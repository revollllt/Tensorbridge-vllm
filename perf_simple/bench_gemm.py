#!/usr/bin/env python3
"""Per-layer GEMM timing for the three arms, on this model's real MLP shapes.

Each arm is driven through the plugin's own linear method — `create_weights`,
`process_weights_after_loading`, `apply` — so what runs here is the production
code path, including the activation quantisation that `apply` performs.

Everything is timed inside a captured CUDA graph, because that is how the engine
runs it: `FULL_DECODE_ONLY` replays a captured decode step rather than
relaunching each kernel from the host. Timing eager launches instead measures
the host path, which the engine does not pay, and the difference is not small —
at M=1 the same TensorBridge layer reads +95% against Marlin launched eagerly
and -3% replayed from a graph. Only the graph number predicts what
`bench_latency.py` will show.

Calls are captured in a loop into one graph rather than one graph per call, so
the graph launch is amortised the way it is in the engine, where a single graph
holds a whole decode step.

Arms are interleaved as a palindrome and each round contributes one sample, so a
drift during the run affects every arm roughly equally.

Usage:
    python bench_gemm.py --model /path/to/Qwen3.6-27B-NVFP4 --output results/gemm.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import arms

# The first two are this checkpoint's MLP projections as the engine sees them —
# gate and up fused into one N, then down — and account for all 192 transformer
# NVFP4 layers. Matching them is what lets this table explain the end-to-end one.
#
# The last two are smaller, narrower shapes. They are not in this checkpoint;
# they are here so a ranking is not read off two shapes that happen to share a
# hidden size of 5120 and a very wide inner dimension.
DEFAULT_SHAPES = "34816,5120 5120,17408 6144,4096 2048,6144"

LABELS = ("official", "normal_a8", "tensorbridge")

# with_quant is what production runs today. gemm_only hands the kernel an
# activation that is already FP8, which is what the layer would cost if the
# quantisation were fused into whatever produces the activation. Marlin is
# W4A16 and has no such step, so it is identical in both modes and acts as a
# control on the difference.
MODES = ("with_quant", "gemm_only")

# TensorBridge's implicit analytic alpha (123/128) was derived only over the
# verified nonzero raw E4M3 scale domain, and the plugin fails closed outside it
# rather than extrapolating. Block scales are drawn as raw bytes in that range so
# the benchmark exercises the same default a real checkpoint does.
E4M3_SCALE_BYTE_MIN, E4M3_SCALE_BYTE_MAX = 0x39, 0x7E


def random_nvfp4_weights(torch, n: int, k: int):
    """ModelOpt NVFP4 tensors in the layout `create_weights` allocates."""
    scale_bytes = torch.randint(
        E4M3_SCALE_BYTE_MIN, E4M3_SCALE_BYTE_MAX + 1,
        (n, k // 16), dtype=torch.uint8, device="cuda",
    )
    return {
        "weight": torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda"),
        "weight_scale": scale_bytes.view(torch.float8_e4m3fn),
        # The plugin rejects non-positive or non-finite global scales.
        "weight_scale_2": torch.full((1,), 0.05, dtype=torch.float32, device="cuda"),
        "input_scale": torch.full((1,), 1.0, dtype=torch.float32, device="cuda"),
    }


def build_plugin_arm(torch, arm: str, n: int, k: int, dtype, weights):
    """Build one plugin linear method; return its with-quant and GEMM-only calls.

    Both arms quantise activations to FP8 inside `apply`. That cost is real
    today but it is a separate elementwise pass over the activation, and fusing
    it into the kernel that produces the activation would remove it. The
    GEMM-only closure hands the kernel an already-quantised input so the two
    numbers bracket what this layer costs before and after such a fusion.
    """
    from vllm.plugins.tensorbridge import (
        TensorBridgeNormalA8LinearMethod,
        TensorBridgeNvfp4LinearMethod,
    )

    method_cls = {
        "tensorbridge": TensorBridgeNvfp4LinearMethod,
        "normal_a8": TensorBridgeNormalA8LinearMethod,
    }[arm]
    method = method_cls(prefix=f"bench.{arm}")

    layer = torch.nn.Module()
    # create_weights allocates with a bare torch.empty, so the parameters land
    # wherever the default device points. The engine builds models under a CUDA
    # device context; without it the tensors stay on the host and the first
    # kernel launch faults with an illegal memory access.
    with torch.device("cuda"):
        method.create_weights(
            layer,
            input_size_per_partition=k,
            output_partition_sizes=[n],
            input_size=k,
            output_size=n,
            params_dtype=dtype,
            # vLLM's parameter classes require a loader. Weights are written
            # directly below, so it is never called.
            weight_loader=lambda *_args, **_kwargs: None,
        )
    for name, value in weights.items():
        getattr(layer, name).data.copy_(value)
    method.process_weights_after_loading(layer)

    def with_quant(x):
        return method.apply(layer, x)

    def build_gemm_only(x):
        """Pre-quantise once, then close over the result."""
        if arm == "tensorbridge":
            from tensorbridge.api.v1 import TensorBridgeLayerMethod

            # may_quant_input returns its inputs untouched when an input_scale
            # is supplied, so passing both skips the quantisation kernel while
            # running exactly the same GEMM.
            quantised, scale = TensorBridgeLayerMethod.may_quant_input(
                layer=layer, inputs=x
            )
            return lambda: TensorBridgeLayerMethod.forward_layer(
                layer=layer,
                inputs=quantised,
                input_scale=scale,
                compute_config=method.compute_config,
            )

        # normal_a8: apply_weights quantises and then calls apply_scaled_mm.
        # Call the second half directly with a pre-quantised activation.
        fp8 = method.fp8_linear
        w, w_s, x_s, x_s_ub = fp8._get_layer_params(layer)
        quantised, scale = fp8.quant_fp8(x.view(-1, x.shape[-1]), x_s, x_s_ub)
        return lambda: fp8.apply_scaled_mm(
            A=quantised, B=w, out_dtype=dtype, As=scale, Bs=w_s,
            bias=None, output_shape=[x.shape[0], n],
        )

    return with_quant, build_gemm_only


def build_marlin_arm(torch, n: int, k: int, dtype, weights):
    """`official`: the same code MarlinNvFp4LinearKernel.apply_weights runs.

    Returns the same pair shape as the plugin arms so the caller does not have
    to special-case it.
    """
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
        prepare_fp4_layer_for_marlin,
    )

    layer = torch.nn.Module()
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.params_dtype = dtype
    layer.weight = weights["weight"].clone()
    layer.weight_scale = weights["weight_scale"].clone()
    layer.weight_global_scale = weights["weight_scale_2"].clone()
    prepare_fp4_layer_for_marlin(layer)

    def call(x):
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=n,
            size_k=k,
            bias=None,
        )

    # W4A16 consumes BF16 directly, so there is no quantisation to remove and
    # the two modes are the same measurement.
    return call, lambda x: (lambda: call(x))


def capture_graph(torch, fn, x, iters: int, warmup: int):
    """Capture `iters` calls into one CUDA graph.

    Warmup runs on a side stream because capture requires the work to have been
    replayed at least once off the default stream.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn(x)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn(x)
    return graph


def time_graph_us(torch, graph, iters: int, warmup: int) -> float:
    """Mean microseconds per captured call over one replay."""
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000.0 / iters


def summarize(values: list[float]) -> dict:
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "median_us": statistics.median(values),
        "stdev_us": stdev,
        # 95% interval on the mean. Two arms whose intervals overlap have not
        # been separated by this experiment, however different their medians.
        "ci95_us": 1.96 * stdev / len(values) ** 0.5 if values else 0.0,
        "samples": values,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, help="unused; recorded for provenance")
    p.add_argument("--output", type=Path)
    p.add_argument("--shapes", default=DEFAULT_SHAPES, help='space separated "N,K"')
    p.add_argument("--batch-sizes", default="1,4,16,32,64,128", help="M values")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--iters", type=int, default=40, help="calls captured per graph")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dist-port", type=int, default=29591,
                   help="port for the one-rank process group")
    args = p.parse_args()

    shapes = [tuple(int(v) for v in s.split(",")) for s in args.shapes.split()]
    batch_sizes = [int(v) for v in args.batch_sizes.split(",") if v]

    # The plugin must be registered before its method classes are usable, and
    # arms.apply must precede that because register() reads the environment.
    env = arms.apply("tensorbridge")
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.plugins.tensorbridge import register

    register()
    # The linear methods ask for the tensor-parallel world size while creating
    # weights. There is no engine here to set that up, so stand up a one-rank
    # group; TP=1 is also what the end-to-end runs use.
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{args.dist_port}",
    )

    dtype = torch.bfloat16
    rows = []
    # Everything from model-parallel setup onward reads the ambient vLLM config:
    # initialize_model_parallel does, and so does the CustomOp machinery when
    # layers are built and when they run.
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(1, 1)
        for n, k in shapes:
            torch.manual_seed(args.seed + n * 5 + k * 7)
            weights = random_nvfp4_weights(torch, n, k)
            builders = {
                "official": build_marlin_arm(torch, n, k, dtype, weights),
                "normal_a8": build_plugin_arm(torch, "normal_a8", n, k, dtype, weights),
                "tensorbridge": build_plugin_arm(torch, "tensorbridge", n, k, dtype, weights),
            }

            for m in batch_sizes:
                x = torch.randn((m, k), dtype=dtype, device="cuda")

                # One key per (arm, mode). Interleaving all six in the palindrome
                # rather than running the modes as separate sweeps means a drift
                # cannot land on one mode and not the other.
                calls = {}
                for label in LABELS:
                    with_quant, build_gemm_only = builders[label]
                    calls[(label, "with_quant")] = lambda _x, f=with_quant: f(_x)
                    gemm_only = build_gemm_only(x)
                    calls[(label, "gemm_only")] = lambda _x, f=gemm_only: f()

                graphs, capture_errors = {}, {}
                for key, call in calls.items():
                    try:
                        graphs[key] = capture_graph(
                            torch, call, x, iters=args.iters, warmup=args.warmup
                        )
                    except Exception as error:  # noqa: BLE001
                        # A kernel that cannot be captured is a finding, not a
                        # crash: record it and keep the rest.
                        capture_errors["/".join(key)] = repr(error)

                keys = list(calls)
                order = [key for key in (keys + keys[::-1]) * args.rounds if key in graphs]
                samples: dict = {key: [] for key in graphs}
                for key in order:
                    samples[key].append(
                        time_graph_us(torch, graphs[key], args.iters, args.warmup)
                    )

                stats = {mode: {} for mode in MODES}
                for (label, mode), values in samples.items():
                    stats[mode][label] = summarize(values)

                row = {"M": m, "N": n, "K": k, "stats": stats,
                       "capture_errors": capture_errors, "vs_official": {}}
                for mode in MODES:
                    table = stats[mode]
                    if "official" not in table:
                        continue
                    base = table["official"]
                    for label in LABELS:
                        if label == "official" or label not in table:
                            continue
                        delta = table[label]["median_us"] - base["median_us"]
                        ci = table[label]["ci95_us"] + base["ci95_us"]
                        row["vs_official"].setdefault(label, {})[mode] = {
                            # Negative means the arm is faster than Marlin.
                            "gap_pct": (table[label]["median_us"] / base["median_us"] - 1) * 100,
                            "resolvable": abs(delta) > ci,
                        }
                rows.append(row)

                parts = []
                for mode in MODES:
                    tag = "q" if mode == "with_quant" else "g"
                    parts += [f"{tag}:{lab}={stats[mode][lab]['median_us']:.2f}"
                              for lab in LABELS if lab in stats[mode]]
                print(f"[gemm] N={n:<6} K={k:<6} M={m:<5} " + "  ".join(parts))
                for key, error in capture_errors.items():
                    print(f"[gemm]   capture failed for {key}: {error}")

                graphs.clear()
                torch.cuda.empty_cache()

    result = {
        "environment": env,
        "versions": arms.versions(),
        "gpu": arms.gpu_info(),
        "model": str(args.model) if args.model else None,
        "probe": {
            "shapes": [list(s) for s in shapes],
            "batch_sizes": batch_sizes,
            "rounds": args.rounds,
            "warmup": args.warmup,
            "iters_per_graph": args.iters,
            "cuda_graph": True,
            "activation_quant_included": True,
        },
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[gemm] wrote {args.output}")

    for mode, title in (
        ("with_quant", "with activation quantisation (what runs today)"),
        ("gemm_only", "GEMM only (activation already FP8, i.e. quant fused away)"),
    ):
        print(f"\n{title}")
        header = f"{'M':>6}{'N':>8}{'K':>8}"
        for label in LABELS[1:]:
            header += f"{label:>16}"
        print(header)
        for row in rows:
            line = f"{row['M']:>6}{row['N']:>8}{row['K']:>8}"
            for label in LABELS[1:]:
                entry = row["vs_official"].get(label, {}).get(mode)
                cell = "-" if entry is None else (
                    f"{entry['gap_pct']:+.1f}%" + ("" if entry["resolvable"] else " ~")
                )
                line += f"{cell:>16}"
            print(line)
    print("\n  gap vs Marlin inside a CUDA graph; negative = faster.")
    print("  ~ = inside CI95, not resolvable.")
    print("  Marlin is W4A16, so it is the same measurement in both tables and")
    print("  the difference between them is the activation quantisation alone.")


if __name__ == "__main__":
    main()
