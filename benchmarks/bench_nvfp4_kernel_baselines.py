#!/usr/bin/env python3
"""Compare TensorBridge NVFP4A8 against every NVFP4-capable kernel vLLM can pick.

The four labels are NOT numerically equivalent. They are the distinct precision
points actually reachable on this hardware, measured on identical shapes:

    tensorbridge_nvfp4a8   W4 weights, FP8 activations, TensorBridge FPMA-SNC
    cutlass_w4a8           W4 weights, FP8 activations, vLLM CUTLASS
    marlin_nvfp4_w4a16     W4 weights, BF16 activations, vLLM Marlin
    cutlass_fp8_w8a8       FP8 weights, FP8 activations, vLLM CUTLASS

Marlin is the one that matters for an end-to-end claim: on Hopper it is what
`init_nvfp4_linear_kernel()` actually selects, because CUTLASS and FlashInfer
NVFP4 both require sm_100. The CUTLASS W4A8 column is the same-precision
algorithmic reference; the FP8 column is the "don't quantise weights to 4 bits"
alternative.

All labels are timed in one process, interleaved in a palindrome, reusing the
timing primitives from the TensorBridge kernel repo so numbers stay comparable
with the two-way runs produced by bench_nvfp4_optimal_vs_cutlass.py.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

LABELS = (
    "tensorbridge_nvfp4a8",
    "tensorbridge_vllm_quant",
    "cutlass_w4a8",
    "marlin_nvfp4_w4a16",
    "cutlass_fp8_w8a8",
)


def _bootstrap(tensorbridge_root: Path) -> Any:
    """Put the TensorBridge checkout ahead of any installed wheel.

    bench_nvfp4_common itself does `sys.path.insert(0, REPO_ROOT)`, so importing
    it from a clean worktree is what pins which kernel source gets measured.
    """
    scripts_dir = tensorbridge_root / "scripts"
    for entry in (str(scripts_dir), str(tensorbridge_root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import bench_nvfp4_common as bc

    bc.load_benchmark_deps()
    return bc


def _build_tensorbridge(
    bc: Any,
    torch: Any,
    m: int,
    n: int,
    k: int,
    dtype: Any,
    include_quant: bool,
    quantizer: str = "tensorbridge",
):
    """Production path: swizzle64_raw layout, SNC on, workload router.

    `quantizer` selects which activation quantiser feeds the same GEMM. Both
    emit per-token FP8 with an identical scale convention (measured: scale
    ratio 1.0000000-1.0000001, and identical max error against the unquantised
    input), so the two variants differ only in the quantiser's cost. With
    include_quant off they do identical work and act as a null control.
    """
    weight_hp = torch.randn((n, k), dtype=dtype, device="cuda")
    layer = bc.make_layer(n, k, dtype, layout=bc.INTERLEAVE_LAYOUT, use_snc=True)
    layer.load_from_unquantized(weight_hp)
    layer.transform()
    del weight_hp

    meta = layer.tensorbridge_metas[""]
    if not getattr(meta, "use_nvfp4_swizzle64_raw", False):
        raise RuntimeError("expected an nvfp4_swizzle64_raw layer")
    storage = bc.storage_audit(layer, n, k)
    if not storage["contract_4p5bpe_ok"]:
        raise RuntimeError(f"raw+scale storage contract failed: {storage}")

    cfg = bc.get_heuristics_config(meta, shape_m=m, gemm_type="dense", use_stream_k=None)
    # Mirror bench_nvfp4_common.run_shape exactly: the production device flag
    # arrives through BENCH_CURRENT_FLAGS_OVERRIDE, and setting it turns the
    # (empty) built-in CURRENT_FLAGS off. Diverging here would compile a
    # different cubin than the two-way runs we compare against.
    current_override = os.environ.get("BENCH_CURRENT_FLAGS_OVERRIDE", "")
    os.environ["BENCH_EXTRA_NVFP4_FLAGS"] = current_override
    configs, effective_flags = bc.prepare_current_configs(
        layer, cfg, use_current_flags=not current_override
    )

    inputs_hp = torch.randn((m, k), dtype=dtype, device="cuda")
    inputs_q, input_scale = bc.ops.quant_input(
        inputs=inputs_hp, dtype="float8e4m3", group_size=None
    )
    if input_scale is not None and input_scale.numel() == 0:
        input_scale = None
    out = torch.empty((m, n), dtype=dtype, device="cuda")

    def gemm_only():
        return bc.launch_tensorbridge(layer, configs, inputs_q, input_scale, out)

    def with_tb_quant():
        # What TensorBridgeLayerMethod.forward_layer actually does per call:
        # may_quant_input then the kernel. Marlin needs no such step, so
        # excluding it would hand the A8 paths a discount they never get.
        q, s = bc.ops.quant_input(inputs=inputs_hp, dtype="float8e4m3", group_size=None)
        if s is not None and s.numel() == 0:
            s = None
        return bc.launch_tensorbridge(layer, configs, q, s, out)

    def with_vllm_quant():
        # may_quant_input returns early when an input_scale is supplied, so the
        # plugin can pre-quantise without the kernel repo depending on vLLM.
        from vllm import _custom_ops as vllm_ops

        q, s = vllm_ops.scaled_fp8_quant(inputs_hp, use_per_token_if_dynamic=True)
        return bc.launch_tensorbridge(layer, configs, q, s, out)

    if not include_quant:
        fn = gemm_only
    elif quantizer == "vllm":
        fn = with_vllm_quant
    else:
        fn = with_tb_quant
    return fn, {"config": cfg, "effective_compile_flags": effective_flags, "storage": storage}


def _build_marlin_nvfp4(torch: Any, m: int, n: int, k: int, dtype: Any):
    """vLLM Marlin NVFP4: 4-bit weights dequantised into a BF16 mainloop."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
        prepare_fp4_layer_for_marlin,
    )

    if k % 16 != 0:
        raise RuntimeError(f"K={k} must be a multiple of the NVFP4 group size 16")

    # A throwaway module is enough: prepare_fp4_layer_for_marlin only reads the
    # weight tensors plus the partition sizes, and writes the repacked results
    # back onto the same attributes.
    layer = torch.nn.Module()
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.params_dtype = dtype
    layer.weight = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, device="cuda"
    )
    weight_scale = torch.randn((n, k // 16), dtype=torch.float32, device="cuda")
    layer.weight_scale = weight_scale.abs().add_(0.5).to(torch.float8_e4m3fn)
    layer.weight_global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    prepare_fp4_layer_for_marlin(layer)

    x = torch.randn((m, k), dtype=dtype, device="cuda")

    def fn():
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

    return fn, {}


def _build_cutlass_fp8(torch: Any, m: int, n: int, k: int, dtype: Any, include_quant: bool):
    """vLLM CUTLASS FP8 W8A8 with per-token and per-channel scales."""
    from vllm import _custom_ops as vllm_ops

    finfo = torch.finfo(torch.float8_e4m3fn)
    a = (
        torch.randn((m, k), dtype=torch.float32, device="cuda")
        .clamp(finfo.min, finfo.max)
        .to(torch.float8_e4m3fn)
    )
    # cutlass_scaled_mm wants b as (K, N); build it transposed so the tensor is
    # column-major, which is the layout vLLM stores linear weights in.
    b = (
        torch.randn((n, k), dtype=torch.float32, device="cuda")
        .clamp(finfo.min, finfo.max)
        .to(torch.float8_e4m3fn)
        .t()
    )
    scale_a = torch.rand((m, 1), dtype=torch.float32, device="cuda").add_(0.5)
    scale_b = torch.rand((1, n), dtype=torch.float32, device="cuda").add_(0.5)
    x = torch.randn((m, k), dtype=dtype, device="cuda")

    def gemm_only():
        return vllm_ops.cutlass_scaled_mm(
            a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=dtype, bias=None
        )

    def with_quant():
        aq, sa = vllm_ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        return vllm_ops.cutlass_scaled_mm(
            aq, b, scale_a=sa, scale_b=scale_b, out_dtype=dtype, bias=None
        )

    return (with_quant if include_quant else gemm_only), {}


def run_shape(bc: Any, args: argparse.Namespace, shape: tuple[int, int, int]) -> dict[str, Any]:
    torch = bc.torch
    m, n, k = shape
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    torch.manual_seed(args.seed + m * 3 + n * 5 + k * 7)

    iq = args.include_activation_quant
    builders: dict[str, Callable[[], tuple[Callable[[], Any], dict[str, Any]]]] = {
        "tensorbridge_nvfp4a8": lambda: _build_tensorbridge(bc, torch, m, n, k, dtype, iq),
        "tensorbridge_vllm_quant": lambda: _build_tensorbridge(
            bc, torch, m, n, k, dtype, iq, quantizer="vllm"
        ),
        "cutlass_w4a8": lambda: _build_cutlass_w4a8(bc, torch, m, n, k, dtype, iq),
        # Marlin is W4A16: it consumes BF16 activations, so there is no
        # quantisation step to add and this label is identical in both modes.
        "marlin_nvfp4_w4a16": lambda: _build_marlin_nvfp4(torch, m, n, k, dtype),
        "cutlass_fp8_w8a8": lambda: _build_cutlass_fp8(torch, m, n, k, dtype, iq),
    }

    fns: dict[str, Callable[[], Any]] = {}
    details: dict[str, Any] = {}
    for label in LABELS:
        fn, detail = builders[label]()
        fns[label] = fn
        details[label] = detail

    # Prime every label before any timing so no label pays first-call cost.
    for label in LABELS:
        for _ in range(args.prime):
            fns[label]()
    torch.cuda.synchronize()

    l2_flush_buf = None
    if args.flush_l2:
        l2_flush_buf = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    records = []
    for i, label in enumerate(bc.palindrome_labels(args.rounds, LABELS)):
        bc.flush_l2(l2_flush_buf)
        us = bc.time_event(
            fns[label],
            warmup=args.warmup,
            iters=args.iters,
            l2_flush_buf=l2_flush_buf,
            flush_l2_each_iter=False,
        )
        records.append({"i": i, "round": i // (2 * len(LABELS)), "label": label, "us": us})

    by_label = {lbl: [r["us"] for r in records if r["label"] == lbl] for lbl in LABELS}
    stats = {lbl: bc.summarize(values) for lbl, values in by_label.items()}

    reference = stats["tensorbridge_nvfp4a8"]["median_us"]
    comparisons = {}
    for label in LABELS:
        if label == "tensorbridge_nvfp4a8":
            continue
        other = stats[label]["median_us"]
        delta = reference - other
        ci = stats["tensorbridge_nvfp4a8"]["ci95_us"] + stats[label]["ci95_us"]
        comparisons[label] = {
            # gap > 0 means TensorBridge took longer than this baseline.
            "tensorbridge_gap_pct_median": (reference / other - 1.0) * 100.0,
            "tensorbridge_minus_baseline_us": delta,
            "ci95_sum_us": ci,
            # A difference smaller than the summed CI95 is not resolvable and
            # must not be reported as a speedup either way.
            "resolvable": abs(delta) > ci,
        }

    row = {
        "M": m,
        "N": n,
        "K": k,
        "dtype": args.dtype,
        "stats": stats,
        "tflops": {lbl: bc.tflops(m, n, k, stats[lbl]["median_us"]) for lbl in LABELS},
        "comparisons": comparisons,
        "tensorbridge_detail": details["tensorbridge_nvfp4a8"],
        "records": records,
    }
    summary = " ".join(
        f"{lbl.split('_')[0]}={stats[lbl]['median_us']:.2f}" for lbl in LABELS
    )
    print(f"[baselines] M={m} N={n} K={k} {summary}", flush=True)
    return row


def _build_cutlass_w4a8(bc: Any, torch: Any, m: int, n: int, k: int, dtype: Any, include_quant: bool):
    kwargs = bc.build_random_cutlass_w4a8_inputs(n, k, m)
    x = torch.randn((m, k), dtype=dtype, device="cuda")
    scale_shape = kwargs["a_token_scales"].shape

    def gemm_only():
        return torch.ops._C.cutlass_w4a8_mm(
            kwargs["a"],
            kwargs["b_q"],
            kwargs["b_group_scales"],
            kwargs["b_group_size"],
            kwargs["b_channel_scales"],
            kwargs["a_token_scales"],
            None,
            None,
        )

    def with_quant():
        from vllm import _custom_ops as vllm_ops

        aq, sa = vllm_ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        return torch.ops._C.cutlass_w4a8_mm(
            aq,
            kwargs["b_q"],
            kwargs["b_group_scales"],
            kwargs["b_group_size"],
            kwargs["b_channel_scales"],
            sa.reshape(scale_shape),
            None,
            None,
        )

    return (with_quant if include_quant else gemm_only), {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensorbridge-root",
        type=Path,
        default=Path("/data/user/jzou521/codes/cuda/tensorbridge-pinned"),
        help="TensorBridge checkout whose kernel source is measured",
    )
    parser.add_argument("--shape-file", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--prime", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flush-l2", action="store_true")
    parser.add_argument(
        "--include-activation-quant",
        action="store_true",
        help="time the BF16->FP8 activation quantisation together with the GEMM "
             "for the A8 labels; Marlin (W4A16) is unaffected",
    )
    parser.add_argument("--shape-limit", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bc = _bootstrap(args.tensorbridge_root)
    bc.load_vllm_cutlass_ops()
    torch = bc.torch
    if not hasattr(torch.ops._C, "cutlass_w4a8_mm"):
        print("[baselines] cutlass_w4a8_mm unavailable; check VLLM_CUTLASS_ROOT", file=sys.stderr)
        return 2

    shapes = []
    for line in args.shape_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "M,")):
            continue
        m, n, k = (int(part) for part in line.split(","))
        shapes.append((m, n, k))
    if args.shape_limit:
        shapes = shapes[: args.shape_limit]
    print(f"[baselines] {len(shapes)} shapes, labels={LABELS}", flush=True)

    rows, failures = [], []
    for shape in shapes:
        try:
            rows.append(run_shape(bc, args, shape))
        except Exception as error:  # noqa: BLE001 - one bad shape must not lose the run
            if not args.continue_on_error:
                raise
            failures.append({"shape": list(shape), "error": repr(error)})
            print(f"[baselines] FAILED {shape}: {error!r}", file=sys.stderr, flush=True)

    payload = {
        "schema_version": 1,
        "experiment": "nvfp4_kernel_baselines",
        "labels": list(LABELS),
        "tensorbridge_root": str(args.tensorbridge_root),
        "rounds": args.rounds,
        "prime": args.prime,
        "warmup": args.warmup,
        "iters": args.iters,
        "flush_l2": args.flush_l2,
        "include_activation_quant": args.include_activation_quant,
        "dtype": args.dtype,
        "environment": bc.environment_meta(),
        "slurm": {
            key: os.environ.get(key)
            for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK")
        },
        "generated_unix_time": int(time.time()),
        "rows": rows,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"[baselines] wrote {args.output} ({len(rows)} rows, {len(failures)} failures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
