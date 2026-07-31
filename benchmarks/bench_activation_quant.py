#!/usr/bin/env python3
"""Can vLLM's FP8 activation quantiser stand in for TensorBridge's?

Two questions, in order, because the second only matters if the first holds:

1. Are they interchangeable? Both claim dynamic per-token FP8, but a different
   scale convention (reciprocal, rounding, clamping) would silently corrupt
   results. This checks the dequantised tensors agree, not just the shapes.
2. How much does each cost? TensorBridge quantises with a Triton kernel that
   launches one program per token with BLOCK = next_pow2(K); at small M that
   leaves an H100 almost idle. vLLM ships a hand-written CUDA kernel.

Timed the same way as the GEMM benchmarks so the numbers compose.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

LABELS = ("tensorbridge_quant_input", "vllm_scaled_fp8_quant")


def _bootstrap(tensorbridge_root: Path) -> Any:
    for entry in (str(tensorbridge_root / "scripts"), str(tensorbridge_root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import bench_nvfp4_common as bc

    bc.load_benchmark_deps()
    return bc


def check_equivalence(bc: Any, torch: Any, m: int, k: int, dtype: Any) -> dict[str, Any]:
    """Compare what each quantiser reconstructs, not what it stores."""
    from vllm import _custom_ops as vllm_ops

    x = torch.randn((m, k), dtype=dtype, device="cuda")

    tb_q, tb_s = bc.ops.quant_input(inputs=x, dtype="float8e4m3", group_size=None)
    vl_q, vl_s = vllm_ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)

    tb_s2 = tb_s.reshape(m, 1).float()
    vl_s2 = vl_s.reshape(m, 1).float()
    # Both are meant to satisfy x ~= q * scale. Comparing reconstructions is
    # what makes a swap safe; identical scale tensors are neither necessary
    # nor sufficient.
    tb_deq = tb_q.float() * tb_s2
    vl_deq = vl_q.float() * vl_s2

    scale_ratio = (tb_s2 / vl_s2).flatten()
    deq_absdiff = (tb_deq - vl_deq).abs()
    denom = x.abs().amax().clamp_min(1e-30)

    return {
        "tb_scale_shape": list(tb_s.shape),
        "vllm_scale_shape": list(vl_s.shape),
        "tb_quant_dtype": str(tb_q.dtype),
        "vllm_quant_dtype": str(vl_q.dtype),
        "scale_ratio_min": float(scale_ratio.min()),
        "scale_ratio_max": float(scale_ratio.max()),
        "bitwise_equal_codes": bool(
            torch.equal(tb_q.view(torch.uint8), vl_q.view(torch.uint8))
        ),
        "dequant_max_abs_diff": float(deq_absdiff.amax()),
        "dequant_max_rel_diff": float(deq_absdiff.amax() / denom),
        "tb_vs_input_max_rel_err": float(
            (tb_deq - x.float()).abs().amax() / denom
        ),
        "vllm_vs_input_max_rel_err": float(
            (vl_deq - x.float()).abs().amax() / denom
        ),
    }


def run_shape(bc: Any, args: argparse.Namespace, m: int, k: int) -> dict[str, Any]:
    from vllm import _custom_ops as vllm_ops

    torch = bc.torch
    dtype = torch.bfloat16
    torch.manual_seed(args.seed + m * 3 + k * 7)

    equivalence = check_equivalence(bc, torch, m, k, dtype)

    x = torch.randn((m, k), dtype=dtype, device="cuda")
    fns = {
        "tensorbridge_quant_input": lambda: bc.ops.quant_input(
            inputs=x, dtype="float8e4m3", group_size=None
        )[0],
        "vllm_scaled_fp8_quant": lambda: vllm_ops.scaled_fp8_quant(
            x, use_per_token_if_dynamic=True
        )[0],
    }
    for label in LABELS:
        for _ in range(args.prime):
            fns[label]()
    torch.cuda.synchronize()

    records = []
    for i, label in enumerate(bc.palindrome_labels(args.rounds, LABELS)):
        us = bc.time_event(fns[label], warmup=args.warmup, iters=args.iters)
        records.append({"i": i, "label": label, "us": us})
    stats = {
        label: bc.summarize([r["us"] for r in records if r["label"] == label])
        for label in LABELS
    }

    tb = stats["tensorbridge_quant_input"]["median_us"]
    vl = stats["vllm_scaled_fp8_quant"]["median_us"]
    delta = tb - vl
    ci = stats["tensorbridge_quant_input"]["ci95_us"] + stats["vllm_scaled_fp8_quant"]["ci95_us"]
    row = {
        "M": m,
        "K": k,
        "stats": stats,
        "tb_over_vllm": tb / vl,
        "resolvable": abs(delta) > ci,
        "equivalence": equivalence,
        "records": records,
    }
    print(
        f"[quant] M={m:>5} K={k:>6} tb={tb:8.2f}us vllm={vl:8.2f}us "
        f"ratio={tb / vl:5.2f}x  deq_rel_diff={equivalence['dequant_max_rel_diff']:.2e}",
        flush=True,
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensorbridge-root",
        type=Path,
        default=Path("/data/user/jzou521/codes/cuda/tensorbridge-pinned"),
    )
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--prime", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bc = _bootstrap(args.tensorbridge_root)

    # The (M, K) pairs the GEMM benchmark exercised, so the two results compose.
    ms = [16, 128, 512, 4096]
    ks = [2048, 4096, 7168, 8192, 12288, 18432, 28672]
    rows = [run_shape(bc, args, m, k) for m in ms for k in ks]

    payload = {
        "schema_version": 1,
        "experiment": "activation_quant_comparison",
        "labels": list(LABELS),
        "tensorbridge_root": str(args.tensorbridge_root),
        "rounds": args.rounds,
        "prime": args.prime,
        "warmup": args.warmup,
        "iters": args.iters,
        "environment": bc.environment_meta(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"[quant] wrote {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
