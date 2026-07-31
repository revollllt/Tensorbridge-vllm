#!/usr/bin/env python3
"""Group an Nsight Systems kernel summary into a GPU-time breakdown.

Wall-clock benchmarks charge every kernel its host launch path. Inside a
replayed CUDA graph that path does not recur, so a kernel that looks expensive
in an eager benchmark can be nearly free in production. This reads the GPU
timeline instead and answers where decode time actually goes.

Produce the input with:

    nsys profile --trace=cuda --sample=none --cpuctxsw=none \\
        --cuda-graph-trace=node --output=prof \\
        python bench_latency.py --arm tensorbridge --model $M --batch-sizes 128
    nsys stats --report cuda_gpu_kern_sum --format csv --output prof prof.nsys-rep

`--cuda-graph-trace=node` is load-bearing: the default reports one entry per
graph launch and hides every kernel inside it.

Usage:
    python summarize_profile.py prof_cuda_gpu_kern_sum.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Substring -> label. Ordered: the first match wins, so put specific patterns
# before general ones.
GROUPS = [
    ("_quant_tensor_kernel", "TensorBridge activation quant"),
    ("scaled_fp8_quant", "vLLM FP8 activation quant"),
    ("void tensorbridge<", "TensorBridge NVFP4 GEMM"),
    ("gated_delta_rule", "gated delta rule (linear attn)"),
    ("cutlass", "CUTLASS (FP8 layers)"),
    ("marlin", "Marlin (NVFP4 W4A16)"),
    ("flash", "flash attention"),
    ("elementwise_kernel", "elementwise"),
    ("reduce", "reduce"),
    ("silu", "silu"),
]


def label_for(name: str) -> str:
    lowered = name.lower()
    for needle, label in GROUPS:
        if needle.lower() in lowered:
            return label
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--top-other", type=int, default=5,
                        help="also list this many largest unclassified kernels")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv_file.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no rows in {args.csv_file}")

    total_ns = sum(int(r["Total Time (ns)"]) for r in rows)
    grouped: dict[str, list[int]] = {}
    for row in rows:
        label = label_for(row["Name"])
        time_ns, calls = int(row["Total Time (ns)"]), int(row["Instances"])
        bucket = grouped.setdefault(label, [0, 0])
        bucket[0] += time_ns
        bucket[1] += calls

    print(f"total GPU kernel time: {total_ns / 1e9:.3f}s over {len(rows)} distinct kernels")
    print()
    print(f"{'group':<32}{'GPU %':>9}{'total ms':>12}{'calls':>10}{'us/call':>10}")
    print("-" * 73)
    for label, (time_ns, calls) in sorted(grouped.items(), key=lambda kv: -kv[1][0]):
        print(f"{label:<32}{time_ns / total_ns * 100:>9.2f}{time_ns / 1e6:>12.1f}"
              f"{calls:>10}{time_ns / calls / 1000:>10.2f}")

    others = sorted(
        (r for r in rows if label_for(r["Name"]) == "other"),
        key=lambda r: -int(r["Total Time (ns)"]),
    )[: args.top_other]
    if others:
        print(f"\nlargest unclassified kernels")
        for row in others:
            name = row["Name"].strip('"')[:58]
            print(f"  {float(row['Time (%)']):>6.2f}%  {name}")


if __name__ == "__main__":
    main()
