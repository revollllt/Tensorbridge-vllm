#!/usr/bin/env python3
"""Decode latency across batch sizes for one TensorBridge arm.

One engine serves the whole batch sweep. The arm is fixed at weight-load time so
arms need separate processes, but batch size does not — and engine startup
(21 GB checkpoint plus NVRTC compilation) costs more than the measurement, so
starting one engine per batch size would spend most of the wall clock loading.

Every batch size gets its own captured CUDA graph. That matters more than it
looks: a decode step at an uncaptured size silently falls back to eager, which
reads as an unexplained slowdown at that batch alone, so the script checks the
resolved capture list and refuses to run if any size is missing.

`ignore_eos` keeps every sequence the same length, which keeps the running batch
pinned at the captured size for the whole generation.

Usage:
    python bench_latency.py --arm tensorbridge --model /path/to/Qwen3.6-27B-NVFP4 \
        --output results/tensorbridge_run0.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import arms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=sorted(arms.ARMS))
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--output", type=Path, help="write the result JSON here")
    p.add_argument("--batch-sizes", default="1,4,16,32,64,128")
    p.add_argument("--input-len", type=int, default=128)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    env = arms.apply(args.arm)  # must precede the vllm import

    import numpy as np
    from vllm import LLM, SamplingParams

    started = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        quantization="modelopt_mixed",
        # This checkpoint declares no torch_dtype, so "auto" would resolve from
        # vLLM's fallback rather than from the checkpoint. Marlin dequantizes
        # into this dtype, so it has to be pinned and identical across arms.
        dtype="bfloat16",
        # The vision tower needs vllm.vllm_flash_attn.layers, which the
        # precompiled wheel does not ship; the engine dies at startup without
        # this. Nothing in a text decode benchmark needs the encoder anyway.
        language_model_only=True,
        max_model_len=args.max_model_len,
        max_num_seqs=max(batch_sizes),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,  # else repeated iterations reuse KV
        seed=args.seed,
        compilation_config={
            "mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_num_of_warmups": 1,
            "cudagraph_capture_sizes": sorted(batch_sizes),
        },
    )
    init_seconds = time.perf_counter() - started

    captured = sorted(llm.llm_engine.vllm_config.compilation_config.cudagraph_capture_sizes or [])
    missing = [b for b in batch_sizes if b not in captured]
    if missing:
        raise SystemExit(f"batch sizes {missing} were not captured as CUDA graphs; got {captured}")

    rng = np.random.default_rng(args.seed)
    sweep = []
    for batch_size in batch_sizes:
        prompts = [
            {"prompt_token_ids": row.tolist()}
            for row in rng.integers(0, 10000, size=(batch_size, args.input_len))
        ]
        params = SamplingParams(
            n=1, temperature=1.0, top_p=1.0, ignore_eos=True,
            max_tokens=args.output_len, detokenize=False,
        )

        def one_pass():
            begin = time.perf_counter()
            llm.generate(prompts, sampling_params=params, use_tqdm=False)
            return time.perf_counter() - begin

        for _ in range(args.warmup):
            one_pass()
        latencies = [one_pass() for _ in range(args.iters)]

        median = statistics.median(latencies)
        sweep.append({
            "batch_size": batch_size,
            "latencies": latencies,
            "median_seconds": median,
            "min_seconds": min(latencies),
            "max_seconds": max(latencies),
            # Decode tokens only; prefill is excluded because it is a fixed cost
            # per iteration and not what a decode kernel comparison is about.
            "decode_tokens_per_second": batch_size * (args.output_len - 1) / median,
        })
        print(f"[latency] arm={args.arm} bs={batch_size:<4} median={median:.3f}s "
              f"min={min(latencies):.3f} max={max(latencies):.3f}")

    result = {
        "arm": args.arm,
        "model": str(args.model),
        "environment": env,
        "versions": arms.versions(),
        "gpu": arms.gpu_info(),
        "probe": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "max_model_len": args.max_model_len,
            "batch_sizes": batch_sizes,
            "iters": args.iters,
            "warmup": args.warmup,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "captured_cudagraph_sizes": captured,
        "engine_init_seconds": init_seconds,
        "sweep": sweep,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[latency] wrote {args.output}")


if __name__ == "__main__":
    main()
