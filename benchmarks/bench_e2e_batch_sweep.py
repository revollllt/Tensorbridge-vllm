#!/usr/bin/env python3
"""Sweep decode batch size end-to-end inside a single engine.

`vllm bench latency` takes one batch size per invocation, so scanning six of
them cost six engine starts per arm -- and engine start (checkpoint load plus
NVRTC JIT) dominates the wall clock. The backend is chosen at weight-load time
so arms genuinely need separate processes, but batch size does not: one engine
can capture a CUDA graph per batch size and serve the whole sweep.

Same probe as the single-batch runs: random prompt token ids, ignore_eos so the
decode batch stays pinned at the captured size, detokenisation off.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        str(p): float(np.percentile(ordered, p)) for p in (10, 25, 50, 75, 90, 99)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-sizes", required=True, help="comma separated")
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--num-iters-warmup", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", required=True, help="value of TENSORBRIDGE_VLLM_BACKEND")
    parser.add_argument("--phase", default="measured")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--order", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    max_batch = max(batch_sizes)

    from vllm import LLM, SamplingParams

    engine_args: dict[str, Any] = {
        "model": args.model,
        "quantization": "modelopt_mixed",
        "dtype": "bfloat16",
        "language_model_only": True,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": max_batch,
        "enable_prefix_caching": False,
        "seed": args.seed,
        "compilation_config": {
            "mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_num_of_warmups": 1,
            # One captured graph per batch size in the sweep. A decode step at a
            # size that was not captured silently falls back to eager, which
            # would show up as an unexplained slowdown at that batch only.
            "cudagraph_capture_sizes": sorted(batch_sizes),
        },
    }

    started = time.perf_counter()
    llm = LLM(**engine_args)
    engine_init_seconds = time.perf_counter() - started

    resolved = llm.llm_engine.vllm_config.compilation_config
    resolved_capture = sorted(resolved.cudagraph_capture_sizes or [])
    missing = [b for b in batch_sizes if b not in resolved_capture]
    if missing:
        raise RuntimeError(
            f"requested batch sizes {missing} were not captured as CUDA graphs; "
            f"resolved capture sizes = {resolved_capture}"
        )

    rng = np.random.default_rng(args.seed)
    results = []
    for batch_size in batch_sizes:
        sampling_params = SamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=args.output_len,
            detokenize=False,
        )
        prompts = [
            {"prompt_token_ids": row.tolist()}
            for row in rng.integers(0, 10000, size=(batch_size, args.input_len))
        ]

        def one_pass() -> float:
            begin = time.perf_counter()
            llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
            return time.perf_counter() - begin

        for _ in range(args.num_iters_warmup):
            one_pass()
        latencies = [one_pass() for _ in range(args.num_iters)]

        results.append(
            {
                "batch_size": batch_size,
                "latencies": latencies,
                "median_seconds": statistics.median(latencies),
                "mean_seconds": statistics.fmean(latencies),
                "percentiles": _percentiles(latencies),
                "decode_tokens": batch_size * args.output_len,
            }
        )
        print(
            f"[sweep] arm={args.arm} bs={batch_size:<4} "
            f"median={statistics.median(latencies):.3f}s "
            f"min={min(latencies):.3f} max={max(latencies):.3f}",
            flush=True,
        )

    try:
        import torch

        torch_version = torch.__version__
    except Exception:  # noqa: BLE001
        torch_version = None
    try:
        import vllm

        vllm_version = vllm.__version__
    except Exception:  # noqa: BLE001
        vllm_version = None

    payload = {
        "schema_version": 1,
        "experiment": "tensorbridge_e2e_batch_sweep",
        "status": "passed",
        "benchmark_context": {
            "phase": args.phase,
            "index": str(args.index),
            "arm": args.arm,
            "order": args.order,
        },
        "production_contract": {
            "transformer_nvfp4_backend": args.arm,
            "fpma_global_scale_alpha": float(
                os.environ.get("TENSORBRIDGE_NVFP4_FPMA_ALPHA", "1.0")
            ),
            "fpma_prefold_selector": os.environ.get(
                "TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR", "none"
            ),
            "fpma_ulp_correction": os.environ.get(
                "TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION", "0"
            )
            == "1",
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "cpu_thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "visible_gpu_count": len(
                [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x]
            ),
            "torch": torch_version,
            "vllm": vllm_version,
            "tensorbridge_compiler": os.environ.get("TENSORBRIDGE_COMPILER", "nvrtc"),
            "gpu_clock_pin_status": os.environ.get("TENSORBRIDGE_GPU_CLOCK_PIN_STATUS"),
        },
        "probe": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "max_model_len": args.max_model_len,
            "num_iters": args.num_iters,
            "num_iters_warmup": args.num_iters_warmup,
            "batch_sizes": batch_sizes,
        },
        "engine_args": {
            key: value for key, value in engine_args.items() if key != "compilation_config"
        }
        | {"compilation_config": engine_args["compilation_config"]},
        "resolved_cudagraph_capture_sizes": resolved_capture,
        "timing": {"engine_init_seconds": engine_init_seconds},
        "sweep": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[sweep] engine_init={engine_init_seconds:.1f}s -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
