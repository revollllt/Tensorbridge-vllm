#!/usr/bin/env python3
"""Attach provenance to a `vllm bench latency` result.

The bench CLI writes only `{avg_latency, latencies, percentiles}`. A speedup
claim needs to prove the arms differed in exactly one way, so this records the
host, the resolved runtime, and the probe shape alongside the samples, in the
layout `scripts/analyze_perf_baseline.py` already checks.

Inputs arrive via TB_* environment variables because the caller is a Slurm
wrapper that already holds them.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise SystemExit(f"[annotate] missing required environment variable {name}")
    return value


def main() -> int:
    raw_path = Path(_env("TB_RAW"))
    if not raw_path.is_file():
        raise SystemExit(f"[annotate] bench output missing: {raw_path}")
    raw = json.loads(raw_path.read_text())
    latencies = [float(value) for value in raw["latencies"]]
    if not latencies:
        raise SystemExit("[annotate] bench produced no latency samples")

    arm = _env("TB_ARM")
    started = float(_env("TB_STARTED"))
    ended = float(_env("TB_ENDED"))
    warmup = int(_env("TB_PROBE_WARMUP"))
    # The CLI does not time engine startup separately. Everything the process
    # spent outside the measured iterations is startup plus warmup; warmup is
    # removed using the measured mean so the remainder is comparable per arm.
    measured_seconds = sum(latencies)
    warmup_estimate = warmup * (measured_seconds / len(latencies))
    startup_seconds = max(0.0, (ended - started) - measured_seconds - warmup_estimate)

    try:
        import torch

        torch_version = torch.__version__
    except Exception:  # noqa: BLE001 - provenance must never break the run
        torch_version = None
    try:
        import vllm

        vllm_version = vllm.__version__
    except Exception:  # noqa: BLE001
        vllm_version = None

    record = {
        "schema_version": 1,
        "experiment": "tensorbridge_e2e_latency",
        "status": "passed",
        "benchmark_context": {
            "phase": _env("TB_PHASE"),
            "index": _env("TB_INDEX"),
            "arm": arm,
            "order": _env("TB_ORDER"),
        },
        "production_contract": {
            "transformer_nvfp4_backend": arm,
            # The measured arms all run the neutral FPMA settings; the plugin
            # refuses any compensation unless the backend is tensorbridge.
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
            "input_len": int(_env("TB_PROBE_INPUT_LEN")),
            "output_len": int(_env("TB_PROBE_OUTPUT_LEN")),
            "batch_size": int(_env("TB_PROBE_BATCH")),
            "max_model_len": int(_env("TB_PROBE_MAX_MODEL_LEN")),
            "num_iters": int(_env("TB_PROBE_ITERS")),
            "num_iters_warmup": warmup,
        },
        "engine_args": {
            "model": _env("TB_MODEL"),
            "quantization": "modelopt_mixed",
            "dtype": "bfloat16",
            "tensor_parallel_size": int(_env("TB_TP")),
            "gpu_memory_utilization": float(_env("TB_GPU_MEM")),
            "max_num_seqs": int(_env("TB_PROBE_BATCH")),
            "compilation_config": json.loads(_env("TB_COMPILATION")),
        },
        "timing": {
            "engine_init_seconds": startup_seconds,
            "process_wall_seconds": ended - started,
        },
        "execution_validation": {
            # One entry per measured bench iteration. The CLI already discarded
            # its own warmup iterations, so none of these are cold.
            "runs": [
                {"repeat_index": i, "generation_seconds": value}
                for i, value in enumerate(latencies)
            ],
        },
        "vllm_bench_latency": raw,
    }

    out_path = Path(_env("TB_OUT"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"[annotate] {arm} idx={record['benchmark_context']['index']} "
        f"median={sorted(latencies)[len(latencies) // 2]:.3f}s "
        f"startup={startup_seconds:.1f}s -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
