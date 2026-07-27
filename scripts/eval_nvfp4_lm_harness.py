#!/usr/bin/env python3
"""Evaluate ModelOpt NVFP4 arms with lm-evaluation-harness through vLLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


for _name, _value in {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OPENBLAS_NUM_THREADS": "8",
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
    "NUMEXPR_NUM_THREADS": "8",
    "MALLOC_ARENA_MAX": "2",
}.items():
    os.environ[_name] = _value

from vllm.plugins.tensorbridge_evaluation.lm_harness import (
    ARMS,
    SUITES,
    RunConfig,
    run_evaluation,
    write_failure_artifact,
)


DEFAULT_MODEL = Path("/data/user/jzou521/models/nvidia/Qwen3.6-27B-NVFP4")
DEFAULT_CHECKPOINT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/manifests/qwen3.6-27b-nvfp4.sha256.json"
)


def _batch_size(value: str) -> str | int:
    if value == "auto" or value.startswith("auto:"):
        return value
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "batch size must be auto, auto:N, or positive int"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("batch size must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        default=DEFAULT_CHECKPOINT_MANIFEST,
    )
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--limit-count", type=int)
    limit.add_argument("--limit-fraction", type=float)
    limit.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--num-fewshot", type=int)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--batch-size", type=_batch_size, default="auto")
    parser.add_argument("--bootstrap-iters", type=int)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--think-end-token")
    parser.add_argument("--max-gen-toks", type=int)
    parser.add_argument("--allow-runtime-version-mismatch", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.bootstrap_iters is not None and args.bootstrap_iters < 0:
        raise ValueError("bootstrap_iters must be non-negative")
    samples_dir = args.samples_dir or args.output_json.parent / f"{args.output_json.stem}_samples"
    config = RunConfig(
        model=args.model,
        checkpoint_manifest=args.checkpoint_manifest,
        arm=args.arm,
        suite=args.suite,
        output_json=args.output_json,
        samples_dir=samples_dir,
        sample_manifest=args.sample_manifest,
        limit_count=args.limit_count,
        limit_fraction=args.limit_fraction,
        num_fewshot=args.num_fewshot,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        batch_size=args.batch_size,
        bootstrap_iters=args.bootstrap_iters,
        enable_thinking=args.enable_thinking,
        think_end_token=args.think_end_token,
        max_gen_toks=args.max_gen_toks,
        allow_runtime_version_mismatch=args.allow_runtime_version_mismatch,
        overwrite=args.overwrite,
    )
    try:
        result = run_evaluation(config)
    except Exception as error:
        write_failure_artifact(config, error)
        raise
    summary = {
        "status": result["status"],
        "arm": result["arm"]["key"],
        "suite": result["protocol"]["suite"],
        "tasks": result["protocol"]["tasks"],
        "checkpoint_content_sha256": result["checkpoint"]["start"][
            "checkpoint_content_sha256"
        ],
        "results": result["lm_eval"].get("results", {}),
        "output_json": str(args.output_json),
        "sample_artifacts": result["sample_artifacts"],
    }
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
