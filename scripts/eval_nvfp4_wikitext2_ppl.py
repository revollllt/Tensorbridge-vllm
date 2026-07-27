#!/usr/bin/env python3
"""Evaluate WikiText-2 perplexity through vLLM ModelOpt or TensorBridge NVFP4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import socket
import statistics
import struct
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


for _name, _value in {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}.items():
    os.environ.setdefault(_name, _value)

from vllm.plugins.tensorbridge_evaluation.ppl import (  # noqa: E402
    build_prompt_blocks,
    prompt_token_capacity,
    score_prompt_logprobs,
)
from tensorbridge.api.v1 import default_fpma_alpha  # noqa: E402


DEFAULT_MODEL = "/data/user/jzou521/models/nvidia/Qwen3.6-27B-NVFP4"
DEFAULT_DATASET_ARROW = (
    "/data/user/jzou521/.cache/huggingface/datasets/wikitext/"
    "wikitext-2-raw-v1/0.0.0/51aa13ed94b80c1a/wikitext-test.arrow"
)

RESULT_SCHEMA_VERSION = 2
EXECUTION_TRACE_SCHEMA_VERSION = 3
GRAPH_MEMORY_WARMUP_RUNS = 2
GRAPH_MIN_MEASURED_RUNS = 10
GRAPH_MIN_REPEAT_RUNS = GRAPH_MEMORY_WARMUP_RUNS + GRAPH_MIN_MEASURED_RUNS
GRAPH_MEMORY_MAX_GROWTH_MIB = 64
GRAPH_PPL_RATIO_UPPER = 1.02
GRAPH_NOISE_SD_RATIO_MAX = 1.5
GRAPH_RESAMPLE_COUNT = 10_000
GRAPH_PERMUTATION_ALPHA = 0.01
GRAPH_TOKEN_MEAN_SHIFT_P95_MAX = 0.25
GRAPH_TOKEN_MEAN_SHIFT_MAX = 0.5


class ExecutionValidationError(AssertionError):
    """An execution-mode failure with JSON-safe diagnostics."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__(message)


class ExecutionInconclusiveError(ExecutionValidationError):
    """A valid experiment that needs more evidence before pass or fail."""


def _positive_int_csv(value: str) -> list[int]:
    parts = [item.strip() for item in value.split(",")]
    if not parts or any(not item for item in parts):
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    try:
        values = [int(item) for item in parts]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all comma-separated integers must be positive")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("comma-separated integers must be unique")
    return values


def _cudagraph_compilation_config(args: argparse.Namespace) -> dict[str, Any] | None:
    config: dict[str, Any] = {}
    if args.execution_mode == "cudagraph":
        config.update(
            {
                "mode": args.compilation_mode,
                "cudagraph_mode": args.cudagraph_mode,
                "cudagraph_num_of_warmups": 1,
            }
        )
        if args.cudagraph_capture_sizes is not None:
            config["cudagraph_capture_sizes"] = args.cudagraph_capture_sizes
    if args.disable_allreduce_rms_fusion:
        # The optional FlashInfer fusion requires CUDA symmetric-memory
        # multicast, which is not reliable on every Hopper driver revision.
        config["pass_config"] = {"fuse_allreduce_rms": False}
    return config or None


def _assigned_gpu_devices(environ: Mapping[str, str] | None = None) -> str | None:
    environ = os.environ if environ is None else environ
    # Slurm can expose a physical GPU in SLURM_JOB_GPUS while renumbering the
    # job-visible device to zero, which is also how nvidia-smi reports it.
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "SLURM_STEP_GPUS",
        "SLURM_JOB_GPUS",
    ):
        value = environ.get(name)
        if value:
            return value
    return None


def _gpu_memory_snapshot(assigned_devices: str | None) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"nvidia-smi memory snapshot failed: {error}") from error
    snapshots = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",", 2)]
        if len(fields) != 3:
            raise RuntimeError(f"malformed nvidia-smi memory row: {line!r}")
        index, uuid, used_mib = fields
        snapshots.append(
            {"index": int(index), "uuid": uuid, "memory_used_mib": int(used_mib)}
        )
    assigned = (
        [item.strip() for item in assigned_devices.split(",") if item.strip()]
        if assigned_devices
        else []
    )
    if assigned:
        snapshots = [
            item
            for item in snapshots
            if any(
                token == str(item["index"])
                or token == item["uuid"]
                or item["uuid"].startswith(token)
                for token in assigned
            )
        ]
        if len(snapshots) != len(assigned):
            raise RuntimeError(
                "nvidia-smi did not resolve every assigned GPU: "
                f"assigned={assigned}, matched={snapshots}"
            )
    if not snapshots:
        raise RuntimeError("nvidia-smi returned no GPU memory rows")
    return snapshots


def _memory_stability_gate(
    snapshots: list[list[dict[str, Any]]],
    *,
    warmup_runs: int = GRAPH_MEMORY_WARMUP_RUNS,
    max_growth_mib: int = GRAPH_MEMORY_MAX_GROWTH_MIB,
) -> dict[str, Any]:
    if warmup_runs < 0 or max_growth_mib < 0:
        raise ValueError("memory gate parameters must be non-negative")
    evaluated = snapshots[warmup_runs:]
    if len(evaluated) < 3:
        raise ValueError("memory stability gate requires at least three post-warmup runs")
    expected_uuids = {item["uuid"] for item in evaluated[0]}
    per_gpu = []
    passed = True
    for snapshot in evaluated:
        if {item["uuid"] for item in snapshot} != expected_uuids:
            raise ValueError("GPU UUIDs changed across memory snapshots")
    for uuid in sorted(expected_uuids):
        values = [
            next(
                item["memory_used_mib"]
                for item in snapshot
                if item["uuid"] == uuid
            )
            for snapshot in evaluated
        ]
        growth_mib = values[-1] - values[0]
        span_mib = max(values) - min(values)
        strictly_increasing = all(right > left for left, right in zip(values, values[1:]))
        gpu_passed = (
            growth_mib <= max_growth_mib
            and span_mib <= max_growth_mib
            and not strictly_increasing
        )
        passed = passed and gpu_passed
        per_gpu.append(
            {
                "uuid": uuid,
                "memory_used_mib": values,
                "growth_mib": growth_mib,
                "span_mib": span_mib,
                "strictly_increasing": strictly_increasing,
                "passed": gpu_passed,
            }
        )
    return {
        "metric": "nvidia_smi_device_memory_used",
        "scope": "assigned_gpu_driver_level_smoke",
        "warmup_runs_excluded": warmup_runs,
        "evaluated_runs": len(evaluated),
        "max_growth_mib": max_growth_mib,
        "per_gpu": per_gpu,
        "passed": passed,
    }


def _cudagraph_metric_loggers(llm) -> list[tuple[int, Any]]:
    manager = getattr(llm.llm_engine, "logger_manager", None)
    found: list[tuple[int, Any]] = []
    if manager is None:
        return found
    for aggregate_logger in manager.stat_loggers:
        per_engine = getattr(aggregate_logger, "per_engine_stat_loggers", None)
        if per_engine is not None:
            candidates = per_engine.items()
        else:
            candidates = [(0, aggregate_logger)]
        for engine_index, logger in candidates:
            cudagraph_logging = getattr(logger, "cudagraph_logging", None)
            if cudagraph_logging is not None:
                found.append((int(engine_index), cudagraph_logging))
    return found


def _reset_cudagraph_runtime_stats(llm) -> None:
    loggers = _cudagraph_metric_loggers(llm)
    if not loggers:
        raise RuntimeError("vLLM CUDA Graph metric logger is unavailable")
    for _, logger in loggers:
        logger.reset()


def _cudagraph_runtime_stats(llm) -> dict[str, Any]:
    records = []
    for engine_index, logger in _cudagraph_metric_loggers(llm):
        for stat in logger.stats:
            records.append(
                {
                    "engine_index": engine_index,
                    "num_unpadded_tokens": int(stat.num_unpadded_tokens),
                    "num_padded_tokens": int(stat.num_padded_tokens),
                    "num_paddings": int(stat.num_paddings),
                    "runtime_mode": str(stat.runtime_mode),
                }
            )
    mode_counts: dict[str, int] = {}
    for record in records:
        mode = record["runtime_mode"]
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    graph_dispatches = sum(
        count for mode, count in mode_counts.items() if mode != "NONE"
    )
    return {
        "records": records,
        "runtime_mode_counts": dict(sorted(mode_counts.items())),
        "total_iterations": len(records),
        "graph_dispatches": graph_dispatches,
    }


def _cudagraph_runtime_gate(
    runtime_stats: dict[str, Any],
    *,
    cudagraph_mode: str,
    expected_decode_dispatches: int,
    expected_decode_batch: int,
) -> dict[str, Any]:
    records = runtime_stats["records"]
    errors = []
    if cudagraph_mode == "FULL_DECODE_ONLY":
        if len(records) != expected_decode_dispatches + 1:
            errors.append(
                f"expected {expected_decode_dispatches + 1} scheduler iterations, "
                f"got {len(records)}"
            )
        if not records or records[0]["runtime_mode"] != "NONE":
            errors.append("prefill iteration must use runtime mode NONE")
        for index, record in enumerate(records[1:], start=1):
            if record["runtime_mode"] != "FULL":
                errors.append(f"decode iteration {index} did not use FULL")
            if record["num_unpadded_tokens"] != expected_decode_batch:
                errors.append(
                    f"decode iteration {index} unpadded batch is "
                    f"{record['num_unpadded_tokens']}, expected {expected_decode_batch}"
                )
            if record["num_padded_tokens"] != expected_decode_batch:
                errors.append(
                    f"decode iteration {index} padded batch is "
                    f"{record['num_padded_tokens']}, expected {expected_decode_batch}"
                )
            if record["num_paddings"] != 0:
                errors.append(f"decode iteration {index} unexpectedly used padding")
    elif runtime_stats["graph_dispatches"] < expected_decode_dispatches:
        errors.append(
            f"observed {runtime_stats['graph_dispatches']} graph dispatches, "
            f"expected at least {expected_decode_dispatches}"
        )
    return {
        "passed": not errors,
        "cudagraph_mode": cudagraph_mode,
        "expected_decode_dispatches": expected_decode_dispatches,
        "expected_decode_batch": expected_decode_batch,
        "errors": errors,
    }


def _validation_trace(blocks, outputs) -> dict[str, Any]:
    target_logprobs: list[float] = []
    targets: list[dict[str, Any]] = []
    generated_token_ids: list[list[list[int]]] = []
    generated_token_logprobs: list[list[list[float]]] = []
    digest = hashlib.sha256()
    digest.update(b"tensorbridge-execution-trace-v3\0")
    for block_index, (block, output) in enumerate(zip(blocks, outputs, strict=True)):
        prompt_logprobs = output.prompt_logprobs
        output_ids = [int(token) for token in output.prompt_token_ids]
        digest.update(struct.pack("<qq", block_index, block.scored_tokens))
        for position in range(block.local_target_start, len(output_ids)):
            target_token = output_ids[position]
            value = prompt_logprobs[position][target_token]
            logprob = float(getattr(value, "logprob", value))
            global_token_offset = (
                block.global_target_start + position - block.local_target_start
            )
            target_logprobs.append(logprob)
            targets.append(
                {
                    "block_index": block_index,
                    "global_token_offset": global_token_offset,
                    "target_token_id": target_token,
                    "logprob": logprob,
                }
            )
            digest.update(struct.pack("<qqd", global_token_offset, target_token, logprob))
        request_tokens = []
        request_logprobs = []
        completions = getattr(output, "outputs", [])
        digest.update(struct.pack("<q", len(completions)))
        for completion in completions:
            token_ids = [int(token) for token in completion.token_ids]
            completion_logprobs = getattr(completion, "logprobs", None)
            if completion_logprobs is None:
                raise RuntimeError("generated-token logprobs were not returned by vLLM")
            if len(completion_logprobs) != len(token_ids):
                raise RuntimeError(
                    "generated-token/logprob length mismatch: "
                    f"{len(token_ids)} vs {len(completion_logprobs)}"
                )
            request_tokens.append(token_ids)
            token_logprobs = []
            digest.update(struct.pack("<q", len(token_ids)))
            for token, position_logprobs in zip(
                token_ids, completion_logprobs, strict=True
            ):
                if token not in position_logprobs:
                    raise RuntimeError(
                        f"generated token {token} is missing from returned logprobs"
                    )
                value = position_logprobs[token]
                logprob = float(getattr(value, "logprob", value))
                if not math.isfinite(logprob):
                    raise RuntimeError(
                        f"non-finite generated-token logprob for token {token}: {logprob}"
                    )
                token_logprobs.append(logprob)
                digest.update(struct.pack("<qd", token, logprob))
            request_logprobs.append(token_logprobs)
        generated_token_ids.append(request_tokens)
        generated_token_logprobs.append(request_logprobs)
    return {
        "sha256": digest.hexdigest(),
        "target_logprobs": target_logprobs,
        "targets": targets,
        "generated_token_ids": generated_token_ids,
        "generated_token_logprobs": generated_token_logprobs,
    }


def _primary_trace(result: dict[str, Any]) -> dict[str, Any]:
    validation = result["execution_validation"]
    primary_index = int(validation["primary_run_index"])
    return validation["runs"][primary_index]


def _preflight_eager_reference(
    reference: Any, reference_path: Path
) -> dict[str, Any]:
    errors = []
    if not isinstance(reference, dict):
        errors.append("top-level JSON must be an object")
    else:
        if reference.get("schema_version") != RESULT_SCHEMA_VERSION:
            errors.append(
                f"schema_version must be {RESULT_SCHEMA_VERSION}, "
                f"got {reference.get('schema_version')!r}"
            )
        if reference.get("status") != "passed":
            errors.append("reference status must be passed")
        validation = reference.get("execution_validation")
        if not isinstance(validation, dict):
            errors.append("execution_validation is missing")
        else:
            if validation.get("execution_trace_schema_version") != (
                EXECUTION_TRACE_SCHEMA_VERSION
            ):
                errors.append(
                    "execution_trace_schema_version must be "
                    f"{EXECUTION_TRACE_SCHEMA_VERSION}"
                )
            if validation.get("requested_mode") != "eager":
                errors.append("reference requested_mode must be eager")
            runs = validation.get("runs")
            if not isinstance(runs, list) or len(runs) < GRAPH_MIN_REPEAT_RUNS:
                errors.append(
                    f"reference requires at least {GRAPH_MIN_REPEAT_RUNS} runs"
                )
            else:
                if validation.get("repeat_runs") != len(runs):
                    errors.append("repeat_runs does not match the run records")
                for index, run in enumerate(runs):
                    if not isinstance(run, dict):
                        errors.append(f"run {index} is not an object")
                        continue
                    metrics = run.get("metrics")
                    mean_nll = metrics.get("mean_nll") if isinstance(metrics, dict) else None
                    if not isinstance(mean_nll, (int, float)) or not math.isfinite(
                        float(mean_nll)
                    ):
                        errors.append(f"run {index} has no finite mean_nll")
                    targets = run.get("targets")
                    if not isinstance(targets, list) or not targets:
                        errors.append(f"run {index} has no target trace")
                    token_ids = run.get("generated_token_ids")
                    token_logprobs = run.get("generated_token_logprobs")
                    if not isinstance(token_ids, list) or not isinstance(
                        token_logprobs, list
                    ):
                        errors.append(
                            f"run {index} lacks trace-v3 generated-token data"
                        )
                    elif len(token_ids) != len(token_logprobs):
                        errors.append(
                            f"run {index} generated request counts do not align"
                        )
                try:
                    measured = runs[GRAPH_MEMORY_WARMUP_RUNS:]
                    expected_tokens = reference.get("blocking", {}).get(
                        "requested_output_tokens"
                    )
                    canonical_targets = _target_identities(measured[0])
                    canonical_ids, _ = _generated_sequences(
                        measured[0], expected_tokens=expected_tokens
                    )
                    for index, run in enumerate(measured, start=GRAPH_MEMORY_WARMUP_RUNS):
                        if _target_identities(run) != canonical_targets:
                            errors.append(f"run {index} target identities changed")
                        generated_ids, _ = _generated_sequences(
                            run, expected_tokens=expected_tokens
                        )
                        if generated_ids != canonical_ids:
                            errors.append(f"run {index} generated token IDs changed")
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(f"trace-v3 structure is invalid: {error}")
    if errors:
        raise ExecutionValidationError(
            "eager reference is incompatible with execution trace v3",
            {"reference_path": str(reference_path), "preflight_errors": errors},
        )
    return reference


def _measured_runs(result: dict[str, Any]) -> list[dict[str, Any]]:
    runs = result["execution_validation"]["runs"]
    return runs[GRAPH_MEMORY_WARMUP_RUNS:]


def _target_identities(run: dict[str, Any]) -> list[tuple[int, int, int]]:
    return [
        (
            int(target["block_index"]),
            int(target["global_token_offset"]),
            int(target["target_token_id"]),
        )
        for target in run["targets"]
    ]


def _generated_sequences(
    run: dict[str, Any], *, expected_tokens: int | None = None
) -> tuple[list[list[int]], list[list[float]]]:
    token_ids = run["generated_token_ids"]
    token_logprobs = run["generated_token_logprobs"]
    if len(token_ids) != len(token_logprobs):
        raise ValueError("generated request counts do not align")
    sequences = []
    logprob_sequences = []
    for request_index, (request_ids, request_logprobs) in enumerate(
        zip(token_ids, token_logprobs, strict=True)
    ):
        if len(request_ids) != 1 or len(request_logprobs) != 1:
            raise ValueError(
                f"request {request_index} must contain exactly one completion"
            )
        sequence = [int(token) for token in request_ids[0]]
        logprobs = [float(value) for value in request_logprobs[0]]
        if len(sequence) != len(logprobs):
            raise ValueError(
                f"request {request_index} generated token/logprob lengths differ"
            )
        if len(sequence) < 2:
            raise ValueError("decode validation requires at least two generated tokens")
        if expected_tokens is not None and len(sequence) != expected_tokens:
            raise ValueError(
                f"request {request_index} generated {len(sequence)} tokens, "
                f"expected {expected_tokens}"
            )
        if not all(math.isfinite(value) for value in logprobs):
            raise ValueError("generated-token logprobs must be finite")
        sequences.append(sequence)
        logprob_sequences.append(logprobs)
    return sequences, logprob_sequences


def _generated_nll_parts(run: dict[str, Any]) -> tuple[float, float]:
    _, logprob_sequences = _generated_sequences(run)
    prefill = [-values[0] for values in logprob_sequences]
    decode = [-value for values in logprob_sequences for value in values[1:]]
    return statistics.fmean(prefill), statistics.fmean(decode)


def _welch_equivalence(
    reference_values: list[float], candidate_values: list[float]
) -> dict[str, Any]:
    if (
        len(reference_values) < GRAPH_MIN_MEASURED_RUNS
        or len(candidate_values) < GRAPH_MIN_MEASURED_RUNS
    ):
        raise ValueError(
            f"equivalence requires {GRAPH_MIN_MEASURED_RUNS} measured runs per arm"
        )
    reference_mean = statistics.fmean(reference_values)
    candidate_mean = statistics.fmean(candidate_values)
    reference_variance = statistics.variance(reference_values)
    candidate_variance = statistics.variance(candidate_values)
    reference_term = reference_variance / len(reference_values)
    candidate_term = candidate_variance / len(candidate_values)
    standard_error = math.sqrt(reference_term + candidate_term)
    delta = candidate_mean - reference_mean
    if standard_error == 0.0:
        degrees_of_freedom = None
        critical_value = 0.0
        ci_lower = delta
        ci_upper = delta
    else:
        denominator = (
            reference_term * reference_term / (len(reference_values) - 1)
            + candidate_term * candidate_term / (len(candidate_values) - 1)
        )
        degrees_of_freedom = (
            (reference_term + candidate_term) ** 2 / denominator
            if denominator > 0.0
            else math.inf
        )
        from scipy.stats import t as student_t

        critical_value = float(student_t.ppf(0.95, degrees_of_freedom))
        ci_lower = delta - critical_value * standard_error
        ci_upper = delta + critical_value * standard_error
    # Requiring both reciprocal geometric ratios to stay within 2% gives a
    # symmetric NLL margin and avoids favoring either execution arm.
    upper_bound = math.log(GRAPH_PPL_RATIO_UPPER)
    lower_bound = -upper_bound
    passed = ci_lower >= lower_bound and ci_upper <= upper_bound
    entirely_outside = ci_lower >= upper_bound or ci_upper <= lower_bound
    decision = "passed" if passed else ("failed" if entirely_outside else "inconclusive")
    return {
        "decision": decision,
        "passed": passed,
        "reference_runs": len(reference_values),
        "candidate_runs": len(candidate_values),
        "reference_mean_nll": reference_mean,
        "candidate_mean_nll": candidate_mean,
        "delta_mean_nll": delta,
        "reference_sd": math.sqrt(reference_variance),
        "candidate_sd": math.sqrt(candidate_variance),
        "standard_error": standard_error,
        "degrees_of_freedom": degrees_of_freedom,
        "critical_value_90pct": critical_value,
        "delta_nll_ci90": [ci_lower, ci_upper],
        "ppl_ratio": _safe_exp(delta),
        "ppl_ratio_ci90": [_safe_exp(ci_lower), _safe_exp(ci_upper)],
        "chosen_probability_ratio": _safe_exp(-delta),
        "chosen_probability_ratio_ci90": [
            _safe_exp(-ci_upper),
            _safe_exp(-ci_lower),
        ],
        "equivalence_bounds_nll": [lower_bound, upper_bound],
        "equivalence_bounds_ppl_ratio": [
            1.0 / GRAPH_PPL_RATIO_UPPER,
            GRAPH_PPL_RATIO_UPPER,
        ],
    }


def _safe_exp(value: float) -> float | None:
    try:
        result = math.exp(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _upper_ratio_percentile(
    values: list[float | None], percentile: float
) -> float | None:
    """Treat undefined ratios as unbounded instead of silently dropping them."""
    finite = sorted(
        value for value in values if value is not None and math.isfinite(value)
    )
    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if upper >= len(finite):
        return None
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _bootstrap_scalar_sd_ratio(
    reference_values: list[float], candidate_values: list[float], *, seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)

    def ratio(left: list[float], right: list[float]) -> float | None:
        reference_sd = statistics.stdev(left)
        candidate_sd = statistics.stdev(right)
        if reference_sd == 0.0:
            return 1.0 if candidate_sd == 0.0 else None
        result = candidate_sd / reference_sd
        return result if math.isfinite(result) else None

    point_ratio = ratio(reference_values, candidate_values)
    ratios: list[float | None] = []
    for _ in range(GRAPH_RESAMPLE_COUNT):
        left = [rng.choice(reference_values) for _ in reference_values]
        right = [rng.choice(candidate_values) for _ in candidate_values]
        value = ratio(left, right)
        ratios.append(value if value is not None and math.isfinite(value) else None)
    upper90 = _upper_ratio_percentile(ratios, 0.90)
    valid_resamples = sum(value is not None for value in ratios)
    return {
        "method": "independent_run_bootstrap",
        "resamples": GRAPH_RESAMPLE_COUNT,
        "seed": seed,
        "point_ratio": point_ratio,
        "upper90": upper90,
        "max_upper90": GRAPH_NOISE_SD_RATIO_MAX,
        "valid_resamples": valid_resamples,
        "invalid_unbounded_resamples": len(ratios) - valid_resamples,
        "passed": point_ratio is not None
        and upper90 is not None
        and upper90 <= GRAPH_NOISE_SD_RATIO_MAX,
    }


def _decode_logprob_vectors(runs: list[dict[str, Any]]) -> list[list[float]]:
    vectors = []
    for run in runs:
        _, sequences = _generated_sequences(run)
        vectors.append([value for sequence in sequences for value in sequence[1:]])
    width = len(vectors[0])
    if width == 0 or any(len(vector) != width for vector in vectors):
        raise ValueError("decode logprob vectors do not align")
    return vectors


def _vector_rms_noise(vectors: list[list[float]]) -> float:
    variances = [
        statistics.variance(vector[index] for vector in vectors)
        for index in range(len(vectors[0]))
    ]
    return math.sqrt(statistics.fmean(variances))


def _bootstrap_vector_rms_noise_ratio(
    reference_vectors: list[list[float]],
    candidate_vectors: list[list[float]],
    *,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)

    def ratio(left: list[list[float]], right: list[list[float]]) -> float | None:
        reference_noise = _vector_rms_noise(left)
        candidate_noise = _vector_rms_noise(right)
        if reference_noise == 0.0:
            return 1.0 if candidate_noise == 0.0 else None
        result = candidate_noise / reference_noise
        return result if math.isfinite(result) else None

    point_ratio = ratio(reference_vectors, candidate_vectors)
    ratios: list[float | None] = []
    for _ in range(GRAPH_RESAMPLE_COUNT):
        left = [rng.choice(reference_vectors) for _ in reference_vectors]
        right = [rng.choice(candidate_vectors) for _ in candidate_vectors]
        value = ratio(left, right)
        ratios.append(value if value is not None and math.isfinite(value) else None)
    upper90 = _upper_ratio_percentile(ratios, 0.90)
    valid_resamples = sum(value is not None for value in ratios)
    return {
        "method": "independent_run_vector_bootstrap",
        "resamples": GRAPH_RESAMPLE_COUNT,
        "seed": seed,
        "point_ratio": point_ratio,
        "upper90": upper90,
        "max_upper90": GRAPH_NOISE_SD_RATIO_MAX,
        "valid_resamples": valid_resamples,
        "invalid_unbounded_resamples": len(ratios) - valid_resamples,
        "passed": point_ratio is not None
        and upper90 is not None
        and upper90 <= GRAPH_NOISE_SD_RATIO_MAX,
    }


def _decode_vector_permutation_test(
    reference_vectors: list[list[float]],
    candidate_vectors: list[list[float]],
    *,
    seed: int,
) -> dict[str, Any]:
    width = len(reference_vectors[0])

    def statistics_for(
        left: list[list[float]], right: list[list[float]]
    ) -> tuple[list[float], float, float]:
        shifts = []
        t_values = []
        for index in range(width):
            left_values = [row[index] for row in left]
            right_values = [row[index] for row in right]
            shift = statistics.fmean(right_values) - statistics.fmean(left_values)
            shifts.append(shift)
            standard_error = math.sqrt(
                statistics.variance(left_values) / len(left_values)
                + statistics.variance(right_values) / len(right_values)
            )
            if standard_error == 0.0:
                t_values.append(0.0 if shift == 0.0 else math.inf)
            else:
                t_values.append(abs(shift) / standard_error)
        rms = math.sqrt(statistics.fmean(value * value for value in shifts))
        return shifts, rms, max(t_values)

    signed_shifts, observed_rms, observed_max_t = statistics_for(
        reference_vectors, candidate_vectors
    )
    pooled = reference_vectors + candidate_vectors
    left_size = len(reference_vectors)
    indices = list(range(len(pooled)))
    rng = random.Random(seed)
    rms_exceedances = 0
    max_t_exceedances = 0
    for _ in range(GRAPH_RESAMPLE_COUNT):
        rng.shuffle(indices)
        left = [pooled[index] for index in indices[:left_size]]
        right = [pooled[index] for index in indices[left_size:]]
        _, rms, max_t = statistics_for(left, right)
        rms_exceedances += rms >= observed_rms
        max_t_exceedances += max_t >= observed_max_t
    rms_p = (rms_exceedances + 1) / (GRAPH_RESAMPLE_COUNT + 1)
    max_t_p = (max_t_exceedances + 1) / (GRAPH_RESAMPLE_COUNT + 1)
    return {
        "method": "arm_label_permutation",
        "permutations": GRAPH_RESAMPLE_COUNT,
        "seed": seed,
        "signed_position_mean_shifts": signed_shifts,
        "observed_rms_mean_shift": observed_rms,
        "observed_max_t": None if not math.isfinite(observed_max_t) else observed_max_t,
        "rms_p_value": rms_p,
        "max_t_p_value": max_t_p,
        "alpha": GRAPH_PERMUTATION_ALPHA,
        "passed": rms_p >= GRAPH_PERMUTATION_ALPHA
        and max_t_p >= GRAPH_PERMUTATION_ALPHA,
    }


def _generated_position_mean_shift(
    reference_runs: list[dict[str, Any]], candidate_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    reference_vectors = [_generated_sequences(run)[1] for run in reference_runs]
    candidate_vectors = [_generated_sequences(run)[1] for run in candidate_runs]
    template = reference_vectors[0]
    if any(
        [len(values) for values in vector] != [len(values) for values in template]
        for vector in reference_vectors + candidate_vectors
    ):
        raise ValueError("generated-token logprob shapes do not align across runs")
    shifts = []
    for request_index, values in enumerate(template):
        for token_index in range(1, len(values)):
            reference_mean = statistics.fmean(
                run[request_index][token_index] for run in reference_vectors
            )
            candidate_mean = statistics.fmean(
                run[request_index][token_index] for run in candidate_vectors
            )
            shifts.append(abs(candidate_mean - reference_mean))
    p95 = _percentile(shifts, 0.95)
    maximum = max(shifts)
    return {
        "decode_positions": len(shifts),
        "mean_abs": statistics.fmean(shifts),
        "p95_abs": p95,
        "max_abs": maximum,
        "p95_limit": GRAPH_TOKEN_MEAN_SHIFT_P95_MAX,
        "max_limit": GRAPH_TOKEN_MEAN_SHIFT_MAX,
        "passed": p95 <= GRAPH_TOKEN_MEAN_SHIFT_P95_MAX
        and maximum <= GRAPH_TOKEN_MEAN_SHIFT_MAX,
    }


def _trace_difference(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_targets = reference["targets"]
    candidate_targets = candidate["targets"]
    first_target_difference = None
    max_abs_logprob = 0.0
    max_abs_logprob_location = None
    for index, (left, right) in enumerate(
        zip(reference_targets, candidate_targets, strict=False)
    ):
        identities = (
            (left["block_index"], left["global_token_offset"], left["target_token_id"]),
            (right["block_index"], right["global_token_offset"], right["target_token_id"]),
        )
        difference = abs(left["logprob"] - right["logprob"])
        if identities[0] != identities[1] or difference != 0.0:
            if first_target_difference is None:
                first_target_difference = {
                    "target_index": index,
                    "reference": left,
                    "candidate": right,
                    "abs_logprob_difference": difference,
                }
            if difference > max_abs_logprob:
                max_abs_logprob = difference
                max_abs_logprob_location = {
                    "target_index": index,
                    "reference": left,
                    "candidate": right,
                    "abs_logprob_difference": difference,
                }
    generated_exact = (
        reference["generated_token_ids"] == candidate["generated_token_ids"]
    )
    generated_logprobs_exact = (
        reference["generated_token_logprobs"]
        == candidate["generated_token_logprobs"]
    )
    max_abs_generated_logprob = None
    max_abs_generated_logprob_location = None
    if generated_exact:
        max_abs_generated_logprob = 0.0
        for block_index, (left_completions, right_completions) in enumerate(
            zip(
                reference["generated_token_logprobs"],
                candidate["generated_token_logprobs"],
                strict=True,
            )
        ):
            for completion_index, (left_values, right_values) in enumerate(
                zip(left_completions, right_completions, strict=True)
            ):
                for token_index, (left, right) in enumerate(
                    zip(left_values, right_values, strict=True)
                ):
                    difference = abs(left - right)
                    if difference > max_abs_generated_logprob:
                        max_abs_generated_logprob = difference
                        max_abs_generated_logprob_location = {
                            "block_index": block_index,
                            "completion_index": completion_index,
                            "token_index": token_index,
                            "token_id": reference["generated_token_ids"][block_index][
                                completion_index
                            ][token_index],
                            "reference_logprob": left,
                            "candidate_logprob": right,
                            "abs_logprob_difference": difference,
                        }
    return {
        "reference_target_count": len(reference_targets),
        "candidate_target_count": len(candidate_targets),
        "first_target_difference": first_target_difference,
        "max_abs_target_logprob": max_abs_logprob,
        "max_abs_target_logprob_location": max_abs_logprob_location,
        "generated_token_ids_exact": generated_exact,
        "generated_token_logprobs_exact": generated_logprobs_exact,
        "max_abs_generated_token_logprob": max_abs_generated_logprob,
        "max_abs_generated_token_logprob_location": (
            max_abs_generated_logprob_location
        ),
        "reference_generated_token_ids": reference["generated_token_ids"],
        "candidate_generated_token_ids": candidate["generated_token_ids"],
    }


def _compare_execution_results(
    eager: dict[str, Any], graph: dict[str, Any]
) -> dict[str, Any]:
    protocol_paths = (
        ("schema_version",),
        ("checkpoint_mode",),
        ("model_path",),
        ("runtime", "vllm"),
        ("runtime", "quant_config_class"),
        ("production_contract",),
        ("dataset",),
        ("blocking",),
        ("engine_args", "tensor_parallel_size"),
        ("engine_args", "max_model_len"),
        ("engine_args", "max_num_seqs"),
        ("engine_args", "gpu_memory_utilization"),
        ("execution_validation", "execution_trace_schema_version"),
        ("execution_validation", "sampling_logprob_contract"),
    )

    def lookup(value: dict[str, Any], path: tuple[str, ...]) -> Any:
        current: Any = value
        for key in path:
            current = current[key]
        return current

    protocol_mismatches = []
    if eager.get("status") != "passed":
        protocol_mismatches.append(
            {"path": "status", "eager": eager.get("status"), "expected": "passed"}
        )
    eager_mode = eager.get("execution_validation", {}).get("requested_mode")
    if eager_mode != "eager":
        protocol_mismatches.append(
            {
                "path": "execution_validation.requested_mode",
                "eager": eager_mode,
                "expected": "eager",
            }
        )
    graph_mode = graph.get("execution_validation", {}).get("requested_mode")
    if graph_mode != "cudagraph":
        protocol_mismatches.append(
            {
                "path": "execution_validation.requested_mode",
                "cudagraph": graph_mode,
                "expected": "cudagraph",
            }
        )
    for path in protocol_paths:
        eager_value = lookup(eager, path)
        graph_value = lookup(graph, path)
        if eager_value != graph_value:
            protocol_mismatches.append(
                {
                    "path": ".".join(path),
                    "eager": eager_value,
                    "cudagraph": graph_value,
                }
            )

    eager_runs = _measured_runs(eager)
    graph_runs = _measured_runs(graph)
    expected_generated_tokens = int(eager["blocking"]["requested_output_tokens"])
    alignment_errors = []
    canonical_targets = _target_identities(eager_runs[0])
    canonical_generated_ids, _ = _generated_sequences(
        eager_runs[0], expected_tokens=expected_generated_tokens
    )
    for arm, runs in (("eager", eager_runs), ("cudagraph", graph_runs)):
        for run_index, run in enumerate(runs):
            if _target_identities(run) != canonical_targets:
                alignment_errors.append(
                    f"{arm} measured run {run_index} target identities differ"
                )
            generated_ids, _ = _generated_sequences(
                run, expected_tokens=expected_generated_tokens
            )
            if generated_ids != canonical_generated_ids:
                alignment_errors.append(
                    f"{arm} measured run {run_index} generated token IDs differ"
                )

    generated_token_ids_exact = not any(
        "generated token IDs" in error for error in alignment_errors
    )
    if alignment_errors:
        return {
            "decision": "failed",
            "passed": False,
            "protocol_mismatches": protocol_mismatches,
            "alignment_errors": alignment_errors,
            "generated_token_ids_exact": generated_token_ids_exact,
            "warmup_runs_excluded": GRAPH_MEMORY_WARMUP_RUNS,
            "measured_runs_per_arm": {
                "eager": len(eager_runs),
                "cudagraph": len(graph_runs),
            },
            "equivalence": None,
            "noise_noninferiority": None,
            "generated_decode_position_mean_shift": None,
            "statistics_skipped": "token identities or conditioning contexts differ",
        }
    eager_prompt_nll = [float(run["metrics"]["mean_nll"]) for run in eager_runs]
    graph_prompt_nll = [float(run["metrics"]["mean_nll"]) for run in graph_runs]
    prompt_equivalence = _welch_equivalence(eager_prompt_nll, graph_prompt_nll)
    eager_generated_parts = [_generated_nll_parts(run) for run in eager_runs]
    graph_generated_parts = [_generated_nll_parts(run) for run in graph_runs]
    eager_prefill_nll = [value[0] for value in eager_generated_parts]
    graph_prefill_nll = [value[0] for value in graph_generated_parts]
    eager_decode_nll = [value[1] for value in eager_generated_parts]
    graph_decode_nll = [value[1] for value in graph_generated_parts]
    prefill_equivalence = _welch_equivalence(
        eager_prefill_nll,
        graph_prefill_nll,
    )
    decode_equivalence = _welch_equivalence(
        eager_decode_nll,
        graph_decode_nll,
    )
    decode_position_shift = _generated_position_mean_shift(eager_runs, graph_runs)
    eager_decode_vectors = _decode_logprob_vectors(eager_runs)
    graph_decode_vectors = _decode_logprob_vectors(graph_runs)
    noise = {
        "prompt": _bootstrap_scalar_sd_ratio(
            eager_prompt_nll, graph_prompt_nll, seed=2026072201
        ),
        "generated_prefill_position0": _bootstrap_scalar_sd_ratio(
            eager_prefill_nll, graph_prefill_nll, seed=2026072202
        ),
        "generated_decode_positions1plus": _bootstrap_scalar_sd_ratio(
            eager_decode_nll, graph_decode_nll, seed=2026072203
        ),
        "generated_decode_vector_rms": _bootstrap_vector_rms_noise_ratio(
            eager_decode_vectors, graph_decode_vectors, seed=2026072204
        ),
    }
    decode_mode_effect = _decode_vector_permutation_test(
        eager_decode_vectors, graph_decode_vectors, seed=2026072205
    )
    equivalences = {
        "prompt_target_nll": prompt_equivalence,
        "generated_prefill_position0_nll": prefill_equivalence,
        "generated_decode_positions1plus_nll": decode_equivalence,
    }
    hard_gate_passed = (
        not protocol_mismatches
        and not alignment_errors
        and generated_token_ids_exact
        and decode_mode_effect["passed"]
    )
    passed = (
        hard_gate_passed
        and all(item["passed"] for item in noise.values())
        and all(item["passed"] for item in equivalences.values())
    )
    if passed:
        decision = "passed"
    elif not hard_gate_passed or decode_equivalence["decision"] == "failed":
        decision = "failed"
    else:
        # Prefill executes eagerly in FULL_DECODE_ONLY. A mismatched prefill
        # control or a wide CI cannot be attributed to graph replay.
        decision = "inconclusive"
    return {
        "decision": decision,
        "passed": passed,
        "protocol_mismatches": protocol_mismatches,
        "alignment_errors": alignment_errors,
        "generated_token_ids_exact": generated_token_ids_exact,
        "warmup_runs_excluded": GRAPH_MEMORY_WARMUP_RUNS,
        "measured_runs_per_arm": {
            "eager": len(eager_runs),
            "cudagraph": len(graph_runs),
        },
        "equivalence": equivalences,
        "noise_noninferiority": noise,
        "generated_decode_mode_effect": decode_mode_effect,
        "generated_decode_position_mean_shift": {
            **decode_position_shift,
            "advisory_only": True,
        },
        "streamk_contract": (
            "floating-point multi-slice reduction is not bit deterministic; "
            "compare eager and graph distributions"
        ),
    }


def _resolve_fpma_alpha(
    backend: str,
    alpha: float | None,
    *,
    prefold_selector: str = "none",
    ulp_correction: bool = False,
) -> float:
    if alpha is not None:
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError(f"invalid FPMA alpha: {alpha!r}")
        return alpha
    if backend != "tensorbridge":
        return 1.0
    return default_fpma_alpha(
        prefold_selector=prefold_selector,
        ulp_correction=ulp_correction,
    )


def _resolve_fpma_alpha_input(
    backend: str,
    cli_alpha: float | None,
    *,
    env_alpha: str | None,
    prefold_selector: str = "none",
    ulp_correction: bool = False,
) -> tuple[float, str]:
    if cli_alpha is not None:
        source = "explicit_cli"
        candidate = cli_alpha
    elif env_alpha is not None:
        source = "explicit_env"
        try:
            candidate = float(env_alpha)
        except ValueError as error:
            raise ValueError(
                f"invalid TENSORBRIDGE_NVFP4_FPMA_ALPHA={env_alpha!r}"
            ) from error
    else:
        candidate = None
        source = (
            "analytic_v1"
            if backend == "tensorbridge"
            and prefold_selector == "none"
            and not ulp_correction
            else "neutral_default"
        )
    alpha = _resolve_fpma_alpha(
        backend,
        candidate,
        prefold_selector=prefold_selector,
        ulp_correction=ulp_correction,
    )
    return alpha, source


def _configure_fpma_alpha_environment(alpha: float, source: str) -> None:
    if source in {"explicit_cli", "explicit_env"}:
        os.environ["TENSORBRIDGE_NVFP4_FPMA_ALPHA"] = str(alpha)
    elif source in {"analytic_v1", "neutral_default"}:
        # Preserve the integration's implicit-default marker so analytic_v1
        # still performs its verified-scale-domain check during weight loading.
        os.environ.pop("TENSORBRIDGE_NVFP4_FPMA_ALPHA", None)
    else:
        raise ValueError(f"unknown FPMA alpha source: {source!r}")


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _write_result(result: dict[str, Any], output: Path | None) -> None:
    encoded = json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded, flush=True)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    eager_reference = None
    eager_reference_bytes = None
    if not args.model.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {args.model}")
    if not args.dataset_arrow.is_file():
        raise FileNotFoundError(f"WikiText-2 Arrow file does not exist: {args.dataset_arrow}")

    if args.backend != "tensorbridge" and (
        args.fpma_alpha != 1.0
        or args.fpma_prefold_selector != "none"
        or args.fpma_ulp_correction
    ):
        raise ValueError("FPMA compensation options apply only to the tensorbridge backend")
    if args.repeat_runs <= 0:
        raise ValueError("repeat_runs must be positive")
    if args.requested_output_tokens <= 0:
        raise ValueError("requested_output_tokens must be positive")
    if args.execution_mode == "eager" and args.cudagraph_capture_sizes is not None:
        raise ValueError("cudagraph capture sizes require execution_mode=cudagraph")
    if args.execution_mode == "eager" and args.eager_reference_json is not None:
        raise ValueError("eager reference applies only to execution_mode=cudagraph")
    if args.execution_mode == "cudagraph":
        if args.compilation_mode == "NONE" and args.cudagraph_mode in {
            "PIECEWISE",
            "FULL_AND_PIECEWISE",
        }:
            raise ValueError(
                f"{args.cudagraph_mode} requires compilation_mode=VLLM_COMPILE"
            )
        if args.repeat_runs < GRAPH_MIN_REPEAT_RUNS:
            raise ValueError(
                f"CUDA Graph validation requires at least {GRAPH_MIN_REPEAT_RUNS} runs"
            )
        if args.requested_output_tokens < 2:
            raise ValueError("CUDA Graph validation requires at least one decode forward")
        if not args.validation_details:
            raise ValueError("CUDA Graph validation requires --validation-details")
        if args.eager_reference_json is None:
            raise ValueError("CUDA Graph validation requires --eager-reference-json")
        if not args.eager_reference_json.is_file():
            raise FileNotFoundError(
                f"eager reference does not exist: {args.eager_reference_json}"
            )
        eager_reference_bytes = args.eager_reference_json.read_bytes()
        try:
            eager_reference_json = json.loads(eager_reference_bytes)
        except json.JSONDecodeError as error:
            raise ExecutionValidationError(
                "eager reference is not valid JSON",
                {
                    "reference_path": str(args.eager_reference_json),
                    "json_error": str(error),
                },
            ) from error
        eager_reference = _preflight_eager_reference(
            eager_reference_json, args.eager_reference_json
        )
        # Preserve per-iteration CUDA Graph stats until each short repeat consumes them.
        os.environ["VLLM_LOG_STATS_INTERVAL"] = "3600"
    os.environ["TENSORBRIDGE_VLLM_BACKEND"] = args.backend
    _configure_fpma_alpha_environment(args.fpma_alpha, args.fpma_alpha_source)
    os.environ["TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR"] = args.fpma_prefold_selector
    os.environ["TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR_CHUNK_ROWS"] = str(
        args.fpma_selector_chunk_rows
    )
    os.environ["TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION"] = str(
        int(args.fpma_ulp_correction)
    )
    os.environ["TENSORBRIDGE_NORMAL_A8_CHUNK_ROWS"] = str(args.normal_a8_chunk_rows)
    os.environ["TENSORBRIDGE_STRICT_QWEN36_LAYOUT"] = "1"
    os.environ["TENSORBRIDGE_COMPILER"] = "nvrtc"
    os.environ["VLLM_PLUGINS"] = "tensorbridge"
    os.environ["VLLM_NVFP4_GEMM_BACKEND"] = "marlin"

    from datasets import Dataset
    from transformers import AutoTokenizer
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    from vllm.plugins.tensorbridge import register

    register()
    from vllm.model_executor.layers.quantization import get_quantization_config

    quant_config_cls = get_quantization_config("modelopt_mixed")
    if quant_config_cls.__name__ != "TensorBridgeModelOptMixedConfig":
        raise RuntimeError(f"TensorBridge vLLM plugin is not active: {quant_config_cls}")

    versions = {
        "vllm": getattr(vllm, "__version__", _version("vllm")),
        "transformers": _version("transformers"),
        "datasets": _version("datasets"),
        "torch": _version("torch"),
    }
    if not args.allow_runtime_version_mismatch and not versions["vllm"].startswith("0.20.2"):
        raise RuntimeError(f"expected vLLM 0.20.2, got {versions['vllm']}")
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count != args.tensor_parallel_size:
        raise RuntimeError(
            "visible GPU count must equal tensor_parallel_size: "
            f"visible={visible_gpu_count}, tp={args.tensor_parallel_size}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )

    dataset = Dataset.from_file(str(args.dataset_arrow))
    text = "\n\n".join(str(item) for item in dataset["text"])
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    requested_output_tokens = args.requested_output_tokens
    max_prompt_tokens = prompt_token_capacity(
        args.max_model_len, requested_output_tokens
    )
    all_blocks = build_prompt_blocks(
        token_ids,
        max_model_len=max_prompt_tokens,
        target_tokens_per_block=args.target_tokens_per_block,
    )
    if args.block_start < 0:
        raise ValueError("block_start must be non-negative")
    if args.block_start >= len(all_blocks):
        raise ValueError(
            f"block_start {args.block_start} is outside {len(all_blocks)} total blocks"
        )
    if args.max_blocks is not None and args.max_blocks <= 0:
        raise ValueError("max_blocks must be positive")
    selected_stop = len(all_blocks)
    if args.max_blocks is not None:
        selected_stop = min(selected_stop, args.block_start + args.max_blocks)
    blocks = all_blocks[args.block_start:selected_stop]
    if not blocks:
        raise ValueError("WikiText-2 tokenization produced fewer than two tokens")

    engine_args = {
        "model": str(args.model),
        "quantization": "modelopt_mixed",
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "language_model_only": True,
        "enforce_eager": args.execution_mode == "eager",
        "enable_prefix_caching": False,
        "disable_log_stats": args.execution_mode == "eager",
        "max_num_seqs": args.max_num_seqs,
    }
    compilation_config = _cudagraph_compilation_config(args)
    if compilation_config is not None:
        engine_args["compilation_config"] = compilation_config
    if args.execution_mode == "cudagraph":
        engine_args["cudagraph_metrics"] = True
    expected_scored_tokens = sum(block.scored_tokens for block in blocks)
    selected_prompt_tokens = sum(len(block.prompt_token_ids) for block in blocks)
    engine_init_started = time.perf_counter()
    llm = LLM(**engine_args)
    engine_init_seconds = time.perf_counter() - engine_init_started
    resolved_compilation = llm.llm_engine.vllm_config.compilation_config
    resolved_compilation_mode = resolved_compilation.mode.name
    resolved_cudagraph_mode = resolved_compilation.cudagraph_mode.name
    resolved_capture_sizes = list(resolved_compilation.cudagraph_capture_sizes or [])
    resolved_allreduce_rms_fusion = bool(
        resolved_compilation.pass_config.fuse_allreduce_rms
    )
    if args.execution_mode == "eager" and resolved_cudagraph_mode != "NONE":
        raise RuntimeError(f"eager mode resolved to cudagraph={resolved_cudagraph_mode}")
    if args.disable_allreduce_rms_fusion and resolved_allreduce_rms_fusion:
        raise RuntimeError("requested disabled allreduce-RMS fusion resolved enabled")
    if args.execution_mode == "cudagraph":
        if resolved_compilation_mode != args.compilation_mode:
            raise RuntimeError(
                f"requested compilation mode {args.compilation_mode} resolved to "
                f"{resolved_compilation_mode}"
            )
        if resolved_cudagraph_mode != args.cudagraph_mode:
            raise RuntimeError(
                f"requested CUDA Graph mode {args.cudagraph_mode} resolved to "
                f"{resolved_cudagraph_mode}"
            )
        if args.cudagraph_mode == "FULL_DECODE_ONLY":
            expected_capture_sizes = [len(blocks)]
            if resolved_capture_sizes != expected_capture_sizes:
                raise RuntimeError(
                    "FULL_DECODE_ONLY validation requires one exact capture size: "
                    f"resolved={resolved_capture_sizes}, expected={expected_capture_sizes}"
                )
    assigned_devices = _assigned_gpu_devices()
    engine_init_memory = _gpu_memory_snapshot(assigned_devices)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=requested_output_tokens,
        prompt_logprobs=1,
        logprobs=1,
        detokenize=False,
        flat_logprobs=False,
        ignore_eos=True,
    )
    prompts = [{"prompt_token_ids": block.prompt_token_ids} for block in blocks]
    repeat_records = []
    run_memory_snapshots = []
    reference_digest = None
    reference_trace = None
    repeat_mismatches = []
    for repeat_index in range(args.repeat_runs):
        if args.execution_mode == "cudagraph":
            _reset_cudagraph_runtime_stats(llm)
        generation_started = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        generation_seconds = time.perf_counter() - generation_started
        runtime_stats = (
            _cudagraph_runtime_stats(llm)
            if args.execution_mode == "cudagraph"
            else None
        )
        minimum_decode_graph_dispatches = requested_output_tokens - 1
        runtime_gate = (
            _cudagraph_runtime_gate(
                runtime_stats,
                cudagraph_mode=args.cudagraph_mode,
                expected_decode_dispatches=minimum_decode_graph_dispatches,
                expected_decode_batch=len(blocks),
            )
            if runtime_stats is not None
            else None
        )
        if runtime_gate is not None and not runtime_gate["passed"]:
            raise ExecutionValidationError(
                "CUDA Graph runtime dispatch did not cover every decode forward",
                {
                    "repeat_index": repeat_index,
                    "runtime_gate": runtime_gate,
                    "runtime_stats": runtime_stats,
                },
            )
        scoring_started = time.perf_counter()
        metrics = score_prompt_logprobs(
            blocks,
            outputs,
            block_index_offset=args.block_start,
            collect_nonfinite=True,
            scan_context_for_nonfinite=True,
        )
        scoring_seconds = time.perf_counter() - scoring_started
        if metrics["scored_tokens"] != expected_scored_tokens:
            raise AssertionError(
                f"scored token mismatch: {metrics['scored_tokens']} vs "
                f"{expected_scored_tokens}"
            )
        trace = _validation_trace(blocks, outputs)
        if reference_digest is None:
            reference_digest = trace["sha256"]
            reference_trace = trace
        elif trace["sha256"] != reference_digest:
            assert reference_trace is not None
            repeat_mismatches.append(
                {
                    "reference_repeat_index": 0,
                    "candidate_repeat_index": repeat_index,
                    "reference_sha256": reference_digest,
                    "candidate_sha256": trace["sha256"],
                    "difference": _trace_difference(reference_trace, trace),
                }
            )
        actual_generated_tokens = sum(
            len(completion.token_ids)
            for output in outputs
            for completion in getattr(output, "outputs", [])
        )
        generated_lengths = [
            len(completion.token_ids)
            for output in outputs
            for completion in getattr(output, "outputs", [])
        ]
        if len(generated_lengths) != len(blocks) or any(
            length != requested_output_tokens for length in generated_lengths
        ):
            raise AssertionError(
                "each request must return exactly one full-length completion: "
                f"lengths={generated_lengths}, expected={requested_output_tokens}"
            )
        expected_generated_tokens = len(blocks) * requested_output_tokens
        if actual_generated_tokens != expected_generated_tokens:
            raise AssertionError(
                f"generated token mismatch: {actual_generated_tokens} vs "
                f"{expected_generated_tokens}"
            )
        gpu_memory = _gpu_memory_snapshot(assigned_devices)
        run_memory_snapshots.append(gpu_memory)
        repeat_record = {
            "repeat_index": repeat_index,
            "generation_seconds": generation_seconds,
            "scoring_seconds": scoring_seconds,
            "metrics": metrics,
            "output_sha256": trace["sha256"],
            "generated_token_ids": trace["generated_token_ids"],
            "generated_token_logprobs": trace["generated_token_logprobs"],
            "gpu_memory": gpu_memory,
            "cudagraph_runtime": runtime_stats,
            "cudagraph_runtime_gate": runtime_gate,
        }
        if args.validation_details:
            repeat_record["target_logprobs"] = trace["target_logprobs"]
            repeat_record["targets"] = trace["targets"]
        repeat_records.append(repeat_record)

    memory_stability = None
    if args.execution_mode == "cudagraph":
        memory_stability = _memory_stability_gate(run_memory_snapshots)
        llm.llm_engine.do_log_stats()
        if not memory_stability["passed"]:
            raise ExecutionValidationError(
                "CUDA Graph driver-level GPU memory stability gate failed",
                {"memory_stability": memory_stability},
            )
    generated_id_mismatches = [
        mismatch
        for mismatch in repeat_mismatches
        if not mismatch["difference"]["generated_token_ids_exact"]
    ]
    if generated_id_mismatches:
        raise ExecutionValidationError(
            "repeated execution changed greedy generated token IDs",
            {
                "generated_token_id_mismatches": generated_id_mismatches,
                "memory_stability": memory_stability,
                "runs": repeat_records,
            },
        )

    primary_run_index = len(repeat_records) - 1
    primary_run = repeat_records[primary_run_index]
    metrics = primary_run["metrics"]
    generation_seconds = primary_run["generation_seconds"]
    scoring_seconds = primary_run["scoring_seconds"]

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment": "tensorbridge_nvfp4_wikitext2_ppl",
        "status": "passed",
        "checkpoint_mode": args.backend,
        "model_path": str(args.model),
        "runtime": {
            **versions,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_count": visible_gpu_count,
            "gpu_clock_pin_status": os.environ.get("TENSORBRIDGE_GPU_CLOCK_PIN_STATUS"),
            "vllm_module": str(Path(vllm.__file__).resolve()),
            "quant_config_class": quant_config_cls.__name__,
            "tensorbridge_compiler": os.environ["TENSORBRIDGE_COMPILER"],
            "tensorbridge_extra_nvrtc_flags": os.environ.get(
                "TENSORBRIDGE_EXTRA_NVRTC_FLAGS", ""
            ),
            "cpu_thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "production_contract": {
            "snc_enabled": args.backend == "tensorbridge",
            "scale_clamp": False,
            "strict_qwen36_layout": True,
            "checkpoint_nvfp4_layers": 193,
            "tensorbridge_nvfp4_layers": 192 if args.backend == "tensorbridge" else 0,
            "normal_a8_nvfp4_layers": 192 if args.backend == "normal_a8" else 0,
            "marlin_nvfp4_layers": 193 if args.backend == "official" else 1,
            "expected_fp8_layers": 208,
            "lm_head_weight_dtype": "nvfp4",
            "lm_head_activation_dtype": "bfloat16",
            "lm_head_backend": "marlin_w4a16",
            "transformer_nvfp4_backend": args.backend,
            "fpma_global_scale_alpha": args.fpma_alpha,
            "fpma_global_scale_alpha_source": args.fpma_alpha_source,
            "fpma_prefold_selector": args.fpma_prefold_selector,
            "fpma_prefold_selector_chunk_rows": args.fpma_selector_chunk_rows,
            "fpma_ulp_correction": args.fpma_ulp_correction,
            "fpma_ulp_encoding": (
                "ulp_scale_msb_flag_v1" if args.fpma_ulp_correction else None
            ),
            "streamk_mode": (
                "auto_router_multi_slice_nondeterministic"
                if args.backend == "tensorbridge"
                else None
            ),
            "allreduce_rms_fusion_disabled": args.disable_allreduce_rms_fusion,
            "normal_a8_chunk_rows": args.normal_a8_chunk_rows,
            "normal_a8_kernel": (
                "vllm_cutlass_fp8" if args.backend == "normal_a8" else None
            ),
        },
        "dataset": {
            "name": "wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "test",
            "cache_file": str(args.dataset_arrow),
            "rows": len(dataset),
            "joiner": "\\n\\n",
            "add_special_tokens": False,
            "total_tokens": len(token_ids),
        },
        "blocking": {
            "max_model_len": args.max_model_len,
            "max_prompt_tokens": max_prompt_tokens,
            "requested_output_tokens": requested_output_tokens,
            "target_tokens_per_block": args.target_tokens_per_block,
            "max_blocks": args.max_blocks,
            "total_num_blocks": len(all_blocks),
            "selected_block_start": args.block_start,
            "selected_block_stop": selected_stop,
            "selected_num_blocks": len(blocks),
            "num_blocks": len(blocks),
            "expected_scored_tokens": expected_scored_tokens,
        },
        "engine_args": engine_args,
        "execution_validation": {
            "execution_trace_schema_version": EXECUTION_TRACE_SCHEMA_VERSION,
            "sampling_logprob_contract": {
                "prompt_logprobs": 1,
                "generated_logprobs": 1,
                "flat_logprobs": False,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "requested_mode": args.execution_mode,
            "requested_cudagraph_mode": (
                args.cudagraph_mode if args.execution_mode == "cudagraph" else None
            ),
            "requested_compilation_mode": (
                args.compilation_mode if args.execution_mode == "cudagraph" else None
            ),
            "requested_cudagraph_capture_sizes": args.cudagraph_capture_sizes,
            "resolved_compilation_mode": resolved_compilation_mode,
            "resolved_cudagraph_mode": resolved_cudagraph_mode,
            "resolved_cudagraph_capture_sizes": resolved_capture_sizes,
            "resolved_allreduce_rms_fusion": resolved_allreduce_rms_fusion,
            "repeat_runs": args.repeat_runs,
            "primary_run_index": primary_run_index,
            "all_repeats_exact": len({run["output_sha256"] for run in repeat_records})
            == 1,
            "repeat_mismatches": repeat_mismatches,
            "numeric_warmup_runs": GRAPH_MEMORY_WARMUP_RUNS,
            "numeric_measured_runs": max(
                0, len(repeat_records) - GRAPH_MEMORY_WARMUP_RUNS
            ),
            "engine_init_gpu_memory": engine_init_memory,
            "memory_stability": memory_stability,
            "runs": repeat_records,
        },
        "benchmark_context": (
            {
                "phase": os.environ.get("TENSORBRIDGE_PERF_PHASE"),
                "index": os.environ.get("TENSORBRIDGE_PERF_INDEX"),
                "arm": os.environ.get("TENSORBRIDGE_PERF_ARM"),
                "order": os.environ.get("TENSORBRIDGE_PERF_ORDER"),
            }
            if os.environ.get("TENSORBRIDGE_PERF_ARM")
            else None
        ),
        "timing": {
            "engine_init_seconds": engine_init_seconds,
            "generation_seconds": generation_seconds,
            "scoring_seconds": scoring_seconds,
            "prompt_tokens_processed": selected_prompt_tokens,
            "generated_tokens": len(blocks) * requested_output_tokens,
            "prompt_tokens_per_second": selected_prompt_tokens / generation_seconds,
            "unique_scored_tokens_per_second": expected_scored_tokens / generation_seconds,
        },
        "metrics": metrics,
    }
    if args.eager_reference_json is not None:
        assert eager_reference is not None
        assert eager_reference_bytes is not None
        comparison = _compare_execution_results(eager_reference, result)
        comparison["reference_path"] = str(args.eager_reference_json)
        comparison["reference_sha256"] = hashlib.sha256(
            eager_reference_bytes
        ).hexdigest()
        result["execution_validation"]["eager_reference_comparison"] = comparison
        if not comparison["passed"]:
            error_type = (
                ExecutionInconclusiveError
                if comparison["decision"] == "inconclusive"
                else ExecutionValidationError
            )
            raise error_type(
                "eager and CUDA Graph execution results exceeded correctness gates",
                {
                    "eager_reference_comparison": comparison,
                    "cudagraph_execution_validation": result["execution_validation"],
                },
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--dataset-arrow", type=Path, default=Path(DEFAULT_DATASET_ARROW))
    parser.add_argument(
        "--backend",
        choices=("official", "normal_a8", "tensorbridge"),
        default="tensorbridge",
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--target-tokens-per-block", type=int, default=1024)
    parser.add_argument("--block-start", type=int, default=0)
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--tensor-parallel-size", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "cudagraph"),
        default="eager",
    )
    parser.add_argument(
        "--cudagraph-mode",
        choices=("PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"),
        default="FULL_AND_PIECEWISE",
    )
    parser.add_argument(
        "--compilation-mode",
        choices=("NONE", "VLLM_COMPILE"),
        default="VLLM_COMPILE",
    )
    parser.add_argument("--cudagraph-capture-sizes", type=_positive_int_csv)
    parser.add_argument("--disable-allreduce-rms-fusion", action="store_true")
    parser.add_argument("--requested-output-tokens", type=int, default=1)
    parser.add_argument("--repeat-runs", type=int, default=1)
    parser.add_argument("--validation-details", action="store_true")
    parser.add_argument("--eager-reference-json", type=Path)
    parser.add_argument("--normal-a8-chunk-rows", type=int, default=256)
    parser.add_argument("--fpma-alpha", type=float)
    parser.add_argument(
        "--fpma-prefold-selector",
        choices=("none", "normal_b8_sse"),
        default="none",
    )
    parser.add_argument("--fpma-selector-chunk-rows", type=int, default=256)
    parser.add_argument("--fpma-ulp-correction", action="store_true")
    parser.add_argument("--allow-runtime-version-mismatch", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    args.fpma_alpha, args.fpma_alpha_source = _resolve_fpma_alpha_input(
        args.backend,
        args.fpma_alpha,
        env_alpha=os.environ.get("TENSORBRIDGE_NVFP4_FPMA_ALPHA"),
        prefold_selector=args.fpma_prefold_selector,
        ulp_correction=args.fpma_ulp_correction,
    )

    try:
        result = _run(args)
    except Exception as error:
        failure = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "experiment": "tensorbridge_nvfp4_wikitext2_ppl",
            "status": (
                "inconclusive"
                if isinstance(error, ExecutionInconclusiveError)
                else "failed"
            ),
            "checkpoint_mode": args.backend,
            "model_path": str(args.model),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if hasattr(error, "total_nonfinite"):
            failure["total_nonfinite"] = error.total_nonfinite
        if hasattr(error, "diagnostics"):
            failure["diagnostics"] = error.diagnostics
        _write_result(failure, args.output_json)
        raise
    _write_result(result, args.output_json)


if __name__ == "__main__":
    main()
