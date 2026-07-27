#!/usr/bin/env python3
"""Fail-closed paired analysis for TensorBridge accuracy confirmation runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    REPO_ROOT / "vllm/plugins/tensorbridge_evaluation/protocols/accuracy_confirm_v1.json"
)
OUTPUT_FORMAT = "tensorbridge_nvfp4_lm_confirmation_analysis"
EXPECTED_PROTOCOL_SHA256 = "ed170a281382fd023aebb39cffe9c8030c6b72f3e8b7ffd53449d5ba014ad5da"
EXPECTED_ARMS = {
    "official": {
        "label": "official_marlin_w4a16",
        "backend": "official",
        "alpha": 1.0,
        "selector": "none",
        "ulp_correction": False,
    },
    "normal_a8": {
        "label": "normal_a8_cutlass",
        "backend": "normal_a8",
        "alpha": 1.0,
        "selector": "none",
        "ulp_correction": False,
    },
    "fpma_default": {
        "label": "tensorbridge_default_fpma_snc",
        "backend": "tensorbridge",
        "alpha": 1.0,
        "selector": "none",
        "ulp_correction": False,
    },
    "ulp_v1": {
        "label": "tensorbridge_ulp_scale_msb_flag_v1",
        "backend": "tensorbridge",
        "alpha": 1.0,
        "selector": "none",
        "ulp_correction": True,
    },
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GSM8K_SYSTEM_INSTRUCTION = (
    "Use no more than six short sentences or equations. End with a separate final "
    "line in the exact form The answer is N. Replace N with the numeric answer only, "
    "omit units, and write nothing after that line."
)
_CACHE_SEED_SHA256 = {
    "fpma_default": "f4dfa648fb1c00544671c33780496aac5c8b4372ad36784f63f36361cb02d5b2",
    "ulp_v1": "da6b312e7ee12d4921f0e1178a27816d81065393fbdeb6770376f6ba391bbbbb",
}
_RUNTIME_VERSIONS = {
    "datasets": "4.8.5",
    "lm_eval": "0.4.11",
    "torch": "2.11.0+cu128",
    "transformers": "5.9.0",
    "vllm": "0.20.2+cu128",
}


@dataclass(frozen=True)
class ConfirmationSpec:
    suite: str
    task: str
    benchmark: str
    metrics: tuple[str, ...]
    primary_metric: str
    filter_name: str
    expected_doc_ids: tuple[int, ...]
    analysis_doc_ids: tuple[int, ...]
    excluded_doc_ids: tuple[int, ...]
    dataset_contract: dict[str, Any]
    sample_manifest_path: Path | None
    sample_manifest_sha256: str | None
    sample_selection: dict[str, Any] | None
    format_regex: str | None
    format_min_valid: int | None
    noninferiority_margin: float


@dataclass
class LoadedRun:
    arm: str
    result_path: Path
    result_sha256: str
    sample_path: Path
    sample_sha256: str
    identities: dict[int, dict[str, Any]]
    correctness: dict[str, dict[int, bool]]
    raw_correctness: dict[str, dict[int, bool]]
    format_valid: dict[int, bool] | None
    checkpoint_identity: dict[str, Any]
    source_identity: dict[str, Any]
    runtime_versions: dict[str, str]
    cache_dir: Path


def _fail(message: str) -> None:
    raise ValueError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _ids_sha256(doc_ids: tuple[int, ...] | list[int]) -> str:
    return _sha256(json.dumps(list(doc_ids), separators=(",", ":")).encode("utf-8"))


def _load_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {label} {path}: {error}") from error
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object: {path}")
    return value, encoded


def _parse_id_range(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        _fail(f"{label} must use the inclusive 'start..end' form")
    match = re.fullmatch(r"(\d+)\.\.(\d+)", value)
    if match is None:
        _fail(f"{label} must use the inclusive 'start..end' form")
    start, end = (int(item) for item in match.groups())
    if start > end:
        _fail(f"{label} has a reversed range")
    return tuple(range(start, end + 1))


def _resolve_existing_path(
    value: str,
    *,
    relative_to: tuple[Path, ...],
    label: str,
) -> Path:
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [base / raw for base in relative_to]
    existing: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file() and resolved not in existing:
            existing.append(resolved)
    if not existing:
        _fail(f"{label} does not exist: {value}")
    if len(existing) != 1:
        _fail(f"{label} is ambiguous: {value} resolved to {existing}")
    return existing[0]


def _validate_analysis_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    _require(protocol.get("schema_version") == 1, "unsupported protocol schema")
    _require(
        protocol.get("format") == "tensorbridge_accuracy_confirmation_protocol",
        "unexpected protocol format",
    )
    _require(
        protocol.get("created_before_confirmation_runs") is True,
        "protocol is not marked as preregistered",
    )
    _require(
        protocol.get("arms") == list(EXPECTED_ARMS),
        "protocol arm list or order does not match the four confirmation arms",
    )
    _require(protocol.get("primary_candidate") == "fpma_default", "primary candidate changed")
    _require(protocol.get("exploratory_candidate") == "ulp_v1", "ULP candidate changed")
    _require(
        protocol.get("checkpoint")
        == {
            "repo_id": "nvidia/Qwen3.6-27B-NVFP4",
            "revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
            "content_sha256": "4ec0960247ca03fd10a9883d20de08d3795760ac1043fe7a9db6151b4074203f",
        },
        "checkpoint preregistration changed",
    )

    analysis = protocol.get("analysis")
    if not isinstance(analysis, dict):
        _fail("protocol analysis block is missing")
    _require(analysis.get("paired") is True, "confirmation analysis must be paired")
    _require(
        analysis.get("noninferiority_decision")
        == "paired_point_estimate_delta_greater_than_or_equal_to_margin",
        "unsupported noninferiority decision rule",
    )
    ci = analysis.get("paired_confidence_interval")
    if not isinstance(ci, dict):
        _fail("paired confidence interval protocol is missing")
    _require(
        ci.get("method") == "paired_nonparametric_bootstrap_percentile",
        "unsupported confidence interval method",
    )
    _require(ci.get("confidence_level") == 0.95, "confidence level must remain 0.95")
    _require(ci.get("resamples") == 10_000, "bootstrap resamples must remain 10000")
    _require(ci.get("seed") == 20_260_717, "bootstrap seed must remain 20260717")
    mcnemar = analysis.get("mcnemar")
    if not isinstance(mcnemar, dict):
        _fail("McNemar protocol is missing")
    _require(mcnemar.get("method") == "exact_binomial", "unsupported McNemar method")
    _require(mcnemar.get("alternative") == "two-sided", "McNemar must be two-sided")
    _require(
        analysis.get("invalid_generation_is_incorrect") is True,
        "invalid-generation policy changed",
    )
    _require(
        analysis.get("do_not_drop_ambiguous_or_discordant_samples") is True,
        "sample exclusion policy changed",
    )
    _require(
        analysis.get("confidence_intervals_are_reported_but_not_a_gate") is True,
        "confidence-interval decision policy changed",
    )
    _require(
        protocol.get("comparisons")
        == {
            "primary": "fpma_default - normal_a8",
            "activation_quantization_control": "normal_a8 - official",
            "exploratory_ulp_vs_default": "ulp_v1 - fpma_default",
            "exploratory_ulp_vs_normal": "ulp_v1 - normal_a8",
        },
        "preregistered comparison set changed",
    )
    return analysis


def _load_sample_manifest(
    protocol_path: Path,
    task_protocol: dict[str, Any],
    task: str,
) -> tuple[Path, dict[str, Any], str]:
    reference = task_protocol.get("sample_manifest")
    expected_sha = task_protocol.get("sample_manifest_sha256")
    if not isinstance(reference, str) or not isinstance(expected_sha, str):
        _fail("generation protocol is missing its sample manifest and SHA256")
    _require(_SHA256_RE.fullmatch(expected_sha) is not None, "invalid manifest SHA256")
    manifest_path = _resolve_existing_path(
        reference,
        relative_to=(REPO_ROOT, protocol_path.parent, Path.cwd()),
        label="sample manifest",
    )
    manifest, encoded = _load_json_object(manifest_path, "sample manifest")
    actual_sha = _sha256(encoded)
    _require(actual_sha == expected_sha, "sample manifest SHA256 differs from the protocol")
    _require(manifest.get("schema_version") == 1, "unsupported sample manifest schema")
    _require(
        manifest.get("format") == "tensorbridge_lm_eval_sample_manifest",
        "unexpected sample manifest format",
    )
    tasks = manifest.get("tasks")
    _require(isinstance(tasks, dict) and set(tasks) == {task}, "manifest task mismatch")
    ids = tasks[task]
    _require(
        isinstance(ids, list) and ids and all(type(doc_id) is int for doc_id in ids),
        "manifest doc IDs must be a non-empty integer list",
    )
    _require(ids == sorted(set(ids)), "manifest doc IDs must be unique and ascending")

    dataset = manifest.get("dataset")
    selection = manifest.get("selection")
    if not isinstance(dataset, dict) or not isinstance(selection, dict):
        _fail("manifest dataset or selection block is missing")
    _require(selection.get("count") == len(ids), "manifest selection count mismatch")
    _require(selection.get("ids_sha256") == _ids_sha256(ids), "manifest IDs SHA256 mismatch")
    _require(selection.get("algorithm") == "sha256_rank_v1", "unsupported sample selection")
    split_size = dataset.get("size")
    candidate_start = selection.get("candidate_start")
    namespace = selection.get("namespace")
    _require(type(split_size) is int and split_size > 0, "invalid manifest split size")
    _require(
        type(candidate_start) is int and 0 <= candidate_start < split_size,
        "invalid candidate start",
    )
    _require(isinstance(namespace, str) and namespace, "invalid sample namespace")
    ranked = sorted(
        range(candidate_start, split_size),
        key=lambda doc_id: (
            hashlib.sha256(f"{namespace}:{doc_id}".encode("utf-8")).digest(),
            doc_id,
        ),
    )[: len(ids)]
    _require(ids == sorted(ranked), "manifest IDs do not match sha256_rank_v1")
    return manifest_path, manifest, actual_sha


def _build_spec(
    protocol: dict[str, Any], protocol_path: Path, suite: str
) -> ConfirmationSpec:
    _validate_analysis_protocol(protocol)
    tasks = protocol.get("tasks")
    margins = protocol.get("noninferiority_margins")
    if not isinstance(tasks, dict) or not isinstance(margins, dict):
        _fail("protocol task or margin block is missing")
    if suite not in {"confirm_mc", "confirm_generation"}:
        _fail(f"only confirmation suites are supported, got {suite!r}")
    task_protocol = tasks.get(suite)
    if not isinstance(task_protocol, dict):
        _fail(f"protocol is missing suite {suite}")
    task = task_protocol.get("task")
    benchmark = task_protocol.get("benchmark", "GSM8K")
    if not isinstance(task, str) or not task:
        _fail(f"protocol task is invalid for {suite}")

    if suite == "confirm_mc":
        _require(task == "tensorbridge_arc_challenge_confirm", "ARC task changed")
        _require(task_protocol.get("bootstrap_iters") == 0, "ARC bootstrap protocol changed")
        _require(task_protocol.get("chat_template") is False, "ARC chat protocol changed")
        _require(task_protocol.get("primary_metric") == "acc_norm", "ARC primary metric changed")
        _require(task_protocol.get("secondary_metric") == "acc", "ARC secondary metric changed")
        size = task_protocol.get("evaluation_docs")
        _require(type(size) is int and size > 0, "invalid ARC evaluation size")
        expected_ids = tuple(range(size))
        excluded = _parse_id_range(
            task_protocol.get("excluded_development_doc_ids"),
            "ARC excluded development IDs",
        )
        analysis_ids = tuple(doc_id for doc_id in expected_ids if doc_id not in set(excluded))
        _require(
            len(analysis_ids) == task_protocol.get("primary_analysis_docs"),
            "ARC primary analysis count mismatch",
        )
        _require(
            analysis_ids
            == _parse_id_range(
                task_protocol.get("primary_analysis_doc_ids"), "ARC primary analysis IDs"
            ),
            "ARC primary analysis range mismatch",
        )
        dataset_contract = {
            "path": "allenai/ai2_arc",
            "name": "ARC-Challenge",
            "split": "test",
            "size": size,
            "revision": task_protocol.get("dataset_revision"),
            "datasets_fingerprint": task_protocol.get("datasets_fingerprint"),
            "canonicalization": "sorted_minified_utf8_jsonl_with_doc_id_v1",
            "canonical_jsonl_sha256": task_protocol.get("canonical_jsonl_sha256"),
        }
        _require(
            all(isinstance(dataset_contract[key], str) and dataset_contract[key] for key in (
                "revision",
                "datasets_fingerprint",
                "canonical_jsonl_sha256",
            )),
            "ARC dataset hashes are missing",
        )
        margin = margins.get("arc_challenge_acc_norm_vs_normal_a8")
        _require(margin == -0.02, "ARC noninferiority margin changed")
        return ConfirmationSpec(
            suite=suite,
            task=task,
            benchmark=str(benchmark),
            metrics=("acc_norm", "acc"),
            primary_metric="acc_norm",
            filter_name="none",
            expected_doc_ids=expected_ids,
            analysis_doc_ids=analysis_ids,
            excluded_doc_ids=excluded,
            dataset_contract=dataset_contract,
            sample_manifest_path=None,
            sample_manifest_sha256=None,
            sample_selection=None,
            format_regex=None,
            format_min_valid=None,
            noninferiority_margin=float(margin),
        )

    _require(task == "tensorbridge_gsm8k_relative_smoke", "GSM8K task changed")
    _require(task_protocol.get("bootstrap_iters") == 0, "GSM8K bootstrap protocol changed")
    _require(task_protocol.get("chat_template") is True, "GSM8K chat protocol changed")
    _require(task_protocol.get("thinking") is False, "GSM8K thinking protocol changed")
    _require(task_protocol.get("temperature") == 0.0, "GSM8K temperature changed")
    _require(task_protocol.get("max_gen_toks") == 1024, "GSM8K generation cap changed")
    _require(
        task_protocol.get("invalid_or_truncated_generation_is_incorrect") is True,
        "GSM8K invalid-generation policy changed",
    )
    _require(
        task_protocol.get("primary_metric") == "exact_match,final-answer",
        "GSM8K metric or filter changed",
    )
    manifest_path, manifest, manifest_sha = _load_sample_manifest(
        protocol_path, task_protocol, task
    )
    expected_ids = tuple(manifest["tasks"][task])
    _require(task_protocol.get("samples") == len(expected_ids), "GSM8K sample count mismatch")
    regex = task_protocol.get("valid_format_regex")
    if not isinstance(regex, str):
        _fail("GSM8K valid-format regex is missing")
    try:
        re.compile(regex)
    except re.error as error:
        raise ValueError(f"invalid GSM8K format regex: {error}") from error
    format_gate = protocol.get("format_gate")
    if not isinstance(format_gate, dict):
        _fail("GSM8K format gate is missing")
    _require(
        format_gate.get("gsm8k_total") == len(expected_ids),
        "GSM8K format-gate denominator mismatch",
    )
    minimum = format_gate.get("gsm8k_min_valid")
    _require(minimum == 127 and len(expected_ids) == 128, "GSM8K format gate changed")
    margin = margins.get("gsm8k_exact_match_vs_normal_a8")
    _require(margin == -0.05, "GSM8K noninferiority margin changed")
    return ConfirmationSpec(
        suite=suite,
        task=task,
        benchmark=str(benchmark),
        metrics=("exact_match",),
        primary_metric="exact_match",
        filter_name="final-answer",
        expected_doc_ids=expected_ids,
        analysis_doc_ids=expected_ids,
        excluded_doc_ids=(),
        dataset_contract=manifest["dataset"],
        sample_manifest_path=manifest_path,
        sample_manifest_sha256=manifest_sha,
        sample_selection=manifest["selection"],
        format_regex=regex,
        format_min_valid=minimum,
        noninferiority_margin=float(margin),
    )


def _binary_metric(value: Any, label: str) -> bool:
    if type(value) not in {int, float, bool}:
        _fail(f"{label} is not binary numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
        _fail(f"{label} is not 0 or 1")
    return bool(numeric)


def _raw_generation(row: dict[str, Any], doc_id: int) -> str:
    responses = row.get("resps")
    if (
        not isinstance(responses, list)
        or len(responses) != 1
        or not isinstance(responses[0], list)
        or len(responses[0]) != 1
        or not isinstance(responses[0][0], str)
    ):
        _fail(f"GSM8K doc {doc_id} does not contain exactly one raw generation")
    return responses[0][0]


def _read_sample_rows(
    sample_path: Path,
    artifact: dict[str, Any],
    spec: ConfirmationSpec,
) -> tuple[
    dict[int, dict[str, Any]],
    dict[str, dict[int, bool]],
    dict[str, dict[int, bool]],
    dict[int, bool] | None,
]:
    encoded = sample_path.read_bytes()
    _require(encoded.endswith(b"\n"), f"sample artifact must end in LF: {sample_path}")
    actual_sha = _sha256(encoded)
    _require(
        artifact.get("sha256") == actual_sha,
        f"sample artifact SHA256 mismatch: {sample_path}",
    )
    try:
        lines = encoded.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"sample artifact is not UTF-8: {sample_path}") from error
    _require(all(lines), f"sample artifact has a blank JSONL row: {sample_path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid sample JSON at {sample_path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            _fail(f"sample row must be an object at {sample_path}:{line_number}")
        rows.append(row)

    expected_count = len(spec.expected_doc_ids)
    _require(artifact.get("rows") == len(rows) == expected_count, "sample row count mismatch")
    _require(artifact.get("unique_docs") == expected_count, "sample unique-doc count mismatch")
    _require(artifact.get("filters") == [spec.filter_name], "sample filter metadata mismatch")

    identities: dict[int, dict[str, Any]] = {}
    raw_correctness = {metric: {} for metric in spec.metrics}
    correctness = {metric: {} for metric in spec.metrics}
    format_valid: dict[int, bool] | None = {} if spec.format_regex is not None else None
    observed_ids: list[int] = []
    format_pattern = re.compile(spec.format_regex) if spec.format_regex is not None else None
    for row_number, row in enumerate(rows, start=1):
        doc_id = row.get("doc_id")
        _require(type(doc_id) is int, f"sample row {row_number} has a non-integer doc_id")
        _require(doc_id not in identities, f"duplicate sample doc_id {doc_id}")
        observed_ids.append(doc_id)
        _require(row.get("filter") == spec.filter_name, f"unexpected filter for doc {doc_id}")
        _require(isinstance(row.get("doc"), dict), f"missing document for doc {doc_id}")
        row_metrics = row.get("metrics")
        _require(
            isinstance(row_metrics, list)
            and len(row_metrics) == len(spec.metrics)
            and set(row_metrics) == set(spec.metrics),
            f"sample metric list mismatch for doc {doc_id}",
        )
        identity = {"doc": row["doc"]}
        for key in ("doc_hash", "prompt_hash", "target_hash"):
            value = row.get(key)
            _require(
                isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
                f"missing or invalid {key} for doc {doc_id}",
            )
            identity[key] = value
        identities[doc_id] = identity

        for metric in spec.metrics:
            raw_value = _binary_metric(row.get(metric), f"{metric} for doc {doc_id}")
            raw_correctness[metric][doc_id] = raw_value
            correctness[metric][doc_id] = raw_value
        if format_pattern is not None:
            assert format_valid is not None
            valid = format_pattern.search(_raw_generation(row, doc_id)) is not None
            format_valid[doc_id] = valid
            if not valid:
                correctness[spec.primary_metric][doc_id] = False

    _require(tuple(observed_ids) == spec.expected_doc_ids, "sample doc IDs or order mismatch")
    return identities, correctness, raw_correctness, format_valid


def _full_dataset_sha256(identities: dict[int, dict[str, Any]], doc_ids: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for doc_id in doc_ids:
        digest.update(_canonical_json({"doc_id": doc_id, "doc": identities[doc_id]["doc"]}))
        digest.update(b"\n")
    return digest.hexdigest()


def _selected_docs_sha256(
    identities: dict[int, dict[str, Any]], doc_ids: tuple[int, ...]
) -> str:
    records = [
        {
            "doc_id": doc_id,
            "doc_sha256": _sha256(_canonical_json(identities[doc_id]["doc"])),
        }
        for doc_id in doc_ids
    ]
    return _sha256(_canonical_json(records))


def _validate_dataset_verification(result: dict[str, Any], spec: ConfirmationSpec) -> None:
    runtime = result.get("runtime")
    verification = runtime.get("dataset_verification") if isinstance(runtime, dict) else None
    if not isinstance(verification, dict):
        _fail("result is missing fail-closed dataset verification")
    _require(verification.get("contract") == spec.dataset_contract, "dataset contract mismatch")
    pre_run = verification.get("pre_run")
    logged = verification.get("logged_samples")
    _require(
        isinstance(pre_run, dict) and pre_run.get("verified") is True,
        "dataset precheck failed",
    )
    _require(
        pre_run.get("size") == spec.dataset_contract["size"]
        and pre_run.get("datasets_fingerprint") == spec.dataset_contract["datasets_fingerprint"]
        and pre_run.get("canonical_jsonl_sha256")
        == spec.dataset_contract["canonical_jsonl_sha256"],
        "dataset precheck hashes mismatch",
    )
    if spec.sample_selection is not None:
        _require(
            pre_run.get("selected_docs_sha256")
            == spec.sample_selection["selected_docs_sha256"],
            "selected-doc precheck hash mismatch",
        )
    _require(
        isinstance(logged, dict) and logged.get("verified") is True,
        "logged samples unverified",
    )
    logged_tasks = logged.get("tasks")
    _require(
        isinstance(logged_tasks, dict) and set(logged_tasks) == {spec.task},
        "logged task mismatch",
    )
    task_check = logged_tasks[spec.task]
    if not isinstance(task_check, dict) or task_check.get("verified") is not True:
        _fail("logged task verification failed")
    if spec.sample_selection is None:
        _require(
            task_check.get("kind") == "full_split"
            and task_check.get("size") == len(spec.expected_doc_ids)
            and task_check.get("canonical_jsonl_sha256")
            == spec.dataset_contract["canonical_jsonl_sha256"],
            "logged full-split verification mismatch",
        )
    else:
        _require(
            task_check.get("kind") == "selected_docs"
            and task_check.get("size") == len(spec.expected_doc_ids)
            and task_check.get("ids_sha256") == spec.sample_selection["ids_sha256"]
            and task_check.get("selected_docs_sha256")
            == spec.sample_selection["selected_docs_sha256"],
            "logged selected-doc verification mismatch",
        )


def _validate_result_protocol(result_protocol: dict[str, Any], spec: ConfirmationSpec) -> None:
    _require(result_protocol.get("suite") == spec.suite, "result suite mismatch")
    _require(result_protocol.get("tasks") == [spec.task], "result task mismatch")
    _require(result_protocol.get("num_fewshot") == 0, "confirmation must be zero-shot")
    _require(result_protocol.get("bootstrap_iters") == 0, "lm-eval bootstrap must be disabled")
    _require(result_protocol.get("limit") == {"kind": "none", "value": None}, "result used a limit")
    _require(
        result_protocol.get("analysis_exclude_doc_ids") == list(spec.excluded_doc_ids),
        "result analysis exclusions differ from the preregistration",
    )
    _require(result_protocol.get("batch_size") == "auto", "lm-eval batch mode changed")
    _require(result_protocol.get("response_cache") is None, "response cache was enabled")
    _require(
        result_protocol.get("seeds")
        == {"python": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},
        "evaluation seeds changed",
    )
    expected_engine_args = {
        "pretrained": "/data/user/jzou521/models/nvidia/Qwen3.6-27B-NVFP4",
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "quantization": "modelopt_mixed",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.5,
        "max_num_seqs": 8,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "disable_log_stats": True,
        "seed": 1234,
        "language_model_only": True,
        "enable_thinking": False,
        "max_gen_toks": 1024 if spec.suite == "confirm_generation" else 256,
    }
    _require(
        result_protocol.get("engine_args") == expected_engine_args,
        "vLLM engine arguments changed",
    )
    if spec.suite == "confirm_mc":
        _require(result_protocol.get("generation") is False, "ARC result is marked as generation")
        _require(result_protocol.get("apply_chat_template") is False, "ARC used a chat template")
        _require(result_protocol.get("prompt_format") == "completion", "ARC prompt changed")
        _require(result_protocol.get("system_instruction") is None, "ARC used an instruction")
        _require(result_protocol.get("fewshot_as_multiturn") is False, "ARC fewshot mode changed")
        _require(
            result_protocol.get("dataset_contract") == spec.dataset_contract,
            "ARC contract mismatch",
        )
        _require(result_protocol.get("sample_selection") is None, "ARC unexpectedly used samples")
        return

    _require(result_protocol.get("generation") is True, "GSM8K result is not generation")
    _require(result_protocol.get("apply_chat_template") is True, "GSM8K omitted chat templating")
    _require(result_protocol.get("enable_thinking") is False, "GSM8K enabled thinking")
    _require(result_protocol.get("think_end_token") is None, "GSM8K think token changed")
    _require(result_protocol.get("prompt_format") == "chat_nonthinking", "GSM8K prompt changed")
    _require(result_protocol.get("fewshot_as_multiturn") is False, "GSM8K fewshot mode changed")
    _require(
        result_protocol.get("system_instruction") == _GSM8K_SYSTEM_INSTRUCTION,
        "GSM8K system instruction changed",
    )
    _require(result_protocol.get("max_gen_toks") == 1024, "GSM8K generation cap changed")
    _require(
        result_protocol.get("generation_kwargs") == {"temperature": 0.0, "max_gen_toks": 1024},
        "GSM8K generation settings changed",
    )
    _require(
        result_protocol.get("dataset_contract") is None,
        "GSM8K has an unexpected full contract",
    )
    selection = result_protocol.get("sample_selection")
    if not isinstance(selection, dict):
        _fail("GSM8K result is missing sample selection metadata")
    _require(
        selection.get("manifest_sha256") == spec.sample_manifest_sha256,
        "result manifest hash mismatch",
    )
    _require(selection.get("dataset") == spec.dataset_contract, "result manifest dataset mismatch")
    _require(
        selection.get("selection") == spec.sample_selection,
        "result sample selection mismatch",
    )
    _require(
        selection.get("tasks") == {spec.task: list(spec.expected_doc_ids)},
        "result sample IDs mismatch",
    )


def _expected_task_config(spec: ConfirmationSpec) -> dict[str, Any]:
    common = {
        "task": spec.task,
        "dataset_path": spec.dataset_contract["path"],
        "dataset_name": spec.dataset_contract["name"],
        "dataset_kwargs": {"revision": spec.dataset_contract["revision"]},
        "description": "",
        "test_split": spec.dataset_contract["split"],
        "num_fewshot": 0,
        "repeats": 1,
        "target_delimiter": " ",
        "fewshot_delimiter": "\n\n",
        "unsafe_code": False,
    }
    if spec.suite == "confirm_mc":
        return common | {
            "output_type": "multiple_choice",
            "training_split": "train",
            "validation_split": "validation",
            "doc_to_text": "Question: {{question}}\nAnswer:",
            "doc_to_target": "{{choices.label.index(answerKey)}}",
            "doc_to_choice": "{{choices.text}}",
            "should_decontaminate": True,
            "doc_to_decontamination_query": "Question: {{question}}\nAnswer:",
            "fewshot_config": {
                "sampler": "default",
                "samples": None,
                "doc_to_text": "Question: {{question}}\nAnswer:",
                "doc_to_target": "{{choices.label.index(answerKey)}}",
                "doc_to_choice": "{{choices.text}}",
                "target_delimiter": " ",
                "fewshot_delimiter": "\n\n",
                "gen_prefix": None,
                "split": None,
                "fewshot_indices": None,
                "process_docs": None,
            },
            "metric_list": [
                {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
                {
                    "metric": "acc_norm",
                    "aggregation": "mean",
                    "higher_is_better": True,
                },
            ],
            "tag": ["tensorbridge_accuracy_confirm"],
            "metadata": {
                "version": 1.0,
                "benchmark_scope": "tensorbridge_accuracy_confirmation",
            },
        }
    assert spec.format_regex is not None
    return common | {
        "output_type": "generate_until",
        "training_split": "train",
        "doc_to_text": "Q: {{question}}\nA:",
        "doc_to_target": "{{answer}}",
        "should_decontaminate": False,
        "fewshot_config": {
            "sampler": "default",
            "samples": None,
            "doc_to_text": "Q: {{question}}\nA:",
            "doc_to_target": "{{answer}}",
            "doc_to_choice": None,
            "target_delimiter": " ",
            "fewshot_delimiter": "\n\n",
            "gen_prefix": None,
            "split": None,
            "fewshot_indices": None,
            "process_docs": None,
        },
        "metric_list": [
            {
                "metric": "exact_match",
                "aggregation": "mean",
                "higher_is_better": True,
                "ignore_case": True,
                "ignore_punctuation": False,
                "regexes_to_ignore": [",", "\\$", "(?s).*#### ", "\\.$"],
            }
        ],
        "generation_kwargs": {
            "until": ["Q:", "</s>", "<|im_end|>"],
            "do_sample": False,
            "temperature": 0.0,
            "max_gen_toks": 1024,
        },
        "filter_list": [
            {
                "name": "final-answer",
                "filter": [
                    {"function": "regex", "regex_pattern": spec.format_regex},
                    {"function": "take_first"},
                ],
            }
        ],
        "metadata": {
            "version": 1.0,
            "benchmark_scope": "tensorbridge_relative_accuracy",
        },
    }


def _expected_runtime_environment(arm: str) -> dict[str, str]:
    return {
        "TENSORBRIDGE_VLLM_BACKEND": EXPECTED_ARMS[arm]["backend"],
        "TENSORBRIDGE_NVFP4_FPMA_ALPHA": "1.0",
        "TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR": "none",
        "TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION": "1" if arm == "ulp_v1" else "0",
        "TENSORBRIDGE_NVFP4_ALLOW_SCALE_CLAMP": "0",
        "TENSORBRIDGE_STRICT_QWEN36_LAYOUT": "1",
        "TENSORBRIDGE_COMPILER": "nvrtc",
        "TENSORBRIDGE_EXTRA_NVRTC_FLAGS": "",
        "TENSORBRIDGE_NORMAL_A8_CHUNK_ROWS": "256",
        "VLLM_PLUGINS": "tensorbridge",
        "VLLM_NVFP4_GEMM_BACKEND": "marlin",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TENSORBRIDGE_DISABLE_PARALLEL_BUILD": "1",
    }


def _validate_lm_eval(
    result: dict[str, Any],
    spec: ConfirmationSpec,
    raw_correctness: dict[str, dict[int, bool]],
) -> None:
    lm_eval = result.get("lm_eval")
    if not isinstance(lm_eval, dict):
        _fail("result is missing lm_eval output")
    for key in ("results", "configs", "n-samples"):
        value = lm_eval.get(key)
        _require(
            isinstance(value, dict) and set(value) == {spec.task},
            f"lm_eval {key} task mismatch",
        )
    config = lm_eval["configs"][spec.task]
    if not isinstance(config, dict):
        _fail("lm_eval task config is invalid")
    expected_config = _expected_task_config(spec)
    _require(
        set(config) == set(expected_config),
        "lm_eval task config keys changed: "
        f"actual={sorted(config)}, expected={sorted(expected_config)}",
    )
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
    _require(not mismatches, f"lm_eval scoring config mismatch: {mismatches}")
    if spec.suite == "confirm_generation":
        filters = config["filter_list"]
        filter_config = filters[0]
        _require(
            isinstance(filter_config, dict)
            and filter_config.get("name") == spec.filter_name
            and isinstance(filter_config.get("filter"), list)
            and len(filter_config["filter"]) == 2
            and filter_config["filter"][0].get("regex_pattern") == spec.format_regex,
            "GSM8K filter chain differs from the protocol",
        )

    counts = lm_eval["n-samples"][spec.task]
    _require(isinstance(counts, dict), "lm_eval sample counts are invalid")
    _require(
        counts.get("original") == spec.dataset_contract["size"],
        "original sample count mismatch",
    )
    expected_effective = len(spec.expected_doc_ids)
    allowed_effective = {expected_effective}
    if spec.sample_selection is not None:
        allowed_effective.add(spec.dataset_contract["size"])
    _require(counts.get("effective") in allowed_effective, "effective sample count mismatch")

    aggregates = lm_eval["results"][spec.task]
    if not isinstance(aggregates, dict):
        _fail("lm_eval aggregates are invalid")
    for metric in spec.metrics:
        key = f"{metric},{spec.filter_name}"
        value = aggregates.get(key)
        _require(type(value) in {int, float} and math.isfinite(value), f"missing aggregate {key}")
        row_mean = sum(raw_correctness[metric].values()) / len(spec.expected_doc_ids)
        _require(
            math.isclose(float(value), row_mean, rel_tol=0.0, abs_tol=1e-12),
            f"aggregate {key} disagrees with sample rows",
        )


def _load_run(result_path: Path, spec: ConfirmationSpec, protocol: dict[str, Any]) -> LoadedRun:
    result, encoded = _load_json_object(result_path, "result")
    _require(result.get("schema_version") == 1, f"unsupported result schema: {result_path}")
    _require(result.get("experiment") == "tensorbridge_nvfp4_lm_harness", "unexpected experiment")
    _require(result.get("status") == "passed", f"result did not pass: {result_path}")
    arm_config = result.get("arm")
    if not isinstance(arm_config, dict):
        _fail(f"result arm config is invalid: {result_path}")
    arm = arm_config.get("key")
    _require(arm in EXPECTED_ARMS, f"unexpected result arm: {arm!r}")
    expected_arm = {"key": arm} | EXPECTED_ARMS[arm]
    _require(arm_config == expected_arm, f"precision config mismatch for arm {arm}")

    checkpoint = result.get("checkpoint")
    start = checkpoint.get("start") if isinstance(checkpoint, dict) else None
    end = checkpoint.get("end") if isinstance(checkpoint, dict) else None
    protocol_checkpoint = protocol.get("checkpoint")
    _require(
        isinstance(start, dict)
        and isinstance(protocol_checkpoint, dict)
        and start.get("checkpoint_content_sha256") == protocol_checkpoint.get("content_sha256"),
        f"checkpoint content mismatch for arm {arm}",
    )
    _require(
        checkpoint.get("unchanged") is True and start == end,
        f"checkpoint changed during the {arm} run",
    )
    production = result.get("production_contract")
    if not isinstance(production, dict):
        _fail(f"production contract is missing for arm {arm}")
    _require(
        production.get("checkpoint_nvfp4_layers") == 193
        and production.get("expected_fp8_layers") == 208
        and production.get("snc_enabled") == (arm in {"fpma_default", "ulp_v1"})
        and production.get("scale_clamp") is False
        and production.get("strict_qwen36_layout") is True
        and production.get("lm_head_backend") == "marlin_w4a16"
        and production.get("fpma_ulp_encoding")
        == ("ulp_scale_msb_flag_v1" if arm == "ulp_v1" else None),
        f"production precision contract mismatch for arm {arm}",
    )

    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        _fail(f"runtime metadata is missing for arm {arm}")
    _require(
        runtime.get("gpu") == "NVIDIA H100 80GB HBM3"
        and runtime.get("capability") == [9, 0],
        f"unexpected GPU for arm {arm}",
    )
    _require(
        runtime.get("cpu_thread_limits")
        == {
            "MKL_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
        },
        f"CPU thread limits changed for arm {arm}",
    )
    _require(
        runtime.get("configured_environment") == _expected_runtime_environment(arm),
        f"configured precision environment mismatch for arm {arm}",
    )
    _require(
        runtime.get("quant_config_class") == "TensorBridgeModelOptMixedConfig",
        f"vLLM quantization plugin mismatch for arm {arm}",
    )
    expected_nvrtc_flags = "-DTENSORBRIDGE_NVFP4_SWZ64_B_DUAL_MMA_PREINT_LOAD=1"
    if arm == "ulp_v1":
        expected_nvrtc_flags += (
            " -DTENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION=1"
            " -DTENSORBRIDGE_NVFP4_FPMA_ULP_SCALE_MSB_FLAG_V1=1"
        )
    _require(
        runtime.get("tensorbridge_nvrtc_flags") == expected_nvrtc_flags,
        f"effective NVRTC flags mismatch for arm {arm}",
    )
    cache_dir_value = runtime.get("tensorbridge_cache_dir")
    _require(
        isinstance(cache_dir_value, str)
        and cache_dir_value
        and Path(cache_dir_value).is_absolute(),
        f"TensorBridge cache path is invalid for arm {arm}",
    )
    cache_dir = Path(cache_dir_value).resolve()
    source = runtime.get("source")
    _require(
        isinstance(source, dict)
        and set(source) == {"start", "end", "unchanged"}
        and source.get("unchanged") is True
        and source.get("start") == source.get("end"),
        f"source changed during the {arm} run",
    )
    source_start = source["start"]
    _require(
        isinstance(source_start, dict)
        and set(source_start) == {"tensorbridge_git", "tensorbridge_tree", "vllm_git"},
        f"source identity is incomplete for arm {arm}",
    )
    tree = source_start["tensorbridge_tree"]
    _require(
        isinstance(tree, dict)
        and type(tree.get("files")) is int
        and tree["files"] > 0
        and _SHA256_RE.fullmatch(str(tree.get("sha256"))) is not None,
        f"TensorBridge tree identity is invalid for arm {arm}",
    )
    for component in ("tensorbridge_git", "vllm_git"):
        git = source_start[component]
        _require(
            isinstance(git, dict)
            and git.get("available") is True
            and isinstance(git.get("head"), str)
            and _SHA256_RE.fullmatch(str(git.get("status_sha256"))) is not None
            and _SHA256_RE.fullmatch(str(git.get("tracked_diff_sha256"))) is not None,
            f"{component} identity is invalid for arm {arm}",
        )
    versions = runtime.get("versions")
    _require(
        versions == _RUNTIME_VERSIONS,
        f"runtime versions changed for arm {arm}",
    )
    cache_seed = runtime.get("tensorbridge_cache_seed")
    expected_seed = _CACHE_SEED_SHA256.get(arm)
    if expected_seed is None:
        _require(cache_seed is None, f"unexpected cache seed for arm {arm}")
    else:
        _require(
            isinstance(cache_seed, dict)
            and cache_seed.get("verified") is True
            and cache_seed.get("sha256") == expected_seed
            and isinstance(cache_seed.get("source"), str),
            f"cache seed provenance mismatch for arm {arm}",
        )

    result_protocol = result.get("protocol")
    if not isinstance(result_protocol, dict):
        _fail(f"result protocol is missing for arm {arm}")
    _validate_result_protocol(result_protocol, spec)
    _validate_dataset_verification(result, spec)

    artifacts = result.get("sample_artifacts")
    _require(isinstance(artifacts, dict) and set(artifacts) == {spec.task}, "sample task mismatch")
    artifact = artifacts[spec.task]
    if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
        _fail(f"sample artifact metadata is invalid for arm {arm}")
    sample_path = _resolve_existing_path(
        artifact["path"],
        relative_to=(Path.cwd(), REPO_ROOT, result_path.parent),
        label=f"sample artifact for {arm}",
    )
    identities, correctness, raw_correctness, format_valid = _read_sample_rows(
        sample_path, artifact, spec
    )
    _validate_lm_eval(result, spec, raw_correctness)

    if spec.sample_selection is None:
        observed_digest = _full_dataset_sha256(identities, spec.expected_doc_ids)
        expected_digest = spec.dataset_contract["canonical_jsonl_sha256"]
        _require(
            observed_digest == expected_digest,
            f"full ARC document hash mismatch for arm {arm}",
        )
    else:
        observed_digest = _selected_docs_sha256(identities, spec.expected_doc_ids)
        expected_digest = spec.sample_selection["selected_docs_sha256"]
        _require(
            observed_digest == expected_digest,
            f"selected GSM8K document hash mismatch for arm {arm}",
        )

    return LoadedRun(
        arm=arm,
        result_path=result_path,
        result_sha256=_sha256(encoded),
        sample_path=sample_path,
        sample_sha256=artifact["sha256"],
        identities=identities,
        correctness=correctness,
        raw_correctness=raw_correctness,
        format_valid=format_valid,
        checkpoint_identity=start,
        source_identity=source_start,
        runtime_versions=versions,
        cache_dir=cache_dir,
    )


def _accuracy(values: dict[int, bool], doc_ids: tuple[int, ...]) -> dict[str, Any]:
    correct = sum(values[doc_id] for doc_id in doc_ids)
    return {"correct": correct, "total": len(doc_ids), "accuracy": correct / len(doc_ids)}


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _paired_bootstrap_ci(
    deltas: list[int], *, confidence_level: float, resamples: int, seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)
    sample_count = len(deltas)
    estimates: list[float] = []
    for _ in range(resamples):
        total = 0
        for _ in range(sample_count):
            total += deltas[rng.randrange(sample_count)]
        estimates.append(total / sample_count)
    estimates.sort()
    tail = (1.0 - confidence_level) / 2.0
    return {
        "method": "paired_nonparametric_bootstrap_percentile",
        "quantile_interpolation": "linear_type7",
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
        "lower": _percentile(estimates, tail),
        "upper": _percentile(estimates, 1.0 - tail),
    }


def _exact_mcnemar(loss_flips: int, gain_flips: int) -> dict[str, Any]:
    discordant = loss_flips + gain_flips
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(loss_flips, gain_flips) + 1))
        p_value = min(1.0, (2 * tail) / (2**discordant))
    return {
        "method": "exact_binomial",
        "alternative": "two-sided",
        "discordant": discordant,
        "p_value": p_value,
    }


def _paired_stats(
    baseline: dict[int, bool],
    candidate: dict[int, bool],
    doc_ids: tuple[int, ...],
    analysis_protocol: dict[str, Any],
) -> dict[str, Any]:
    loss_flip_ids = [
        doc_id for doc_id in doc_ids if baseline[doc_id] and not candidate[doc_id]
    ]
    gain_flip_ids = [
        doc_id for doc_id in doc_ids if candidate[doc_id] and not baseline[doc_id]
    ]
    loss_flips = len(loss_flip_ids)
    gain_flips = len(gain_flip_ids)
    both_correct = sum(candidate[doc_id] and baseline[doc_id] for doc_id in doc_ids)
    both_incorrect = len(doc_ids) - loss_flips - gain_flips - both_correct
    deltas = [int(candidate[doc_id]) - int(baseline[doc_id]) for doc_id in doc_ids]
    ci_protocol = analysis_protocol["paired_confidence_interval"]
    return {
        "samples": len(doc_ids),
        "baseline_accuracy": sum(baseline[doc_id] for doc_id in doc_ids) / len(doc_ids),
        "candidate_accuracy": sum(candidate[doc_id] for doc_id in doc_ids) / len(doc_ids),
        "candidate_loss_flips": loss_flips,
        "candidate_gain_flips": gain_flips,
        "candidate_loss_flip_doc_ids": loss_flip_ids,
        "candidate_gain_flip_doc_ids": gain_flip_ids,
        "concordant_both_correct": both_correct,
        "concordant_both_incorrect": both_incorrect,
        "paired_accuracy_delta": sum(deltas) / len(doc_ids),
        "exact_mcnemar": _exact_mcnemar(loss_flips, gain_flips),
        "paired_confidence_interval": _paired_bootstrap_ci(
            deltas,
            confidence_level=ci_protocol["confidence_level"],
            resamples=ci_protocol["resamples"],
            seed=ci_protocol["seed"],
        ),
    }


def _comparison_summary(
    *,
    runs: dict[str, LoadedRun],
    arm_summaries: dict[str, Any],
    spec: ConfirmationSpec,
    analysis_protocol: dict[str, Any],
    candidate: str,
    baseline: str,
    role: str,
    noninferiority_check: bool,
    required_gate: bool,
) -> dict[str, Any]:
    metric_results: dict[str, Any] = {}
    for metric in spec.metrics:
        stats = _paired_stats(
            runs[baseline].correctness[metric],
            runs[candidate].correctness[metric],
            spec.analysis_doc_ids,
            analysis_protocol,
        )
        if metric == spec.primary_metric and noninferiority_check:
            passed = stats["paired_accuracy_delta"] >= spec.noninferiority_margin
            stats["noninferiority"] = {
                "decision_rule": analysis_protocol["noninferiority_decision"],
                "margin": spec.noninferiority_margin,
                "confidence_interval_is_not_a_gate": True,
                "point_estimate_gate_passed": passed,
            }
        metric_results[metric] = stats

    format_gate = None
    if runs[candidate].format_valid is not None:
        candidate_format = arm_summaries[candidate]["format"]
        baseline_format = arm_summaries[baseline]["format"]
        format_gate = {
            "minimum_valid": candidate_format["minimum_valid"],
            "candidate": {
                "arm": candidate,
                "valid": candidate_format["valid"],
                "total": candidate_format["total"],
                "gate_passed": candidate_format["gate_passed"],
            },
            "baseline": {
                "arm": baseline,
                "valid": baseline_format["valid"],
                "total": baseline_format["total"],
                "gate_passed": baseline_format["gate_passed"],
            },
            "both_passed": candidate_format["gate_passed"]
            and baseline_format["gate_passed"],
        }
    summary = {
        "role": role,
        "candidate": candidate,
        "baseline": baseline,
        "metrics": metric_results,
        "format_gate": format_gate,
        "noninferiority_check_applies": noninferiority_check,
        "required_for_primary_confirmation": required_gate,
    }
    if noninferiority_check:
        primary_gate = metric_results[spec.primary_metric]["noninferiority"][
            "point_estimate_gate_passed"
        ]
        threshold_passed = primary_gate and (
            format_gate is None or format_gate["both_passed"]
        )
        summary["threshold_and_format_checks_passed"] = threshold_passed
        summary["all_required_gates_passed"] = threshold_passed if required_gate else None
    else:
        summary["threshold_and_format_checks_passed"] = None
        summary["all_required_gates_passed"] = None
    return summary


def analyze_confirmation(
    result_paths: list[Path], protocol_path: Path = DEFAULT_PROTOCOL
) -> dict[str, Any]:
    """Validate four confirmation artifacts and return paired JSON-ready statistics."""
    _require(len(result_paths) == len(EXPECTED_ARMS), "exactly four result JSON files are required")
    resolved_results = [path.expanduser().resolve() for path in result_paths]
    _require(len(set(resolved_results)) == len(resolved_results), "duplicate result path")
    protocol_path = protocol_path.expanduser().resolve()
    protocol, protocol_encoded = _load_json_object(protocol_path, "protocol")
    _require(
        _sha256(protocol_encoded) == EXPECTED_PROTOCOL_SHA256,
        "accuracy confirmation protocol SHA256 changed",
    )

    peeked_suites: list[str] = []
    for result_path in resolved_results:
        result, _ = _load_json_object(result_path, "result")
        result_protocol = result.get("protocol")
        if not isinstance(result_protocol, dict) or not isinstance(
            result_protocol.get("suite"), str
        ):
            _fail(f"cannot determine confirmation suite from {result_path}")
        peeked_suites.append(result_protocol["suite"])
    _require(len(set(peeked_suites)) == 1, "result JSON files belong to different suites")
    spec = _build_spec(protocol, protocol_path, peeked_suites[0])
    analysis_protocol = protocol["analysis"]

    runs: dict[str, LoadedRun] = {}
    sample_paths: set[Path] = set()
    cache_dirs: set[Path] = set()
    for result_path in resolved_results:
        run = _load_run(result_path, spec, protocol)
        _require(run.arm not in runs, f"duplicate result arm {run.arm}")
        _require(run.sample_path not in sample_paths, "two arms point to the same sample artifact")
        _require(run.cache_dir not in cache_dirs, "two arms used the same TensorBridge cache")
        runs[run.arm] = run
        sample_paths.add(run.sample_path)
        cache_dirs.add(run.cache_dir)
    _require(set(runs) == set(EXPECTED_ARMS), "the four required arms are not all present")

    reference = runs["normal_a8"].identities
    reference_checkpoint = runs["normal_a8"].checkpoint_identity
    reference_source = runs["normal_a8"].source_identity
    reference_versions = runs["normal_a8"].runtime_versions
    for arm, run in runs.items():
        _require(run.identities == reference, f"sample hashes or documents differ for arm {arm}")
        _require(
            run.checkpoint_identity == reference_checkpoint,
            f"checkpoint identity differs across arms for {arm}",
        )
        _require(run.source_identity == reference_source, f"source identity differs for arm {arm}")
        _require(
            run.runtime_versions == reference_versions,
            f"runtime versions differ for arm {arm}",
        )
    identity_records = [
        {"doc_id": doc_id} | reference[doc_id] for doc_id in spec.expected_doc_ids
    ]
    identity_sha = _sha256(_canonical_json(identity_records))

    arm_summaries: dict[str, Any] = {}
    for arm in EXPECTED_ARMS:
        run = runs[arm]
        analysis_metrics = {
            metric: _accuracy(run.correctness[metric], spec.analysis_doc_ids)
            for metric in spec.metrics
        }
        full_metrics = {
            metric: _accuracy(run.correctness[metric], spec.expected_doc_ids)
            for metric in spec.metrics
        }
        summary: dict[str, Any] = {
            "analysis_metrics": analysis_metrics,
            "full_evaluation_metrics": full_metrics,
        }
        if run.format_valid is not None:
            valid = sum(run.format_valid.values())
            minimum = spec.format_min_valid
            assert minimum is not None
            raw_metric = _accuracy(run.raw_correctness[spec.primary_metric], spec.analysis_doc_ids)
            summary["lm_eval_raw_exact_match"] = raw_metric
            summary["format"] = {
                "valid": valid,
                "total": len(spec.expected_doc_ids),
                "valid_rate": valid / len(spec.expected_doc_ids),
                "minimum_valid": minimum,
                "gate_passed": valid >= minimum,
            }
        arm_summaries[arm] = summary

    comparison_specs = (
        ("fpma_default", "normal_a8", "primary", True, True),
        ("normal_a8", "official", "activation_quantization_control", False, False),
        ("ulp_v1", "fpma_default", "exploratory_ulp_vs_default", False, False),
        ("ulp_v1", "normal_a8", "exploratory_ulp_vs_normal", True, False),
    )
    comparisons = {
        f"{candidate}_vs_{baseline}": _comparison_summary(
            runs=runs,
            arm_summaries=arm_summaries,
            spec=spec,
            analysis_protocol=analysis_protocol,
            candidate=candidate,
            baseline=baseline,
            role=role,
            noninferiority_check=check,
            required_gate=required,
        )
        for candidate, baseline, role, check, required in comparison_specs
    }

    inputs = {
        arm: {
            "result_path": str(runs[arm].result_path),
            "result_sha256": runs[arm].result_sha256,
            "sample_artifact_path": str(runs[arm].sample_path),
            "sample_artifact_sha256": runs[arm].sample_sha256,
        }
        for arm in EXPECTED_ARMS
    }
    protocol_record: dict[str, Any] = {
        "path": str(protocol_path),
        "sha256": _sha256(protocol_encoded),
        "suite": spec.suite,
        "task": spec.task,
        "benchmark": spec.benchmark,
        "primary_metric": spec.primary_metric,
        "secondary_metrics": [metric for metric in spec.metrics if metric != spec.primary_metric],
        "analysis_exclude_doc_ids": list(spec.excluded_doc_ids),
        "noninferiority_margin": spec.noninferiority_margin,
        "paired_confidence_interval": analysis_protocol["paired_confidence_interval"],
        "mcnemar": analysis_protocol["mcnemar"],
    }
    if spec.sample_manifest_path is not None:
        protocol_record["sample_manifest"] = {
            "path": str(spec.sample_manifest_path),
            "sha256": spec.sample_manifest_sha256,
        }

    primary_passed = comparisons["fpma_default_vs_normal_a8"][
        "all_required_gates_passed"
    ]
    exploratory_ulp_passed = comparisons["ulp_v1_vs_normal_a8"][
        "threshold_and_format_checks_passed"
    ]
    return {
        "schema_version": 1,
        "format": OUTPUT_FORMAT,
        "status": "analysis_completed",
        "scientific_decision": {
            "primary_confirmation_passed": primary_passed,
            "exploratory_ulp_threshold_passed": exploratory_ulp_passed,
            "confidence_intervals_are_reported_but_not_a_gate": True,
        },
        "protocol": protocol_record,
        "inputs": inputs,
        "sample_pairing": {
            "task": spec.task,
            "filter": spec.filter_name,
            "evaluation_docs": len(spec.expected_doc_ids),
            "analysis_docs": len(spec.analysis_doc_ids),
            "excluded_docs": len(spec.excluded_doc_ids),
            "evaluation_doc_ids_sha256": _ids_sha256(spec.expected_doc_ids),
            "analysis_doc_ids_sha256": _ids_sha256(spec.analysis_doc_ids),
            "cross_arm_identity_sha256": identity_sha,
            "cross_arm_match": True,
            "cross_arm_checkpoint_match": True,
            "cross_arm_source_match": True,
            "cross_arm_runtime_versions_match": True,
        },
        "arms": arm_summaries,
        "comparisons": comparisons,
    }


def _write_output(path: Path, encoded: str, overwrite: bool, protected: set[Path]) -> None:
    path = path.expanduser().resolve()
    _require(path not in protected, "output path would overwrite an input or protocol file")
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(encoded, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", nargs="+", type=Path, help="the four arm result JSON files")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, help="also write the JSON report to this path")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = analyze_confirmation(args.result_json, args.protocol)
    encoded = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    if args.output is not None:
        protected = {path.expanduser().resolve() for path in args.result_json}
        protected.add(args.protocol.expanduser().resolve())
        for record in report["inputs"].values():
            protected.add(Path(record["sample_artifact_path"]).resolve())
        manifest = report["protocol"].get("sample_manifest")
        if isinstance(manifest, dict):
            protected.add(Path(manifest["path"]).resolve())
        _write_output(args.output, encoded, args.overwrite, protected)
    print(encoded, end="")


if __name__ == "__main__":
    main()
