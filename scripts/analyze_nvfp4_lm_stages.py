#!/usr/bin/env python3
"""Fail-closed paired analysis for TensorBridge six-arm accuracy stages."""

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
from types import SimpleNamespace
from typing import Any

if __package__:
    from scripts import analyze_nvfp4_lm_confirmation as confirmation
else:  # Support ``python scripts/analyze_nvfp4_lm_stages.py``.
    import analyze_nvfp4_lm_confirmation as confirmation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    REPO_ROOT / "vllm/plugins/tensorbridge_evaluation/protocols/accuracy_expand_v2.json"
)
DEFAULT_CONFIRM_PROTOCOL = (
    REPO_ROOT / "vllm/plugins/tensorbridge_evaluation/protocols/accuracy_confirm_v1.json"
)
OUTPUT_FORMAT = "tensorbridge_nvfp4_lm_stage_analysis"
EXPECTED_PROTOCOL_SHA256 = (
    "c56962917b63290a879832b562421f8ae938820d387d6be431b73fded7085a08"
)
EXPECTED_CONFIRM_PROTOCOL_SHA256 = (
    "ed170a281382fd023aebb39cffe9c8030c6b72f3e8b7ffd53449d5ba014ad5da"
)
PREREGISTERED_CONFIRM_PROTOCOL_SHA256 = (
    "3a3ab3a617143b29eb94af5a75316b74f44d66841d2970a85206c4ac091350a2"
)

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
    "selector_alpha1": {
        "label": "tensorbridge_selector_alpha_1",
        "backend": "tensorbridge",
        "alpha": 1.0,
        "selector": "normal_b8_sse",
        "ulp_correction": False,
    },
    "ulp_v1": {
        "label": "tensorbridge_ulp_scale_msb_flag_v1",
        "backend": "tensorbridge",
        "alpha": 1.0,
        "selector": "none",
        "ulp_correction": True,
    },
    "alpha_0960": {
        "label": "tensorbridge_global_alpha_0_960",
        "backend": "tensorbridge",
        "alpha": 0.960,
        "selector": "none",
        "ulp_correction": False,
    },
}

EXPECTED_COMPARISONS = (
    ("normal_a8", "official"),
    ("fpma_default", "normal_a8"),
    ("selector_alpha1", "fpma_default"),
    ("selector_alpha1", "normal_a8"),
    ("ulp_v1", "fpma_default"),
    ("ulp_v1", "normal_a8"),
    ("alpha_0960", "fpma_default"),
    ("alpha_0960", "normal_a8"),
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_HEAD_RE = re.compile(r"[0-9a-f]{40}\Z")
_CALLABLE_ADDRESS_RE = re.compile(r"(?<= at )0x[0-9a-fA-F]+")
_GSM8K_SYSTEM_INSTRUCTION = (
    "Use no more than six short sentences or equations. End with a separate final "
    "line in the exact form The answer is N. Replace N with the numeric answer only, "
    "omit units, and write nothing after that line."
)
_GSM8K_FORMAT_REGEX = (
    r"(?m)^The answer is (-?[0-9][0-9,]*(?:\.[0-9]+)?)\.?\s*\Z"
)
_BASE_NVRTC_FLAGS = "-DTENSORBRIDGE_NVFP4_SWZ64_B_DUAL_MMA_PREINT_LOAD=1"
_ULP_NVRTC_FLAGS = (
    _BASE_NVRTC_FLAGS
    + " -DTENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION=1"
    + " -DTENSORBRIDGE_NVFP4_FPMA_ULP_SCALE_MSB_FLAG_V1=1"
)
_RUNTIME_VERSIONS = {
    "datasets": "4.8.5",
    "lm_eval": "0.4.11",
    "torch": "2.11.0+cu128",
    "transformers": "5.9.0",
    "vllm": "0.20.2+cu128",
}
_DEFAULT_CACHE_SEED = (
    "f4dfa648fb1c00544671c33780496aac5c8b4372ad36784f63f36361cb02d5b2"
)
_ULP_CACHE_SEED = (
    "da6b312e7ee12d4921f0e1178a27816d81065393fbdeb6770376f6ba391bbbbb"
)
_CACHE_SEEDS = {
    "fpma_default": _DEFAULT_CACHE_SEED,
    "selector_alpha1": _DEFAULT_CACHE_SEED,
    "ulp_v1": _ULP_CACHE_SEED,
    "alpha_0960": _DEFAULT_CACHE_SEED,
}
_EXPECTED_CHECKPOINT = {
    "repo_id": "nvidia/Qwen3.6-27B-NVFP4",
    "revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
    "content_sha256": (
        "4ec0960247ca03fd10a9883d20de08d3795760ac1043fe7a9db6151b4074203f"
    ),
}
_EXPECTED_CHECKPOINT_SOURCE = {
    "provider": "huggingface",
    "repo_id": _EXPECTED_CHECKPOINT["repo_id"],
    "revision": _EXPECTED_CHECKPOINT["revision"],
    "revision_kind": "git_commit",
}
_MODEL_PATH = "/data/user/jzou521/models/nvidia/Qwen3.6-27B-NVFP4"

_GSM8K_DATASET = {
    "path": "openai/gsm8k",
    "name": "main",
    "split": "test",
    "size": 1319,
    "revision": "740312add88f781978c0658806c59bc2815b9866",
    "datasets_fingerprint": "59ec1b7f9357c7a2",
    "canonicalization": "sorted_minified_utf8_jsonl_with_doc_id_v1",
    "canonical_jsonl_sha256": (
        "f788a609d63ba8cbe610b0e64994033027d559c435775af0e471bb10ec1ee326"
    ),
}
_STAGE1_DATASET_CONTRACTS = {
    "hellaswag": {
        "path": "Rowan/hellaswag",
        "name": None,
        "split": "validation",
        "size": 10042,
        "revision": None,
        "datasets_fingerprint": "71a62ba9d49be089",
        "cached_builder_hash": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "canonicalization": "sorted_minified_utf8_jsonl_with_doc_id_v1",
        "canonical_jsonl_sha256": (
            "a09a861cd677a5df4bbe076e9079bd4c5f66487e849b1a4a9e0f32279a0b742e"
        ),
    },
    "winogrande": {
        "path": "allenai/winogrande",
        "name": "winogrande_xl",
        "split": "validation",
        "size": 1267,
        "revision": None,
        "datasets_fingerprint": "5b125086384c0403",
        "cached_builder_hash": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        "canonicalization": "sorted_minified_utf8_jsonl_with_doc_id_v1",
        "canonical_jsonl_sha256": (
            "f1507af323384e8047d617a63d268bac5a134d7f30f62b1152ce1490475da868"
        ),
    },
}
_STAGE2_DATASET_CONTRACTS = {
    "mmlu_pro": {
        "path": "TIGER-Lab/MMLU-Pro",
        "name": "default",
        "split": "test",
        "size": 12032,
        "revision": None,
        "datasets_fingerprint": "0072dd0a32d256fc",
        "cached_builder_hash": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "canonicalization": "sorted_minified_utf8_jsonl_with_doc_id_v1",
        "canonical_jsonl_sha256": (
            "372a592e9d8e15302d6e7ccab29e040ddae786b5dcddce18365b96ff43939f68"
        ),
    }
}
_STAGE_TASK_SOURCE_SHA256 = {
    "stage1_mc": {
        "hellaswag": (
            "652027d438d4bc2f23b3525f7e3f255eae0be75cff322d0b7f28bfebf28484c0"
        ),
        "winogrande": (
            "abf47611da8131be4ea5450e1189203de902c8c6d00593f45bf59e62f03b1509"
        ),
    },
    "stage2_mmlu_pro": {
        "mmlu_pro": (
            "7cff51577642fb1ae79a3c7991ce773f2505afb6fc8e7802988c94ad094d10ed"
        )
    },
    "stage3_generation": {
        "tensorbridge_gsm8k_relative_smoke": (
            "47c193604c56a717778641320d09ed49cab619f00406ce3588cc235c447a36a5"
        )
    },
}
_STAGE_SELECTED_DOC_SHA256 = {
    "hellaswag": (
        "39af1ea86866600455e0543d95601d1e158b0b42c384667ddb9dd3346e09024a"
    ),
    "winogrande": (
        "2745f5df477b490dc9f3b1d02b4a1a0eac4a9c7794be959a44451835f6ebdd33"
    ),
}
_STAGE1_COMPOSITE_DOC_SHA256 = (
    "b8f345a9030494ab72895765d51cc312f0600043fc6f0eac489e516e451c310a"
)
_STAGE2_COMPOSITE_DOC_SHA256 = (
    "8fc7f27b644625b3d0d6efa9b2d83523b7f7ff33c8cb91616acd266a32dda312"
)
_MMLU_PRO_LEAVES = {
    "mmlu_pro_biology": 717,
    "mmlu_pro_business": 789,
    "mmlu_pro_chemistry": 1132,
    "mmlu_pro_computer_science": 410,
    "mmlu_pro_economics": 844,
    "mmlu_pro_engineering": 969,
    "mmlu_pro_health": 818,
    "mmlu_pro_history": 381,
    "mmlu_pro_law": 1101,
    "mmlu_pro_math": 1351,
    "mmlu_pro_other": 924,
    "mmlu_pro_philosophy": 499,
    "mmlu_pro_physics": 1299,
    "mmlu_pro_psychology": 798,
}
_STAGE3_MANIFEST_SHA256 = (
    "37077471979c08a0b71ecdb384c0d4b5c7d9e37f59225ba724bb0ba3693915e4"
)
_STAGE3_IDS_SHA256 = (
    "481e6b075c43ca12c490c460a38430aa57dd22dc83aa5915edc7a268af771cb4"
)
_STAGE3_SELECTED_DOC_SHA256 = (
    "ee105ca1a10576013a6708dba5da76d5330cecf4481e6acc02db9018a79bf263"
)


PairKey = tuple[str, int, str]


@dataclass(frozen=True)
class LeafSpec:
    task: str
    metrics: tuple[str, ...]
    primary_metric: str
    filter_name: str
    expected_doc_ids: tuple[int, ...]
    analysis_doc_ids: tuple[int, ...]
    original_size: int
    selected_docs_sha256: str | None = None

    @property
    def expected_keys(self) -> tuple[PairKey, ...]:
        return tuple((self.task, doc_id, self.filter_name) for doc_id in self.expected_doc_ids)

    @property
    def analysis_keys(self) -> tuple[PairKey, ...]:
        return tuple((self.task, doc_id, self.filter_name) for doc_id in self.analysis_doc_ids)


@dataclass(frozen=True)
class StageSpec:
    suite: str
    requested_tasks: tuple[str, ...]
    leaves: tuple[LeafSpec, ...]
    num_fewshot: int
    max_model_len: int
    max_gen_toks: int
    generation: bool
    apply_chat_template: bool
    prompt_format: str
    limit_count: int | None
    dataset_contracts: dict[str, dict[str, Any]] | None
    manifest_path: Path | None
    manifest_sha256: str | None
    manifest: dict[str, Any] | None
    format_regex: str | None
    format_min_valid: int | None
    post_confirmation: bool
    confirmation_spec: Any | None = None

    @property
    def leaf_by_task(self) -> dict[str, LeafSpec]:
        return {leaf.task: leaf for leaf in self.leaves}


@dataclass
class LoadedRun:
    arm: str
    result_path: Path
    result_sha256: str
    sample_paths: dict[str, Path]
    sample_sha256: dict[str, str]
    identities: dict[PairKey, dict[str, Any]]
    correctness: dict[tuple[str, str], dict[PairKey, bool]]
    raw_correctness: dict[tuple[str, str], dict[PairKey, bool]]
    format_valid: dict[PairKey, bool] | None
    normalized_configs: dict[str, Any]
    checkpoint_identity: dict[str, Any]
    source_identity: dict[str, Any]
    runtime_versions: dict[str, str]
    module_paths: dict[str, str]
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


def _ids_sha256(doc_ids: list[int] | tuple[int, ...]) -> str:
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


def _normalize_callable_addresses(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_callable_addresses(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_normalize_callable_addresses(child) for child in value]
    if isinstance(value, str) and (
        value.startswith("<function ") or "partial(<function " in value
    ):
        return _CALLABLE_ADDRESS_RE.sub("<ADDR>", value)
    return value


def _binary_metric(value: Any, label: str) -> bool:
    if type(value) not in {int, float, bool}:
        _fail(f"{label} is not binary numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
        _fail(f"{label} is not 0 or 1")
    return bool(numeric)


def _load_manifest(
    path: Path,
    expected_sha256: str,
    task: str,
) -> dict[str, Any]:
    manifest, encoded = _load_json_object(path, "sample manifest")
    _require(_sha256(encoded) == expected_sha256, "sample manifest raw SHA256 mismatch")
    selection = manifest.get("selection")
    _require(isinstance(selection, dict), "sample manifest selection is missing")
    algorithm = selection.get("algorithm")
    expected_keys = {"schema_version", "format", "dataset", "selection", "tasks"}
    if algorithm == "sha256_rank_excluding_ids_v1":
        expected_keys.add("excluded_doc_ids")
    _require(set(manifest) == expected_keys, "sample manifest keys changed")
    _require(manifest.get("schema_version") == 1, "sample manifest schema changed")
    _require(
        manifest.get("format") == "tensorbridge_lm_eval_sample_manifest",
        "sample manifest format changed",
    )
    _require(manifest.get("dataset") == _GSM8K_DATASET, "sample dataset contract changed")
    tasks = manifest.get("tasks")
    _require(isinstance(tasks, dict) and set(tasks) == {task}, "manifest task mismatch")
    ids = tasks[task]
    _require(
        isinstance(ids, list) and ids and all(type(doc_id) is int for doc_id in ids),
        "manifest IDs must be non-empty integers",
    )
    _require(ids == sorted(set(ids)), "manifest IDs must be unique and ascending")
    _require(selection.get("count") == len(ids), "manifest sample count mismatch")
    _require(selection.get("ids_sha256") == _ids_sha256(ids), "manifest IDs SHA mismatch")
    split_size = _GSM8K_DATASET["size"]
    candidate_start = selection.get("candidate_start")
    namespace = selection.get("namespace")
    _require(
        type(candidate_start) is int and 0 <= candidate_start < split_size,
        "manifest candidate start is invalid",
    )
    _require(isinstance(namespace, str) and namespace, "manifest namespace is invalid")
    excluded: list[int] = []
    if algorithm == "sha256_rank_excluding_ids_v1":
        excluded_value = manifest.get("excluded_doc_ids")
        _require(
            isinstance(excluded_value, list)
            and all(type(doc_id) is int for doc_id in excluded_value),
            "manifest exclusions are invalid",
        )
        excluded = excluded_value
        _require(excluded == sorted(set(excluded)), "manifest exclusions are not closed")
        _require(
            selection.get("excluded_ids_sha256") == _ids_sha256(excluded),
            "manifest exclusion SHA mismatch",
        )
    else:
        _require(algorithm == "sha256_rank_v1", "unsupported sample selection algorithm")
    excluded_set = set(excluded)
    ranked = sorted(
        (
            doc_id
            for doc_id in range(candidate_start, split_size)
            if doc_id not in excluded_set
        ),
        key=lambda doc_id: (
            hashlib.sha256(f"{namespace}:{doc_id}".encode("utf-8")).digest(),
            doc_id,
        ),
    )[: len(ids)]
    _require(ids == sorted(ranked), "manifest IDs do not match SHA256 ranking")
    return manifest


def _validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    _require(protocol.get("schema_version") == 1, "unsupported expansion protocol schema")
    _require(
        protocol.get("format") == "tensorbridge_accuracy_expansion_protocol",
        "unexpected expansion protocol format",
    )
    _require(
        protocol.get("created_before_prospective_stage_runs") is True,
        "prospective stages were not frozen before execution",
    )
    relation = protocol.get("relation_to_confirmation_v1")
    _require(
        isinstance(relation, dict)
        and relation.get("protocol_sha256")
        == PREREGISTERED_CONFIRM_PROTOCOL_SHA256
        and relation.get("unchanged") is True
        and relation.get("user_authorized_expansion_after_v1") is True
        and relation.get("post_confirmation_results_are_not_reclassified_as_preregistered")
        is True,
        "confirmation-v1 relationship changed",
    )
    _require(protocol.get("checkpoint") == _EXPECTED_CHECKPOINT, "checkpoint changed")
    _require(protocol.get("arms") == list(EXPECTED_ARMS), "six-arm order changed")
    zero_cost = protocol.get("zero_cost_contract")
    _require(
        isinstance(zero_cost, dict)
        and zero_cost.get("meaning")
        == "no additional steady_state_kernel_instructions_or_weight_bytes"
        and zero_cost.get("selector_alpha1")
        == {
            "alpha": 1.0,
            "selector": "normal_b8_sse",
            "selector_chunk_rows": 256,
            "load_time_work_or_temporary_memory_may_increase": True,
        }
        and zero_cost.get("alpha_0960")
        == {
            "alpha": 0.96,
            "selector": "none",
            "selected_on_wikitext2": True,
            "held_out_interpretation_required": True,
        },
        "zero-cost arm contract changed",
    )
    sensitivity = protocol.get("post_confirmation_sensitivity")
    _require(
        isinstance(sensitivity, dict)
        and sensitivity.get("suites") == ["confirm_mc", "confirm_generation"]
        and sensitivity.get("new_arms_only") == ["selector_alpha1", "alpha_0960"]
        and sensitivity.get("reuse_existing_v1_arms")
        == ["official", "normal_a8", "fpma_default", "ulp_v1"],
        "post-confirmation sensitivity contract changed",
    )
    analysis = protocol.get("analysis")
    _require(isinstance(analysis, dict), "expansion analysis block is missing")
    expected_comparisons = [
        f"{candidate} - {baseline}"
        for candidate, baseline in EXPECTED_COMPARISONS
    ]
    _require(analysis.get("paired") is True, "analysis must remain paired")
    _require(
        analysis.get("pairing_key") == ["leaf_task", "doc_id", "filter"],
        "pairing key changed",
    )
    _require(analysis.get("comparisons") == expected_comparisons, "comparison set changed")
    _require(analysis.get("screening_margin_vs_normal_a8") == -0.05, "screening margin changed")
    _require(
        analysis.get("screening_decision")
        == "paired_point_estimate_delta_greater_than_or_equal_to_margin",
        "screening rule changed",
    )
    _require(
        analysis.get("confidence_intervals_are_reported_but_not_a_gate") is True,
        "confidence interval decision rule changed",
    )
    ci = analysis.get("paired_confidence_interval")
    _require(
        isinstance(ci, dict)
        and ci.get("method") == "paired_nonparametric_bootstrap_percentile"
        and ci.get("confidence_level") == 0.95
        and ci.get("resamples") == 10_000
        and ci.get("seed") == 20_260_718
        and ci.get("mmlu_pro_resampling_unit")
        == "document_within_fixed_category",
        "paired bootstrap protocol changed",
    )
    _require(
        analysis.get("mcnemar")
        == {"method": "exact_binomial", "alternative": "two-sided"},
        "McNemar protocol changed",
    )
    for flag in (
        "invalid_generation_is_incorrect",
        "do_not_drop_ambiguous_or_discordant_samples",
        "no_result_dependent_arm_dropping_between_stages",
        "all_six_arms_run_in_all_three_stages",
    ):
        _require(analysis.get(flag) is True, f"analysis flag changed: {flag}")
    execution = protocol.get("execution")
    _require(isinstance(execution, dict), "execution protocol is missing")
    _require(
        execution.get("gpu") == "NVIDIA H100 80GB HBM3"
        and execution.get("gpus_per_job") == 1
        and execution.get("cpus_per_job") == 8
        and execution.get("host_memory_gib") == 80
        and execution.get("cpu_thread_limit") == 8
        and execution.get("tensor_parallel_size") == 1
        and execution.get("dtype") == "bfloat16"
        and execution.get("quantization") == "modelopt_mixed"
        and execution.get("batch_size") == "auto"
        and execution.get("gpu_memory_utilization") == 0.5
        and execution.get("max_num_seqs") == 8
        and execution.get("max_model_len_by_stage")
        == {
            "stage1_mc": 4096,
            "stage2_mmlu_pro": 16384,
            "stage3_generation": 4096,
        }
        and execution.get("runtime_versions") == _RUNTIME_VERSIONS
        and execution.get("allow_runtime_version_mismatch") is False
        and execution.get("bootstrap_iters") == 0
        and execution.get("seeds")
        == {"python": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234, "model": 1234}
        and execution.get("independent_physical_cache_per_arm_and_stage") is True
        and execution.get("default_abi_cache_seed_sha256") == _DEFAULT_CACHE_SEED
        and execution.get("ulp_v1_cache_seed_sha256") == _ULP_CACHE_SEED,
        "execution contract changed",
    )
    _require(protocol.get("stop_after_stage3") is True, "stage stop rule changed")
    _require(
        protocol.get("additional_tasks_not_authorized") is True,
        "additional-task authorization changed",
    )
    return analysis


def _protocol_snapshot(
    contract: dict[str, Any], task_source_sha256: str, selected_docs_sha256: str
) -> dict[str, Any]:
    return {
        key: value
        for key, value in contract.items()
        if key not in {"revision", "canonicalization"}
    } | {
        "task_source_sha256": task_source_sha256,
        "selected_docs_sha256": selected_docs_sha256,
    }


def _validate_prospective_stages(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stages = protocol.get("prospective_stages")
    _require(
        isinstance(stages, list)
        and [stage.get("suite") for stage in stages if isinstance(stage, dict)]
        == ["stage1_mc", "stage2_mmlu_pro", "stage3_generation"],
        "prospective stage list or order changed",
    )
    by_suite = {stage["suite"]: stage for stage in stages}
    stage1 = by_suite["stage1_mc"]
    _require(stage1.get("index") == 1, "Stage 1 index changed")
    _require(stage1.get("tasks") == ["hellaswag", "winogrande"], "Stage 1 tasks changed")
    _require(
        stage1.get("selection")
        == {
            "kind": "first_n_after_lm_eval_task_preprocessing",
            "limit_count_per_task": 512,
            "expected_total": 1024,
        },
        "Stage 1 sample selection changed",
    )
    _require(
        stage1.get("metrics")
        == {"hellaswag": "acc_norm,none", "winogrande": "acc,none"},
        "Stage 1 metrics changed",
    )
    expected_stage1_snapshots = {
        task: _protocol_snapshot(
            _STAGE1_DATASET_CONTRACTS[task],
            _STAGE_TASK_SOURCE_SHA256["stage1_mc"][task],
            _STAGE_SELECTED_DOC_SHA256[task],
        )
        for task in ("hellaswag", "winogrande")
    }
    _require(
        stage1.get("dataset_snapshots") == expected_stage1_snapshots,
        "Stage 1 dataset snapshots changed",
    )
    _require(
        stage1.get("composite_sample_identity_sha256")
        == _STAGE1_COMPOSITE_DOC_SHA256,
        "Stage 1 composite sample identity changed",
    )

    stage2 = by_suite["stage2_mmlu_pro"]
    _require(stage2.get("index") == 2, "Stage 2 index changed")
    _require(stage2.get("tasks") == ["mmlu_pro"], "Stage 2 tasks changed")
    _require(
        stage2.get("selection")
        == {
            "kind": "first_n_per_leaf_after_lm_eval_task_preprocessing",
            "limit_count_per_leaf": 64,
            "leaf_count": 14,
            "expected_total": 896,
        },
        "Stage 2 sample selection changed",
    )
    _require(stage2.get("num_fewshot") == 5, "Stage 2 fewshot count changed")
    _require(stage2.get("max_model_len") == 16384, "Stage 2 model length changed")
    _require(
        stage2.get("metric") == "exact_match,custom-extract",
        "Stage 2 metric changed",
    )
    _require(stage2.get("leaf_tasks") == _MMLU_PRO_LEAVES, "MMLU-Pro leaves changed")
    expected_stage2_snapshot = _protocol_snapshot(
        _STAGE2_DATASET_CONTRACTS["mmlu_pro"],
        _STAGE_TASK_SOURCE_SHA256["stage2_mmlu_pro"]["mmlu_pro"],
        _STAGE2_COMPOSITE_DOC_SHA256,
    )
    _require(
        stage2.get("dataset_snapshot") == expected_stage2_snapshot,
        "Stage 2 dataset snapshot changed",
    )
    _require(
        stage2.get("aggregation")
        == "equal_64_sample_micro_average_across_14_leaves",
        "Stage 2 aggregation changed",
    )

    stage3 = by_suite["stage3_generation"]
    _require(stage3.get("index") == 3, "Stage 3 index changed")
    _require(
        stage3.get("tasks") == ["tensorbridge_gsm8k_relative_smoke"],
        "Stage 3 task changed",
    )
    _require(
        stage3.get("task_source_sha256")
        == _STAGE_TASK_SOURCE_SHA256["stage3_generation"][
            "tensorbridge_gsm8k_relative_smoke"
        ],
        "Stage 3 task source changed",
    )
    _require(
        stage3.get("selection")
        == {
            "kind": "sha256_rank_excluding_ids_v1",
            "count": 256,
            "excluded_count": 144,
            "excluded_ids_sha256": (
                "f6cf62e20cba6e0366d6a4dbdf19715d806afa2f7d5042ed26dea058a50a8bce"
            ),
            "manifest": (
                "vllm/plugins/tensorbridge_evaluation/samples/"
                "gsm8k_test_stage3_sha256_rank_256.json"
            ),
            "manifest_sha256": _STAGE3_MANIFEST_SHA256,
            "ids_sha256": _STAGE3_IDS_SHA256,
            "selected_docs_sha256": _STAGE3_SELECTED_DOC_SHA256,
        },
        "Stage 3 sample selection changed",
    )
    _require(
        stage3.get("metric") == "exact_match,final-answer",
        "Stage 3 metric changed",
    )
    _require(
        stage3.get("format_gate") == {"minimum_valid": 254, "total": 256},
        "Stage 3 format gate changed",
    )
    _require(
        stage3.get("thinking") is False
        and stage3.get("temperature") == 0.0
        and stage3.get("max_gen_toks") == 1024,
        "Stage 3 generation settings changed",
    )
    return by_suite


def _build_spec(
    protocol: dict[str, Any],
    protocol_path: Path,
    suite: str,
    confirm_protocol_path: Path = DEFAULT_CONFIRM_PROTOCOL,
) -> StageSpec:
    _validate_protocol(protocol)
    stages = _validate_prospective_stages(protocol)
    if suite == "stage1_mc":
        return StageSpec(
            suite=suite,
            requested_tasks=("hellaswag", "winogrande"),
            leaves=(
                LeafSpec(
                    task="hellaswag",
                    metrics=("acc", "acc_norm"),
                    primary_metric="acc_norm",
                    filter_name="none",
                    expected_doc_ids=tuple(range(512)),
                    analysis_doc_ids=tuple(range(512)),
                    original_size=10042,
                    selected_docs_sha256=_STAGE_SELECTED_DOC_SHA256["hellaswag"],
                ),
                LeafSpec(
                    task="winogrande",
                    metrics=("acc",),
                    primary_metric="acc",
                    filter_name="none",
                    expected_doc_ids=tuple(range(512)),
                    analysis_doc_ids=tuple(range(512)),
                    original_size=1267,
                    selected_docs_sha256=_STAGE_SELECTED_DOC_SHA256["winogrande"],
                ),
            ),
            num_fewshot=0,
            max_model_len=4096,
            max_gen_toks=256,
            generation=False,
            apply_chat_template=False,
            prompt_format="completion",
            limit_count=512,
            dataset_contracts=_STAGE1_DATASET_CONTRACTS,
            manifest_path=None,
            manifest_sha256=None,
            manifest=None,
            format_regex=None,
            format_min_valid=None,
            post_confirmation=False,
        )
    if suite == "stage2_mmlu_pro":
        leaves = tuple(
            LeafSpec(
                task=task,
                metrics=("exact_match",),
                primary_metric="exact_match",
                filter_name="custom-extract",
                expected_doc_ids=tuple(range(64)),
                analysis_doc_ids=tuple(range(64)),
                original_size=original_size,
            )
            for task, original_size in _MMLU_PRO_LEAVES.items()
        )
        return StageSpec(
            suite=suite,
            requested_tasks=("mmlu_pro",),
            leaves=leaves,
            num_fewshot=5,
            max_model_len=16384,
            max_gen_toks=2048,
            generation=True,
            apply_chat_template=True,
            prompt_format="chat_nonthinking",
            limit_count=64,
            dataset_contracts=_STAGE2_DATASET_CONTRACTS,
            manifest_path=None,
            manifest_sha256=None,
            manifest=None,
            format_regex=None,
            format_min_valid=None,
            post_confirmation=False,
        )
    if suite == "stage3_generation":
        stage = stages[suite]
        selection = stage["selection"]
        manifest_path = _resolve_existing_path(
            selection["manifest"],
            relative_to=(REPO_ROOT, protocol_path.parent, Path.cwd()),
            label="Stage 3 sample manifest",
        )
        manifest = _load_manifest(
            manifest_path,
            selection["manifest_sha256"],
            "tensorbridge_gsm8k_relative_smoke",
        )
        _require(
            manifest["selection"]["ids_sha256"] == selection["ids_sha256"]
            and manifest["selection"]["selected_docs_sha256"]
            == selection["selected_docs_sha256"]
            and manifest["selection"]["excluded_ids_sha256"]
            == selection["excluded_ids_sha256"]
            and len(manifest["excluded_doc_ids"]) == selection["excluded_count"],
            "Stage 3 manifest differs from the expansion protocol",
        )
        ids = tuple(manifest["tasks"]["tensorbridge_gsm8k_relative_smoke"])
        return StageSpec(
            suite=suite,
            requested_tasks=("tensorbridge_gsm8k_relative_smoke",),
            leaves=(
                LeafSpec(
                    task="tensorbridge_gsm8k_relative_smoke",
                    metrics=("exact_match",),
                    primary_metric="exact_match",
                    filter_name="final-answer",
                    expected_doc_ids=ids,
                    analysis_doc_ids=ids,
                    original_size=1319,
                    selected_docs_sha256=_STAGE3_SELECTED_DOC_SHA256,
                ),
            ),
            num_fewshot=0,
            max_model_len=4096,
            max_gen_toks=1024,
            generation=True,
            apply_chat_template=True,
            prompt_format="chat_nonthinking",
            limit_count=None,
            dataset_contracts=None,
            manifest_path=manifest_path,
            manifest_sha256=selection["manifest_sha256"],
            manifest=manifest,
            format_regex=_GSM8K_FORMAT_REGEX,
            format_min_valid=254,
            post_confirmation=False,
        )
    if suite not in {"confirm_mc", "confirm_generation"}:
        _fail(f"unsupported suite: {suite!r}")

    confirm_protocol_path = confirm_protocol_path.expanduser().resolve()
    confirm_protocol, encoded = _load_json_object(
        confirm_protocol_path, "confirmation-v1 protocol"
    )
    _require(
        _sha256(encoded) == EXPECTED_CONFIRM_PROTOCOL_SHA256,
        "confirmation-v1 protocol SHA256 changed",
    )
    confirm_spec = confirmation._build_spec(
        confirm_protocol, confirm_protocol_path, suite
    )
    leaf = LeafSpec(
        task=confirm_spec.task,
        metrics=confirm_spec.metrics,
        primary_metric=confirm_spec.primary_metric,
        filter_name=confirm_spec.filter_name,
        expected_doc_ids=confirm_spec.expected_doc_ids,
        analysis_doc_ids=confirm_spec.analysis_doc_ids,
        original_size=confirm_spec.dataset_contract["size"],
        selected_docs_sha256=(
            confirm_spec.sample_selection["selected_docs_sha256"]
            if confirm_spec.sample_selection is not None
            else None
        ),
    )
    return StageSpec(
        suite=suite,
        requested_tasks=(confirm_spec.task,),
        leaves=(leaf,),
        num_fewshot=0,
        max_model_len=4096,
        max_gen_toks=1024 if suite == "confirm_generation" else 256,
        generation=suite == "confirm_generation",
        apply_chat_template=suite == "confirm_generation",
        prompt_format=("chat_nonthinking" if suite == "confirm_generation" else "completion"),
        limit_count=None,
        dataset_contracts=None,
        manifest_path=confirm_spec.sample_manifest_path,
        manifest_sha256=confirm_spec.sample_manifest_sha256,
        manifest=None,
        format_regex=confirm_spec.format_regex,
        format_min_valid=confirm_spec.format_min_valid,
        post_confirmation=True,
        confirmation_spec=confirm_spec,
    )


_HELLASWAG_PROCESS_DOCS = '''def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)
'''
_WINOGRANDE_DOC_TO_CHOICE = '''def doc_to_choice(doc):
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]
'''
_WINOGRANDE_DOC_TO_TARGET = '''def doc_to_target(doc):
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()
'''
_WINOGRANDE_DOC_TO_TEXT = '''def doc_to_text(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]
'''


def _common_fewshot_config(
    *,
    doc_to_text: Any,
    doc_to_target: Any,
    doc_to_choice: Any,
    sampler: str = "default",
    split: str | None = None,
    process_docs: Any = None,
) -> dict[str, Any]:
    return {
        "sampler": sampler,
        "samples": None,
        "doc_to_text": doc_to_text,
        "doc_to_target": doc_to_target,
        "doc_to_choice": doc_to_choice,
        "target_delimiter": " ",
        "fewshot_delimiter": "\n\n",
        "gen_prefix": None,
        "split": split,
        "fewshot_indices": None,
        "process_docs": process_docs,
    }


def _expected_hellaswag_config() -> dict[str, Any]:
    return {
        "task": "hellaswag",
        "dataset_path": "Rowan/hellaswag",
        "description": "",
        "training_split": "train",
        "validation_split": "validation",
        "process_docs": _HELLASWAG_PROCESS_DOCS,
        "doc_to_text": "{{query}}",
        "doc_to_target": "{{label}}",
        "doc_to_choice": "choices",
        "should_decontaminate": False,
        "num_fewshot": 0,
        "fewshot_config": _common_fewshot_config(
            doc_to_text="{{query}}",
            doc_to_target="{{label}}",
            doc_to_choice="choices",
            process_docs="<function process_docs at <ADDR>>",
        ),
        "metric_list": [
            {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
            {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True},
        ],
        "output_type": "multiple_choice",
        "repeats": 1,
        "target_delimiter": " ",
        "fewshot_delimiter": "\n\n",
        "tag": ["multiple_choice"],
        "metadata": {"version": 1.0},
        "unsafe_code": False,
    }


def _expected_winogrande_config() -> dict[str, Any]:
    return {
        "task": "winogrande",
        "dataset_path": "allenai/winogrande",
        "dataset_name": "winogrande_xl",
        "description": "",
        "training_split": "train",
        "validation_split": "validation",
        "doc_to_text": _WINOGRANDE_DOC_TO_TEXT,
        "doc_to_target": _WINOGRANDE_DOC_TO_TARGET,
        "doc_to_choice": _WINOGRANDE_DOC_TO_CHOICE,
        "should_decontaminate": True,
        "doc_to_decontamination_query": "sentence",
        "num_fewshot": 0,
        "fewshot_config": _common_fewshot_config(
            doc_to_text="<function doc_to_text at <ADDR>>",
            doc_to_target="<function doc_to_target at <ADDR>>",
            doc_to_choice="<function doc_to_choice at <ADDR>>",
        ),
        "metric_list": [
            {"metric": "acc", "aggregation": "mean", "higher_is_better": True}
        ],
        "output_type": "multiple_choice",
        "repeats": 1,
        "target_delimiter": " ",
        "fewshot_delimiter": "\n\n",
        "metadata": {"version": 1.0},
        "unsafe_code": False,
    }


def _mmlu_subject(task: str) -> tuple[str, str]:
    alias = task.removeprefix("mmlu_pro_")
    subject = alias.replace("_", " ")
    return alias, subject


def _expected_mmlu_config(task: str) -> dict[str, Any]:
    alias, subject = _mmlu_subject(task)
    partial_prefix = "functools.partial(<function"
    return {
        "task": task,
        "task_alias": alias,
        "dataset_path": "TIGER-Lab/MMLU-Pro",
        "description": (
            "The following are multiple choice questions (with answers) about "
            f"{subject}. Think step by step and then finish your answer with "
            '"the answer is (X)" where X is the correct letter choice.\n'
        ),
        "test_split": "test",
        "fewshot_split": "validation",
        "process_docs": (
            f"{partial_prefix} process_docs at <ADDR>>, subject={subject!r})"
        ),
        "doc_to_text": (
            f"{partial_prefix} format_cot_example at <ADDR>>, including_answer=False)"
        ),
        "doc_to_target": "answer",
        "should_decontaminate": False,
        "num_fewshot": 5,
        "fewshot_config": _common_fewshot_config(
            doc_to_text=(
                f"{partial_prefix} format_cot_example at <ADDR>>, including_answer=True)"
            ),
            doc_to_target="",
            doc_to_choice=None,
            sampler="first_n",
            split="validation",
            process_docs=(
                f"{partial_prefix} process_docs at <ADDR>>, subject={subject!r})"
            ),
        ),
        "metric_list": [
            {
                "metric": "exact_match",
                "aggregation": "mean",
                "higher_is_better": True,
                "ignore_case": True,
                "ignore_punctuation": True,
            }
        ],
        "output_type": "generate_until",
        "generation_kwargs": {
            "until": ["Question:"],
            "do_sample": False,
            "temperature": 0.0,
            "max_gen_toks": 2048,
        },
        "filter_list": [
            {
                "name": "custom-extract",
                "filter": [
                    {
                        "function": "regex",
                        "regex_pattern": r"answer is \(?([ABCDEFGHIJ])\)?",
                    },
                    {"function": "take_first"},
                ],
            }
        ],
        "repeats": 1,
        "target_delimiter": " ",
        "fewshot_delimiter": "\n\n",
        "metadata": {"version": 3.0},
        "unsafe_code": False,
    }


def _expected_task_config(spec: StageSpec, leaf: LeafSpec) -> dict[str, Any]:
    if leaf.task == "hellaswag":
        return _expected_hellaswag_config()
    if leaf.task == "winogrande":
        return _expected_winogrande_config()
    if leaf.task.startswith("mmlu_pro_"):
        return _expected_mmlu_config(leaf.task)
    config_spec = spec.confirmation_spec
    if config_spec is None:
        config_spec = SimpleNamespace(
            suite=spec.suite,
            task=leaf.task,
            dataset_contract=_GSM8K_DATASET,
            format_regex=spec.format_regex,
        )
    return confirmation._expected_task_config(config_spec)


def _expected_environment(arm: str, *, include_selector_chunk: bool = True) -> dict[str, str]:
    arm_config = EXPECTED_ARMS[arm]
    environment = {
        "TENSORBRIDGE_VLLM_BACKEND": arm_config["backend"],
        "TENSORBRIDGE_NVFP4_FPMA_ALPHA": str(arm_config["alpha"]),
        "TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR": arm_config["selector"],
        "TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION": (
            "1" if arm_config["ulp_correction"] else "0"
        ),
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
    if include_selector_chunk:
        environment["TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR_CHUNK_ROWS"] = "256"
    return environment


def _allowed_environments(arm: str, spec: StageSpec) -> list[dict[str, str]]:
    current = _expected_environment(arm, include_selector_chunk=True)
    if not spec.post_confirmation:
        return [current]
    # Both post-confirm cohorts were launched before chunk-row provenance was added.
    return [current, _expected_environment(arm, include_selector_chunk=False)]


def _expected_engine_args(spec: StageSpec) -> dict[str, Any]:
    return {
        "pretrained": _MODEL_PATH,
        "quantization": "modelopt_mixed",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": spec.max_model_len,
        "gpu_memory_utilization": 0.5,
        "language_model_only": True,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "disable_log_stats": True,
        "max_num_seqs": 8,
        "seed": 1234,
        "enable_thinking": False,
        "max_gen_toks": spec.max_gen_toks,
    }


def _validate_result_protocol(result_protocol: dict[str, Any], spec: StageSpec) -> None:
    old_keys = {
        "suite",
        "tasks",
        "num_fewshot",
        "apply_chat_template",
        "fewshot_as_multiturn",
        "system_instruction",
        "enable_thinking",
        "think_end_token",
        "max_gen_toks",
        "generation",
        "prompt_format",
        "min_model_len",
        "analysis_exclude_doc_ids",
        "dataset_contract",
        "generation_kwargs",
        "limit",
        "sample_selection",
        "batch_size",
        "bootstrap_iters",
        "response_cache",
        "seeds",
        "engine_args",
    }
    expected_keys = old_keys | (
        {"dataset_contracts", "analysis_protocol"}
        if not spec.post_confirmation
        else set()
    )
    if spec.post_confirmation and "dataset_contracts" in result_protocol:
        expected_keys.add("dataset_contracts")
    _require(set(result_protocol) == expected_keys, "result protocol keys changed")
    if not spec.post_confirmation:
        analysis_protocol = result_protocol.get("analysis_protocol")
        analysis_path = (
            analysis_protocol.get("path")
            if isinstance(analysis_protocol, dict)
            else None
        )
        _require(
            isinstance(analysis_protocol, dict)
            and set(analysis_protocol) == {"path", "sha256"}
            and isinstance(analysis_path, str)
            and Path(analysis_path).is_absolute()
            and Path(analysis_path).name == "accuracy_expand_v2.json"
            and analysis_protocol.get("sha256") == EXPECTED_PROTOCOL_SHA256,
            "result analysis protocol identity changed",
        )
    _require(result_protocol.get("suite") == spec.suite, "result suite mismatch")
    _require(
        result_protocol.get("tasks") == list(spec.requested_tasks),
        "result requested tasks changed",
    )
    _require(result_protocol.get("num_fewshot") == spec.num_fewshot, "fewshot count changed")
    _require(
        result_protocol.get("apply_chat_template") == spec.apply_chat_template,
        "chat-template setting changed",
    )
    _require(
        result_protocol.get("fewshot_as_multiturn") is False,
        "fewshot multiturn setting changed",
    )
    expected_instruction = _GSM8K_SYSTEM_INSTRUCTION if spec.format_regex else None
    _require(
        result_protocol.get("system_instruction") == expected_instruction,
        "system instruction changed",
    )
    _require(result_protocol.get("enable_thinking") is False, "thinking was enabled")
    _require(result_protocol.get("think_end_token") is None, "think-end token changed")
    _require(result_protocol.get("max_gen_toks") == spec.max_gen_toks, "generation cap changed")
    _require(result_protocol.get("generation") == spec.generation, "generation mode changed")
    _require(result_protocol.get("prompt_format") == spec.prompt_format, "prompt format changed")
    expected_min_len = 16384 if spec.suite == "stage2_mmlu_pro" else 4096
    _require(
        result_protocol.get("min_model_len") == expected_min_len,
        "minimum model length changed",
    )
    exclusions = (
        list(spec.confirmation_spec.excluded_doc_ids)
        if spec.confirmation_spec is not None
        else []
    )
    _require(
        result_protocol.get("analysis_exclude_doc_ids") == exclusions,
        "analysis exclusion list changed",
    )
    _require(result_protocol.get("batch_size") == "auto", "batch mode changed")
    _require(result_protocol.get("bootstrap_iters") == 0, "lm-eval bootstrap was enabled")
    _require(result_protocol.get("response_cache") is None, "response cache was enabled")
    _require(
        result_protocol.get("seeds")
        == {"python": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},
        "evaluation seeds changed",
    )
    _require(
        result_protocol.get("engine_args") == _expected_engine_args(spec),
        "vLLM engine arguments changed",
    )
    expected_generation_kwargs = None
    if spec.suite == "stage2_mmlu_pro":
        expected_generation_kwargs = {"temperature": 0.0}
    elif spec.format_regex is not None:
        expected_generation_kwargs = {
            "temperature": 0.0,
            "max_gen_toks": spec.max_gen_toks,
        }
    _require(
        result_protocol.get("generation_kwargs") == expected_generation_kwargs,
        "generation kwargs changed",
    )

    if spec.limit_count is not None:
        expected_limit = {
            "kind": "count",
            "value": spec.limit_count,
            "from_suite_default": True,
        }
    else:
        expected_limit = {"kind": "none", "value": None}
    _require(result_protocol.get("limit") == expected_limit, "sample limit changed")

    if spec.suite == "stage1_mc":
        _require(
            result_protocol.get("dataset_contract") is None,
            "unexpected single dataset contract",
        )
        _require(
            result_protocol.get("dataset_contracts") == _STAGE1_DATASET_CONTRACTS,
            "Stage 1 dataset contracts changed",
        )
        _require(result_protocol.get("sample_selection") is None, "Stage 1 used explicit samples")
    elif spec.suite == "stage2_mmlu_pro":
        _require(
            result_protocol.get("dataset_contract") is None,
            "unexpected single dataset contract",
        )
        _require(
            result_protocol.get("dataset_contracts") == _STAGE2_DATASET_CONTRACTS,
            "Stage 2 dataset contracts changed",
        )
        _require(result_protocol.get("sample_selection") is None, "Stage 2 used explicit samples")
    elif spec.post_confirmation:
        confirmation._validate_result_protocol(result_protocol, spec.confirmation_spec)
        if "dataset_contracts" in result_protocol:
            _require(
                result_protocol["dataset_contracts"] is None,
                "confirmation gained dataset contracts",
            )
    else:
        _require(result_protocol.get("dataset_contract") is None, "Stage 3 gained a full contract")
        _require(
            result_protocol.get("dataset_contracts") is None,
            "Stage 3 gained dataset contracts",
        )
        _validate_result_sample_selection(result_protocol.get("sample_selection"), spec)


def _validate_result_sample_selection(value: Any, spec: StageSpec) -> None:
    _require(isinstance(value, dict), "result sample selection is missing")
    _require(spec.manifest is not None, "internal manifest spec is missing")
    expected_keys = {
        "kind",
        "manifest_path",
        "manifest_sha256",
        "format",
        "schema_version",
        "dataset",
        "selection",
        "tasks",
        "excluded_doc_ids",
    }
    _require(set(value) == expected_keys, "result sample-selection keys changed")
    _require(value.get("kind") == "explicit_doc_ids", "sample-selection kind changed")
    manifest_path = value.get("manifest_path")
    _require(
        isinstance(manifest_path, str)
        and Path(manifest_path).name == spec.manifest_path.name,
        "result manifest path changed",
    )
    for key in ("format", "schema_version", "dataset", "selection", "tasks", "excluded_doc_ids"):
        _require(value.get(key) == spec.manifest.get(key), f"sample-selection {key} changed")
    _require(value.get("manifest_sha256") == spec.manifest_sha256, "result manifest SHA changed")


def _raw_generation(row: dict[str, Any], key: PairKey) -> str:
    responses = row.get("resps")
    if (
        not isinstance(responses, list)
        or len(responses) != 1
        or not isinstance(responses[0], list)
        or len(responses[0]) != 1
        or not isinstance(responses[0][0], str)
    ):
        _fail(f"generation sample {key!r} does not have exactly one raw response")
    return responses[0][0]


def _read_sample_rows(
    sample_path: Path,
    artifact: dict[str, Any],
    leaf: LeafSpec,
    spec: StageSpec,
) -> tuple[
    dict[PairKey, dict[str, Any]],
    dict[tuple[str, str], dict[PairKey, bool]],
    dict[tuple[str, str], dict[PairKey, bool]],
    dict[PairKey, bool] | None,
]:
    encoded = sample_path.read_bytes()
    _require(encoded.endswith(b"\n"), f"sample artifact must end in LF: {sample_path}")
    _require(artifact.get("sha256") == _sha256(encoded), "sample artifact SHA mismatch")
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
        _require(isinstance(row, dict), f"sample row {line_number} is not an object")
        rows.append(row)
    _require(
        artifact.get("rows") == len(rows) == len(leaf.expected_doc_ids),
        f"sample row count mismatch for {leaf.task}",
    )
    _require(
        artifact.get("unique_docs") == len(leaf.expected_doc_ids),
        f"sample unique-doc count mismatch for {leaf.task}",
    )
    _require(
        artifact.get("filters") == [leaf.filter_name],
        f"sample filter metadata mismatch for {leaf.task}",
    )

    identities: dict[PairKey, dict[str, Any]] = {}
    correctness = {(leaf.task, metric): {} for metric in leaf.metrics}
    raw_correctness = {(leaf.task, metric): {} for metric in leaf.metrics}
    format_valid: dict[PairKey, bool] | None = (
        {} if spec.format_regex is not None else None
    )
    format_pattern = re.compile(spec.format_regex) if spec.format_regex else None
    observed_ids: list[int] = []
    for row_number, row in enumerate(rows, start=1):
        doc_id = row.get("doc_id")
        _require(type(doc_id) is int, f"sample row {row_number} has non-integer doc_id")
        observed_ids.append(doc_id)
        _require(
            row.get("filter") == leaf.filter_name,
            f"unexpected filter for {leaf.task} doc {doc_id}",
        )
        key = (leaf.task, doc_id, leaf.filter_name)
        _require(key not in identities, f"duplicate sample pairing key {key!r}")
        _require(isinstance(row.get("doc"), dict), f"missing document for {key!r}")
        row_metrics = row.get("metrics")
        _require(
            isinstance(row_metrics, list)
            and len(row_metrics) == len(leaf.metrics)
            and set(row_metrics) == set(leaf.metrics),
            f"sample metric list mismatch for {key!r}",
        )
        identity = {"doc": row["doc"]}
        for hash_key in ("doc_hash", "prompt_hash", "target_hash"):
            value = row.get(hash_key)
            _require(
                isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
                f"missing or invalid {hash_key} for {key!r}",
            )
            identity[hash_key] = value
        identities[key] = identity
        for metric in leaf.metrics:
            raw = _binary_metric(row.get(metric), f"{metric} for {key!r}")
            raw_correctness[(leaf.task, metric)][key] = raw
            correctness[(leaf.task, metric)][key] = raw
        if format_pattern is not None:
            assert format_valid is not None
            valid = format_pattern.search(_raw_generation(row, key)) is not None
            format_valid[key] = valid
            if not valid:
                correctness[(leaf.task, leaf.primary_metric)][key] = False
    _require(
        tuple(observed_ids) == leaf.expected_doc_ids,
        f"sample document IDs or order changed for {leaf.task}",
    )
    return identities, correctness, raw_correctness, format_valid


def _selected_docs_sha256(
    identities: dict[PairKey, dict[str, Any]], leaf: LeafSpec
) -> str:
    records = [
        {
            "doc_id": doc_id,
            "doc_sha256": _sha256(
                _canonical_json(identities[(leaf.task, doc_id, leaf.filter_name)]["doc"])
            ),
        }
        for doc_id in leaf.expected_doc_ids
    ]
    return _sha256(_canonical_json(records))


def _composite_docs_sha256(
    identities: dict[PairKey, dict[str, Any]], leaves: tuple[LeafSpec, ...]
) -> str:
    records = [
        {
            "leaf_task": leaf.task,
            "doc_id": doc_id,
            "doc_sha256": _sha256(
                _canonical_json(identities[(leaf.task, doc_id, leaf.filter_name)]["doc"])
            ),
        }
        for leaf in sorted(leaves, key=lambda item: item.task)
        for doc_id in leaf.expected_doc_ids
    ]
    return _sha256(_canonical_json(records))


def _full_dataset_sha256(
    identities: dict[PairKey, dict[str, Any]], leaf: LeafSpec
) -> str:
    digest = hashlib.sha256()
    for doc_id in leaf.expected_doc_ids:
        key = (leaf.task, doc_id, leaf.filter_name)
        digest.update(_canonical_json({"doc_id": doc_id, "doc": identities[key]["doc"]}))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_sample_document_contract(
    identities: dict[PairKey, dict[str, Any]], spec: StageSpec
) -> None:
    if spec.suite == "stage1_mc":
        for leaf in spec.leaves:
            _require(
                _selected_docs_sha256(identities, leaf) == leaf.selected_docs_sha256,
                f"selected-document SHA mismatch for {leaf.task}",
            )
        _require(
            _composite_docs_sha256(identities, spec.leaves)
            == _STAGE1_COMPOSITE_DOC_SHA256,
            "Stage 1 composite selected-document SHA mismatch",
        )
    elif spec.suite == "stage2_mmlu_pro":
        _require(
            _composite_docs_sha256(identities, spec.leaves)
            == _STAGE2_COMPOSITE_DOC_SHA256,
            "Stage 2 composite selected-document SHA mismatch",
        )
    elif spec.manifest_path is not None:
        leaf = spec.leaves[0]
        _require(
            _selected_docs_sha256(identities, leaf) == leaf.selected_docs_sha256,
            "selected GSM8K document SHA mismatch",
        )
    elif spec.suite == "confirm_mc":
        expected = spec.confirmation_spec.dataset_contract["canonical_jsonl_sha256"]
        _require(
            _full_dataset_sha256(identities, spec.leaves[0]) == expected,
            "full ARC document SHA mismatch",
        )


def _validate_dataset_verification(result: dict[str, Any], spec: StageSpec) -> None:
    if spec.post_confirmation:
        confirmation._validate_dataset_verification(result, spec.confirmation_spec)
        return
    runtime = result.get("runtime")
    verification = runtime.get("dataset_verification") if isinstance(runtime, dict) else None
    _require(isinstance(verification, dict), "result is missing dataset verification")
    if spec.dataset_contracts is not None:
        _require(
            runtime.get("sample_dataset_verification") is None,
            "limited stage unexpectedly used a sample manifest",
        )
        _require(
            verification.get("contract") == spec.dataset_contracts,
            "stage dataset contracts changed",
        )
        pre_run = verification.get("pre_run")
        _require(
            isinstance(pre_run, dict) and set(pre_run) == set(spec.dataset_contracts),
            "stage dataset prechecks are incomplete",
        )
        for task, contract in spec.dataset_contracts.items():
            record = pre_run[task]
            _require(
                isinstance(record, dict)
                and record.get("verified") is True
                and record.get("size") == contract["size"]
                and record.get("datasets_fingerprint") == contract["datasets_fingerprint"]
                and record.get("canonical_jsonl_sha256")
                == contract["canonical_jsonl_sha256"],
                f"dataset precheck failed for {task}",
            )
        logged = verification.get("logged_samples")
        expected_composite = (
            _STAGE1_COMPOSITE_DOC_SHA256
            if spec.suite == "stage1_mc"
            else _STAGE2_COMPOSITE_DOC_SHA256
        )
        expected_ids = tuple(range(spec.limit_count or 0))
        _require(
            isinstance(logged, dict)
            and set(logged)
            == {
                "verified",
                "kind",
                "suite",
                "tasks",
                "composite_selected_docs_sha256",
            }
            and logged.get("verified") is True
            and logged.get("kind") == "stage_selected_docs"
            and logged.get("suite") == spec.suite
            and logged.get("composite_selected_docs_sha256") == expected_composite,
            "limited stage logged-sample verification changed",
        )
        task_records = logged.get("tasks")
        leaf_tasks = {leaf.task for leaf in spec.leaves}
        _require(
            isinstance(task_records, dict) and set(task_records) == leaf_tasks,
            "limited stage logged task set changed",
        )
        for leaf in spec.leaves:
            task_record = task_records[leaf.task]
            _require(
                isinstance(task_record, dict)
                and set(task_record)
                == {
                    "verified",
                    "kind",
                    "size",
                    "ids_sha256",
                    "selected_docs_sha256",
                }
                and task_record.get("verified") is True
                and task_record.get("kind") == "selected_docs"
                and task_record.get("size") == len(expected_ids)
                and task_record.get("ids_sha256") == _ids_sha256(expected_ids)
                and isinstance(task_record.get("selected_docs_sha256"), str)
                and _SHA256_RE.fullmatch(task_record["selected_docs_sha256"])
                is not None,
                f"logged selected-document verification changed for {leaf.task}",
            )
            if leaf.selected_docs_sha256 is not None:
                _require(
                    task_record["selected_docs_sha256"]
                    == leaf.selected_docs_sha256,
                    f"logged selected-document SHA mismatch for {leaf.task}",
                )
        return
    _require(spec.manifest is not None, "Stage 3 manifest is missing")
    _require(verification.get("contract") == _GSM8K_DATASET, "Stage 3 dataset contract changed")
    pre_run = verification.get("pre_run")
    _require(
        isinstance(pre_run, dict)
        and pre_run.get("verified") is True
        and pre_run.get("size") == 1319
        and pre_run.get("datasets_fingerprint") == _GSM8K_DATASET["datasets_fingerprint"]
        and pre_run.get("canonical_jsonl_sha256")
        == _GSM8K_DATASET["canonical_jsonl_sha256"]
        and pre_run.get("selected_docs_sha256") == _STAGE3_SELECTED_DOC_SHA256,
        "Stage 3 dataset precheck failed",
    )
    _require(
        runtime.get("sample_dataset_verification") == pre_run,
        "Stage 3 sample dataset verification changed",
    )
    logged = verification.get("logged_samples")
    leaf = spec.leaves[0]
    task_record = logged.get("tasks", {}).get(leaf.task) if isinstance(logged, dict) else None
    _require(
        isinstance(logged, dict)
        and logged.get("verified") is True
        and isinstance(task_record, dict)
        and task_record.get("verified") is True
        and task_record.get("kind") == "selected_docs"
        and task_record.get("size") == 256
        and task_record.get("ids_sha256") == _STAGE3_IDS_SHA256
        and task_record.get("selected_docs_sha256") == _STAGE3_SELECTED_DOC_SHA256,
        "Stage 3 logged sample verification failed",
    )


_LM_EVAL_BASE_KEYS = {
    "config",
    "configs",
    "date",
    "eot_token_id",
    "git_hash",
    "group_subtasks",
    "higher_is_better",
    "lm_eval_version",
    "max_length",
    "n-samples",
    "n-shot",
    "pretty_env_info",
    "results",
    "tokenizer_bos_token",
    "tokenizer_eos_token",
    "tokenizer_pad_token",
    "transformers_version",
    "upper_git_hash",
    "versions",
}


def _expected_lm_eval_config(spec: StageSpec) -> dict[str, Any]:
    gen_kwargs = None
    if spec.suite == "stage2_mmlu_pro":
        gen_kwargs = {"temperature": 0.0}
    elif spec.format_regex is not None:
        gen_kwargs = {"temperature": 0.0, "max_gen_toks": spec.max_gen_toks}
    return {
        "model": "vllm",
        "model_args": _expected_engine_args(spec),
        "batch_size": "auto",
        "batch_sizes": [],
        "device": None,
        "use_cache": None,
        "limit": spec.limit_count,
        "bootstrap_iters": 0,
        "gen_kwargs": gen_kwargs,
        "random_seed": 0,
        "numpy_seed": 1234,
        "torch_seed": 1234,
        "fewshot_seed": 1234,
    }


def _aggregate_expected_keys(leaf: LeafSpec) -> set[str]:
    keys = {"alias"}
    for metric in leaf.metrics:
        keys.add(f"{metric},{leaf.filter_name}")
        keys.add(f"{metric}_stderr,{leaf.filter_name}")
    return keys


def _expected_aggregate_alias(spec: StageSpec, leaf: LeafSpec) -> str:
    if spec.suite == "stage2_mmlu_pro":
        alias, _ = _mmlu_subject(leaf.task)
        return f" - {alias}"
    return leaf.task


def _validate_lm_eval(
    result: dict[str, Any],
    spec: StageSpec,
    raw_correctness: dict[tuple[str, str], dict[PairKey, bool]],
) -> dict[str, Any]:
    lm_eval = result.get("lm_eval")
    _require(isinstance(lm_eval, dict), "result is missing lm_eval output")
    expected_lm_keys = _LM_EVAL_BASE_KEYS | (
        {"groups"} if spec.suite == "stage2_mmlu_pro" else set()
    )
    _require(set(lm_eval) == expected_lm_keys, "lm_eval output keys changed")
    leaf_tasks = {leaf.task for leaf in spec.leaves}
    group_tasks = {"mmlu_pro"} if spec.suite == "stage2_mmlu_pro" else set()
    leaf_only_maps = ("configs", "n-shot", "n-samples")
    leaf_and_group_maps = ("results", "versions", "higher_is_better")
    for name in leaf_only_maps:
        value = lm_eval.get(name)
        _require(
            isinstance(value, dict) and set(value) == leaf_tasks,
            f"lm_eval {name} task set changed",
        )
    for name in leaf_and_group_maps:
        value = lm_eval.get(name)
        _require(
            isinstance(value, dict) and set(value) == leaf_tasks | group_tasks,
            f"lm_eval {name} task set changed",
        )
    expected_groups = (
        {"mmlu_pro": list(_MMLU_PRO_LEAVES)}
        if spec.suite == "stage2_mmlu_pro"
        else {leaf.task: [] for leaf in spec.leaves}
    )
    _require(lm_eval.get("group_subtasks") == expected_groups, "lm_eval group closure changed")
    if spec.suite == "stage2_mmlu_pro":
        groups = lm_eval.get("groups")
        _require(isinstance(groups, dict) and set(groups) == {"mmlu_pro"}, "MMLU group changed")
    _require(lm_eval.get("config") == _expected_lm_eval_config(spec), "lm_eval run config changed")
    _require(lm_eval.get("max_length") == spec.max_model_len, "lm_eval max length changed")
    _require(lm_eval.get("lm_eval_version") == "0.4.11", "lm-eval version changed")
    _require(
        lm_eval.get("transformers_version") == _RUNTIME_VERSIONS["transformers"],
        "lm_eval transformers version changed",
    )

    normalized_configs: dict[str, Any] = {}
    for leaf in spec.leaves:
        config = lm_eval["configs"][leaf.task]
        _require(isinstance(config, dict), f"invalid task config for {leaf.task}")
        normalized = _normalize_callable_addresses(config)
        expected_config = _expected_task_config(spec, leaf)
        _require(
            set(normalized) == set(expected_config),
            f"lm_eval task config keys changed for {leaf.task}",
        )
        mismatches = {
            key: {"actual": normalized.get(key), "expected": expected}
            for key, expected in expected_config.items()
            if normalized.get(key) != expected
        }
        _require(not mismatches, f"lm_eval task config changed for {leaf.task}: {mismatches}")
        normalized_configs[leaf.task] = normalized

        _require(
            lm_eval["n-shot"][leaf.task] == spec.num_fewshot,
            f"effective fewshot count changed for {leaf.task}",
        )
        expected_higher = {metric: True for metric in leaf.metrics}
        _require(
            lm_eval["higher_is_better"][leaf.task] == expected_higher,
            f"metric direction changed for {leaf.task}",
        )
        counts = lm_eval["n-samples"][leaf.task]
        _require(isinstance(counts, dict), f"sample counts are invalid for {leaf.task}")
        if spec.manifest_path is None:
            _require(
                counts == {
                    "original": leaf.original_size,
                    "effective": len(leaf.expected_doc_ids),
                },
                f"sample counts changed for {leaf.task}",
            )
        else:
            _require(
                counts.get("original") == leaf.original_size
                and counts.get("selected_effective") == len(leaf.expected_doc_ids)
                and counts.get("lm_eval_reported_effective") == counts.get("effective")
                and counts.get("effective") in {leaf.original_size, len(leaf.expected_doc_ids)}
                and set(counts)
                == {
                    "original",
                    "effective",
                    "selected_effective",
                    "lm_eval_reported_effective",
                },
                f"explicit sample counts changed for {leaf.task}",
            )
        aggregates = lm_eval["results"][leaf.task]
        _require(isinstance(aggregates, dict), f"invalid aggregates for {leaf.task}")
        _require(
            set(aggregates) == _aggregate_expected_keys(leaf),
            f"aggregate metric keys changed for {leaf.task}",
        )
        _require(
            aggregates.get("alias") == _expected_aggregate_alias(spec, leaf),
            f"aggregate alias changed for {leaf.task}",
        )
        for metric in leaf.metrics:
            metric_key = f"{metric},{leaf.filter_name}"
            value = aggregates.get(metric_key)
            _require(
                type(value) in {int, float} and math.isfinite(value),
                f"aggregate {metric_key} is missing for {leaf.task}",
            )
            raw_values = raw_correctness[(leaf.task, metric)]
            row_mean = sum(raw_values.values()) / len(raw_values)
            _require(
                math.isclose(float(value), row_mean, rel_tol=0.0, abs_tol=1e-12),
                f"aggregate {metric_key} disagrees with sample rows",
            )
            _require(
                aggregates.get(f"{metric}_stderr,{leaf.filter_name}") == "N/A",
                f"lm-eval bootstrap stderr was unexpectedly enabled for {leaf.task}",
            )
    if spec.suite == "stage2_mmlu_pro":
        group_higher = lm_eval["higher_is_better"]["mmlu_pro"]
        _require(group_higher == {"exact_match": True}, "MMLU group metric direction changed")
        group = lm_eval["results"]["mmlu_pro"]
        _require(isinstance(group, dict), "MMLU group aggregate is invalid")
        group_value = group.get("exact_match,custom-extract")
        all_values = [
            value
            for leaf in spec.leaves
            for value in raw_correctness[(leaf.task, "exact_match")].values()
        ]
        _require(
            type(group_value) in {int, float}
            and math.isfinite(group_value)
            and math.isclose(
                float(group_value), sum(all_values) / len(all_values), rel_tol=0.0, abs_tol=1e-12
            ),
            "MMLU group aggregate disagrees with the equal-size micro average",
        )
    return normalized_configs


def _validate_git_record(value: Any, label: str) -> None:
    _require(isinstance(value, dict), f"{label} provenance is missing")
    _require(
        set(value)
        == {
            "available",
            "root",
            "head",
            "dirty",
            "status_sha256",
            "tracked_diff_sha256",
        },
        f"{label} provenance keys changed",
    )
    _require(value.get("available") is True, f"{label} git provenance is unavailable")
    _require(
        isinstance(value.get("root"), str) and Path(value["root"]).is_absolute(),
        f"{label} git root is invalid",
    )
    _require(
        isinstance(value.get("head"), str)
        and _GIT_HEAD_RE.fullmatch(value["head"]) is not None,
        f"{label} git head is invalid",
    )
    _require(type(value.get("dirty")) is bool, f"{label} dirty flag is invalid")
    for key in ("status_sha256", "tracked_diff_sha256"):
        _require(
            isinstance(value.get(key), str)
            and _SHA256_RE.fullmatch(value[key]) is not None,
            f"{label} {key} is invalid",
        )


def _validate_source_identity(source: Any, arm: str) -> dict[str, Any]:
    _require(
        isinstance(source, dict)
        and set(source) == {"start", "end", "unchanged"}
        and source.get("unchanged") is True
        and source.get("start") == source.get("end"),
        f"source changed during the {arm} run",
    )
    identity = source["start"]
    _require(
        isinstance(identity, dict)
        and set(identity) == {"tensorbridge_git", "tensorbridge_tree", "vllm_git"},
        f"source identity is incomplete for {arm}",
    )
    tree = identity["tensorbridge_tree"]
    _require(
        isinstance(tree, dict)
        and set(tree) == {"sha256", "files"}
        and type(tree.get("files")) is int
        and tree["files"] > 0
        and isinstance(tree.get("sha256"), str)
        and _SHA256_RE.fullmatch(tree["sha256"]) is not None,
        f"TensorBridge tree identity is invalid for {arm}",
    )
    _validate_git_record(identity["tensorbridge_git"], f"TensorBridge ({arm})")
    _validate_git_record(identity["vllm_git"], f"vLLM ({arm})")
    return identity


def _validate_task_sources(runtime: dict[str, Any], spec: StageSpec) -> None:
    expected = _STAGE_TASK_SOURCE_SHA256.get(spec.suite, {})
    if spec.post_confirmation:
        _require(
            "task_sources" not in runtime or runtime.get("task_sources") == {},
            "confirmation task-source metadata changed",
        )
        return
    records = runtime.get("task_sources")
    _require(
        isinstance(records, dict) and set(records) == set(expected),
        "task source set changed",
    )
    for task, expected_sha in expected.items():
        record = records[task]
        _require(
            isinstance(record, dict)
            and set(record) == {"files", "sha256", "verified"}
            and type(record.get("files")) is int
            and record["files"] > 0
            and record.get("sha256") == expected_sha
            and record.get("verified") is True,
            f"task source verification failed for {task}",
        )


def _load_run(
    result_path: Path,
    spec: StageSpec,
    protocol: dict[str, Any],
) -> LoadedRun:
    result, encoded = _load_json_object(result_path, "result")
    _require(
        set(result)
        == {
            "schema_version",
            "experiment",
            "status",
            "arm",
            "model_path",
            "checkpoint",
            "runtime",
            "protocol",
            "production_contract",
            "timing",
            "lm_eval",
            "sample_artifacts",
        },
        f"result top-level keys changed: {result_path}",
    )
    _require(result.get("schema_version") == 1, "unsupported result schema")
    _require(result.get("experiment") == "tensorbridge_nvfp4_lm_harness", "unexpected experiment")
    _require(result.get("status") == "passed", f"result did not pass: {result_path}")
    _require(result.get("model_path") == _MODEL_PATH, "model path changed")
    arm_config = result.get("arm")
    _require(isinstance(arm_config, dict), "result arm config is invalid")
    arm = arm_config.get("key")
    _require(arm in EXPECTED_ARMS, f"unexpected result arm: {arm!r}")
    _require(
        arm_config == {"key": arm} | EXPECTED_ARMS[arm],
        f"precision config mismatch for arm {arm}",
    )

    checkpoint = result.get("checkpoint")
    _require(isinstance(checkpoint, dict), f"checkpoint record is missing for {arm}")
    _require(
        set(checkpoint) == {"start", "end", "unchanged"}
        and checkpoint.get("unchanged") is True
        and checkpoint.get("start") == checkpoint.get("end"),
        f"checkpoint changed during the {arm} run",
    )
    checkpoint_identity = checkpoint["start"]
    _require(isinstance(checkpoint_identity, dict), f"checkpoint identity is invalid for {arm}")
    _require(
        checkpoint_identity.get("checkpoint_content_sha256")
        == protocol["checkpoint"]["content_sha256"]
        and checkpoint_identity.get("source") == _EXPECTED_CHECKPOINT_SOURCE
        and checkpoint_identity.get("expected_checkpoint_verified") is True,
        f"checkpoint identity changed for {arm}",
    )
    for key in ("manifest_sha256", "checkpoint_content_sha256"):
        value = checkpoint_identity.get(key)
        _require(
            isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
            f"checkpoint {key} is invalid for {arm}",
        )
    manifest_path = checkpoint_identity.get("manifest_path")
    _require(
        checkpoint_identity.get("model_path") == _MODEL_PATH
        and isinstance(manifest_path, str)
        and Path(manifest_path).is_absolute(),
        f"checkpoint input paths are invalid for {arm}",
    )

    production = result.get("production_contract")
    _require(
        production
        == {
            "checkpoint_nvfp4_layers": 193,
            "expected_fp8_layers": 208,
            "snc_enabled": EXPECTED_ARMS[arm]["backend"] == "tensorbridge",
            "scale_clamp": False,
            "strict_qwen36_layout": True,
            "lm_head_backend": "marlin_w4a16",
            "fpma_ulp_encoding": (
                "ulp_scale_msb_flag_v1" if arm == "ulp_v1" else None
            ),
        },
        f"production precision contract changed for {arm}",
    )

    runtime = result.get("runtime")
    _require(isinstance(runtime, dict), f"runtime metadata is missing for {arm}")
    _require(
        runtime.get("gpu") == "NVIDIA H100 80GB HBM3"
        and runtime.get("capability") == [9, 0],
        f"unexpected GPU for {arm}",
    )
    _require(
        runtime.get("cpu_thread_limits")
        == {
            "MKL_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
        },
        f"CPU thread limits changed for {arm}",
    )
    configured = runtime.get("configured_environment")
    _require(
        configured in _allowed_environments(arm, spec),
        f"configured environment changed for {arm}",
    )
    _require(
        runtime.get("quant_config_class") == "TensorBridgeModelOptMixedConfig",
        f"vLLM quantization plugin changed for {arm}",
    )
    expected_flags = _ULP_NVRTC_FLAGS if arm == "ulp_v1" else _BASE_NVRTC_FLAGS
    _require(
        runtime.get("tensorbridge_nvrtc_flags") == expected_flags,
        f"effective NVRTC flags changed for {arm}",
    )
    cache_value = runtime.get("tensorbridge_cache_dir")
    _require(
        isinstance(cache_value, str) and cache_value and Path(cache_value).is_absolute(),
        f"TensorBridge cache path is invalid for {arm}",
    )
    cache_dir = Path(cache_value).resolve()
    expected_seed = _CACHE_SEEDS.get(arm)
    seed = runtime.get("tensorbridge_cache_seed")
    if expected_seed is None:
        _require(seed is None, f"unexpected cache seed for {arm}")
    else:
        _require(
            isinstance(seed, dict)
            and set(seed) == {"source", "sha256", "verified"}
            and isinstance(seed.get("source"), str)
            and seed.get("sha256") == expected_seed
            and seed.get("verified") is True,
            f"cache seed provenance changed for {arm}",
        )
    versions = runtime.get("versions")
    _require(versions == _RUNTIME_VERSIONS, f"runtime versions changed for {arm}")
    module_paths = runtime.get("module_paths")
    _require(
        isinstance(module_paths, dict)
        and set(module_paths) == {"lm_eval", "vllm", "tensorbridge_tasks"}
        and all(
            isinstance(path, str) and Path(path).is_absolute()
            for path in module_paths.values()
        ),
        f"runtime module paths are invalid for {arm}",
    )
    _validate_task_sources(runtime, spec)
    source_identity = _validate_source_identity(runtime.get("source"), arm)

    result_protocol = result.get("protocol")
    _require(isinstance(result_protocol, dict), f"result protocol is missing for {arm}")
    _validate_result_protocol(result_protocol, spec)
    _validate_dataset_verification(result, spec)

    artifacts = result.get("sample_artifacts")
    leaf_tasks = {leaf.task for leaf in spec.leaves}
    _require(
        isinstance(artifacts, dict) and set(artifacts) == leaf_tasks,
        f"sample artifact task set changed for {arm}",
    )
    sample_paths: dict[str, Path] = {}
    sample_sha256: dict[str, str] = {}
    identities: dict[PairKey, dict[str, Any]] = {}
    correctness: dict[tuple[str, str], dict[PairKey, bool]] = {}
    raw_correctness: dict[tuple[str, str], dict[PairKey, bool]] = {}
    format_valid: dict[PairKey, bool] | None = {} if spec.format_regex else None
    for leaf in spec.leaves:
        artifact = artifacts[leaf.task]
        _require(
            isinstance(artifact, dict)
            and set(artifact) == {"path", "rows", "unique_docs", "filters", "sha256"}
            and isinstance(artifact.get("path"), str),
            f"sample artifact metadata changed for {arm}/{leaf.task}",
        )
        sample_path = _resolve_existing_path(
            artifact["path"],
            relative_to=(Path.cwd(), REPO_ROOT, result_path.parent),
            label=f"sample artifact for {arm}/{leaf.task}",
        )
        leaf_identities, leaf_correctness, leaf_raw, leaf_format = _read_sample_rows(
            sample_path, artifact, leaf, spec
        )
        _require(not set(identities) & set(leaf_identities), "duplicate cross-task pairing key")
        identities.update(leaf_identities)
        correctness.update(leaf_correctness)
        raw_correctness.update(leaf_raw)
        if format_valid is not None:
            assert leaf_format is not None
            format_valid.update(leaf_format)
        sample_paths[leaf.task] = sample_path
        sample_sha256[leaf.task] = artifact["sha256"]
    _require(
        len(set(sample_paths.values())) == len(sample_paths),
        f"one run reuses sample artifact paths for {arm}",
    )
    _validate_sample_document_contract(identities, spec)
    normalized_configs = _validate_lm_eval(result, spec, raw_correctness)
    return LoadedRun(
        arm=arm,
        result_path=result_path,
        result_sha256=_sha256(encoded),
        sample_paths=sample_paths,
        sample_sha256=sample_sha256,
        identities=identities,
        correctness=correctness,
        raw_correctness=raw_correctness,
        format_valid=format_valid,
        normalized_configs=normalized_configs,
        checkpoint_identity=checkpoint_identity,
        source_identity=source_identity,
        runtime_versions=versions,
        module_paths=module_paths,
        cache_dir=cache_dir,
    )


def _accuracy(values: dict[PairKey, bool], keys: tuple[PairKey, ...]) -> dict[str, Any]:
    correct = sum(values[key] for key in keys)
    return {"correct": correct, "total": len(keys), "accuracy": correct / len(keys)}


def _pair_key_record(key: PairKey) -> dict[str, Any]:
    leaf_task, doc_id, filter_name = key
    return {"leaf_task": leaf_task, "doc_id": doc_id, "filter": filter_name}


def _paired_core(
    baseline: dict[PairKey, bool],
    candidate: dict[PairKey, bool],
    keys: tuple[PairKey, ...],
) -> tuple[dict[str, Any], list[int]]:
    _require(keys and set(baseline) >= set(keys), "baseline pairing keys are incomplete")
    _require(set(candidate) >= set(keys), "candidate pairing keys are incomplete")
    loss_keys = [key for key in keys if baseline[key] and not candidate[key]]
    gain_keys = [key for key in keys if candidate[key] and not baseline[key]]
    both_correct = sum(baseline[key] and candidate[key] for key in keys)
    both_incorrect = len(keys) - len(loss_keys) - len(gain_keys) - both_correct
    deltas = [int(candidate[key]) - int(baseline[key]) for key in keys]
    return (
        {
            "samples": len(keys),
            "baseline_accuracy": sum(baseline[key] for key in keys) / len(keys),
            "candidate_accuracy": sum(candidate[key] for key in keys) / len(keys),
            "candidate_loss_flips": len(loss_keys),
            "candidate_gain_flips": len(gain_keys),
            "candidate_loss_flip_keys": [_pair_key_record(key) for key in loss_keys],
            "candidate_gain_flip_keys": [_pair_key_record(key) for key in gain_keys],
            "concordant_both_correct": both_correct,
            "concordant_both_incorrect": both_incorrect,
            "paired_accuracy_delta": sum(deltas) / len(keys),
            "exact_mcnemar": confirmation._exact_mcnemar(len(loss_keys), len(gain_keys)),
        },
        deltas,
    )


def _paired_stats(
    baseline: dict[PairKey, bool],
    candidate: dict[PairKey, bool],
    keys: tuple[PairKey, ...],
    analysis_protocol: dict[str, Any],
) -> dict[str, Any]:
    stats, deltas = _paired_core(baseline, candidate, keys)
    ci = analysis_protocol["paired_confidence_interval"]
    stats["paired_confidence_interval"] = confirmation._paired_bootstrap_ci(
        deltas,
        confidence_level=ci["confidence_level"],
        resamples=ci["resamples"],
        seed=ci["seed"],
    )
    return stats


def _stratified_bootstrap_ci(
    deltas_by_category: dict[str, list[int]],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    _require(deltas_by_category, "stratified bootstrap has no categories")
    _require(all(values for values in deltas_by_category.values()), "empty bootstrap category")
    rng = random.Random(seed)
    categories = sorted(deltas_by_category)
    total_samples = sum(len(deltas_by_category[category]) for category in categories)
    estimates: list[float] = []
    for _ in range(resamples):
        total = 0
        for category in categories:
            values = deltas_by_category[category]
            for _ in values:
                total += values[rng.randrange(len(values))]
        estimates.append(total / total_samples)
    estimates.sort()
    tail = (1.0 - confidence_level) / 2.0
    return {
        "method": "paired_stratified_nonparametric_bootstrap_percentile",
        "resampling_unit": "document_within_fixed_category",
        "categories": len(categories),
        "quantile_interpolation": "linear_type7",
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
        "lower": confirmation._percentile(estimates, tail),
        "upper": confirmation._percentile(estimates, 1.0 - tail),
    }


def _mmlu_micro_stats(
    baseline: LoadedRun,
    candidate: LoadedRun,
    spec: StageSpec,
    analysis_protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metric = "exact_match"
    all_keys = tuple(key for leaf in spec.leaves for key in leaf.analysis_keys)
    baseline_values: dict[PairKey, bool] = {}
    candidate_values: dict[PairKey, bool] = {}
    categories: dict[str, Any] = {}
    deltas_by_category: dict[str, list[int]] = {}
    for leaf in spec.leaves:
        baseline_leaf = baseline.correctness[(leaf.task, metric)]
        candidate_leaf = candidate.correctness[(leaf.task, metric)]
        baseline_values.update(baseline_leaf)
        candidate_values.update(candidate_leaf)
        category_stats, deltas = _paired_core(
            baseline_leaf, candidate_leaf, leaf.analysis_keys
        )
        ci = analysis_protocol["paired_confidence_interval"]
        category_stats["paired_confidence_interval"] = confirmation._paired_bootstrap_ci(
            deltas,
            confidence_level=ci["confidence_level"],
            resamples=ci["resamples"],
            seed=ci["seed"],
        )
        categories[leaf.task] = category_stats
        deltas_by_category[leaf.task] = deltas
    micro, _ = _paired_core(baseline_values, candidate_values, all_keys)
    ci = analysis_protocol["paired_confidence_interval"]
    micro["paired_confidence_interval"] = _stratified_bootstrap_ci(
        deltas_by_category,
        confidence_level=ci["confidence_level"],
        resamples=ci["resamples"],
        seed=ci["seed"],
    )
    micro["aggregation"] = "equal_64_sample_micro_average_across_14_leaves"
    return micro, categories


def _format_summary(run: LoadedRun, spec: StageSpec) -> dict[str, Any] | None:
    if run.format_valid is None:
        return None
    minimum = spec.format_min_valid
    assert minimum is not None
    valid = sum(run.format_valid.values())
    total = len(run.format_valid)
    return {
        "valid": valid,
        "total": total,
        "valid_rate": valid / total,
        "minimum_valid": minimum,
        "gate_passed": valid >= minimum,
    }


def _comparison_summary(
    *,
    runs: dict[str, LoadedRun],
    arm_summaries: dict[str, Any],
    spec: StageSpec,
    analysis_protocol: dict[str, Any],
    candidate: str,
    baseline: str,
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    primary_deltas: list[dict[str, Any]] = []
    if spec.suite == "stage2_mmlu_pro":
        micro, categories = _mmlu_micro_stats(
            runs[baseline], runs[candidate], spec, analysis_protocol
        )
        tasks["mmlu_pro"] = {
            "primary_metric": "exact_match",
            "micro_average": micro,
            "categories": categories,
        }
        primary_deltas.append(
            {"unit": "mmlu_pro_micro_average", "delta": micro["paired_accuracy_delta"]}
        )
    else:
        for leaf in spec.leaves:
            metric_stats: dict[str, Any] = {}
            for metric in leaf.metrics:
                metric_stats[metric] = _paired_stats(
                    runs[baseline].correctness[(leaf.task, metric)],
                    runs[candidate].correctness[(leaf.task, metric)],
                    leaf.analysis_keys,
                    analysis_protocol,
                )
            tasks[leaf.task] = {
                "primary_metric": leaf.primary_metric,
                "metrics": metric_stats,
            }
            primary_deltas.append(
                {
                    "unit": leaf.task,
                    "delta": metric_stats[leaf.primary_metric]["paired_accuracy_delta"],
                }
            )

    format_gate = None
    if spec.format_regex is not None:
        candidate_format = arm_summaries[candidate]["format"]
        baseline_format = arm_summaries[baseline]["format"]
        format_gate = {
            "minimum_valid": spec.format_min_valid,
            "candidate": {"arm": candidate} | candidate_format,
            "baseline": {"arm": baseline} | baseline_format,
            "both_passed": (
                candidate_format["gate_passed"] and baseline_format["gate_passed"]
            ),
        }
    screening_applies = baseline == "normal_a8" and candidate in {
        "fpma_default",
        "selector_alpha1",
        "ulp_v1",
        "alpha_0960",
    }
    screening = None
    if screening_applies:
        margin = analysis_protocol["screening_margin_vs_normal_a8"]
        point_estimates_passed = all(item["delta"] >= margin for item in primary_deltas)
        format_passed = format_gate is None or format_gate["both_passed"]
        screening = {
            "decision_rule": analysis_protocol["screening_decision"],
            "margin": margin,
            "units": primary_deltas,
            "point_estimate_checks_passed": point_estimates_passed,
            "format_check_passed": format_passed,
            "confidence_intervals_are_reported_but_not_a_gate": True,
            "screening_check_passed": point_estimates_passed and format_passed,
        }
    return {
        "candidate": candidate,
        "baseline": baseline,
        "tasks": tasks,
        "format_gate": format_gate,
        "screening_vs_normal_a8": screening,
    }


def _source_without_status(identity: dict[str, Any]) -> dict[str, Any]:
    tensorbridge_git = identity["tensorbridge_git"]
    return {
        "tensorbridge_tree": identity["tensorbridge_tree"],
        "tensorbridge_git": {
            key: tensorbridge_git[key]
            for key in ("available", "root", "head", "tracked_diff_sha256")
        },
        "vllm_git": identity["vllm_git"],
    }


def _validate_cross_arm_source(runs: dict[str, LoadedRun], spec: StageSpec) -> dict[str, Any]:
    if not spec.post_confirmation:
        reference = runs["normal_a8"].source_identity
        for arm, run in runs.items():
            _require(run.source_identity == reference, f"source identity differs for {arm}")
        return {
            "policy": "all_six_arms_exact_source_identity",
            "prospective_stage_exact_match": True,
        }
    reused = ("official", "normal_a8", "fpma_default", "ulp_v1")
    reused_reference = _source_without_status(runs[reused[0]].source_identity)
    for arm in reused[1:]:
        _require(
            _source_without_status(runs[arm].source_identity) == reused_reference,
            f"reused confirmation source differs for {arm}",
        )
    new_reference = runs["selector_alpha1"].source_identity
    _require(
        runs["alpha_0960"].source_identity == new_reference,
        "new post-confirmation arms used different source identities",
    )
    _require(
        _source_without_status(new_reference) == reused_reference,
        "post-confirmation cohorts used different executable sources",
    )
    return {
        "policy": "post_confirmation_two_cohort_source_identity",
        "reused_v1_arms": list(reused),
        "reused_v1_status_sha256_may_differ": True,
        "reused_v1_tree_head_tracked_diff_and_vllm_match": True,
        "new_arm_exact_match": True,
        "cross_cohort_tree_head_tracked_diff_and_vllm_match": True,
        "cross_cohort_status_sha256_may_differ": True,
    }


def analyze_stages(
    result_paths: list[Path],
    protocol_path: Path = DEFAULT_PROTOCOL,
    confirm_protocol_path: Path = DEFAULT_CONFIRM_PROTOCOL,
) -> dict[str, Any]:
    """Validate one six-arm suite and return JSON-ready paired statistics."""
    _require(len(result_paths) == len(EXPECTED_ARMS), "exactly six result JSON files are required")
    resolved_results = [path.expanduser().resolve() for path in result_paths]
    _require(len(set(resolved_results)) == len(resolved_results), "duplicate result path")
    protocol_path = protocol_path.expanduser().resolve()
    protocol, encoded = _load_json_object(protocol_path, "accuracy expansion protocol")
    _require(
        _sha256(encoded) == EXPECTED_PROTOCOL_SHA256,
        "accuracy expansion protocol raw SHA256 changed",
    )
    _validate_protocol(protocol)

    suites: list[str] = []
    for result_path in resolved_results:
        result, _ = _load_json_object(result_path, "result")
        result_protocol = result.get("protocol")
        _require(
            isinstance(result_protocol, dict)
            and isinstance(result_protocol.get("suite"), str),
            f"cannot determine suite from {result_path}",
        )
        suites.append(result_protocol["suite"])
    _require(len(set(suites)) == 1, "result files belong to different suites")
    spec = _build_spec(
        protocol,
        protocol_path,
        suites[0],
        confirm_protocol_path=confirm_protocol_path,
    )
    analysis_protocol = protocol["analysis"]

    runs: dict[str, LoadedRun] = {}
    all_sample_paths: set[Path] = set()
    cache_dirs: set[Path] = set()
    for result_path in resolved_results:
        run = _load_run(result_path, spec, protocol)
        _require(run.arm not in runs, f"duplicate result arm {run.arm}")
        _require(run.cache_dir not in cache_dirs, "two arms used the same physical cache")
        for sample_path in run.sample_paths.values():
            _require(
                sample_path not in all_sample_paths,
                "two arms point to the same sample artifact",
            )
            all_sample_paths.add(sample_path)
        cache_dirs.add(run.cache_dir)
        runs[run.arm] = run
    _require(set(runs) == set(EXPECTED_ARMS), "the six required arms are not all present")

    reference = runs["normal_a8"]
    for arm, run in runs.items():
        _require(run.identities == reference.identities, f"sample identity differs for {arm}")
        _require(
            run.normalized_configs == reference.normalized_configs,
            f"normalized task configs differ for {arm}",
        )
        _require(
            run.checkpoint_identity == reference.checkpoint_identity,
            f"checkpoint identity differs for {arm}",
        )
        _require(
            run.runtime_versions == reference.runtime_versions,
            f"runtime versions differ for {arm}",
        )
        _require(
            run.module_paths == reference.module_paths,
            f"runtime module paths differ for {arm}",
        )
    source_summary = _validate_cross_arm_source(runs, spec)

    identity_records = [
        _pair_key_record(key) | reference.identities[key]
        for key in sorted(reference.identities)
    ]
    identity_sha = _sha256(_canonical_json(identity_records))
    arm_summaries: dict[str, Any] = {}
    for arm in EXPECTED_ARMS:
        run = runs[arm]
        task_summaries: dict[str, Any] = {}
        for leaf in spec.leaves:
            analysis_metrics = {
                metric: _accuracy(
                    run.correctness[(leaf.task, metric)], leaf.analysis_keys
                )
                for metric in leaf.metrics
            }
            full_metrics = {
                metric: _accuracy(run.correctness[(leaf.task, metric)], leaf.expected_keys)
                for metric in leaf.metrics
            }
            task_summaries[leaf.task] = {
                "primary_metric": leaf.primary_metric,
                "analysis_metrics": analysis_metrics,
                "full_evaluation_metrics": full_metrics,
            }
        summary: dict[str, Any] = {"tasks": task_summaries}
        if spec.suite == "stage2_mmlu_pro":
            total_correct = sum(
                task_summaries[leaf.task]["analysis_metrics"][leaf.primary_metric]["correct"]
                for leaf in spec.leaves
            )
            total = sum(len(leaf.analysis_doc_ids) for leaf in spec.leaves)
            summary["micro_average"] = {
                "metric": "exact_match",
                "correct": total_correct,
                "total": total,
                "accuracy": total_correct / total,
                "aggregation": "equal_64_sample_micro_average_across_14_leaves",
            }
        format_summary = _format_summary(run, spec)
        if format_summary is not None:
            summary["format"] = format_summary
            leaf = spec.leaves[0]
            summary["lm_eval_raw_exact_match"] = _accuracy(
                run.raw_correctness[(leaf.task, leaf.primary_metric)], leaf.analysis_keys
            )
        arm_summaries[arm] = summary

    comparisons = {
        f"{candidate}_vs_{baseline}": _comparison_summary(
            runs=runs,
            arm_summaries=arm_summaries,
            spec=spec,
            analysis_protocol=analysis_protocol,
            candidate=candidate,
            baseline=baseline,
        )
        for candidate, baseline in EXPECTED_COMPARISONS
    }
    screening_candidates = ("fpma_default", "selector_alpha1", "ulp_v1", "alpha_0960")
    screening = {
        candidate: comparisons[f"{candidate}_vs_normal_a8"]["screening_vs_normal_a8"][
            "screening_check_passed"
        ]
        for candidate in screening_candidates
    }
    inputs = {
        arm: {
            "result_path": str(runs[arm].result_path),
            "result_sha256": runs[arm].result_sha256,
            "checkpoint": {
                "model_path": runs[arm].checkpoint_identity["model_path"],
                "manifest_path": runs[arm].checkpoint_identity["manifest_path"],
                "manifest_sha256": runs[arm].checkpoint_identity["manifest_sha256"],
                "content_sha256": runs[arm].checkpoint_identity[
                    "checkpoint_content_sha256"
                ],
            },
            "sample_artifacts": {
                task: {
                    "path": str(runs[arm].sample_paths[task]),
                    "sha256": runs[arm].sample_sha256[task],
                }
                for task in sorted(runs[arm].sample_paths)
            },
        }
        for arm in EXPECTED_ARMS
    }
    supporting_files: dict[str, Any] = {}
    if spec.manifest_path is not None:
        supporting_files["sample_manifest"] = {
            "path": str(spec.manifest_path),
            "sha256": spec.manifest_sha256,
        }
    if spec.post_confirmation:
        supporting_files["confirmation_v1_protocol"] = {
            "path": str(confirm_protocol_path.expanduser().resolve()),
            "sha256": EXPECTED_CONFIRM_PROTOCOL_SHA256,
        }
    return {
        "schema_version": 1,
        "format": OUTPUT_FORMAT,
        "status": "analysis_completed",
        "scientific_decision": {
            "suite": spec.suite,
            "prospective_stage": not spec.post_confirmation,
            "post_confirmation_sensitivity": spec.post_confirmation,
            "screening_checks_vs_normal_a8": screening,
            "all_screening_checks_passed": all(screening.values()),
            "confidence_intervals_are_reported_but_not_a_gate": True,
            "no_result_dependent_arm_dropping_authorized": True,
        },
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(encoded),
            "suite": spec.suite,
            "requested_tasks": list(spec.requested_tasks),
            "leaf_tasks": [leaf.task for leaf in spec.leaves],
            "pairing_key": analysis_protocol["pairing_key"],
            "screening_margin_vs_normal_a8": analysis_protocol[
                "screening_margin_vs_normal_a8"
            ],
            "paired_confidence_interval": analysis_protocol[
                "paired_confidence_interval"
            ],
            "mcnemar": analysis_protocol["mcnemar"],
            "supporting_files": supporting_files,
        },
        "inputs": inputs,
        "sample_pairing": {
            "pairs": len(reference.identities),
            "cross_arm_identity_sha256": identity_sha,
            "cross_arm_sample_match": True,
            "cross_arm_task_config_match_after_callable_address_normalization": True,
            "cross_arm_checkpoint_match": True,
            "cross_arm_runtime_versions_match": True,
            "cross_arm_module_paths_match": True,
            "source": source_summary,
        },
        "arms": arm_summaries,
        "comparisons": comparisons,
    }


def _write_output(
    path: Path,
    encoded: str,
    overwrite: bool,
    protected: set[Path],
    protected_roots: set[Path] | None = None,
) -> None:
    path = path.expanduser().resolve()
    _require(path not in protected, "output path would overwrite an input, protocol, or manifest")
    for root in protected_roots or set():
        root = root.expanduser().resolve()
        _require(
            path != root and root not in path.parents,
            "output path would overwrite the checkpoint model tree",
        )
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
    parser.add_argument("result_json", nargs="+", type=Path, help="the six arm result JSON files")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--confirm-protocol", type=Path, default=DEFAULT_CONFIRM_PROTOCOL)
    parser.add_argument("--output", type=Path, help="also write the JSON report to this path")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = analyze_stages(
        args.result_json,
        protocol_path=args.protocol,
        confirm_protocol_path=args.confirm_protocol,
    )
    encoded = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    if args.output is not None:
        protected = {path.expanduser().resolve() for path in args.result_json}
        protected_roots: set[Path] = set()
        protected.add(args.protocol.expanduser().resolve())
        if report["protocol"]["suite"].startswith("confirm_"):
            protected.add(args.confirm_protocol.expanduser().resolve())
        for arm in report["inputs"].values():
            protected.add(Path(arm["checkpoint"]["manifest_path"]).resolve())
            protected_roots.add(Path(arm["checkpoint"]["model_path"]).resolve())
            for artifact in arm["sample_artifacts"].values():
                protected.add(Path(artifact["path"]).resolve())
        for supporting in report["protocol"]["supporting_files"].values():
            protected.add(Path(supporting["path"]).resolve())
        _write_output(
            args.output,
            encoded,
            args.overwrite,
            protected,
            protected_roots,
        )
    print(encoded, end="")


if __name__ == "__main__":
    main()
