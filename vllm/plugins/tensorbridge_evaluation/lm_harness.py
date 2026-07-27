"""Fail-closed lm-evaluation-harness runner helpers for TensorBridge NVFP4."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import wraps
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any, Callable


@dataclass(frozen=True)
class ArmConfig:
    key: str
    label: str
    backend: str
    alpha: float
    selector: str
    ulp_correction: bool


@dataclass(frozen=True)
class SuiteConfig:
    key: str
    tasks: tuple[str, ...]
    num_fewshot: int
    apply_chat_template: bool
    enable_thinking: bool
    max_gen_toks: int
    generation: bool
    default_limit_count: int | None = None
    min_model_len: int = 4096
    fewshot_as_multiturn: bool = False
    system_instruction: str | None = None
    sample_manifest: Path | None = None
    sample_manifest_sha256: str | None = None
    analysis_exclude_doc_ids: tuple[int, ...] = ()
    dataset_contract: dict[str, Any] | None = None
    dataset_contracts: dict[str, dict[str, Any]] | None = None


@dataclass(frozen=True)
class RunConfig:
    model: Path
    checkpoint_manifest: Path
    arm: str
    suite: str
    output_json: Path
    samples_dir: Path
    sample_manifest: Path | None = None
    limit_count: int | None = None
    limit_fraction: float | None = None
    num_fewshot: int | None = None
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.5
    max_num_seqs: int = 8
    batch_size: str | int = "auto"
    bootstrap_iters: int | None = None
    enable_thinking: bool | None = None
    think_end_token: str | None = None
    max_gen_toks: int | None = None
    allow_runtime_version_mismatch: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class RuntimeBindings:
    simple_evaluate: Callable[..., dict[str, Any] | None]
    json_default: Callable[[Any], Any]
    versions: dict[str, str]
    module_paths: dict[str, str]
    quant_config_class: str
    gpu: str | None
    capability: tuple[int, int] | None
    task_manager: Any | None = None


ARMS = {
    arm.key: arm
    for arm in (
        ArmConfig("official", "official_marlin_w4a16", "official", 1.0, "none", False),
        ArmConfig("normal_a8", "normal_a8_cutlass", "normal_a8", 1.0, "none", False),
        ArmConfig(
            "fpma_default",
            "tensorbridge_default_fpma_snc",
            "tensorbridge",
            1.0,
            "none",
            False,
        ),
        ArmConfig(
            "selector_alpha1",
            "tensorbridge_selector_alpha_1",
            "tensorbridge",
            1.0,
            "normal_b8_sse",
            False,
        ),
        ArmConfig(
            "ulp_v1",
            "tensorbridge_ulp_scale_msb_flag_v1",
            "tensorbridge",
            1.0,
            "none",
            True,
        ),
        ArmConfig(
            "alpha_0960",
            "tensorbridge_global_alpha_0_960",
            "tensorbridge",
            0.960,
            "none",
            False,
        ),
    )
}


_GSM8K_FINAL_ANSWER_INSTRUCTION = (
    "Use no more than six short sentences or equations. End with a separate final "
    "line in the exact form The answer is N. Replace N with the numeric answer only, "
    "omit units, and write nothing after that line."
)
_SAMPLE_ROOT = Path(__file__).with_name("samples")
_PROTOCOL_ROOT = Path(__file__).with_name("protocols")
_STAGE_PROTOCOL_PATH = _PROTOCOL_ROOT / "accuracy_expand_v2.json"
_STAGE_PROTOCOL_SHA256 = (
    "c56962917b63290a879832b562421f8ae938820d387d6be431b73fded7085a08"
)
_CONFIRM_SAMPLE_MANIFEST_SHA256 = (
    "37bca36cf4be344ed07209b2caa76688969ac20664f3ad4e263409296455f2df"
)
_STAGE3_SAMPLE_MANIFEST_SHA256 = (
    "37077471979c08a0b71ecdb384c0d4b5c7d9e37f59225ba724bb0ba3693915e4"
)
_ARC_CHALLENGE_TEST_CONTRACT = {
    "path": "allenai/ai2_arc",
    "name": "ARC-Challenge",
    "split": "test",
    "size": 1172,
    "revision": "210d026faf9955653af8916fad021475a3f00453",
    "datasets_fingerprint": "a4361c3f3e560fcd",
    "canonicalization": "sorted_minified_utf8_jsonl_with_doc_id_v1",
    "canonical_jsonl_sha256": (
        "6f763973e8abb704484d01687eb7b193dd6ec7a5e9399bacb71e33d4eb07a0f2"
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
_EXPECTED_STAGE_TASK_SOURCES = {
    "stage1_mc": {
        "hellaswag": "652027d438d4bc2f23b3525f7e3f255eae0be75cff322d0b7f28bfebf28484c0",
        "winogrande": "abf47611da8131be4ea5450e1189203de902c8c6d00593f45bf59e62f03b1509",
    },
    "stage2_mmlu_pro": {
        "mmlu_pro": "7cff51577642fb1ae79a3c7991ce773f2505afb6fc8e7802988c94ad094d10ed",
    },
}
_EXPECTED_LOCAL_STAGE_TASK_SOURCES = {
    "stage3_generation": {
        "tensorbridge_gsm8k_relative_smoke": {
            "relative_path": "tensorbridge_gsm8k_relative_smoke.yaml",
            "sha256": "47c193604c56a717778641320d09ed49cab619f00406ce3588cc235c447a36a5",
        }
    }
}
_EXPECTED_STAGE_CHECKPOINT_CONTENT_SHA256 = (
    "4ec0960247ca03fd10a9883d20de08d3795760ac1043fe7a9db6151b4074203f"
)
_EXPECTED_STAGE_RUNTIME_VERSIONS = {
    "lm_eval": "0.4.11",
    "vllm": "0.20.2+cu128",
    "transformers": "5.9.0",
    "datasets": "4.8.5",
    "torch": "2.11.0+cu128",
}
_STAGE1_SELECTED_DOC_SHA256 = {
    "hellaswag": "39af1ea86866600455e0543d95601d1e158b0b42c384667ddb9dd3346e09024a",
    "winogrande": "2745f5df477b490dc9f3b1d02b4a1a0eac4a9c7794be959a44451835f6ebdd33",
}
_STAGE1_COMPOSITE_SELECTED_DOC_SHA256 = (
    "b8f345a9030494ab72895765d51cc312f0600043fc6f0eac489e516e451c310a"
)
_STAGE2_LEAF_TASKS = (
    "mmlu_pro_biology",
    "mmlu_pro_business",
    "mmlu_pro_chemistry",
    "mmlu_pro_computer_science",
    "mmlu_pro_economics",
    "mmlu_pro_engineering",
    "mmlu_pro_health",
    "mmlu_pro_history",
    "mmlu_pro_law",
    "mmlu_pro_math",
    "mmlu_pro_other",
    "mmlu_pro_philosophy",
    "mmlu_pro_physics",
    "mmlu_pro_psychology",
)
_STAGE2_COMPOSITE_SELECTED_DOC_SHA256 = (
    "8fc7f27b644625b3d0d6efa9b2d83523b7f7ff33c8cb91616acd266a32dda312"
)


SUITES = {
    suite.key: suite
    for suite in (
        SuiteConfig("smoke_mc", ("arc_challenge",), 0, False, False, 256, False, 16),
        SuiteConfig(
            "smoke_generation",
            ("tensorbridge_gsm8k_relative_smoke",),
            0,
            True,
            False,
            1024,
            True,
            16,
            system_instruction=_GSM8K_FINAL_ANSWER_INSTRUCTION,
        ),
        SuiteConfig(
            "confirm_mc",
            ("tensorbridge_arc_challenge_confirm",),
            0,
            False,
            False,
            256,
            False,
            analysis_exclude_doc_ids=tuple(range(16)),
            dataset_contract=_ARC_CHALLENGE_TEST_CONTRACT,
        ),
        SuiteConfig(
            "confirm_generation",
            ("tensorbridge_gsm8k_relative_smoke",),
            0,
            True,
            False,
            1024,
            True,
            system_instruction=_GSM8K_FINAL_ANSWER_INSTRUCTION,
            sample_manifest=_SAMPLE_ROOT / "gsm8k_test_sha256_rank_128.json",
            sample_manifest_sha256=_CONFIRM_SAMPLE_MANIFEST_SHA256,
        ),
        SuiteConfig(
            "mc_core",
            ("arc_challenge", "hellaswag"),
            0,
            False,
            False,
            256,
            False,
        ),
        SuiteConfig(
            "mmlu_pro",
            ("mmlu_pro",),
            5,
            True,
            False,
            2048,
            True,
            min_model_len=16384,
        ),
        SuiteConfig(
            "generation_core",
            ("tensorbridge_gsm8k_relative_smoke",),
            0,
            True,
            False,
            1024,
            True,
            system_instruction=_GSM8K_FINAL_ANSWER_INSTRUCTION,
        ),
        SuiteConfig(
            "stage1_mc",
            ("hellaswag", "winogrande"),
            0,
            False,
            False,
            256,
            False,
            512,
            dataset_contracts=_STAGE1_DATASET_CONTRACTS,
        ),
        SuiteConfig(
            "stage2_mmlu_pro",
            ("mmlu_pro",),
            5,
            True,
            False,
            2048,
            True,
            64,
            min_model_len=16384,
            dataset_contracts=_STAGE2_DATASET_CONTRACTS,
        ),
        SuiteConfig(
            "stage3_generation",
            ("tensorbridge_gsm8k_relative_smoke",),
            0,
            True,
            False,
            1024,
            True,
            system_instruction=_GSM8K_FINAL_ANSWER_INSTRUCTION,
            sample_manifest=_SAMPLE_ROOT / "gsm8k_test_stage3_sha256_rank_256.json",
            sample_manifest_sha256=_STAGE3_SAMPLE_MANIFEST_SHA256,
        ),
    )
}


_PRECISION_ENV_KEYS = {
    "TENSORBRIDGE_VLLM_BACKEND",
    "TENSORBRIDGE_NVFP4_FPMA_ALPHA",
    "TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR",
    "TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR_CHUNK_ROWS",
    "TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION",
    "TENSORBRIDGE_NVFP4_ALLOW_SCALE_CLAMP",
    "TENSORBRIDGE_STRICT_QWEN36_LAYOUT",
    "TENSORBRIDGE_COMPILER",
    "TENSORBRIDGE_EXTRA_NVRTC_FLAGS",
    "VLLM_PLUGINS",
}


CHECKPOINT_MANIFEST_FORMAT = "tensorbridge_checkpoint_sha256_manifest"
CHECKPOINT_METADATA_FILES = (
    "config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "chat_template.jinja",
    "generation_config.json",
)


EXPECTED_CHECKPOINT_SOURCE = {
    "provider": "huggingface",
    "repo_id": "nvidia/Qwen3.6-27B-NVFP4",
    "revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
    "revision_kind": "git_commit",
}
EXPECTED_WEIGHT_SHA256 = {
    "model-00001-of-00003.safetensors": (
        "b4a0d9a57ff1859dac1144b53ca285011db072737d8813fc16d8d1e07ecae17d"
    ),
    "model-00002-of-00003.safetensors": (
        "06da4242b0f491118d19d4d4c7564307a7bd6059c6bed284e08c93f6fc5a556d"
    ),
    "model-00003-of-00003.safetensors": (
        "e90f5b2bb16814a0565de284ea179edec201edfb120d13f1debaab66f9e60845"
    ),
}
EXPECTED_METADATA_SHA256 = {
    "chat_template.jinja": (
        "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
    ),
    "config.json": (
        "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338"
    ),
    "generation_config.json": (
        "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
    ),
    "hf_quant_config.json": (
        "fd7200cd8bca2a8a5d777061521abf83e2deb97ab6bc2f04e7a0a3d3f8ecd5c1"
    ),
    "model.safetensors.index.json": (
        "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2"
    ),
    "tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "tokenizer_config.json": (
        "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b"
    ),
    "vocab.json": (
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
    ),
}


_EVALUATION_ENV_KEYS = _PRECISION_ENV_KEYS | {
    "TENSORBRIDGE_NORMAL_A8_CHUNK_ROWS",
    "TENSORBRIDGE_NVFP4_CPP_ROUTER",
    "TENSORBRIDGE_NVFP4_CPP_ROUTER_STRICT",
    "TENSORBRIDGE_NVFP4_SWZ64_DUAL_MMA_PREINT_LAYOUT",
    "TENSORBRIDGE_CACHE_DIR",
    "TENSORBRIDGE_CACHE_SEED_SOURCE",
    "TENSORBRIDGE_CACHE_SEED_SHA256",
    "TENSORBRIDGE_CACHE_SEED_VERIFIED",
    "VLLM_NVFP4_GEMM_BACKEND",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "TOKENIZERS_PARALLELISM",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "TENSORBRIDGE_DISABLE_PARALLEL_BUILD",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "MALLOC_ARENA_MAX",
}


def resolve_arm(key: str) -> ArmConfig:
    try:
        return ARMS[key]
    except KeyError as error:
        raise ValueError(f"unknown lm-eval arm: {key}") from error


def resolve_suite(key: str) -> SuiteConfig:
    try:
        return SUITES[key]
    except KeyError as error:
        raise ValueError(f"unknown lm-eval suite: {key}") from error


def resolve_limit(
    suite: SuiteConfig,
    limit_count: int | None,
    limit_fraction: float | None,
) -> tuple[int | float | None, dict[str, Any]]:
    if limit_count is not None and limit_fraction is not None:
        raise ValueError("limit_count and limit_fraction are mutually exclusive")
    if limit_count is not None:
        if limit_count <= 0:
            raise ValueError("limit_count must be positive")
        return limit_count, {"kind": "count", "value": limit_count}
    if limit_fraction is not None:
        if not 0.0 < limit_fraction < 1.0:
            raise ValueError("limit_fraction must be strictly between 0 and 1")
        return limit_fraction, {"kind": "fraction", "value": limit_fraction}
    if suite.default_limit_count is not None:
        return suite.default_limit_count, {
            "kind": "count",
            "value": suite.default_limit_count,
            "from_suite_default": True,
        }
    return None, {"kind": "none", "value": None}


SAMPLE_MANIFEST_FORMAT = "tensorbridge_lm_eval_sample_manifest"
EXPECTED_SAMPLE_MANIFEST_SHA256 = (
    _CONFIRM_SAMPLE_MANIFEST_SHA256
)
EXPECTED_SAMPLE_DATASET = {
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
EXPECTED_SAMPLE_SELECTION = {
    "algorithm": "sha256_rank_v1",
    "candidate_start": 16,
    "namespace": "tensorbridge-gsm8k-confirm-v1",
    "count": 128,
    "ids_sha256": "a43574fc29a99293c793b08c17a02b720cd4f9487e9fd33ef299515903924fc2",
    "selected_docs_sha256": (
        "4cba37cf8fc51ba80de102927527591be096d2f4e1c0694ea6f7ddefd5c13c78"
    ),
}


def _ids_sha256(ids: list[int]) -> str:
    encoded = json.dumps(ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_ranked_ids(
    namespace: str, split_size: int, count: int, candidate_start: int
) -> list[int]:
    return _sha256_ranked_ids_excluding(
        namespace,
        split_size,
        count,
        candidate_start,
        excluded_doc_ids=(),
    )


def _sha256_ranked_ids_excluding(
    namespace: str,
    split_size: int,
    count: int,
    candidate_start: int,
    excluded_doc_ids: tuple[int, ...] | list[int],
) -> list[int]:
    excluded = set(excluded_doc_ids)
    ranked = sorted(
        (doc_id for doc_id in range(candidate_start, split_size) if doc_id not in excluded),
        key=lambda doc_id: (
            hashlib.sha256(f"{namespace}:{doc_id}".encode("utf-8")).digest(),
            doc_id,
        ),
    )
    if count > len(ranked):
        raise ValueError("sample count exceeds the remaining candidate documents")
    # lm-eval 0.4.11 mislabels documents if explicit sample IDs are not ascending.
    return sorted(ranked[:count])


def _canonical_doc_bytes(doc: dict[str, Any]) -> bytes:
    return json.dumps(
        dict(doc), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _selected_docs_sha256(ids: list[int], docs: dict[int, dict[str, Any]]) -> str:
    records = [
        {
            "doc_id": doc_id,
            "doc_sha256": hashlib.sha256(_canonical_doc_bytes(docs[doc_id])).hexdigest(),
        }
        for doc_id in ids
    ]
    encoded = json.dumps(
        records, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_dataset(
    path: str, name: str | None, split: str, revision: str | None
):
    from datasets import load_dataset

    return load_dataset(path, name, split=split, revision=revision)


def _full_dataset_sha256(dataset: Any) -> str:
    digest = hashlib.sha256()
    for doc_id, doc in enumerate(dataset):
        canonical = json.dumps(
            {"doc_id": doc_id, "doc": dict(doc)},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(canonical)
        digest.update(b"\n")
    return digest.hexdigest()


def _verify_loaded_dataset_contract(
    contract: dict[str, Any], dataset: Any
) -> dict[str, Any]:
    if contract.get("canonicalization") != "sorted_minified_utf8_jsonl_with_doc_id_v1":
        raise ValueError("unsupported dataset canonicalization")
    if len(dataset) != contract["size"]:
        raise ValueError(f"dataset size mismatch: {len(dataset)} vs {contract['size']}")
    fingerprint = getattr(dataset, "_fingerprint", None)
    if fingerprint != contract["datasets_fingerprint"]:
        raise ValueError(
            "dataset fingerprint mismatch: "
            f"{fingerprint!r} vs {contract['datasets_fingerprint']!r}"
        )
    full_digest = _full_dataset_sha256(dataset)
    if full_digest != contract["canonical_jsonl_sha256"]:
        raise ValueError("dataset canonical JSONL SHA256 mismatch")
    return {
        "verified": True,
        "size": len(dataset),
        "datasets_fingerprint": fingerprint,
        "canonical_jsonl_sha256": full_digest,
    }


def verify_dataset_contract(
    contract: dict[str, Any],
    dataset_loader: Callable[[str, str | None, str, str | None], Any] = _load_dataset,
) -> dict[str, Any]:
    dataset = dataset_loader(
        contract["path"], contract["name"], contract["split"], contract["revision"]
    )
    return _verify_loaded_dataset_contract(contract, dataset)


def verify_sample_dataset(
    sample_selection: dict[str, Any],
    dataset_loader: Callable[[str, str | None, str, str | None], Any] = _load_dataset,
) -> dict[str, Any]:
    contract = sample_selection["dataset"]
    dataset = dataset_loader(
        contract["path"], contract["name"], contract["split"], contract["revision"]
    )
    verified = _verify_loaded_dataset_contract(contract, dataset)
    selected_ids = next(iter(sample_selection["tasks"].values()))
    selected_set = set(selected_ids)
    selected_docs = {
        doc_id: dict(dataset[doc_id])
        for doc_id in selected_ids
        if 0 <= doc_id < len(dataset)
    }
    if set(selected_docs) != selected_set:
        raise ValueError("sample dataset is missing selected document IDs")
    selected_digest = _selected_docs_sha256(selected_ids, selected_docs)
    if selected_digest != sample_selection["selection"]["selected_docs_sha256"]:
        raise ValueError("selected sample document SHA256 mismatch")
    return verified | {"selected_docs_sha256": selected_digest}


def load_sample_manifest(
    path: Path,
    requested_tasks: tuple[str, ...] | list[str],
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    path = path.resolve()
    encoded = path.read_bytes()
    manifest_sha256 = hashlib.sha256(encoded).hexdigest()
    expected_sha256 = expected_sha256 or EXPECTED_SAMPLE_MANIFEST_SHA256
    if manifest_sha256 != expected_sha256:
        raise ValueError(
            "sample manifest raw SHA256 mismatch: "
            f"{manifest_sha256} vs {expected_sha256}"
        )
    manifest = json.loads(encoded)
    if not isinstance(manifest, dict):
        raise ValueError("sample manifest must be a JSON object")
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("sample manifest selection must be an object")
    algorithm = selection.get("algorithm")
    expected_keys = {"schema_version", "format", "dataset", "selection", "tasks"}
    if algorithm == "sha256_rank_excluding_ids_v1":
        expected_keys.add("excluded_doc_ids")
    if set(manifest) != expected_keys:
        raise ValueError(
            "sample manifest keys mismatch: "
            f"actual={sorted(manifest)}, expected={sorted(expected_keys)}"
        )
    if manifest["schema_version"] != 1 or manifest["format"] != SAMPLE_MANIFEST_FORMAT:
        raise ValueError("unsupported sample manifest schema or format")

    dataset = manifest["dataset"]
    tasks = manifest["tasks"]
    if expected_sha256 == EXPECTED_SAMPLE_MANIFEST_SHA256:
        if dataset != EXPECTED_SAMPLE_DATASET:
            raise ValueError("sample manifest dataset contract is invalid")
        if selection != EXPECTED_SAMPLE_SELECTION:
            raise ValueError("sample manifest selection contract is invalid")
    if not isinstance(dataset, dict):
        raise ValueError("sample manifest dataset contract must be an object")
    dataset_keys = {
        "path",
        "name",
        "split",
        "size",
        "revision",
        "datasets_fingerprint",
        "canonicalization",
        "canonical_jsonl_sha256",
    }
    if set(dataset) != dataset_keys:
        raise ValueError("sample manifest dataset contract keys changed")
    if dataset.get("canonicalization") != "sorted_minified_utf8_jsonl_with_doc_id_v1":
        raise ValueError("sample manifest canonicalization is invalid")
    if not isinstance(tasks, dict) or set(tasks) != set(requested_tasks):
        raise ValueError("sample manifest task set does not match requested tasks")

    split_size = dataset["size"]
    count = selection["count"]
    candidate_start = selection["candidate_start"]
    namespace = selection["namespace"]
    if type(split_size) is not int or split_size <= 0:
        raise ValueError("sample manifest split size must be a positive integer")
    if type(candidate_start) is not int or not 0 <= candidate_start < split_size:
        raise ValueError("sample manifest candidate_start is out of range")
    if type(count) is not int or count <= 0:
        raise ValueError("sample manifest count is out of range")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("sample manifest selection algorithm is invalid")

    excluded_doc_ids: list[int] = []
    if algorithm == "sha256_rank_v1":
        selection_keys = {
            "algorithm",
            "candidate_start",
            "namespace",
            "count",
            "ids_sha256",
            "selected_docs_sha256",
        }
    elif algorithm == "sha256_rank_excluding_ids_v1":
        selection_keys = {
            "algorithm",
            "candidate_start",
            "namespace",
            "count",
            "excluded_ids_sha256",
            "ids_sha256",
            "selected_docs_sha256",
        }
        excluded_doc_ids = manifest["excluded_doc_ids"]
        if (
            not isinstance(excluded_doc_ids, list)
            or any(type(doc_id) is not int for doc_id in excluded_doc_ids)
            or excluded_doc_ids != sorted(set(excluded_doc_ids))
            or any(not 0 <= doc_id < split_size for doc_id in excluded_doc_ids)
        ):
            raise ValueError("sample manifest excluded document IDs are invalid")
        if selection.get("excluded_ids_sha256") != _ids_sha256(excluded_doc_ids):
            raise ValueError("sample manifest excluded document ID hash mismatch")
    else:
        raise ValueError("sample manifest selection algorithm is invalid")
    if set(selection) != selection_keys:
        raise ValueError("sample manifest selection contract keys changed")

    expected_ids = _sha256_ranked_ids_excluding(
        namespace,
        split_size,
        count,
        candidate_start,
        excluded_doc_ids,
    )
    expected_digest = _ids_sha256(expected_ids)
    if selection["ids_sha256"] != expected_digest:
        raise ValueError("sample manifest ids_sha256 mismatch")
    resolved: dict[str, list[int]] = {}
    for task, ids in tasks.items():
        if not isinstance(ids, list) or any(type(doc_id) is not int for doc_id in ids):
            raise ValueError(f"sample IDs for {task} must be an integer list")
        if ids != expected_ids:
            raise ValueError(f"sample IDs for {task} do not match {algorithm}")
        resolved[task] = list(ids)

    metadata = {
        "kind": "explicit_doc_ids",
        "manifest_path": str(path),
        "manifest_sha256": manifest_sha256,
        "format": manifest["format"],
        "schema_version": manifest["schema_version"],
        "dataset": dataset,
        "selection": selection,
        "tasks": tasks,
    }
    if excluded_doc_ids:
        metadata["excluded_doc_ids"] = excluded_doc_ids
    return resolved, metadata


def resolve_protocol(config: RunConfig) -> dict[str, Any]:
    suite = resolve_suite(config.suite)
    tasks = suite.tasks
    if not tasks or any(not task.strip() for task in tasks):
        raise ValueError("at least one non-empty task is required")
    if suite.key.startswith("confirm_") or suite.key.startswith("stage"):
        immutable_overrides = {
            "num_fewshot": (config.num_fewshot, suite.num_fewshot),
            "max_gen_toks": (config.max_gen_toks, suite.max_gen_toks),
            "enable_thinking": (config.enable_thinking, suite.enable_thinking),
            "bootstrap_iters": (config.bootstrap_iters, 0),
        }
        conflicts = {
            key: actual
            for key, (actual, expected) in immutable_overrides.items()
            if actual is not None and actual != expected
        }
        if suite.key.startswith("stage"):
            immutable_overrides.update(
                {
                    "max_model_len": (config.max_model_len, suite.min_model_len),
                    "gpu_memory_utilization": (
                        config.gpu_memory_utilization,
                        0.5,
                    ),
                    "max_num_seqs": (config.max_num_seqs, 8),
                    "batch_size": (config.batch_size, "auto"),
                    "allow_runtime_version_mismatch": (
                        config.allow_runtime_version_mismatch,
                        False,
                    ),
                }
            )
            conflicts = {
                key: actual
                for key, (actual, expected) in immutable_overrides.items()
                if actual is not None and actual != expected
            }
            if config.limit_fraction is not None:
                conflicts["limit_fraction"] = config.limit_fraction
            if config.limit_count not in (None, suite.default_limit_count):
                conflicts["limit_count"] = config.limit_count
            if config.sample_manifest is not None and (
                suite.sample_manifest is None
                or config.sample_manifest.resolve() != suite.sample_manifest.resolve()
            ):
                conflicts["sample_manifest"] = str(config.sample_manifest)
        if conflicts or config.think_end_token is not None:
            contract = (
                "confirmation protocol" if suite.key.startswith("confirm_") else "stage"
            )
            raise ValueError(
                f"{contract} overrides are not allowed: "
                f"{conflicts}, think_end_token={config.think_end_token!r}"
            )

    apply_chat_template = suite.apply_chat_template
    enable_thinking = suite.enable_thinking
    if config.enable_thinking is not None:
        enable_thinking = config.enable_thinking
    if enable_thinking and not suite.generation:
        raise ValueError("thinking is only valid for generation suites")
    if enable_thinking and not config.think_end_token:
        raise ValueError("thinking requires think_end_token")
    if not apply_chat_template and enable_thinking:
        raise ValueError("thinking requires a chat template")

    max_gen_toks = (
        config.max_gen_toks
        if config.max_gen_toks is not None
        else suite.max_gen_toks
    )
    if max_gen_toks <= 0:
        raise ValueError("max_gen_toks must be positive")
    if config.num_fewshot is not None and config.num_fewshot < 0:
        raise ValueError("num_fewshot must be non-negative")

    limit, limit_metadata = resolve_limit(
        suite, config.limit_count, config.limit_fraction
    )
    sample_manifest = config.sample_manifest or suite.sample_manifest
    lm_eval_samples = None
    sample_selection = None
    if suite.dataset_contract is not None and (
        limit is not None or sample_manifest is not None
    ):
        raise ValueError("a full dataset contract cannot be combined with a limit or manifest")
    if sample_manifest is not None:
        if limit is not None:
            raise ValueError("sample manifest and limit are mutually exclusive")
        lm_eval_samples, sample_selection = load_sample_manifest(
            sample_manifest,
            tasks,
            expected_sha256=suite.sample_manifest_sha256,
        )
    analysis_protocol = None
    if suite.key.startswith("stage"):
        encoded = _STAGE_PROTOCOL_PATH.read_bytes()
        actual_sha256 = hashlib.sha256(encoded).hexdigest()
        if actual_sha256 != _STAGE_PROTOCOL_SHA256:
            raise ValueError("accuracy expansion protocol SHA256 changed")
        analysis_protocol = {
            "path": str(_STAGE_PROTOCOL_PATH.resolve()),
            "sha256": actual_sha256,
        }
    protocol = {
        "suite": suite.key,
        "tasks": list(tasks),
        "num_fewshot": (
            config.num_fewshot if config.num_fewshot is not None else suite.num_fewshot
        ),
        "apply_chat_template": apply_chat_template,
        "fewshot_as_multiturn": suite.fewshot_as_multiturn,
        "system_instruction": suite.system_instruction,
        "enable_thinking": enable_thinking,
        "think_end_token": config.think_end_token if enable_thinking else None,
        "max_gen_toks": max_gen_toks,
        "generation": suite.generation,
        "prompt_format": (
            "chat_thinking"
            if enable_thinking
            else ("chat_nonthinking" if apply_chat_template else "completion")
        ),
        "min_model_len": suite.min_model_len,
        "analysis_exclude_doc_ids": list(suite.analysis_exclude_doc_ids),
        "dataset_contract": (
            dict(suite.dataset_contract) if suite.dataset_contract is not None else None
        ),
        "dataset_contracts": (
            {
                task: dict(contract)
                for task, contract in suite.dataset_contracts.items()
            }
            if suite.dataset_contracts is not None
            else None
        ),
        "generation_kwargs": (
            {"temperature": 0.0, "max_gen_toks": max_gen_toks}
            if suite.key
            in {
                "smoke_generation",
                "confirm_generation",
                "generation_core",
                "stage3_generation",
            }
            else ({"temperature": 0.0} if suite.generation else None)
        ),
        "limit": limit_metadata,
        "sample_selection": sample_selection,
        "lm_eval_limit": limit,
        "lm_eval_samples": lm_eval_samples,
    }
    if analysis_protocol is not None:
        protocol["analysis_protocol"] = analysis_protocol
    return protocol


def configure_environment(
    arm: ArmConfig, *, selector_chunk_rows: int | None = None
) -> dict[str, str]:
    desired = {
        "TENSORBRIDGE_VLLM_BACKEND": arm.backend,
        "TENSORBRIDGE_NVFP4_FPMA_ALPHA": str(arm.alpha),
        "TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR": arm.selector,
        "TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION": str(int(arm.ulp_correction)),
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
    chunk_key = "TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR_CHUNK_ROWS"
    if selector_chunk_rows is not None:
        if selector_chunk_rows <= 0:
            raise ValueError("selector chunk rows must be positive")
        desired[chunk_key] = str(selector_chunk_rows)
    elif os.environ.get(chunk_key) is not None:
        raise RuntimeError(
            f"conflicting inherited precision environment: {chunk_key}="
            f"{os.environ[chunk_key]!r}, expected unset"
        )
    for key, value in desired.items():
        inherited = os.environ.get(key)
        if key in _PRECISION_ENV_KEYS and inherited not in (None, value):
            raise RuntimeError(
                f"conflicting inherited precision environment: {key}={inherited!r}, "
                f"expected {value!r}"
            )
        os.environ[key] = value
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "8"
    os.environ["MALLOC_ARENA_MAX"] = "2"
    return desired


def _capture_evaluation_environment() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in _EVALUATION_ENV_KEYS}


def _restore_evaluation_environment(state: dict[str, str | None]) -> None:
    for key, value in state.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _restores_evaluation_environment(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        state = _capture_evaluation_environment()
        try:
            return function(*args, **kwargs)
        finally:
            _restore_evaluation_environment(state)

    return wrapped


def load_runtime(allow_version_mismatch: bool = False) -> RuntimeBindings:
    versions = {
        "lm_eval": importlib.metadata.version("lm-eval"),
        "vllm": importlib.metadata.version("vllm"),
        "transformers": importlib.metadata.version("transformers"),
        "datasets": importlib.metadata.version("datasets"),
        "torch": importlib.metadata.version("torch"),
    }
    if not allow_version_mismatch:
        if versions["lm_eval"] != "0.4.11":
            raise RuntimeError(f"expected lm-eval 0.4.11, got {versions['lm_eval']}")
        if not versions["vllm"].startswith("0.20.2"):
            raise RuntimeError(f"expected vLLM 0.20.2, got {versions['vllm']}")

    from vllm.plugins.tensorbridge import register

    register()

    import lm_eval
    import torch
    import vllm
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import handle_non_serializable
    from vllm.model_executor.layers.quantization import get_quantization_config

    quant_class = get_quantization_config("modelopt_mixed").__name__
    if quant_class != "TensorBridgeModelOptMixedConfig":
        raise RuntimeError(f"TensorBridge vLLM plugin is inactive: {quant_class}")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    task_path = Path(__file__).with_name("tasks")
    return RuntimeBindings(
        simple_evaluate=lm_eval.simple_evaluate,
        json_default=handle_non_serializable,
        versions=versions,
        module_paths={
            "lm_eval": lm_eval.__file__,
            "vllm": vllm.__file__,
            "tensorbridge_tasks": str(task_path),
        },
        quant_config_class=quant_class,
        gpu=gpu,
        capability=capability,
        task_manager=TaskManager(include_path=str(task_path)),
    )


def build_model_args(config: RunConfig, protocol: dict[str, Any]) -> dict[str, Any]:
    if config.max_model_len <= 0 or config.max_num_seqs <= 0:
        raise ValueError("max_model_len and max_num_seqs must be positive")
    if not 0.0 < config.gpu_memory_utilization < 1.0:
        raise ValueError("gpu_memory_utilization must be strictly between 0 and 1")
    if config.max_model_len < protocol["min_model_len"]:
        raise ValueError(
            f"suite {protocol['suite']} requires max_model_len >= "
            f"{protocol['min_model_len']}, got {config.max_model_len}"
        )
    args: dict[str, Any] = {
        "pretrained": str(config.model),
        "quantization": "modelopt_mixed",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": config.max_model_len,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "language_model_only": True,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "disable_log_stats": True,
        "max_num_seqs": config.max_num_seqs,
        "seed": 1234,
        "enable_thinking": protocol["enable_thinking"],
        "max_gen_toks": protocol["max_gen_toks"],
    }
    if protocol["enable_thinking"]:
        args["think_end_token"] = protocol["think_end_token"]
    return args


def _assert_finite(value: Any, path: str = "result") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}: {value!r}")
    if not isinstance(value, (str, bytes, bool, int, float, dict, list, tuple)):
        item = getattr(value, "item", None)
        if callable(item):
            scalar = item()
            if scalar is not value:
                _assert_finite(scalar, path)
                return
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def _covered_tasks(result: dict[str, Any], requested: list[str]) -> set[str]:
    configs = set(result.get("configs", {}))
    groups = result.get("group_subtasks", {}) or {}
    for task in requested:
        if task in configs:
            continue
        subtasks = groups.get(task)
        if not subtasks:
            raise ValueError(f"requested task is missing from lm-eval result: {task}")
        flattened = set(subtasks if isinstance(subtasks, list) else [subtasks])
        if not flattened.issubset(configs):
            raise ValueError(f"lm-eval group {task} is missing leaf task results")
    return configs


def _validate_sample_coverage(
    result: dict[str, Any],
    samples: dict[str, list[dict[str, Any]]],
    selected_ids: dict[str, list[int]] | None = None,
    sample_selection: dict[str, Any] | None = None,
    dataset_contract: dict[str, Any] | None = None,
    stage_suite: str | None = None,
) -> dict[str, Any] | None:
    if selected_ids is not None and dataset_contract is not None:
        raise ValueError("selected samples and a full dataset contract are mutually exclusive")
    configs = set(result.get("configs", {}))
    if configs != set(samples):
        raise ValueError(
            "sample task set does not match lm-eval configs: "
            f"configs={sorted(configs)}, samples={sorted(samples)}"
        )
    counts = result.get("n-samples", {})
    if set(counts) != configs:
        raise ValueError("n-samples task set does not match lm-eval configs")
    if selected_ids is not None and set(selected_ids) != configs:
        raise ValueError("selected sample task set does not match lm-eval configs")
    if dataset_contract is not None and len(configs) != 1:
        raise ValueError("a full dataset contract requires exactly one lm-eval task")
    logged_verification: dict[str, Any] = {}
    logged_docs: dict[str, tuple[list[int], dict[int, dict[str, Any]]]] = {}
    for task, rows in samples.items():
        pairs: set[tuple[str, str]] = set()
        docs: set[str] = set()
        filters_by_doc: dict[str, set[str]] = {}
        ordered_doc_ids: list[Any] = []
        selected_docs: dict[int, dict[str, Any]] = {}
        for row in rows:
            if "doc_id" not in row:
                raise ValueError(f"sample for {task} is missing doc_id")
            missing_hashes = {
                key for key in ("doc_hash", "prompt_hash", "target_hash") if key not in row
            }
            if missing_hashes:
                raise ValueError(
                    f"sample for {task} is missing hashes: {sorted(missing_hashes)}"
                )
            doc_id = json.dumps(row["doc_id"], sort_keys=True, default=str)
            filter_name = str(row.get("filter", "none"))
            pair = (doc_id, filter_name)
            if pair in pairs:
                raise ValueError(f"duplicate (doc_id, filter) sample for {task}: {pair}")
            pairs.add(pair)
            filters_by_doc.setdefault(doc_id, set()).add(filter_name)
            if (
                type(row["doc_id"]) is not int or not isinstance(row.get("doc"), dict)
            ):
                raise ValueError(f"sample for {task} has invalid doc_id or doc")
            if doc_id not in docs:
                ordered_doc_ids.append(row["doc_id"])
                docs.add(doc_id)
                selected_docs[row["doc_id"]] = row["doc"]
            elif _canonical_doc_bytes(
                selected_docs[row["doc_id"]]
            ) != _canonical_doc_bytes(row["doc"]):
                raise ValueError(f"sample filters disagree on document content for {task}")
        expected_filters = {str(row.get("filter", "none")) for row in rows}
        incomplete_filters = {
            doc_id: sorted(filters)
            for doc_id, filters in filters_by_doc.items()
            if filters != expected_filters
        }
        if incomplete_filters:
            raise ValueError(
                f"sample documents for {task} do not share the same filters: "
                f"{incomplete_filters}"
            )
        effective = (
            len(selected_ids[task])
            if selected_ids is not None
            else int(counts[task]["effective"])
        )
        if len(docs) != effective:
            raise ValueError(
                f"sample coverage mismatch for {task}: {len(docs)} unique docs vs {effective}"
            )
        logged_docs[task] = (ordered_doc_ids, selected_docs)
        if selected_ids is not None and ordered_doc_ids != selected_ids[task]:
            raise ValueError(f"sample document IDs for {task} do not match the manifest")
        if selected_ids is not None:
            if sample_selection is None:
                raise ValueError("selected samples require selection metadata")
            digest = _selected_docs_sha256(selected_ids[task], selected_docs)
            if digest != sample_selection["selection"]["selected_docs_sha256"]:
                raise ValueError(f"sample document content for {task} does not match")
            logged_verification[task] = {
                "verified": True,
                "kind": "selected_docs",
                "size": len(selected_ids[task]),
                "ids_sha256": _ids_sha256(selected_ids[task]),
                "selected_docs_sha256": digest,
            }
        elif dataset_contract is not None:
            expected_doc_ids = list(range(dataset_contract["size"]))
            if ordered_doc_ids != expected_doc_ids:
                raise ValueError(
                    f"sample document IDs for {task} do not cover the full dataset"
                )
            full_digest = _full_dataset_sha256(
                selected_docs[doc_id] for doc_id in expected_doc_ids
            )
            if full_digest != dataset_contract["canonical_jsonl_sha256"]:
                raise ValueError(
                    f"sample document content for {task} does not match the full dataset"
                )
            logged_verification[task] = {
                "verified": True,
                "kind": "full_split",
                "size": len(expected_doc_ids),
                "canonical_jsonl_sha256": full_digest,
            }
    if stage_suite is not None:
        return _validate_stage_logged_samples(stage_suite, logged_docs)
    return {"verified": True, "tasks": logged_verification} if logged_verification else None


def _composite_selected_docs_sha256(
    logged_docs: dict[str, tuple[list[int], dict[int, dict[str, Any]]]],
) -> str:
    records = []
    for task in sorted(logged_docs):
        doc_ids, docs = logged_docs[task]
        records.extend(
            {
                "leaf_task": task,
                "doc_id": doc_id,
                "doc_sha256": hashlib.sha256(
                    _canonical_doc_bytes(docs[doc_id])
                ).hexdigest(),
            }
            for doc_id in doc_ids
        )
    encoded = json.dumps(
        records, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_stage_logged_samples(
    suite: str,
    logged_docs: dict[str, tuple[list[int], dict[int, dict[str, Any]]]],
) -> dict[str, Any]:
    if suite == "stage1_mc":
        expected_tasks = set(_STAGE1_SELECTED_DOC_SHA256)
        expected_ids = list(range(512))
        expected_composite = _STAGE1_COMPOSITE_SELECTED_DOC_SHA256
    elif suite == "stage2_mmlu_pro":
        expected_tasks = set(_STAGE2_LEAF_TASKS)
        expected_ids = list(range(64))
        expected_composite = _STAGE2_COMPOSITE_SELECTED_DOC_SHA256
    else:
        raise ValueError(f"unsupported stage logged-sample contract: {suite}")
    if set(logged_docs) != expected_tasks:
        raise ValueError(
            f"{suite} leaf task set changed: "
            f"actual={sorted(logged_docs)}, expected={sorted(expected_tasks)}"
        )
    task_records: dict[str, Any] = {}
    for task in sorted(logged_docs):
        doc_ids, docs = logged_docs[task]
        if doc_ids != expected_ids:
            raise ValueError(f"{suite} selected document IDs changed for {task}")
        selected_sha256 = _selected_docs_sha256(doc_ids, docs)
        if suite == "stage1_mc" and (
            selected_sha256 != _STAGE1_SELECTED_DOC_SHA256[task]
        ):
            raise ValueError(f"{suite} selected document content changed for {task}")
        task_records[task] = {
            "verified": True,
            "kind": "selected_docs",
            "size": len(doc_ids),
            "ids_sha256": _ids_sha256(doc_ids),
            "selected_docs_sha256": selected_sha256,
        }
    composite = _composite_selected_docs_sha256(logged_docs)
    if composite != expected_composite:
        raise ValueError(f"{suite} composite selected document SHA256 changed")
    return {
        "verified": True,
        "kind": "stage_selected_docs",
        "suite": suite,
        "tasks": task_records,
        "composite_selected_docs_sha256": composite,
    }


def _validate_n_samples(
    result: dict[str, Any],
    limit: int | float | None,
    selected_ids: dict[str, list[int]] | None = None,
    selected_split_size: int | None = None,
) -> None:
    if selected_ids is not None and set(result.get("n-samples", {})) != set(selected_ids):
        raise ValueError("selected sample task set does not match n-samples")
    for task, counts in result.get("n-samples", {}).items():
        original = int(counts["original"])
        effective = int(counts["effective"])
        if selected_ids is not None:
            if selected_split_size is None or original != selected_split_size:
                raise ValueError(
                    f"unexpected original sample count for {task}: "
                    f"{original} vs {selected_split_size}"
                )
            expected = len(selected_ids[task])
            # lm-eval 0.4.11 reports the full split as effective in samples mode.
            if effective not in {expected, original}:
                raise ValueError(
                    f"unexpected lm-eval sample count for {task}: {effective}"
                )
            counts["lm_eval_reported_effective"] = effective
            counts["selected_effective"] = expected
        elif limit is None:
            expected = original
        elif isinstance(limit, int):
            expected = min(limit, original)
        else:
            expected = min(math.ceil(original * limit), original)
        if selected_ids is None and effective != expected:
            raise ValueError(
                f"unexpected effective sample count for {task}: {effective} vs {expected}"
            )


def _validate_result_schema(result: dict[str, Any], expected_num_fewshot: int) -> None:
    required = {
        "results",
        "configs",
        "group_subtasks",
        "versions",
        "n-shot",
        "higher_is_better",
        "n-samples",
        "config",
        "git_hash",
        "date",
        "samples",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(f"lm-eval result is missing keys: {sorted(missing)}")
    configs = set(result["configs"])
    n_shot = result["n-shot"]
    if set(n_shot) != configs:
        raise ValueError("n-shot task set does not match lm-eval configs")
    mismatches = {
        task: value
        for task, value in n_shot.items()
        if int(value) != expected_num_fewshot
    }
    if mismatches:
        raise ValueError(
            f"unexpected effective fewshot counts (expected {expected_num_fewshot}): "
            f"{mismatches}"
        )


def _safe_task_filename(task: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task).strip("._") or "task"


def _encode_json(value: Any, json_default: Callable[[Any], Any]) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        default=json_default,
    )


def _write_atomic(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _checkpoint_shards(model: Path) -> tuple[str, ...]:
    index_path = model / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read checkpoint index: {index_path}") from error
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index must contain a non-empty weight_map")
    shards: set[str] = set()
    for value in weight_map.values():
        if not isinstance(value, str) or not value.endswith(".safetensors"):
            raise ValueError(f"invalid checkpoint shard name: {value!r}")
        if Path(value).name != value or value in {".", ".."}:
            raise ValueError(f"checkpoint shard must be a basename: {value!r}")
        shards.add(value)
    return tuple(sorted(shards))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_checkpoint_file(model: Path, relative_path: str) -> dict[str, Any]:
    path = model / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {path}")
    before = path.stat()
    sha256 = _sha256_file(path)
    after = path.stat()
    before_state = (before.st_size, before.st_mtime_ns)
    after_state = (after.st_size, after.st_mtime_ns)
    if before_state != after_state:
        raise RuntimeError(f"checkpoint file changed while hashing: {path}")
    return {
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": sha256,
    }


def _checkpoint_content_sha256(files: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(files):
        record = files[relative_path]
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(record["sha256"]))
        digest.update(b"\0")
    return digest.hexdigest()


def build_checkpoint_manifest(model: Path, workers: int = 1) -> dict[str, Any]:
    if not model.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model}")
    if workers <= 0:
        raise ValueError("checkpoint hash workers must be positive")
    shards = _checkpoint_shards(model)
    relative_paths = (*CHECKPOINT_METADATA_FILES, *shards)
    with ThreadPoolExecutor(max_workers=min(workers, len(relative_paths))) as pool:
        records = list(
            pool.map(lambda name: _hash_checkpoint_file(model, name), relative_paths)
        )
    files = dict(zip(relative_paths, records, strict=True))
    source: dict[str, str] | None = None
    if model.resolve().name == EXPECTED_CHECKPOINT_SOURCE["repo_id"].split("/")[-1]:
        if set(shards) != set(EXPECTED_WEIGHT_SHA256):
            raise ValueError("checkpoint shard list does not match the pinned source")
        expected_files = EXPECTED_METADATA_SHA256 | EXPECTED_WEIGHT_SHA256
        for name, expected in expected_files.items():
            if files[name]["sha256"] != expected:
                raise ValueError(f"checkpoint pinned SHA256 mismatch: {name}")
        for name, expected in EXPECTED_WEIGHT_SHA256.items():
            files[name]["expected_sha256"] = expected
            files[name]["expected_hash_source"] = (
                "huggingface_lfs_oid_at_pinned_revision"
            )
            files[name]["verified"] = True
        source = dict(EXPECTED_CHECKPOINT_SOURCE)
    return {
        "schema_version": 1,
        "format": CHECKPOINT_MANIFEST_FORMAT,
        "hash_algorithm": "sha256",
        "model_name": model.resolve().name,
        "source": source,
        "metadata_files": list(CHECKPOINT_METADATA_FILES),
        "weight_shards": list(shards),
        "weight_bytes": sum(files[name]["size_bytes"] for name in shards),
        "checkpoint_content_sha256": _checkpoint_content_sha256(files),
        "files": files,
    }


def write_checkpoint_manifest(
    model: Path,
    output: Path,
    workers: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest = build_checkpoint_manifest(model, workers=workers)
    _write_atomic(output, _encode_json(manifest, str) + "\n", overwrite)
    return manifest


def verify_checkpoint_manifest(model: Path, manifest_path: Path) -> dict[str, Any]:
    try:
        encoded = manifest_path.read_bytes()
    except OSError as error:
        raise FileNotFoundError(
            f"checkpoint manifest does not exist: {manifest_path}"
        ) from error
    try:
        manifest = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid checkpoint manifest JSON: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest must be a JSON object")
    expected_header = {
        "schema_version": 1,
        "format": CHECKPOINT_MANIFEST_FORMAT,
        "hash_algorithm": "sha256",
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"checkpoint manifest {key} must be {expected!r}, "
                f"got {manifest.get(key)!r}"
            )

    current_shards = _checkpoint_shards(model)
    metadata_files = manifest.get("metadata_files")
    weight_shards = manifest.get("weight_shards")
    files = manifest.get("files")
    if metadata_files != list(CHECKPOINT_METADATA_FILES):
        raise ValueError("checkpoint manifest metadata file contract mismatch")
    if weight_shards != list(current_shards):
        raise ValueError("checkpoint manifest shard list does not match the model index")
    expected_paths = {*CHECKPOINT_METADATA_FILES, *current_shards}
    if not isinstance(files, dict) or set(files) != expected_paths:
        raise ValueError("checkpoint manifest file set mismatch")

    expected_checkpoint_verified = False
    expected_model_name = EXPECTED_CHECKPOINT_SOURCE["repo_id"].split("/")[-1]
    if model.resolve().name == expected_model_name:
        if manifest.get("source") != EXPECTED_CHECKPOINT_SOURCE:
            raise ValueError("checkpoint manifest pinned source mismatch")
        if set(current_shards) != set(EXPECTED_WEIGHT_SHA256):
            raise ValueError("checkpoint shard list does not match the pinned source")
        for name, expected in EXPECTED_METADATA_SHA256.items():
            if files[name].get("sha256") != expected:
                raise ValueError(f"checkpoint expected SHA256 mismatch: {name}")
        for name, expected in EXPECTED_WEIGHT_SHA256.items():
            record = files[name]
            if (
                record.get("sha256") != expected
                or record.get("expected_sha256") != expected
                or record.get("expected_hash_source")
                != "huggingface_lfs_oid_at_pinned_revision"
                or record.get("verified") is not True
            ):
                raise ValueError(f"checkpoint expected SHA256 mismatch: {name}")
        expected_checkpoint_verified = True

    for relative_path in sorted(expected_paths):
        record = files[relative_path]
        if not isinstance(record, dict):
            raise ValueError(f"invalid manifest record for {relative_path}")
        sha256 = record.get("sha256")
        size_bytes = record.get("size_bytes")
        mtime_ns = record.get("mtime_ns")
        if (
            not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(mtime_ns, int)
        ):
            raise ValueError(f"invalid manifest record for {relative_path}")

    content_sha256 = _checkpoint_content_sha256(files)
    if manifest.get("checkpoint_content_sha256") != content_sha256:
        raise ValueError("checkpoint manifest aggregate content hash mismatch")

    metadata_sha256: dict[str, str] = {}
    for relative_path in CHECKPOINT_METADATA_FILES:
        record = files[relative_path]
        actual = _hash_checkpoint_file(model, relative_path)
        if actual["size_bytes"] != record["size_bytes"]:
            raise ValueError(f"checkpoint metadata size mismatch: {relative_path}")
        if actual["sha256"] != record["sha256"]:
            raise ValueError(f"checkpoint metadata SHA256 mismatch: {relative_path}")
        metadata_sha256[relative_path] = actual["sha256"]

    shard_records: dict[str, dict[str, Any]] = {}
    for relative_path in current_shards:
        path = model / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint shard does not exist: {path}")
        stat = path.stat()
        record = files[relative_path]
        if stat.st_size != record["size_bytes"]:
            raise ValueError(f"checkpoint shard size mismatch: {relative_path}")
        if stat.st_mtime_ns != record["mtime_ns"]:
            raise ValueError(f"checkpoint shard mtime mismatch: {relative_path}")
        shard_records[relative_path] = {
            "size_bytes": record["size_bytes"],
            "mtime_ns": record["mtime_ns"],
            "sha256": record["sha256"],
        }

    return {
        "model_path": str(model.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "checkpoint_content_sha256": content_sha256,
        "source": manifest.get("source"),
        "expected_checkpoint_verified": expected_checkpoint_verified,
        "metadata_sha256": metadata_sha256,
        "weight_shards": shard_records,
        "verification": {
            "metadata": "sha256",
            "weight_shards": "precomputed_sha256_with_size_and_mtime",
        },
    }


def write_sample_artifacts(
    samples: dict[str, list[dict[str, Any]]],
    samples_dir: Path,
    json_default: Callable[[Any], Any],
    overwrite: bool,
) -> dict[str, dict[str, Any]]:
    if samples_dir.exists() and any(samples_dir.iterdir()):
        raise FileExistsError(
            f"samples dir must be unique and empty, even with overwrite: {samples_dir}"
        )
    samples_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    for task in sorted(samples):
        rows = samples[task]
        _assert_finite(rows, f"samples.{task}")
        lines = [
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                default=json_default,
            )
            for row in rows
        ]
        encoded = ("\n".join(lines) + "\n").encode("utf-8")
        path = samples_dir / f"{_safe_task_filename(task)}.jsonl"
        _write_atomic(path, encoded.decode("utf-8"), overwrite)
        filters = sorted({str(row.get("filter", "none")) for row in rows})
        unique_docs = {
            json.dumps(row["doc_id"], sort_keys=True, default=str) for row in rows
        }
        artifacts[task] = {
            "path": str(path),
            "rows": len(rows),
            "unique_docs": len(unique_docs),
            "filters": filters,
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    return artifacts


def _run_git(path: Path, *args: str) -> str | None:
    process = subprocess.run(
        ["git", "-C", str(path), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if process.returncode != 0:
        return None
    return process.stdout.decode("utf-8", errors="replace")


def git_provenance(path: Path) -> dict[str, Any]:
    if path.is_file():
        path = path.parent
    root_text = _run_git(path, "rev-parse", "--show-toplevel")
    if root_text is None:
        return {"available": False}
    root = Path(root_text.strip())
    head = (_run_git(root, "rev-parse", "HEAD") or "").strip() or None
    status = _run_git(root, "status", "--porcelain=v1") or ""
    diff = _run_git(root, "diff", "--binary", "HEAD") or ""
    return {
        "available": True,
        "root": str(root),
        "head": head,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


def source_tree_sha256(repo: Path) -> dict[str, Any]:
    paths = [
        repo / "pyproject.toml",
        repo / "constraints" / "tensorbridge.json",
        repo / "vllm" / "plugins" / "tensorbridge.py",
        repo / "vllm" / "plugins" / "tensorbridge_qwen35.py",
        repo / "scripts" / "eval_nvfp4_lm_harness.py",
    ]
    for root in (repo / "vllm" / "plugins" / "tensorbridge_evaluation",):
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".json", ".yaml", ".cuh", ".h", ".cpp"}
        )
    digest = hashlib.sha256()
    existing = sorted({path.resolve() for path in paths if path.is_file()})
    for path in existing:
        relative = path.relative_to(repo.resolve()).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"sha256": digest.hexdigest(), "files": len(existing)}


def verify_stage_task_sources(lm_eval_module: Path, suite: str) -> dict[str, Any]:
    expected = _EXPECTED_STAGE_TASK_SOURCES.get(suite, {})
    task_root = lm_eval_module.resolve().parent / "tasks"
    records: dict[str, Any] = {}
    for task, expected_sha256 in expected.items():
        source_dir = task_root / task
        paths = sorted(
            path
            for path in source_dir.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and (
                path.suffix in {".py", ".yaml"}
                or path.name == "_default_template_yaml"
            )
        )
        if not paths:
            raise ValueError(f"lm-eval task source is missing for {task}")
        digest = hashlib.sha256()
        for path in paths:
            relative = path.relative_to(task_root).as_posix().encode()
            digest.update(relative)
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"lm-eval task source SHA256 changed for {task}: "
                f"{actual_sha256} vs {expected_sha256}"
            )
        records[task] = {
            "files": len(paths),
            "sha256": actual_sha256,
            "verified": True,
        }
    local_task_root = Path(__file__).with_name("tasks")
    for task, contract in _EXPECTED_LOCAL_STAGE_TASK_SOURCES.get(suite, {}).items():
        path = local_task_root / contract["relative_path"]
        if not path.is_file():
            raise ValueError(f"local TensorBridge task source is missing for {task}")
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != contract["sha256"]:
            raise ValueError(
                f"local TensorBridge task source SHA256 changed for {task}: "
                f"{actual_sha256} vs {contract['sha256']}"
            )
        if task in records:
            raise ValueError(f"duplicate stage task source record for {task}")
        records[task] = {
            "files": 1,
            "sha256": actual_sha256,
            "verified": True,
        }
    return records


def _validate_stage_checkpoint_identity(identity: dict[str, Any], suite: str) -> None:
    if not suite.startswith("stage"):
        return
    if identity.get("expected_checkpoint_verified") is not True:
        raise ValueError("stage evaluation requires the frozen NVIDIA checkpoint")
    if identity.get("source") != EXPECTED_CHECKPOINT_SOURCE:
        raise ValueError("stage checkpoint source or revision changed")
    if (
        identity.get("checkpoint_content_sha256")
        != _EXPECTED_STAGE_CHECKPOINT_CONTENT_SHA256
    ):
        raise ValueError("stage checkpoint content SHA256 changed")


def _validate_stage_runtime(runtime: RuntimeBindings, suite: str) -> None:
    if not suite.startswith("stage"):
        return
    if runtime.gpu != "NVIDIA H100 80GB HBM3" or runtime.capability != (9, 0):
        raise ValueError(
            "stage evaluation requires one NVIDIA H100 80GB HBM3 with SM90"
        )
    if runtime.versions != _EXPECTED_STAGE_RUNTIME_VERSIONS:
        raise ValueError("stage evaluation runtime versions changed")


@_restores_evaluation_environment
def run_evaluation(
    config: RunConfig,
    runtime_loader: Callable[[bool], RuntimeBindings] = load_runtime,
    dataset_loader: Callable[[str, str | None, str, str | None], Any] = _load_dataset,
) -> dict[str, Any]:
    if not config.model.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {config.model}")
    if config.output_json.exists() and not config.overwrite:
        raise FileExistsError(f"output already exists: {config.output_json}")
    if config.samples_dir.exists() and any(config.samples_dir.iterdir()):
        raise FileExistsError(
            f"samples directory must be unique and empty: {config.samples_dir}"
        )

    arm = resolve_arm(config.arm)
    protocol = resolve_protocol(config)
    checkpoint_before = verify_checkpoint_manifest(
        config.model, config.checkpoint_manifest
    )
    _validate_stage_checkpoint_identity(checkpoint_before, protocol["suite"])
    configured_env = configure_environment(
        arm,
        selector_chunk_rows=256 if protocol["suite"].startswith("stage") else None,
    )
    sample_dataset_verification = None
    dataset_pre_run_verification = None
    if protocol["sample_selection"] is not None:
        sample_dataset_verification = verify_sample_dataset(
            protocol["sample_selection"], dataset_loader
        )
        dataset_pre_run_verification = sample_dataset_verification
    elif protocol["dataset_contract"] is not None:
        dataset_pre_run_verification = verify_dataset_contract(
            protocol["dataset_contract"], dataset_loader
        )
    elif protocol["dataset_contracts"] is not None:
        dataset_pre_run_verification = {
            task: verify_dataset_contract(contract, dataset_loader)
            for task, contract in protocol["dataset_contracts"].items()
        }
    runtime = runtime_loader(config.allow_runtime_version_mismatch)
    _validate_stage_runtime(runtime, protocol["suite"])
    model_args = build_model_args(config, protocol)
    task_sources = verify_stage_task_sources(
        Path(runtime.module_paths["lm_eval"]), protocol["suite"]
    )
    bootstrap_iters = config.bootstrap_iters
    if bootstrap_iters is None:
        is_subset = (
            protocol["limit"]["kind"] != "none"
            or protocol["lm_eval_samples"] is not None
        )
        is_confirmation = protocol["suite"].startswith("confirm_")
        bootstrap_iters = 0 if is_subset or is_confirmation else 1000
    if bootstrap_iters < 0:
        raise ValueError("bootstrap_iters must be non-negative")

    repo = Path(__file__).resolve().parents[3]
    vllm_module = Path(runtime.module_paths["vllm"])
    source_before = {
        "tensorbridge_git": git_provenance(repo),
        "tensorbridge_tree": source_tree_sha256(repo),
        "vllm_git": git_provenance(vllm_module),
    }
    started = time.perf_counter()
    raw = runtime.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=protocol["tasks"],
        num_fewshot=protocol["num_fewshot"],
        batch_size=config.batch_size,
        use_cache=None,
        cache_requests=False,
        rewrite_requests_cache=False,
        delete_requests_cache=False,
        limit=protocol["lm_eval_limit"],
        samples=protocol["lm_eval_samples"],
        bootstrap_iters=bootstrap_iters,
        check_integrity=False,
        write_out=False,
        log_samples=True,
        task_manager=runtime.task_manager,
        system_instruction=protocol["system_instruction"],
        apply_chat_template=protocol["apply_chat_template"],
        fewshot_as_multiturn=protocol["fewshot_as_multiturn"],
        gen_kwargs=protocol["generation_kwargs"],
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    elapsed = time.perf_counter() - started
    if raw is None:
        raise RuntimeError("lm-eval returned no result on the primary rank")
    _validate_result_schema(raw, protocol["num_fewshot"])
    samples = raw.pop("samples", None)
    if not isinstance(samples, dict):
        raise ValueError("lm-eval result is missing per-task samples")

    _covered_tasks(raw, protocol["tasks"])
    _assert_finite(raw, "lm_eval")
    selected_ids = protocol["lm_eval_samples"]
    selected_split_size = (
        protocol["sample_selection"]["dataset"]["size"]
        if protocol["sample_selection"] is not None
        else None
    )
    _validate_n_samples(
        raw,
        protocol["lm_eval_limit"],
        selected_ids=selected_ids,
        selected_split_size=selected_split_size,
    )
    logged_dataset_verification = _validate_sample_coverage(
        raw,
        samples,
        selected_ids=selected_ids,
        sample_selection=protocol["sample_selection"],
        dataset_contract=protocol["dataset_contract"],
        stage_suite=(
            protocol["suite"] if protocol["dataset_contracts"] is not None else None
        ),
    )
    source_after = {
        "tensorbridge_git": git_provenance(repo),
        "tensorbridge_tree": source_tree_sha256(repo),
        "vllm_git": git_provenance(vllm_module),
    }
    if source_before != source_after:
        raise RuntimeError("source state changed while lm-eval was running")
    if protocol["sample_selection"] is not None:
        try:
            selected_after, selection_after = load_sample_manifest(
                Path(protocol["sample_selection"]["manifest_path"]),
                protocol["tasks"],
                expected_sha256=protocol["sample_selection"]["manifest_sha256"],
            )
        except Exception as error:
            raise RuntimeError("sample manifest changed while lm-eval was running") from error
        if selected_after != selected_ids or selection_after != protocol["sample_selection"]:
            raise RuntimeError("sample manifest changed while lm-eval was running")
    checkpoint_after = verify_checkpoint_manifest(
        config.model, config.checkpoint_manifest
    )
    if checkpoint_before != checkpoint_after:
        raise RuntimeError("checkpoint state changed while lm-eval was running")
    sample_artifacts = write_sample_artifacts(
        samples, config.samples_dir, runtime.json_default, config.overwrite
    )
    result = {
        "schema_version": 1,
        "experiment": "tensorbridge_nvfp4_lm_harness",
        "status": "passed",
        "arm": asdict(arm),
        "model_path": str(config.model),
        "checkpoint": {
            "start": checkpoint_before,
            "end": checkpoint_after,
            "unchanged": True,
        },
        "runtime": {
            "versions": runtime.versions,
            "module_paths": runtime.module_paths,
            "task_sources": task_sources,
            "quant_config_class": runtime.quant_config_class,
            "gpu": runtime.gpu,
            "capability": runtime.capability,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "cpu_thread_limits": {
                key: os.environ.get(key)
                for key in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "configured_environment": configured_env,
            "tensorbridge_nvrtc_flags": os.environ.get(
                "TENSORBRIDGE_EXTRA_NVRTC_FLAGS", ""
            ),
            "tensorbridge_cache_dir": os.environ.get("TENSORBRIDGE_CACHE_DIR"),
            "tensorbridge_cache_seed": (
                {
                    "source": os.environ.get("TENSORBRIDGE_CACHE_SEED_SOURCE"),
                    "sha256": os.environ.get("TENSORBRIDGE_CACHE_SEED_SHA256"),
                    "verified": os.environ.get("TENSORBRIDGE_CACHE_SEED_VERIFIED")
                    == "1",
                }
                if os.environ.get("TENSORBRIDGE_CACHE_SEED_SOURCE")
                else None
            ),
            "sample_dataset_verification": sample_dataset_verification,
            "dataset_verification": (
                {
                    "contract": (
                        protocol["sample_selection"]["dataset"]
                        if protocol["sample_selection"] is not None
                        else (
                            protocol["dataset_contract"]
                            if protocol["dataset_contract"] is not None
                            else protocol["dataset_contracts"]
                        )
                    ),
                    "pre_run": dataset_pre_run_verification,
                    "logged_samples": logged_dataset_verification,
                }
                if dataset_pre_run_verification is not None
                else None
            ),
            "source": {"start": source_before, "end": source_after, "unchanged": True},
        },
        "protocol": {
            key: value
            for key, value in protocol.items()
            if key not in {"lm_eval_limit", "lm_eval_samples"}
        }
        | {
            "batch_size": config.batch_size,
            "bootstrap_iters": bootstrap_iters,
            "response_cache": None,
            "seeds": {"python": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},
            "engine_args": model_args,
        },
        "production_contract": {
            "checkpoint_nvfp4_layers": 193,
            "expected_fp8_layers": 208,
            "snc_enabled": arm.backend == "tensorbridge",
            "scale_clamp": False,
            "strict_qwen36_layout": True,
            "lm_head_backend": "marlin_w4a16",
            "fpma_ulp_encoding": (
                "ulp_scale_msb_flag_v1" if arm.ulp_correction else None
            ),
        },
        "timing": {"evaluation_seconds": elapsed},
        "lm_eval": raw,
        "sample_artifacts": sample_artifacts,
    }
    _assert_finite(result)
    _write_atomic(
        config.output_json,
        _encode_json(result, runtime.json_default) + "\n",
        config.overwrite,
    )
    return result


def write_failure_artifact(config: RunConfig, error: BaseException) -> None:
    try:
        suite = resolve_suite(config.suite)
        suite_manifest = suite.sample_manifest
        suite_manifest_sha256 = suite.sample_manifest_sha256
        dataset_contract = suite.dataset_contract or suite.dataset_contracts
    except ValueError:
        suite_manifest = None
        suite_manifest_sha256 = None
        dataset_contract = None
    effective_manifest = config.sample_manifest or suite_manifest
    manifest_record = None
    if effective_manifest is not None:
        effective_manifest = effective_manifest.resolve()
        manifest_record = {
            "path": str(effective_manifest),
            "expected_sha256": (
                suite_manifest_sha256 or EXPECTED_SAMPLE_MANIFEST_SHA256
            ),
            "actual_sha256": None,
            "matches_expected": False,
        }
        try:
            if effective_manifest.is_file():
                actual_sha256 = hashlib.sha256(effective_manifest.read_bytes()).hexdigest()
                manifest_record["actual_sha256"] = actual_sha256
                manifest_record["matches_expected"] = (
                    actual_sha256 == manifest_record["expected_sha256"]
                )
        except OSError as manifest_error:
            manifest_record["read_error"] = (
                f"{type(manifest_error).__name__}: {manifest_error}"
            )
    failure = {
        "schema_version": 1,
        "experiment": "tensorbridge_nvfp4_lm_harness",
        "status": "failed",
        "arm": config.arm,
        "suite": config.suite,
        "model_path": str(config.model),
        "checkpoint_manifest": str(config.checkpoint_manifest),
        "sample_manifest": manifest_record,
        "dataset_contract": dataset_contract,
        "error_type": type(error).__name__,
        "error": str(error),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    try:
        _write_atomic(
            config.output_json,
            _encode_json(failure, str) + "\n",
            config.overwrite,
        )
    except FileExistsError:
        pass
