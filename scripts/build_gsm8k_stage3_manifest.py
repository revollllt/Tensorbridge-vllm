#!/usr/bin/env python3
"""Build the frozen non-overlapping GSM8K sample manifest for accuracy stage 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


for _name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.setdefault(_name, "1")

from vllm.plugins.tensorbridge_evaluation import lm_harness


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIRM_MANIFEST = (
    REPO_ROOT
    / "vllm/plugins/tensorbridge_evaluation/samples/gsm8k_test_sha256_rank_128.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "vllm/plugins/tensorbridge_evaluation/samples/gsm8k_test_stage3_sha256_rank_256.json"
)
TASK = "tensorbridge_gsm8k_relative_smoke"
NAMESPACE = "tensorbridge-gsm8k-stage3-v2"
COUNT = 256
CANDIDATE_START = 16
EXPECTED_EXCLUDED_IDS_SHA256 = (
    "f6cf62e20cba6e0366d6a4dbdf19715d806afa2f7d5042ed26dea058a50a8bce"
)


def build_manifest() -> dict[str, object]:
    encoded = CONFIRM_MANIFEST.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != lm_harness.EXPECTED_SAMPLE_MANIFEST_SHA256:
        raise ValueError("confirmation sample manifest SHA256 changed")
    confirm = json.loads(encoded)
    confirm_ids = confirm.get("tasks", {}).get(TASK)
    if not isinstance(confirm_ids, list) or len(confirm_ids) != 128:
        raise ValueError("confirmation sample IDs are invalid")

    excluded_ids = sorted(set(range(CANDIDATE_START)) | set(confirm_ids))
    if lm_harness._ids_sha256(excluded_ids) != EXPECTED_EXCLUDED_IDS_SHA256:
        raise ValueError("stage-3 exclusion set changed")

    dataset_contract = dict(lm_harness.EXPECTED_SAMPLE_DATASET)
    dataset = lm_harness._load_dataset(
        dataset_contract["path"],
        dataset_contract["name"],
        dataset_contract["split"],
        dataset_contract["revision"],
    )
    lm_harness._verify_loaded_dataset_contract(dataset_contract, dataset)
    selected_ids = lm_harness._sha256_ranked_ids_excluding(
        NAMESPACE,
        dataset_contract["size"],
        COUNT,
        CANDIDATE_START,
        excluded_ids,
    )
    selected_docs = {doc_id: dict(dataset[doc_id]) for doc_id in selected_ids}
    selection = {
        "algorithm": "sha256_rank_excluding_ids_v1",
        "candidate_start": CANDIDATE_START,
        "namespace": NAMESPACE,
        "count": COUNT,
        "excluded_ids_sha256": lm_harness._ids_sha256(excluded_ids),
        "ids_sha256": lm_harness._ids_sha256(selected_ids),
        "selected_docs_sha256": lm_harness._selected_docs_sha256(
            selected_ids, selected_docs
        ),
    }
    return {
        "schema_version": 1,
        "format": lm_harness.SAMPLE_MANIFEST_FORMAT,
        "dataset": dataset_contract,
        "selection": selection,
        "excluded_doc_ids": excluded_ids,
        "tasks": {TASK: selected_ids},
    }


def _write_atomic(path: Path, encoded: str, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(encoded, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest()
    encoded = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    _write_atomic(args.output, encoded, args.overwrite)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                "selected_ids_sha256": manifest["selection"]["ids_sha256"],
                "selected_docs_sha256": manifest["selection"][
                    "selected_docs_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
