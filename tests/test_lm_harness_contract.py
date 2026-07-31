from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import pytest

from vllm.plugins.tensorbridge_evaluation import lm_harness


def _clear_precision_environment(monkeypatch):
    for key in lm_harness._PRECISION_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _runtime(simple_evaluate, module_path: Path):
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("", encoding="utf-8")
    return lm_harness.RuntimeBindings(
        simple_evaluate=simple_evaluate,
        json_default=str,
        versions={
            "lm_eval": "0.4.11",
            "vllm": "0.20.2+cu128",
            "transformers": "5.9.0",
            "datasets": "4.8.5",
            "torch": "2.11.0+cu128",
        },
        module_paths={"lm_eval": str(module_path), "vllm": str(module_path)},
        quant_config_class="TensorBridgeModelOptMixedConfig",
        gpu="NVIDIA H100 80GB HBM3",
        capability=(9, 0),
    )


def _sample(doc_id, filter_name="none", doc=None):
    return {
        "doc_id": doc_id,
        "filter": filter_name,
        "doc": doc or {"question": f"question-{doc_id}", "answer": f"answer-{doc_id}"},
        "doc_hash": f"doc-{doc_id}",
        "prompt_hash": f"prompt-{doc_id}",
        "target_hash": f"target-{doc_id}",
        "resps": [["A"]],
        "filtered_resps": ["A"],
    }


def _raw_result(
    task="arc_challenge",
    original=20,
    effective=16,
    filters=("none",),
    doc_ids=None,
    docs=None,
    reported_effective=None,
):
    doc_ids = list(range(effective)) if doc_ids is None else list(doc_ids)
    samples = [
        _sample(doc_id, filter_name, None if docs is None else docs[doc_id])
        for doc_id in doc_ids
        for filter_name in filters
    ]
    return {
        "results": {task: {"acc,none": 0.5}},
        "configs": {task: {"task": task}},
        "group_subtasks": {task: []},
        "versions": {task: 1.0},
        "n-samples": {
            task: {
                "original": original,
                "effective": (
                    effective if reported_effective is None else reported_effective
                ),
            }
        },
        "n-shot": {task: 0},
        "higher_is_better": {task: {"acc": True}},
        "config": {"model": "vllm"},
        "git_hash": "test-git-hash",
        "date": "2026-07-17T00:00:00",
        "samples": {task: samples},
    }


def _config(tmp_path, **overrides):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    manifest = tmp_path / "checkpoint_manifest.json"
    if not manifest.exists():
        shard = "model-00001-of-00001.safetensors"
        for name in lm_harness.CHECKPOINT_METADATA_FILES:
            path = model / name
            if name == "model.safetensors.index.json":
                path.write_text(
                    json.dumps({"weight_map": {"layer.weight": shard}}),
                    encoding="utf-8",
                )
            else:
                path.write_text("{}\n", encoding="utf-8")
        (model / shard).write_bytes(b"weights")
        lm_harness.write_checkpoint_manifest(model, manifest)
    values = {
        "model": model,
        "checkpoint_manifest": manifest,
        "arm": "ulp_v1",
        "suite": "smoke_mc",
        "output_json": tmp_path / "result.json",
        "samples_dir": tmp_path / "samples",
    }
    values.update(overrides)
    return lm_harness.RunConfig(**values)


class _FakeDataset(list):
    _fingerprint = "fake-fingerprint"


def _fake_sample_manifest(tmp_path, monkeypatch):
    docs = _FakeDataset(
        {"question": f"question-{doc_id}", "answer": f"answer-{doc_id}"}
        for doc_id in range(20)
    )
    namespace = "test-sample-manifest"
    ids = lm_harness._sha256_ranked_ids(namespace, len(docs), 3, 2)
    full = hashlib.sha256()
    for doc_id, doc in enumerate(docs):
        encoded = json.dumps(
            {"doc_id": doc_id, "doc": doc},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        full.update(encoded)
        full.update(b"\n")
    dataset = {
        "path": "fake/gsm8k",
        "name": "main",
        "split": "test",
        "size": len(docs),
        "revision": "fake-revision",
        "datasets_fingerprint": docs._fingerprint,
        "canonicalization": "sorted_minified_utf8_jsonl_with_doc_id_v1",
        "canonical_jsonl_sha256": full.hexdigest(),
    }
    selection = {
        "algorithm": "sha256_rank_v1",
        "candidate_start": 2,
        "namespace": namespace,
        "count": len(ids),
        "ids_sha256": lm_harness._ids_sha256(ids),
        "selected_docs_sha256": lm_harness._selected_docs_sha256(
            ids, {doc_id: docs[doc_id] for doc_id in ids}
        ),
    }
    manifest = {
        "schema_version": 1,
        "format": lm_harness.SAMPLE_MANIFEST_FORMAT,
        "dataset": dataset,
        "selection": selection,
        "tasks": {"tensorbridge_gsm8k_relative_smoke": ids},
    }
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(lm_harness, "EXPECTED_SAMPLE_DATASET", dataset)
    monkeypatch.setattr(lm_harness, "EXPECTED_SAMPLE_SELECTION", selection)
    manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        lm_harness,
        "EXPECTED_SAMPLE_MANIFEST_SHA256",
        manifest_sha256,
    )
    monkeypatch.setitem(
        lm_harness.SUITES,
        "confirm_generation",
        replace(
            lm_harness.SUITES["confirm_generation"],
            sample_manifest_sha256=manifest_sha256,
        ),
    )
    return path, docs, ids


def _fake_dataset_contract(docs):
    return {
        "path": "fake/arc",
        "name": "ARC-Challenge",
        "split": "test",
        "size": len(docs),
        "revision": "fake-revision",
        "datasets_fingerprint": docs._fingerprint,
        "canonicalization": "sorted_minified_utf8_jsonl_with_doc_id_v1",
        "canonical_jsonl_sha256": lm_harness._full_dataset_sha256(docs),
    }


def _replace_confirm_mc_contract(monkeypatch, contract):
    monkeypatch.setitem(
        lm_harness.SUITES,
        "confirm_mc",
        replace(lm_harness.SUITES["confirm_mc"], dataset_contract=contract),
    )


def test_arm_contracts_are_fixed():
    expected = {
        "official": ("official", 1.0, "none", False),
        "normal_a8": ("normal_a8", 1.0, "none", False),
        "fpma_default": ("tensorbridge", 1.0, "none", False),
        "selector_alpha1": ("tensorbridge", 1.0, "normal_b8_sse", False),
        "ulp_v1": ("tensorbridge", 1.0, "none", True),
        "alpha_0960": ("tensorbridge", 0.960, "none", False),
        "alpha_0961": ("tensorbridge", 0.961, "none", False),
    }
    assert {
        key: (arm.backend, arm.alpha, arm.selector, arm.ulp_correction)
        for key, arm in lm_harness.ARMS.items()
    } == expected
    with pytest.raises(ValueError, match="unknown lm-eval arm"):
        lm_harness.resolve_arm("unknown")

    assert set(lm_harness.EXPECTED_METADATA_SHA256) == set(
        lm_harness.CHECKPOINT_METADATA_FILES
    )
    assert lm_harness.EXPECTED_CHECKPOINT_SOURCE["revision"] == (
        "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
    )


def test_suite_protocols_keep_mc_and_generation_separate(tmp_path):
    mc = lm_harness.resolve_protocol(_config(tmp_path))
    assert mc["tasks"] == ["arc_challenge"]
    assert mc["apply_chat_template"] is False
    assert mc["lm_eval_limit"] == 16

    generation = lm_harness.resolve_protocol(
        _config(tmp_path, suite="smoke_generation")
    )
    assert generation["tasks"] == ["tensorbridge_gsm8k_relative_smoke"]
    assert generation["apply_chat_template"] is True
    assert generation["enable_thinking"] is False
    assert generation["prompt_format"] == "chat_nonthinking"
    assert generation["system_instruction"] == (
        "Use no more than six short sentences or equations. End with a separate final "
        "line in the exact form The answer is N. Replace N with the numeric answer only, "
        "omit units, and write nothing after that line."
    )
    assert generation["generation_kwargs"] == {
        "temperature": 0.0,
        "max_gen_toks": 1024,
    }

    confirm_mc = lm_harness.resolve_protocol(_config(tmp_path, suite="confirm_mc"))
    assert confirm_mc["tasks"] == ["tensorbridge_arc_challenge_confirm"]
    assert confirm_mc["lm_eval_limit"] is None
    assert confirm_mc["lm_eval_samples"] is None
    assert confirm_mc["analysis_exclude_doc_ids"] == list(range(16))
    assert confirm_mc["dataset_contract"] == lm_harness._ARC_CHALLENGE_TEST_CONTRACT
    with pytest.raises(ValueError, match="full dataset contract"):
        lm_harness.resolve_protocol(
            _config(tmp_path, suite="confirm_mc", limit_count=32)
        )
    with pytest.raises(ValueError, match="confirmation protocol overrides"):
        lm_harness.resolve_protocol(
            _config(tmp_path, suite="confirm_mc", num_fewshot=1)
        )
    with pytest.raises(ValueError, match="confirmation protocol overrides"):
        lm_harness.resolve_protocol(
            _config(tmp_path, suite="confirm_generation", max_gen_toks=512)
        )

    confirm_generation = lm_harness.resolve_protocol(
        _config(tmp_path, suite="confirm_generation")
    )
    selected = confirm_generation["lm_eval_samples"][
        "tensorbridge_gsm8k_relative_smoke"
    ]
    assert len(selected) == 128
    assert selected == sorted(selected)
    assert min(selected) >= 16
    assert lm_harness._ids_sha256(selected) == (
        "a43574fc29a99293c793b08c17a02b720cd4f9487e9fd33ef299515903924fc2"
    )
    assert confirm_generation["lm_eval_limit"] is None
    assert confirm_generation["sample_selection"]["manifest_sha256"] == (
        lm_harness.EXPECTED_SAMPLE_MANIFEST_SHA256
    )
    assert confirm_generation["dataset_contract"] is None
    with pytest.raises(ValueError, match="mutually exclusive"):
        lm_harness.resolve_protocol(
            _config(tmp_path, suite="confirm_generation", limit_count=4)
        )

    with pytest.raises(ValueError, match="only valid for generation"):
        lm_harness.resolve_protocol(_config(tmp_path, enable_thinking=True))
    with pytest.raises(ValueError, match="requires think_end_token"):
        lm_harness.resolve_protocol(
            _config(tmp_path, suite="smoke_generation", enable_thinking=True)
        )
    thinking = lm_harness.resolve_protocol(
        _config(
            tmp_path,
            suite="smoke_generation",
            enable_thinking=True,
            think_end_token="</think>",
        )
    )
    assert thinking["prompt_format"] == "chat_thinking"

    for invalid in (0, -1):
        with pytest.raises(ValueError, match="max_gen_toks must be positive"):
            lm_harness.resolve_protocol(_config(tmp_path, max_gen_toks=invalid))

    mmlu = lm_harness.resolve_protocol(_config(tmp_path, suite="mmlu_pro"))
    assert mmlu["num_fewshot"] == 5
    assert mmlu["fewshot_as_multiturn"] is False
    assert mmlu["min_model_len"] == 16384
    with pytest.raises(ValueError, match="requires max_model_len"):
        lm_harness.build_model_args(_config(tmp_path, suite="mmlu_pro"), mmlu)
    args = lm_harness.build_model_args(
        _config(tmp_path, suite="mmlu_pro", max_model_len=16384), mmlu
    )
    assert args["max_model_len"] == 16384

    stage1 = lm_harness.resolve_protocol(_config(tmp_path, suite="stage1_mc"))
    assert stage1["tasks"] == ["hellaswag", "winogrande"]
    assert stage1["limit"] == {
        "kind": "count",
        "value": 512,
        "from_suite_default": True,
    }
    assert stage1["lm_eval_limit"] == 512
    assert stage1["analysis_protocol"]["sha256"] == (
        lm_harness._STAGE_PROTOCOL_SHA256
    )
    with pytest.raises(ValueError, match="stage overrides"):
        lm_harness.resolve_protocol(
            _config(tmp_path, suite="stage1_mc", limit_count=511)
        )
    for override in (
        {"max_model_len": 8192},
        {"gpu_memory_utilization": 0.6},
        {"max_num_seqs": 4},
        {"batch_size": 1},
        {"allow_runtime_version_mismatch": True},
    ):
        with pytest.raises(ValueError, match="stage overrides"):
            lm_harness.resolve_protocol(
                _config(tmp_path, suite="stage1_mc", **override)
            )

    stage2 = lm_harness.resolve_protocol(
        _config(tmp_path, suite="stage2_mmlu_pro", max_model_len=16384)
    )
    assert stage2["tasks"] == ["mmlu_pro"]
    assert stage2["num_fewshot"] == 5
    assert stage2["lm_eval_limit"] == 64

    stage3 = lm_harness.resolve_protocol(
        _config(tmp_path, suite="stage3_generation")
    )
    stage3_ids = stage3["lm_eval_samples"]["tensorbridge_gsm8k_relative_smoke"]
    assert len(stage3_ids) == 256
    assert not set(stage3_ids) & set(stage3["sample_selection"]["excluded_doc_ids"])
    assert stage3["generation_kwargs"] == {
        "temperature": 0.0,
        "max_gen_toks": 1024,
    }
    assert stage3["sample_selection"]["manifest_sha256"] == (
        lm_harness._STAGE3_SAMPLE_MANIFEST_SHA256
    )
    expansion = json.loads(
        lm_harness._STAGE_PROTOCOL_PATH.read_text(encoding="utf-8")
    )
    stage3_contract = next(
        stage
        for stage in expansion["prospective_stages"]
        if stage["suite"] == "stage3_generation"
    )
    assert stage3_contract["task_source_sha256"] == (
        lm_harness._EXPECTED_LOCAL_STAGE_TASK_SOURCES["stage3_generation"]
        ["tensorbridge_gsm8k_relative_smoke"]["sha256"]
    )


def test_limit_contract_rejects_ambiguous_values():
    suite = lm_harness.SUITES["mc_core"]
    assert lm_harness.resolve_limit(suite, 32, None)[0] == 32
    assert lm_harness.resolve_limit(suite, None, 0.1)[0] == 0.1
    with pytest.raises(ValueError, match="mutually exclusive"):
        lm_harness.resolve_limit(suite, 32, 0.1)
    with pytest.raises(ValueError, match="positive"):
        lm_harness.resolve_limit(suite, 0, None)
    with pytest.raises(ValueError, match="strictly between"):
        lm_harness.resolve_limit(suite, None, 1.0)


def test_confirmation_protocol_binds_manifest_and_task_identity():
    repo = Path(lm_harness.__file__).resolve().parents[3]
    manifest_path = (
        repo
        / "vllm/plugins/tensorbridge_evaluation/samples/gsm8k_test_sha256_rank_128.json"
    )
    protocol_path = (
        repo / "vllm/plugins/tensorbridge_evaluation/protocols/accuracy_confirm_v1.json"
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    assert manifest_sha256 == lm_harness.EXPECTED_SAMPLE_MANIFEST_SHA256
    assert (
        protocol["tasks"]["confirm_generation"]["sample_manifest_sha256"]
        == manifest_sha256
    )
    assert protocol["tasks"]["confirm_mc"]["task"] == (
        "tensorbridge_arc_challenge_confirm"
    )
    assert protocol["tasks"]["confirm_mc"]["canonical_jsonl_sha256"] == (
        lm_harness._ARC_CHALLENGE_TEST_CONTRACT["canonical_jsonl_sha256"]
    )


def test_sample_manifest_binds_selection_and_dataset(tmp_path, monkeypatch):
    path, docs, ids = _fake_sample_manifest(tmp_path, monkeypatch)
    selected, metadata = lm_harness.load_sample_manifest(
        path, ("tensorbridge_gsm8k_relative_smoke",)
    )
    assert selected == {"tensorbridge_gsm8k_relative_smoke": ids}
    verified = lm_harness.verify_sample_dataset(
        metadata, lambda path, name, split, revision: docs
    )
    assert verified["verified"] is True
    assert verified["selected_docs_sha256"] == metadata["selection"][
        "selected_docs_sha256"
    ]

    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["tasks"]["tensorbridge_gsm8k_relative_smoke"] = list(reversed(ids))
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="raw SHA256 mismatch"):
        lm_harness.load_sample_manifest(
            path, ("tensorbridge_gsm8k_relative_smoke",)
        )
    monkeypatch.setattr(
        lm_harness,
        "EXPECTED_SAMPLE_MANIFEST_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="do not match sha256_rank_v1"):
        lm_harness.load_sample_manifest(
            path, ("tensorbridge_gsm8k_relative_smoke",)
        )


def test_sample_manifest_supports_frozen_exclusion_set(tmp_path):
    docs = _FakeDataset(
        {"question": f"question-{doc_id}", "answer": str(doc_id)}
        for doc_id in range(20)
    )
    dataset = _fake_dataset_contract(docs)
    excluded = [0, 1, 2, 7]
    namespace = "test-excluding-manifest"
    ids = lm_harness._sha256_ranked_ids_excluding(
        namespace, len(docs), 4, 2, excluded
    )
    selection = {
        "algorithm": "sha256_rank_excluding_ids_v1",
        "candidate_start": 2,
        "namespace": namespace,
        "count": len(ids),
        "excluded_ids_sha256": lm_harness._ids_sha256(excluded),
        "ids_sha256": lm_harness._ids_sha256(ids),
        "selected_docs_sha256": lm_harness._selected_docs_sha256(
            ids, {doc_id: docs[doc_id] for doc_id in ids}
        ),
    }
    manifest = {
        "schema_version": 1,
        "format": lm_harness.SAMPLE_MANIFEST_FORMAT,
        "dataset": dataset,
        "selection": selection,
        "excluded_doc_ids": excluded,
        "tasks": {"tensorbridge_gsm8k_relative_smoke": ids},
    }
    path = tmp_path / "excluding.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    selected, metadata = lm_harness.load_sample_manifest(
        path,
        ("tensorbridge_gsm8k_relative_smoke",),
        expected_sha256=manifest_sha256,
    )
    assert selected == {"tensorbridge_gsm8k_relative_smoke": ids}
    assert metadata["excluded_doc_ids"] == excluded
    assert not set(ids) & set(excluded)
    verified = lm_harness.verify_sample_dataset(
        metadata, lambda path, name, split, revision: docs
    )
    assert verified["selected_docs_sha256"] == selection["selected_docs_sha256"]


def test_dataset_contract_verifies_full_canonical_split():
    docs = _FakeDataset(
        {"question": f"question-{doc_id}", "answer": str(doc_id)}
        for doc_id in range(3)
    )
    contract = _fake_dataset_contract(docs)
    verified = lm_harness.verify_dataset_contract(
        contract, lambda path, name, split, revision: docs
    )
    assert verified == {
        "verified": True,
        "size": len(docs),
        "datasets_fingerprint": docs._fingerprint,
        "canonical_jsonl_sha256": contract["canonical_jsonl_sha256"],
    }

    changed = _FakeDataset(dict(doc) for doc in docs)
    changed[1] = {"question": "changed", "answer": "1"}
    with pytest.raises(ValueError, match="canonical JSONL SHA256"):
        lm_harness.verify_dataset_contract(
            contract, lambda path, name, split, revision: changed
        )


def test_runner_passes_explicit_samples_and_normalizes_upstream_count(
    tmp_path, monkeypatch
):
    _clear_precision_environment(monkeypatch)
    path, docs, ids = _fake_sample_manifest(tmp_path, monkeypatch)
    captured = {}

    def simple_evaluate(**kwargs):
        captured.update(kwargs)
        return _raw_result(
            task="tensorbridge_gsm8k_relative_smoke",
            original=len(docs),
            effective=len(ids),
            doc_ids=ids,
            docs=docs,
            reported_effective=len(docs),
        )

    runtime = _runtime(simple_evaluate, tmp_path / "fake_vllm" / "__init__.py")
    result = lm_harness.run_evaluation(
        _config(
            tmp_path,
            suite="confirm_generation",
            sample_manifest=path,
        ),
        runtime_loader=lambda _: runtime,
        dataset_loader=lambda path, name, split, revision: docs,
    )
    task = "tensorbridge_gsm8k_relative_smoke"
    assert captured["limit"] is None
    assert captured["samples"] == {task: ids}
    assert captured["bootstrap_iters"] == 0
    assert result["lm_eval"]["n-samples"][task]["lm_eval_reported_effective"] == len(
        docs
    )
    assert result["lm_eval"]["n-samples"][task]["selected_effective"] == len(ids)
    assert result["sample_artifacts"][task]["unique_docs"] == len(ids)
    assert result["runtime"]["sample_dataset_verification"]["verified"] is True
    assert result["runtime"]["dataset_verification"]["logged_samples"][
        "verified"
    ] is True


def test_runner_verifies_full_dataset_before_and_after_evaluation(
    tmp_path, monkeypatch
):
    _clear_precision_environment(monkeypatch)
    docs = _FakeDataset(
        {"question": f"question-{doc_id}", "answer": str(doc_id)}
        for doc_id in range(3)
    )
    contract = _fake_dataset_contract(docs)
    _replace_confirm_mc_contract(monkeypatch, contract)
    captured = {}

    def simple_evaluate(**kwargs):
        captured.update(kwargs)
        return _raw_result(
            task="tensorbridge_arc_challenge_confirm",
            original=len(docs),
            effective=len(docs),
            doc_ids=range(len(docs)),
            docs=docs,
        )

    runtime = _runtime(simple_evaluate, tmp_path / "fake_vllm" / "__init__.py")
    result = lm_harness.run_evaluation(
        _config(tmp_path, suite="confirm_mc"),
        runtime_loader=lambda _: runtime,
        dataset_loader=lambda path, name, split, revision: docs,
    )
    verification = result["runtime"]["dataset_verification"]
    task = "tensorbridge_arc_challenge_confirm"

    assert captured["tasks"] == [task]
    assert captured["limit"] is None
    assert captured["bootstrap_iters"] == 0
    assert verification["contract"] == contract
    assert verification["pre_run"]["canonical_jsonl_sha256"] == (
        contract["canonical_jsonl_sha256"]
    )
    assert verification["logged_samples"]["tasks"][task] == {
        "verified": True,
        "kind": "full_split",
        "size": len(docs),
        "canonical_jsonl_sha256": contract["canonical_jsonl_sha256"],
    }


def test_runner_rejects_logged_full_dataset_content_mismatch(tmp_path, monkeypatch):
    _clear_precision_environment(monkeypatch)
    docs = _FakeDataset(
        {"question": f"question-{doc_id}", "answer": str(doc_id)}
        for doc_id in range(3)
    )
    logged_docs = _FakeDataset(dict(doc) for doc in docs)
    logged_docs[1] = {"question": "changed", "answer": "1"}
    contract = _fake_dataset_contract(docs)
    _replace_confirm_mc_contract(monkeypatch, contract)

    def simple_evaluate(**kwargs):
        del kwargs
        return _raw_result(
            task="tensorbridge_arc_challenge_confirm",
            original=len(docs),
            effective=len(docs),
            doc_ids=range(len(docs)),
            docs=logged_docs,
        )

    runtime = _runtime(simple_evaluate, tmp_path / "fake_vllm" / "__init__.py")
    with pytest.raises(ValueError, match="does not match the full dataset"):
        lm_harness.run_evaluation(
            _config(tmp_path, suite="confirm_mc"),
            runtime_loader=lambda _: runtime,
            dataset_loader=lambda path, name, split, revision: docs,
        )


def test_runner_rejects_selected_document_content_mismatch(tmp_path, monkeypatch):
    _clear_precision_environment(monkeypatch)
    path, docs, ids = _fake_sample_manifest(tmp_path, monkeypatch)
    wrong_docs = _FakeDataset(dict(doc) for doc in docs)
    wrong_docs[ids[0]] = {"question": "changed", "answer": "changed"}

    def simple_evaluate(**kwargs):
        del kwargs
        return _raw_result(
            task="tensorbridge_gsm8k_relative_smoke",
            original=len(docs),
            effective=len(ids),
            doc_ids=ids,
            docs=wrong_docs,
            reported_effective=len(docs),
        )

    runtime = _runtime(simple_evaluate, tmp_path / "fake_vllm" / "__init__.py")
    with pytest.raises(ValueError, match="document content"):
        lm_harness.run_evaluation(
            _config(
                tmp_path,
                suite="confirm_generation",
                sample_manifest=path,
            ),
            runtime_loader=lambda _: runtime,
            dataset_loader=lambda path, name, split, revision: docs,
        )


def test_runner_rejects_manifest_changes_during_evaluation(tmp_path, monkeypatch):
    _clear_precision_environment(monkeypatch)
    path, docs, ids = _fake_sample_manifest(tmp_path, monkeypatch)

    def simple_evaluate(**kwargs):
        del kwargs
        path.write_bytes(path.read_bytes() + b"\n")
        return _raw_result(
            task="tensorbridge_gsm8k_relative_smoke",
            original=len(docs),
            effective=len(ids),
            doc_ids=ids,
            docs=docs,
            reported_effective=len(docs),
        )

    runtime = _runtime(simple_evaluate, tmp_path / "fake_vllm" / "__init__.py")
    with pytest.raises(RuntimeError, match="sample manifest changed"):
        lm_harness.run_evaluation(
            _config(
                tmp_path,
                suite="confirm_generation",
                sample_manifest=path,
            ),
            runtime_loader=lambda _: runtime,
            dataset_loader=lambda path, name, split, revision: docs,
        )


def test_runner_rejects_multi_filter_document_drift(tmp_path, monkeypatch):
    _clear_precision_environment(monkeypatch)
    path, docs, ids = _fake_sample_manifest(tmp_path, monkeypatch)

    def simple_evaluate(**kwargs):
        del kwargs
        raw = _raw_result(
            task="tensorbridge_gsm8k_relative_smoke",
            original=len(docs),
            effective=len(ids),
            filters=("first", "second"),
            doc_ids=ids,
            docs=docs,
            reported_effective=len(docs),
        )
        raw["samples"]["tensorbridge_gsm8k_relative_smoke"][1]["doc"] = {
            "question": "changed",
            "answer": "changed",
        }
        return raw

    runtime = _runtime(simple_evaluate, tmp_path / "fake_vllm" / "__init__.py")
    with pytest.raises(ValueError, match="filters disagree"):
        lm_harness.run_evaluation(
            _config(
                tmp_path,
                suite="confirm_generation",
                sample_manifest=path,
            ),
            runtime_loader=lambda _: runtime,
            dataset_loader=lambda path, name, split, revision: docs,
        )


def test_environment_rejects_inherited_precision_flags(monkeypatch):
    _clear_precision_environment(monkeypatch)
    state = lm_harness._capture_evaluation_environment()
    try:
        monkeypatch.setenv("TENSORBRIDGE_EXTRA_NVRTC_FLAGS", "-DUNREVIEWED=1")
        with pytest.raises(RuntimeError, match="conflicting inherited"):
            lm_harness.configure_environment(lm_harness.ARMS["ulp_v1"])

        monkeypatch.delenv("TENSORBRIDGE_EXTRA_NVRTC_FLAGS")
        configured = lm_harness.configure_environment(lm_harness.ARMS["ulp_v1"])
        assert configured["TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION"] == "1"
        assert "TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR_CHUNK_ROWS" not in configured
        assert configured["TENSORBRIDGE_EXTRA_NVRTC_FLAGS"] == ""
        lm_harness._restore_evaluation_environment(state)
        configured = lm_harness.configure_environment(
            lm_harness.ARMS["selector_alpha1"], selector_chunk_rows=256
        )
        assert configured["TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR_CHUNK_ROWS"] == "256"
    finally:
        lm_harness._restore_evaluation_environment(state)


def test_runner_passes_fail_closed_vllm_contract_and_writes_samples(tmp_path, monkeypatch):
    _clear_precision_environment(monkeypatch)
    captured = {}

    def simple_evaluate(**kwargs):
        captured.update(kwargs)
        return _raw_result()

    runtime = _runtime(simple_evaluate, tmp_path / "fake_vllm" / "__init__.py")
    config = _config(tmp_path)
    register_environment = {
        "TENSORBRIDGE_NVFP4_CPP_ROUTER": "1",
        "TENSORBRIDGE_NVFP4_CPP_ROUTER_STRICT": "1",
        "TENSORBRIDGE_NVFP4_SWZ64_DUAL_MMA_PREINT_LAYOUT": "1",
    }
    assert set(register_environment) <= lm_harness._EVALUATION_ENV_KEYS
    environment_before = lm_harness._capture_evaluation_environment()

    def runtime_loader(_):
        os.environ.update(register_environment)
        return runtime

    result = lm_harness.run_evaluation(config, runtime_loader=runtime_loader)
    assert lm_harness._capture_evaluation_environment() == environment_before

    assert captured["model"] == "vllm"
    assert captured["tasks"] == ["arc_challenge"]
    assert captured["limit"] == 16
    assert captured["samples"] is None
    assert captured["log_samples"] is True
    assert captured["system_instruction"] is None
    assert captured["task_manager"] is None
    assert captured["use_cache"] is None
    assert captured["model_args"]["quantization"] == "modelopt_mixed"
    assert captured["model_args"]["language_model_only"] is True
    assert captured["model_args"]["enforce_eager"] is True
    assert result["runtime"]["quant_config_class"] == "TensorBridgeModelOptMixedConfig"
    assert result["checkpoint"]["unchanged"] is True
    assert len(result["checkpoint"]["start"]["checkpoint_content_sha256"]) == 64
    assert "samples" not in result["lm_eval"]

    sample_path = Path(result["sample_artifacts"]["arc_challenge"]["path"])
    encoded = sample_path.read_bytes()
    assert result["sample_artifacts"]["arc_challenge"]["rows"] == 16
    assert result["sample_artifacts"]["arc_challenge"]["unique_docs"] == 16
    assert result["sample_artifacts"]["arc_challenge"]["sha256"] == hashlib.sha256(
        encoded
    ).hexdigest()
    persisted = json.loads(config.output_json.read_text(encoding="utf-8"))
    assert persisted["status"] == "passed"
    assert persisted["arm"]["key"] == "ulp_v1"


def test_multi_filter_samples_count_unique_documents(tmp_path, monkeypatch):
    _clear_precision_environment(monkeypatch)
    captured = {}

    def simple_evaluate(**kwargs):
        captured.update(kwargs)
        return _raw_result(
            task="tensorbridge_gsm8k_relative_smoke",
            original=2,
            effective=2,
            filters=("strict-match", "flexible-extract"),
        )

    runtime = _runtime(simple_evaluate, tmp_path / "fake_vllm" / "__init__.py")
    result = lm_harness.run_evaluation(
        _config(tmp_path, suite="smoke_generation"),
        runtime_loader=lambda _: runtime,
    )
    assert captured["system_instruction"] == (
        "Use no more than six short sentences or equations. End with a separate final "
        "line in the exact form The answer is N. Replace N with the numeric answer only, "
        "omit units, and write nothing after that line."
    )
    artifact = result["sample_artifacts"]["tensorbridge_gsm8k_relative_smoke"]
    assert artifact["rows"] == 4
    assert artifact["unique_docs"] == 2
    assert artifact["filters"] == ["flexible-extract", "strict-match"]


def test_runner_rejects_nonfinite_and_partial_final_results(tmp_path, monkeypatch):
    _clear_precision_environment(monkeypatch)

    def nonfinite(**kwargs):
        del kwargs
        raw = _raw_result()
        raw["results"]["arc_challenge"]["acc,none"] = float("nan")
        return raw

    runtime = _runtime(nonfinite, tmp_path / "fake_vllm" / "__init__.py")
    with pytest.raises(ValueError, match="non-finite"):
        lm_harness.run_evaluation(_config(tmp_path), runtime_loader=lambda _: runtime)

    _clear_precision_environment(monkeypatch)

    def partial(**kwargs):
        del kwargs
        return _raw_result(
            task="tensorbridge_gsm8k_relative_smoke", original=20, effective=16
        )

    runtime = _runtime(partial, tmp_path / "fake_vllm2" / "__init__.py")
    with pytest.raises(ValueError, match="effective sample count"):
        lm_harness.run_evaluation(
            _config(
                tmp_path,
                suite="generation_core",
                output_json=tmp_path / "full.json",
                samples_dir=tmp_path / "full_samples",
            ),
            runtime_loader=lambda _: runtime,
        )


def test_runner_rejects_samples_without_pairing_hashes(tmp_path, monkeypatch):
    _clear_precision_environment(monkeypatch)

    def missing_hash(**kwargs):
        del kwargs
        raw = _raw_result()
        del raw["samples"]["arc_challenge"][0]["prompt_hash"]
        return raw

    runtime = _runtime(missing_hash, tmp_path / "fake_vllm" / "__init__.py")
    with pytest.raises(ValueError, match="missing hashes"):
        lm_harness.run_evaluation(_config(tmp_path), runtime_loader=lambda _: runtime)


def test_runner_rejects_effective_fewshot_mismatch(tmp_path, monkeypatch):
    _clear_precision_environment(monkeypatch)

    def wrong_nshot(**kwargs):
        del kwargs
        raw = _raw_result()
        raw["n-shot"]["arc_challenge"] = 1
        return raw

    runtime = _runtime(wrong_nshot, tmp_path / "fake_vllm" / "__init__.py")
    with pytest.raises(ValueError, match="effective fewshot"):
        lm_harness.run_evaluation(_config(tmp_path), runtime_loader=lambda _: runtime)


def test_runner_rejects_source_changes_during_evaluation(tmp_path, monkeypatch):
    _clear_precision_environment(monkeypatch)
    runtime = _runtime(
        lambda **kwargs: _raw_result(), tmp_path / "fake_vllm" / "__init__.py"
    )
    tree_states = iter(
        [
            {"sha256": "before", "files": 1},
            {"sha256": "after", "files": 1},
        ]
    )
    monkeypatch.setattr(lm_harness, "source_tree_sha256", lambda path: next(tree_states))
    monkeypatch.setattr(
        lm_harness,
        "git_provenance",
        lambda path: {"available": True, "head": "same", "dirty": True},
    )
    with pytest.raises(RuntimeError, match="source state changed"):
        lm_harness.run_evaluation(_config(tmp_path), runtime_loader=lambda _: runtime)
    assert not (tmp_path / "samples").exists()


def test_checkpoint_manifest_hashes_metadata_and_fast_checks_shards(tmp_path):
    config = _config(tmp_path)
    identity = lm_harness.verify_checkpoint_manifest(
        config.model, config.checkpoint_manifest
    )
    assert identity["verification"] == {
        "metadata": "sha256",
        "weight_shards": "precomputed_sha256_with_size_and_mtime",
    }
    assert len(identity["manifest_sha256"]) == 64
    assert len(identity["weight_shards"]) == 1
    assert identity["source"] is None
    assert identity["expected_checkpoint_verified"] is False

    (config.model / "config.json").write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="metadata (size|SHA256) mismatch"):
        lm_harness.verify_checkpoint_manifest(
            config.model, config.checkpoint_manifest
        )


def test_checkpoint_manifest_rejects_same_size_shard_replacement(tmp_path):
    config = _config(tmp_path)
    shard = config.model / "model-00001-of-00001.safetensors"
    before = shard.stat()
    shard.write_bytes(b"changed")
    os.utime(
        shard,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
    )
    with pytest.raises(ValueError, match="checkpoint shard mtime mismatch"):
        lm_harness.verify_checkpoint_manifest(
            config.model, config.checkpoint_manifest
        )


def test_git_provenance_accepts_a_file_path():
    provenance = lm_harness.git_provenance(Path(lm_harness.__file__))
    assert provenance["available"] is True
    assert provenance["head"]


def test_stage_task_source_hash_is_fail_closed(tmp_path, monkeypatch):
    module = tmp_path / "lm_eval" / "__init__.py"
    source = module.parent / "tasks" / "example" / "task.yaml"
    source.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    source.write_text("task: example\n", encoding="utf-8")
    digest = hashlib.sha256()
    digest.update(b"example/task.yaml\0")
    digest.update(source.read_bytes())
    digest.update(b"\0")
    monkeypatch.setitem(
        lm_harness._EXPECTED_STAGE_TASK_SOURCES,
        "test_stage",
        {"example": digest.hexdigest()},
    )

    verified = lm_harness.verify_stage_task_sources(module, "test_stage")
    assert verified == {
        "example": {"files": 1, "sha256": digest.hexdigest(), "verified": True}
    }
    source.write_text("task: changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="task source SHA256 changed"):
        lm_harness.verify_stage_task_sources(module, "test_stage")


def test_stage3_local_task_source_hash_is_fail_closed(tmp_path):
    module = tmp_path / "lm_eval" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    verified = lm_harness.verify_stage_task_sources(module, "stage3_generation")
    assert verified == {
        "tensorbridge_gsm8k_relative_smoke": {
            "files": 1,
            "sha256": (
                "47c193604c56a717778641320d09ed49cab619f00406ce3588cc235c447a36a5"
            ),
            "verified": True,
        }
    }


def test_stage_checkpoint_and_gpu_contracts_are_fail_closed(tmp_path):
    identity = {
        "expected_checkpoint_verified": True,
        "source": dict(lm_harness.EXPECTED_CHECKPOINT_SOURCE),
        "checkpoint_content_sha256": (
            lm_harness._EXPECTED_STAGE_CHECKPOINT_CONTENT_SHA256
        ),
    }
    lm_harness._validate_stage_checkpoint_identity(identity, "stage1_mc")
    with pytest.raises(ValueError, match="frozen NVIDIA checkpoint"):
        lm_harness._validate_stage_checkpoint_identity(
            identity | {"expected_checkpoint_verified": False}, "stage1_mc"
        )

    runtime = _runtime(lambda **kwargs: _raw_result(), tmp_path / "vllm" / "__init__.py")
    lm_harness._validate_stage_runtime(runtime, "stage1_mc")
    with pytest.raises(ValueError, match="requires one NVIDIA H100"):
        lm_harness._validate_stage_runtime(
            replace(runtime, gpu="NVIDIA A100-SXM4-80GB", capability=(8, 0)),
            "stage1_mc",
        )
    with pytest.raises(ValueError, match="runtime versions changed"):
        lm_harness._validate_stage_runtime(
            replace(runtime, versions=runtime.versions | {"lm_eval": "0.4.12"}),
            "stage1_mc",
        )


def test_stage_logged_samples_bind_leaf_ids_and_processed_documents(monkeypatch):
    logged_docs = {
        task: (
            list(range(512)),
            {
                doc_id: {"task": task, "doc_id": doc_id, "text": f"row-{doc_id}"}
                for doc_id in range(512)
            },
        )
        for task in ("hellaswag", "winogrande")
    }
    selected = {
        task: lm_harness._selected_docs_sha256(doc_ids, docs)
        for task, (doc_ids, docs) in logged_docs.items()
    }
    composite = lm_harness._composite_selected_docs_sha256(logged_docs)
    monkeypatch.setattr(lm_harness, "_STAGE1_SELECTED_DOC_SHA256", selected)
    monkeypatch.setattr(
        lm_harness, "_STAGE1_COMPOSITE_SELECTED_DOC_SHA256", composite
    )
    verified = lm_harness._validate_stage_logged_samples("stage1_mc", logged_docs)
    assert verified["verified"] is True
    assert verified["composite_selected_docs_sha256"] == composite
    assert set(verified["tasks"]) == {"hellaswag", "winogrande"}

    changed = {
        task: (list(doc_ids), dict(docs))
        for task, (doc_ids, docs) in logged_docs.items()
    }
    changed["hellaswag"][1][0] = {"changed": True}
    with pytest.raises(ValueError, match="selected document content changed"):
        lm_harness._validate_stage_logged_samples("stage1_mc", changed)


def test_failure_artifact_is_machine_readable(tmp_path):
    config = _config(tmp_path)
    lm_harness.write_failure_artifact(config, RuntimeError("boom"))
    failure = json.loads(config.output_json.read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["error_type"] == "RuntimeError"
    assert failure["error"] == "boom"

    confirm_config = _config(
        tmp_path,
        suite="confirm_generation",
        output_json=tmp_path / "confirm_failure.json",
    )
    lm_harness.write_failure_artifact(confirm_config, RuntimeError("confirm boom"))
    confirm_failure = json.loads(
        confirm_config.output_json.read_text(encoding="utf-8")
    )
    manifest = confirm_failure["sample_manifest"]
    assert Path(manifest["path"]) == (
        lm_harness.SUITES["confirm_generation"].sample_manifest.resolve()
    )
    assert manifest["expected_sha256"] == (
        lm_harness.EXPECTED_SAMPLE_MANIFEST_SHA256
    )
    assert manifest["actual_sha256"] == (
        lm_harness.EXPECTED_SAMPLE_MANIFEST_SHA256
    )
    assert manifest["matches_expected"] is True
