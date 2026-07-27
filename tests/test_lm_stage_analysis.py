import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import analyze_nvfp4_lm_stages as analysis


def _tiny_spec(*, suite="stage1_mc", generation=False, post_confirmation=False):
    leaf = analysis.LeafSpec(
        task="task_a",
        metrics=("acc",),
        primary_metric="acc",
        filter_name="none",
        expected_doc_ids=(0, 1, 2, 3),
        analysis_doc_ids=(0, 1, 2, 3),
        original_size=4,
    )
    return analysis.StageSpec(
        suite=suite,
        requested_tasks=("task_a",),
        leaves=(leaf,),
        num_fewshot=0,
        max_model_len=4096,
        max_gen_toks=256,
        generation=generation,
        apply_chat_template=generation,
        prompt_format="chat_nonthinking" if generation else "completion",
        limit_count=None,
        dataset_contracts=None,
        manifest_path=None,
        manifest_sha256=None,
        manifest=None,
        format_regex=None,
        format_min_valid=None,
        post_confirmation=post_confirmation,
    )


def _analysis_protocol(resamples=20):
    return {
        "screening_margin_vs_normal_a8": -0.05,
        "screening_decision": (
            "paired_point_estimate_delta_greater_than_or_equal_to_margin"
        ),
        "paired_confidence_interval": {
            "method": "paired_nonparametric_bootstrap_percentile",
            "confidence_level": 0.95,
            "resamples": resamples,
            "seed": 20_260_718,
            "mmlu_pro_resampling_unit": "document_within_fixed_category",
        },
    }


def _sample_row(doc_id, *, response="The answer is 1", exact_match=1.0):
    digest = f"{doc_id + 1:064x}"
    return {
        "doc_id": doc_id,
        "filter": "final-answer",
        "metrics": ["exact_match"],
        "doc": {"question": f"question-{doc_id}", "answer": "#### 1"},
        "doc_hash": digest,
        "prompt_hash": digest,
        "target_hash": digest,
        "exact_match": exact_match,
        "resps": [[response]],
    }


def _write_jsonl(path, rows):
    encoded = (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
            for row in rows
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    return {
        "path": str(path),
        "rows": len(rows),
        "unique_docs": len({row["doc_id"] for row in rows}),
        "filters": ["final-answer"],
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def test_frozen_protocol_builds_all_five_supported_suite_specs():
    protocol, encoded = analysis._load_json_object(
        analysis.DEFAULT_PROTOCOL, "protocol"
    )
    assert hashlib.sha256(encoded).hexdigest() == analysis.EXPECTED_PROTOCOL_SHA256

    stage1 = analysis._build_spec(
        protocol, analysis.DEFAULT_PROTOCOL, "stage1_mc"
    )
    stage2 = analysis._build_spec(
        protocol, analysis.DEFAULT_PROTOCOL, "stage2_mmlu_pro"
    )
    stage3 = analysis._build_spec(
        protocol, analysis.DEFAULT_PROTOCOL, "stage3_generation"
    )
    confirm_mc = analysis._build_spec(
        protocol, analysis.DEFAULT_PROTOCOL, "confirm_mc"
    )
    confirm_generation = analysis._build_spec(
        protocol, analysis.DEFAULT_PROTOCOL, "confirm_generation"
    )

    assert [leaf.task for leaf in stage1.leaves] == ["hellaswag", "winogrande"]
    assert [len(leaf.expected_doc_ids) for leaf in stage1.leaves] == [512, 512]
    assert analysis._STAGE1_COMPOSITE_DOC_SHA256 == (
        "b8f345a9030494ab72895765d51cc312f0600043fc6f0eac489e516e451c310a"
    )
    assert protocol["prospective_stages"][0][
        "composite_sample_identity_sha256"
    ] == analysis._STAGE1_COMPOSITE_DOC_SHA256
    assert len(stage2.leaves) == 14
    assert all(leaf.expected_doc_ids == tuple(range(64)) for leaf in stage2.leaves)
    assert len(stage3.leaves[0].expected_doc_ids) == 256
    assert analysis._ids_sha256(stage3.leaves[0].expected_doc_ids) == (
        analysis._STAGE3_IDS_SHA256
    )
    stage3_protocol = next(
        item
        for item in protocol["prospective_stages"]
        if item["suite"] == "stage3_generation"
    )
    assert stage3_protocol["task_source_sha256"] == (
        analysis._STAGE_TASK_SOURCE_SHA256["stage3_generation"][
            "tensorbridge_gsm8k_relative_smoke"
        ]
    )
    assert confirm_mc.leaves[0].analysis_doc_ids == tuple(range(16, 1172))
    assert len(confirm_generation.leaves[0].expected_doc_ids) == 128


def test_protocol_validation_rejects_comparison_or_stage_drift():
    protocol, _ = analysis._load_json_object(analysis.DEFAULT_PROTOCOL, "protocol")
    changed = copy.deepcopy(protocol)
    changed["analysis"]["comparisons"].pop()
    with pytest.raises(ValueError, match="comparison set changed"):
        analysis._validate_protocol(changed)

    changed = copy.deepcopy(protocol)
    changed["prospective_stages"][1]["leaf_tasks"].pop("mmlu_pro_math")
    with pytest.raises(ValueError, match="MMLU-Pro leaves changed"):
        analysis._validate_prospective_stages(changed)


def test_stage3_local_task_source_record_is_required():
    protocol, _ = analysis._load_json_object(analysis.DEFAULT_PROTOCOL, "protocol")
    spec = analysis._build_spec(
        protocol, analysis.DEFAULT_PROTOCOL, "stage3_generation"
    )
    source_sha = analysis._STAGE_TASK_SOURCE_SHA256["stage3_generation"][
        "tensorbridge_gsm8k_relative_smoke"
    ]
    runtime = {
        "task_sources": {
            "tensorbridge_gsm8k_relative_smoke": {
                "files": 1,
                "sha256": source_sha,
                "verified": True,
            }
        }
    }
    analysis._validate_task_sources(runtime, spec)

    runtime["task_sources"]["tensorbridge_gsm8k_relative_smoke"][
        "sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="task source verification failed"):
        analysis._validate_task_sources(runtime, spec)


def test_limited_stage_requires_runner_selected_document_verification():
    protocol, _ = analysis._load_json_object(analysis.DEFAULT_PROTOCOL, "protocol")
    spec = analysis._build_spec(protocol, analysis.DEFAULT_PROTOCOL, "stage1_mc")
    pre_run = {
        task: {
            "verified": True,
            "size": contract["size"],
            "datasets_fingerprint": contract["datasets_fingerprint"],
            "canonical_jsonl_sha256": contract["canonical_jsonl_sha256"],
        }
        for task, contract in analysis._STAGE1_DATASET_CONTRACTS.items()
    }
    task_records = {
        leaf.task: {
            "verified": True,
            "kind": "selected_docs",
            "size": 512,
            "ids_sha256": analysis._ids_sha256(tuple(range(512))),
            "selected_docs_sha256": leaf.selected_docs_sha256,
        }
        for leaf in spec.leaves
    }
    verification = {
        "contract": analysis._STAGE1_DATASET_CONTRACTS,
        "pre_run": pre_run,
        "logged_samples": {
            "verified": True,
            "kind": "stage_selected_docs",
            "suite": "stage1_mc",
            "tasks": task_records,
            "composite_selected_docs_sha256": (
                analysis._STAGE1_COMPOSITE_DOC_SHA256
            ),
        },
    }
    result = {
        "runtime": {
            "sample_dataset_verification": None,
            "dataset_verification": verification,
        }
    }
    analysis._validate_dataset_verification(result, spec)

    verification["logged_samples"]["composite_selected_docs_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="logged-sample verification changed"):
        analysis._validate_dataset_verification(result, spec)


def test_prospective_result_protocol_requires_frozen_analysis_identity():
    protocol, _ = analysis._load_json_object(analysis.DEFAULT_PROTOCOL, "protocol")
    spec = analysis._build_spec(protocol, analysis.DEFAULT_PROTOCOL, "stage1_mc")
    result_protocol = {
        "suite": "stage1_mc",
        "tasks": ["hellaswag", "winogrande"],
        "num_fewshot": 0,
        "apply_chat_template": False,
        "fewshot_as_multiturn": False,
        "system_instruction": None,
        "enable_thinking": False,
        "think_end_token": None,
        "max_gen_toks": 256,
        "generation": False,
        "prompt_format": "completion",
        "min_model_len": 4096,
        "analysis_exclude_doc_ids": [],
        "dataset_contract": None,
        "dataset_contracts": analysis._STAGE1_DATASET_CONTRACTS,
        "generation_kwargs": None,
        "limit": {"kind": "count", "value": 512, "from_suite_default": True},
        "sample_selection": None,
        "analysis_protocol": {
            "path": str(analysis.DEFAULT_PROTOCOL.resolve()),
            "sha256": analysis.EXPECTED_PROTOCOL_SHA256,
        },
        "batch_size": "auto",
        "bootstrap_iters": 0,
        "response_cache": None,
        "seeds": {"python": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},
        "engine_args": analysis._expected_engine_args(spec),
    }
    analysis._validate_result_protocol(result_protocol, spec)

    result_protocol["analysis_protocol"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="analysis protocol identity changed"):
        analysis._validate_result_protocol(result_protocol, spec)


def test_composite_document_hash_uses_leaf_task_field():
    leaves = tuple(
        analysis.LeafSpec(
            task=task,
            metrics=("acc",),
            primary_metric="acc",
            filter_name="none",
            expected_doc_ids=(0,),
            analysis_doc_ids=(0,),
            original_size=1,
        )
        for task in ("leaf_a", "leaf_b")
    )
    identities = {
        (leaf.task, 0, "none"): {
            "doc": {"value": leaf.task},
            "doc_hash": "0" * 64,
            "prompt_hash": "1" * 64,
            "target_hash": "2" * 64,
        }
        for leaf in leaves
    }
    records = [
        {
            "leaf_task": leaf.task,
            "doc_id": 0,
            "doc_sha256": hashlib.sha256(
                analysis._canonical_json({"value": leaf.task})
            ).hexdigest(),
        }
        for leaf in leaves
    ]
    expected = hashlib.sha256(analysis._canonical_json(records)).hexdigest()

    assert analysis._composite_docs_sha256(identities, leaves) == expected


def test_mmlu_group_leaf_alias_matches_lm_eval_print_alias():
    protocol, _ = analysis._load_json_object(analysis.DEFAULT_PROTOCOL, "protocol")
    spec = analysis._build_spec(
        protocol, analysis.DEFAULT_PROTOCOL, "stage2_mmlu_pro"
    )
    aliases = {
        leaf.task: analysis._expected_aggregate_alias(spec, leaf)
        for leaf in spec.leaves
    }
    assert aliases["mmlu_pro_biology"] == " - biology"
    assert aliases["mmlu_pro_computer_science"] == " - computer_science"


def test_zero_steady_state_cost_arms_have_closed_environment_contracts():
    selector = analysis._expected_environment("selector_alpha1")
    alpha = analysis._expected_environment("alpha_0960")

    assert selector["TENSORBRIDGE_NVFP4_FPMA_ALPHA"] == "1.0"
    assert selector["TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR"] == "normal_b8_sse"
    assert selector["TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR_CHUNK_ROWS"] == "256"
    assert alpha["TENSORBRIDGE_NVFP4_FPMA_ALPHA"] == "0.96"
    assert alpha["TENSORBRIDGE_NVFP4_PREFOLD_SELECTOR"] == "none"
    assert analysis._CACHE_SEEDS["selector_alpha1"] == analysis._DEFAULT_CACHE_SEED
    assert analysis._CACHE_SEEDS["alpha_0960"] == analysis._DEFAULT_CACHE_SEED


def test_post_confirmation_allows_pre_chunk_provenance_for_new_arms_only_there():
    stage = _tiny_spec()
    confirm = _tiny_spec(suite="confirm_mc", post_confirmation=True)
    for arm in ("selector_alpha1", "alpha_0960"):
        old_environment = analysis._expected_environment(
            arm, include_selector_chunk=False
        )
        assert old_environment not in analysis._allowed_environments(arm, stage)
        assert old_environment in analysis._allowed_environments(arm, confirm)


def test_task_configs_are_closed_after_callable_address_normalization():
    hellaswag = analysis._expected_hellaswag_config()
    assert len(hellaswag) == 20
    assert [item["metric"] for item in hellaswag["metric_list"]] == [
        "acc",
        "acc_norm",
    ]
    mmlu = analysis._expected_mmlu_config("mmlu_pro_computer_science")
    assert len(mmlu) == 21
    assert mmlu["task_alias"] == "computer_science"
    assert "about computer science" in mmlu["description"]

    actual = copy.deepcopy(mmlu)
    actual["doc_to_text"] = actual["doc_to_text"].replace("<ADDR>", "0x7ffabc")
    actual["fewshot_config"]["process_docs"] = actual["fewshot_config"][
        "process_docs"
    ].replace("<ADDR>", "0x1234")
    assert analysis._normalize_callable_addresses(actual) == mmlu

    actual["metric_list"][0]["ignore_case"] = False
    assert analysis._normalize_callable_addresses(actual) != mmlu


def test_generation_reader_keeps_invalid_format_incorrect(tmp_path):
    rows = [
        _sample_row(0, response="work\nThe answer is 1"),
        _sample_row(1, response="The answer is 1\ntrailing text"),
    ]
    sample_path = tmp_path / "samples.jsonl"
    artifact = _write_jsonl(sample_path, rows)
    leaf = analysis.LeafSpec(
        task="gsm",
        metrics=("exact_match",),
        primary_metric="exact_match",
        filter_name="final-answer",
        expected_doc_ids=(0, 1),
        analysis_doc_ids=(0, 1),
        original_size=2,
    )
    spec = analysis.StageSpec(
        suite="stage3_generation",
        requested_tasks=("gsm",),
        leaves=(leaf,),
        num_fewshot=0,
        max_model_len=4096,
        max_gen_toks=1024,
        generation=True,
        apply_chat_template=True,
        prompt_format="chat_nonthinking",
        limit_count=None,
        dataset_contracts=None,
        manifest_path=tmp_path / "manifest.json",
        manifest_sha256="0" * 64,
        manifest={},
        format_regex=analysis._GSM8K_FORMAT_REGEX,
        format_min_valid=1,
        post_confirmation=False,
    )

    _, correctness, raw_correctness, valid = analysis._read_sample_rows(
        sample_path, artifact, leaf, spec
    )
    first = ("gsm", 0, "final-answer")
    second = ("gsm", 1, "final-answer")
    assert raw_correctness[("gsm", "exact_match")] == {
        first: True,
        second: True,
    }
    assert correctness[("gsm", "exact_match")] == {
        first: True,
        second: False,
    }
    assert valid == {first: True, second: False}


def test_sample_reader_rejects_duplicate_pairing_keys(tmp_path):
    sample_path = tmp_path / "duplicate.jsonl"
    artifact = _write_jsonl(sample_path, [_sample_row(0), _sample_row(0)])
    artifact["unique_docs"] = 2
    leaf = analysis.LeafSpec(
        task="gsm",
        metrics=("exact_match",),
        primary_metric="exact_match",
        filter_name="final-answer",
        expected_doc_ids=(0, 0),
        analysis_doc_ids=(0, 0),
        original_size=2,
    )
    spec = _tiny_spec(suite="stage3_generation", generation=True)
    spec = analysis.StageSpec(
        **{
            **spec.__dict__,
            "leaves": (leaf,),
            "format_regex": analysis._GSM8K_FORMAT_REGEX,
            "format_min_valid": 1,
        }
    )
    with pytest.raises(ValueError, match="duplicate sample pairing key"):
        analysis._read_sample_rows(sample_path, artifact, leaf, spec)


def test_exact_mcnemar_and_pair_flip_keys_are_reported():
    keys = tuple(("leaf", doc_id, "none") for doc_id in range(4))
    baseline = dict(zip(keys, (True, True, False, False), strict=True))
    candidate = dict(zip(keys, (True, False, True, False), strict=True))

    first = analysis._paired_stats(
        baseline, candidate, keys, _analysis_protocol()
    )
    second = analysis._paired_stats(
        baseline, candidate, keys, _analysis_protocol()
    )

    assert first == second
    assert first["candidate_loss_flip_keys"] == [
        {"leaf_task": "leaf", "doc_id": 1, "filter": "none"}
    ]
    assert first["candidate_gain_flip_keys"] == [
        {"leaf_task": "leaf", "doc_id": 2, "filter": "none"}
    ]
    assert first["paired_accuracy_delta"] == 0.0
    assert first["exact_mcnemar"]["p_value"] == 1.0


def test_category_stratified_bootstrap_is_deterministic():
    deltas = {"a": [1, 0, -1, 0], "b": [0, 1, 0, 0]}
    first = analysis._stratified_bootstrap_ci(
        deltas, confidence_level=0.95, resamples=100, seed=20_260_718
    )
    second = analysis._stratified_bootstrap_ci(
        deltas, confidence_level=0.95, resamples=100, seed=20_260_718
    )

    assert first == second
    assert first["method"] == (
        "paired_stratified_nonparametric_bootstrap_percentile"
    )
    assert first["resampling_unit"] == "document_within_fixed_category"
    assert first["categories"] == 2


def test_screening_uses_point_estimates_and_not_confidence_interval():
    spec = _tiny_spec()
    leaf = spec.leaves[0]
    keys = leaf.analysis_keys
    normal_values = dict(zip(keys, (True, True, False, False), strict=True))
    default_values = dict(zip(keys, (True, True, True, False), strict=True))
    runs = {
        "normal_a8": SimpleNamespace(
            correctness={(leaf.task, "acc"): normal_values}
        ),
        "fpma_default": SimpleNamespace(
            correctness={(leaf.task, "acc"): default_values}
        ),
    }
    summary = analysis._comparison_summary(
        runs=runs,
        arm_summaries={"normal_a8": {}, "fpma_default": {}},
        spec=spec,
        analysis_protocol=_analysis_protocol(),
        candidate="fpma_default",
        baseline="normal_a8",
    )

    screening = summary["screening_vs_normal_a8"]
    assert screening["screening_check_passed"] is True
    assert screening["confidence_intervals_are_reported_but_not_a_gate"] is True


def _source_identity(seed):
    digest = f"{seed:064x}"
    git = {
        "available": True,
        "root": "/repo",
        "head": "1" * 40,
        "dirty": True,
        "status_sha256": digest,
        "tracked_diff_sha256": "2" * 64,
    }
    return {
        "tensorbridge_git": git,
        "tensorbridge_tree": {"sha256": "3" * 64, "files": 10},
        "vllm_git": {
            **git,
            "root": "/vllm",
            "status_sha256": "4" * 64,
        },
    }


def test_source_policy_is_exact_for_stages_and_cohorted_for_confirmation():
    base = _source_identity(1)
    runs = {
        arm: SimpleNamespace(source_identity=copy.deepcopy(base))
        for arm in analysis.EXPECTED_ARMS
    }
    runs["normal_a8"].source_identity["tensorbridge_git"]["status_sha256"] = "5" * 64
    with pytest.raises(ValueError, match="source identity differs"):
        analysis._validate_cross_arm_source(runs, _tiny_spec())

    confirm_spec = _tiny_spec(suite="confirm_mc", post_confirmation=True)
    summary = analysis._validate_cross_arm_source(runs, confirm_spec)
    assert summary["reused_v1_status_sha256_may_differ"] is True
    assert summary["new_arm_exact_match"] is True
    assert summary["cross_cohort_tree_head_tracked_diff_and_vllm_match"] is True

    runs["alpha_0960"].source_identity["tensorbridge_tree"]["files"] = 11
    with pytest.raises(ValueError, match="new post-confirmation arms"):
        analysis._validate_cross_arm_source(runs, confirm_spec)

    runs["alpha_0960"].source_identity = copy.deepcopy(
        runs["selector_alpha1"].source_identity
    )
    for arm in ("selector_alpha1", "alpha_0960"):
        runs[arm].source_identity["tensorbridge_tree"]["sha256"] = "6" * 64
    with pytest.raises(ValueError, match="different executable sources"):
        analysis._validate_cross_arm_source(runs, confirm_spec)


def test_analysis_requires_exactly_six_result_files():
    with pytest.raises(ValueError, match="exactly six"):
        analysis.analyze_stages([])


def test_output_writer_protects_every_input_kind(tmp_path):
    protected = tmp_path / "manifest.json"
    protected.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite an input"):
        analysis._write_output(protected, "{}\n", True, {protected.resolve()})

    model_root = tmp_path / "model"
    model_root.mkdir()
    with pytest.raises(ValueError, match="checkpoint model tree"):
        analysis._write_output(
            model_root / "config.json",
            "{}\n",
            True,
            set(),
            {model_root.resolve()},
        )
