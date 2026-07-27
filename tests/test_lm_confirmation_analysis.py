import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import analyze_nvfp4_lm_confirmation as analysis


def _sample_row(doc_id, *, exact_match=1.0, response="The answer is 1"):
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


def _generation_spec(count):
    doc_ids = tuple(range(count))
    return analysis.ConfirmationSpec(
        suite="confirm_generation",
        task="tensorbridge_gsm8k_relative_smoke",
        benchmark="GSM8K",
        metrics=("exact_match",),
        primary_metric="exact_match",
        filter_name="final-answer",
        expected_doc_ids=doc_ids,
        analysis_doc_ids=doc_ids,
        excluded_doc_ids=(),
        dataset_contract={
            "path": "openai/gsm8k",
            "name": "main",
            "split": "test",
            "size": count,
            "revision": "test-revision",
        },
        sample_manifest_path=None,
        sample_manifest_sha256=None,
        sample_selection=None,
        format_regex=(
            r"(?m)^The answer is (-?[0-9][0-9,]*(?:\.[0-9]+)?)\.?\s*\Z"
        ),
        format_min_valid=max(0, count - 1),
        noninferiority_margin=-0.05,
    )


def test_preregistered_protocol_builds_both_confirmation_specs():
    protocol, _ = analysis._load_json_object(analysis.DEFAULT_PROTOCOL, "protocol")
    assert hashlib.sha256(analysis.DEFAULT_PROTOCOL.read_bytes()).hexdigest() == (
        analysis.EXPECTED_PROTOCOL_SHA256
    )

    arc = analysis._build_spec(protocol, analysis.DEFAULT_PROTOCOL, "confirm_mc")
    gsm = analysis._build_spec(
        protocol, analysis.DEFAULT_PROTOCOL, "confirm_generation"
    )

    assert arc.primary_metric == "acc_norm"
    assert arc.metrics == ("acc_norm", "acc")
    assert len(arc.expected_doc_ids) == 1172
    assert arc.analysis_doc_ids == tuple(range(16, 1172))
    assert gsm.primary_metric == "exact_match"
    assert len(gsm.expected_doc_ids) == 128
    assert analysis._ids_sha256(gsm.expected_doc_ids) == (
        "a43574fc29a99293c793b08c17a02b720cd4f9487e9fd33ef299515903924fc2"
    )

    changed = copy.deepcopy(protocol)
    del changed["comparisons"]["exploratory_ulp_vs_default"]
    with pytest.raises(ValueError, match="comparison set changed"):
        analysis._build_spec(changed, analysis.DEFAULT_PROTOCOL, "confirm_mc")

    changed = copy.deepcopy(protocol)
    changed["noninferiority_margins"]["arc_challenge_acc_norm_vs_normal_a8"] = -0.03
    with pytest.raises(ValueError, match="margin changed"):
        analysis._build_spec(changed, analysis.DEFAULT_PROTOCOL, "confirm_mc")


def test_generation_format_is_anchored_and_invalid_format_is_incorrect(tmp_path):
    rows = [
        _sample_row(0, response="work\nThe answer is 1"),
        _sample_row(1, response="The answer is 1\ntrailing text"),
    ]
    sample_path = tmp_path / "samples.jsonl"
    artifact = _write_jsonl(sample_path, rows)

    _, correctness, raw_correctness, format_valid = analysis._read_sample_rows(
        sample_path, artifact, _generation_spec(2)
    )

    assert raw_correctness["exact_match"] == {0: True, 1: True}
    assert format_valid == {0: True, 1: False}
    assert correctness["exact_match"] == {0: True, 1: False}


def test_sample_reader_rejects_duplicate_doc_ids(tmp_path):
    rows = [_sample_row(0), _sample_row(0)]
    sample_path = tmp_path / "duplicate.jsonl"
    artifact = _write_jsonl(sample_path, rows)
    artifact["unique_docs"] = 2

    with pytest.raises(ValueError, match="duplicate sample doc_id"):
        analysis._read_sample_rows(sample_path, artifact, _generation_spec(2))


def test_lm_eval_task_config_is_closed():
    spec = _generation_spec(2)
    task = spec.task
    raw_correctness = {"exact_match": {0: True, 1: True}}
    result = {
        "lm_eval": {
            "results": {task: {"exact_match,final-answer": 1.0}},
            "configs": {task: analysis._expected_task_config(spec)},
            "n-samples": {task: {"original": 2, "effective": 2}},
        }
    }
    analysis._validate_lm_eval(result, spec, raw_correctness)

    result["lm_eval"]["configs"][task]["process_results"] = "changed"
    with pytest.raises(ValueError, match="config keys changed"):
        analysis._validate_lm_eval(result, spec, raw_correctness)


def test_exact_mcnemar_and_paired_bootstrap_are_deterministic():
    assert analysis._exact_mcnemar(2, 0)["p_value"] == 0.5
    assert analysis._exact_mcnemar(0, 0)["p_value"] == 1.0
    protocol = {
        "noninferiority_decision": (
            "paired_point_estimate_delta_greater_than_or_equal_to_margin"
        ),
        "paired_confidence_interval": {
            "method": "paired_nonparametric_bootstrap_percentile",
            "confidence_level": 0.95,
            "resamples": 10_000,
            "seed": 20_260_717,
        },
    }
    baseline = {0: True, 1: True, 2: False, 3: False}
    candidate = {0: True, 1: False, 2: True, 3: False}

    first = analysis._paired_stats(baseline, candidate, tuple(range(4)), protocol)
    second = analysis._paired_stats(baseline, candidate, tuple(range(4)), protocol)

    assert first == second
    assert first["candidate_loss_flips"] == 1
    assert first["candidate_gain_flips"] == 1
    assert first["candidate_loss_flip_doc_ids"] == [1]
    assert first["candidate_gain_flip_doc_ids"] == [2]
    assert first["paired_accuracy_delta"] == 0.0
    assert first["exact_mcnemar"]["p_value"] == 1.0


def test_analysis_requires_exactly_four_result_files():
    with pytest.raises(ValueError, match="exactly four"):
        analysis.analyze_confirmation([])


def test_primary_gate_requires_baseline_and_candidate_format():
    spec = _generation_spec(2)
    protocol = {
        "noninferiority_decision": (
            "paired_point_estimate_delta_greater_than_or_equal_to_margin"
        ),
        "paired_confidence_interval": {
            "confidence_level": 0.95,
            "resamples": 20,
            "seed": 20_260_717,
        },
    }
    runs = {
        "normal_a8": SimpleNamespace(
            correctness={"exact_match": {0: True, 1: False}},
            format_valid={0: False, 1: False},
        ),
        "fpma_default": SimpleNamespace(
            correctness={"exact_match": {0: True, 1: True}},
            format_valid={0: True, 1: True},
        ),
    }
    arm_summaries = {
        "normal_a8": {
            "format": {
                "valid": 0,
                "total": 2,
                "minimum_valid": 1,
                "gate_passed": False,
            }
        },
        "fpma_default": {
            "format": {
                "valid": 2,
                "total": 2,
                "minimum_valid": 1,
                "gate_passed": True,
            }
        },
    }
    summary = analysis._comparison_summary(
        runs=runs,
        arm_summaries=arm_summaries,
        spec=spec,
        analysis_protocol=protocol,
        candidate="fpma_default",
        baseline="normal_a8",
        role="primary",
        noninferiority_check=True,
        required_gate=True,
    )
    assert summary["metrics"]["exact_match"]["noninferiority"][
        "point_estimate_gate_passed"
    ] is True
    assert summary["format_gate"]["both_passed"] is False
    assert summary["all_required_gates_passed"] is False


def test_output_writer_protects_inputs(tmp_path):
    protected = tmp_path / "samples.jsonl"
    protected.write_text("raw\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite an input"):
        analysis._write_output(protected, "{}\n", True, {protected.resolve()})
