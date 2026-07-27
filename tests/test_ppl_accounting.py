import argparse
import copy
import json
import math
import os
import sys
from types import SimpleNamespace

import pytest

import scripts.eval_nvfp4_wikitext2_ppl as ppl_eval
from scripts.eval_nvfp4_wikitext2_ppl import (
    _assigned_gpu_devices,
    _bootstrap_scalar_sd_ratio,
    _bootstrap_vector_rms_noise_ratio,
    _compare_execution_results,
    _configure_fpma_alpha_environment,
    _cudagraph_compilation_config,
    _cudagraph_runtime_gate,
    _cudagraph_runtime_stats,
    _memory_stability_gate,
    _positive_int_csv,
    _preflight_eager_reference,
    _reset_cudagraph_runtime_stats,
    _resolve_fpma_alpha,
    _resolve_fpma_alpha_input,
    _trace_difference,
    _validation_trace,
    _welch_equivalence,
    ExecutionInconclusiveError,
    ExecutionValidationError,
)
from vllm.plugins.tensorbridge_evaluation.ppl import (
    NonFiniteLogprobError,
    build_prompt_blocks,
    prompt_token_capacity,
    score_prompt_logprobs,
)


def test_ppl_fpma_alpha_default_is_backend_aware():
    assert _resolve_fpma_alpha("tensorbridge", None) == 0.961
    assert _resolve_fpma_alpha("official", None) == 1.0
    assert _resolve_fpma_alpha("normal_a8", None) == 1.0
    assert _resolve_fpma_alpha("tensorbridge", None, ulp_correction=True) == 1.0
    assert (
        _resolve_fpma_alpha(
            "tensorbridge",
            None,
            prefold_selector="normal_b8_sse",
        )
        == 1.0
    )
    assert _resolve_fpma_alpha("tensorbridge", 0.960) == 0.960


def test_ppl_implicit_alpha_preserves_analytic_domain_gate(monkeypatch):
    monkeypatch.setenv("TENSORBRIDGE_NVFP4_FPMA_ALPHA", "0.5")
    _configure_fpma_alpha_environment(0.961, "analytic_v1")
    assert "TENSORBRIDGE_NVFP4_FPMA_ALPHA" not in os.environ

    _configure_fpma_alpha_environment(0.960, "explicit_cli")
    assert os.environ["TENSORBRIDGE_NVFP4_FPMA_ALPHA"] == "0.96"


def test_ppl_explicit_alpha_precedence_and_provenance():
    alpha, source = _resolve_fpma_alpha_input(
        "tensorbridge",
        None,
        env_alpha="0.975",
    )
    assert (alpha, source) == (0.975, "explicit_env")

    alpha, source = _resolve_fpma_alpha_input(
        "tensorbridge",
        0.960,
        env_alpha="0.975",
    )
    assert (alpha, source) == (0.960, "explicit_cli")


def test_cudagraph_capture_sizes_parser_is_strict():
    assert _positive_int_csv("1,2,4,8,32") == [1, 2, 4, 8, 32]
    for invalid in ("", "0,1", "1,-2", "1,1", "1,nope", "1,,2", "1,2,"):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int_csv(invalid)


def test_cudagraph_config_can_disable_optional_allreduce_rms_fusion():
    args = SimpleNamespace(
        execution_mode="cudagraph",
        compilation_mode="NONE",
        cudagraph_mode="FULL_DECODE_ONLY",
        cudagraph_capture_sizes=[1],
        disable_allreduce_rms_fusion=True,
    )
    config = _cudagraph_compilation_config(args)
    assert config == {
        "mode": "NONE",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_num_of_warmups": 1,
        "cudagraph_capture_sizes": [1],
        "pass_config": {"fuse_allreduce_rms": False},
    }

    args.execution_mode = "eager"
    assert _cudagraph_compilation_config(args) == {
        "pass_config": {"fuse_allreduce_rms": False}
    }
    args.disable_allreduce_rms_fusion = False
    assert _cudagraph_compilation_config(args) is None


def test_assigned_gpu_devices_prefers_job_visible_numbering():
    assert (
        _assigned_gpu_devices(
            {
                "CUDA_VISIBLE_DEVICES": "0,1",
                "SLURM_STEP_GPUS": "4,5",
                "SLURM_JOB_GPUS": "4,5",
            }
        )
        == "0,1"
    )
    assert _assigned_gpu_devices({"SLURM_JOB_GPUS": "4"}) == "4"
    assert _assigned_gpu_devices({}) is None


class _Logprob:
    def __init__(self, logprob):
        self.logprob = logprob


def _outputs_for(blocks, logprobs, generated=None):
    outputs = []
    offset = 0
    if generated is None:
        generated = [[] for _ in blocks]
    for block, request_completions in zip(blocks, generated, strict=True):
        prompt = block.prompt_token_ids
        values = [None] * len(prompt)
        for position in range(block.local_target_start, len(prompt)):
            values[position] = {prompt[position]: _Logprob(logprobs[offset])}
            offset += 1
        completions = []
        for token_ids in request_completions:
            completions.append(
                SimpleNamespace(
                    token_ids=token_ids,
                    logprobs=[
                        {token: _Logprob(-0.01 * (index + 1))}
                        for index, token in enumerate(token_ids)
                    ],
                )
            )
        outputs.append(
            SimpleNamespace(
                prompt_token_ids=prompt,
                prompt_logprobs=values,
                outputs=completions,
            )
        )
    assert offset == len(logprobs)
    return outputs


def test_validation_trace_is_stable_and_covers_scored_targets():
    blocks = build_prompt_blocks(
        list(range(10)),
        max_model_len=6,
        target_tokens_per_block=3,
    )
    values = [-0.25 * (index + 1) for index in range(9)]
    first = _validation_trace(blocks, _outputs_for(blocks, values))
    second = _validation_trace(blocks, _outputs_for(blocks, values))

    assert first["target_logprobs"] == values
    assert first["generated_token_ids"] == [[] for _ in blocks]
    assert first["generated_token_logprobs"] == [[] for _ in blocks]
    assert [item["global_token_offset"] for item in first["targets"]] == list(
        range(1, 10)
    )
    assert first["sha256"] == second["sha256"]

    changed = values.copy()
    changed[-1] -= 0.125
    third = _validation_trace(blocks, _outputs_for(blocks, changed))
    assert third["sha256"] != first["sha256"]
    difference = _trace_difference(first, third)
    assert difference["max_abs_target_logprob"] == pytest.approx(0.125)
    assert difference["generated_token_ids_exact"]


def test_validation_trace_frames_generated_tokens_and_completion_boundaries():
    blocks = build_prompt_blocks([1, 2], max_model_len=2, target_tokens_per_block=1)
    values = [-0.5]
    split = _validation_trace(
        blocks,
        _outputs_for(blocks, values, generated=[[[11], [12]]]),
    )
    joined = _validation_trace(
        blocks,
        _outputs_for(blocks, values, generated=[[[11, 12], []]]),
    )
    changed = _validation_trace(
        blocks,
        _outputs_for(blocks, values, generated=[[[11], [13]]]),
    )
    changed_score_outputs = _outputs_for(
        blocks, values, generated=[[[11], [12]]]
    )
    changed_score_outputs[0].outputs[0].logprobs[0][11] = _Logprob(-0.02)
    changed_score = _validation_trace(blocks, changed_score_outputs)

    assert split["generated_token_ids"] == [[[11], [12]]]
    assert split["generated_token_logprobs"] == [[[-0.01], [-0.01]]]
    assert split["sha256"] != joined["sha256"]
    assert split["sha256"] != changed["sha256"]
    assert not _trace_difference(split, changed)["generated_token_ids_exact"]
    score_difference = _trace_difference(split, changed_score)
    assert score_difference["generated_token_ids_exact"]
    assert not score_difference["generated_token_logprobs_exact"]
    assert score_difference["max_abs_generated_token_logprob"] == pytest.approx(0.01)


def test_cudagraph_runtime_stats_are_consumed_from_vllm_logger_shape():
    stats = [
        SimpleNamespace(
            num_unpadded_tokens=1,
            num_padded_tokens=1,
            num_paddings=0,
            runtime_mode="FULL",
        ),
        SimpleNamespace(
            num_unpadded_tokens=31,
            num_padded_tokens=32,
            num_paddings=1,
            runtime_mode="NONE",
        ),
    ]

    class _Logger:
        def __init__(self):
            self.stats = stats.copy()

        def reset(self):
            self.stats = []

    logger = _Logger()
    aggregate = SimpleNamespace(
        per_engine_stat_loggers={0: SimpleNamespace(cudagraph_logging=logger)}
    )
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            logger_manager=SimpleNamespace(stat_loggers=[aggregate])
        )
    )

    summary = _cudagraph_runtime_stats(llm)
    assert summary["runtime_mode_counts"] == {"FULL": 1, "NONE": 1}
    assert summary["graph_dispatches"] == 1
    _reset_cudagraph_runtime_stats(llm)
    assert logger.stats == []


def test_full_decode_only_runtime_gate_requires_one_eager_prefill_and_full_decode():
    records = [
        {
            "runtime_mode": "NONE",
            "num_unpadded_tokens": 32,
            "num_padded_tokens": 32,
            "num_paddings": 0,
        }
    ] + [
        {
            "runtime_mode": "FULL",
            "num_unpadded_tokens": 1,
            "num_padded_tokens": 1,
            "num_paddings": 0,
        }
        for _ in range(7)
    ]
    stats = {"records": records, "graph_dispatches": 7}
    gate = _cudagraph_runtime_gate(
        stats,
        cudagraph_mode="FULL_DECODE_ONLY",
        expected_decode_dispatches=7,
        expected_decode_batch=1,
    )
    assert gate["passed"]

    stats["records"][3]["runtime_mode"] = "NONE"
    gate = _cudagraph_runtime_gate(
        stats,
        cudagraph_mode="FULL_DECODE_ONLY",
        expected_decode_dispatches=7,
        expected_decode_batch=1,
    )
    assert not gate["passed"]


@pytest.mark.parametrize(
    ("record_index", "field", "value"),
    [
        (1, "runtime_mode", "PIECEWISE"),
        (1, "num_unpadded_tokens", 2),
        (1, "num_padded_tokens", 2),
        (1, "num_paddings", 1),
    ],
)
def test_full_decode_only_runtime_gate_rejects_decode_fallback_or_padding(
    record_index, field, value
):
    records = [
        {
            "runtime_mode": "NONE",
            "num_unpadded_tokens": 32,
            "num_padded_tokens": 32,
            "num_paddings": 0,
        }
    ] + [
        {
            "runtime_mode": "FULL",
            "num_unpadded_tokens": 1,
            "num_padded_tokens": 1,
            "num_paddings": 0,
        }
        for _ in range(7)
    ]
    records[record_index][field] = value
    gate = _cudagraph_runtime_gate(
        {"records": records, "graph_dispatches": 7},
        cudagraph_mode="FULL_DECODE_ONLY",
        expected_decode_dispatches=7,
        expected_decode_batch=1,
    )
    assert not gate["passed"]


def _memory_snapshot(first, second):
    return [
        {"index": 0, "uuid": "GPU-a", "memory_used_mib": first},
        {"index": 1, "uuid": "GPU-b", "memory_used_mib": second},
    ]


def test_cudagraph_memory_gate_excludes_warmup_and_rejects_continual_growth():
    stable = [
        _memory_snapshot(80, 90),
        _memory_snapshot(100, 110),
        _memory_snapshot(120, 130),
        _memory_snapshot(120, 130),
        _memory_snapshot(120, 130),
    ]
    assert _memory_stability_gate(stable)["passed"]

    growing = stable[:2] + [
        _memory_snapshot(120, 130),
        _memory_snapshot(121, 130),
        _memory_snapshot(122, 130),
    ]
    gate = _memory_stability_gate(growing)
    assert not gate["passed"]
    assert gate["per_gpu"][0]["strictly_increasing"]


def _execution_result(mode, logprob):
    mean_nll = -logprob
    trace = {
        "targets": [
            {
                "block_index": 0,
                "global_token_offset": 1,
                "target_token_id": 2,
                "logprob": logprob,
            }
        ],
        "generated_token_ids": [[[7, 8, 9, 10, 11, 12, 13, 14]]],
        "generated_token_logprobs": [
            [[-0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]]
        ],
        "metrics": {"mean_nll": mean_nll, "ppl": math.exp(mean_nll)},
    }
    runs = []
    for index in range(12):
        run = copy.deepcopy(trace)
        run["repeat_index"] = index
        run["output_sha256"] = f"sha-{index}"
        runs.append(run)
    return {
        "schema_version": 2,
        "status": "passed",
        "checkpoint_mode": "tensorbridge",
        "model_path": "/model",
        "runtime": {"vllm": "0.20.2", "quant_config_class": "TensorBridge"},
        "production_contract": {"fpma_global_scale_alpha": 0.961},
        "dataset": {"name": "wikitext"},
        "blocking": {
            "selected_block_start": 0,
            "selected_block_stop": 1,
            "requested_output_tokens": 8,
        },
        "engine_args": {
            "tensor_parallel_size": 1,
            "max_model_len": 256,
            "max_num_seqs": 1,
            "gpu_memory_utilization": 0.5,
        },
        "execution_validation": {
            "requested_mode": mode,
            "execution_trace_schema_version": 3,
            "sampling_logprob_contract": {
                "prompt_logprobs": 1,
                "generated_logprobs": 1,
                "flat_logprobs": False,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "repeat_runs": len(runs),
            "primary_run_index": len(runs) - 1,
            "runs": runs,
        },
        "metrics": {"mean_nll": mean_nll, "ppl": math.exp(mean_nll)},
    }


def test_eager_cudagraph_comparison_uses_aligned_token_level_gates():
    eager = _execution_result("eager", -0.5)
    graph = _execution_result("cudagraph", -0.5000001)
    comparison = _compare_execution_results(eager, graph)
    assert comparison["passed"]
    assert comparison["generated_token_ids_exact"]

    for run in graph["execution_validation"]["runs"]:
        run["metrics"]["mean_nll"] = 0.54
        run["metrics"]["ppl"] = math.exp(0.54)
    comparison = _compare_execution_results(eager, graph)
    assert not comparison["passed"]
    assert comparison["equivalence"]["prompt_target_nll"]["decision"] == "failed"


def test_execution_comparison_rejects_decode_shift_and_token_change():
    eager = _execution_result("eager", -0.5)
    graph = _execution_result("cudagraph", -0.5)
    for run in graph["execution_validation"]["runs"]:
        run["generated_token_logprobs"] = [
            [[-0.2, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5]]
        ]
    comparison = _compare_execution_results(eager, graph)
    assert not comparison["passed"]
    assert comparison["equivalence"]["generated_decode_positions1plus_nll"][
        "decision"
    ] == "failed"

    graph = _execution_result("cudagraph", -0.5)
    graph["execution_validation"]["runs"][-1]["generated_token_ids"] = [
        [[7, 8, 9, 10, 11, 12, 13, 15]]
    ]
    comparison = _compare_execution_results(eager, graph)
    assert not comparison["passed"]
    assert not comparison["generated_token_ids_exact"]


def test_execution_comparison_detects_canceling_decode_vector_shift():
    eager = _execution_result("eager", -0.5)
    graph = _execution_result("cudagraph", -0.5)
    signed_shifts = [0.03, -0.03, 0.03, -0.03, 0.03, -0.03, 0.0]
    for run in graph["execution_validation"]["runs"]:
        values = run["generated_token_logprobs"][0][0]
        run["generated_token_logprobs"][0][0] = [
            values[0],
            *(value + shift for value, shift in zip(values[1:], signed_shifts)),
        ]

    comparison = _compare_execution_results(eager, graph)
    assert comparison["equivalence"]["generated_decode_positions1plus_nll"][
        "passed"
    ]
    assert not comparison["generated_decode_mode_effect"]["passed"]
    assert comparison["generated_decode_mode_effect"]["rms_p_value"] < 0.01
    assert comparison["decision"] == "failed"


def test_execution_comparison_treats_prefill_mismatch_as_inconclusive():
    eager = _execution_result("eager", -0.5)
    graph = _execution_result("cudagraph", -0.5)
    for run in graph["execution_validation"]["runs"]:
        run["generated_token_logprobs"][0][0][0] = -0.5
    comparison = _compare_execution_results(eager, graph)
    assert comparison["decision"] == "inconclusive"
    assert not comparison["passed"]


def test_execution_comparison_excludes_two_warmup_runs():
    eager = _execution_result("eager", -0.5)
    graph = _execution_result("cudagraph", -0.5)
    for run in graph["execution_validation"]["runs"][:2]:
        run["metrics"]["mean_nll"] = 100.0
        run["generated_token_logprobs"][0][0] = [-100.0] * 8
    assert _compare_execution_results(eager, graph)["passed"]


def test_welch_equivalence_reports_pass_inconclusive_and_fail():
    assert _welch_equivalence([1.0] * 10, [1.0] * 10)["decision"] == "passed"
    assert _welch_equivalence([1.0] * 10, [1.019] * 10)["decision"] == "passed"
    assert _welch_equivalence([1.0] * 10, [0.981] * 10)["decision"] == "passed"
    assert _welch_equivalence([1.0] * 10, [1.020] * 10)["decision"] == "failed"
    assert _welch_equivalence([1.0] * 10, [0.980] * 10)["decision"] == "failed"
    noisy = [0.95, 1.05] * 5
    assert _welch_equivalence(noisy, noisy)["decision"] == "inconclusive"
    assert _welch_equivalence([1.0] * 10, [1.1] * 10)["decision"] == "failed"


def test_bootstrap_noise_gate_does_not_drop_unbounded_zero_reference_ratios():
    scalar = _bootstrap_scalar_sd_ratio([1.0] * 10, [1.0, 1.1] * 5, seed=1)
    vector = _bootstrap_vector_rms_noise_ratio(
        [[1.0, 2.0]] * 10,
        [[1.0, 2.0], [1.1, 2.2]] * 5,
        seed=2,
    )

    for result in (scalar, vector):
        assert result["point_ratio"] is None
        assert result["upper90"] is None
        assert result["invalid_unbounded_resamples"] > 0
        assert not result["passed"]


def test_main_preserves_inconclusive_status_and_json_safe_diagnostics(
    monkeypatch, tmp_path
):
    output = tmp_path / "inconclusive.json"

    def raise_inconclusive(_args):
        raise ExecutionInconclusiveError("need more runs", {"upper_bound": math.inf})

    monkeypatch.setattr(ppl_eval, "_run", raise_inconclusive)
    monkeypatch.setattr(sys, "argv", ["eval", "--output-json", str(output)])
    with pytest.raises(ExecutionInconclusiveError, match="need more runs"):
        ppl_eval.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "inconclusive"
    assert payload["diagnostics"]["upper_bound"] is None


def test_eager_reference_preflight_rejects_old_trace_before_execution(tmp_path):
    reference = _execution_result("eager", -0.5)
    assert _preflight_eager_reference(reference, tmp_path / "current.json") is reference

    old = copy.deepcopy(reference)
    old["schema_version"] = 1
    old["execution_validation"].pop("execution_trace_schema_version")
    for run in old["execution_validation"]["runs"]:
        run.pop("generated_token_logprobs")
    with pytest.raises(ExecutionValidationError, match="trace v3") as error:
        _preflight_eager_reference(old, tmp_path / "old.json")
    assert error.value.diagnostics["preflight_errors"]


def test_prompt_blocks_cover_every_target_once_with_overlap():
    tokens = list(range(13))
    blocks = build_prompt_blocks(
        tokens,
        max_model_len=6,
        target_tokens_per_block=3,
    )

    assert [(block.global_target_start, block.global_target_end) for block in blocks] == [
        (1, 4),
        (4, 7),
        (7, 10),
        (10, 13),
    ]
    assert sum(block.scored_tokens for block in blocks) == len(tokens) - 1
    assert all(len(block.prompt_token_ids) <= 6 for block in blocks)
    assert all(block.local_target_start >= 1 for block in blocks)


def test_max_blocks_limits_scored_targets_not_context():
    blocks = build_prompt_blocks(
        list(range(30)),
        max_model_len=8,
        target_tokens_per_block=4,
        max_blocks=2,
    )
    assert len(blocks) == 2
    assert sum(block.scored_tokens for block in blocks) == 8


def test_prompt_capacity_reserves_vllm_generation_token():
    engine_limit = 2048
    output_tokens = 1
    prompt_limit = prompt_token_capacity(engine_limit, output_tokens)
    blocks = build_prompt_blocks(
        list(range(5000)),
        max_model_len=prompt_limit,
        target_tokens_per_block=1024,
        max_blocks=4,
    )

    assert prompt_limit == 2047
    assert len(blocks) == 4
    assert all(len(block.prompt_token_ids) + output_tokens <= engine_limit for block in blocks)


def test_prompt_logprob_score_uses_only_target_tokens():
    blocks = build_prompt_blocks(
        list(range(5)),
        max_model_len=4,
        target_tokens_per_block=2,
    )
    outputs = _outputs_for(blocks, [math.log(0.5), math.log(0.25)] * 2)
    metrics = score_prompt_logprobs(blocks, outputs)

    assert metrics["scored_tokens"] == 4
    assert metrics["ppl"] == pytest.approx(math.sqrt(8.0))


def test_prompt_logprob_score_rejects_missing_target_and_prompt_mismatch():
    blocks = build_prompt_blocks([1, 2, 3], max_model_len=3, target_tokens_per_block=2)
    outputs = _outputs_for(blocks, [0.0, 0.0])
    outputs[0].prompt_logprobs[1] = {999: _Logprob(0.0)}
    with pytest.raises(KeyError, match="target token"):
        score_prompt_logprobs(blocks, outputs)

    outputs = _outputs_for(blocks, [0.0, 0.0])
    outputs[0].prompt_token_ids = [9, 2, 3]
    with pytest.raises(ValueError, match="prompt token mismatch"):
        score_prompt_logprobs(blocks, outputs)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_prompt_logprob_score_rejects_nonfinite_values(bad):
    blocks = build_prompt_blocks([1, 2], max_model_len=2, target_tokens_per_block=1)
    with pytest.raises(NonFiniteLogprobError, match="non-finite") as error:
        score_prompt_logprobs(blocks, _outputs_for(blocks, [bad]))
    assert error.value.total_nonfinite == 1
    assert error.value.diagnostics[0]["block_index"] == 0
    assert error.value.diagnostics[0]["global_token_offset"] == 1
