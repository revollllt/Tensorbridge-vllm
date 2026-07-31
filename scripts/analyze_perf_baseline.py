#!/usr/bin/env python3
"""Turn same-node perf result JSON into a defensible speedup report.

The analyzer refuses to emit a ratio it cannot defend. It first measures the
harness noise floor from repeated runs of an identical configuration, then
reports every cross-configuration ratio next to that floor. A ratio inside the
floor is labelled unreportable rather than published.

Job 416484 is the reason this exists: two runs of the `alpha` arm, which differ
by one FP32 constant and share every device instruction, came out 1.92x apart.
Any effect smaller than that spread was indistinguishable from the environment.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

# Fields that must agree across every run before ratios mean anything. A
# difference here is a confounder, not a performance result.
_INVARIANT_RUNTIME_FIELDS = (
    "hostname",
    "cpu_thread_limits",
    "slurm_cpus_per_task",
    "visible_gpu_count",
    "torch",
    "vllm",
    "tensorbridge_compiler",
)
_INVARIANT_BLOCKING_FIELDS = (
    "max_model_len",
    "max_prompt_tokens",
    "requested_output_tokens",
    "target_tokens_per_block",
    "selected_num_blocks",
    "selected_block_start",
    "expected_scored_tokens",
)
_INVARIANT_ENGINE_FIELDS = (
    "max_num_seqs",
    "tensor_parallel_size",
    "gpu_memory_utilization",
    "enforce_eager",
    "dtype",
    "quantization",
)


class FairnessError(RuntimeError):
    """A confounder makes the requested comparison indefensible."""


def _group_key(result: dict[str, Any]) -> str:
    contract = result["production_contract"]
    backend = contract["transformer_nvfp4_backend"]
    alpha = contract["fpma_global_scale_alpha"]
    selector = contract["fpma_prefold_selector"]
    ulp = contract["fpma_ulp_correction"]
    if backend != "tensorbridge":
        return backend
    parts = [backend]
    if alpha != 1.0:
        parts.append(f"alpha{alpha}")
    if selector not in (None, "none"):
        parts.append(f"sel_{selector}")
    if ulp:
        parts.append("ulp")
    return "+".join(parts) if len(parts) > 1 else f"{backend}_default"


def _repeat_seconds(result: dict[str, Any], warmup: int) -> list[float]:
    runs = result["execution_validation"]["runs"]
    measured = runs[warmup:]
    if not measured:
        raise FairnessError(
            f"warmup={warmup} consumed all {len(runs)} repeats; lower --warmup-repeats"
        )
    return [float(run["generation_seconds"]) for run in measured]


def load_runs(paths: list[Path], warmup: int) -> list[dict[str, Any]]:
    """Read measured (non-prime) results and attach their timing samples."""
    records = []
    for path in sorted(paths):
        result = json.loads(path.read_text())
        context = result.get("benchmark_context") or {}
        phase = context.get("phase")
        if phase == "prime":
            continue
        if result.get("status") == "failed":
            # A run that aborted writes an error dump instead of a result. It
            # carries no timing, and silently dropping it would understate the
            # cohort, so surface it and let the caller decide.
            raise FairnessError(
                f"{path.name} recorded a failed run "
                f"({result.get('error_type')}: {result.get('error')}); "
                "exclude it explicitly or re-run that arm"
            )
        samples = _repeat_seconds(result, warmup)
        records.append(
            {
                "path": str(path),
                "phase": phase,
                # Position in the interleaved order. Mirror-pairing needs it, so
                # a result without one cannot participate in paired ratios.
                "index": int(context["index"]) if context.get("index") is not None else None,
                "group": _group_key(result),
                "backend": result["production_contract"]["transformer_nvfp4_backend"],
                "hostname": result["runtime"]["hostname"],
                "gpu_clock_pin_status": result["runtime"].get("gpu_clock_pin_status"),
                "slurm_job_id": result["runtime"].get("slurm_job_id"),
                "engine_init_seconds": float(result["timing"]["engine_init_seconds"]),
                "samples": samples,
                "median_seconds": statistics.median(samples),
                "raw": result,
            }
        )
    if not records:
        raise FairnessError("no measured results found (only prime runs?)")
    return records


def check_fairness(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fail closed on any confounder that would invalidate a ratio."""
    violations: list[str] = []

    def _require_uniform(label: str, values: list[Any]) -> Any:
        encoded = {json.dumps(value, sort_keys=True) for value in values}
        if len(encoded) > 1:
            violations.append(f"{label} differs across runs: {sorted(encoded)}")
        return values[0]

    hostname = _require_uniform("runtime.hostname", [r["hostname"] for r in records])
    for field in _INVARIANT_RUNTIME_FIELDS[1:]:
        _require_uniform(f"runtime.{field}", [r["raw"]["runtime"].get(field) for r in records])

    # The PPL probe describes its workload under "blocking"; the latency probe
    # under "probe". Check whichever is present, and every key it declares, so
    # a new probe field cannot slip past the gate by not being on a fixed list.
    if any("blocking" in r["raw"] for r in records):
        for field in _INVARIANT_BLOCKING_FIELDS:
            _require_uniform(
                f"blocking.{field}", [r["raw"].get("blocking", {}).get(field) for r in records]
            )
    probe_fields = sorted(
        {key for r in records for key in r["raw"].get("probe", {})}
    )
    for field in probe_fields:
        _require_uniform(
            f"probe.{field}", [r["raw"].get("probe", {}).get(field) for r in records]
        )
    if not probe_fields and not any("blocking" in r["raw"] for r in records):
        violations.append("no probe/blocking description found; workload shape is unverified")

    for field in _INVARIANT_ENGINE_FIELDS:
        _require_uniform(
            f"engine_args.{field}", [r["raw"].get("engine_args", {}).get(field) for r in records]
        )

    clock_states = {r["gpu_clock_pin_status"] for r in records}
    return {
        "hostname": hostname,
        "gpu_clock_pin_status": sorted(str(state) for state in clock_states),
        "clocks_pinned": all(
            str(state).startswith("pinned") for state in clock_states
        ),
        "violations": violations,
        "passed": not violations,
    }


def _spread(values: list[float]) -> dict[str, Any]:
    if len(values) < 2:
        return {"n": len(values), "median": values[0] if values else None}
    mean = statistics.fmean(values)
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        # max/min is the honest statement of "how far apart can two runs of the
        # same thing land", which is exactly what a speedup claim competes with.
        "max_over_min": max(values) / min(values),
        "mean": mean,
        "cv_percent": 100.0 * statistics.stdev(values) / mean,
    }


def noise_floor(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Spread among runs of an identical configuration: the reportability bar."""
    floors = {}
    for group in sorted({r["group"] for r in records}):
        members = [r for r in records if r["group"] == group]
        if len(members) < 2:
            continue
        cross = _spread([r["median_seconds"] for r in members])
        within = [_spread(r["samples"]) for r in members if len(r["samples"]) > 1]
        folded = _paired_ratios(members, members)
        floors[group] = {
            "cross_process": cross,
            "within_process_max_cv_percent": (
                max(s["cv_percent"] for s in within) if within else None
            ),
            "fold_paired_ratios": [pair["ratio"] for pair in folded],
            "engine_init": _spread([r["engine_init_seconds"] for r in members]),
        }
    if not floors:
        raise FairnessError(
            "no configuration was repeated, so there is no noise floor and no "
            "ratio can be defended; add a null-control arm"
        )
    return floors


def _ordered(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (r for r in runs if r["index"] is not None), key=lambda r: r["index"]
    )


def _paired_ratios(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Pair runs so that position-in-time bias cancels as common mode.

    The interleaved order is a palindrome, e.g. ``d a s u u s a d``. Across two
    different configurations the k-th occurrence of each lands symmetrically
    about the centre, so pairing k-th with k-th yields temporally adjacent
    pairs -- the same pairing job 416672 recorded as ``position_pairs``.

    Within a single configuration (the null control) the two halves of the
    palindrome are the thing being compared, so occurrence k pairs with
    occurrence M-1-k instead.
    """
    left_runs = _ordered(left)
    right_runs = _ordered(right)
    if not left_runs or not right_runs:
        return []

    same_group = [r["path"] for r in left_runs] == [r["path"] for r in right_runs]
    if same_group:
        count = len(left_runs)
        couples = [(left_runs[k], left_runs[count - 1 - k]) for k in range(count // 2)]
    else:
        couples = list(zip(left_runs, right_runs))

    return [
        {
            "left_index": l_run["index"],
            "right_index": r_run["index"],
            "left_seconds": l_run["median_seconds"],
            "right_seconds": r_run["median_seconds"],
            "ratio": l_run["median_seconds"] / r_run["median_seconds"],
        }
        for l_run, r_run in couples
        if l_run["path"] != r_run["path"]
    ]


def compare(
    records: list[dict[str, Any]],
    floors: dict[str, Any],
    baseline: str | None,
) -> list[dict[str, Any]]:
    """Ratio every group against the baseline, judged against the noise floor."""
    groups = sorted({r["group"] for r in records})
    if baseline is None:
        baseline = "official" if "official" in groups else groups[0]
    if baseline not in groups:
        raise FairnessError(f"baseline {baseline!r} not among groups {groups}")

    # The bar is the worst same-configuration spread anywhere in the cohort:
    # a claim must beat the noise we actually observed, not the best case.
    floor_ratio = max(
        floor["cross_process"]["max_over_min"] for floor in floors.values()
    )
    base_runs = [r for r in records if r["group"] == baseline]
    base_median = statistics.median([r["median_seconds"] for r in base_runs])

    comparisons = []
    for group in groups:
        if group == baseline:
            continue
        runs = [r for r in records if r["group"] == group]
        median = statistics.median([r["median_seconds"] for r in runs])
        # Speedup > 1 means the arm finished faster than the baseline.
        naive = base_median / median
        paired = _paired_ratios(base_runs, runs)
        paired_ratios = [pair["ratio"] for pair in paired]
        reportable = abs(math.log(naive)) > abs(math.log(floor_ratio))
        comparisons.append(
            {
                "group": group,
                "baseline": baseline,
                "n_runs": len(runs),
                "median_generation_seconds": median,
                "baseline_median_generation_seconds": base_median,
                "naive_median_speedup": naive,
                "position_paired_speedups": paired_ratios,
                "position_paired_median_speedup": (
                    statistics.median(paired_ratios) if paired_ratios else None
                ),
                "noise_floor_ratio": floor_ratio,
                "reportable": reportable,
                "verdict": (
                    "reportable"
                    if reportable
                    else "INSIDE NOISE FLOOR - do not report as a speedup"
                ),
                "engine_init_excluded": True,
            }
        )
    return comparisons


def _format_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    fairness = report["fairness"]
    lines.append("=" * 78)
    lines.append("PERF BASELINE REPORT")
    lines.append("=" * 78)
    lines.append(f"host              : {fairness['hostname']}")
    lines.append(f"gpu clock pinning : {', '.join(fairness['gpu_clock_pin_status'])}")
    lines.append(f"fairness gates    : {'PASSED' if fairness['passed'] else 'FAILED'}")
    for violation in fairness["violations"]:
        lines.append(f"  ! {violation}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("NOISE FLOOR (identical configuration, repeated)")
    lines.append("-" * 78)
    lines.append(
        f"{'group':<26}{'n':>3}{'median s':>11}{'max/min':>10}{'CV %':>8}{'fold-pair range':>18}"
    )
    for group, floor in report["noise_floor"].items():
        cross = floor["cross_process"]
        pairs = floor["fold_paired_ratios"]
        pair_text = f"{min(pairs):.3f}-{max(pairs):.3f}" if pairs else "-"
        lines.append(
            f"{group:<26}{cross['n']:>3}{cross['median']:>11.3f}"
            f"{cross['max_over_min']:>10.3f}{cross['cv_percent']:>8.2f}{pair_text:>18}"
        )
    lines.append("")
    lines.append(
        "  max/min is how far apart two runs of the SAME configuration landed."
    )
    lines.append("  Any speedup claim must beat this ratio to mean anything.")
    lines.append(
        "  fold-pair range is the same spread after the palindrome cancels drift."
    )
    lines.append("")

    engine = report["engine_init"]
    lines.append("-" * 78)
    lines.append("ENGINE INIT (reported separately; never inside a speedup)")
    lines.append("-" * 78)
    lines.append(f"{'group':<26}{'n':>3}{'median s':>11}{'max/min':>10}")
    for group, spread in engine.items():
        lines.append(
            f"{group:<26}{spread['n']:>3}{spread['median']:>11.1f}"
            f"{spread.get('max_over_min', float('nan')):>10.3f}"
        )
    lines.append("")
    lines.append("  TensorBridge pays NVRTC JIT here; official/normal_a8 do not.")
    lines.append("  That gap is a build-cost difference, not a kernel speed difference.")
    lines.append("")

    if report["comparisons"]:
        lines.append("-" * 78)
        lines.append("SPEEDUP vs BASELINE (>1 means faster than baseline)")
        lines.append("-" * 78)
        lines.append(
            f"{'group':<26}{'naive':>9}{'paired':>9}{'floor':>9}  verdict"
        )
        for row in report["comparisons"]:
            paired = row["position_paired_median_speedup"]
            paired_text = f"{paired:.3f}" if paired is not None else "-"
            lines.append(
                f"{row['group']:<26}{row['naive_median_speedup']:>9.3f}"
                f"{paired_text:>9}{row['noise_floor_ratio']:>9.3f}  {row['verdict']}"
            )
    else:
        lines.append("-" * 78)
        lines.append("NO CROSS-CONFIGURATION COMPARISON")
        lines.append("-" * 78)
        lines.append("  This cohort contains a single configuration.")
        lines.append("  That is the null-control case: the numbers above ARE the result.")
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_dirs",
        nargs="+",
        type=Path,
        help="same_node_<jobid> directories, or directories of result JSON",
    )
    parser.add_argument(
        "--warmup-repeats",
        type=int,
        default=1,
        help="in-process repeats to discard before timing (default 1)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="group key used as the speedup denominator (default: official)",
    )
    parser.add_argument("--output", type=Path, help="write the machine-readable report here")
    args = parser.parse_args()

    paths: list[Path] = []
    for directory in args.result_dirs:
        if directory.is_file():
            paths.append(directory)
        else:
            paths.extend(sorted(directory.glob("*.json")))
    if not paths:
        print("no result JSON found", file=sys.stderr)
        return 2

    try:
        records = load_runs(paths, args.warmup_repeats)
        fairness = check_fairness(records)
        floors = noise_floor(records)
        comparisons = compare(records, floors, args.baseline)
    except FairnessError as error:
        print(f"[perf-baseline] {error}", file=sys.stderr)
        return 3

    report = {
        "schema_version": 1,
        "warmup_repeats": args.warmup_repeats,
        "fairness": fairness,
        "noise_floor": floors,
        "engine_init": {
            group: _spread(
                [r["engine_init_seconds"] for r in records if r["group"] == group]
            )
            for group in sorted({r["group"] for r in records})
        },
        "comparisons": comparisons,
        "runs": [
            {key: value for key, value in record.items() if key != "raw"}
            for record in records
        ],
    }

    print(_format_report(report))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\n[perf-baseline] wrote {args.output}")
    return 0 if fairness["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
