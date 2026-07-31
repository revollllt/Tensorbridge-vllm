#!/usr/bin/env python3
"""Summarise a results directory: per-batch speedup against a noise floor.

Two things make a latency ratio defensible here.

Position pairing. Arms run in separate processes, so a ratio built from arm
medians mixes in whatever drifted between processes. Runs are interleaved as a
palindrome (`official tensorbridge tensorbridge official`), and the k-th run of
one arm is paired with the k-th run of the other, which puts each pair at
mirrored positions in time so a linear drift cancels.

A noise floor. Each arm appears twice, and two runs of the *same* arm must land
at the same number. However far apart they actually land is the resolution of
the experiment, and an effect smaller than that is not reported as a speedup
regardless of how it looks. Job 416484 in the full harness is why: two runs of
one arm, differing by one FP32 constant and sharing every device instruction,
came out 1.92x apart.

Usage:
    python compare.py results/
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

BASELINE = "official"


def load(results: Path) -> list[dict]:
    runs = []
    for path in sorted(results.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "sweep" not in data:
            continue
        data["path"] = path.name
        runs.append(data)
    if not runs:
        raise SystemExit(f"no latency results in {results}")
    return runs


def check_comparable(runs: list[dict]) -> None:
    """Abort on anything that would make a ratio mean something else."""
    def uniform(label, values):
        encoded = {json.dumps(v, sort_keys=True) for v in values}
        if len(encoded) > 1:
            raise SystemExit(f"{label} differs across runs: {sorted(encoded)}")

    uniform("probe", [r["probe"] for r in runs])
    uniform("model", [r["model"] for r in runs])
    uniform("versions", [r["versions"] for r in runs])
    uniform("gpu name", [r["gpu"].get("name") for r in runs])
    uniform("gpu driver", [r["gpu"].get("driver") for r in runs])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--baseline", default=BASELINE)
    args = parser.parse_args()

    runs = load(args.results)
    check_comparable(runs)

    by_arm: dict[str, list[dict]] = {}
    for run in runs:
        by_arm.setdefault(run["arm"], []).append(run)
    for arm in by_arm:
        by_arm[arm].sort(key=lambda r: r["path"])

    gpu = runs[0]["gpu"]
    clock_note = ""
    if gpu.get("max_graphics_clock") and gpu.get("current_graphics_clock"):
        clock_note = (f"  (max {gpu['max_graphics_clock']}, "
                      f"was {gpu['current_graphics_clock']} at record time)")
    print(f"{gpu.get('name', '?')} driver {gpu.get('driver', '?')}{clock_note}")
    print(f"runs: " + ", ".join(f"{a}x{len(r)}" for a, r in sorted(by_arm.items())))
    print()

    if args.baseline not in by_arm:
        raise SystemExit(f"baseline {args.baseline!r} not among {sorted(by_arm)}")
    base_runs = by_arm[args.baseline]
    batch_sizes = [s["batch_size"] for s in runs[0]["sweep"]]

    def median_at(run, batch):
        return next(s["median_seconds"] for s in run["sweep"] if s["batch_size"] == batch)

    for arm, arm_runs in sorted(by_arm.items()):
        if arm == args.baseline:
            continue
        print(f"{arm} vs {args.baseline}")
        print(f"  {'batch':>6}{'baseline s':>13}{arm:>13}{'speedup':>11}"
              f"{'floor':>9}  verdict")
        for batch in batch_sizes:
            base = [median_at(r, batch) for r in base_runs]
            cand = [median_at(r, batch) for r in arm_runs]
            pairs = [b / c for b, c in zip(base, cand)]
            speedup = statistics.median(pairs)
            # Worst same-arm spread at this batch: what two identical runs did.
            floor = max(
                max(base) / min(base) if len(base) > 1 else 1.0,
                max(cand) / min(cand) if len(cand) > 1 else 1.0,
            )
            reportable = abs(speedup - 1) > abs(floor - 1)
            verdict = "reportable" if reportable else "inside noise floor"
            print(f"  {batch:>6}{statistics.median(base):>13.3f}"
                  f"{statistics.median(cand):>13.3f}{speedup:>10.3f}x"
                  f"{floor:>9.3f}  {verdict}")
        print()

    if any(len(r) < 2 for r in by_arm.values()):
        print("Every arm needs at least two runs, or there is no noise floor and")
        print("no ratio above can be defended. Run the palindrome twice through.")


if __name__ == "__main__":
    main()
