#!/usr/bin/env python3
"""Paired GSM8K comparison across TensorBridge lm-eval arms.

The arms score the same frozen document set, so an unpaired two-proportion
comparison throws away most of the available power. This analyzer verifies the
pairing first (identical doc_id and doc_hash across arms), then reports exact
McNemar statistics over the discordant pairs.

The pairing check is not a formality: a silent doc-set mismatch would make every
downstream delta meaningless, so a mismatch is a hard error rather than a
warning.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path


DEFAULT_ARMS = ("official", "normal_a8", "alpha_0961")


def load_arm(run_dir: Path, arm: str) -> tuple[dict[int, int], dict[int, str]]:
    pattern = str(run_dir / f"{arm}_*_samples" / "*.jsonl")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no sample JSONL for arm {arm!r} under {run_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous sample JSONL for arm {arm!r}: {matches}")
    scores: dict[int, int] = {}
    hashes: dict[int, str] = {}
    with open(matches[0], encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            doc_id = int(row["doc_id"])
            if doc_id in scores:
                raise ValueError(f"duplicate doc_id {doc_id} for arm {arm!r}")
            scores[doc_id] = int(row["exact_match"])
            hashes[doc_id] = row["doc_hash"]
    return scores, hashes


def assert_paired(hashes: dict[str, dict[int, str]]) -> set[int]:
    arms = list(hashes)
    reference = arms[0]
    ids = set(hashes[reference])
    for arm in arms[1:]:
        if set(hashes[arm]) != ids:
            missing = ids ^ set(hashes[arm])
            raise ValueError(
                f"doc_id set differs between {reference!r} and {arm!r}; "
                f"{len(missing)} ids only present in one arm"
            )
    for doc_id in sorted(ids):
        digests = {arm: hashes[arm][doc_id] for arm in arms}
        if len(set(digests.values())) != 1:
            raise ValueError(f"doc_hash mismatch at doc_id {doc_id}: {digests}")
    return ids


def mcnemar_exact(
    base: dict[int, int], cand: dict[int, int], ids: set[int]
) -> dict[str, float | int]:
    """Exact two-sided McNemar over discordant pairs, plus a paired delta CI."""
    wins = sum(1 for i in ids if cand[i] == 1 and base[i] == 0)
    losses = sum(1 for i in ids if cand[i] == 0 and base[i] == 1)
    discordant = wins + losses
    total = len(ids)
    delta_pp = (wins - losses) / total * 100.0
    # Under H0 each discordant pair is a fair coin, so the tail is binomial.
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(wins, losses)
        tail = sum(math.comb(discordant, j) for j in range(smaller + 1))
        p_value = min(1.0, 2.0 * tail / 2**discordant)
    stderr_pp = math.sqrt(discordant) / total * 100.0 if discordant else 0.0
    return {
        "wins": wins,
        "losses": losses,
        "discordant": discordant,
        "delta_pp": delta_pp,
        "stderr_pp": stderr_pp,
        "ci_low_pp": delta_pp - 1.96 * stderr_pp,
        "ci_high_pp": delta_pp + 1.96 * stderr_pp,
        "p_value": p_value,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="directory holding <arm>_<job>_samples/ (e.g. "
        "benchmarks/results/lm_eval/<run_id>/generation_core)",
    )
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument(
        "--primary",
        nargs=2,
        metavar=("BASE", "CANDIDATE"),
        default=["normal_a8", "alpha_0961"],
        help="the required primary comparison; normal_a8 is the exact-arithmetic "
        "counterpart of the FPMA path, so it isolates FPMA from activation dtype",
    )
    parser.add_argument("--json", type=Path, help="also write the report here")
    args = parser.parse_args()

    scores: dict[str, dict[int, int]] = {}
    hashes: dict[str, dict[int, str]] = {}
    for arm in args.arms:
        scores[arm], hashes[arm] = load_arm(args.run_dir, arm)

    ids = assert_paired(hashes)
    print(f"pairing verified: {len(ids)} docs, doc_id and doc_hash identical across arms\n")

    print(f"{'arm':14}{'exact_match':>13}{'correct':>9}{'n':>7}")
    per_arm = {}
    for arm in args.arms:
        correct = sum(scores[arm][i] for i in ids)
        acc = correct / len(ids)
        per_arm[arm] = {"exact_match": acc, "correct": correct, "n": len(ids)}
        print(f"{arm:14}{acc * 100:12.4f}%{correct:>9}{len(ids):>7}")

    pairs = [(a, b) for a in args.arms for b in args.arms if a != b and args.arms.index(a) < args.arms.index(b)]
    primary = (args.primary[0], args.primary[1])
    if primary not in pairs and (primary[1], primary[0]) not in pairs:
        raise ValueError(f"primary comparison {primary} is not among the loaded arms")

    print(
        f"\n{'comparison':30}{'win/loss':>10}{'delta pp':>10}{'se pp':>8}"
        f"{'95% CI pp':>24}{'p exact':>9}"
    )
    report = {"run_dir": str(args.run_dir), "arms": per_arm, "comparisons": {}}
    ordered = [primary] + [p for p in pairs if p != primary and (p[1], p[0]) != primary]
    for base, cand in ordered:
        stat = mcnemar_exact(scores[base], scores[cand], ids)
        report["comparisons"][f"{cand} - {base}"] = stat
        tag = "  <- primary" if (base, cand) == primary else ""
        ci = f"[{stat['ci_low_pp']:+.4f}, {stat['ci_high_pp']:+.4f}]"
        print(
            f"{cand + ' - ' + base:30}{f'{stat['wins']}/{stat['losses']}':>10}"
            f"{stat['delta_pp']:>10.4f}{stat['stderr_pp']:>8.4f}{ci:>24}"
            f"{stat['p_value']:>9.4f}{tag}"
        )

    unanimous_right = sum(1 for i in ids if all(scores[a][i] == 1 for a in args.arms))
    unanimous_wrong = sum(1 for i in ids if all(scores[a][i] == 0 for a in args.arms))
    disputed = len(ids) - unanimous_right - unanimous_wrong
    report["agreement"] = {
        "all_correct": unanimous_right,
        "all_wrong": unanimous_wrong,
        "disputed": disputed,
    }
    print(
        f"\nagreement: all-correct {unanimous_right}, all-wrong {unanimous_wrong}, "
        f"disputed {disputed}"
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
