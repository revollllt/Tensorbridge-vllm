#!/usr/bin/env python3
"""Summarise a results directory: PPL per arm, and a paired GSM8K comparison.

The arms answer the same GSM8K documents, so the comparison is paired. An exact
McNemar test over the documents where two arms disagree is much more sensitive
than comparing their overall rates, which throws the pairing away. With ~1300
documents and a handful of disagreements, the unpaired form cannot resolve
anything below about a point; the paired form resolves a few tenths.

The pairing is checked, not assumed: mismatched document sets or hashes abort.

Usage:
    python compare.py results/
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

# normal_a8 is the FPMA baseline: same B8 weights, same FP8 activation, same
# epilogue scale, differing only in whether B8 was expanded exactly at load or
# approximated in the mainloop. Comparing against official instead would mix in
# the activation dtype change.
PRIMARY = ("normal_a8", "alpha_0961")


def load_ppl(results: Path) -> dict[str, dict]:
    out = {}
    for path in sorted(results.glob("*_ppl.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[data["arm"]] = data["metrics"]
    return out


def load_gsm8k(results: Path) -> tuple[dict[str, dict[int, int]], dict[str, dict[int, str]]]:
    scores, hashes = {}, {}
    for path in sorted(glob.glob(str(results / "*_gsm8k_samples" / "*.jsonl"))):
        arm = Path(path).parent.name.replace("_gsm8k_samples", "")
        scores[arm], hashes[arm] = {}, {}
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                doc_id = int(row["doc_id"])
                scores[arm][doc_id] = int(row["exact_match"])
                hashes[arm][doc_id] = row["doc_hash"]
    return scores, hashes


def check_paired(hashes: dict[str, dict[int, str]]) -> list[int]:
    arms = list(hashes)
    reference = arms[0]
    ids = sorted(hashes[reference])
    for arm in arms[1:]:
        if sorted(hashes[arm]) != ids:
            raise SystemExit(f"document sets differ between {reference} and {arm}")
        for doc_id in ids:
            if hashes[arm][doc_id] != hashes[reference][doc_id]:
                raise SystemExit(f"doc_hash differs at doc_id {doc_id}: not the same question")
    return ids


def mcnemar(base: dict[int, int], cand: dict[int, int], ids: list[int]) -> dict:
    wins = sum(1 for i in ids if cand[i] and not base[i])
    losses = sum(1 for i in ids if base[i] and not cand[i])
    n = wins + losses
    delta = (wins - losses) / len(ids) * 100
    stderr = math.sqrt(n) / len(ids) * 100 if n else 0.0
    # Under the null each disagreement is a coin flip, so the tail is binomial.
    p = 1.0
    if n:
        tail = sum(math.comb(n, j) for j in range(min(wins, losses) + 1))
        p = min(1.0, 2 * tail / 2**n)
    return {"wins": wins, "losses": losses, "delta_pp": delta,
            "ci": (delta - 1.96 * stderr, delta + 1.96 * stderr), "p": p}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()

    ppl = load_ppl(args.results)
    if ppl:
        print("WikiText-2 perplexity")
        print(f"  {'arm':14}{'mean NLL':>15}{'PPL':>14}{'scored':>10}")
        for arm, m in sorted(ppl.items(), key=lambda kv: kv[1]["ppl"]):
            print(f"  {arm:14}{m['mean_nll']:15.10f}{m['ppl']:14.9f}{m['scored_tokens']:>10}")
        print()

    scores, hashes = load_gsm8k(args.results)
    if not scores:
        return
    ids = check_paired(hashes)
    print(f"GSM8K exact-match  (pairing verified over {len(ids)} documents)")
    print(f"  {'arm':14}{'exact_match':>13}{'correct':>10}")
    for arm in sorted(scores, key=lambda a: -sum(scores[a].values())):
        correct = sum(scores[arm][i] for i in ids)
        print(f"  {arm:14}{correct / len(ids) * 100:12.4f}%{correct:>10}")

    arm_list = sorted(scores)
    pairs = [(a, b) for i, a in enumerate(arm_list) for b in arm_list[i + 1:]]
    if PRIMARY[0] in scores and PRIMARY[1] in scores:
        pairs = [PRIMARY] + [p for p in pairs if set(p) != set(PRIMARY)]

    print(f"\n  {'comparison':30}{'win/loss':>10}{'delta pp':>10}{'95% CI pp':>24}{'p':>9}")
    for base, cand in pairs:
        s = mcnemar(scores[base], scores[cand], ids)
        ci = f"[{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]"
        win_loss = f"{s['wins']}/{s['losses']}"
        tag = "  <- primary" if (base, cand) == PRIMARY else ""
        print(f"  {cand + ' - ' + base:30}{win_loss:>10}"
              f"{s['delta_pp']:>10.4f}{ci:>24}{s['p']:>9.4f}{tag}")

    unanimous = sum(1 for i in ids if len({scores[a][i] for a in arm_list}) == 1)
    print(f"\n  arms agree on {unanimous}/{len(ids)} documents, "
          f"disagree on {len(ids) - unanimous}")


if __name__ == "__main__":
    main()
