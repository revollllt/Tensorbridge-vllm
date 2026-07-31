#!/usr/bin/env python3
"""Full GSM8K exact-match for one TensorBridge arm, via lm-evaluation-harness.

lm-eval owns the prompting, generation, answer extraction, and scoring; this
script only pins the arm, the engine, and the decode settings, then records what
ran. Re-implementing the metric would risk drifting from the numbers this is
meant to reproduce.

The task file next to this script fixes the dataset revision and the answer
filter: the model must close with a line matching

    The answer is <number>.

and only that final line is read. A correct computation that ends in another
format scores zero, which is deliberate — the arms are compared under identical
formatting pressure, so the difference between them stays attributable to the
kernel.

Usage:
    python eval_gsm8k.py --arm alpha_0961 --model /path/to/Qwen3.6-27B-NVFP4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import arms

HERE = Path(__file__).resolve().parent

SYSTEM_INSTRUCTION = (
    "Use no more than six short sentences or equations. End with a separate final "
    "line in the exact form The answer is N. Replace N with the numeric answer only, "
    "omit units, and write nothing after that line."
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=sorted(arms.ARMS))
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--output", type=Path, help="write the result JSON here")
    p.add_argument("--samples-dir", type=Path, help="write per-document records here")
    p.add_argument("--limit", type=int, help="first N documents only, for a smoke run")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-gen-toks", type=int, default=1024)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = p.parse_args()

    env = arms.apply(args.arm)  # must precede the vllm import

    import lm_eval
    from lm_eval.tasks import TaskManager

    quant_config_class = arms.confirm_active()
    print(f"[gsm8k] plugin active: {quant_config_class}")

    started = time.perf_counter()
    raw = lm_eval.simple_evaluate(
        model="vllm",
        model_args={
            "pretrained": str(args.model),
            "quantization": "modelopt_mixed",
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            # Qwen3.6 is multimodal; building the vision tower breaks engine init
            # on this install and is unused by a text benchmark. See eval_ppl.py.
            "language_model_only": True,
            # CUDA graphs on. GSM8K is decode-bound, so eager mode would expose
            # each arm's per-layer Python dispatch cost on top of its kernel
            # time, and the arms do not share a dispatch path. Graph replay keeps
            # the comparison on the kernels.
            "enforce_eager": False,
            "enable_prefix_caching": False,  # keep documents independent
            "disable_log_stats": True,
            "max_gen_toks": args.max_gen_toks,
            "seed": 1234,
            # lm-eval defaults this to True. Qwen3.6 then reasons at length and
            # keeps commenting after its final line, which the answer filter
            # anchors to end-of-string and rejects -- correct arithmetic scored
            # as wrong, at roughly a quarter of the true rate.
            "enable_thinking": False,
        },
        tasks=["gsm8k_final_answer"],
        task_manager=TaskManager(include_path=str(HERE)),
        num_fewshot=0,
        batch_size="auto",
        limit=args.limit,
        apply_chat_template=True,
        system_instruction=SYSTEM_INSTRUCTION,
        gen_kwargs={"temperature": 0.0, "max_gen_toks": args.max_gen_toks},
        log_samples=True,
        # Greedy decoding makes these mostly inert, but an unpinned seed is one
        # more reason a rerun could disagree.
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    elapsed = time.perf_counter() - started
    if raw is None:
        raise RuntimeError("lm-eval returned no result")

    samples = raw.pop("samples", {})
    task_result = raw["results"]["gsm8k_final_answer"]
    n_samples = raw["n-samples"]["gsm8k_final_answer"]
    metrics = {
        "exact_match": task_result["exact_match,final-answer"],
        "exact_match_stderr": task_result["exact_match_stderr,final-answer"],
        "n": n_samples["effective"],
    }
    result = {
        "arm": args.arm,
        "model": str(args.model),
        "environment": env,
        "quant_config_class": quant_config_class,
        "versions": arms.versions(),
        "metrics": metrics,
        "lm_eval_results": raw["results"],
        "n_samples": raw["n-samples"],
        "timing": {"evaluate_seconds": elapsed},
    }
    print(json.dumps(metrics, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str),
                               encoding="utf-8")
        print(f"[gsm8k] wrote {args.output}")

    # Per-document records. compare.py needs these: the arms score the same
    # documents, and the paired test over them is far more sensitive than
    # comparing two aggregate rates.
    if args.samples_dir:
        args.samples_dir.mkdir(parents=True, exist_ok=True)
        path = args.samples_dir / "gsm8k_final_answer.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in samples.get("gsm8k_final_answer", []):
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(f"[gsm8k] wrote {path}")

    # A misconfigured engine does not raise here, it returns a plausible number.
    # Dropping enable_thinking=False once scored 27.7% with the arithmetic
    # correct throughout: the model kept commenting past its final answer line
    # and the end-anchored filter rejected every one. Every arm lands near 96%,
    # so anything this low is a broken setup rather than a weak arm. Artifacts
    # are already written, so the run is still available for diagnosis.
    if not args.limit and metrics["exact_match"] < 0.80:
        raise SystemExit(
            f"[gsm8k] exact_match {metrics['exact_match']:.4f} is far below the ~0.96 "
            f"every arm reaches. Check the engine arguments against the reference "
            f"harness in vllm/plugins/tensorbridge_evaluation/lm_harness.py before "
            f"trusting this run."
        )


if __name__ == "__main__":
    main()
