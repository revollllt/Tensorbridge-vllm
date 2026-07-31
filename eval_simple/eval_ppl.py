#!/usr/bin/env python3
"""WikiText-2 perplexity for one TensorBridge arm.

Scoring uses vLLM's prompt logprobs, not generation: each block is submitted as a
prompt and the model reports the logprob it assigned to every token already in
that prompt.

Blocking is strided. A block holds up to `max_model_len - 1` tokens but only its
last `--target-tokens` are scored; the tokens before them are context. Because
the window advances by exactly the target width, every token is scored once and
almost every scored token carries ~1k tokens of left context. Cutting the text
into disjoint 2048-token chunks instead would leave each chunk's opening tokens
nearly context-free and inflate perplexity.

Usage:
    python eval_ppl.py --arm alpha_0961 --model /path/to/Qwen3.6-27B-NVFP4
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import arms


def build_blocks(token_ids, max_prompt_tokens, target_tokens):
    """Return (prompt_ids, local_target_start) per block, covering each token once."""
    blocks = []
    target_start = 1  # token 0 has no predecessor, so nothing predicts it
    while target_start < len(token_ids):
        target_end = min(target_start + target_tokens, len(token_ids))
        prompt_start = max(0, target_end - max_prompt_tokens)
        prompt_ids = [int(t) for t in token_ids[prompt_start:target_end]]
        blocks.append((prompt_ids, target_start - prompt_start))
        target_start = target_end
    return blocks


def score(blocks, outputs):
    """Sum -logprob over each block's target region. Returns (nll_sum, n)."""
    nlls = []
    for index, ((prompt_ids, local_start), out) in enumerate(zip(blocks, outputs)):
        if [int(t) for t in out.prompt_token_ids] != prompt_ids:
            raise RuntimeError(f"block {index}: engine returned a different prompt")
        logprobs = out.prompt_logprobs
        if logprobs is None or len(logprobs) != len(prompt_ids):
            raise RuntimeError(f"block {index}: prompt logprob length mismatch")
        for pos in range(local_start, len(prompt_ids)):
            candidates = logprobs[pos]
            token = prompt_ids[pos]
            if candidates is None or token not in candidates:
                raise RuntimeError(f"block {index}: no logprob for position {pos}")
            value = candidates[token]
            value = float(getattr(value, "logprob", value))
            # A non-finite logprob means the kernel produced garbage. Averaging it
            # away would hide exactly the failure this evaluation exists to catch.
            if not math.isfinite(value):
                raise RuntimeError(f"block {index}: non-finite logprob at {pos}")
            nlls.append(-value)
    # fsum, not sum: ~300k additions of similar magnitude drift in plain float.
    return math.fsum(nlls), len(nlls)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=sorted(arms.ARMS))
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--output", type=Path, help="write the result JSON here")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--target-tokens", type=int, default=1024)
    p.add_argument("--max-blocks", type=int, help="limit blocks for a smoke run")
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = p.parse_args()

    env = arms.apply(args.arm)  # must precede the vllm import

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    quant_config_class = arms.confirm_active()
    print(f"[ppl] plugin active: {quant_config_class}")

    # wikitext-2-raw-v1 test split, joined with blank lines, tokenized as one
    # stream. No special tokens: they would be scored as if they were text.
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    blocks = build_blocks(token_ids, args.max_model_len - 1, args.target_tokens)
    if args.max_blocks:
        blocks = blocks[: args.max_blocks]
    expected = sum(len(ids) - start for ids, start in blocks)
    print(f"[ppl] arm={args.arm} tokens={len(token_ids)} blocks={len(blocks)} "
          f"scored={expected}")

    started = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        quantization="modelopt_mixed",
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Qwen3.6 is multimodal. Without this vLLM builds the vision tower too,
        # which pulls in vllm.vllm_flash_attn.layers.rotary — absent from this
        # install — and engine init dies. Text-only evaluation needs none of it.
        language_model_only=True,
        # CUDA graphs on, matching eval_gsm8k.py. Scoring here is prefill-only
        # (one output token, read back through prompt_logprobs), so graph replay
        # has little to capture and mostly costs capture time at startup. It is
        # enabled anyway so both benchmarks share one engine configuration.
        enforce_eager=False,
        enable_prefix_caching=False,  # else blocks would reuse each other's KV
        disable_log_stats=True,
    )
    init_seconds = time.perf_counter() - started

    started = time.perf_counter()
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids, _ in blocks],
        SamplingParams(
            temperature=0.0,
            max_tokens=1,  # vLLM needs to emit something; the token is discarded
            prompt_logprobs=1,
            logprobs=1,
            detokenize=False,
            ignore_eos=True,
        ),
    )
    generate_seconds = time.perf_counter() - started

    nll_sum, scored = score(blocks, outputs)
    if scored != expected:
        raise RuntimeError(f"scored {scored} tokens, expected {expected}")
    mean_nll = nll_sum / scored
    result = {
        "arm": args.arm,
        "model": str(args.model),
        "environment": env,
        "quant_config_class": quant_config_class,
        "versions": arms.versions(),
        "blocking": {
            "total_tokens": len(token_ids),
            "blocks": len(blocks),
            "max_model_len": args.max_model_len,
            "target_tokens_per_block": args.target_tokens,
        },
        "metrics": {
            "scored_tokens": scored,
            "nll_sum": nll_sum,
            "mean_nll": mean_nll,
            "ppl": math.exp(mean_nll),
        },
        "timing": {
            "engine_init_seconds": init_seconds,
            "generate_seconds": generate_seconds,
        },
    }
    print(json.dumps(result["metrics"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[ppl] wrote {args.output}")


if __name__ == "__main__":
    main()
