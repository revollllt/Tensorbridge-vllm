"""Token blocking and prompt-logprob accounting for perplexity evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class PromptBlock:
    prompt_token_ids: list[int]
    local_target_start: int
    global_target_start: int
    global_target_end: int

    @property
    def scored_tokens(self) -> int:
        return self.global_target_end - self.global_target_start


class NonFiniteLogprobError(ValueError):
    """A prompt-logprob failure with JSON-safe location diagnostics."""

    def __init__(self, total_nonfinite: int, diagnostics: list[dict[str, Any]]):
        self.total_nonfinite = total_nonfinite
        self.diagnostics = diagnostics
        first = diagnostics[0]
        super().__init__(
            "non-finite target logprobs: "
            f"count={total_nonfinite}, first_block={first['block_index']}, "
            f"first_prompt_position={first['prompt_position']}, "
            f"first_global_token_offset={first['global_token_offset']}, "
            f"first_value={first['logprob']}"
        )


def prompt_token_capacity(max_model_len: int, requested_output_tokens: int) -> int:
    """Reserve output positions so prompt plus generation fits the engine limit."""
    if requested_output_tokens < 1:
        raise ValueError("requested_output_tokens must be positive")
    capacity = max_model_len - requested_output_tokens
    if capacity < 2:
        raise ValueError("max_model_len must leave room for context and output tokens")
    return capacity


def build_prompt_blocks(
    token_ids: Sequence[int],
    *,
    max_model_len: int,
    target_tokens_per_block: int,
    max_blocks: int | None = None,
) -> list[PromptBlock]:
    """Cover each target token once while retaining overlap as context only."""
    if max_model_len < 2:
        raise ValueError("max_model_len must be at least 2")
    if not 1 <= target_tokens_per_block < max_model_len:
        raise ValueError("target_tokens_per_block must be in [1, max_model_len)")
    if max_blocks is not None and max_blocks <= 0:
        raise ValueError("max_blocks must be positive")
    if len(token_ids) < 2:
        return []

    blocks: list[PromptBlock] = []
    target_start = 1
    while target_start < len(token_ids):
        if max_blocks is not None and len(blocks) >= max_blocks:
            break
        target_end = min(target_start + target_tokens_per_block, len(token_ids))
        prompt_start = max(0, target_end - max_model_len)
        prompt_ids = [int(token) for token in token_ids[prompt_start:target_end]]
        local_target_start = target_start - prompt_start
        if local_target_start < 1:
            raise AssertionError("every scored token must have a predecessor in its prompt")
        blocks.append(
            PromptBlock(
                prompt_token_ids=prompt_ids,
                local_target_start=local_target_start,
                global_target_start=target_start,
                global_target_end=target_end,
            )
        )
        target_start = target_end
    return blocks


def _extract_logprob(value: Any) -> float:
    logprob = getattr(value, "logprob", value)
    return float(logprob)


def _candidate_summary(candidates: Any) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for token_id, value in candidates.items():
        rank = getattr(value, "rank", None)
        summary.append(
            {
                "token_id": int(token_id),
                "logprob": repr(_extract_logprob(value)),
                "rank": None if rank is None else int(rank),
            }
        )
    return summary


def score_prompt_logprobs(
    blocks: Sequence[PromptBlock],
    outputs: Sequence[Any],
    *,
    block_index_offset: int = 0,
    collect_nonfinite: bool = False,
    scan_context_for_nonfinite: bool = False,
    max_nonfinite_reports: int = 64,
) -> dict[str, float | int]:
    """Accumulate NLL only over each block's non-overlapping target region."""
    if len(outputs) != len(blocks):
        raise ValueError(f"expected {len(blocks)} outputs, got {len(outputs)}")

    if block_index_offset < 0:
        raise ValueError("block_index_offset must be non-negative")
    if max_nonfinite_reports <= 0:
        raise ValueError("max_nonfinite_reports must be positive")

    token_nlls: list[float] = []
    nonfinite_reports: list[dict[str, Any]] = []
    total_nonfinite = 0
    for local_block_index, (block, output) in enumerate(
        zip(blocks, outputs, strict=True)
    ):
        block_index = block_index_offset + local_block_index
        output_ids = [int(token) for token in output.prompt_token_ids]
        if output_ids != block.prompt_token_ids:
            raise ValueError(f"prompt token mismatch for block {block_index}")
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None or len(prompt_logprobs) != len(output_ids):
            raise ValueError(f"prompt logprob length mismatch for block {block_index}")
        if prompt_logprobs[0] is not None:
            raise ValueError(f"the first prompt logprob must be None for block {block_index}")

        scan_start = 1 if scan_context_for_nonfinite else block.local_target_start
        global_prompt_start = block.global_target_start - block.local_target_start
        for position in range(scan_start, len(output_ids)):
            candidates = prompt_logprobs[position]
            if candidates is None:
                if position < block.local_target_start:
                    continue
                raise ValueError(f"missing prompt logprobs at block {block_index}, pos {position}")
            target_token = output_ids[position]
            if target_token not in candidates:
                if position < block.local_target_start:
                    continue
                raise KeyError(
                    f"target token {target_token} missing at block {block_index}, pos {position}"
                )
            logprob = _extract_logprob(candidates[target_token])
            if not math.isfinite(logprob):
                total_nonfinite += 1
                if len(nonfinite_reports) < max_nonfinite_reports:
                    nonfinite_reports.append(
                        {
                            "block_index": block_index,
                            "prompt_position": position,
                            "global_token_offset": global_prompt_start + position,
                            "target_token_id": target_token,
                            "logprob": repr(logprob),
                            "prompt_len": len(output_ids),
                            "global_prompt_start": global_prompt_start,
                            "local_target_start": block.local_target_start,
                            "global_target_start": block.global_target_start,
                            "global_target_end": block.global_target_end,
                            "is_scored_target": position >= block.local_target_start,
                            "candidate_count": len(candidates),
                            "candidates": _candidate_summary(candidates),
                        }
                    )
                if not collect_nonfinite:
                    raise NonFiniteLogprobError(total_nonfinite, nonfinite_reports)
                continue
            if position >= block.local_target_start:
                token_nlls.append(-logprob)

    if total_nonfinite:
        raise NonFiniteLogprobError(total_nonfinite, nonfinite_reports)

    expected = sum(block.scored_tokens for block in blocks)
    if len(token_nlls) != expected:
        raise AssertionError(f"scored {len(token_nlls)} tokens, expected {expected}")
    if not token_nlls:
        raise ValueError("no tokens were scored")
    nll_sum = math.fsum(token_nlls)
    mean_nll = nll_sum / len(token_nlls)
    ppl = math.exp(mean_nll)
    if not math.isfinite(ppl):
        raise ValueError(f"non-finite perplexity: {ppl}")
    return {
        "scored_tokens": len(token_nlls),
        "nll_sum": nll_sum,
        "mean_nll": mean_nll,
        "ppl": ppl,
    }
