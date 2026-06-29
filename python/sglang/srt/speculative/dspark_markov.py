"""DSpark Markov head sampling utilities.

DSpark (from DeepSeek's DeepSpec) is block-wise speculative decoding: each
draft step proposes a block of tokens through a small Markov head instead of
one token at a time. The Markov head adds a low-rank bias to the base logits
based on the previous token, enabling sequential dependency modeling within
a block.

This module provides the core Markov bias computation and sequential sampling
logic, ported from DeepSeek's DeepSpec / vLLM DSpark implementation.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def markov_bias(
    prev_token_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
) -> torch.Tensor:
    """Compute the Markov bias for a batch of previous token ids.

    Args:
        prev_token_ids: [batch_size] — previous token ids.
        markov_w1: [vocab_size, markov_rank] — Markov embedding table.
        markov_w2: [vocab_size, markov_rank] — Markov projection table.

    Returns:
        [batch_size, vocab_size] — Markov bias to add to base logits.
    """
    low_rank = markov_w1.index_select(0, prev_token_ids)
    return torch.matmul(low_rank, markov_w2.t())


def sequential_markov_sample_greedy(
    base_logits: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
) -> torch.Tensor:
    """Sequentially sample a block of tokens using greedy argmax with Markov bias.

    For each position in the block, the Markov bias is computed from the
    previous token (starting from the anchor token) and added to the base
    logits. The token is then sampled via greedy argmax.

    Args:
        base_logits: [batch_size, num_positions, vocab_size] — base logits
            from the draft model's lm_head.
        anchor_token_ids: [batch_size] — the token preceding the first draft
            position (the last verified token from the target model).
        markov_w1: [vocab_size, markov_rank] — Markov embedding table.
        markov_w2: [vocab_size, markov_rank] — Markov projection table.

    Returns:
        [batch_size, num_positions] — sampled token ids.
    """
    batch_size, num_positions, vocab_size = base_logits.shape
    tokens = base_logits.new_empty(
        (batch_size, num_positions), dtype=torch.int64
    )
    prev = anchor_token_ids
    for position in range(num_positions):
        logits = base_logits[:, position] + markov_bias(
            prev, markov_w1, markov_w2
        )
        sampled = torch.argmax(logits, dim=-1)
        tokens[:, position] = sampled
        prev = sampled
    return tokens


def sequential_markov_sample(
    base_logits: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    sample_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sequentially sample a block of tokens with Markov bias and a custom sampler.

    Args:
        base_logits: [batch_size, num_positions, vocab_size] — base logits.
        anchor_token_ids: [batch_size] — token preceding the first draft position.
        markov_w1: [vocab_size, markov_rank] — Markov embedding table.
        markov_w2: [vocab_size, markov_rank] — Markov projection table.
        sample_fn: callable that takes [batch_size, vocab_size] logits and
            returns [batch_size] sampled token ids.

    Returns:
        tokens: [batch_size, num_positions] — sampled token ids.
    """
    batch_size, num_positions, vocab_size = base_logits.shape
    tokens = base_logits.new_empty(
        (batch_size, num_positions), dtype=torch.int64
    )
    prev = anchor_token_ids
    for position in range(num_positions):
        logits = base_logits[:, position] + markov_bias(
            prev, markov_w1, markov_w2
        )
        sampled = sample_fn(logits)
        tokens[:, position] = sampled
        prev = sampled
    return tokens, None


__all__ = [
    "markov_bias",
    "sequential_markov_sample_greedy",
    "sequential_markov_sample",
]
