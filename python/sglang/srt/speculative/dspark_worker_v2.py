"""DSpark speculative decoding worker (spec-v2).

DSpark (from DeepSeek's DeepSpec) is block-wise speculative decoding that
extends DFlash with a Markov head. Instead of greedy argmax for draft token
sampling, DSpark uses sequential Markov sampling: each draft position's
logits are corrected by a low-rank bias derived from the previous token.

This worker inherits all DFlash behavior (block drafting, target verify,
accept/reject) and only overrides the draft token sampling step.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.dspark_markov import (
    sequential_markov_sample_greedy,
)

logger = logging.getLogger(__name__)


class DSparkWorkerV2(DFlashWorkerV2):
    """DSPARK speculative decoding worker (spec-v2).

    Identical to DFlash except draft tokens are sampled via sequential
    Markov sampling instead of greedy argmax. The Markov head adds a
    low-rank bias to the base logits based on the previous token, enabling
    better sequential dependency modeling within a draft block.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dspark_markov_w1: Optional[torch.Tensor] = None
        self._dspark_markov_w2: Optional[torch.Tensor] = None
        self._dspark_markov_resolved = False

    def _resolve_markov_weights(self) -> bool:
        """Resolve Markov head weights from the draft model.

        Returns True if Markov weights are available, False otherwise.
        When False, the worker falls back to DFlash greedy sampling.
        """
        if self._dspark_markov_resolved:
            return self._dspark_markov_w1 is not None

        self._dspark_markov_resolved = True
        draft_model = self.draft_model

        markov_w1 = getattr(draft_model, "markov_w1", None)
        markov_w2 = getattr(draft_model, "markov_w2", None)

        if markov_w1 is not None and markov_w2 is not None:
            self._dspark_markov_w1 = markov_w1
            self._dspark_markov_w2 = markov_w2
            if self.tp_rank == 0:
                logger.info(
                    "DSPARK: Markov head resolved from draft model. "
                    "w1 shape=%s, w2 shape=%s",
                    tuple(markov_w1.shape),
                    tuple(markov_w2.shape),
                )
            return True

        if self.tp_rank == 0:
            logger.warning(
                "DSPARK: Draft model has no markov_w1/markov_w2 weights. "
                "Falling back to DFlash greedy sampling."
            )
        return False

    def _sample_draft_tokens_from_output(
        self,
        *,
        draft_logits_output,
        block_ids: torch.Tensor,
        lm_head,
        bs: int,
    ) -> torch.Tensor:
        """DSpark: sample draft tokens using sequential Markov sampling.

        Falls back to DFlash greedy sampling if Markov weights are unavailable.
        """
        if not self._resolve_markov_weights():
            return super()._sample_draft_tokens_from_output(
                draft_logits_output=draft_logits_output,
                block_ids=block_ids,
                lm_head=lm_head,
                bs=bs,
            )

        block_size = int(self.block_size)
        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DSPARK draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, block_size, -1)

        # Compute base logits for positions 1..block_size-1 (skip anchor at 0).
        # Use the draft model's logits if available; otherwise compute from
        # hidden states via the (vocab-parallel) LM head with TP all-gather.
        base_logits = draft_logits_output.next_token_logits
        if base_logits is not None:
            base_logits = base_logits.view(bs, block_size, -1)[:, 1:, :]
        else:
            base_logits = self._compute_full_vocab_logits(
                draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
                lm_head,
            ).view(bs, block_size - 1, -1)

        # Sequential Markov sampling: each position's logits are corrected
        # by a low-rank bias from the previous token.
        anchor_token_ids = block_ids[:, 0].to(torch.long)
        draft_next = sequential_markov_sample_greedy(
            base_logits=base_logits,
            anchor_token_ids=anchor_token_ids,
            markov_w1=self._dspark_markov_w1,
            markov_w2=self._dspark_markov_w2,
        )

        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)
        return draft_tokens

    def _compute_full_vocab_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head,
    ) -> torch.Tensor:
        """Compute full-vocabulary logits from hidden states.

        Handles TP by all-gathering per-rank logits. Used as a fallback when
        the draft model's forward did not produce next_token_logits.
        """
        from sglang.srt.distributed import tensor_model_parallel_all_gather

        weight = lm_head.weight
        weight_dtype = weight.dtype
        hs = (
            hidden_states
            if hidden_states.dtype == weight_dtype
            else hidden_states.to(weight_dtype)
        )
        local_logits = torch.matmul(hs, weight.t())

        tp_size = getattr(lm_head, "shard_indices", None)
        if tp_size is not None or hasattr(lm_head, "shard_indices"):
            local_logits = tensor_model_parallel_all_gather(local_logits)

        return local_logits


__all__ = ["DSparkWorkerV2"]
