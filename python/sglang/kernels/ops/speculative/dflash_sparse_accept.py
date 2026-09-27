"""CPU reference helpers for DFlash2 sparse sampling acceptance."""

from __future__ import annotations

import torch


def _validate_sparse_inputs(
    candidates: torch.Tensor,
    candidate_ids: torch.Tensor,
    q_rows: torch.Tensor,
) -> None:
    if candidates.ndim != 2:
        raise ValueError("candidates must have shape [batch, block]")
    if candidate_ids.ndim != 3 or q_rows.ndim != 3:
        raise ValueError("candidate_ids and q_rows must have shape [batch, gamma, top_k]")
    if candidate_ids.shape != q_rows.shape:
        raise ValueError("candidate_ids and q_rows shapes must match")
    if candidates.shape[1] != candidate_ids.shape[1] + 1:
        raise ValueError("candidates block size must be gamma + 1")


def select_draft_q_values(
    *,
    candidates: torch.Tensor,
    candidate_ids: torch.Tensor,
    q_rows: torch.Tensor,
) -> torch.Tensor:
    """Return the selector q for each sampled draft token."""
    _validate_sparse_inputs(candidates, candidate_ids, q_rows)
    draft_tokens = candidates[:, 1:].unsqueeze(-1)
    matches = candidate_ids == draft_tokens
    return q_rows.float().masked_fill(~matches, 0.0).sum(dim=-1)


def sparse_residual_rows(
    *,
    target_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    q_rows: torch.Tensor,
) -> torch.Tensor:
    """Reference residual for one probability row, matching the dense kernel."""
    if target_probs.ndim != 2:
        raise ValueError("target_probs must have shape [rows, vocab]")
    if candidate_ids.ndim != 2 or q_rows.ndim != 2:
        raise ValueError("candidate_ids and q_rows must have shape [rows, top_k]")
    if candidate_ids.shape != q_rows.shape:
        raise ValueError("candidate_ids and q_rows shapes must match")

    safe_q = torch.nan_to_num(q_rows.float(), nan=0.0)
    dense_q = torch.zeros_like(target_probs, dtype=torch.float32)
    dense_q.scatter_(1, candidate_ids, safe_q)
    return torch.relu(target_probs.float() - dense_q)
