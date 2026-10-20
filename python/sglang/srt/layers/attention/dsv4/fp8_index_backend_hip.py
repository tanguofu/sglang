"""FP8 low-ratio indexer plumbing for gfx942.

Two entry points the HIP low-ratio backend calls instead of AITER's FlyDSL fp4
kernels: one stores index keys as fp8, one produces the logits the paged top-k
transform consumes.  Both keep the shapes and the relu-then-weighted-sum
semantics of the fp4 path, so the candidate logic and the top-k transform above
them are unchanged.
"""

from __future__ import annotations

from typing import Tuple

import torch

from sglang.kernels.ops.attention.dsv4.index_k_fp8 import (
    INDEX_K_FP8_SLOT_BYTES,
    mqa_logits_fp8,
    quantize_query_fp8,
    store_index_k_fp8,
)
from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail


def store_index_k_fp8_layer(
    pool,
    layer,
    indexer,
    latent: torch.Tensor,
    freqs_cis: torch.Tensor,
    pos: torch.Tensor,
    out_loc: torch.Tensor,
    layer_id: int,
) -> None:
    """Normalize, rotate, quantize and store one index-K slot per token."""
    ratio = layer.compress_ratio
    k = indexer.k_norm(indexer.forward_wk(latent))
    # A compressed key is written once per group and carries the group's first
    # position, exactly as the fp4 store does.
    group_pos = (pos // ratio) * ratio
    k = rope_tail(k, freqs_cis[group_pos], indexer.rope_head_dim)
    store_index_k_fp8(
        k,
        out_loc.to(torch.int64),
        pool.get_index_k_with_scale_buffer(layer_id),
        page_size=pool.get_index_k_page_size(ratio),
    )


def query_fp8(layer, x, q_lora, pos) -> Tuple[torch.Tensor, torch.Tensor]:
    """Indexer query as fp8 plus the head weights carrying its dequant scale."""
    indexer = layer.indexer
    q, _ = indexer.wq_b(q_lora)
    q = q.view(q.shape[0], indexer.n_local_heads, indexer.index_head_dim)
    # No fp4 fake quantization: the payload is fp8, so round once, in the
    # quantizer that also folds the scale into the weights.
    q = rope_tail(q, layer.freqs_cis[pos], indexer.rope_head_dim)
    return quantize_query_fp8(q, indexer.head_weights(x))


def fp8_index_logits(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    cache: torch.Tensor,
    page_size: int,
    page_table: torch.Tensor,
    lens: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """``[rows, max_seq_len]`` logits over each row's visible compressed slots.

    One kernel serves both decode (a row per request) and the scored rows of a
    prefill: a row's visible range is ``lens[row]`` compressed positions reached
    through its own page-table row.
    """
    return mqa_logits_fp8(
        q_fp8,
        weights,
        cache,
        page_table,
        lens.to(torch.int32),
        max_seq_len,
        page_size=page_size,
    )


__all__ = [
    "INDEX_K_FP8_SLOT_BYTES",
    "fp8_index_logits",
    "query_fp8",
    "store_index_k_fp8_layer",
]
