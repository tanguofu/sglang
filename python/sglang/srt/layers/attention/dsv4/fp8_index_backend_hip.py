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
from sglang.srt.environ import envs

_DEBUGGED: set = set()
_STORED: set = set()


def store_index_k_fp8_group(
    pool,
    layer,
    indexer,
    latent: torch.Tensor,
    freqs: torch.Tensor,
    group_pos: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Normalize, rotate, quantize and store one index-K slot per group.

    :param freqs: the whole RoPE table; the group's row is gathered here.
    :param group_pos: each group's first position.
    :param slots: the index-K slot of each group.
    """
    ratio = layer.compress_ratio
    k = indexer.k_norm(indexer.forward_wk(latent))
    k = rope_tail(k, freqs[group_pos], indexer.rope_head_dim)
    cache = pool.get_index_k_with_scale_buffer(layer.layer_id)
    store_index_k_fp8(
        k,
        slots.to(torch.int64),
        cache,
        page_size=pool.get_index_k_page_size(ratio),
    )
    if envs.SGLANG_DS41_FP8_INDEX_DEBUG.get() and ("g", layer.layer_id) not in _STORED:
        _STORED.add(("g", layer.layer_id))
        torch.cuda.synchronize()
        raw = cache.view(torch.uint8)
        print(
            "[fp8-store-g] layer=%d ratio=%d page=%d slots[:8]=%s k|max|=%.4f nonzero=%.5f"
            % (
                layer.layer_id,
                ratio,
                pool.get_index_k_page_size(ratio),
                slots[:8].tolist(),
                k.abs().max().item(),
                (raw != 0).float().mean().item(),
            ),
            flush=True,
        )


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
    """Store from a per-token batch: the group's first position rotates the key."""
    ratio = layer.compress_ratio
    group_pos = (pos // ratio) * ratio
    k = indexer.k_norm(indexer.forward_wk(latent))
    k = rope_tail(k, freqs_cis[group_pos], indexer.rope_head_dim)
    cache = pool.get_index_k_with_scale_buffer(layer.layer_id)
    store_index_k_fp8(
        k,
        out_loc.to(torch.int64),
        cache,
        page_size=pool.get_index_k_page_size(ratio),
    )
    if envs.SGLANG_DS41_FP8_INDEX_DEBUG.get() and layer.layer_id not in _STORED:
        _STORED.add(layer.layer_id)
        torch.cuda.synchronize()
        raw = cache.view(torch.uint8)
        print(
            "[fp8-store] layer=%d ratio=%d slots[:4]=%s k|max|=%.4f "
            "nonzero=%.4f buf|max|=%d"
            % (
                layer.layer_id,
                ratio,
                out_loc[:4].tolist(),
                k.abs().max().item(),
                (raw != 0).float().mean().item(),
                int(raw.max().item()),
            ),
            flush=True,
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
    logits = mqa_logits_fp8(
        q_fp8,
        weights,
        cache,
        page_table,
        lens.to(torch.int32),
        max_seq_len,
        page_size=page_size,
    )
    if envs.SGLANG_DS41_FP8_INDEX_DEBUG.get():
        _debug_row0(q_fp8, weights, cache, page_table, lens, page_size, logits)
    return logits


def _debug_row0(q_fp8, weights, cache, page_table, lens, page_size, logits) -> None:
    """Compare row 0's logits with a torch reference over the same slots."""
    from sglang.kernels.ops.attention.dsv4.index_k_fp8 import INDEX_K_FP8_SLOT_BYTES
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

    rows, ln = q_fp8.shape[0], int(lens[0].item())
    key = (rows, ln, int(logits.shape[1]))
    if key in _DEBUGGED or rows == 0 or ln == 0:
        return
    _DEBUGGED.add(key)

    dim = q_fp8.shape[-1]
    page_ids = page_table[0, : (ln + page_size - 1) // page_size].to(torch.int64)
    offs = torch.arange(ln, device=page_ids.device) % page_size
    reps = torch.arange(ln, device=page_ids.device) // page_size
    slots = page_ids[reps] * page_size + offs

    # Same flat addressing the kernel uses: slot * 132 bytes, scale word last.
    k_flat = cache.view(fp8_dtype).reshape(-1)
    s_flat = cache.view(torch.float32).reshape(-1)
    cols = torch.arange(dim, device=slots.device)
    payload = k_flat[slots[:, None] * INDEX_K_FP8_SLOT_BYTES + cols[None, :]].float()
    scale = s_flat[slots * (INDEX_K_FP8_SLOT_BYTES // 4) + dim // 4]
    kd = payload * scale.unsqueeze(-1)

    qd = q_fp8[0].view(q_fp8.shape[2], q_fp8.shape[3]).float()
    ref = (qd @ kd.T).relu().mul(weights[0].unsqueeze(-1)).sum(0)
    got = logits[0, :ln].float()
    d = (got - ref).abs()
    order_got = torch.topk(got, min(64, ln)).indices
    order_ref = torch.topk(ref, min(64, ln)).indices
    overlap = len(set(order_got.tolist()) & set(order_ref.tolist()))
    print(
        "[fp8-idx] rows=%d ln=%d pt_w=%d page=%d slots[:3]=%s kd|max|=%.4f "
        "diff[max=%.4f mean=%.4f] top64=%d/64"
        % (
            rows,
            ln,
            page_table.shape[1],
            page_size,
            slots[:3].tolist(),
            kd.abs().max().item(),
            d.max().item(),
            d.mean().item(),
            overlap,
        ),
        flush=True,
    )


def max_seq_len_of(logits) -> int:
    return logits.shape[1]


__all__ = [
    "INDEX_K_FP8_SLOT_BYTES",
    "fp8_index_logits",
    "query_fp8",
    "store_index_k_fp8_group",
    "store_index_k_fp8_layer",
]
