"""FP8 index-K for the DeepSeek-V4.1 low-ratio indexer on gfx942.

The low-ratio (compress ratio 1 and 2) indexer stores its index keys as fp4 and
scores them with AITER's FlyDSL kernels, which use the gfx950-only
``llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4`` instruction.  gfx942 has no FP4
hardware, so that path aborts there.

This module keeps the same math with an fp8 payload instead:

* one index-K row is 128 fp8 bytes plus one fp32 dequant scale, laid out as a
  contiguous 132-byte slot in the pool (the fused row the CUDA C4 indexer uses);
* a query row is quantized per token, and its scale is folded into the indexer
  head weights, which is exact because the scale multiplies every head alike;
* the logits come from AITER's fp8 paged MQA-logits kernels, which support
  gfx942 (``_GLUON_PA_MQA_LOGITS_ARCHS``).

fp8 e4m3 keeps three mantissa bits against fp4's one, so the index selection is
at least as accurate as the fp4 reference it replaces.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, fp8_max

# Bytes of one index-K slot: 128 fp8 payload + 4 bytes of fp32 scale.
INDEX_K_FP8_SLOT_BYTES = 132
# Logits scratch, keyed by element count: the width tracks the context length and
# an agentic session only ever grows it, so a fresh allocation per call would
# strand every block instead of reusing one.
_LOGITS_POOL: dict = {}


def _alloc_logits(rows: int, width: int, device: torch.device) -> torch.Tensor:
    key = (device, rows * width)
    buf = _LOGITS_POOL.get(key)
    if buf is None:
        if len(_LOGITS_POOL) > 8:
            _LOGITS_POOL.clear()
        buf = torch.empty(rows * width, dtype=torch.float32, device=device)
        _LOGITS_POOL[key] = buf
    return buf.view(rows, width)


# Smallest scale divisor: keeps an all-zero row finite instead of scaling by 0.
_AMAX_FLOOR = 1e-4  # mirrored as a literal inside the Triton kernels


@triton.jit
def _quantize_q_fp8_kernel(
    q_ptr,
    out_ptr,
    w_ptr,
    w_out_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    MAX: tl.constexpr,
):
    """One token per program: quantize all heads with one scale, fold it into the weights."""
    t = tl.program_id(0)
    offs = tl.arange(0, H * D)
    v = tl.load(q_ptr + t * H * D + offs).to(tl.float32)
    amax = tl.max(tl.abs(v))
    scale = tl.maximum(amax, 1e-4) / MAX
    q = tl.clamp(v * (1.0 / scale), -MAX, MAX).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + t * H * D + offs, q)

    h = tl.arange(0, H)
    w = tl.load(w_ptr + t * H + h).to(tl.float32)
    tl.store(w_out_ptr + t * H + h, w * scale)


@triton.jit
def _store_index_k_fp8_kernel(
    k_ptr,
    loc_ptr,
    buf_f8_ptr,
    buf_f32_ptr,
    D: tl.constexpr,
    SLOT: tl.constexpr,
    MAX: tl.constexpr,
    NSLOTS: tl.constexpr,
):
    """Quantize one index-K row and store it into its 132-byte slot.

    ``buf_f8_ptr`` and ``buf_f32_ptr`` alias the same pool allocation; the scale
    sits in the four bytes a slot keeps past its payload.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, D)
    v = tl.load(k_ptr + row * D + offs).to(tl.float32)
    amax = tl.max(tl.abs(v))
    scale = tl.maximum(amax, 1e-4) / MAX
    q = tl.clamp(v * (1.0 / scale), -MAX, MAX).to(buf_f8_ptr.dtype.element_ty)

    slot = tl.load(loc_ptr + row)
    # Slot 0 is the reserved dummy row the pool keeps empty; a compressor leaves
    # -1 (or a stale id) on the rows that own no group, so bound the write.
    if slot > 0:
        if slot < NSLOTS:
            base = slot * SLOT
            tl.store(buf_f8_ptr + base + offs, q)
            tl.store(buf_f32_ptr + slot * (SLOT // 4) + D // 4, scale)


def quantize_query_fp8(
    q: torch.Tensor, weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token fp8 query and the head weights with the query scale folded in.

    :param q: ``[tokens, heads, 128]`` bf16, RoPE applied, before quantization.
    :param weights: ``[tokens, heads]``, already carrying ``head_weight_scale``.
    :return: ``q_fp8`` ``[tokens, 1, heads, 128]`` and ``weights`` fp32
        ``[tokens, heads]``.
    """
    tokens, heads, dim = q.shape
    q = q.contiguous()
    out = torch.empty_like(q, dtype=fp8_dtype)
    w = weights.to(torch.float32).contiguous()
    w_out = torch.empty_like(w)
    _quantize_q_fp8_kernel[(tokens,)](
        q, out, w, w_out, H=heads, D=dim, MAX=float(fp8_max), num_warps=4
    )
    return out.view(tokens, 1, heads, dim), w_out


def store_index_k_fp8(
    k: torch.Tensor,
    loc: torch.Tensor,
    cache: torch.Tensor,
    *,
    page_size: int,
) -> None:
    """Quantize and store one index-K row per token.

    :param k: ``[tokens, 128]`` bf16, normalized and rotated.
    :param loc: ``[tokens]`` int64 slot ids; 0 is the reserved dummy.
    :param cache: the layer's pool, ``[pages, page_size * 132]`` uint8.
    """
    tokens, dim = k.shape
    if tokens == 0:
        return
    k = k.contiguous()
    loc = loc.contiguous()
    buf_f8 = cache.view(fp8_dtype)
    buf_f32 = cache.view(torch.float32)
    _store_index_k_fp8_kernel[(tokens,)](
        k,
        loc,
        buf_f8,
        buf_f32,
        D=dim,
        SLOT=INDEX_K_FP8_SLOT_BYTES,
        MAX=float(fp8_max),
        NSLOTS=int(cache.shape[0]),
        num_warps=4,
    )
    _ = page_size  # the slot index already encodes page and offset


def kv_cache_view(cache: torch.Tensor, page_size: int) -> torch.Tensor:
    """The pool as the ``[pages, page_size, 1, 132]`` layout the logits kernels read."""
    pages = cache.shape[0] // page_size
    return cache.view(pages, page_size, 1, INDEX_K_FP8_SLOT_BYTES)

@triton.jit
def _mqa_logits_fp8_kernel(
    q_ptr,
    w_ptr,
    payload_ptr,
    scale_ptr,
    page_table_ptr,
    lens_ptr,
    out_ptr,
    pt_stride,
    out_stride,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    SLOT: tl.constexpr,
    NSLOTS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One query row against one block of its visible compressed slots.

    ``logits[row, j] = sum_h relu(q[row, h] . k[slot(j)]) * w[row, h]``, the
    indexer's score with the query's dequant scale already folded into ``w``.
    """
    row = tl.program_id(0)
    blk = tl.program_id(1)
    ln = tl.load(lens_ptr + row)
    if ln <= 0:
        return

    cols = blk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < ln
    page = tl.load(page_table_ptr + row * pt_stride + cols // PAGE, mask=mask, other=0)
    # A table can carry -1 on slots the request has not filled; clamp so a stray
    # id reads the reserved row instead of walking off the pool.
    slot = tl.maximum(page, 0) * PAGE + (cols % PAGE)
    slot = tl.minimum(slot, NSLOTS - 1)

    offs_d = tl.arange(0, D)
    k = tl.load(
        payload_ptr + slot[:, None] * SLOT + offs_d[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    scale = tl.load(scale_ptr + slot * (SLOT // 4) + D // 4, mask=mask, other=1.0)
    k = (k.to(tl.float32) * scale[:, None]).to(tl.bfloat16)

    offs_h = tl.arange(0, H)
    q = tl.load(q_ptr + row * H * D + offs_h[:, None] * D + offs_d[None, :]).to(
        tl.bfloat16
    )
    w = tl.load(w_ptr + row * H + offs_h)

    s = tl.dot(q, tl.trans(k))  # [H, BLOCK_N] fp32
    s = tl.maximum(s, 0.0) * w[:, None]
    # Columns past the row's length are written as zero rather than masked out:
    # the logits buffer is reused across layers, so it must not keep stale
    # scores from an earlier, longer row.
    tl.store(out_ptr + row * out_stride + cols, tl.where(mask, tl.sum(s, 0), 0.0))


def mqa_logits_fp8(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    cache: torch.Tensor,
    page_table: torch.Tensor,
    lens: torch.Tensor,
    max_seq_len: int,
    *,
    page_size: int,
    block_n: int = 64,
) -> torch.Tensor:
    """``[rows, max_seq_len]`` fp32 logits over each row's visible compressed slots."""
    rows, _, heads, dim = q_fp8.shape
    # The paged top-k transform reads score rows as 16-byte units, so the row
    # stride must be a multiple of four fp32 columns. Round to a coarse bucket as
    # well: a width that tracks the context length would hand the caching
    # allocator a monotonically growing size sequence and strand every block.
    width = (max_seq_len + 255) // 256 * 256
    out = _alloc_logits(rows, width, q_fp8.device)
    if rows == 0 or width == 0:
        return out
    payload = cache.view(fp8_dtype)
    scales = cache.view(torch.float32)
    grid = (rows, triton.cdiv(max_seq_len, block_n))
    _mqa_logits_fp8_kernel[grid](
        q_fp8,
        weights,
        payload,
        scales,
        page_table,
        lens,
        out,
        page_table.stride(0),
        out.stride(0),
        H=heads,
        D=dim,
        PAGE=page_size,
        SLOT=INDEX_K_FP8_SLOT_BYTES,
        NSLOTS=int(cache.shape[0]),
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=1,
    )
    return out
