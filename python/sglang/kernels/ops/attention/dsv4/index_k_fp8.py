"""FP8 index-K for the DeepSeek-V4.1 low-ratio indexer on gfx942.

The low-ratio (compress ratio 1 and 2) indexer stores its index keys as fp4 and
scores them with AITER's FlyDSL kernels, which use the gfx950-only
``llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4`` instruction.  gfx942 has no FP4
hardware, so that path aborts there.

This module keeps the same math with an fp8 payload instead:

* one index-K row is 128 fp8 bytes plus one fp32 dequant scale, laid out as a
  contiguous 132-byte slot in the pool;
* a query row is quantized per token, and its scale is folded into the indexer
  head weights, which is exact because the scale multiplies every head alike;
* the logits kernel keeps the reference semantics: ``sum_h relu(q_h . k) * w_h``.

fp8 e4m3 keeps three mantissa bits against fp4's one, so the index selection is
at least as accurate as the fp4 reference it replaces.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, fp8_max

# Bytes of one index-K slot: 128 fp8 payload + 4 bytes of fp32 scale.
INDEX_K_FP8_SLOT_BYTES = 132

# Columns of compressed positions one `_mqa_logits_fp8_kernel` program scores.
# One program owns BLOCK_N of a row's columns and reduces the whole head and
# head_dim axes itself, so BLOCK_N only changes how the row is cut, never the
# arithmetic: the kernel is bit-exact across BLOCK_N (verified 64 vs 128 vs 256
# at lc = 320 / 3328 / 32768, torch.equal).  It is not free, though: at
# BLOCK_N=64 the [32, 64] output tile leaves the four warps with too little
# work to hide the fp8->bf16 conversions, and the kernel runs at roughly a
# third of the rate BLOCK_N=128 reaches on gfx942 (3.0-3.2x measured at
# rows = 64..16384, lc = 8192..65536).  Fall back with
# SGLANG_HIP_DSV41_INDEX_LOGITS_BLOCK_N=64.
def _logits_block_n() -> int:
    return int(os.environ.get("SGLANG_HIP_DSV41_INDEX_LOGITS_BLOCK_N", "128"))


# Query rows one `_mqa_logits_fp8_tiled_kernel` program scores.  Rows of a tile
# are stacked into the M axis of one dot, so the K tile is loaded once for the
# whole tile instead of once per row.  1 keeps the one-row-per-program kernel.
# A tile must not straddle two requests: it reads the page table of its first
# row only, and every token row of a request carries that request's pages, so
# the mapping is only well defined inside one request.  The caller builds the
# tiles from the per-request row counts, which the extend path already has on
# the host; decode (one row per request, captured in a graph) never tiles.
def _logits_block_m() -> int:
    return int(os.environ.get("SGLANG_HIP_DSV41_INDEX_LOGITS_BLOCK_M", "8"))


# Score only the candidate blocks the source layer published, instead of the
# whole context, on the consumer layers.  The consumer's top-k runs inside those
# blocks, so a score outside them is computed and then dropped: skipping them
# leaves the compact row bit-identical.  See mqa_logits_fp8_candidates.
def _score_candidates_only() -> bool:
    return bool(int(os.environ.get("SGLANG_HIP_DSV41_INDEX_CANDIDATE_ONLY", "1")))

# Logits scratch, keyed by element count.  The width tracks the context length
# and an agentic session only ever grows it, so a fresh allocation per layer per
# step would hand the caching allocator a monotonically growing size sequence and
# strand every block instead of reusing one.
_LOGITS_POOL: dict = {}

# Smallest scale divisor: keeps an all-zero row finite instead of scaling by 0.
# Mirrored as a literal inside the Triton kernels.
_AMAX_FLOOR = 1e-4


def _alloc_logits(rows: int, width: int, device: torch.device) -> torch.Tensor:
    # Inside a capture, allocate from the graph pool the way torch expects: the
    # pooled block is for eager steps only.
    if torch.cuda.is_current_stream_capturing():
        return torch.empty((rows, width), dtype=torch.float32, device=device)
    key = (device, rows * width)
    buf = _LOGITS_POOL.get(key)
    if buf is None:
        if len(_LOGITS_POOL) > 8:
            _LOGITS_POOL.clear()
        buf = torch.empty(rows * width, dtype=torch.float32, device=device)
        _LOGITS_POOL[key] = buf
    return buf.view(rows, width)


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
    # The grid spans the whole page-table bucket, so most blocks of a short
    # request own no column at all.  Returning here skips a dot product whose
    # result would be masked away; the caller zeroes the buffer first, so the
    # skipped columns still read as zero.
    if blk * BLOCK_N >= ln:
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


@triton.jit
def _mqa_logits_fp8_tiled_kernel(
    q_ptr,
    w_ptr,
    payload_ptr,
    scale_ptr,
    page_table_ptr,
    lens_ptr,
    tile_lo_ptr,
    tile_hi_ptr,
    out_ptr,
    pt_stride,
    out_stride,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    SLOT: tl.constexpr,
    NSLOTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """BLOCK_M query rows of one request against one block of their slots.

    Same arithmetic as `_mqa_logits_fp8_kernel`, so the scores are bit-identical
    for the same BLOCK_N: the only thing a tile changes is that the K tile is
    gathered once for BLOCK_M rows.  Rows are stacked along the dot's M axis as
    [BLOCK_M * H, D] -- a row's own H heads stay a contiguous M slice, because q
    is [rows, heads, D] -- and the [BLOCK_M * H, BLOCK_N] result is folded back
    to [BLOCK_M, H, BLOCK_N] before the per-row head reduction.

    The page table is read at the tile's first row.  Every row of a tile belongs
    to one request and a request's token rows all carry the same pages, so that
    one row's mapping is the tile's; a tile that straddled two requests would
    score one request's queries against the other's keys.
    """
    tile = tl.program_id(0)
    blk = tl.program_id(1)
    lo = tl.load(tile_lo_ptr + tile)
    hi = tl.load(tile_hi_ptr + tile)
    m = tl.arange(0, BLOCK_M)
    rows = lo + m
    rmask = rows < hi
    ln = tl.load(lens_ptr + rows, mask=rmask, other=0)
    cols = blk * BLOCK_N + tl.arange(0, BLOCK_N)
    # Rows of a tile advance together, so the block is skipped only when the
    # longest row of the tile reaches no further; a tile whose rows are short
    # would otherwise pay for the whole rectangle.  The caller zeroes the
    # buffer, so the skipped columns still read as zero.
    if tl.max(ln) <= blk * BLOCK_N:
        return
    cmask = cols[None, :] < ln[:, None]

    offs_d = tl.arange(0, D)
    page = tl.load(page_table_ptr + lo * pt_stride + cols // PAGE)
    slot = tl.minimum(tl.maximum(page, 0) * PAGE + (cols % PAGE), NSLOTS - 1)
    k = tl.load(payload_ptr + slot[:, None] * SLOT + offs_d[None, :]).to(tl.bfloat16)
    scale = tl.load(scale_ptr + slot * (SLOT // 4) + D // 4)
    k = (k.to(tl.float32) * scale[:, None]).to(tl.bfloat16)

    q = tl.load(
        q_ptr + rows[:, None] * (H * D) + tl.arange(0, H * D)[None, :],
        mask=rmask[:, None],
        other=0.0,
    )
    q = tl.reshape(q, (BLOCK_M * H, D)).to(tl.bfloat16)
    w = tl.load(
        w_ptr + rows[:, None] * H + tl.arange(0, H)[None, :],
        mask=rmask[:, None],
        other=0.0,
    )

    s = tl.dot(q, tl.trans(k))  # [BLOCK_M * H, BLOCK_N] fp32
    s = tl.reshape(s, (BLOCK_M, H, BLOCK_N))
    s = tl.maximum(s, 0.0) * w[:, :, None]
    tl.store(
        out_ptr + rows[:, None] * out_stride + cols[None, :],
        tl.where(cmask, tl.sum(s, 1), 0.0),
        mask=rmask[:, None],
    )


@triton.jit
def _mqa_logits_fp8_candidate_kernel(
    q_ptr,
    w_ptr,
    payload_ptr,
    scale_ptr,
    page_table_ptr,
    lens_ptr,
    ids_ptr,
    compact_lens_ptr,
    out_ptr,
    pt_stride,
    ids_stride,
    out_stride,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    SLOT: tl.constexpr,
    NSLOTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One query row against one block of the compact candidate row.

    Column ``c`` of the compact row is compressed position
    ``ids[row, c // BLOCK_SIZE] * BLOCK_SIZE + c % BLOCK_SIZE``, the layout
    `gather_candidate_blocks` produces from the full logits.  Scoring it here
    instead of scoring the whole context and gathering is bit-exact for the
    columns the top-k reads: the arithmetic per position is the production
    kernel's, and a column the gather would fill with -inf (an unused id, or a
    position past the row's reach) is written as -inf here too.

    The whole rectangle is written, so the buffer needs no zeroing and the
    columns past a row's kept blocks read as -inf exactly as the gather leaves
    them.
    """
    row = tl.program_id(0)
    blk = tl.program_id(1)
    # A row keeps at most ceil(lc / BLOCK_SIZE) blocks, so the tail of a short
    # row owns no candidate: leave it -inf, the buffer was filled with -inf.
    if blk * BLOCK_N >= tl.load(compact_lens_ptr + row):
        return
    c = blk * BLOCK_N + tl.arange(0, BLOCK_N)
    ln = tl.load(lens_ptr + row)
    bid = tl.load(ids_ptr + row * ids_stride + c // BLOCK_SIZE)
    pos = bid * BLOCK_SIZE + (c % BLOCK_SIZE)
    valid = (bid >= 0) & (pos < ln)
    safe = tl.maximum(pos, 0)
    offs_d = tl.arange(0, D)
    page = tl.load(
        page_table_ptr + row * pt_stride + safe // PAGE, mask=valid, other=0
    )
    slot = tl.minimum(tl.maximum(page, 0) * PAGE + (safe % PAGE), NSLOTS - 1)

    k = tl.load(
        payload_ptr + slot[:, None] * SLOT + offs_d[None, :], mask=valid[:, None], other=0.0
    ).to(tl.bfloat16)
    scale = tl.load(
        scale_ptr + slot * (SLOT // 4) + D // 4, mask=valid, other=1.0
    )
    k = (k.to(tl.float32) * scale[:, None]).to(tl.bfloat16)

    offs_h = tl.arange(0, H)
    q = tl.load(q_ptr + row * H * D + offs_h[:, None] * D + offs_d[None, :]).to(
        tl.bfloat16
    )
    w = tl.load(w_ptr + row * H + offs_h)

    s = tl.dot(q, tl.trans(k))  # [H, BLOCK_N] fp32
    s = tl.maximum(s, 0.0) * w[:, None]
    tl.store(
        out_ptr + row * out_stride + c, tl.where(valid, tl.sum(s, 0), float("-inf"))
    )


# Row tiles, keyed by the request layout.  The build is a handful of tiny
# kernels, so it costs a flat ~0.4 ms no matter how small the batch is; a
# chunked prefill repeats one layout across its chunks, so keeping the result
# makes the steady state pay it once.
_TILE_INDEX_CACHE: dict = {}

# Columns below which tiling cannot pay its own index build back.
_TILE_MIN_COLUMNS = 4_000_000


def _logits_tiles(row_splits, block_m, device):
    """Per-row-tile ``(lo, hi)`` of a slice split into `row_splits` requests.

    Tiles are cut inside a request, never across two: a tile reads the page
    table of its first row only.
    """
    key = (device, block_m, tuple(row_splits))
    cached = _TILE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    counts = torch.tensor(row_splits, dtype=torch.int32, device=device)
    starts = torch.cumsum(counts, 0) - counts
    per_req = (counts + block_m - 1) // block_m
    # the tile count is known on the host, so the grid needs no device sync
    total = sum((n + block_m - 1) // block_m for n in row_splits)
    which = torch.repeat_interleave(
        torch.arange(counts.shape[0], dtype=torch.int32, device=device), per_req
    )
    tile0 = torch.cumsum(per_req, 0) - per_req
    within = torch.arange(total, dtype=torch.int32, device=device) - torch.repeat_interleave(
        tile0, per_req
    )
    lo = starts[which] + within * block_m
    hi = torch.minimum(lo + block_m, starts[which] + counts[which])
    if len(_TILE_INDEX_CACHE) > 32:
        _TILE_INDEX_CACHE.clear()
    _TILE_INDEX_CACHE[key] = (lo, hi, total)
    return _TILE_INDEX_CACHE[key]


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
    _store_index_k_fp8_kernel[(tokens,)](
        k,
        loc,
        cache.view(fp8_dtype),
        cache.view(torch.float32),
        D=dim,
        SLOT=INDEX_K_FP8_SLOT_BYTES,
        MAX=float(fp8_max),
        NSLOTS=int(cache.numel()) // INDEX_K_FP8_SLOT_BYTES,
        num_warps=4,
    )
    _ = page_size  # the slot id already encodes page and offset


def mqa_logits_fp8(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    cache: torch.Tensor,
    page_table: torch.Tensor,
    lens: torch.Tensor,
    max_seq_len: int,
    *,
    page_size: int,
    block_n: Optional[int] = None,
    block_m: Optional[int] = None,
    row_splits: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """``[rows, max_seq_len]`` fp32 logits over each row's visible compressed slots.

    :param row_splits: row counts of the requests packed into ``q_fp8``, in row
        order.  Row tiling needs it to keep a tile inside one request; without
        it (or with ``block_m == 1``) one program scores one row.
    """
    rows, _, heads, dim = q_fp8.shape
    if block_n is None:
        block_n = _logits_block_n()
    if block_m is None:
        block_m = _logits_block_m()
    # The paged top-k transform reads a score row as 16-byte units, so its width
    # must be a multiple of four fp32 columns; round to a coarse bucket as well so
    # a growing context does not allocate a new rectangle on every step.
    width = (max_seq_len + 255) // 256 * 256
    out = _alloc_logits(rows, width, q_fp8.device)
    if rows == 0 or width == 0:
        return out
    # The pooled buffer can still hold a longer row's scores, and the kernel now
    # leaves every column past a row's length untouched.
    out.zero_()
    payload = cache.view(fp8_dtype)
    scales = cache.view(torch.float32)
    nslots = int(cache.numel()) // INDEX_K_FP8_SLOT_BYTES
    lens_i32 = lens.to(torch.int32)
    # Tiling needs the tile index, which is built from a host-side list and so
    # cannot be produced inside a capture; a captured step keeps one row per
    # program.  Decode rows are one per request and never tile either.
    if (
        block_m > 1
        and row_splits is not None
        and rows * max_seq_len >= _TILE_MIN_COLUMNS
        and not torch.cuda.is_current_stream_capturing()
    ):
        lo, hi, tiles = _logits_tiles(row_splits, block_m, q_fp8.device)
        _mqa_logits_fp8_tiled_kernel[(tiles, triton.cdiv(max_seq_len, block_n))](
            q_fp8,
            weights,
            payload,
            scales,
            page_table,
            lens_i32,
            lo,
            hi,
            out,
            page_table.stride(0),
            out.stride(0),
            H=heads,
            D=dim,
            PAGE=page_size,
            SLOT=INDEX_K_FP8_SLOT_BYTES,
            NSLOTS=nslots,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=1,
        )
        return out
    grid = (rows, triton.cdiv(max_seq_len, block_n))
    _mqa_logits_fp8_kernel[grid](
        q_fp8,
        weights,
        payload,
        scales,
        page_table,
        lens_i32,
        out,
        page_table.stride(0),
        out.stride(0),
        H=heads,
        D=dim,
        PAGE=page_size,
        SLOT=INDEX_K_FP8_SLOT_BYTES,
        NSLOTS=nslots,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=1,
    )
    return out


def mqa_logits_fp8_candidates(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    cache: torch.Tensor,
    page_table: torch.Tensor,
    lens: torch.Tensor,
    ids: torch.Tensor,
    compact_lens: torch.Tensor,
    *,
    page_size: int,
    block_size: int,
    block_n: Optional[int] = None,
) -> torch.Tensor:
    """``[rows, ids.shape[1] * block_size]`` fp32: each row's score at its
    candidate positions only.

    This is the compact row `gather_candidate_blocks` builds by picking columns
    out of the full logits, computed directly instead.  Bit-exact with it: same
    position -> same slot, same dot, and a column the gather would leave -inf is
    -inf here.  The consumer layer's top-k reads no other column, so its
    selection is unchanged.

    :param lens: each row's compressed context length.
    :param ids: ``[rows, topk_blocks]`` int32 candidate block ids, -1 padded.
    :param compact_lens: ``[rows]`` int32 width of each row's kept prefix.
    """
    rows, _, heads, dim = q_fp8.shape
    if block_n is None:
        block_n = _logits_block_n()
    width = ids.shape[1] * block_size
    out = torch.full(
        (rows, width), float("-inf"), dtype=torch.float32, device=q_fp8.device
    )
    if rows == 0 or width == 0:
        return out
    _mqa_logits_fp8_candidate_kernel[(rows, triton.cdiv(width, block_n))](
        q_fp8,
        weights,
        cache.view(fp8_dtype),
        cache.view(torch.float32),
        page_table,
        lens.to(torch.int32),
        ids,
        compact_lens.to(torch.int32),
        out,
        page_table.stride(0),
        ids.stride(0),
        out.stride(0),
        H=heads,
        D=dim,
        PAGE=page_size,
        SLOT=INDEX_K_FP8_SLOT_BYTES,
        NSLOTS=int(cache.numel()) // INDEX_K_FP8_SLOT_BYTES,
        BLOCK_SIZE=block_size,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=1,
    )
    return out


