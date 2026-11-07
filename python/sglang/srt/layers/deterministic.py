"""Deterministic reductions for the DeepSeek-V4 attention path.

Two library choices on ROCm make a reduction's grouping depend on where an
element sits rather than on its value:

* hipBLASLt answers large-K GEMMs with stream-K, whose K split follows the work
  distribution over the M tiles. Two bit-identical rows of one batch then get
  different fp32 accumulation orders and come back an ulp apart.
* GroupCoordinator.all_reduce picks between several communicators by message
  size, and the kernels do not share a reduction order.

An ulp is harmless once, but the models here stack 61 layers of attention, MoE
and mHC mixing, and the perturbation grows until a sampled token changes: a
deterministic-looking server whose rows are not reproducible. Everything in
this module exists to make one reduction's order a function of the element
alone, so identical inputs give identical bits.

The overrides are scoped to single calls: hipBLASLt stays in place everywhere a
row's position cannot matter, so this buys determinism without moving the whole
engine off the faster library.
"""

import os
from contextlib import contextmanager

import torch
import triton
import triton.language as tl

# Chosen by the environment so an A/B run does not need a code change.
_OVERRIDE = os.environ.get("SGLANG_DS41_BLAS_LIB")

# Sites to leave on hipBLASLt, to price each one separately:
# `einsum` = wo_a absorb, `linear` = mHC mixing, `scoped` = wo_b output.
_OFF = {name for name in os.environ.get("SGLANG_DS41_DET_OFF", "").split(",") if name}


def _target() -> str:
    """The BLAS library used for row-uniform calls; rocBLAS on ROCm."""
    return _OVERRIDE or "cublas"


@contextmanager
def row_uniform_blas():
    """Run one reduction with an accumulation order that cannot follow the row.

    `cublas` is the ROCm build's name for rocBLAS (the selector predates the
    fork); it keeps one reduction order per output element for these shapes.
    """
    if "scoped" in _OFF:
        yield
        return
    try:
        previous = torch.backends.cuda.preferred_blas_library()
    except Exception:  # not a ROCm build, or the selector is unavailable
        yield
        return
    target = _target()
    if previous == target:
        yield
        return
    try:
        torch.backends.cuda.preferred_blas_library(target)
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(previous)


def row_uniform_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    """`torch.einsum` under :func:`row_uniform_blas`."""
    if "einsum" in _OFF:
        return torch.einsum(equation, *operands)
    with row_uniform_blas():
        return torch.einsum(equation, *operands)


def row_uniform_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """`F.linear` under :func:`row_uniform_blas`."""
    if "linear" in _OFF:
        return torch.nn.functional.linear(x, weight)
    with row_uniform_blas():
        return torch.nn.functional.linear(x, weight)


# --- the lm_head projection -------------------------------------------------
#
# The lm_head is the one GEMM whose output the sampler reads directly, so an ulp
# there moves a token instead of being diluted by the rest of the layer.
# hipBLASLt splits K by the work distribution over the M tiles, so the same row
# accumulates in a different order when the batch holds 32 rows than when
# speculative decoding widens it to 128: measured on gfx942 at [M, 5120] @
# [16160, 5120].T (the TP8 shard of a 129280 vocab), M=32 versus M=128 differ by
# 1 bf16 ulp on 0.035 % of the logits, and all 32 rows carry ~6 moved elements.
#
# Splitting the batch into fixed-size calls makes every call the shape plain
# decode already makes, so rows 0..31 of a 128-row verify batch come back
# bit-identical to the 32-row batch they stand in for. A batch of chunk rows or
# fewer is one unchanged call, so the non-speculative baseline does not move --
# which is why the chunk size must be at least `--max-running-requests`; a
# smaller one would split plain decode too and move it.
#
# Off by default: at M=128 the four calls measured 372 us against 140 us for one
# on an idle gfx942, so treat the ratio as indicative and gate the rollout.
_LMHEAD_CHUNK_ROWS = int(os.environ.get("SGLANG_DS41_LMHEAD_ROW_CHUNK", "0") or 0)


def lm_head_chunk_rows(rows: int) -> int:
    """Rows per lm_head call for this batch, or 0 to keep today's single call."""
    if _LMHEAD_CHUNK_ROWS <= 0 or rows <= _LMHEAD_CHUNK_ROWS:
        return 0
    return _LMHEAD_CHUNK_ROWS


def row_chunked_lm_head(hidden_states: torch.Tensor, weight: torch.Tensor):
    """`hidden_states @ weight.T` as fixed-size row chunks, so the accumulation
    order of a row cannot follow how many rows the batch happens to have."""
    chunk = lm_head_chunk_rows(hidden_states.shape[0])
    if not chunk or hidden_states.dim() != 2 or weight.dim() != 2:
        return torch.matmul(hidden_states, weight.T)
    out = torch.empty(
        (hidden_states.shape[0], weight.shape[0]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    for start in range(0, hidden_states.shape[0], chunk):
        stop = start + chunk
        # copy_ rather than `out=` so hipBLASLt sees the same call it sees today.
        out[start:stop].copy_(torch.matmul(hidden_states[start:stop], weight.T))
    return out


# ---------------------------------------------------------------------------
# The mHC mixing projection: a bit-identical stand-in for rocBLAS' SGEMM
# ---------------------------------------------------------------------------

# Off by default so it can be rolled out gradually and rolled back in a second.
_BITEXACT_MIX = os.environ.get("SGLANG_DS41_MHC_BITEXACT_GEMM", "0") == "1"

# rocBLAS has three row-count regimes, measured on gfx942 at K=20480:
#   M <= 16    17 us    bit-regime A
#   17..32     33 us    bit-regime B
#   M >= 33     7.9 ms  bit-regime C  (one 64-CTA kernel for every M to 16384)
# The kernel below reproduces C, so it takes over at 33; below that rocBLAS is
# both faster and the arithmetic the baseline already produces. The 240x cliff
# and the bit split at 33 are what make a prompt's output depend on how many
# rows the scheduler happened to put in the forward.
_BITEXACT_MIX_MIN_ROWS = 33

# Rows per CTA. The K loop is one serial fp32 FMA chain, so M is the only source
# of parallelism: wide tiles once there are enough rows to fill the machine,
# narrow ones so a small decode batch still spreads over the CUs. Measured best
# tile per row count on gfx942 (BK=32, 4 warps, 2 stages).
_BITEXACT_MIX_BLOCK_K = 32
_BITEXACT_MIX_TILES = ((6144, 128), (4096, 64))
_BITEXACT_MIX_BLOCK_M_DEFAULT = 16

# --- the speculative-verify row counts ------------------------------------
#
# Speculative decoding widens a verify batch to batch_size * verify_width, so
# mixing lands at M = 64..192 instead of the plain-decode M = batch_size. There
# `ceil(M / BM)` is a handful of CTAs on an 80-CU part, the K loop is latency
# bound rather than work bound and the cost is flat in M -- measured 385.9 us at
# M=64 and 384.3 us at M=96 -- so turning speculation on adds ~28 ms/step of
# pure mixing overhead, more than the accept rate buys back.
#
# Off by default. `tune` (also `1`) keeps the kernel but launches it with a tile
# measured for these row counts; `chunk` splits the batch into <=32-row pieces so
# each one stays on rocBLAS' fast path. M <= 32 -- plain decode, the common case
# -- and M > 192 are never touched.
#
# `chunk` is faster but NOT bit-exact: rocBLAS' 17..32-row bit regime is not the
# regime the kernel reproduces, so 32-row chunks move ~99% of the elements by up
# to 5.6e-5 (production-ish scales). It stays reachable for a gated A/B and must
# not be the default.
_MIX_SMALLM_FIX = os.environ.get("SGLANG_DS41_MHC_MIX_SMALLM_FIX", "0").strip().lower()

# Highest row count the fix may touch. Past this the row tiles already spread
# over the machine and the kernel is the faster of the two.
_MIX_SMALLM_MAX_ROWS = 192

# Rows per rocBLAS call for `chunk`. 32 is the top of the 17..32 bit regime, and
# requiring an exact multiple keeps every chunk the same size so one row cannot
# be reduced in a different order from its neighbour.
_MIX_SMALLM_CHUNK_ROWS = 32

# (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages) for `tune`.
# Cutting BLOCK_SIZE_N to 16 puts two CTAs on the 24 mixing columns instead of
# one, which is what buys the 2.2x: at BM=16 the launch is only ceil(M/16) CTAs
# and the rest of the part idles. Measured on gfx942 (K=20480, MIX=24), us/call:
#   16,32,32,4,2 (= today)  451   16,16,64,4,2   268   16,8,64,4,2  1206
#   16,32,128,4,2           494   16,16,64,2,2   211   16,16,128,2,2 195
# Every one of them bit-exact against today's output. The bench sees 451 us where
# the profile sees 384 -- idle-card clocks -- but both are flat in M, so the ratio
# is the transferable part.
_MIX_SMALLM_TILE = (16, 16, 128, 2, 2)


def _mix_smallm_strategy(m: int) -> str:
    """Which small-M path this row count takes, or "" for today's behaviour."""
    if _MIX_SMALLM_FIX in ("0", "", "off", "none"):
        return ""
    if m < _BITEXACT_MIX_MIN_ROWS or m > _MIX_SMALLM_MAX_ROWS:
        return ""
    if _MIX_SMALLM_FIX == "chunk":
        # Any row count, not just multiples of the chunk: the chunk runner pads
        # the tail up to a full chunk, so every rocBLAS call stays in the 32-row
        # bit regime. Rejecting the remainder sent those batches (bs 9..15,
        # 18..23, 25..31 at gamma=3) to the serial Triton kernel, which costs
        # ~30 ms/step -- the c=18 trough.
        return "chunk"
    if _MIX_SMALLM_FIX in ("1", "on", "true", "tune"):
        return "tune"
    return ""


def _chunked_row_uniform_linear(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
) -> None:
    """The projection as a run of equal <=32-row :func:`row_uniform_linear` calls.

    One launch per chunk instead of one per batch: at these row counts rocBLAS
    is ~12x faster per call than the kernel above, and the extra launches are
    an order of magnitude cheaper than the chain they replace.
    """
    rows = x.shape[0]
    chunk = _MIX_SMALLM_CHUNK_ROWS
    tail = rows % chunk
    if tail:
        # A short final call would land in a different rocBLAS row regime and
        # move the bits for exactly those rows. Pad with zeros (each output row
        # only depends on its own input row) so every call is a full chunk.
        x = torch.cat([x, x.new_zeros((chunk - tail, *x.shape[1:]))], dim=0)
    for start in range(0, x.shape[0], chunk):
        stop = start + chunk
        valid = min(stop, rows)
        out[start:valid].copy_(row_uniform_linear(x[start:stop], weight)[: valid - start])


@triton.jit
def _row_uniform_mix_kernel(
    X,
    W,
    O,
    M,
    K,
    sx,
    sw,
    MIX: tl.constexpr,
    MP: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    MASK_K: tl.constexpr,
):
    """One fp32 accumulator per output element, walked over K in increasing order.

    The loop must stay a single chain: any split of K regroups the additions and
    costs ~2.6e-6 of relative accuracy, which is what moves HumanEval.
    """
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    offs_n = tl.program_id(1) * MP + tl.arange(0, MP)
    mask_m = offs_m < M
    mask_n = offs_n < MIX
    acc = tl.zeros([BM, MP], dtype=tl.float32)
    for kb in range(0, K, BK):
        offs_k = kb + tl.arange(0, BK)
        if MASK_K:
            mask_k = offs_k < K
            a = tl.load(
                X + offs_m[:, None] * sx + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            b = tl.load(
                W + offs_n[None, :] * sw + offs_k[:, None],
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
            )
        else:
            a = tl.load(X + offs_m[:, None] * sx + offs_k[None, :], mask=mask_m[:, None], other=0.0)
            b = tl.load(W + offs_n[None, :] * sw + offs_k[:, None], mask=mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(
        O + offs_m[:, None] * MIX + offs_n[None, :],
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _bit_exact_mix_usable(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Every condition the kernel relies on; anything else stays on rocBLAS."""
    return (
        _BITEXACT_MIX
        and torch.version.hip is not None
        and x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.float32
        and weight.dtype == torch.float32
        and x.stride(1) == 1
        and weight.stride(1) == 1
        and weight.shape[1] == x.shape[1]
        and x.shape[0] >= _BITEXACT_MIX_MIN_ROWS
    )


def row_uniform_mhc_mix(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The mHC mixing projection, bit-identical to :func:`row_uniform_linear`.

    rocBLAS' fp32 SGEMM on gfx942 is a serial FMA chain over K, which this kernel
    reproduces element for element -- measured `torch.equal` at every row count
    from 64 to 16384. It is worth 6.6x on the prefill shape (8.05 -> 1.22 ms) and
    17x on decode shapes (7.9 -> 0.46 ms), because rocBLAS launches 64 CTAs no
    matter how many rows it is given.
    """
    if not _bit_exact_mix_usable(x, weight):
        return row_uniform_linear(x, weight)

    m, k = x.shape
    mix = weight.shape[0]
    out = torch.empty((m, mix), dtype=torch.float32, device=x.device)
    strategy = _mix_smallm_strategy(m)
    if strategy == "chunk":
        _chunked_row_uniform_linear(x, weight, out)
        return out
    block_m = _BITEXACT_MIX_BLOCK_M_DEFAULT
    block_k = _BITEXACT_MIX_BLOCK_K
    block_n = max(16, triton.next_power_of_2(mix))
    warps = 4
    stages = 2
    if strategy == "tune":
        block_m, block_n, block_k, warps, stages = _MIX_SMALLM_TILE
    else:
        for min_rows, tile in _BITEXACT_MIX_TILES:
            if m >= min_rows:
                block_m = tile
                break
    # The K loop is one serial chain per output element, so a CTA's cost does not
    # change with how many of them run: at 24 mixing columns and BM=16 the launch
    # is only ceil(M/16) CTAs, a handful of the 80 CUs. `tune` may cut the
    # column tile to put more of them to work on the same chain.
    grid = (triton.cdiv(m, block_m), triton.cdiv(mix, block_n))
    _row_uniform_mix_kernel[grid](
        x,
        weight,
        out,
        m,
        k,
        x.stride(0),
        weight.stride(0),
        MIX=mix,
        MP=block_n,
        BM=block_m,
        BK=block_k,
        MASK_K=(k % block_k != 0),
        num_warps=warps,
        num_stages=stages,
    )
    return out
