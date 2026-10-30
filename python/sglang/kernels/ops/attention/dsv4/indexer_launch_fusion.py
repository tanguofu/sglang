"""Launch-fusion kernels for the DS4.1 indexer / low-ratio compressor path.

Decode on gfx942 runs this path fully inside a CUDA graph, so every ATen op
costs its ~2.6 us launch floor and nothing else.  Two blocks on this path are
made of integer bookkeeping, bit-preserving permutations and one fp32 complex
multiply, which means they collapse into a handful of Triton launches without
touching any accumulation order:

* :func:`fused_pair_partners_and_slots` replaces
  ``DeepseekV4AttnBackend._low_ratio_pair_partners`` plus the two integer
  selects its caller computes (~38 ATen launches -> 2).  The block performs **no
  floating-point arithmetic at all**: integer compare, boolean logic, a ring
  read, a roll (a bit-preserving permutation) and two selects.  There is no
  accumulation order to change, so it is not in the forbidden ``reduce_kernel``
  class.
* :func:`rope_tail_fused` replaces :func:`rope_tail` and the ``freqs[positions]``
  gather that feeds it (~5 ATen launches -> 1).  Its only arithmetic is one
  complex64 multiply in fp32 followed by one round to bf16, both correctly
  rounded and both in a fixed expression order.
* :func:`group_pos_and_slots` is the integer-select half on its own, for when
  the pairing fusion is switched off (4 ATen launches -> 2).

The ring hazard in the first block is respected by splitting it in two: the
kernel that reads the ring and derives every index is a separate launch from the
kernel that writes the ring, so the read is ordered before the write by stream
semantics rather than by an in-kernel barrier.

Every kernel is gated by an env switch defaulting to off, and every guard that
cannot be checked before launch falls back to the untouched torch path.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Read per call so a process can A/B both sides of a fusion without a restart --
# that is how the bit-exactness harness compares them.
ENV_FUSE_PAIR_PARTNERS = "SGLANG_DS41_FUSE_PAIR_PARTNERS"
ENV_FUSE_ROPE_TAIL = "SGLANG_DS41_FUSE_ROPE_TAIL"
ENV_FUSE_INT_SELECTS = "SGLANG_DS41_FUSE_INT_SELECTS"


def fuse_enabled(name: str) -> bool:
    return os.environ.get(name, "0") != "0"


def _pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def _warps(block: int) -> int:
    return max(1, min(8, block // 32))


# ---------------------------------------------------------------------------
# rope_tail
# ---------------------------------------------------------------------------


@triton.jit
def _rope_tail_kernel(
    X,
    F,
    OUT,
    POS,
    HEADS,
    D: tl.constexpr,
    RD: tl.constexpr,
    RD2: tl.constexpr,
    INV: tl.constexpr,
    GATHER: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per row of ``x`` (a row is ``D`` features, ``HEADS`` rows per
    token).  The head is copied bit-for-bit; the tail is rotated as complex64 in
    fp32 and rounded once, exactly as ``rope_tail`` does."""
    row = tl.program_id(0)
    off = tl.arange(0, BLOCK_D)
    m = off < D
    x = tl.load(X + row * D + off, mask=m, other=0.0).to(tl.float32)

    cut = D - RD
    tail = off >= cut
    rel = off - cut
    j = tl.where(rel > 0, rel, 0) // 2

    t = row // HEADS
    if GATHER:
        t = tl.load(POS + t).to(tl.int32)
    fbase = t * RD2 + j

    fr = tl.load(F + fbase * 2, mask=tail, other=0.0)
    fi = tl.load(F + fbase * 2 + 1, mask=tail, other=0.0)
    if INV:
        fi = -fi

    base = cut + 2 * j
    ar = tl.load(X + row * D + base, mask=tail, other=0.0).to(tl.float32)
    ai = tl.load(X + row * D + base + 1, mask=tail, other=0.0).to(tl.float32)
    # Same expression order as thrust::complex::operator* on finite operands:
    # two fp32 multiplies and one add per component, no contraction.
    real = ar * fr - ai * fi
    imag = ar * fi + ai * fr

    rot = tl.where((rel & 1) == 1, imag, real)
    tl.store(OUT + row * D + off, tl.where(tail, rot, x).to(OUT.dtype.element_ty), mask=m)


def rope_tail_fusable(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rope_dim: int,
    positions: Optional[torch.Tensor] = None,
) -> bool:
    """True when :func:`rope_tail_fused` can take over.  Every condition here is
    a structural one; anything else keeps the torch path."""
    return (
        x.is_cuda
        and x.is_contiguous()
        and x.ndim in (2, 3)
        and freqs.is_cuda
        and freqs.is_contiguous()
        and freqs.ndim == 2
        and freqs.is_complex()
        and freqs.dtype == torch.complex64
        and rope_dim > 0
        and rope_dim % 2 == 0
        and rope_dim <= x.shape[-1]
        and freqs.shape[-1] == rope_dim // 2
        # Without positions the table is indexed by each token's own ordinal, so
        # it must have exactly one row per token; with them its length is
        # whatever the caller's index range needs.
        and (positions is not None or freqs.shape[0] == x.shape[0])
    )


def rope_tail_fused(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rope_dim: int,
    inverse: bool = False,
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``rope_tail`` with the ``freqs[positions]`` gather folded in.

    :param freqs: the whole ``[L, rope_dim // 2]`` complex64 table.
    :param positions: per-*token* table indices, or None to index the table by
        each token's own ordinal (what ``rope_tail`` does today).
    """
    assert rope_tail_fusable(x, freqs, rope_dim)
    d = x.shape[-1]
    rows = x.numel() // d
    heads = rows // x.shape[0]
    out = torch.empty_like(x)
    f = torch.view_as_real(freqs).view(-1, 2)
    _rope_tail_kernel[(rows,)](
        x,
        f,
        out,
        # Unused when GATHER is false, but the signature still needs an int row.
        positions
        if positions is not None
        else torch.empty(0, dtype=torch.int32, device=x.device),
        heads,
        D=d,
        RD=rope_dim,
        RD2=rope_dim // 2,
        INV=inverse,
        GATHER=positions is not None,
        BLOCK_D=_pow2(d),
        num_warps=_warps(_pow2(d)),
    )
    return out


# ---------------------------------------------------------------------------
# _low_ratio_pair_partners (+ the two integer selects its caller computes)
# ---------------------------------------------------------------------------


@triton.jit
def _pair_partners_read_kernel(
    REQ,
    POS,
    RAW_LOC,
    OUT_LOC,
    BUF,
    KV,
    SCORE,
    P_KV,
    P_SCORE,
    W_LOC,
    G_POS,
    SLOTS,
    T,
    S,
    D: tl.constexpr,
    RING,
    TRIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Everything that only reads the ring, plus the two integer selects.

    Writes nothing into the ring, so this launch and the write launch below can
    be queued back to back: the read is ordered before the write by the stream.
    """
    i = tl.program_id(0)
    im1 = (i - 1 + T) % T

    pos_i = tl.load(POS + i)
    req_i = tl.load(REQ + i)
    # `pad` is `raw_out_loc == 0`; computing it here keeps the bool tensor (and
    # its launch) out of the graph entirely.
    pad_i = (tl.load(RAW_LOC + i) == 0).to(tl.int32)
    pad_m1 = (tl.load(RAW_LOC + im1) == 0).to(tl.int32)

    # --- ring read -------------------------------------------------------
    # A negative state_loc means "no slot" and lands on the empty last row,
    # which is what `buf[-1]` selects in torch.  Both arms of the select are
    # plain arithmetic, so neither side is ever dereferenced out of bounds.
    read_pos = pos_i - 1
    bad = (pad_i != 0) | (read_pos < 0)
    rl = tl.where(bad, S - 1, req_i * RING + (read_pos % RING)).to(tl.int32)

    d = tl.arange(0, BLOCK_D)
    dm = d < D
    two_d = 2 * D
    ck = tl.load(BUF + rl * two_d + d, mask=dm, other=0.0)
    cs = tl.load(BUF + rl * two_d + D + d, mask=dm, other=0.0)

    # --- in_batch --------------------------------------------------------
    req_m1 = tl.load(REQ + im1)
    pos_m1 = tl.load(POS + im1)
    in_batch = (
        (i > 0)
        & (req_i == req_m1)
        & (pos_i == pos_m1 + 1)
        & (pad_i == 0)
        & (pad_m1 == 0)
    )

    kv_r = tl.load(KV + im1 * D + d, mask=dm, other=0.0)
    sc_r = tl.load(SCORE + im1 * D + d, mask=dm, other=0.0)
    tl.store(P_KV + i * D + d, tl.where(in_batch, kv_r, ck), mask=dm)
    tl.store(P_SCORE + i * D + d, tl.where(in_batch, sc_r, cs), mask=dm)

    # --- ring write location ---------------------------------------------
    keep = pad_i == 0
    if TRIM:
        if i < T - RING:
            req_f = tl.load(REQ + i + RING)
            pad_f = (tl.load(RAW_LOC + i + RING) == 0).to(tl.int32)
            keep = keep & ((req_i != req_f) | (pad_f != 0))
    write_pos = tl.where(keep, pos_i, -1)
    wl = tl.where(write_pos < 0, -1, req_i * RING + (write_pos % RING))
    tl.store(W_LOC + i, wl.to(tl.int32))

    # --- the two integer selects -----------------------------------------
    # torch's % follows Python, so `pos % 2` is 0 or 1 for every int and
    # `pos & 1` is the same value under two's complement (and sign-safe).
    tl.store(G_POS + i, (pos_i - (pos_i & 1)).to(G_POS.dtype.element_ty))
    ol = tl.load(OUT_LOC + i)
    tl.store(SLOTS + i, tl.where(ol >= 0, ol, 0).to(SLOTS.dtype.element_ty))


@triton.jit
def _pair_partners_write_kernel(
    BUF, W_LOC, KV, SCORE, T, S, D: tl.constexpr, BLOCK_D: tl.constexpr
):
    """``T + 1`` programs: one per token, plus one that resets the empty row the
    way ``set_state_by_state_loc``'s trailing ``clear()`` does."""
    i = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    dm = d < D
    two_d = 2 * D
    if i < T:
        # `loc < 0` means "no slot"; those rows are simply not written, which is
        # what the torch path ends up with anyway (it writes the sentinel row and
        # then clears it).
        loc = tl.load(W_LOC + i).to(tl.int32)
        ok = dm & (loc >= 0)
        v = tl.load(KV + i * D + d, mask=ok, other=0.0)
        s = tl.load(SCORE + i * D + d, mask=ok, other=0.0)
        tl.store(BUF + loc * two_d + d, v, mask=ok)
        tl.store(BUF + loc * two_d + D + d, s, mask=ok)
    else:
        tl.store(BUF + (S - 1) * two_d + d, 0.0, mask=dm)
        tl.store(BUF + (S - 1) * two_d + D + d, float("-inf"), mask=dm)


def pair_partners_fusable(
    buffer: torch.Tensor, kv: torch.Tensor, score: torch.Tensor, req, pos, raw_loc, out_loc
) -> bool:
    """Structural preconditions for :func:`fused_pair_partners_and_slots`."""
    if not (
        kv.is_cuda
        and kv.is_contiguous()
        and kv.ndim == 2
        and kv.dtype == torch.float32
        and score.shape == kv.shape
        and score.dtype == kv.dtype
        and score.is_contiguous()
        and buffer.is_cuda
        and buffer.is_contiguous()
        and buffer.ndim == 2
        and buffer.shape[-1] == 2 * kv.shape[-1]
        and req.shape == pos.shape == raw_loc.shape == out_loc.shape
        and pos.shape[0] == kv.shape[0]
    ):
        return False
    d = kv.shape[-1]
    # Triton block sizes are powers of two and the ring row stride is 2*D.
    return 0 < d <= 4096 and _pow2(d) <= 4096


def fused_pair_partners_and_slots(
    *,
    buffer: torch.Tensor,
    ring: int,
    kv: torch.Tensor,
    score: torch.Tensor,
    req: torch.Tensor,
    pos: torch.Tensor,
    raw_loc: torch.Tensor,
    out_loc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-launch replacement for ``_low_ratio_pair_partners`` plus the
    ``group_pos`` / ``slots`` selects computed by its caller.

    Returns ``(partner_kv, partner_score, group_pos, slots)`` and leaves the
    ring write queued on the current stream *after* the read, which is the
    ordering the original comments at the call site require.
    """
    assert pair_partners_fusable(buffer, kv, score, req, pos, raw_loc, out_loc)
    t = kv.shape[0]
    if t == 0:
        return kv, score, pos, out_loc
    d = kv.shape[-1]
    block_d = _pow2(d)
    warps = _warps(block_d)

    partner_kv = torch.empty_like(kv)
    partner_score = torch.empty_like(score)
    write_loc = torch.empty((t,), dtype=torch.int32, device=kv.device)
    group_pos = torch.empty_like(pos)
    slots = torch.empty_like(out_loc)

    _pair_partners_read_kernel[(t,)](
        req,
        pos,
        raw_loc,
        out_loc,
        buffer,
        kv,
        score,
        partner_kv,
        partner_score,
        write_loc,
        group_pos,
        slots,
        t,
        buffer.shape[0],
        D=d,
        RING=int(ring),
        TRIM=t > int(ring),
        BLOCK_D=block_d,
        num_warps=warps,
    )
    _pair_partners_write_kernel[(t + 1,)](
        buffer, write_loc, kv, score, t, buffer.shape[0], D=d, BLOCK_D=block_d, num_warps=warps
    )
    return partner_kv, partner_score, group_pos, slots


# ---------------------------------------------------------------------------
# the two integer selects, on their own
# ---------------------------------------------------------------------------


def group_pos_and_slots(pos: torch.Tensor, out_loc: torch.Tensor):
    """``group_pos = where(pos % 2 == 1, pos - 1, pos)`` and
    ``slots = where(out_loc >= 0, out_loc, zeros_like(out_loc))``.

    Both are pure integer identities with no floating point and nothing to
    round: ``pos % 2`` is 0 or 1 for every int under torch's Python-style
    remainder, so ``pos - pos % 2`` is the same number, and ``clamp(min=0)`` is
    the same select.  Four ATen launches become two.
    """
    if fuse_enabled(ENV_FUSE_INT_SELECTS):
        return pos - pos % 2, out_loc.clamp(min=0)
    return (
        torch.where(pos % 2 == 1, pos - 1, pos),
        torch.where(out_loc >= 0, out_loc, torch.zeros_like(out_loc)),
    )


__all__ = [
    "ENV_FUSE_INT_SELECTS",
    "ENV_FUSE_PAIR_PARTNERS",
    "ENV_FUSE_ROPE_TAIL",
    "fuse_enabled",
    "fused_pair_partners_and_slots",
    "group_pos_and_slots",
    "pair_partners_fusable",
    "rope_tail_fusable",
    "rope_tail_fused",
]
