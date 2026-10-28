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


# ---------------------------------------------------------------------------
# The mHC mixing projection: a bit-identical stand-in for rocBLAS' SGEMM
# ---------------------------------------------------------------------------

# Off by default so it can be rolled out gradually and rolled back in a second.
_BITEXACT_MIX = os.environ.get("SGLANG_DS41_MHC_BITEXACT_GEMM", "0") == "1"

# rocBLAS answers this GEMM with one 64-CTA kernel for every row count from 64
# to 16384, so it costs ~8 ms regardless of M; it only drops to tens of
# microseconds below 64 rows, where it also switches kernels (and therefore
# bits). Stay on rocBLAS there: there is nothing to win and the small-M bits are
# what the baseline already produces.
_BITEXACT_MIX_MIN_ROWS = 64

# Rows per CTA. The K loop is one serial fp32 FMA chain, so M is the only source
# of parallelism: wide tiles once there are enough rows to fill the machine,
# narrow ones so a small decode batch still spreads over the CUs. Measured best
# tile per row count on gfx942 (BK=32, 4 warps, 2 stages).
_BITEXACT_MIX_BLOCK_K = 32
_BITEXACT_MIX_TILES = ((6144, 128), (4096, 64))
_BITEXACT_MIX_BLOCK_M_DEFAULT = 16


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
    offs_n = tl.arange(0, MP)
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
    block_m = _BITEXACT_MIX_BLOCK_M_DEFAULT
    for min_rows, tile in _BITEXACT_MIX_TILES:
        if m >= min_rows:
            block_m = tile
            break
    _row_uniform_mix_kernel[(triton.cdiv(m, block_m),)](
        x,
        weight,
        out,
        m,
        k,
        x.stride(0),
        weight.stride(0),
        MIX=mix,
        MP=max(16, triton.next_power_of_2(mix)),
        BM=block_m,
        BK=_BITEXACT_MIX_BLOCK_K,
        MASK_K=(k % _BITEXACT_MIX_BLOCK_K != 0),
        num_warps=4,
        num_stages=2,
    )
    return out
