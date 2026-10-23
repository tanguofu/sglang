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

# Chosen by the environment so an A/B run does not need a code change.
_OVERRIDE = os.environ.get("SGLANG_DS41_BLAS_LIB")


def _target() -> str:
    """The BLAS library used for row-uniform calls; rocBLAS on ROCm."""
    return _OVERRIDE or "cublas"


@contextmanager
def row_uniform_blas():
    """Run one reduction with an accumulation order that cannot follow the row.

    `cublas` is the ROCm build's name for rocBLAS (the selector predates the
    fork); it keeps one reduction order per output element for these shapes.
    """
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
    with row_uniform_blas():
        return torch.einsum(equation, *operands)


def row_uniform_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """`F.linear` under :func:`row_uniform_blas`."""
    with row_uniform_blas():
        return torch.nn.functional.linear(x, weight)
