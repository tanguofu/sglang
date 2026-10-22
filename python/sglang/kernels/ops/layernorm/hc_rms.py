"""Fused RMS statistics for the DSV4 hyper-connection pre-mix.

The reference path materialises three intermediates per call:

    x_flat = x.flatten(1).float()                                  # cast kernel
    rsqrt  = torch.rsqrt(x_flat.square().mean(-1, keepdim=True)+eps)  # square, mean, add, rsqrt
    mixes  = F.linear(x_flat, hc_fn) * rsqrt

which is five kernels over a K = hc_mult * hidden row for every layer.  One pass
that writes the fp32 operand and the per-row rsqrt replaces them; the arithmetic
is the same fp32 square-and-mean, so the result is unchanged to fp32 rounding.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Debug switch: ``1`` makes rms_cast decline, so the caller keeps its torch path
# and the fused kernel can be ruled in or out of a numerical investigation.
_DISABLED = os.environ.get("SGLANG_DS41_DISABLE_HC_RMS", "0") == "1"


@triton.jit
def _rms_cast_kernel(
    x_ptr,
    xf_ptr,
    rs_ptr,
    D,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One row per program: fp32 copy of the row plus ``rsqrt(mean(x^2) + eps)``."""
    row = tl.program_id(0)
    base = row.to(tl.int64) * D
    sumsq = tl.zeros((), dtype=tl.float32)
    for off in range(0, D, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < D
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        tl.store(xf_ptr + base + idx, v, mask=mask)
        sumsq += tl.sum(v * v, axis=0)
    tl.store(rs_ptr + row, 1.0 / tl.sqrt(sumsq / D + EPS))


def rms_cast(
    x: torch.Tensor, eps: float, *, block: int = 4096
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``x`` flattened to 2-D as fp32 plus its per-row ``rsqrt(mean(x^2)+eps)``.

    Returns ``None`` when the layout is not one the kernel handles, so the caller
    keeps its torch path.
    """
    if _DISABLED:
        return None
    if x.dim() < 2 or not x.is_contiguous():
        return None
    # ``flatten(1)``: every dim after the leading batch dim contributes to K.
    rows = x.shape[0]
    dim = x.numel() // rows
    if rows == 0 or dim == 0:
        return None
    xf = torch.empty((rows, dim), dtype=torch.float32, device=x.device)
    rs = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    _rms_cast_kernel[(rows,)](
        x,
        xf,
        rs,
        dim,
        EPS=float(eps),
        BLOCK=block,
        num_warps=8,
    )
    return xf, rs


__all__ = ["rms_cast"]
