"""Aiter RMSNorm vs a torch reference."""

from __future__ import annotations

import pytest
import torch


def _ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


@pytest.mark.parametrize("rows,hidden", [(8, 256), (32, 1024), (64, 4096)])
def test_aiter_rms_norm(rows: int, hidden: int) -> None:
    try:
        from aiter import rms_norm
    except Exception as exc:
        pytest.skip(f"aiter.rms_norm unavailable: {exc}")

    torch.manual_seed(4)
    eps = 1e-6
    x = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
    out = rms_norm(x, w, eps)
    ref = _ref_rms_norm(x, w, eps)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
