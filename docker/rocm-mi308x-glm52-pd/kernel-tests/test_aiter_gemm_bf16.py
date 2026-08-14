"""Aiter tuned GEMM (bf16) vs torch.nn.functional.linear."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


def _tgemm_mm(
    x: torch.Tensor, weight: torch.Tensor, otype: torch.dtype
) -> torch.Tensor:
    from aiter.tuned_gemm import tgemm

    return tgemm.mm(x, weight, None, otype, None, None)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (16, 32, 256),
        (64, 256, 512),
        (128, 32, 1024),
    ],
)
def test_aiter_bf16_gemm(m: int, n: int, k: int) -> None:
    try:
        _tgemm_mm(
            torch.zeros(1, 8, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(8, 8, dtype=torch.bfloat16, device="cuda"),
            torch.bfloat16,
        )
    except Exception as exc:
        pytest.skip(f"aiter.tuned_gemm unavailable: {exc}")

    torch.manual_seed(2)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    out = _tgemm_mm(x, w, torch.bfloat16)
    ref = F.linear(x, w)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
