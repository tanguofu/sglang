"""Aiter RoPE vs a rotary-embedding torch reference (small head dim)."""

from __future__ import annotations

import pytest
import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _ref_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [T, H, D], cos/sin: [T, D]
    return x * cos.unsqueeze(1) + _rotate_half(x) * sin.unsqueeze(1)


def test_aiter_rope_or_skip() -> None:
    impl = None
    try:
        from aiter.ops.triton.rope import rope as impl  # type: ignore
    except Exception:
        try:
            from aiter import rope as impl  # type: ignore
        except Exception:
            impl = None
    if impl is None:
        pytest.skip("aiter rope not importable on this image")

    torch.manual_seed(5)
    t, h, d = 16, 8, 64
    x = torch.randn(t, h, d, dtype=torch.bfloat16, device="cuda")
    pos = torch.arange(t, device="cuda", dtype=torch.float32)
    half = d // 2
    freq = 1.0 / (10000 ** (torch.arange(0, half, device="cuda", dtype=torch.float32) / half))
    ang = pos[:, None] * freq[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1).to(x.dtype)
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1).to(x.dtype)
    try:
        out = impl(x, cos, sin)
    except TypeError as exc:
        pytest.skip(f"rope signature mismatch: {exc}")
    ref = _ref_rope(x.float(), cos.float(), sin.float())
    torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-2)
