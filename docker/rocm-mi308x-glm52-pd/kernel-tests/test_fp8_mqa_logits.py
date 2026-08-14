"""FlyDSL + Triton fp8 MQA logits vs a torch reference (DSA indexer)."""

from __future__ import annotations

from typing import Callable

import pytest
import torch

e4m3 = getattr(torch, "float8_e4m3fnuz", None) or torch.float8_e4m3fn
fp8_max = float(torch.finfo(e4m3).max)


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x64, y64 = x.double(), y.double()
    denom = (x64 * x64 + y64 * y64).sum()
    if float(denom) == 0.0:
        return 0.0
    return float(1 - 2 * (x64 * y64).sum() / denom)


def _cast_kv_fp8(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = kv.abs().float().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = amax / fp8_max
    return (kv.float() / scale).to(e4m3), scale.squeeze(-1)


def _ref_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> torch.Tensor:
    seq_kv = kv.shape[0]
    idx = torch.arange(seq_kv, device=q.device)
    mask = (idx[None, :] >= ks[:, None]) & (idx[None, :] < ke[:, None])
    score = torch.einsum("mhd,nd->hmn", q.float(), kv.float())
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    return logits.masked_fill(~mask, float("-inf"))


def _import_impls() -> dict[str, Callable]:
    impls: dict[str, Callable] = {}
    try:
        from aiter.ops.flydsl import flydsl_fp8_mqa_logits

        impls["flydsl"] = flydsl_fp8_mqa_logits
    except Exception:
        try:
            from aiter.ops.flydsl.kernels.fp8_mqa_logits import flydsl_fp8_mqa_logits

            impls["flydsl"] = flydsl_fp8_mqa_logits
        except Exception:
            pass
    try:
        from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits

        impls["triton"] = fp8_mqa_logits
    except Exception:
        try:
            from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

            impls["triton"] = fp8_mqa_logits
        except Exception:
            pass
    return impls


@pytest.mark.parametrize("s_q,s_k", [(1, 16), (17, 76), (32, 256)])
@pytest.mark.parametrize("num_heads", [16])
def test_fp8_mqa_logits_matches_ref(s_q: int, s_k: int, num_heads: int) -> None:
    impls = _import_impls()
    if not impls:
        pytest.skip("fp8_mqa_logits (flydsl/triton) not importable")

    torch.manual_seed(0)
    head_dim = 128
    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(s_k, head_dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(s_q, num_heads, dtype=torch.float32, device="cuda")
    ks = torch.zeros(s_q, dtype=torch.int32, device="cuda")
    ke = torch.arange(s_q, dtype=torch.int32, device="cuda") + (s_k - s_q)

    kv_fp8, scales = _cast_kv_fp8(kv)
    kv_deq = (kv_fp8.float() * scales.unsqueeze(-1)).to(torch.bfloat16)
    q_fp8 = q.to(e4m3)
    ref = _ref_logits(q, kv_deq, weights, ks, ke)
    ref_mask = ref == float("-inf")

    for name, fn in impls.items():
        out = fn(q_fp8, kv_fp8, scales, weights, ks, ke, True)
        out_mask = out == float("-inf")
        assert torch.equal(out_mask, ref_mask), f"{name}: -inf mask mismatch"
        if not bool(ref_mask.all()):
            diff = _calc_diff(
                out.masked_fill(out_mask, 0), ref.masked_fill(ref_mask, 0)
            )
            assert diff < 2e-3, f"{name} calc_diff={diff}"
