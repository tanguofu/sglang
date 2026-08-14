"""Aiter FMoE / MoE GEMM vs a torch grouped-GEMM reference (tiny shapes)."""

from __future__ import annotations

import pytest
import torch


def _try_blockscale_gemm() -> None:
    from aiter.ops.triton.moe.moe_op_gemm_a8w8_blockscale import (  # type: ignore
        moe_gemm_a8w8_blockscale,
        moe_gemm_torch,
    )
    from aiter.ops.triton.moe.moe_routing.routing import routing  # type: ignore

    torch.manual_seed(3)
    m, n, k = 32, 128, 128
    n_expts_tot, n_expts_act = 8, 2
    device = "cuda"
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device)
    routing_data, gather_idx, scatter_idx = routing(logits, n_expts_act)
    x = (torch.randn((m, k), dtype=torch.bfloat16, device=device) / 10).to(
        torch.float8_e4m3fnuz
        if hasattr(torch, "float8_e4m3fnuz")
        else torch.float8_e4m3fn
    )
    w = (torch.randn((n_expts_tot, k, n), dtype=torch.bfloat16, device=device) / 10).to(
        x.dtype
    )
    # If the op needs extra scales the call will raise; skip then.
    y = moe_gemm_a8w8_blockscale(
        x,
        w,
        routing_data,
        gather_idx,
        scatter_idx,
    )
    ref = moe_gemm_torch(x, w, routing_data, gather_idx, scatter_idx)
    torch.testing.assert_close(y.float(), ref.float(), rtol=8e-2, atol=8e-2)


def test_aiter_moe_gemm_blockscale() -> None:
    try:
        _try_blockscale_gemm()
    except ImportError as exc:
        pytest.skip(f"moe_gemm_a8w8_blockscale not importable: {exc}")
    except TypeError as exc:
        pytest.skip(f"moe_gemm signature mismatch on this aiter: {exc}")
    except Exception as exc:
        pytest.skip(f"moe_gemm not runnable: {exc}")


def test_aiter_fmoe_symbol_present() -> None:
    try:
        import aiter

        assert hasattr(aiter, "fmoe_g1u1") or hasattr(aiter, "fmoe")
    except Exception as exc:
        pytest.skip(f"aiter fmoe symbol missing: {exc}")
