"""DSA fused_store_index_k_cache vs per-token fp8 quant reference (HIP enabled)."""

from __future__ import annotations

from typing import Any

import pytest
import torch

PAGE_SIZE = 64
HEAD_DIM = 128
BYTES_PER_TOKEN = HEAD_DIM + 4


def _load_fused() -> tuple[Any, Any]:
    try:
        from sglang.jit_kernel.fused_store_index_cache import (
            can_use_dsa_fused_store,
            fused_store_index_k_cache,
        )

        return can_use_dsa_fused_store, fused_store_index_k_cache
    except ImportError:
        from sglang.kernels.ops.attention.fused_store_index_cache import (
            can_use_dsa_fused_store,
            fused_store_index_k_cache,
        )

        return can_use_dsa_fused_store, fused_store_index_k_cache


def _fp8_dtype() -> torch.dtype:
    if hasattr(torch, "float8_e4m3fnuz"):
        try:
            from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

            if is_fp8_fnuz():
                return torch.float8_e4m3fnuz
        except Exception:
            return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def _read_token(buf: torch.Tensor, token_idx: int) -> tuple[torch.Tensor, float]:
    page = token_idx // PAGE_SIZE
    offset = token_idx % PAGE_SIZE
    page_bytes = PAGE_SIZE * BYTES_PER_TOKEN
    flat = buf.reshape(-1)
    val = flat[page * page_bytes + offset * HEAD_DIM :][:HEAD_DIM]
    scale = (
        flat[page * page_bytes + HEAD_DIM * PAGE_SIZE + offset * 4 :][:4]
        .view(torch.float32)
        .item()
    )
    return val.view(_fp8_dtype()).float(), scale


@pytest.mark.parametrize("num_tokens", [8, 64, 128])
def test_fused_store_roundtrip(num_tokens: int) -> None:
    try:
        can_use, fused_store = _load_fused()
    except ImportError:
        pytest.skip("fused_store_index_cache not importable")

    if not can_use(torch.bfloat16, torch.int64, PAGE_SIZE):
        pytest.skip("fused_store JIT unavailable on this image")

    torch.manual_seed(1)
    key = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    loc = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    n_pages = int(loc.max().item()) // PAGE_SIZE + 2
    buf = torch.zeros(
        (n_pages, PAGE_SIZE * BYTES_PER_TOKEN), dtype=torch.uint8, device="cuda"
    )
    fused_store(key, buf, loc, page_size=PAGE_SIZE)
    torch.cuda.synchronize()

    fp8_max = float(torch.finfo(_fp8_dtype()).max)
    amax = key.abs().float().amax(dim=-1).clamp(min=1e-4)
    ref_scale = amax / fp8_max
    ref_deq = (key.float() / ref_scale.unsqueeze(-1)).to(_fp8_dtype()).float() * (
        ref_scale.unsqueeze(-1)
    )

    got = []
    for i in range(num_tokens):
        fp8_vals, scale = _read_token(buf, int(loc[i].item()))
        got.append(fp8_vals * scale)
    got_t = torch.stack(got, dim=0)
    # FP8 E4M3 1-ULP; HIP pack vs torch .to() can differ at ties.
    torch.testing.assert_close(got_t, ref_deq, rtol=0.15, atol=2.0)
