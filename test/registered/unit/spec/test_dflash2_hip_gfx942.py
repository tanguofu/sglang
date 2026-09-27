import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.arg_groups.speculative_hook import _handle_dflash
from sglang.srt.models import dflash
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _enable_fake_aiter(monkeypatch, tp_size=1):
    monkeypatch.setattr(dflash, "is_hip", lambda: True, raising=False)
    monkeypatch.setattr(
        dflash,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=tp_size),
        raising=False,
    )
    monkeypatch.setattr(
        dflash,
        "get_bool_env_var",
        lambda name, default="false": name == "SGLANG_USE_AITER",
        raising=False,
    )


def test_dflash_hip_gate_requires_explicit_env(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_DFLASH_HIP", raising=False)
    monkeypatch.setattr(
        "sglang.srt.utils.is_hip", lambda: True, raising=False
    )
    server_args = SimpleNamespace(device="rocm")

    with pytest.raises(ValueError, match="SGLANG_ENABLE_DFLASH_HIP"):
        _handle_dflash(server_args)


def test_dflash_hip_gate_allows_experimental_path(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_DFLASH_HIP", "1")
    monkeypatch.setattr(
        "sglang.srt.utils.is_hip", lambda: True, raising=False
    )
    monkeypatch.setattr(
        "sglang.srt.arg_groups.overrides.resolved_view",
        lambda args: SimpleNamespace(enable_dp_attention=False),
        raising=False,
    )
    monkeypatch.setattr(
        "sglang.srt.arg_groups.speculative_hook._resolve_dflash_draft_attention_backend",
        lambda args: None,
        raising=False,
    )
    server_args = SimpleNamespace(
        device="rocm",
        pp_size=1,
        speculative_draft_model_path="dummy",
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_dflash_block_size=8,
        speculative_num_draft_tokens=8,
        speculative_draft_window_size=8,
        max_running_requests=1,
        enable_mixed_chunk=False,
    )

    _handle_dflash(server_args)

    assert server_args.speculative_num_draft_tokens == 8


def test_dflash_hip_gate_handles_unresolved_device(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_DFLASH_HIP", "1")
    monkeypatch.setattr(
        "sglang.srt.utils.is_hip", lambda: True, raising=False
    )
    monkeypatch.setattr(
        "sglang.srt.arg_groups.overrides.resolved_view",
        lambda args: SimpleNamespace(enable_dp_attention=False),
        raising=False,
    )
    monkeypatch.setattr(
        "sglang.srt.arg_groups.speculative_hook._resolve_dflash_draft_attention_backend",
        lambda args: None,
        raising=False,
    )
    server_args = SimpleNamespace(
        device=None,
        pp_size=1,
        speculative_draft_model_path="dummy",
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_dflash_block_size=8,
        speculative_num_draft_tokens=8,
        speculative_draft_window_size=8,
        max_running_requests=1,
        enable_mixed_chunk=False,
    )

    _handle_dflash(server_args)

    assert server_args.speculative_num_draft_tokens == 8


def test_aiter_topk_contract_and_determinism(monkeypatch):
    _enable_fake_aiter(monkeypatch, tp_size=8)
    torch.manual_seed(0)
    scores = torch.randn(32, 128, dtype=torch.float32)
    k = 8
    expected_values, expected_indices = torch.topk(scores, k, dim=-1)

    def fake_topk(
        logits,
        next_n,
        seq_lens,
        indices,
        num_rows,
        stride0,
        stride1,
        k,
        stable,
    ):
        assert logits.dtype == torch.float32
        assert logits.is_contiguous()
        assert next_n == 1
        assert num_rows == scores.shape[0]
        assert stride0 == scores.stride(0)
        assert stride1 == scores.stride(1)
        assert stable is True
        assert indices.dtype == torch.int32
        assert indices.shape == (scores.shape[0], k)
        assert torch.all(seq_lens == scores.shape[-1])
        indices.copy_(expected_indices.to(torch.int32))

    monkeypatch.setattr(dflash, "_aiter_gemm_a16w16", lambda *args, **kwargs: None)
    monkeypatch.setattr(dflash, "_aiter_top_k_per_row_decode", fake_topk)

    actual_values, actual_indices = dflash._radix_topk(scores, k)

    torch.testing.assert_close(actual_values, expected_values)
    torch.testing.assert_close(actual_indices, expected_indices)


def test_aiter_gemm_contract_and_accuracy(monkeypatch):
    _enable_fake_aiter(monkeypatch)
    torch.manual_seed(0)
    hidden = torch.randn(3, 4, dtype=torch.bfloat16)
    weight = torch.randn(16, 4, dtype=torch.bfloat16)
    k = 4

    def fake_gemm(a, b, otype=None):
        assert a is hidden
        assert torch.equal(b, weight)
        assert otype == torch.float32
        return torch.matmul(a, b.T).float()

    monkeypatch.setattr(dflash, "_aiter_gemm_a16w16", fake_gemm)
    monkeypatch.setattr(
        dflash,
        "_radix_topk",
        lambda scores, k: torch.topk(scores, k, dim=-1),
    )
    monkeypatch.setattr(
        dflash,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=1),
    )
    model = SimpleNamespace(
        lm_head=SimpleNamespace(weight=weight, org_vocab_size=weight.shape[0]),
        candidate_selector=SimpleNamespace(top_k=k),
        _transform_unary_logits=lambda logits: logits.float(),
    )

    candidate_ids, unary_logits = dflash.DFlash2DraftModel.compute_candidates(
        model, hidden
    )

    expected_logits, expected_ids = torch.topk(
        torch.matmul(hidden, weight.T), k, dim=-1
    )
    torch.testing.assert_close(candidate_ids, expected_ids)
    torch.testing.assert_close(unary_logits, expected_logits.float())


def test_aiter_gemm_is_disabled_but_topk_stays_enabled_for_tp_greater_than_one(
    monkeypatch,
):
    _enable_fake_aiter(monkeypatch, tp_size=8)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        dflash,
        "_aiter_top_k_per_row_decode",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        dflash,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=8),
    )

    assert dflash._use_aiter_gemm() is False
    assert dflash._use_aiter_topk(31) is False
    assert dflash._use_aiter_topk(32) is True


def test_aiter_topk_has_no_row_threshold_for_tp_one(monkeypatch):
    _enable_fake_aiter(monkeypatch, tp_size=1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert dflash._use_aiter_topk(1) is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
