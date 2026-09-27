import importlib.util
import sys
from pathlib import Path

import torch


HELPER_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/kernels/ops/speculative/dflash_sparse_accept.py"
)


def load_helper():
    spec = importlib.util.spec_from_file_location("dflash_sparse_accept", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_draft_q_values_matches_sampled_path():
    helper = load_helper()
    candidates = torch.tensor([[7, 10, 20, 30]], dtype=torch.int64)
    candidate_ids = torch.tensor(
        [[[10, 11, 12], [20, 21, 22], [30, 31, 32]]], dtype=torch.int64
    )
    q_rows = torch.tensor(
        [[[0.2, 0.5, 0.3], [0.7, 0.2, 0.1], [0.4, 0.4, 0.2]]],
        dtype=torch.float32,
    )

    selected_q = helper.select_draft_q_values(
        candidates=candidates,
        candidate_ids=candidate_ids,
        q_rows=q_rows,
    )

    torch.testing.assert_close(selected_q, torch.tensor([[0.2, 0.7, 0.4]]))


def test_sparse_residual_matches_dense_scatter_reference():
    helper = load_helper()
    vocab_size = 12
    target_probs = torch.tensor(
        [[0.05, 0.2, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.025, 0.025]],
        dtype=torch.float32,
    )
    candidate_ids = torch.tensor([[1, 4, 8]], dtype=torch.int64)
    q_rows = torch.tensor([[0.2, 0.1, 0.05]], dtype=torch.float32)

    dense_q = torch.zeros_like(target_probs)
    dense_q.scatter_(1, candidate_ids, q_rows)
    expected = torch.relu(target_probs - dense_q)
    actual = helper.sparse_residual_rows(
        target_probs=target_probs,
        candidate_ids=candidate_ids,
        q_rows=q_rows,
    )

    torch.testing.assert_close(actual, expected)


def test_sparse_residual_treats_nan_q_as_zero():
    helper = load_helper()
    target_probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
    candidate_ids = torch.tensor([[1]], dtype=torch.int64)
    q_rows = torch.tensor([[float("nan")]], dtype=torch.float32)

    actual = helper.sparse_residual_rows(
        target_probs=target_probs,
        candidate_ids=candidate_ids,
        q_rows=q_rows,
    )

    torch.testing.assert_close(actual, target_probs)
