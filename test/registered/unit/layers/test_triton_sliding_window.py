import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.triton_backend import update_sliding_window_buffer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestTritonSlidingWindow(unittest.TestCase):
    def test_cpu_mirror_avoids_gpu_scalar_readback(self):
        window_kv_indptr = torch.zeros(3, dtype=torch.int32)
        req_to_token = torch.zeros((2, 8), dtype=torch.int64)
        seq_lens = torch.tensor([5, 10], dtype=torch.int64)
        seq_lens_cpu = torch.tensor([5, 10], dtype=torch.int64)
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
        token_to_kv_pool = Mock()
        token_to_kv_pool.translate_loc_from_full_to_swa = Mock(side_effect=lambda x: x)

        with (
            patch(
                "torch.Tensor.item",
                side_effect=AssertionError("GPU scalar readback is forbidden"),
            ),
            patch("torch.empty", wraps=torch.empty) as torch_empty,
            patch(
                "sglang.srt.layers.attention.triton_backend.create_flashinfer_kv_indices_triton"
            ),
        ):
            result = update_sliding_window_buffer(
                window_kv_indptr,
                req_to_token,
                7,
                seq_lens,
                req_pool_indices,
                2,
                torch.device("cpu"),
                token_to_kv_pool,
                seq_lens_cpu=seq_lens_cpu,
            )

        self.assertEqual(torch_empty.call_args.args[0], 12)
        self.assertEqual(result[1].shape[0], 12)
        token_to_kv_pool.translate_loc_from_full_to_swa.assert_called_once()
        self.assertEqual(
            token_to_kv_pool.translate_loc_from_full_to_swa.call_args.args[0].shape[0],
            12,
        )


if __name__ == "__main__":
    unittest.main()
