import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class TestDSATargetVerifyMetadata(unittest.TestCase):
    def test_target_verify_uses_gpu_side_uniform_draft_lens(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.speculative_num_draft_tokens = 4
        backend.dsa_index_kpool = 1
        backend.real_page_size = 1
        backend.use_mha = False
        backend.dsa_decode_impl = "tilelang"
        backend.dsa_prefill_impl = "tilelang"
        backend.set_dsa_prefill_impl = Mock()
        backend.get_topk_transform_method = Mock(return_value=None)

        seq_lens = torch.tensor([10, 20], dtype=torch.int32)
        forward_batch = SimpleNamespace(
            batch_size=2,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.clone(),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            forward_mode=ForwardMode.TARGET_VERIFY,
        )
        backend.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(40, dtype=torch.int64).reshape(2, 20)
        )

        captured = {}

        def capture_seqlens_expand(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            raise RuntimeError("stop after target-verify metadata")

        with patch.object(
            dsa_backend,
            "seqlens_expand_triton",
            side_effect=capture_seqlens_expand,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after target-verify"):
                backend.init_forward_metadata(forward_batch)

        draft_lens = captured["args"][0]
        self.assertEqual(draft_lens.device.type, seq_lens.device.type)
        self.assertEqual(draft_lens.dtype, torch.int32)
        self.assertEqual(draft_lens.tolist(), [4, 4])


if __name__ == "__main__":
    unittest.main()
