import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDFlashDisaggregation(unittest.TestCase):
    def test_dflash_builds_decode_draft_input(self):
        batch = SimpleNamespace(
            seq_lens=torch.tensor([4, 7], dtype=torch.int64),
            req_pool_indices=torch.tensor([11, 21], dtype=torch.int64),
            enable_overlap=False,
        )
        last_tokens = torch.tensor([101, 102], dtype=torch.int64)

        spec_info = SpeculativeAlgorithm.DFLASH.build_disagg_draft_input(
            batch, last_tokens, Mock()
        )

        self.assertIsInstance(spec_info, DFlashDraftInputV2)
        self.assertTrue(torch.equal(spec_info.bonus_tokens, last_tokens))
        self.assertTrue(torch.equal(spec_info.new_seq_lens, batch.seq_lens))


if __name__ == "__main__":
    unittest.main()
