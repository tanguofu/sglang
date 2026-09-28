import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsa_backend import _reshape_split_mla_q
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestReshapeSplitMlaQ(unittest.TestCase):
    def test_zero_rope_dimension(self):
        layer = SimpleNamespace(tp_q_head_num=8, v_head_dim=128, head_dim=128)
        q = torch.randn(4, 8, 128)
        q_rope = torch.empty(4, 8, 0)

        q_nope, reshaped_q_rope = _reshape_split_mla_q(q, q_rope, layer)

        self.assertEqual(tuple(q_nope.shape), (4, 8, 128))
        self.assertEqual(tuple(reshaped_q_rope.shape), (4, 8, 0))


if __name__ == "__main__":
    unittest.main()
