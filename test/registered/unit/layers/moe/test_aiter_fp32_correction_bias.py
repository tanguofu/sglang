"""Pin the aiter boundary so an fp32 MoE correction bias is not recast to bf16."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import torch

from sglang.srt.layers.moe.topk import _aiter_fp32_correction_bias_args
from sglang.test.test_utils import CustomTestCase


class TestAiterFp32CorrectionBias(CustomTestCase):
    def test_fp32_bias_upcasts_logits(self):
        logits = torch.ones(2, 8, dtype=torch.bfloat16)
        bias = torch.linspace(6.8, 7.06, 8, dtype=torch.float32)
        gating, kept = _aiter_fp32_correction_bias_args(logits, bias)
        self.assertEqual(gating.dtype, torch.float32)
        self.assertEqual(kept.dtype, torch.float32)
        self.assertIs(kept, bias)

    def test_bf16_bias_stays_byte_identical(self):
        logits = torch.ones(2, 8, dtype=torch.bfloat16)
        bias = torch.ones(8, dtype=torch.bfloat16)
        gating, kept = _aiter_fp32_correction_bias_args(logits, bias)
        self.assertIs(gating, logits)
        self.assertIs(kept, bias)
