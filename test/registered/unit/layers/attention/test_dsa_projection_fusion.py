"""Pin gfx942 DSA indexer projection-fusion eligibility (#39243)."""

from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

from sglang.srt.layers.attention.dsa.dsa_projection_fusion import (
    dsa_indexer_projection_fusion_enabled,
)
from sglang.test.test_utils import CustomTestCase


def _quant(name="fp8", block=(128, 128), mxfp8=False):
    return SimpleNamespace(
        get_name=lambda: name,
        weight_block_size=list(block) if block else None,
        use_mxfp8=mxfp8,
    )


class TestDsaProjectionFusion(CustomTestCase):
    def _hip(self, **kwargs):
        defaults = dict(
            cuda_full_fusion=False,
            is_hip=True,
            gfx942=True,
            enable_rocm_proj=True,
            disable_fusion=False,
            is_neox_style=False,
            enable_lora=False,
            quant_config=None,
        )
        defaults.update(kwargs)
        return dsa_indexer_projection_fusion_enabled(**defaults)

    def test_glm53_fp8_opt_in(self):
        self.assertTrue(self._hip(quant_config=_quant()))

    def test_opt_in_required(self):
        self.assertFalse(self._hip(enable_rocm_proj=False))

    def test_global_disable_wins(self):
        self.assertFalse(self._hip(disable_fusion=True))

    def test_neox_and_lora_rejected(self):
        self.assertFalse(self._hip(is_neox_style=True))
        self.assertFalse(self._hip(enable_lora=True))

    def test_not_gfx942_or_not_hip(self):
        self.assertFalse(self._hip(gfx942=False))
        self.assertFalse(self._hip(is_hip=False))

    def test_mxfp8_and_wrong_block_rejected(self):
        self.assertFalse(self._hip(quant_config=_quant(mxfp8=True)))
        self.assertFalse(self._hip(quant_config=_quant(block=(64, 64))))
        self.assertFalse(self._hip(quant_config=_quant(name="awq")))

    def test_cuda_full_fusion_still_on(self):
        self.assertTrue(
            dsa_indexer_projection_fusion_enabled(
                cuda_full_fusion=True,
                is_hip=False,
                gfx942=False,
                enable_rocm_proj=False,
                disable_fusion=False,
                is_neox_style=False,
                enable_lora=False,
                quant_config=_quant(),
            )
        )
