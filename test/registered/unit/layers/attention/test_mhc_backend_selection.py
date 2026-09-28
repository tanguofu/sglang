import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.layernorm import mhc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMHCBackendSelection(CustomTestCase):
    def test_gfx942_selects_aiter_when_enabled(self):
        with (
            patch.object(mhc, "is_hip", return_value=True),
            patch.object(mhc, "is_gfx95_supported", return_value=False),
            patch.object(
                mhc, "is_gfx942_supported", return_value=True, create=True
            ),
            patch.dict(os.environ, {"SGLANG_USE_AITER": "1"}),
        ):
            self.assertTrue(mhc._use_aiter_mhc())

    def test_gfx950_selects_aiter_when_enabled(self):
        with (
            patch.object(mhc, "is_hip", return_value=True),
            patch.object(mhc, "is_gfx95_supported", return_value=True),
            patch.object(
                mhc, "is_gfx942_supported", return_value=False, create=True
            ),
            patch.dict(os.environ, {"SGLANG_USE_AITER": "1"}),
        ):
            self.assertTrue(mhc._use_aiter_mhc())

    def test_non_hip_never_selects_aiter(self):
        with (
            patch.object(mhc, "is_hip", return_value=False),
            patch.object(mhc, "is_gfx95_supported", return_value=True),
            patch.object(
                mhc, "is_gfx942_supported", return_value=True, create=True
            ),
            patch.dict(os.environ, {"SGLANG_USE_AITER": "1"}),
        ):
            self.assertFalse(mhc._use_aiter_mhc())

    def test_gfx942_aiter_pre_leaves_rmsnorm_to_caller(self):
        aiter_mhc = MagicMock(return_value=("post", "comb", "layer"))
        aiter_module = MagicMock(mhc_pre=aiter_mhc)

        with (
            patch.dict(sys.modules, {"aiter.ops.mhc": aiter_module}),
            patch.object(mhc, "_AITER_MHC_RUNTIME_DISABLED", False),
            patch.object(mhc, "_AITER_MHC_ACTIVE_LOGGED", False),
            patch.object(mhc, "is_hip", return_value=True),
            patch.object(mhc, "is_gfx95_supported", return_value=False),
            patch.object(mhc, "is_gfx942_supported", return_value=True),
            patch.dict(os.environ, {"SGLANG_USE_AITER": "1"}),
        ):
            result = mhc._mhc_pre_dispatch(
                residual=torch.zeros(2, 3, 4),
                fn=torch.zeros(15, 12),
                hc_scale=torch.ones(3),
                hc_base=torch.zeros(15),
                rms_eps=1e-5,
                hc_pre_eps=1e-6,
                hc_sinkhorn_eps=1e-6,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=1,
                norm_weight=torch.ones(4),
                norm_eps=1e-5,
            )

        self.assertEqual(("post", "comb", "layer", False), result)
        self.assertNotIn("norm_weight", aiter_mhc.call_args.kwargs)
        self.assertNotIn("norm_eps", aiter_mhc.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
