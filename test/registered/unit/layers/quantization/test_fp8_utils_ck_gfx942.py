import sys
import unittest
from contextlib import contextmanager
from unittest import mock

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8_utils import _ck_csv_has_row
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@contextmanager
def patch_ck_lookup(get_config):
    jit_core = mock.MagicMock()
    gemm_op = mock.MagicMock()
    gemm_op.get_CKGEMM_config = get_config
    modules = {
        "aiter": mock.MagicMock(),
        "aiter.jit": mock.MagicMock(),
        "aiter.jit.core": jit_core,
        "aiter.ops": mock.MagicMock(),
        "aiter.ops.gemm_op_a8w8": gemm_op,
    }
    with mock.patch.dict(sys.modules, modules):
        yield jit_core.AITER_CONFIGS, gemm_op.get_CKGEMM_config


class TestCkCsvHasRow(unittest.TestCase):
    def test_helper_returns_false_when_disabled(self):
        # Gate off -> constant False without ever importing aiter.
        with mock.patch.object(fp8_utils, "_use_aiter_ck_gfx942", False):
            _ck_csv_has_row.cache_clear()
            self.assertFalse(_ck_csv_has_row(4, 4096, 2048))
            self.assertFalse(_ck_csv_has_row(16384, 2624, 6144))

    def test_helper_swallows_exception(self):
        # A broken aiter/CSV must degrade to triton, never raise.
        def boom(*args, **kwargs):
            raise RuntimeError("csv unreadable")

        with mock.patch.object(fp8_utils, "_use_aiter_ck_gfx942", True):
            with patch_ck_lookup(boom):
                _ck_csv_has_row.cache_clear()
                self.assertFalse(_ck_csv_has_row(4, 4096, 2048))

    def test_helper_true_on_csv_hit(self):
        fake_config = {"kernelId": 8, "splitK": 3, "us": 8.15}
        get_config = mock.MagicMock(return_value=fake_config)
        with mock.patch.object(fp8_utils, "_use_aiter_ck_gfx942", True):
            with patch_ck_lookup(get_config):
                _ck_csv_has_row.cache_clear()
                self.assertTrue(_ck_csv_has_row(4, 4096, 2048))

    def test_helper_false_on_csv_miss(self):
        get_config = mock.MagicMock(return_value=None)
        with mock.patch.object(fp8_utils, "_use_aiter_ck_gfx942", True):
            with patch_ck_lookup(get_config):
                _ck_csv_has_row.cache_clear()
                # indexer wk (N=128) has no CSV rows -> stays triton.
                self.assertFalse(_ck_csv_has_row(4, 128, 6144))

    def test_env_gate_off_by_default(self):
        # Without SGLANG_ROCM_USE_CK_GEMM_GFX942 the flag must be False so the
        # rollback path (env=0 + pod bounce) restores the triton behavior.
        import os

        saved = os.environ.pop("SGLANG_ROCM_USE_CK_GEMM_GFX942", None)
        try:
            gate = (
                fp8_utils._use_aiter
                and fp8_utils.is_gfx942_supported()
                and not fp8_utils._is_gfx95_supported
                and fp8_utils.get_bool_env_var("SGLANG_ROCM_USE_CK_GEMM_GFX942")
            )
            self.assertFalse(gate)
        finally:
            if saved is not None:
                os.environ["SGLANG_ROCM_USE_CK_GEMM_GFX942"] = saved

    def test_gfx95_branch_unchanged(self):
        # On gfx95 the decision must be byte-identical to the pre-gfx942 logic:
        # the new elif only fires when _use_aiter_ck_gfx942 is True, which
        # requires is_gfx942_supported() and not _is_gfx95_supported.
        for gfx95 in (True, False):
            for ck_gfx942 in (True, False):
                if gfx95 and ck_gfx942:
                    continue  # impossible combination by construction
                with mock.patch.object(fp8_utils, "_use_aiter_gfx95", gfx95):
                    with mock.patch.object(
                        fp8_utils, "_use_aiter_ck_gfx942", ck_gfx942
                    ):
                        with mock.patch.object(
                            fp8_utils, "_use_aiter_bpreshuffle_gfx95", False
                        ):
                            # gfx95 branch: use_triton follows the gfx95 table
                            # regardless of the gfx942 flag.
                            if gfx95:
                                self.assertTrue(
                                    fp8_utils.use_aiter_triton_gemm_w8a8_tuned_gfx950(
                                        7168, 2048
                                    )
                                )


if __name__ == "__main__":
    unittest.main()
