"""Pin FlyDSL MQA i32-safe row limits for long-context prefill chunks."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.test.test_utils import CustomTestCase


class TestDsaMqaI32Chunk(CustomTestCase):
    def test_16k_chunk_crosses_at_96k_not_32k(self):
        # 16k * 32k * 4 = 2.00 GiB-ish element-bytes; still under i32.
        self.assertGreaterEqual(Indexer._mqa_i32_safe_max_rows(32000), 16384)
        # 16k * 64k * 4 wraps i32; 96k is the live PD cliff.
        self.assertLess(Indexer._mqa_i32_safe_max_rows(64000), 16384)
        self.assertLess(Indexer._mqa_i32_safe_max_rows(96000), 16384)
        self.assertLess(Indexer._mqa_i32_safe_max_rows(196000), 16384)

    def test_decode_row_stays_unchunked(self):
        # Decode / last-token extend is 1-4 rows; never force-chunk on k alone.
        self.assertGreater(Indexer._mqa_i32_safe_max_rows(196000), 8)

    def test_empty_k(self):
        self.assertEqual(Indexer._mqa_i32_safe_max_rows(0), 1)
        self.assertEqual(Indexer._mqa_i32_safe_max_rows(-1), 1)
