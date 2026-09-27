import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
)
from sglang.srt.disaggregation.utils import DisaggregationMode


class TestMooncakeHostStaging(unittest.TestCase):
    def _make_manager(self):
        manager = object.__new__(MooncakeKVManager)
        manager.disaggregation_mode = DisaggregationMode.DECODE
        manager.engine = Mock()
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[1000, 2000],
            kv_data_lens=[100, 200],
            aux_data_ptrs=[],
            aux_data_lens=[],
            state_data_ptrs=[],
            state_data_lens=[],
            page_size=2,
            kv_item_lens=[10, 20],
            gpu_id=0,
        )
        return manager

    def test_host_staging_replaces_kv_pointers(self):
        manager = self._make_manager()
        host_ptrs = [5000, 6000]

        def malloc_host(byref_ptr, length):
            byref_ptr._obj.value = host_ptrs.pop(0)
            return 0

        hip_lib = Mock()
        hip_lib.hipMallocHost.side_effect = malloc_host

        with patch.dict(os.environ, {"SGLANG_PD_HOST_STAGING": "1"}):
            with patch("ctypes.CDLL", return_value=hip_lib):
                manager.register_buffer_to_engine()

        self.assertEqual(manager.kv_args.kv_data_ptrs, [5000, 6000])
        self.assertEqual(manager._gpu_ptrs, [1000, 2000])
        manager.engine.batch_register.assert_called_once_with(
            [5000, 6000], [100, 200]
        )

    def test_selective_copy_uses_only_transferred_pages(self):
        manager = self._make_manager()
        manager._host_staging_ptrs = [5000, 6000]
        manager._gpu_ptrs = [1000, 2000]
        hip_lib = Mock()
        hip_lib.hipSetDevice.return_value = 0
        hip_lib.hipMemcpy.return_value = 0

        with patch("ctypes.CDLL", return_value=hip_lib):
            manager._copy_host_to_gpu(np.array([1, 3], dtype=np.int32))

        calls = hip_lib.hipMemcpy.call_args_list
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0].args[2].value, 10)
        self.assertEqual(calls[1].args[2].value, 10)
        self.assertEqual(calls[2].args[2].value, 20)
        self.assertEqual(calls[3].args[2].value, 20)

    def test_selective_copy_splits_large_contiguous_ranges(self):
        manager = self._make_manager()
        manager._host_staging_ptrs = [5000]
        manager._gpu_ptrs = [1000]
        manager.kv_args.page_size = 1
        manager.kv_args.kv_item_lens = [600 * 1024 * 1024]
        hip_lib = Mock()
        hip_lib.hipSetDevice.return_value = 0
        hip_lib.hipMemcpy.return_value = 0

        with patch("ctypes.CDLL", return_value=hip_lib):
            manager._copy_host_to_gpu(np.array([0], dtype=np.int32))

        lengths = [call.args[2].value for call in hip_lib.hipMemcpy.call_args_list]
        self.assertEqual(lengths, [256 * 1024 * 1024, 256 * 1024 * 1024, 88 * 1024 * 1024])

    def test_receiver_poll_copies_on_success(self):
        manager = self._make_manager()
        manager._copy_host_to_gpu = Mock()
        receiver = object.__new__(MooncakeKVReceiver)
        receiver.kv_mgr = manager
        receiver.bootstrap_room = 1
        receiver.conclude_state = None
        receiver._dst_kv_indices = np.array([1, 3], dtype=np.int32)
        receiver.kv_mgr.check_status = Mock(return_value=KVPoll.Success)

        with patch.dict(os.environ, {"SGLANG_PD_HOST_STAGING": "1"}):
            result = receiver.poll()

        self.assertEqual(result, KVPoll.Success)
        manager._copy_host_to_gpu.assert_called_once()
        np.testing.assert_array_equal(
            manager._copy_host_to_gpu.call_args.args[0],
            np.array([1, 3], dtype=np.int32),
        )


if __name__ == "__main__":
    unittest.main()
