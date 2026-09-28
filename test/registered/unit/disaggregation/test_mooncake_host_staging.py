import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    MooncakeKVReceiver,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakeHipLibrary:
    def __init__(self):
        self.malloc_sizes = []
        self.freed_ptrs = []
        self.copy_calls = []
        self.devices = []
        self.next_ptr = 0x10000

    def hipMallocHost(self, out, size):
        self.malloc_sizes.append(size.value)
        self.next_ptr += size.value
        out._obj.value = self.next_ptr
        return 0

    def hipHostFree(self, ptr):
        self.freed_ptrs.append(ptr.value)
        return 0

    def hipSetDevice(self, device):
        self.devices.append(device.value)
        return 0

    def hipMemcpy(self, dst, src, size, kind):
        self.copy_calls.append((dst.value, src.value, size.value, kind.value))
        return 0


def make_manager():
    manager = object.__new__(MooncakeKVManager)
    manager.kv_args = SimpleNamespace(
        kv_data_ptrs=[0x1000, 0x2000],
        kv_data_lens=[100, 200],
        kv_item_lens=[10, 20],
        state_data_ptrs=[[0x3000, 0x4000]],
        state_data_lens=[[30, 40]],
        state_item_lens=[[3, 4]],
        gpu_id=7,
    )
    return manager


class TestMooncakeHostStaging(unittest.TestCase):
    def test_allocate_host_staging_buffers_mirrors_kv_and_state(self):
        manager = make_manager()
        hip = FakeHipLibrary()

        with patch.object(ctypes, "CDLL", return_value=hip):
            manager._allocate_host_staging_buffers()

        self.assertEqual([100, 200, 30, 40], hip.malloc_sizes)
        self.assertEqual([0x1000, 0x2000], manager._gpu_ptrs)
        self.assertEqual([[0x3000, 0x4000]], manager._gpu_state_ptrs)
        self.assertEqual(manager._host_staging_ptrs, manager.kv_args.kv_data_ptrs)
        self.assertEqual(
            manager._host_state_staging_ptrs, manager.kv_args.state_data_ptrs
        )
        self.assertEqual([100, 200], manager._host_staging_lens)
        self.assertEqual([[30, 40]], manager._host_state_staging_lens)

    def test_copy_state_host_to_gpu_copies_only_requested_slots(self):
        manager = make_manager()
        manager._host_state_staging_ptrs = [[0x5000, 0x6000]]
        manager._gpu_state_ptrs = [[0x7000, 0x8000]]
        hip = FakeHipLibrary()

        with patch.object(ctypes, "CDLL", return_value=hip):
            manager._copy_state_host_to_gpu([[2, 3, 5]])

        self.assertEqual([7], hip.devices)
        self.assertEqual(
            [
                (0x7000 + 2 * 3, 0x5000 + 2 * 3, 2 * 3, 1),
                (0x7000 + 5 * 3, 0x5000 + 5 * 3, 3, 1),
                (0x8000 + 2 * 4, 0x6000 + 2 * 4, 2 * 4, 1),
                (0x8000 + 5 * 4, 0x6000 + 5 * 4, 4, 1),
            ],
            hip.copy_calls,
        )

    def test_copy_state_host_to_gpu_skips_missing_indices(self):
        manager = make_manager()
        manager._host_state_staging_ptrs = [[0x5000]]
        manager._gpu_state_ptrs = [[0x7000]]
        hip = FakeHipLibrary()

        with patch.object(ctypes, "CDLL", return_value=hip):
            manager._copy_state_host_to_gpu(None)
            manager._copy_state_host_to_gpu([[]])

        self.assertEqual([], hip.copy_calls)

    def test_receiver_stores_state_indices_and_poll_copies_kv_and_state(self):
        receiver = object.__new__(MooncakeKVReceiver)
        receiver.kv_mgr = MagicMock()
        receiver.kv_mgr.check_status.return_value = KVPoll.Success
        receiver.kv_mgr._host_staging_enabled.return_value = True
        receiver.bootstrap_infos = []
        receiver.bootstrap_room = 1
        receiver.required_dst_info_num = 1
        receiver.conclude_state = None
        receiver.init_time = None

        receiver.send_metadata(
            kv_indices=[1, 2],
            aux_index=3,
            state_indices=[[4, 5]],
        )
        receiver.poll()

        self.assertEqual([1, 2], list(receiver._dst_kv_indices))
        self.assertEqual([[4, 5]], receiver._dst_state_indices)
        receiver.kv_mgr._copy_host_to_gpu.assert_called_once_with([1, 2])
        receiver.kv_mgr._copy_state_host_to_gpu.assert_called_once_with([[4, 5]])

    def test_registration_info_parses_state_host_staging_flag(self):
        message = [
            b"room",
            b"endpoint",
            b"1234",
            b"session",
            b"",
            b"",
            b"",
            b"0",
            b"1",
            b"0",
            b"",
            b"",
            b"",
            b"",
            b"",
            b"",
            b"1",
            b"0",
            b"1",
        ]

        info = KVArgsRegisterInfo.from_zmq(message)

        self.assertTrue(info.dst_state_host_staging)

    def test_register_kv_args_sends_state_host_staging_flag(self):
        receiver = object.__new__(MooncakeKVReceiver)
        receiver.session_id = "session"
        receiver.kv_mgr = MagicMock()
        receiver.kv_mgr.kv_args = SimpleNamespace(
            kv_data_ptrs=[],
            aux_data_ptrs=[],
            state_data_ptrs=[],
            state_item_lens=[],
            state_dim_per_tensor=[],
            state_layer_ids=[],
            kv_layer_ids=[],
            kv_item_lens=[],
            engine_rank=0,
            gpu_id=0,
        )
        receiver.kv_mgr.local_ip = "127.0.0.1"
        receiver.kv_mgr.rank_port = 1234
        receiver.kv_mgr.attn_tp_size = 8
        receiver.kv_mgr.enable_staging = False
        receiver.kv_mgr._host_staging_enabled.return_value = True
        receiver.bootstrap_infos = [
            {"rank_ip": "127.0.0.1", "rank_port": 2345, "is_dummy": False}
        ]
        socket = MagicMock()
        lock = MagicMock()

        with patch.object(
            MooncakeKVReceiver,
            "_connect_to_bootstrap_server",
            return_value=(socket, lock),
        ):
            self.assertTrue(receiver._register_kv_args())

        self.assertEqual(b"1", socket.send_multipart.call_args.args[0][-1])

    def test_state_host_staging_rejects_unequal_tp_sizes(self):
        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(state_types=[])
        manager.attn_tp_size = 8
        manager.pp_size = 1
        target = KVArgsRegisterInfo(
            room="room",
            endpoint="endpoint",
            dst_port=1234,
            mooncake_session_id="session",
            dst_kv_ptrs=[],
            dst_aux_ptrs=[],
            dst_state_data_ptrs=[],
            dst_tp_rank=0,
            dst_attn_tp_size=7,
            dst_kv_item_len=0,
            dst_state_item_lens=[],
            dst_state_dim_per_tensor=[],
            dst_kv_layer_ids=[],
            dst_state_layer_ids=[],
            dst_state_host_staging=True,
        )

        with self.assertRaisesRegex(RuntimeError, "equal attention TP sizes"):
            manager.maybe_send_extra(
                req=MagicMock(),
                prefill_state_indices=[],
                executor=MagicMock(),
                target_rank_registration_info=target,
            )


if __name__ == "__main__":
    unittest.main()
