"""Decode health-check head-of-line blocking in `pop_preallocated`.

`DecodePreallocQueue.pop_preallocated` admits from the head of the queue and
`break`s on the first request it cannot admit. A health-check request appended
behind a request that does not fit (too much KV) therefore never runs, and
because `Scheduler.is_fully_idle(for_health_check=True)` deliberately ignores
the PD queues it is also never deferred into `return_health_check_ipcs`, so no
health signal is ever emitted: `/health` answers 503 forever with a live
scheduler loop and nothing in the logs.

`SGLANG_DS41_HEALTH_NO_HOL_BLOCK=1` moves the (1-token) health request to the
front of the scan so it is admitted instead. Every capacity check still applies
to it, so this cannot over-admit.

Pure Python with stubs — no GPU, no model.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeRequest,
    health_first_scan_order,
)

HEALTH_RID = f"{HEALTH_CHECK_RID_PREFIX}0000"


class _FakeReq:
    def __init__(self, rid, input_len):
        self.rid = rid
        self.origin_input_ids = list(range(input_len))
        self.output_ids = []
        self.finished_reason = None
        self.finished_output = False
        self.return_logprob = False
        self.bootstrap_host = "1.2.3.4"
        self.bootstrap_room = 0
        self.is_rebootstrap = False
        self.kv = SimpleNamespace(req_pool_idx=0, cache_protected_len=0)


def _decode_req(rid, input_len):
    return DecodeRequest(
        req=_FakeReq(rid, input_len),
        kv_receiver=SimpleNamespace(supports_host_destination=False),
        waiting_for_input=True,
    )


class _StubPreallocQueue(DecodePreallocQueue):
    """`DecodePreallocQueue` with everything but the admission loop stubbed."""

    def __init__(self, queue, *, allocatable_tokens):
        self.queue = list(queue)
        self.pp_size = 1
        self.scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[]),
            enable_priority_scheduling=False,
            enable_hisparse=False,
            enable_lora=False,
            enable_decode_hicache=False,
        )
        self.token_to_kv_pool = None
        self.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=256, translate_kv_indices_for_transfer=lambda x: x
        )
        # 8 request rows x 128 slots, enough for the shapes used here.
        self.req_to_token_pool = SimpleNamespace(
            available_size=lambda: 8,
            req_to_token=np.zeros((8, 128), dtype=np.int64),
        )
        self.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 8
        )
        self.kv_manager = SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
        self.tree_cache = None
        self.allocatable_tokens = allocatable_tokens
        self.sent_metadata = []

    # --- stubs for everything the admission loop calls -----------------------
    def _resolve_pending_reqs(self, *args, **kwargs):
        return None

    def _update_handshake_waiters(self, *args, **kwargs):
        return None

    def _uses_swa_tail_prealloc(self):
        return False

    def _uses_swa_reservation(self):
        return False

    def _allocatable_token_budgets(self, **kwargs):
        return self.allocatable_tokens

    def _hicache_pending_restore_tokens(self):
        return 0

    def _rebootstrap_prefill_len(self, req):
        return len(req.origin_input_ids)

    def _pre_alloc_fill_len(self, req):
        return len(req.origin_input_ids)

    def _required_alloc_tokens(self, **kwargs):
        return kwargs["fill_len"] - kwargs["prefix_len"]

    def _required_admission_tokens(
        self, req, required_alloc_tokens, prefix_len, retractable_tokens
    ):
        return required_alloc_tokens

    def _prealloc_reservation_fits(
        self, full_tokens, swa_tokens, *, full_allocatable_tokens, swa_allocatable_tokens
    ):
        # The only real admission gate here: an un-admittable head `break`s the
        # scan before the requests behind it are ever considered.
        return full_tokens <= full_allocatable_tokens

    def _pre_alloc(self, req, prefix_indices, prefix_len, total_prefix_len):
        return None

    def _send_kv_metadata(self, decode_req, *args, **kwargs):
        self.sent_metadata.append(decode_req.req.rid)

    # --- the code under test -------------------------------------------------
    def pop(self):
        return DecodePreallocQueue.pop_preallocated(self)


_PATCHES = [
    mock.patch(
        "sglang.srt.disaggregation.decode.get_disagg",
        lambda: SimpleNamespace(
            disaggregation_decode_host_receive_threshold=1,
            disaggregation_decode_enable_radix_cache=False,
        ),
    ),
    mock.patch(
        "sglang.srt.disaggregation.decode.ensure_mamba_capacity",
        lambda *args, **kwargs: True,
    ),
]


class TestHealthFirstScanOrder(unittest.TestCase):
    def test_disabled_keeps_fifo_order(self):
        queue = [_decode_req("a", 4), _decode_req(HEALTH_RID, 1)]
        with mock.patch("sglang.srt.disaggregation.decode._HEALTH_NO_HOL_BLOCK", False):
            self.assertEqual([d.req.rid for d in health_first_scan_order(queue)], ["a", HEALTH_RID])

    def test_enabled_moves_health_request_to_front(self):
        queue = [_decode_req("a", 4), _decode_req(HEALTH_RID, 1)]
        with mock.patch("sglang.srt.disaggregation.decode._HEALTH_NO_HOL_BLOCK", True):
            self.assertEqual([d.req.rid for d in health_first_scan_order(queue)], [HEALTH_RID, "a"])

    def test_enabled_is_stable_for_the_rest_of_the_queue(self):
        queue = [_decode_req("a", 4), _decode_req("b", 4), _decode_req(HEALTH_RID, 1)]
        with mock.patch("sglang.srt.disaggregation.decode._HEALTH_NO_HOL_BLOCK", True):
            self.assertEqual(
                [d.req.rid for d in health_first_scan_order(queue)],
                [HEALTH_RID, "a", "b"],
            )

    def test_no_health_request_is_a_noop(self):
        queue = [_decode_req("a", 4), _decode_req("b", 4)]
        with mock.patch("sglang.srt.disaggregation.decode._HEALTH_NO_HOL_BLOCK", True):
            self.assertIs(health_first_scan_order(queue), queue)


class TestPopPreallocatedHealthHolBlock(unittest.TestCase):
    def setUp(self):
        for patcher in _PATCHES:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _run(self, enabled, *, allocatable_tokens=100):
        # Head request needs 4096 tokens, only `allocatable_tokens` are free.
        queue = _StubPreallocQueue(
            [_decode_req("big", 4096), _decode_req(HEALTH_RID, 1)],
            allocatable_tokens=allocatable_tokens,
        )
        with mock.patch("sglang.srt.disaggregation.decode._HEALTH_NO_HOL_BLOCK", enabled):
            preallocated, failed = queue.pop()
        return [d.req.rid for d in preallocated], [d.req.rid for d in failed], queue

    def test_default_blocks_the_health_request_behind_an_unadmittable_head(self):
        preallocated, failed, queue = self._run(False)
        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [])
        # Both stay queued: the head `break`s the scan, so the tail is never
        # looked at -> no health signal, permanent 503.
        self.assertEqual([d.req.rid for d in queue.queue], ["big", HEALTH_RID])

    def test_enabled_admits_the_health_request_and_keeps_the_head(self):
        preallocated, failed, queue = self._run(True)
        self.assertEqual(preallocated, [HEALTH_RID])
        self.assertEqual(failed, [])
        # The un-admittable request is not dropped or reordered away.
        self.assertEqual([d.req.rid for d in queue.queue], ["big"])
        self.assertEqual(queue.sent_metadata, [HEALTH_RID])

    def test_enabled_does_not_admit_a_health_request_that_does_not_fit(self):
        preallocated, failed, queue = self._run(True, allocatable_tokens=0)
        # Reordered, but the capacity gate still applies: nothing is admitted.
        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [])
        self.assertEqual([d.req.rid for d in queue.queue], [HEALTH_RID, "big"])
        self.assertEqual(queue.sent_metadata, [])

    def test_enabled_still_admits_the_request_it_jumped_when_that_one_fits(self):
        # A 50-token head and a 1-token health request both fit in 100 tokens:
        # the health request is admitted first, the head is not starved.
        queue = _StubPreallocQueue(
            [_decode_req("mid", 50), _decode_req(HEALTH_RID, 1)],
            allocatable_tokens=100,
        )
        with mock.patch("sglang.srt.disaggregation.decode._HEALTH_NO_HOL_BLOCK", True):
            preallocated, failed = queue.pop()
        self.assertEqual([d.req.rid for d in preallocated], [HEALTH_RID, "mid"])
        self.assertEqual(failed, [])
        self.assertEqual(queue.queue, [])


if __name__ == "__main__":
    unittest.main()
