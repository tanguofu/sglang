import threading
import unittest
from collections import defaultdict
from unittest.mock import patch

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMooncakeSessionBlacklist(CustomTestCase):
    @staticmethod
    def _make_manager(threshold=3, ttl=300.0):
        manager = object.__new__(MooncakeKVManager)
        manager.session_lock = threading.Lock()
        manager.session_failures = defaultdict(int)
        manager.failed_sessions = set()
        manager.failed_session_times = {}
        manager.session_failure_threshold = threshold
        manager.failed_session_ttl_s = ttl
        return manager

    def test_single_failure_does_not_blacklist_session(self):
        manager = self._make_manager()

        manager._record_session_failure("session-1")

        self.assertEqual(manager.session_failures["session-1"], 1)
        self.assertFalse(manager._is_session_blacklisted("session-1"))

    def test_failure_threshold_blacklists_session(self):
        manager = self._make_manager(threshold=3)

        for _ in range(3):
            manager._record_session_failure("session-1")

        self.assertEqual(manager.session_failures["session-1"], 3)
        self.assertTrue(manager._is_session_blacklisted("session-1"))

    def test_blacklist_expires_after_ttl(self):
        manager = self._make_manager(threshold=2, ttl=300.0)

        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic",
            return_value=100.0,
        ):
            manager._record_session_failure("session-1")
            manager._record_session_failure("session-1")
            self.assertTrue(manager._is_session_blacklisted("session-1"))

        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic",
            return_value=401.0,
        ):
            self.assertFalse(manager._is_session_blacklisted("session-1"))

        self.assertNotIn("session-1", manager.session_failures)
        self.assertNotIn("session-1", manager.failed_session_times)

    def test_clear_session_failure_resets_state(self):
        manager = self._make_manager(threshold=1)
        manager._record_session_failure("session-1")

        manager._clear_session_failure("session-1")

        self.assertFalse(manager._is_session_blacklisted("session-1"))
        self.assertNotIn("session-1", manager.session_failures)
        self.assertNotIn("session-1", manager.failed_session_times)


if __name__ == "__main__":
    unittest.main()
