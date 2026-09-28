import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.http_server import (
    _send_disaggregation_warmup_requests,
    _freeze_gc_after_server_warmup,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDisaggregationServerWarmup(unittest.IsolatedAsyncioTestCase):
    async def test_sends_concurrent_scalar_request_to_each_dp_rank(self):
        from sglang.srt.runtime_context import get_context

        # The warmup fan-out width comes from the published topology.
        override = get_context().override_server_args(dp_size=4)
        override.install()
        self.addCleanup(override.restore)
        server_args = SimpleNamespace(dp_size=4)
        all_started = asyncio.Event()
        calls = []
        sessions = []

        class Response:
            status = 200

            async def __aenter__(self):
                if len(calls) == server_args.dp_size:
                    all_started.set()
                await asyncio.wait_for(all_started.wait(), timeout=5)
                return self

            async def __aexit__(self, *args):
                pass

            async def read(self):
                return b""

        class Session:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                sessions.append(self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def post(self, *args, **kwargs):
                calls.append((args, kwargs))
                return Response()

        with patch("sglang.srt.entrypoints.http_server.aiohttp.ClientSession", Session):
            status_codes = await _send_disaggregation_warmup_requests(
                url="http://localhost:30000",
                headers={"Authorization": "Bearer token"},
                ssl_verify=False,
                timeout=123,
            )

        self.assertEqual(status_codes, [200] * server_args.dp_size)
        self.assertEqual(len(calls), server_args.dp_size)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(
            sessions[0].kwargs["headers"], {"Authorization": "Bearer token"}
        )
        self.assertEqual(sessions[0].kwargs["timeout"].total, 123)

        calls_by_rank = {
            kwargs["json"]["routed_dp_rank"]: (args, kwargs) for args, kwargs in calls
        }
        self.assertEqual(set(calls_by_rank), set(range(server_args.dp_size)))

        for dp_rank, (args, kwargs) in calls_by_rank.items():
            self.assertEqual(args, ("http://localhost:30000/generate",))
            self.assertEqual(kwargs["json"]["input_ids"], [10, 11, 12, 13])
            self.assertEqual(kwargs["json"]["bootstrap_host"], FAKE_BOOTSTRAP_HOST)
            self.assertEqual(kwargs["json"]["bootstrap_room"], dp_rank)
            self.assertFalse(kwargs["ssl"])


class TestFreezeGcAfterServerWarmup(unittest.TestCase):
    def test_retries_transient_connection_refused(self):
        class ServerArgs:
            api_key = "test-key"
            admin_api_key = None

            def url(self):
                return "http://127.0.0.1:31000"

            def ssl_verify(self):
                return False

        attempts = []
        sleeps = []

        def post(*args, **kwargs):
            attempts.append((args, kwargs))
            if len(attempts) < 3:
                raise requests.exceptions.ConnectionError("connection refused")
            return SimpleNamespace(raise_for_status=lambda: None)

        with (
            patch("sglang.srt.entrypoints.http_server.requests.post", side_effect=post),
            patch(
                "sglang.srt.entrypoints.http_server.time.sleep",
                side_effect=sleeps.append,
            ),
        ):
            _freeze_gc_after_server_warmup(ServerArgs())

        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleeps, [1, 1])
        self.assertEqual(
            attempts[0][1]["headers"], {"Authorization": "Bearer test-key"}
        )

    def test_stops_after_five_failed_attempts(self):
        class ServerArgs:
            api_key = None
            admin_api_key = None

            def url(self):
                return "http://127.0.0.1:31000"

            def ssl_verify(self):
                return False

        attempts = []
        sleeps = []

        def post(*args, **kwargs):
            attempts.append((args, kwargs))
            raise requests.exceptions.ConnectionError("connection refused")

        with (
            patch("sglang.srt.entrypoints.http_server.requests.post", side_effect=post),
            patch(
                "sglang.srt.entrypoints.http_server.time.sleep",
                side_effect=sleeps.append,
            ),
            self.assertLogs(
                "sglang.srt.entrypoints.http_server", level="WARNING"
            ) as logs,
        ):
            _freeze_gc_after_server_warmup(ServerArgs())

        self.assertEqual(len(attempts), 5)
        self.assertEqual(sleeps, [1, 1, 1, 1])
        self.assertTrue(any("after 5 attempts" in message for message in logs.output))


if __name__ == "__main__":
    unittest.main()
