import unittest

from sglang.srt.entrypoints.openai.serving_chat import (
    normalize_assistant_tool_call_arguments,
    parse_tool_call_arguments,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _assistant_message(arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": arguments},
            }
        ],
    }


class TestToolArgsTolerant(CustomTestCase):
    def test_valid_arguments_are_parsed(self):
        msg = _assistant_message('{"path": "/tmp/a.txt"}')
        normalize_assistant_tool_call_arguments(msg)
        self.assertEqual(
            msg["tool_calls"][0]["function"]["arguments"], {"path": "/tmp/a.txt"}
        )

    def test_truncated_arguments_degrade_instead_of_raising(self):
        raw = '{"path": "/tmp/a.tx'
        msg = _assistant_message(raw)
        normalize_assistant_tool_call_arguments(msg)
        self.assertEqual(msg["tool_calls"][0]["function"]["arguments"], {"_raw": raw})

    def test_empty_arguments_degrade_instead_of_raising(self):
        msg = _assistant_message("")
        normalize_assistant_tool_call_arguments(msg)
        self.assertEqual(msg["tool_calls"][0]["function"]["arguments"], {"_raw": ""})

    def test_non_dict_arguments_degrade_instead_of_raising(self):
        raw = "[1, 2, 3]"
        msg = _assistant_message(raw)
        normalize_assistant_tool_call_arguments(msg)
        self.assertEqual(msg["tool_calls"][0]["function"]["arguments"], {"_raw": raw})

    def test_dict_arguments_are_left_untouched(self):
        msg = _assistant_message({"path": "/tmp/a.txt"})
        normalize_assistant_tool_call_arguments(msg)
        self.assertEqual(
            msg["tool_calls"][0]["function"]["arguments"], {"path": "/tmp/a.txt"}
        )

    def test_non_assistant_and_missing_tool_calls_are_noop(self):
        msg = {"role": "user", "content": "hi"}
        normalize_assistant_tool_call_arguments(msg)
        self.assertEqual(msg, {"role": "user", "content": "hi"})

        msg = {"role": "assistant", "content": "hi"}
        normalize_assistant_tool_call_arguments(msg)
        self.assertEqual(msg, {"role": "assistant", "content": "hi"})

    def test_parse_still_raises_for_direct_callers(self):
        with self.assertRaises(ValueError):
            parse_tool_call_arguments("{not json")


if __name__ == "__main__":
    unittest.main()
