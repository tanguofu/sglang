import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[4]
    / "docker/rocm-mi308x-glm52-pd/scripts/bench_200k_ttft_tfot.py"
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, ids):
        return "".join(ids)


class TestBench200kTtftTfot(unittest.TestCase):
    def test_build_messages_reaches_target_tokens(self):
        spec = importlib.util.spec_from_file_location("bench_200k_ttft_tfot", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        messages = module.build_messages(
            FakeTokenizer(),
            target_tokens=200000,
            salt="salt",
            random_filler=False,
        )

        self.assertEqual(len(messages), 2)
        self.assertGreaterEqual(
            len(messages[1]["content"]),
            199000,
        )


if __name__ == "__main__":
    unittest.main()
