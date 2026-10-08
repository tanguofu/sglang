import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase

    register_cpu_ci(est_time=2, suite="base-a-test-cpu")
except ImportError:
    register_cpu_ci = None
    CustomTestCase = unittest.TestCase

SCRIPT = (
    Path(__file__).parents[4]
    / "docker/rocm-mi308x-glm52-pd/patches/patch_tool_args_tolerant.py"
)


OLD = '''        if "arguments" in function and isinstance(function["arguments"], str):
            try:
                function["arguments"] = parse_tool_call_arguments(function["arguments"])
            except ValueError:
                if strict:
                    raise
'''


class TestPatchToolArgsTolerant(CustomTestCase):
    def test_patch_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "serving_chat.py"
            target.write_text(OLD, encoding="utf-8")

            first = subprocess.run(
                [sys.executable, str(SCRIPT), str(target)],
                capture_output=True,
                text=True,
                check=True,
            )
            patched = target.read_text(encoding="utf-8")
            self.assertIn('function["arguments"] = {"_raw": function["arguments"]}', patched)
            self.assertIn("patch_tool_args_tolerant: SUCCESS", first.stdout)

            second = subprocess.run(
                [sys.executable, str(SCRIPT), str(target)],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("patch_tool_args_tolerant: already applied, skipping", second.stdout)
            self.assertEqual(patched, target.read_text(encoding="utf-8"))

    def test_patch_rejects_unexpected_anchor_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "serving_chat.py"
            target.write_text("print('unexpected source')\n", encoding="utf-8")

            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [sys.executable, str(SCRIPT), str(target)],
                    capture_output=True,
                    text=True,
                    check=True,
                )


if __name__ == "__main__":
    unittest.main()
