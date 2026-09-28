#!/usr/bin/env python3
"""Skip JSON Schema format:regex on tool.parameters.

Codex / Claude Code send JS Unicode-property patterns such as \\p{Cc}.
jsonschema's default format checker compiles pattern with Python re,
which rejects \\p and returns 400 decode_bad_request. Those patterns
are prompts for the model, not executed by the server.

Idempotent.
"""
import sys
from pathlib import Path

TARGET = Path(
    "/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py"
)
if len(sys.argv) > 1:
    TARGET = Path(sys.argv[1])

OLD = "Draft202012Validator.check_schema(tool.function.parameters)"
NEW = "Draft202012Validator.check_schema(tool.function.parameters, format_checker=None)"


def main() -> None:
    src = TARGET.read_text()
    if NEW in src:
        print("patch_tool_schema_regex: already applied, skipping")
        sys.exit(0)
    if src.count(OLD) != 1:
        raise RuntimeError(
            f"patch_tool_schema_regex: anchor count={src.count(OLD)}"
        )
    TARGET.write_text(src.replace(OLD, NEW, 1))
    verify = TARGET.read_text()
    if NEW not in verify:
        raise RuntimeError("patch_tool_schema_regex: patch missing after write")
    print("patch_tool_schema_regex: SUCCESS")


if __name__ == "__main__":
    main()
