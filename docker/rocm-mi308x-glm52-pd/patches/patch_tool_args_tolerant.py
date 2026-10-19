#!/usr/bin/env python3
"""Tolerate invalid JSON in assistant history tool_call arguments (glm47).

Codex/cc-switch replay assistant tool calls back in the conversation history.
When a previous stream was cut mid-tool-call (client abort, 600s first-byte
timeout) or the model emitted a degenerate value, the replayed
``function.arguments`` string is truncated or otherwise invalid JSON.
``normalize_assistant_tool_call_arguments`` (strict for glm47) then raises
"Assistant tool call function.arguments must be valid JSON." and the prefill
worker returns 400 — permanently wedging that conversation, because the client
keeps replaying the same broken history.

Fix: on parse failure, degrade to ``{"_raw": <original string>}`` instead of
raising. The GLM-5.3 chat template iterates ``arguments.items()`` and renders
``<arg_key>_raw</arg_key><arg_value>...so the conversation still renders (the model sees the raw text as a single
argument) and the request proceeds instead of 400-ing.

Idempotent. Safe to run repeatedly.
"""
import sys
from pathlib import Path

TARGET = Path(
    "/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py"
)
if len(sys.argv) > 1:
    TARGET = Path(sys.argv[1])

OLD = """        if "arguments" in function and isinstance(function["arguments"], str):
            try:
                function["arguments"] = parse_tool_call_arguments(function["arguments"])
            except ValueError:
                if strict:
                    raise"""

NEW = """        if "arguments" in function and isinstance(function["arguments"], str):
            try:
                function["arguments"] = parse_tool_call_arguments(function["arguments"])
            except ValueError:
                if strict:
                    # FIX(tool-args-tolerant): a replayed assistant tool call
                    # with invalid JSON arguments (truncated stream, degenerate
                    # value) must not 400 the whole conversation. Degrade to a
                    # single _raw argument so the chat template still renders.
                    function["arguments"] = {"_raw": function["arguments"]}"""


def main() -> None:
    src = TARGET.read_text()
    if "FIX(tool-args-tolerant)" in src:
        print("patch_tool_args_tolerant: already applied, skipping")
        sys.exit(0)
    if src.count(OLD) != 1:
        raise RuntimeError(
            f"patch_tool_args_tolerant: anchor count={src.count(OLD)}"
        )
    TARGET.write_text(src.replace(OLD, NEW, 1))
    verify = TARGET.read_text()
    if "FIX(tool-args-tolerant)" not in verify:
        raise RuntimeError("patch_tool_args_tolerant: patch missing after write")
    print("patch_tool_args_tolerant: SUCCESS")


if __name__ == "__main__":
    main()
