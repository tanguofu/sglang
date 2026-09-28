#!/usr/bin/env python3
"""Apply PD disaggregation patches to serving_responses.py.

Changes (from 1p1d-api-fix commit ae127462e8):
1. import re + _fix_created_at_int() helper (after __future__, not before)
2. import DisaggregationMode for prefill loop-skip check
3. pass bootstrap fields through to GenerateReqInput (both call sites)
4. stream=bool(request.stream) in ChatCompletionRequest (router may send null)
5. break the builtin-tools loop in prefill mode (prefill worker does one forward + KV transfer)
6. apply _fix_created_at_int to model_dump_json in both _send_event functions

Idempotent.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/port-post1/base/srt_entrypoints_openai_serving_responses.py")
src = path.read_text()
changed = []


def apply(label, did):
    changed.append(f"{label}: {'applied' if did else 'skipped'}")


# ---------- 1. import re + _fix_created_at_int helper ----------
if "_fix_created_at_int" in src:
    apply("created_at helper", False)
else:
    # Add `import re` after `import logging`
    if "import re\n" not in src:
        old_imp = "import logging\n"
        new_imp = "import logging\nimport re\n"
        assert old_imp in src, "serving_responses_pd: import logging anchor not found"
        src = src.replace(old_imp, new_imp, 1)

    # Add _fix_created_at_int after the logger definition
    old_logger = 'logger = logging.getLogger(__name__)\n'
    new_logger = (
        'logger = logging.getLogger(__name__)\n'
        '\n'
        '# codex expects created_at as i64; the openai SDK promotes it to float,\n'
        '# producing e.g. "created_at":1784036849.0 which codex fails to deserialize.\n'
        '_CREATED_AT_FLOAT_RE = re.compile(r\'"created_at":(\\d+)\\.0\\b\')\n'
        '\n'
        '\n'
        'def _fix_created_at_int(json_str: str) -> str:\n'
        '    """Strip .0 from created_at float values so codex (i64) can deserialize."""\n'
        '    return _CREATED_AT_FLOAT_RE.sub(r\'"created_at":\\1\', json_str)\n'
    )
    assert old_logger in src, "serving_responses_pd: logger anchor not found"
    src = src.replace(old_logger, new_logger, 1)
    apply("created_at helper", True)

# ---------- 2. import DisaggregationMode ----------
if "from sglang.srt.disaggregation.utils import DisaggregationMode" in src:
    apply("DisaggregationMode import", False)
else:
    # Add after GenerateReqInput import
    old_imp = "from sglang.srt.managers.io_struct import GenerateReqInput\n"
    new_imp = (
        "from sglang.srt.managers.io_struct import GenerateReqInput\n"
        "from sglang.srt.disaggregation.utils import DisaggregationMode\n"
    )
    assert old_imp in src, "serving_responses_pd: GenerateReqInput import anchor not found"
    src = src.replace(old_imp, new_imp, 1)
    apply("DisaggregationMode import", True)

# ---------- 3. bootstrap fields in first GenerateReqInput call ----------
if "bootstrap_host=request.bootstrap_host" in src:
    apply("bootstrap forwarding", False)
else:
    # First call site: anchor on require_reasoning=require_reasoning, which is the
    # last arg before closing paren in v0.5.17. Insert bootstrap fields after it.
    old_fwd = "                        require_reasoning=require_reasoning,\n                    )"
    new_fwd = (
        "                        require_reasoning=require_reasoning,\n"
        "                        bootstrap_host=request.bootstrap_host,\n"
        "                        bootstrap_port=request.bootstrap_port,\n"
        "                        bootstrap_room=request.bootstrap_room,\n"
        "                        disagg_prefill_dp_rank=request.disagg_prefill_dp_rank,\n"
        "                    )"
    )
    if old_fwd not in src:
        # Fallback: older pattern with background=request.background,
        old_fwd = "                        background=request.background,\n                    )"
        new_fwd = (
            "                        background=request.background,\n"
            "                        bootstrap_host=request.bootstrap_host,\n"
            "                        bootstrap_port=request.bootstrap_port,\n"
            "                        bootstrap_room=request.bootstrap_room,\n"
            "                        disagg_prefill_dp_rank=request.disagg_prefill_dp_rank,\n"
            "                    )"
        )
    assert old_fwd in src, "serving_responses_pd: first GenerateReqInput anchor not found"
    src = src.replace(old_fwd, new_fwd, 1)
    apply("bootstrap forwarding", True)

    # Second call site: after background=adapted_request.background,
    old_fwd2 = "                background=adapted_request.background,\n            )"
    new_fwd2 = (
        "                background=adapted_request.background,\n"
        "                bootstrap_host=adapted_request.bootstrap_host,\n"
        "                bootstrap_port=adapted_request.bootstrap_port,\n"
        "                bootstrap_room=adapted_request.bootstrap_room,\n"
        "                disagg_prefill_dp_rank=adapted_request.disagg_prefill_dp_rank,\n"
        "            )"
    )
    if old_fwd2 in src:
        src = src.replace(old_fwd2, new_fwd2, 1)
        changed.append("bootstrap forwarding (2nd site): applied")
    else:
        changed.append("bootstrap forwarding (2nd site): anchor not found, skipped")

# ---------- 4. stream=bool(request.stream) ----------
old_stream = "            stream=request.stream,\n            tools=chat_tools or None,"
new_stream = "            stream=bool(request.stream),\n            tools=chat_tools or None,"
if new_stream in src:
    apply("stream guard", False)
else:
    assert old_stream in src, "serving_responses_pd: stream= anchor not found"
    src = src.replace(old_stream, new_stream, 1)
    apply("stream guard", True)

# ---------- 5. prefill loop-skip ----------
if "disaggregation_mode == DisaggregationMode.PREFILL.value" in src:
    apply("prefill loop-skip", False)
else:
    # Add after the yield context in the builtin tools loop, before need_builtin_tool_call check
    old_loop = (
        "            if not context.need_builtin_tool_call():\n"
        "                # The model did not ask for a tool call, so we're done.\n"
        "                break"
    )
    new_loop = (
        "            # In PD prefill mode, the prefill worker only does one forward\n"
        "            # pass and transfers KV; it must not loop waiting for tool calls.\n"
        "            if self.tokenizer_manager.server_args.disaggregation_mode == DisaggregationMode.PREFILL.value:\n"
        "                break\n"
        "\n"
        "            if not context.need_builtin_tool_call():\n"
        "                # The model did not ask for a tool call, so we're done.\n"
        "                break"
    )
    assert old_loop in src, "serving_responses_pd: prefill loop-skip anchor not found"
    src = src.replace(old_loop, new_loop, 1)
    apply("prefill loop-skip", True)

# ---------- 6. _fix_created_at_int in _send_event functions ----------
count_before = src.count("_fix_created_at_int(event.model_dump_json(indent=None))")
total_send_events = src.count("event.model_dump_json(indent=None)")
if count_before > 0 and count_before >= total_send_events:
    apply("created_at in _send_event", False)
else:
    old_dump = 'f"data: {event.model_dump_json(indent=None)}\\n\\n"'
    new_dump = 'f"data: {_fix_created_at_int(event.model_dump_json(indent=None))}\\n\\n"'
    n = src.count(old_dump)
    if n > 0:
        src = src.replace(old_dump, new_dump)
        apply(f"created_at in _send_event ({n} sites)", True)
    else:
        apply("created_at in _send_event (anchor not found)", False)

path.write_text(src)
print(f"[ok] {path}: " + ", ".join(changed))
