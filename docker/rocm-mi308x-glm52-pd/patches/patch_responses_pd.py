#!/usr/bin/env python3
"""Patch sglang server for /v1/responses in PD (prefill-decode) disaggregation mode.

Applies to the overlaid sglang source at /sgl-workspace/sglang/python/sglang/.
Three changes:
  1. protocol.py: add bootstrap_host/port/room + disagg_prefill_dp_rank to
     ResponsesRequest so router-injected PD bootstrap survives Pydantic parsing.
  2. serving_responses.py: pass bootstrap fields to GenerateReqInput; coerce
     stream to bool (router may forward stream=null); break the builtin-tools
     loop after the first generate when disaggregation_mode==prefill (prefill
     only does prefill+KV-transfer, must not loop).
"""
import re, sys

PROTO = "/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/protocol.py"
RESP = "/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py"

# ---- 1. protocol.py: ResponsesRequest bootstrap fields ----
s = open(PROTO).read()
done_marker = "disagg_prefill_dp_rank: Optional[int] = None\n\n    # Default sampling parameters"
if done_marker not in s:
    anchor = "    # Default sampling parameters\n    _DEFAULT_SAMPLING_PARAMS = {"
    boot = (
        "    # For PD disaggregation\n"
        "    bootstrap_host: Optional[Union[List[str], str]] = None\n"
        "    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None\n"
        "    bootstrap_room: Optional[Union[List[int], int]] = None\n"
        "    disagg_prefill_dp_rank: Optional[int] = None\n"
        "\n"
    )
    assert s.count(anchor) == 1, "protocol anchor not found"
    s = s.replace(anchor, boot + anchor, 1)
    open(PROTO, "w").write(s)
    print("[OK] protocol.py: ResponsesRequest bootstrap fields added")
else:
    print("[OK] protocol.py: already patched")

# ---- 2. serving_responses.py ----
s = open(RESP).read()
changed = False

# 2a. stream=bool(request.stream) in ChatCompletionRequest
a1 = (
    "        chat_request = ChatCompletionRequest(\n"
    "            model=request.model,\n"
    "            messages=messages,\n"
    "            stream=request.stream,\n"
)
b1 = a1.replace("stream=request.stream,", "stream=bool(request.stream),")
if "stream=bool(request.stream)," not in s:
    assert s.count(a1) == 1, "stream anchor not found"
    s = s.replace(a1, b1, 1)
    changed = True
    print("[OK] serving_responses.py: stream coerced to bool")

# 2b. first GenerateReqInput: pass bootstrap from request
a2 = (
    "                        extra_key=self._compute_extra_key(request),\n"
    "                        background=request.background,\n"
    "                    )\n"
)
b2 = (
    "                        extra_key=self._compute_extra_key(request),\n"
    "                        background=request.background,\n"
    "                        bootstrap_host=request.bootstrap_host,\n"
    "                        bootstrap_port=request.bootstrap_port,\n"
    "                        bootstrap_room=request.bootstrap_room,\n"
    "                        disagg_prefill_dp_rank=request.disagg_prefill_dp_rank,\n"
    "                    )\n"
)
if "bootstrap_host=request.bootstrap_host," not in s:
    assert s.count(a2) == 1, "genreq anchor not found"
    s = s.replace(a2, b2, 1)
    changed = True
    print("[OK] serving_responses.py: bootstrap passed to GenerateReqInput")

# 2c. tool-loop GenerateReqInput: carry bootstrap from adapted_request
a3 = (
    "                return_hidden_states=adapted_request.return_hidden_states,\n"
    "                background=adapted_request.background,\n"
    "            )\n"
)
b3 = (
    "                return_hidden_states=adapted_request.return_hidden_states,\n"
    "                background=adapted_request.background,\n"
    "                bootstrap_host=adapted_request.bootstrap_host,\n"
    "                bootstrap_port=adapted_request.bootstrap_port,\n"
    "                bootstrap_room=adapted_request.bootstrap_room,\n"
    "                disagg_prefill_dp_rank=adapted_request.disagg_prefill_dp_rank,\n"
    "            )\n"
)
if "bootstrap_host=adapted_request.bootstrap_host," not in s and a3 in s:
    s = s.replace(a3, b3, 1)
    changed = True
    print("[OK] serving_responses.py: bootstrap added to tool-loop GenerateReqInput")

# 2d. loop-skip: break builtin-tools loop after first generate in PD prefill mode
loop_anchor = (
    "                yield context\n"
    "\n"
    "            if not context.need_builtin_tool_call():\n"
)
loop_ins = (
    "                yield context\n"
    "\n"
    "            # PD prefill mode: engine only prefills + transfers KV (no generation).\n"
    "            # Do not continue the builtin-tools loop; break after the single generate.\n"
    "            if self.tokenizer_manager.server_args.disaggregation_mode == DisaggregationMode.PREFILL:\n"
    "                break\n"
    "\n"
    "            if not context.need_builtin_tool_call():\n"
)
# Need DisaggregationMode import
if "DisaggregationMode" not in s:
    imp_anchor = "from sglang.srt.entrypoints.context import ("
    assert imp_anchor in s, "context import anchor not found"
    s = s.replace(imp_anchor, "from sglang.srt.disaggregation.utils import DisaggregationMode\n" + imp_anchor, 1)
    print("[OK] serving_responses.py: added DisaggregationMode import")
if "DisaggregationMode.PREFILL" not in s:
    assert s.count(loop_anchor) == 1, "loop anchor not found"
    s = s.replace(loop_anchor, loop_ins, 1)
    changed = True
    print("[OK] serving_responses.py: loop-skip in prefill mode")

if changed or "DisaggregationMode.PREFILL" in s:
    open(RESP, "w").write(s)
    print("[OK] serving_responses.py written")
