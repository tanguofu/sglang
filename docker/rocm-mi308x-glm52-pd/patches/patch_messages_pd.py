#!/usr/bin/env python3
"""Patch sglang Anthropic /v1/messages backend for PD (prefill-decode) disaggregation.

Applies to the overlaid sglang source at /sgl-workspace/sglang/python/sglang/.
Two changes:
  1. anthropic/protocol.py: add bootstrap_host/port/room + disagg_prefill_dp_rank
     to AnthropicMessagesRequest so router-injected PD bootstrap survives Pydantic
     parsing (default extra="ignore" would otherwise drop these fields).
  2. anthropic/serving.py: in _convert_to_chat_completion_request, after
     chat_request = ChatCompletionRequest(**request_data), forward the 4 bootstrap
     fields from anthropic_request onto chat_request so they reach the prefill
     worker via GenerateReqInput.

Idempotent: re-running on an already-patched tree is a no-op.
"""

PROTO = "/sgl-workspace/sglang/python/sglang/srt/entrypoints/anthropic/protocol.py"
SERVING = "/sgl-workspace/sglang/python/sglang/srt/entrypoints/anthropic/serving.py"

# ---- 1. protocol.py: add bootstrap fields to AnthropicMessagesRequest ----
s = open(PROTO).read()
if "bootstrap_host: Optional" not in s:
    # Anchor: the betas field is the last field before the validator (verified
    # at protocol.py:380). Pydantic v2 default extra="ignore" would otherwise
    # drop the router-injected bootstrap fields.
    anchor = '    betas: Optional[list[str]] = None\n\n    @field_validator("model")'
    boot = (
        '    betas: Optional[list[str]] = None\n\n'
        '    # For PD disaggregation (injected by the PD router)\n'
        '    bootstrap_host: Optional[Union[list[str], str]] = None\n'
        '    bootstrap_port: Optional[Union[list[Optional[int]], int]] = None\n'
        '    bootstrap_room: Optional[Union[list[int], int]] = None\n'
        '    disagg_prefill_dp_rank: Optional[int] = None\n'
        '\n'
        '    @field_validator("model")'
    )
    assert s.count(anchor) == 1, "anthropic protocol anchor not found"
    s = s.replace(anchor, boot, 1)
    open(PROTO, "w").write(s)
    print("[OK] anthropic/protocol.py: bootstrap fields added to AnthropicMessagesRequest")
else:
    print("[OK] anthropic/protocol.py: already patched")

# ---- 2. serving.py: forward bootstrap fields to chat_request ----
s = open(SERVING).read()
marker = "        chat_request.disagg_prefill_dp_rank = anthropic_request.disagg_prefill_dp_rank"
if marker not in s:
    # Anchor: chat_request is built from request_data at serving.py:563.
    # Bootstrap fields arrive on anthropic_request (added above) but are not
    # part of request_data, so they must be copied across explicitly.
    anchor = "        chat_request = ChatCompletionRequest(**request_data)\n"
    fwd = (
        anchor +
        "\n"
        "        # Forward PD disaggregation bootstrap fields injected by the router.\n"
        "        # These arrive on AnthropicMessagesRequest (added to the protocol\n"
        "        # for PD mode) and must reach GenerateReqInput via serving_chat.\n"
        "        chat_request.bootstrap_host = anthropic_request.bootstrap_host\n"
        "        chat_request.bootstrap_port = anthropic_request.bootstrap_port\n"
        "        chat_request.bootstrap_room = anthropic_request.bootstrap_room\n"
        "        chat_request.disagg_prefill_dp_rank = anthropic_request.disagg_prefill_dp_rank\n"
    )
    assert s.count(anchor) == 1, "anthropic serving anchor not found"
    s = s.replace(anchor, fwd, 1)
    open(SERVING, "w").write(s)
    print("[OK] anthropic/serving.py: bootstrap forwarding added")
else:
    print("[OK] anthropic/serving.py: already patched")

print("\n=== /v1/messages PD patches applied ===")
