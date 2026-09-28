#!/usr/bin/env python3
"""Add PD disaggregation bootstrap fields to ResponsesRequest in protocol.py.

The PD router (sgl-model-gateway) forwards bootstrap_host/port/room from the
prefill worker to the decode worker via the request payload. The upstream
protocol.py has these fields on CompletionRequest and ChatCompletionRequest
but NOT on ResponsesRequest (used by codex). This patch adds them.

Idempotent — skips if fields already present.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/port-post1/base/srt_entrypoints_openai_protocol.py")
src = path.read_text()
changed = []

# Add bootstrap fields to ResponsesRequest
if "bootstrap_host" in src and "class ResponsesRequest" in src:
    # Check if ResponsesRequest specifically has bootstrap_host
    # Find the class body and check
    idx = src.index("class ResponsesRequest")
    # Look ahead 5000 chars for bootstrap_host
    if "bootstrap_host" in src[idx:idx+5000]:
        changed.append("ResponsesRequest bootstrap: skipped")
    else:
        changed.append("ResponsesRequest bootstrap: NEEDS PATCH")
else:
    changed.append("ResponsesRequest bootstrap: NEEDS PATCH")

if "NEEDS PATCH" in changed[-1]:
    # Anchor: the repetition_penalty field followed by Default sampling params
    # in ResponsesRequest. This pattern appears once in the class.
    anchor = (
        "    repetition_penalty: Optional[float] = None\n"
        "\n"
        "    # Default sampling parameters\n"
        "    _DEFAULT_SAMPLING_PARAMS = {"
    )
    replacement = (
        "    repetition_penalty: Optional[float] = None\n"
        "\n"
        "    # PD disaggregation bootstrap fields (forwarded to prefill worker by router)\n"
        "    bootstrap_host: Optional[Union[List[str], str]] = None\n"
        "    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None\n"
        "    bootstrap_room: Optional[Union[List[int], int]] = None\n"
        "    disagg_prefill_dp_rank: Optional[int] = None\n"
        "\n"
        "    # Default sampling parameters\n"
        "    _DEFAULT_SAMPLING_PARAMS = {"
    )
    assert anchor in src, "protocol_responses: ResponsesRequest anchor not found"
    src = src.replace(anchor, replacement, 1)
    changed[-1] = "ResponsesRequest bootstrap: applied"

path.write_text(src)
print(f"[ok] {path}: " + ", ".join(changed))
