#!/usr/bin/env python3
"""Guardrail: make /v1/responses return a clean 404 on this PD deployment.

Codex traffic is transformed to /v1/chat/completions by cc-switch, so the
Responses API is intentionally NOT served here. A stray /v1/responses request
used to reach the KV-transfer path with a list-typed ``bootstrap_room`` and
crash the prefill worker (``TypeError: unhashable type: 'list'``, exit 137).

Rather than half-serving that endpoint, we rewrite the three Responses route
handlers in ``http_server.py`` on disk to return HTTP 404 with an OpenAI-style
error envelope. Like the other patches in this dir, this runs as a standalone
process BEFORE ``exec python3 -m sglang.launch_server``; it physically edits the
site-packages source so the launched server imports the guarded version.

Idempotent. Safe to run repeatedly.

Usage:
    python3 patch_responses_404.py [http_server.py path]
"""
import os
import py_compile
import sys

HTTP_SERVER_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py"
)

MARKER = "FIX(responses-404-guardrail)"

GUARD = (
    "    # {marker}: Responses API is disabled on this PD deployment.\n"
    "    from fastapi.responses import ORJSONResponse as _ORJSONResponse\n"
    "    return _ORJSONResponse(\n"
    "        status_code=404,\n"
    "        content={{\n"
    '            "error": {{\n'
    '                "message": "The Responses API (/v1/responses) is not '
    'available on this deployment. Use /v1/chat/completions.",\n'
    '                "type": "invalid_request_error",\n'
    '                "param": None,\n'
    '                "code": "not_found",\n'
    "            }}\n"
    "        }},\n"
    "    )\n"
).format(marker=MARKER)

# (anchor line that ends the handler signature, indentation-preserving)
ANCHORS = [
    "async def v1_responses_request(request: ResponsesRequest, raw_request: Request):\n",
    "async def v1_retrieve_responses(response_id: str, raw_request: Request):\n",
    "async def v1_cancel_responses(response_id: str, raw_request: Request):\n",
]


def main() -> int:
    with open(HTTP_SERVER_PATH) as f:
        src = f.read()

    if MARKER in src:
        print("patch_responses_404: already patched, skipping")
        return 0

    changed = []
    for anchor in ANCHORS:
        if anchor not in src:
            print(f"patch_responses_404: WARNING anchor not found: {anchor.strip()}")
            continue
        # Insert the guard as the very first statement of the handler body.
        src = src.replace(anchor, anchor + GUARD, 1)
        changed.append(anchor.split("(")[0].replace("async def ", ""))

    if not changed:
        print("patch_responses_404: NO CHANGES (no anchors matched)")
        return 1

    tmp = HTTP_SERVER_PATH + ".resp404"
    with open(tmp, "w") as f:
        f.write(src)
    try:
        py_compile.compile(tmp, doraise=True)
        os.rename(tmp, HTTP_SERVER_PATH)
        print(f"patch_responses_404: SUCCESS - guarded {', '.join(changed)}")
        return 0
    except py_compile.PyCompileError as e:
        os.unlink(tmp)
        print(f"patch_responses_404: FAILED - {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
