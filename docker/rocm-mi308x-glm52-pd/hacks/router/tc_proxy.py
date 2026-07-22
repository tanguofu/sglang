#!/usr/bin/env python3
"""
tool_choice normalizing proxy for SGLang PD Router.

Sits in front of the Rust router and normalizes tool_choice from dict format
{"type": "auto"} to string format "auto" for all request bodies, so that
clients like grok that send dict-format tool_choice can use /v1/responses.

PROXY_PORT  (default 30011) - where this proxy listens (Service targetPort)
BACKEND_PORT (default 30012) - where the Rust router listens
"""

import asyncio
import gzip
import json
import logging
import os

from aiohttp import web, ClientSession, ClientTimeout

PROXY_PORT = int(os.environ.get("PROXY_PORT", "30011"))
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "30012"))
BACKEND_HOST = os.environ.get("BACKEND_HOST", "127.0.0.1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("tc_proxy")

HOP_BY_HOP = frozenset({
    "host", "content-length", "transfer-encoding", "connection",
    "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "upgrade",
})


def normalize_body(body_bytes: bytes, content_encoding: str = "") -> bytes:
    """Parse JSON body and normalize tool_choice dict to string.

    Also strips tool_choice when no tools are specified, since the OpenAI
    spec requires tools to be present when tool_choice is set. grok always
    sends tool_choice even without tools.

    Handles gzip-compressed request bodies.
    """
    if not body_bytes:
        return body_bytes

    # Decompress if needed
    raw_bytes = body_bytes
    if "gzip" in content_encoding.lower():
        try:
            raw_bytes = gzip.decompress(body_bytes)
        except Exception as e:
            log.warning("Failed to decompress gzip body: %s", e)
            return body_bytes

    stripped = raw_bytes.lstrip()
    if not stripped or stripped[0:1] not in (b"{", b"["):
        return body_bytes
    try:
        parsed = json.loads(raw_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        log.warning("JSON parse failed (len=%d, encoding=%s): %s", len(raw_bytes), content_encoding, e)
        return body_bytes
    if not isinstance(parsed, dict):
        return body_bytes

    changed = False
    tc = parsed.get("tool_choice")
    if tc is not None:
        log.info("Incoming tool_choice=%r tools_present=%s", tc, bool(parsed.get("tools")))
    if isinstance(tc, dict):
        type_val = tc.get("type")
        if type_val in ("auto", "required", "none"):
            parsed["tool_choice"] = type_val
            changed = True
            log.info("Normalized tool_choice: %s -> %s", tc, type_val)
        else:
            # Unknown dict format — strip it to avoid Rust deserialization error
            del parsed["tool_choice"]
            changed = True
            log.warning("Stripped unknown tool_choice format: %s", tc)

    tools = parsed.get("tools")
    if "tool_choice" in parsed and not tools:
        del parsed["tool_choice"]
        changed = True
        log.info("Stripped tool_choice (no tools specified)")

    if not changed:
        return body_bytes
    # Re-serialize and handle compression
    new_body = json.dumps(parsed).encode()
    if "gzip" in content_encoding.lower():
        return gzip.compress(new_body)
    return new_body


async def proxy_handler(request: web.Request) -> web.StreamResponse:
    """Forward request to backend, normalizing tool_choice in POST bodies."""
    target_url = f"http://{BACKEND_HOST}:{BACKEND_PORT}{request.path_qs}"

    body = await request.read()
    if request.method == "POST" and body:
        ce = request.headers.get("Content-Encoding", "")
        body = normalize_body(body, ce)

    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    if body:
        headers["Content-Length"] = str(len(body))

    session: ClientSession = request.app["client_session"]
    try:
        resp = await session.request(
            request.method,
            target_url,
            headers=headers,
            data=body if body else None,
            allow_redirects=False,
        )
    except Exception as e:
        log.error("Backend connection failed: %s", e)
        return web.Response(status=502, text=f"Backend unavailable: {e}")

    content_type = resp.headers.get("Content-Type", "")
    is_stream = "text/event-stream" in content_type

    if is_stream:
        proxy_resp = web.StreamResponse(
            status=resp.status,
            headers={
                k: v for k, v in resp.headers.items()
                if k.lower() not in HOP_BY_HOP
            },
        )
        proxy_resp.content_type = "text/event-stream"
        proxy_resp.headers["Cache-Control"] = "no-cache"
        await proxy_resp.prepare(request)
        try:
            async for chunk in resp.content.iter_any():
                await proxy_resp.write(chunk)
            await proxy_resp.write_eof()
        finally:
            resp.release()
        return proxy_resp
    else:
        resp_body = await resp.read()
        result = web.Response(
            status=resp.status,
            headers={
                k: v for k, v in resp.headers.items()
                if k.lower() not in HOP_BY_HOP
            },
            body=resp_body,
        )
        resp.release()
        return result


async def main():
    app = web.Application(client_max_size=100 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", proxy_handler)

    timeout = ClientTimeout(total=600, sock_read=600, sock_connect=30)
    app["client_session"] = ClientSession(
        timeout=timeout, auto_decompress=False
    )

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PROXY_PORT)
    await site.start()
    log.info(
        "tool_choice proxy listening on :%s -> %s:%s",
        PROXY_PORT, BACKEND_HOST, BACKEND_PORT,
    )
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await app["client_session"].close()


if __name__ == "__main__":
    asyncio.run(main())
