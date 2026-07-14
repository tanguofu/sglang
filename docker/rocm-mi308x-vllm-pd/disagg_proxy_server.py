#!/usr/bin/env python3
"""Disaggregated serving proxy for vLLM PD (Prefill-Decode).

Routes requests: client → prefill (max_tokens=1, store KV) → decode (stream output).
KV cache transfers from prefill to decode via LMCache NIXL/UCX RDMA.
"""
import argparse
import asyncio
import httpx
import json
import sys

async def handle_request(prefill_host, prefill_port, decode_host, decode_port, request_body, path, method, headers):
    """Forward request: prefill first (store KV), then decode (stream output)."""
    prefill_url = f"http://{prefill_host}:{prefill_port}{path}"
    decode_url = f"http://{decode_host}:{decode_port}{path}"

    async with httpx.AsyncClient(timeout=600.0) as client:
        if method == "POST" and "/chat/completions" in path:
            body = json.loads(request_body) if isinstance(request_body, bytes) else json.loads(request_body)

            # Step 1: Send to prefill (max_tokens=1, just store KV)
            prefill_body = dict(body)
            prefill_body["max_tokens"] = 1
            prefill_body["stream"] = False
            try:
                resp = await client.post(prefill_url, json=prefill_body, headers=headers)
                print(f"[proxy] prefill status={resp.status_code}", file=sys.stderr)
            except Exception as e:
                print(f"[proxy] prefill error: {e}", file=sys.stderr)

            # Step 2: Send to decode (full generation, stream if requested)
            decode_body = dict(body)
            try:
                resp = await client.post(decode_url, json=decode_body, headers=headers)
                return resp.status_code, resp.headers, resp.content
            except Exception as e:
                return 500, {}, json.dumps({"error": f"decode failed: {str(e)}"}).encode()
        else:
            # Non-chat requests: forward to decode directly
            try:
                resp = await client.request(method, decode_url, content=request_body, headers=headers)
                return resp.status_code, resp.headers, resp.content
            except Exception as e:
                return 500, {}, json.dumps({"error": str(e)}).encode()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--prefiller-host", required=True)
    parser.add_argument("--prefiller-port", type=int, default=8100)
    parser.add_argument("--decoder-host", required=True)
    parser.add_argument("--decoder-port", type=int, default=8200)
    args = parser.parse_args()

    from fastapi import FastAPI, Request
    from fastapi.responses import Response, StreamingResponse
    import uvicorn

    app = FastAPI()

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy(path: str, request: Request):
        body = await request.body()
        status, headers, content = await handle_request(
            args.prefiller_host, args.prefiller_port,
            args.decoder_host, args.decoder_port,
            body, f"/{path}", request.method, dict(request.headers)
        )
        return Response(content=content, status_code=status, headers=dict(headers))

    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
