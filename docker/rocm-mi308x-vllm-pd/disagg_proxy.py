#!/usr/bin/env python3
"""PD Proxy with /v1/responses support for Codex CLI.

Based on vLLM's disagg_proxy_demo.py + added /v1/responses route.
Routes: prefill (max_tokens=1, push KV) → decode (full, pull KV, stream).

Usage:
  python3 disagg_proxy.py --model glm-5.2 \
    --prefill 21.234.170.19:13000 \
    --decode 21.234.170.32:13000 \
    --port 9000
"""
import argparse
import itertools
import json
import logging
import os

import aiohttp
import requests
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class Proxy:
    def __init__(self, prefill: list[str], decode: list[str], model: str):
        self.prefill = prefill
        self.decode = decode
        self.prefill_cycler = itertools.cycle(prefill)
        self.decode_cycler = itertools.cycle(decode)
        self.model = model
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        self.router.post("/v1/chat/completions")(self.create_chat_completion)
        self.router.post("/v1/completions")(self.create_completion)
        self.router.post("/v1/responses")(self.create_response)
        self.router.post("/v1/messages")(self.create_message)
        self.router.get("/status")(self.get_status)

    async def _forward(self, url, data):
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url=url, json=data, headers=headers) as resp:
                if 200 <= resp.status < 300 or 400 <= resp.status < 500:
                    async for chunk in resp.content.iter_chunked(1024):
                        yield chunk
                else:
                    err = await resp.text()
                    raise HTTPException(status_code=resp.status, detail=err)

    async def _pd_route(self, path, request, prefill_key="max_tokens", prefill_val=1):
        """PD routing: prefill (max_tokens=1, push KV) → decode (full, pull KV, stream)."""
        # Step 1: prefill (store KV, push to decode via NIXL)
        kv_req = request.copy()
        kv_req[prefill_key] = prefill_val
        # Also set max_tokens for responses that use max_output_tokens
        if prefill_key == "max_output_tokens":
            kv_req["max_tokens"] = 1
        prefill_inst = next(self.prefill_cycler)
        logger.info(f"PD: prefill → {prefill_inst}{path}")
        try:
            async for _ in self._forward(f"http://{prefill_inst}{path}", kv_req):
                continue
        except HTTPException as e:
            logger.error(f"PD: prefill failed: {e.detail}")
            raise

        # Step 2: decode (pull KV, generate, stream)
        decode_inst = next(self.decode_cycler)
        logger.info(f"PD: decode → {decode_inst}{path}")
        generator = self._forward(f"http://{decode_inst}{path}", request)
        return StreamingResponse(generator)

    async def create_chat_completion(self, raw_request: Request):
        req = await raw_request.json()
        return await self._pd_route("/v1/chat/completions", req, "max_tokens", 1)

    async def create_completion(self, raw_request: Request):
        req = await raw_request.json()
        return await self._pd_route("/v1/completions", req, "max_tokens", 1)

    async def create_response(self, raw_request: Request):
        """Codex /v1/responses — PD route with max_output_tokens=1 for prefill."""
        req = await raw_request.json()
        return await self._pd_route("/v1/responses", req, "max_output_tokens", 1)

    async def create_message(self, raw_request: Request):
        """Anthropic /v1/messages — PD route with max_tokens=1 for prefill."""
        req = await raw_request.json()
        return await self._pd_route("/v1/messages", req, "max_tokens", 1)

    async def get_status(self):
        return {"prefill": self.prefill, "decode": self.decode, "model": self.model}


def parse_args():
    p = argparse.ArgumentParser("PD Proxy with /v1/responses support")
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--prefill", "-p", nargs="+", required=True, help="host:port list")
    p.add_argument("--decode", "-d", nargs="+", required=True, help="host:port list")
    p.add_argument("--port", type=int, default=9000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    proxy = Proxy(args.prefill, args.decode, args.model)
    app = FastAPI()
    app.include_router(proxy.router)
    config = uvicorn.Config(app, host="0.0.0.0", port=args.port, loop="uvloop")
    server = uvicorn.Server(config)
    logger.info(f"PD Proxy on :{args.port} | prefill={args.prefill} decode={args.decode}")
    server.run()
