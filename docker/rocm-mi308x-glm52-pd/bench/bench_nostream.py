#!/usr/bin/env python3
"""Non-streaming benchmark for Codex /v1/responses on sglang-1p1d PD deployment."""

import asyncio
import json
import ssl
import time
import aiohttp

BASE_URL = "https://glm52-pd-1p1d.jmpti.woa.com"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"

PROMPTS = {
    "tiny": "Say hello and explain what 2+2 equals in one sentence.",
    "small": (
        "You are a coding assistant. Help me debug this Python function:\n\n"
        "```python\ndef calculate_average(numbers):\n"
        "    total = 0\n    for n in numbers:\n        total += n\n"
        "    return total / len(numbers)\n```\n\n"
        "It crashes on empty input. Fix it and explain the fix. "
        "Also handle the case where numbers is None. "
        "Show me the corrected function with type hints and a docstring. "
        "Add unit tests using pytest. Explain edge cases I should test. " * 3
    ),
}


async def bench_one(session, prompt_size, max_tokens):
    """Send one non-streaming /v1/responses request and measure latency."""
    payload = {
        "model": MODEL,
        "input": PROMPTS[prompt_size],
        "max_output_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    url = f"{BASE_URL}/v1/responses"

    start = time.monotonic()
    try:
        async with session.post(url, json=payload, headers=headers, ssl=False) as resp:
            elapsed = (time.monotonic() - start) * 1000
            if resp.status != 200:
                body = await resp.text()
                return {"status": resp.status, "error": body[:200], "ms": elapsed}
            data = await resp.json()
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            reasoning_tokens = usage.get("reasoning_tokens", 0)
            tps = (completion_tokens / (elapsed / 1000)) if elapsed > 0 and completion_tokens > 0 else 0
            return {
                "status": 200,
                "ms": round(elapsed, 1),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "tps": round(tps, 1),
            }
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return {"status": 0, "error": str(e), "ms": round(elapsed, 1)}


async def main():
    print(f"# sglang-1p1d PD Non-Streaming Benchmark")
    print(f"# Endpoint: {BASE_URL}")
    print(f"# Model: {MODEL}")
    print(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    connector = aiohttp.TCPConnector(ssl=ssl_ctx, limit=4)

    async with aiohttp.ClientSession(connector=connector) as session:
        for prompt_size in ["tiny", "small"]:
            for max_tokens in [64, 256]:
                results = []
                for i in range(3):
                    r = await bench_one(session, prompt_size, max_tokens)
                    results.append(r)
                    status = r.get("status", 0)
                    ms = r.get("ms", 0)
                    tps = r.get("tps", 0)
                    pt = r.get("prompt_tokens", 0)
                    ct = r.get("completion_tokens", 0)
                    if status == 200:
                        print(f"  {prompt_size:6s} max={max_tokens:4d}  req#{i}: {ms:8.1f}ms  "
                              f"prompt={pt:4d}  completion={ct:4d}  {tps:6.1f} tok/s")
                    else:
                        print(f"  {prompt_size:6s} max={max_tokens:4d}  req#{i}: ERROR status={status} {r.get('error','')[:100]}")

                ok = [r for r in results if r.get("status") == 200]
                if ok:
                    times = [r["ms"] for r in ok]
                    tps_list = [r["tps"] for r in ok]
                    avg_ms = sum(times) / len(times)
                    avg_tps = sum(tps_list) / len(tps_list)
                    print(f"  {prompt_size:6s} max={max_tokens:4d}  AVG: {avg_ms:8.1f}ms  {avg_tps:6.1f} tok/s  ({len(ok)}/{len(results)} ok)")
                print()


asyncio.run(main())
