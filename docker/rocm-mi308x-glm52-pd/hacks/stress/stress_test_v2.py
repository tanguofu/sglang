#!/usr/bin/env python3
"""
Focused stress test to reproduce the sglang event-loop block.

Strategy: Test via router with increasing prefill sizes, monitor for:
  - TTFT > 30s (event loop blocked)
  - request timeout
  - worker entering zombie state

Usage from local machine (router is reachable via kubectl port-forward or directly):
  python3 /tmp/stress_test_v2.py --base-url http://21.234.170.19:30001 --target W2
"""

import argparse
import asyncio
import json
import time
import sys
from dataclasses import dataclass
from typing import Optional

import httpx

API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"


@dataclass
class Result:
    label: str
    prefill: int
    ttft: Optional[float] = None
    total: Optional[float] = None
    out_tokens: int = 0
    status: str = "pending"
    error: str = ""


def make_prompt(target_tokens: int) -> str:
    """Generate a prompt with approximately target_tokens tokens."""
    base = "Tell me a story about "
    filler = "the quick brown fox jumps over the lazy dog. "
    base_tokens = 10
    needed = max(1, target_tokens - base_tokens)
    repeats = needed // 10 + 1
    prompt = base + (filler * repeats)
    prompt += f"\n\nSummarize the above in one sentence. (Target: ~{target_tokens} input tokens)"
    return prompt


async def run_test(
    client: httpx.AsyncClient,
    base_url: str,
    prefill: int,
    timeout: float = 180.0,
    max_tokens: int = 50,
) -> Result:
    label = f"prefill={prefill}"
    r = Result(label=label, prefill=prefill)
    prompt = make_prompt(prefill)
    payload = {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    t0 = time.time()
    first_chunk_t = None
    out_tokens = 0
    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                r.status = "error"
                r.error = f"HTTP {resp.status_code}: {body[:200].decode('utf-8', errors='replace')}"
                r.total = time.time() - t0
                return r
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if first_chunk_t is None:
                    first_chunk_t = time.time()
                    r.ttft = first_chunk_t - t0
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "") or delta.get("reasoning_content", "")
                    if content:
                        out_tokens += 1
        r.total = time.time() - t0
        r.out_tokens = out_tokens
        r.status = "success" if out_tokens > 0 else "empty"
    except httpx.TimeoutException as e:
        r.status = "hang"
        r.error = f"timeout after {timeout}s"
        r.total = time.time() - t0
        if first_chunk_t:
            r.ttft = first_chunk_t - t0
    except Exception as e:
        r.status = "error"
        r.error = f"{type(e).__name__}: {e}"
        r.total = time.time() - t0
        if first_chunk_t:
            r.ttft = first_chunk_t - t0
    return r


async def check_health(client: httpx.AsyncClient, base_url: str) -> tuple[str, float]:
    """Returns (status, response_time)."""
    t0 = time.time()
    try:
        r = await client.get(f"{base_url}/health", timeout=10.0)
        return f"HTTP {r.status_code}", time.time() - t0
    except Exception as e:
        return f"error: {type(e).__name__}", time.time() - t0


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://21.234.170.19:30001")
    parser.add_argument(
        "--prefill-sizes",
        type=str,
        default="64,512,1024,2048,4096",
        help="Comma-separated prefill token sizes",
    )
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--sleep-between", type=float, default=5.0)
    args = parser.parse_args()

    sizes = [int(x) for x in args.prefill_sizes.split(",")]
    print(f"=== Stress test against {args.base_url} ===")
    print(f"Prefill sizes: {sizes}, max_tokens={args.max_tokens}, timeout={args.timeout}s")
    print()

    async with httpx.AsyncClient() as client:
        # Initial health check
        h_status, h_time = await check_health(client, args.base_url)
        print(f"[health] {h_status} ({h_time:.3f}s)")
        print()

        results = []
        for size in sizes:
            print(f"[test] prefill={size} ...", end="", flush=True)
            r = await run_test(
                client,
                args.base_url,
                size,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
            )
            results.append(r)
            if r.status == "success":
                print(
                    f" OK ttft={r.ttft:.2f}s total={r.total:.2f}s out={r.out_tokens}"
                )
            elif r.status == "hang":
                print(
                    f" HANG ttft={r.ttft}s total={r.total:.2f}s err={r.error}"
                )
            else:
                print(f" {r.status} total={r.total:.2f}s err={r.error[:100]}")

            # Check health after each test
            h_status, h_time = await check_health(client, args.base_url)
            print(f"   [health] {h_status} ({h_time:.3f}s)")

            if r.status == "hang":
                print("   !!! HANG detected — stopping test")
                break

            await asyncio.sleep(args.sleep_between)

        print()
        print("=== Summary ===")
        for r in results:
            ttft = f"{r.ttft:.2f}s" if r.ttft is not None else "N/A"
            total = f"{r.total:.2f}s" if r.total is not None else "N/A"
            print(
                f"  {r.label:25s} status={r.status:8s} ttft={ttft:8s} total={total:8s} out={r.out_tokens}"
            )


if __name__ == "__main__":
    asyncio.run(main())
