#!/usr/bin/env python3
"""
Stress test to reproduce codex-like GPU hang on GLM-5.2 2tp8.

Goal: Systematically identify what triggers the c10::hip::memcpy_and_sync hang.
Test matrix:
  1. Prefill size: 64, 512, 1K, 2K, 5K, 10K, 15K tokens
  2. API path: /v1/chat/completions vs /v1/responses (streaming)
  3. Concurrency: 1, 2, 4
  4. Worker: direct to W1/W2 (bypass router) vs through router

Each test:
  - Send request, measure TTFT and total time
  - Check worker health before/after
  - If hang detected (>60s no response), record and move on
"""

import argparse
import asyncio
import json
import time
import sys
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx

# Worker endpoints (direct, bypass router)
W1_URL = "http://21.151.225.152:30000"
W2_URL = "http://21.151.225.172:30000"
ROUTER_URL = "http://21.234.170.19:30001"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"

# Use kubectl port-forward or direct access via router
# For direct worker access, need to be in cluster or use port-forward


@dataclass
class TestResult:
    test_name: str
    prefill_tokens: int
    api_path: str
    concurrency: int
    target: str
    ttft: Optional[float] = None
    total_time: Optional[float] = None
    output_tokens: int = 0
    status: str = "pending"  # pending, success, hang, error
    error: str = ""
    raw_response: str = ""


def make_prompt(target_tokens: int) -> str:
    """Generate a prompt with approximately target_tokens tokens."""
    # English text averages ~0.75 tokens per word
    # Use a repetitive structure to hit exact token counts
    base = "Tell me a story about "
    filler = "the quick brown fox jumps over the lazy dog. "
    # Each filler is ~10 tokens, need target_tokens - base_tokens
    base_tokens = 10
    needed = max(1, target_tokens - base_tokens)
    repeats = needed // 10 + 1
    prompt = base + (filler * repeats)
    # Add a clear task at the end
    prompt += f"\n\nSummarize the above in one sentence. (Target: ~{target_tokens} input tokens)"
    return prompt


async def test_chat_completions(
    client: httpx.AsyncClient,
    base_url: str,
    prefill_tokens: int,
    max_tokens: int = 100,
    timeout: float = 120.0,
) -> TestResult:
    """Test /v1/chat/completions with given prefill size."""
    prompt = make_prompt(prefill_tokens)
    payload = {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    result = TestResult(
        test_name=f"chat_completions_{prefill_tokens}t",
        prefill_tokens=prefill_tokens,
        api_path="/v1/chat/completions",
        concurrency=1,
        target=base_url,
    )

    start = time.time()
    try:
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        ttft = time.time() - start
        result.ttft = ttft
        result.total_time = ttft

        if resp.status_code == 200:
            data = resp.json()
            usage = data.get("usage", {})
            result.output_tokens = usage.get("completion_tokens", 0)
            result.status = "success"
            result.raw_response = json.dumps(usage)
        else:
            result.status = "error"
            result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
    except httpx.TimeoutException:
        result.status = "hang"
        result.error = f"Timeout after {timeout}s"
        result.total_time = time.time() - start
    except Exception as e:
        result.status = "error"
        result.error = f"{type(e).__name__}: {e}"
        result.total_time = time.time() - start

    return result


async def test_responses_streaming(
    client: httpx.AsyncClient,
    base_url: str,
    prefill_tokens: int,
    max_tokens: int = 100,
    timeout: float = 120.0,
) -> TestResult:
    """Test /v1/responses with streaming (codex uses this)."""
    prompt = make_prompt(prefill_tokens)
    payload = {
        "model": "glm-5.2",
        "input": prompt,
        "max_output_tokens": max_tokens,
        "stream": True,
        "reasoning": {"effort": "low"},
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    result = TestResult(
        test_name=f"responses_stream_{prefill_tokens}t",
        prefill_tokens=prefill_tokens,
        api_path="/v1/responses",
        concurrency=1,
        target=base_url,
    )

    start = time.time()
    first_token_time = None
    output_tokens = 0
    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/responses",
            json=payload,
            headers=headers,
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                result.status = "error"
                result.error = f"HTTP {resp.status_code}"
                result.total_time = time.time() - start
                return result

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    evt_type = data.get("type", "")
                    if first_token_time is None and evt_type in (
                        "response.output_text.delta",
                        "response.reasoning_text.delta",
                    ):
                        first_token_time = time.time()
                    if evt_type == "response.completed":
                        usage = data.get("response", {}).get("usage", {})
                        output_tokens = usage.get("output_tokens", 0)
                except json.JSONDecodeError:
                    continue

        result.ttft = (first_token_time - start) if first_token_time else None
        result.total_time = time.time() - start
        result.output_tokens = output_tokens
        result.status = "success" if output_tokens > 0 else "error"
        if output_tokens == 0:
            result.error = "No output tokens received"
    except httpx.TimeoutException:
        result.status = "hang"
        result.error = f"Timeout after {timeout}s (first_token={first_token_time})"
        result.total_time = time.time() - start
    except Exception as e:
        result.status = "error"
        result.error = f"{type(e).__name__}: {e}"
        result.total_time = time.time() - start

    return result


async def check_worker_health(client: httpx.AsyncClient, worker_url: str) -> bool:
    """Quick health check."""
    try:
        resp = await client.get(
            f"{worker_url}/health",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10.0,
        )
        return resp.status_code == 200
    except Exception:
        return False


async def run_test_suite(base_url: str, label: str, prefill_sizes: list, api_paths: list):
    """Run a suite of tests."""
    print(f"\n{'='*60}")
    print(f"Test suite: {label}")
    print(f"Target: {base_url}")
    print(f"Prefill sizes: {prefill_sizes}")
    print(f"API paths: {api_paths}")
    print(f"{'='*60}\n")

    results = []
    async with httpx.AsyncClient() as client:
        for api_path in api_paths:
            for prefill in prefill_sizes:
                # Pre-test health check
                healthy_before = await check_worker_health(client, base_url)
                if not healthy_before:
                    print(f"  [SKIP] {api_path} {prefill}t - worker unhealthy before test")
                    continue

                print(f"  [{time.strftime('%H:%M:%S')}] Testing {api_path} {prefill}t...", end="", flush=True)
                if api_path == "/v1/chat/completions":
                    result = await test_chat_completions(client, base_url, prefill)
                else:
                    result = await test_responses_streaming(client, base_url, prefill)

                # Post-test health check
                await asyncio.sleep(2)
                healthy_after = await check_worker_health(client, base_url)

                if not healthy_after and result.status == "success":
                    result.status = "hang_after"
                    result.error = "Worker unhealthy after successful request"

                results.append(result)

                # Print result
                if result.status == "success":
                    ttft = f"{result.ttft:.2f}s" if result.ttft else "N/A"
                    total = f"{result.total_time:.2f}s"
                    print(f" OK ttft={ttft} total={total} out={result.output_tokens}t")
                elif result.status == "hang":
                    print(f" HANG after {result.total_time:.1f}s")
                    print(f"    -> Worker likely dead, stopping suite")
                    break
                elif result.status == "hang_after":
                    print(f" HANG_AFTER (request OK but worker died)")
                    print(f"    -> Stopping suite")
                    break
                else:
                    print(f" ERROR: {result.error}")

                # Wait between tests
                await asyncio.sleep(5)

    return results


async def main():
    parser = argparse.ArgumentParser(description="GLM-5.2 stress test")
    parser.add_argument(
        "--target",
        choices=["w1", "w2", "router"],
        default="w2",
        help="Target endpoint",
    )
    parser.add_argument(
        "--prefill-sizes",
        type=str,
        default="64,512,1024,2048,5120,10240,15360",
        help="Comma-separated prefill sizes",
    )
    parser.add_argument(
        "--api-paths",
        type=str,
        default="chat,responses",
        help="Comma-separated API paths (chat, responses)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds",
    )
    args = parser.parse_args()

    # Select target
    if args.target == "w1":
        base_url = W1_URL
    elif args.target == "w2":
        base_url = W2_URL
    else:
        base_url = ROUTER_URL

    prefill_sizes = [int(x) for x in args.prefill_sizes.split(",")]
    api_paths = []
    for p in args.api_paths.split(","):
        if p == "chat":
            api_paths.append("/v1/chat/completions")
        elif p == "responses":
            api_paths.append("/v1/responses")

    # Override global timeout
    global DEFAULT_TIMEOUT
    DEFAULT_TIMEOUT = args.timeout

    results = await run_test_suite(base_url, args.target, prefill_sizes, api_paths)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status_icon = {"success": "OK", "hang": "HANG", "error": "ERR", "hang_after": "HANG_AFTER"}.get(r.status, "?")
        ttft = f"{r.ttft:.2f}s" if r.ttft else "N/A"
        total = f"{r.total_time:.2f}s" if r.total_time else "N/A"
        print(f"  [{status_icon:10}] {r.test_name:30} ttft={ttft:8} total={total:8} out={r.output_tokens}t")
        if r.error:
            print(f"             error: {r.error}")

    # Return non-zero if any hang detected
    hangs = [r for r in results if r.status in ("hang", "hang_after")]
    sys.exit(1 if hangs else 0)


if __name__ == "__main__":
    asyncio.run(main())
