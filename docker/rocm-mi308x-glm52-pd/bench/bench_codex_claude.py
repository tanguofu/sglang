#!/usr/bin/env python3
"""Benchmark TTFT and TFOT for Codex /v1/responses and Claude /v1/messages on sglang-1p1d PD deployment.

Metrics per request:
  - TTFT (Time To First Token): first reasoning/thinking delta (live-displayed token)
  - TTFT_text: first text/output_text delta (final answer)
  - TFOT (Time Per Output Token): total_ms / total_output_tokens
  - Decode throughput (tok/s)
  - p50 / p90 / p99 latency
"""

import argparse
import asyncio
import json
import ssl
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import aiohttp


BASE_URL = "https://glm52-pd-1p1d.jmpti.woa.com"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"


PROMPTS = {
    "tiny": "Say hello and explain what 2+2 equals in one sentence.",
    "small": (
        "You are a coding assistant. Help me debug this Python function:\n\n"
        "```python\n"
        "def calculate_average(numbers):\n"
        "    total = 0\n"
        "    for n in numbers:\n"
        "        total += n\n"
        "    return total / len(numbers)\n"
        "```\n\n"
        "It crashes on empty input. Fix it and explain the fix. "
        "Also handle the case where numbers is None. "
        "Show me the corrected function with type hints and a docstring. "
        "Add unit tests using pytest. "
        "Explain edge cases I should test. " * 3
    ),
    "medium": (
        "You are an expert software engineer. Here is my codebase context:\n\n"
        "## Project: ti-cloud-teamai\n"
        "## Tech stack: Python, Go, Kubernetes, Helm\n\n"
        "### Files:\n"
        + "\n".join([
            f"#### `{f}`\n```python\n# ... (file contents elided for brevity)\n"
            f"def function_{i}():\n"
            f"    pass\n```\n"
            for i, f in enumerate([
                "src/main.py", "src/utils.py", "src/api.py", "src/models.py",
                "src/config.py", "src/router.py", "src/cache.py", "src/auth.py",
                "src/db.py", "src/logger.py", "src/metrics.py", "src/health.py",
            ])
        ])
        + "\n### Question:\n"
        + "Review my codebase for common issues. Focus on:\n"
        + "1. Error handling\n2. Type safety\n3. Performance\n4. Security\n"
        + "5. Test coverage\n6. Documentation\n7. Dependency management\n"
        + "Give me a prioritized list of improvements with code examples."
    ),
    "large": (
        "You are a coding assistant with access to the following tools:\n\n"
        + "\n".join([
            f"## Tool {i}: tool_{i}\n"
            f"Description: Performs operation {i} on the input.\n"
            f"Parameters:\n"
            f"  - input (string, required): The input to process\n"
            f"  - options (object, optional): Additional options\n"
            f"  - timeout (integer, optional): Timeout in seconds, default 30\n"
            f"Returns: {{\n"
            f"  \"result\": string,\n"
            f"  \"metadata\": {{\n"
            f"    \"duration_ms\": integer,\n"
            f"    \"status\": \"success\" | \"error\"\n"
            f"  }}\n"
            f"}}\n"
            f"Example:\n"
            f"  tool_{i}(input=\"hello\", options={{\"verbose\": true}})\n"
            for i in range(40)
        ])
        + "\n## Conversation history:\n"
        + "\n".join([
            f"User: Please help me with task {i}.\n"
            f"Assistant: I'll help you with task {i}. Let me analyze the requirements "
            f"and provide a solution. First, I need to understand the context. "
            f"The task involves several components that need to work together. "
            f"Let me break it down step by step. "
            f"[reasoning content for task {i}]\n"
            for i in range(20)
        ])
        + "\n## Current request:\n"
        + "Given the above tools and conversation, help me implement a new feature "
        + "that integrates tools 5, 12, and 23. Provide a complete implementation "
        + "with error handling, logging, and tests. Explain your design decisions."
    ),
}


@dataclass
class RequestResult:
    request_id: int
    api: str            # "codex" or "claude"
    prompt_size: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    text_tokens: int = 0
    ttft_ms: float = 0.0          # first reasoning/thinking delta
    ttft_text_ms: float = 0.0     # first text/output_text delta
    total_ms: float = 0.0
    tfot_ms: float = 0.0
    decode_ms: float = 0.0
    decode_throughput_tps: float = 0.0
    error: Optional[str] = None


async def send_codex_request(session, request_id, prompt_size, max_output_tokens, semaphore) -> RequestResult:
    """Codex /v1/responses streaming benchmark."""
    async with semaphore:
        prompt = PROMPTS[prompt_size]
        payload = {
            "model": MODEL,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        result = RequestResult(request_id=request_id, api="codex", prompt_size=prompt_size)
        url = f"{BASE_URL}/v1/responses"
        start = time.monotonic()
        first_reasoning = first_text = last_token = None
        reasoning_deltas = text_deltas = 0

        try:
            async with session.post(url, json=payload, headers=headers, ssl=False) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    result.error = f"HTTP {resp.status}: {body[:200]}"
                    result.total_ms = (time.monotonic() - start) * 1000
                    return result

                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    ev = data.get("type", "")
                    if ev == "response.reasoning_text.delta":
                        if first_reasoning is None:
                            first_reasoning = time.monotonic()
                            result.ttft_ms = (first_reasoning - start) * 1000
                        reasoning_deltas += 1
                        last_token = time.monotonic()
                    elif ev == "response.output_text.delta":
                        if first_text is None:
                            first_text = time.monotonic()
                            result.ttft_text_ms = (first_text - start) * 1000
                        text_deltas += 1
                        last_token = time.monotonic()
                    elif ev == "response.completed":
                        usage = data.get("response", {}).get("usage", {})
                        result.prompt_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
                        result.output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
                        srv_reasoning = usage.get("output_tokens_details", {}).get("reasoning_tokens", 0) or usage.get("reasoning_tokens", 0)
                        if srv_reasoning > 0:
                            result.reasoning_tokens = srv_reasoning
                            result.text_tokens = max(0, result.output_tokens - srv_reasoning)
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
        finally:
            end = time.monotonic()
            result.total_ms = (end - start) * 1000
            if first_reasoning and last_token:
                result.decode_ms = (last_token - first_reasoning) * 1000
            elif first_text and last_token:
                result.decode_ms = (last_token - first_text) * 1000
            # Estimate token counts from delta events if usage missing
            if result.output_tokens == 0:
                est = reasoning_deltas + text_deltas
                if est > 0:
                    result.output_tokens = est
                    result.reasoning_tokens = reasoning_deltas
                    result.text_tokens = text_deltas
            else:
                if result.reasoning_tokens == 0 and reasoning_deltas > 0:
                    total_d = reasoning_deltas + text_deltas
                    if total_d > 0:
                        result.reasoning_tokens = int(result.output_tokens * reasoning_deltas / total_d)
                        result.text_tokens = result.output_tokens - result.reasoning_tokens
            if result.output_tokens > 0:
                result.tfot_ms = result.total_ms / result.output_tokens
            if result.decode_ms > 0 and result.output_tokens > 0:
                result.decode_throughput_tps = result.output_tokens / (result.decode_ms / 1000)
        return result


async def send_claude_request(session, request_id, prompt_size, max_tokens, semaphore) -> RequestResult:
    """Claude /v1/messages streaming benchmark."""
    async with semaphore:
        prompt = PROMPTS[prompt_size]
        payload = {
            "model": MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        result = RequestResult(request_id=request_id, api="claude", prompt_size=prompt_size)
        url = f"{BASE_URL}/v1/messages"
        start = time.monotonic()
        first_thinking = first_text = last_token = None
        thinking_deltas = text_deltas = 0

        try:
            async with session.post(url, json=payload, headers=headers, ssl=False) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    result.error = f"HTTP {resp.status}: {body[:200]}"
                    result.total_ms = (time.monotonic() - start) * 1000
                    return result

                current_block_type = None
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    ev = data.get("type", "")
                    if ev == "content_block_start":
                        block = data.get("content_block", {})
                        current_block_type = block.get("type")
                    elif ev == "content_block_delta":
                        delta = data.get("delta", {})
                        delta_type = delta.get("type")
                        if delta_type == "thinking_delta":
                            if first_thinking is None:
                                first_thinking = time.monotonic()
                                result.ttft_ms = (first_thinking - start) * 1000
                            thinking_deltas += 1
                            last_token = time.monotonic()
                        elif delta_type == "text_delta":
                            if first_text is None:
                                first_text = time.monotonic()
                                result.ttft_text_ms = (first_text - start) * 1000
                            text_deltas += 1
                            last_token = time.monotonic()
                    elif ev == "message_start":
                        msg = data.get("message", {})
                        u = msg.get("usage", {})
                        if u.get("input_tokens"):
                            result.prompt_tokens = u["input_tokens"]
                    elif ev == "message_delta":
                        # message_delta carries usage with output_tokens at end
                        u = data.get("usage", {})
                        if u.get("output_tokens"):
                            result.output_tokens = u["output_tokens"]
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
        finally:
            end = time.monotonic()
            result.total_ms = (end - start) * 1000
            if first_thinking and last_token:
                result.decode_ms = (last_token - first_thinking) * 1000
            elif first_text and last_token:
                result.decode_ms = (last_token - first_text) * 1000
            if result.output_tokens == 0:
                est = thinking_deltas + text_deltas
                if est > 0:
                    result.output_tokens = est
                    result.reasoning_tokens = thinking_deltas
                    result.text_tokens = text_deltas
            else:
                if result.reasoning_tokens == 0 and thinking_deltas > 0:
                    total_d = thinking_deltas + text_deltas
                    if total_d > 0:
                        result.reasoning_tokens = int(result.output_tokens * thinking_deltas / total_d)
                        result.text_tokens = result.output_tokens - result.reasoning_tokens
            if result.output_tokens > 0:
                result.tfot_ms = result.total_ms / result.output_tokens
            if result.decode_ms > 0 and result.output_tokens > 0:
                result.decode_throughput_tps = result.output_tokens / (result.decode_ms / 1000)
        return result


def pct(vals, p):
    if not vals:
        return 0
    s = sorted(vals)
    idx = max(0, min(len(s) - 1, int(len(s) * p / 100)))
    return s[idx]


def print_stats(results, label):
    ok = [r for r in results if not r.error]
    err = [r for r in results if r.error]
    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"{'='*90}")
    print(f"  Total: {len(results)}  OK: {len(ok)}  Error: {len(err)}")
    if not ok:
        for r in err:
            print(f"    [#{r.request_id}] ERROR: {r.error}")
        return

    def fmt(vals, name, unit="ms"):
        if not vals:
            print(f"  {name:34s}: no data")
            return
        print(
            f"  {name:34s}: "
            f"min={min(vals):8.1f}  "
            f"p50={pct(vals,50):8.1f}  "
            f"p90={pct(vals,90):8.1f}  "
            f"p99={pct(vals,99):8.1f}  "
            f"max={max(vals):8.1f}  "
            f"mean={statistics.mean(vals):8.1f} {unit}"
        )

    ttfts = [r.ttft_ms for r in ok if r.ttft_ms > 0]
    ttfts_text = [r.ttft_text_ms for r in ok if r.ttft_text_ms > 0]
    totals = [r.total_ms for r in ok]
    tfots = [r.tfot_ms for r in ok if r.tfot_ms > 0]
    decodes = [r.decode_ms for r in ok if r.decode_ms > 0]
    tps = [r.decode_throughput_tps for r in ok if r.decode_throughput_tps > 0]
    prompts = [r.prompt_tokens for r in ok if r.prompt_tokens > 0]
    outputs = [r.output_tokens for r in ok if r.output_tokens > 0]
    reasoning = [r.reasoning_tokens for r in ok if r.reasoning_tokens > 0]

    fmt(ttfts, "TTFT (first reasoning/thinking)")
    fmt(ttfts_text, "TTFT_text (first text delta)")
    fmt(decodes, "Decode phase (first->last)")
    fmt(totals, "Total request time")
    fmt(tfots, "TFOT (total/output_tokens)", "ms/tok")
    fmt(tps, "Decode throughput", "tok/s")
    fmt(prompts, "Prompt tokens", "tok")
    fmt(outputs, "Output tokens (reasoning+text)", "tok")
    fmt(reasoning, "Reasoning tokens", "tok")

    if err:
        print(f"\n  Errors:")
        for r in err:
            print(f"    [#{r.request_id}] {r.error}")


async def run_benchmark(api, concurrency, num_requests, prompt_size, max_tokens):
    print(f"\n# Running: api={api}, concurrency={concurrency}, requests={num_requests}, "
          f"prompt={prompt_size}, max_tokens={max_tokens}")
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 2, ssl=False)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        if api == "codex":
            tasks = [send_codex_request(session, i, prompt_size, max_tokens, semaphore) for i in range(num_requests)]
        else:
            tasks = [send_claude_request(session, i, prompt_size, max_tokens, semaphore) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
    return list(results)


def summary_table(all_results):
    print(f"\n{'='*90}")
    print("  SUMMARY (p50 / p90)")
    print(f"{'='*90}")
    print(f"  {'api':8s}  {'size':8s}  {'TTFT(ms)':>12s}  {'TTFT_t(ms)':>12s}  "
          f"{'Total(ms)':>12s}  {'TFOT(ms/tok)':>14s}  {'tps':>9s}  "
          f"{'prompt':>7s}  {'output':>7s}")
    for (api, size), results in all_results.items():
        ok = [r for r in results if not r.error]
        if not ok:
            print(f"  {api:8s}  {size:8s}  no successful results")
            continue
        ttfts = [r.ttft_ms for r in ok if r.ttft_ms > 0]
        ttfts_text = [r.ttft_text_ms for r in ok if r.ttft_text_ms > 0]
        totals = [r.total_ms for r in ok]
        tfots = [r.tfot_ms for r in ok if r.tfot_ms > 0]
        tps = [r.decode_throughput_tps for r in ok if r.decode_throughput_tps > 0]
        prompts = [r.prompt_tokens for r in ok if r.prompt_tokens > 0]
        outputs = [r.output_tokens for r in ok if r.output_tokens > 0]

        def m(v): return statistics.median(v) if v else 0
        def p90(v): return pct(v, 90) if v else 0

        print(
            f"  {api:8s}  {size:8s}  "
            f"{m(ttfts):5.0f}/{p90(ttfts):5.0f}  "
            f"{m(ttfts_text):5.0f}/{p90(ttfts_text):5.0f}  "
            f"{m(totals):5.0f}/{p90(totals):5.0f}  "
            f"{m(tfots):6.1f}/{p90(tfots):6.1f}  "
            f"{m(tps):4.0f}/{p90(tps):4.0f}  "
            f"{m(prompts):7.0f}  {m(outputs):7.0f}"
        )


async def main():
    parser = argparse.ArgumentParser(description="Benchmark TTFT/TFOT for Codex and Claude APIs on sglang-1p1d")
    parser.add_argument("--api", choices=["codex", "claude", "both"], default="both")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--prompt-size", choices=["tiny", "small", "medium", "large"], default="small")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--all-sizes", action="store_true")
    args = parser.parse_args()

    print(f"# sglang-1p1d PD Router TTFT/TFOT Benchmark")
    print(f"# Endpoint: {BASE_URL}")
    print(f"# APIs: {args.api}")
    print(f"# Model: {MODEL}")
    print(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = defaultdict(list)
    apis = ["codex", "claude"] if args.api == "both" else [args.api]
    sizes = ["tiny", "small", "medium", "large"] if args.all_sizes else [args.prompt_size]

    for api in apis:
        for size in sizes:
            results = await run_benchmark(api, args.concurrency, args.requests, size, args.max_tokens)
            all_results[(api, size)] = results
            print_stats(results, f"api={api} prompt={size} concurrency={args.concurrency}")

    summary_table(all_results)


if __name__ == "__main__":
    asyncio.run(main())
