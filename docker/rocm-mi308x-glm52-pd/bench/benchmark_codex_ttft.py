#!/usr/bin/env python3
"""Benchmark TTFT and TFOT for codex-like traffic on GLM-5.2 2tp8.

Codex traffic characteristics:
  - Large prompts (15K+ tokens: skills+tools+mcp+conversation)
  - Reasoning model (generates ildi blocks before final text)
  - Streaming (TTFT matters — codex shows reasoning live)
  - Concurrency (codex CLI may send parallel requests)

Metrics:
  - TTFT (Time To First Token): from request send to first reasoning delta
    (codex displays reasoning live, so first reasoning delta = first visible token)
  - TTFT_text: from request send to first output_text delta (non-reasoning)
  - TFOT (Time Per Output Token): total_time / total_output_tokens
    (includes reasoning + text tokens — total generation speed)
  - Decode throughput: output_tokens / (last_token - first_token)
  - P50/P90/P99 latency

Wire API: /v1/responses (OpenAI Responses API, used by codex)
Event types observed:
  - response.created / response.in_progress
  - response.output_item.added
  - response.reasoning_text.delta / done  (reasoning/thinking phase)
  - response.content_part.added / done
  - response.output_text.delta / done      (final text answer)
  - response.output_item.done
  - response.completed                     (carries usage stats)
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


BASE_URL = "https://glm52-2tp8.jmpti.woa.com/v1"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"


# Codex-like prompt templates (approx token counts)
PROMPTS = {
    # ~100 tokens — minimal baseline
    "tiny": "Say hello and explain what 2+2 equals in one sentence.",

    # ~500 tokens — short codex conversation
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

    # ~2K tokens — typical codex request (system prompt + tools + conversation)
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

    # ~8K tokens — large codex request (full skills+tools+mcp loaded)
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
    """Result of a single benchmark request."""
    request_id: int
    prompt_size: str
    prompt_tokens: int = 0
    output_tokens: int = 0          # total output tokens (reasoning + text)
    reasoning_tokens: int = 0
    text_tokens: int = 0
    ttft_ms: float = 0.0            # Time to first reasoning delta (codex TTFT)
    ttft_text_ms: float = 0.0       # Time to first output_text delta
    total_ms: float = 0.0           # Total request time
    tfot_ms: float = 0.0            # Total time / total output tokens
    decode_ms: float = 0.0          # Time from first token to last token
    decode_throughput_tps: float = 0.0  # output_tokens / decode_seconds
    error: Optional[str] = None


async def send_request(
    session: aiohttp.ClientSession,
    request_id: int,
    prompt_size: str,
    max_output_tokens: int,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    """Send a single streaming request and measure TTFT + decode timing."""
    async with semaphore:
        prompt = PROMPTS[prompt_size]
        payload = {
            "model": MODEL,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        result = RequestResult(request_id=request_id, prompt_size=prompt_size)
        url = f"{BASE_URL}/responses"
        start = time.monotonic()
        first_reasoning_time = None
        first_text_time = None
        last_token_time = None
        reasoning_delta_count = 0
        text_delta_count = 0
        reasoning_char_count = 0
        text_char_count = 0

        try:
            async with session.post(url, json=payload, headers=headers, ssl=False) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    result.error = f"HTTP {resp.status}: {body[:200]}"
                    result.total_ms = (time.monotonic() - start) * 1000
                    return result

                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = data.get("type", "")

                    # First reasoning delta = codex TTFT (user sees reasoning live)
                    if event_type == "response.reasoning_text.delta":
                        if first_reasoning_time is None:
                            first_reasoning_time = time.monotonic()
                            result.ttft_ms = (first_reasoning_time - start) * 1000
                        reasoning_delta_count += 1
                        reasoning_char_count += len(data.get("delta", ""))
                        last_token_time = time.monotonic()

                    # First output text delta = TTFT for visible answer
                    elif event_type == "response.output_text.delta":
                        if first_text_time is None:
                            first_text_time = time.monotonic()
                            result.ttft_text_ms = (first_text_time - start) * 1000
                        text_delta_count += 1
                        text_char_count += len(data.get("delta", ""))
                        last_token_time = time.monotonic()

                    # Final event carries authoritative usage stats
                    elif event_type == "response.completed":
                        usage = data.get("response", {}).get("usage", {})
                        result.prompt_tokens = usage.get("input_tokens", 0)
                        result.output_tokens = usage.get("output_tokens", 0)
                        # sglang doesn't populate reasoning_tokens; fall back to delta counts
                        srv_reasoning = usage.get(
                            "output_tokens_details", {}
                        ).get("reasoning_tokens", 0)
                        if srv_reasoning > 0:
                            result.reasoning_tokens = srv_reasoning
                            result.text_tokens = result.output_tokens - srv_reasoning
                        # else: leave placeholder; filled in finally block from delta counts

        except asyncio.TimeoutError:
            result.error = "Timeout"
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
        finally:
            end = time.monotonic()
            result.total_ms = (end - start) * 1000
            if first_reasoning_time and last_token_time:
                result.decode_ms = (last_token_time - first_reasoning_time) * 1000
            elif first_text_time and last_token_time:
                result.decode_ms = (last_token_time - first_text_time) * 1000

            # If usage stats missing, estimate from delta counts
            if result.output_tokens == 0:
                est_total = reasoning_delta_count + text_delta_count
                if est_total > 0:
                    result.output_tokens = est_total
                    result.reasoning_tokens = reasoning_delta_count
                    result.text_tokens = text_delta_count
            else:
                # output_tokens from server, but reasoning_tokens missing
                if result.reasoning_tokens == 0 and reasoning_delta_count > 0:
                    # Estimate reasoning fraction from delta counts
                    total_deltas = reasoning_delta_count + text_delta_count
                    if total_deltas > 0:
                        result.reasoning_tokens = int(
                            result.output_tokens * reasoning_delta_count / total_deltas
                        )
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


def print_stats(results: list[RequestResult], label: str) -> None:
    """Print summary statistics for a batch of results."""
    ok = [r for r in results if not r.error]
    err = [r for r in results if r.error]

    print(f"\n{'='*78}")
    print(f"  {label}")
    print(f"{'='*78}")
    print(f"  Total: {len(results)}  OK: {len(ok)}  Error: {len(err)}")

    if not ok:
        print("  No successful results.")
        for r in err:
            print(f"    [#{r.request_id}] ERROR: {r.error}")
        return

    def fmt(vals, name, unit="ms"):
        if not vals:
            print(f"  {name:32s}: no data")
            return
        print(
            f"  {name:32s}: "
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

    fmt(ttfts, "TTFT (first reasoning delta)")
    fmt(ttfts_text, "TTFT_text (first output delta)")
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


async def run_benchmark(
    concurrency: int,
    num_requests: int,
    prompt_size: str,
    max_output_tokens: int,
) -> list[RequestResult]:
    """Run the benchmark with given parameters."""
    print(f"\n# Running: concurrency={concurrency}, requests={num_requests}, "
          f"prompt={prompt_size}, max_output_tokens={max_output_tokens}")

    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 2, ssl=False)
    timeout = aiohttp.ClientTimeout(total=600)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            send_request(session, i, prompt_size, max_output_tokens, semaphore)
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)
    return list(results)


async def main():
    parser = argparse.ArgumentParser(description="Benchmark TTFT/TFOT for codex on GLM-5.2")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent requests (default: 1)")
    parser.add_argument("--requests", type=int, default=10,
                        help="Total number of requests per run (default: 10)")
    parser.add_argument("--prompt-size", choices=["tiny", "small", "medium", "large"],
                        default="medium", help="Prompt size (default: medium)")
    parser.add_argument("--max-output-tokens", type=int, default=512,
                        help="Max output tokens per request (default: 512)")
    parser.add_argument("--all-sizes", action="store_true",
                        help="Run all prompt sizes sequentially")
    args = parser.parse_args()

    print(f"# GLM-5.2 2tp8 Codex TTFT/TFOT Benchmark")
    print(f"# Endpoint: {BASE_URL}/responses")
    print(f"# Model: {MODEL}")
    print(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = defaultdict(list)

    if args.all_sizes:
        for size in ["tiny", "small", "medium", "large"]:
            results = await run_benchmark(
                args.concurrency, args.requests, size, args.max_output_tokens
            )
            all_results[size] = results
            print_stats(results, f"prompt={size} concurrency={args.concurrency}")
    else:
        results = await run_benchmark(
            args.concurrency, args.requests, args.prompt_size, args.max_output_tokens
        )
        all_results[args.prompt_size] = results
        print_stats(results, f"prompt={args.prompt_size} concurrency={args.concurrency}")

    # Summary across all runs
    print(f"\n{'='*78}")
    print("  SUMMARY (p50 / p90)")
    print(f"{'='*78}")
    print(f"  {'size':8s}  {'TTFT(ms)':>10s}  {'TTFT_t(ms)':>10s}  "
          f"{'Total(ms)':>10s}  {'TFOT(ms/tok)':>14s}  {'tps':>7s}  "
          f"{'prompt':>7s}  {'output':>7s}")
    for size, results in all_results.items():
        ok = [r for r in results if not r.error]
        if not ok:
            print(f"  {size:8s}  no successful results")
            continue
        ttfts = [r.ttft_ms for r in ok if r.ttft_ms > 0]
        ttfts_text = [r.ttft_text_ms for r in ok if r.ttft_text_ms > 0]
        totals = [r.total_ms for r in ok]
        tfots = [r.tfot_ms for r in ok if r.tfot_ms > 0]
        tps = [r.decode_throughput_tps for r in ok if r.decode_throughput_tps > 0]
        prompts = [r.prompt_tokens for r in ok if r.prompt_tokens > 0]
        outputs = [r.output_tokens for r in ok if r.output_tokens > 0]

        def m(v):
            return statistics.median(v) if v else 0
        def p90(v):
            return pct(v, 90) if v else 0

        print(
            f"  {size:8s}  "
            f"{m(ttfts):6.0f}/{p90(ttfts):4.0f}  "
            f"{m(ttfts_text):6.0f}/{p90(ttfts_text):4.0f}  "
            f"{m(totals):6.0f}/{p90(totals):4.0f}  "
            f"{m(tfots):8.1f}/{p90(tfots):5.1f}  "
            f"{m(tps):4.0f}/{p90(tps):3.0f}  "
            f"{m(prompts):7.0f}  {m(outputs):7.0f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
