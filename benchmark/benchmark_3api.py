"""Benchmark /v1/chat/completions vs /v1/messages vs /v1/responses PD routing.

Tests all 3 API formats through the sgl-model-gateway PD router and compares:
  - latency (TTFT, end-to-end)
  - throughput (tokens/sec)
  - output quality (token count, response validity)

Usage:
  python benchmark_3api.py --base-url https://<httproute> --model glm-5.2 \
    --num-prompts 20 --concurrency 4
"""

import argparse
import asyncio
import json
import time
import sys
import httpx
from typing import Optional


# 10 diverse prompts for quality + latency comparison
DEFAULT_PROMPTS = [
    "Explain the difference between TCP and UDP in 3 sentences.",
    "Write a Python function to check if a string is a palindrome.",
    "What are the main causes of climate change? List 5.",
    "Translate 'Hello, how are you?' into French, Spanish, and Japanese.",
    "Write a haiku about autumn leaves.",
    "Explain quantum entanglement in simple terms.",
    "What is the time complexity of binary search? Why?",
    "List the planets in our solar system in order from the sun.",
    "Write a SQL query to find the top 10 customers by total order value.",
    "What is the CAP theorem in distributed systems?",
    "Describe the water cycle in 4 steps.",
    "What is the difference between a list and a tuple in Python?",
    "Write a regex to validate an email address.",
    "Explain the concept of recursion with a simple example.",
    "What are the benefits of microservices architecture?",
    "How does HTTPS encryption work? Briefly explain.",
    "Write a function to reverse a linked list in Python.",
    "What is the Halting Problem? Why is it undecidable?",
    "List 5 design patterns in object-oriented programming.",
    "Explain the difference between SQL and NoSQL databases.",
]


async def send_chat_completion(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_key: Optional[str] = None,
) -> dict:
    """Send /v1/chat/completions request."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    start = time.perf_counter()
    resp = await client.post(
        f"{base_url}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120.0,
    )
    elapsed = time.perf_counter() - start
    body = resp.json() if resp.status_code == 200 else {}
    output_text = ""
    choices = body.get("choices", [])
    if choices:
        output_text = choices[0].get("message", {}).get("content", "")
    usage = body.get("usage", {})
    return {
        "api": "chat",
        "status": resp.status_code,
        "latency": elapsed,
        "output_tokens": usage.get("completion_tokens", 0),
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_text": output_text,
        "error": body.get("error", {}) if resp.status_code != 200 else None,
    }


async def send_messages(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_key: Optional[str] = None,
) -> dict:
    """Send /v1/messages (Anthropic Messages API) request."""
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if api_key:
        headers["x-api-key"] = api_key
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    start = time.perf_counter()
    resp = await client.post(
        f"{base_url}/v1/messages",
        headers=headers,
        json=payload,
        timeout=120.0,
    )
    elapsed = time.perf_counter() - start
    body = resp.json() if resp.status_code == 200 else {}
    output_text = ""
    content = body.get("content", [])
    if content and isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                output_text += block.get("text", "")
    usage = body.get("usage", {})
    return {
        "api": "messages",
        "status": resp.status_code,
        "latency": elapsed,
        "output_tokens": usage.get("output_tokens", 0),
        "input_tokens": usage.get("input_tokens", 0),
        "output_text": output_text,
        "error": body.get("error", {}) if resp.status_code != 200 else None,
    }


async def send_responses(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_key: Optional[str] = None,
) -> dict:
    """Send /v1/responses (OpenAI Responses API) request."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "stream": False,
    }
    start = time.perf_counter()
    resp = await client.post(
        f"{base_url}/v1/responses",
        headers=headers,
        json=payload,
        timeout=120.0,
    )
    elapsed = time.perf_counter() - start
    body = resp.json() if resp.status_code == 200 else {}
    output_text = ""
    output = body.get("output", [])
    if output and isinstance(output, list):
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                for block in item.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        output_text += block.get("text", "")
    usage = body.get("usage", {})
    return {
        "api": "responses",
        "status": resp.status_code,
        "latency": elapsed,
        "output_tokens": usage.get("output_tokens", 0),
        "input_tokens": usage.get("input_tokens", 0),
        "output_text": output_text,
        "error": body.get("error", {}) if resp.status_code != 200 else None,
    }


async def run_benchmark(
    base_url: str,
    model: str,
    prompts: list,
    max_tokens: int,
    concurrency: int,
    api_key: Optional[str] = None,
) -> dict:
    """Run benchmark for all 3 API formats."""
    results = {"chat": [], "messages": [], "responses": []}
    sem = asyncio.Semaphore(concurrency)

    async def run_one(api_type: str, prompt: str, idx: int):
        async with sem:
            if api_type == "chat":
                return await send_chat_completion(
                    client, base_url, model, prompt, max_tokens, api_key
                )
            elif api_type == "messages":
                return await send_messages(
                    client, base_url, model, prompt, max_tokens, api_key
                )
            else:
                return await send_responses(
                    client, base_url, model, prompt, max_tokens, api_key
                )

    async with httpx.AsyncClient(verify=False) as client:
        for api_type in ["chat", "messages", "responses"]:
            print(f"\n{'='*60}")
            print(f"Testing /v1/{api_type} ...")
            print(f"{'='*60}")
            tasks = [run_one(api_type, p, i) for i, p in enumerate(prompts)]
            api_results = await asyncio.gather(*tasks)
            results[api_type] = api_results

            # Print per-API summary
            successes = [r for r in api_results if r["status"] == 200]
            failures = [r for r in api_results if r["status"] != 200]
            latencies = [r["latency"] for r in successes]
            output_tokens = [r["output_tokens"] for r in successes]

            print(f"  Success: {len(successes)}/{len(api_results)}")
            if failures:
                for f in failures[:3]:
                    print(f"  FAIL status={f['status']}: {f.get('error', '?')}")
            if latencies:
                avg_lat = sum(latencies) / len(latencies)
                min_lat = min(latencies)
                max_lat = max(latencies)
                avg_tok = sum(output_tokens) / len(output_tokens) if output_tokens else 0
                avg_tps = (
                    sum(t / l for t, l in zip(output_tokens, latencies) if l > 0)
                    / len(output_tokens)
                    if output_tokens
                    else 0
                )
                print(f"  Latency avg/min/max: {avg_lat:.3f}/{min_lat:.3f}/{max_lat:.3f}s")
                print(f"  Output tokens avg: {avg_tok:.1f}")
                print(f"  Throughput avg: {avg_tps:.1f} tok/s")

    return results


def print_comparison(results: dict):
    """Print side-by-side comparison table."""
    print(f"\n{'='*80}")
    print("COMPARISON: chat vs messages vs responses")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'chat':>15} {'messages':>15} {'responses':>15}")
    print(f"{'-'*70}")

    for metric in ["success_rate", "avg_latency", "min_latency", "max_latency",
                   "avg_output_tokens", "avg_throughput_tps"]:
        row = f"{metric:<25}"
        for api in ["chat", "messages", "responses"]:
            api_results = results[api]
            successes = [r for r in api_results if r["status"] == 200]
            if not successes:
                row += f"{'N/A':>15}"
                continue
            if metric == "success_rate":
                val = f"{len(successes)}/{len(api_results)}"
            elif metric == "avg_latency":
                val = f"{sum(r['latency'] for r in successes)/len(successes):.3f}s"
            elif metric == "min_latency":
                val = f"{min(r['latency'] for r in successes):.3f}s"
            elif metric == "max_latency":
                val = f"{max(r['latency'] for r in successes):.3f}s"
            elif metric == "avg_output_tokens":
                toks = [r["output_tokens"] for r in successes]
                val = f"{sum(toks)/len(toks):.1f}"
            elif metric == "avg_throughput_tps":
                tps = [r["output_tokens"] / r["latency"] for r in successes if r["latency"] > 0]
                val = f"{sum(tps)/len(tps):.1f}" if tps else "N/A"
            row += f"{val:>15}"
        print(row)

    # Quality check: compare first prompt output across 3 APIs
    print(f"\n{'='*80}")
    print("QUALITY SAMPLE (prompt 0)")
    print(f"{'='*80}")
    for api in ["chat", "messages", "responses"]:
        r = results[api][0]
        text = r["output_text"][:200] if r["output_text"] else "(empty)"
        status = "OK" if r["status"] == 200 else f"FAIL({r['status']})"
        print(f"\n[{api}] status={status} tokens={r['output_tokens']}")
        print(f"  {text}...")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark /v1/chat/completions vs /v1/messages vs /v1/responses"
    )
    parser.add_argument("--base-url", required=True, help="Base URL (e.g. https://xxx)")
    parser.add_argument("--model", default="glm-5.2", help="Model name")
    parser.add_argument("--num-prompts", type=int, default=20, help="Number of prompts")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max output tokens")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent requests per API")
    parser.add_argument("--api-key", default=None, help="API key")
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS[: args.num_prompts]
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Concurrency: {args.concurrency}")

    results = asyncio.run(
        run_benchmark(
            args.base_url,
            args.model,
            prompts,
            args.max_tokens,
            args.concurrency,
            args.api_key,
        )
    )

    print_comparison(results)

    # Save full results
    out_file = "/tmp/benchmark_3api_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {out_file}")


if __name__ == "__main__":
    main()
