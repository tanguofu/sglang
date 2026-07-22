#!/usr/bin/env python3
"""Benchmark Anthropic /v1/messages vs OpenAI /v1/responses on glm52-1tp8."""
import json
import time
import subprocess
import sys
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
BASE = "https://glm52-1tp8.jmpti.woa.com"

def curl(url, headers, payload, stream=False):
    """Execute curl and return (status_code, total_time, ttfb, body_or_chunk_count)."""
    tmpfile = tempfile.mktemp()
    cmd = [
        "curl", "-s", "-o", tmpfile,
        "-w", "%{http_code}|%{time_total}|%{time_starttransfer}",
        url
    ]
    for h in headers:
        cmd.extend(["-H", h])
    cmd.extend(["-d", json.dumps(payload)])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    meta = result.stdout.strip().split("|")
    status_code = int(meta[0]) if meta else 0
    total_time = float(meta[1]) if len(meta) > 1 else 0
    ttfb = float(meta[2]) if len(meta) > 2 else 0

    body = ""
    chunk_count = 0
    if os.path.exists(tmpfile):
        with open(tmpfile, "r") as f:
            body = f.read()
        if stream:
            chunk_count = body.count("\ndata:")
        os.unlink(tmpfile)

    return status_code, total_time, ttfb, body, chunk_count


def parse_anthropic(body):
    """Parse Anthropic /v1/messages response."""
    try:
        d = json.loads(body)
        usage = d.get("usage", {})
        return {
            "input_tokens": usage.get("input_tokens", "?"),
            "output_tokens": usage.get("output_tokens", "?"),
            "stop_reason": d.get("stop_reason", "?"),
            "content_types": [c.get("type") for c in d.get("content", [])],
        }
    except Exception as e:
        return {"error": str(e)}


def parse_openai_responses(body):
    """Parse OpenAI /v1/responses response."""
    try:
        d = json.loads(body)
        usage = d.get("usage", {})
        output_types = [o.get("type") for o in d.get("output", [])]
        return {
            "prompt_tokens": usage.get("prompt_tokens", "?"),
            "completion_tokens": usage.get("completion_tokens", "?"),
            "reasoning_tokens": usage.get("reasoning_tokens", "?"),
            "status": d.get("status", "?"),
            "output_types": output_types,
        }
    except Exception as e:
        return {"error": str(e)}


def run_benchmark(name, prompt, max_tokens, stream=False, runs=3):
    """Run a single benchmark scenario on both protocols."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Prompt: \"{prompt[:80]}{'...' if len(prompt)>80 else ''}\"")
    print(f"  Max tokens: {max_tokens} | Streaming: {stream} | Runs: {runs}")
    print(f"{'='*70}")

    anthropic_headers = [
        "Content-Type: application/json",
        f"Authorization: Bearer {API_KEY}",
        "anthropic-version: 2023-06-01",
    ]
    openai_headers = [
        "Content-Type: application/json",
        f"Authorization: Bearer {API_KEY}",
    ]

    for run in range(1, runs + 1):
        print(f"\n  --- Run {run} ---")

        # Anthropic /v1/messages
        anthropic_payload = {
            "model": "glm-5.2",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if stream:
            anthropic_payload["stream"] = True

        sc, total, ttfb, body, chunks = curl(
            f"{BASE}/v1/messages", anthropic_headers, anthropic_payload, stream=stream
        )
        if not stream:
            info = parse_anthropic(body)
            print(f"  Anthropic  /v1/messages:    HTTP {sc} | {total:.3f}s | TTFB {ttfb:.3f}s | "
                  f"out={info.get('output_tokens','?')} | stop={info.get('stop_reason','?')} | "
                  f"types={info.get('content_types','?')}")
        else:
            print(f"  Anthropic  /v1/messages:    HTTP {sc} | {total:.3f}s | TTFB {ttfb:.3f}s | "
                  f"SSE chunks={chunks}")

        # OpenAI /v1/responses
        openai_payload = {
            "model": "glm-5.2",
            "max_output_tokens": max_tokens,
            "input": prompt,
        }
        if stream:
            openai_payload["stream"] = True

        sc, total, ttfb, body, chunks = curl(
            f"{BASE}/v1/responses", openai_headers, openai_payload, stream=stream
        )
        if not stream:
            info = parse_openai_responses(body)
            print(f"  OpenAI     /v1/responses:   HTTP {sc} | {total:.3f}s | TTFB {ttfb:.3f}s | "
                  f"comp={info.get('completion_tokens','?')} | reason={info.get('reasoning_tokens','?')} | "
                  f"status={info.get('status','?')} | types={info.get('output_types','?')}")
        else:
            print(f"  OpenAI     /v1/responses:   HTTP {sc} | {total:.3f}s | TTFB {ttfb:.3f}s | "
                  f"SSE chunks={chunks}")


def run_concurrent_benchmark(name, prompt, max_tokens, concurrency, runs=1):
    """Run concurrent requests on both protocols."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Concurrency: {concurrency} | Max tokens: {max_tokens} | Streaming: true")
    print(f"{'='*70}")

    anthropic_headers = [
        "Content-Type: application/json",
        f"Authorization: Bearer {API_KEY}",
        "anthropic-version: 2023-06-01",
    ]
    openai_headers = [
        "Content-Type: application/json",
        f"Authorization: Bearer {API_KEY}",
    ]

    for proto_name, endpoint, headers, payload_template in [
        ("Anthropic /v1/messages", f"{BASE}/v1/messages", anthropic_headers, {
            "model": "glm-5.2",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }),
        ("OpenAI    /v1/responses", f"{BASE}/v1/responses", openai_headers, {
            "model": "glm-5.2",
            "max_output_tokens": max_tokens,
            "input": prompt,
            "stream": True,
        }),
    ]:
        print(f"\n  --- {proto_name} ({concurrency} concurrent) ---")

        def single_request(_):
            sc, total, ttfb, body, chunks = curl(endpoint, headers, payload_template, stream=True)
            return sc, total, ttfb, chunks

        start = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(single_request, i) for i in range(concurrency)]
            results = [f.result() for f in as_completed(futures)]
        wall = time.time() - start

        ttfts = [r[2] for r in results if r[0] == 200]
        totals = [r[1] for r in results if r[0] == 200]
        chunks_list = [r[3] for r in results if r[0] == 200]
        successes = sum(1 for r in results if r[0] == 200)

        if ttfts:
            print(f"    Success: {successes}/{concurrency}")
            print(f"    Wall time: {wall:.3f}s")
            print(f"    Throughput: {successes/wall:.2f} req/s")
            print(f"    TTFB  — min: {min(ttfts):.3f}s | max: {max(ttfts):.3f}s | avg: {sum(ttfts)/len(ttfts):.3f}s")
            print(f"    Total — min: {min(totals):.3f}s | max: {max(totals):.3f}s | avg: {sum(totals)/len(totals):.3f}s")
            print(f"    SSE chunks — min: {min(chunks_list)} | max: {max(chunks_list)} | avg: {sum(chunks_list)/len(chunks_list):.0f}")
        else:
            print(f"    All failed! Results: {[(r[0],) for r in results]}")


def run_tool_call_benchmark():
    """Benchmark tool calling on both protocols."""
    print(f"\n{'='*70}")
    print(f"  TOOL CALL BENCHMARK")
    print(f"{'='*70}")

    prompt = "What is the weather in Paris? Use the get_weather tool."

    # Anthropic tool format
    anthropic_headers = [
        "Content-Type: application/json",
        f"Authorization: Bearer {API_KEY}",
        "anthropic-version: 2023-06-01",
    ]
    anthropic_payload = {
        "model": "glm-5.2",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{
            "name": "get_weather",
            "description": "Get weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            }
        }]
    }

    sc, total, ttfb, body, _ = curl(f"{BASE}/v1/messages", anthropic_headers, anthropic_payload)
    info = parse_anthropic(body)
    # Extract tool call details
    try:
        d = json.loads(body)
        tool_uses = [c for c in d.get("content", []) if c.get("type") == "tool_use"]
        if tool_uses:
            tu = tool_uses[0]
            tool_detail = f"name={tu.get('name')} | input={tu.get('input')}"
        else:
            tool_detail = "no tool_use found"
    except:
        tool_detail = "parse error"
    print(f"\n  Anthropic  /v1/messages:    HTTP {sc} | {total:.3f}s | TTFB {ttfb:.3f}s")
    print(f"    stop_reason={info.get('stop_reason','?')} | output_tokens={info.get('output_tokens','?')}")
    print(f"    tool_call: {tool_detail}")

    # OpenAI Responses tool format
    openai_headers = [
        "Content-Type: application/json",
        f"Authorization: Bearer {API_KEY}",
    ]
    openai_payload = {
        "model": "glm-5.2",
        "max_output_tokens": 256,
        "input": prompt,
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            }
        }]
    }

    sc, total, ttfb, body, _ = curl(f"{BASE}/v1/responses", openai_headers, openai_payload)
    info = parse_openai_responses(body)
    # Extract function call details
    try:
        d = json.loads(body)
        func_calls = [o for o in d.get("output", []) if o.get("type") == "function_call"]
        if func_calls:
            fc = func_calls[0]
            tool_detail = f"name={fc.get('name')} | call_id={fc.get('call_id')} | args={fc.get('arguments')}"
        else:
            tool_detail = "no function_call found"
    except:
        tool_detail = "parse error"
    print(f"\n  OpenAI     /v1/responses:   HTTP {sc} | {total:.3f}s | TTFB {ttfb:.3f}s")
    print(f"    status={info.get('status','?')} | completion_tokens={info.get('completion_tokens','?')}")
    print(f"    tool_call: {tool_detail}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Benchmark: Anthropic /v1/messages vs OpenAI /v1/responses     ║")
    print("║  Target: glm52-1tp8.jmpti.woa.com (TP8, GLM-5.2-FP8)          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Health check
    sc, total, _, _, _ = curl(f"{BASE}/health", [], {}, )
    print(f"\nHealth: HTTP {sc} | {total:.3f}s")

    # Round 1: Short prompt, short output (non-stream)
    run_benchmark(
        "ROUND 1: Short prompt / short output (non-stream)",
        "What is 2+2? Answer in one word.",
        max_tokens=64, stream=False, runs=3
    )

    # Round 2: Short prompt, short output (stream)
    run_benchmark(
        "ROUND 2: Short prompt / short output (streaming)",
        "What is 2+2? Answer in one word.",
        max_tokens=64, stream=True, runs=3
    )

    # Round 3: Medium prompt, medium output (non-stream)
    run_benchmark(
        "ROUND 3: Medium prompt / 256 tokens (non-stream)",
        "Explain the difference between TCP and UDP in networking. Provide a detailed comparison.",
        max_tokens=256, stream=False, runs=3
    )

    # Round 4: Medium prompt, medium output (stream)
    run_benchmark(
        "ROUND 4: Medium prompt / 256 tokens (streaming)",
        "Explain the difference between TCP and UDP in networking. Provide a detailed comparison.",
        max_tokens=256, stream=True, runs=3
    )

    # Round 5: Long output (512 tokens, streaming)
    run_benchmark(
        "ROUND 5: Long output / 512 tokens (streaming)",
        "Write a comprehensive guide to Python decorators, including examples of function decorators, class decorators, and practical use cases.",
        max_tokens=512, stream=True, runs=2
    )

    # Round 6: Concurrent streaming (5 parallel)
    run_concurrent_benchmark(
        "ROUND 6: Concurrent streaming (5 parallel, 256 tokens)",
        "Explain how garbage collection works in Python.",
        max_tokens=256, concurrency=5
    )

    # Round 7: Concurrent streaming (10 parallel)
    run_concurrent_benchmark(
        "ROUND 7: Concurrent streaming (10 parallel, 128 tokens)",
        "What are the benefits of using Kubernetes?",
        max_tokens=128, concurrency=10
    )

    # Tool call benchmark
    run_tool_call_benchmark()

    print(f"\n{'='*70}")
    print("  BENCHMARK COMPLETE")
    print(f"{'='*70}")
