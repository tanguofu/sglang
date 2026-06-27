#!/usr/bin/env python3
"""
Unified benchmark script for GLM-5.2-FP8 on SGLang.
Uses only stdlib (asyncio, urllib) — no external deps needed on host.

Usage:
  python3 bench_glm52.py --url http://localhost:30000 \
    --concurrency 1,5,10,20,30 --max-tokens 200 \
    --input-tokens 0,1000 --num-warmup 1 --output results.json
"""

import argparse
import asyncio
import json
import time
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import List

DEFAULT_PROMPT = "Write a detailed analysis of quantum computing."
PADDING_TEXT = "The history of science is a fascinating subject. " * 500


@dataclass
class RequestResult:
    success: bool
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class BenchResult:
    plan: str
    concurrency: int
    max_tokens: int
    input_tokens: int
    num_requests: int
    num_ok: int
    elapsed_s: float
    output_tok_s: float
    total_tok_s: float
    req_s: float
    avg_latency_ms: float
    avg_ttft_ms: float
    p99_latency_ms: float = 0.0
    p99_ttft_ms: float = 0.0


def build_prompt(target_input_tokens: int) -> str:
    if target_input_tokens <= 0:
        return DEFAULT_PROMPT
    target_chars = target_input_tokens * 4
    prompt = PADDING_TEXT
    while len(prompt) < target_chars:
        prompt += " " + PADDING_TEXT
    return prompt[:target_chars]


async def send_request(url: str, payload: dict) -> RequestResult:
    """Send a single non-streaming request and measure latency."""
    result = RequestResult(success=False)
    start = time.monotonic()
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: urllib.request.urlopen(req, timeout=600)
        )
        body = await loop.run_in_executor(None, lambda: resp.read())
        obj = json.loads(body)
        result.latency_ms = (time.monotonic() - start) * 1000
        result.ttft_ms = result.latency_ms  # non-streaming: TTFT ≈ total latency
        usage = obj.get("usage", {})
        result.completion_tokens = usage.get("completion_tokens", 0)
        result.prompt_tokens = usage.get("prompt_tokens", 0)
        result.total_tokens = usage.get("total_tokens", 0)
        result.success = result.completion_tokens > 0
    except Exception as e:
        result.error = str(e)
        result.latency_ms = (time.monotonic() - start) * 1000
    return result


async def bench_once(
    url: str,
    concurrency: int,
    max_tokens: int,
    input_tokens: int,
    num_warmup: int = 1,
    model: str = "GLM-5.2-FP8",
) -> BenchResult:
    prompt = build_prompt(input_tokens)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }

    # Warmup
    for _ in range(num_warmup):
        await send_request(url, payload)

    # Benchmark
    start = time.monotonic()
    tasks = [send_request(url, payload) for _ in range(concurrency)]
    results: List[RequestResult] = await asyncio.gather(*tasks)
    elapsed = time.monotonic() - start

    ok = [r for r in results if r.success]
    num_ok = len(ok)
    total_comp = sum(r.completion_tokens for r in ok)
    total_tok = sum(r.total_tokens for r in ok)

    latencies = sorted([r.latency_ms for r in ok]) if ok else [0]
    ttfts = sorted([r.ttft_ms for r in ok]) if ok else [0]

    def p99(arr):
        if len(arr) <= 1:
            return arr[-1] if arr else 0
        return arr[int(len(arr) * 0.99)]

    return BenchResult(
        plan="",
        concurrency=concurrency,
        max_tokens=max_tokens,
        input_tokens=input_tokens,
        num_requests=concurrency,
        num_ok=num_ok,
        elapsed_s=round(elapsed, 2),
        output_tok_s=round(total_comp / elapsed, 1) if elapsed > 0 else 0,
        total_tok_s=round(total_tok / elapsed, 1) if elapsed > 0 else 0,
        req_s=round(num_ok / elapsed, 2) if elapsed > 0 else 0,
        avg_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0,
        avg_ttft_ms=round(sum(ttfts) / len(ttfts), 1) if ttfts else 0,
        p99_latency_ms=round(p99(latencies), 1),
        p99_ttft_ms=round(p99(ttfts), 1),
    )


async def smoke_test(url: str, model: str = "GLM-5.2-FP8") -> bool:
    print("\n=== Smoke Test ===")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 50,
        "temperature": 0,
    }
    try:
        start = time.monotonic()
        result = await send_request(url, payload)
        elapsed = (time.monotonic() - start) * 1000
        if result.success:
            print(f"  ✓ Basic chat: {elapsed:.0f}ms, {result.completion_tokens} tokens")
            return True
        else:
            print(f"  ✗ Basic chat failed: {result.error}")
            return False
    except Exception as e:
        print(f"  ✗ Smoke test error: {e}")
        return False


def print_table(results: List[BenchResult]):
    print("\n" + "=" * 120)
    print(f"{'Plan':<6} {'Conc':>4} {'InTok':>6} {'OutTok':>6} {'OK':>4} "
          f"{'OutT/s':>8} {'TotT/s':>8} {'Req/s':>6} "
          f"{'AvgLat':>8} {'P99Lat':>8} {'AvgTTFT':>8} {'P99TTFT':>8}")
    print("-" * 120)
    for r in results:
        print(f"{r.plan:<6} {r.concurrency:>4} {r.input_tokens:>6} {r.max_tokens:>6} "
              f"{r.num_ok:>4} {r.output_tok_s:>8.1f} {r.total_tok_s:>8.1f} {r.req_s:>6.2f} "
              f"{r.avg_latency_ms:>7.0f}ms {r.p99_latency_ms:>7.0f}ms "
              f"{r.avg_ttft_ms:>7.0f}ms {r.p99_ttft_ms:>7.0f}ms")
    print("=" * 120)


async def main():
    parser = argparse.ArgumentParser(description="SGLang GLM-5.2 Benchmark")
    parser.add_argument("--url", default="http://localhost:30000")
    parser.add_argument("--concurrency", default="1,5,10,20,30",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--input-tokens", default="0,1000",
                        help="Comma-separated input token counts (0=short)")
    parser.add_argument("--num-warmup", type=int, default=1)
    parser.add_argument("--model", default="GLM-5.2-FP8")
    parser.add_argument("--plan", default="unknown", help="Plan name (B/C/D)")
    parser.add_argument("--output", default="", help="Output JSON file path")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    concurrencies = [int(x) for x in args.concurrency.split(",")]
    input_tokens_list = [int(x) for x in args.input_tokens.split(",")]
    api_url = f"{args.url}/v1/chat/completions"

    # Health check
    print(f"Connecting to {args.url} ...")
    try:
        resp = urllib.request.urlopen(f"{args.url}/v1/models", timeout=10)
        models = json.loads(resp.read())
        model_ids = [m["id"] for m in models.get("data", [])]
        print(f"Available models: {model_ids}")
        # Use actual model name from server
        if model_ids:
            args.model = model_ids[0]
            print(f"Using model: {args.model}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        sys.exit(1)

    # Smoke test
    if not args.skip_smoke:
        smoke_ok = await smoke_test(api_url, args.model)
        if not smoke_ok:
            print("\n⚠ Smoke test failed, but continuing with benchmark...")

    # Run benchmarks
    all_results: List[BenchResult] = []
    for input_tok in input_tokens_list:
        label = "short" if input_tok == 0 else f"{input_tok}tok"
        print(f"\n{'='*60}")
        print(f"  Benchmark: input={label}, max_tokens={args.max_tokens}")
        print(f"{'='*60}")

        for conc in concurrencies:
            print(f"\n  Running concurrency={conc}, input_tokens={input_tok} ...")
            result = await bench_once(
                url=api_url,
                concurrency=conc,
                max_tokens=args.max_tokens,
                input_tokens=input_tok,
                num_warmup=args.num_warmup,
                model=args.model,
            )
            result.plan = args.plan
            all_results.append(result)
            print(f"    → {result.num_ok}/{result.concurrency} OK, "
                  f"output={result.output_tok_s} tok/s, "
                  f"req/s={result.req_s}, "
                  f"avg_lat={result.avg_latency_ms:.0f}ms, "
                  f"ttft={result.avg_ttft_ms:.0f}ms")

    # Print summary table
    print_table(all_results)

    # Save JSON results
    if args.output:
        output_data = {
            "plan": args.plan,
            "url": args.url,
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [asdict(r) for r in all_results],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
