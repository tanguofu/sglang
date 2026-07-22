#!/usr/bin/env python3
"""
SGLang 1P1D PD-disaggregated benchmark — Claude Code scenario.

Measures:
  - TTFT  (Time To First Token) — end-to-end latency from request send to first
           generated token arriving over the stream.
  - TPOT  (Time Per Output Token) — average inter-token latency for the decode
           phase (excluding the first token).
  - Total latency, input/output token counts, throughput.

Scenarios:
  1. Short prompt, short output   — quick chat turn (Claude Code Q&A)
  2. Medium prompt, medium output — code review / explain (typical Claude Code)
  3. Long prompt, medium output   — large file context + edit instruction
  4. Concurrency sweep            — N parallel Claude Code sessions

Usage:
  python3 benchmark_claude_code.py --host http://21.151.225.144:30001 \
         --api-key sk-46faecc9d0bc4dcd9db6a15c73ae91c8 --model glm-5.2
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def stream_request(host, api_key, model, messages, max_tokens, timeout=300):
    """Send a streaming chat completion request and return timing metrics.

    Returns dict with:
      ttft_ms, tpot_ms, output_tokens, total_ms, status, error
    """
    url = f"{host}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    ttft_ms = None
    first_token_time = None
    output_tokens = 0
    error = None

    try:
        resp = requests.post(
            url, headers=headers, json=payload, stream=True, timeout=timeout
        )
        if resp.status_code != 200:
            error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return {
                "ttft_ms": None,
                "tpot_ms": None,
                "output_tokens": 0,
                "total_ms": (time.perf_counter() - t_start) * 1000,
                "status": resp.status_code,
                "error": error,
            }

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                # GLM-5.2 is a reasoning model: tokens may arrive in
                # either delta.content or delta.reasoning_content.
                content = delta.get("content")
                reasoning = delta.get("reasoning_content")
                if content or reasoning:
                    now = time.perf_counter()
                    output_tokens += 1
                    if ttft_ms is None:
                        ttft_ms = (now - t_start) * 1000
                        first_token_time = now
                    else:
                        first_token_time = now  # update for last token

            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens"):
                output_tokens = usage["completion_tokens"]

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000

        if ttft_ms is None:
            error = "no tokens received"
            tpot_ms = None
        elif output_tokens > 1:
            # TPOT = (total - TTFT) / (output_tokens - 1)
            tpot_ms = (total_ms - ttft_ms) / (output_tokens - 1)
        else:
            tpot_ms = None

        return {
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "output_tokens": output_tokens,
            "total_ms": total_ms,
            "status": 200,
            "error": error,
        }

    except requests.exceptions.Timeout:
        return {
            "ttft_ms": ttft_ms,
            "tpot_ms": None,
            "output_tokens": output_tokens,
            "total_ms": (time.perf_counter() - t_start) * 1000,
            "status": 0,
            "error": "timeout",
        }
    except Exception as e:
        return {
            "ttft_ms": ttft_ms,
            "tpot_ms": None,
            "output_tokens": output_tokens,
            "total_ms": (time.perf_counter() - t_start) * 1000,
            "status": 0,
            "error": str(e),
        }


def approx_tokens(text):
    """Rough token estimate (~4 chars/token for mixed EN/CN/code)."""
    return max(1, len(text) // 4)


def run_scenario(name, host, api_key, model, messages, max_tokens, rounds=5):
    """Run a single scenario for N rounds and report stats."""
    input_tokens = sum(approx_tokens(m["content"]) for m in messages)
    print(f"\n{'='*70}")
    print(f"Scenario: {name}")
    print(f"  input ~{input_tokens} tokens, max_output={max_tokens} tokens, rounds={rounds}")
    print(f"{'='*70}")

    results = []
    for i in range(rounds):
        r = stream_request(host, api_key, model, messages, max_tokens)
        if r["error"]:
            print(f"  [{i+1}/{rounds}] ERROR: {r['error']} "
                  f"(tokens={r['output_tokens']}, {r['total_ms']:.0f}ms)")
        else:
            print(f"  [{i+1}/{rounds}] TTFT={r['ttft_ms']:.0f}ms  "
                  f"TPOT={r['tpot_ms']:.1f}ms  "
                  f"out={r['output_tokens']} tok  "
                  f"total={r['total_ms']:.0f}ms")
        results.append(r)
        # small gap between requests to mimic human think time
        time.sleep(0.5)

    ok = [r for r in results if r["error"] is None]
    if not ok:
        print("  >>> ALL FAILED — skipping stats")
        return None

    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] is not None]
    totals = [r["total_ms"] for r in ok]
    outs = [r["output_tokens"] for r in ok]

    def stats(vals):
        if not vals:
            return "n/a"
        return (f"mean={statistics.mean(vals):.0f}ms  "
                f"p50={statistics.median(vals):.0f}ms  "
                f"min={min(vals):.0f}ms  max={max(vals):.0f}ms")

    print(f"\n  Summary ({len(ok)}/{len(results)} ok):")
    print(f"    TTFT  : {stats(ttfts)}")
    print(f"    TPOT  : {stats(tpots)}")
    print(f"    Total : {stats(totals)}")
    print(f"    Output tokens: mean={statistics.mean(outs):.0f}  "
          f"total={sum(outs)}")
    if tpots:
        print(f"    Decode throughput: {1000/statistics.mean(tpots):.1f} tok/s")

    return {
        "scenario": name,
        "input_tokens": input_tokens,
        "max_output": max_tokens,
        "rounds": len(results),
        "ok": len(ok),
        "ttft": {"mean": statistics.mean(ttfts) if ttfts else None,
                 "p50": statistics.median(ttfts) if ttfts else None,
                 "min": min(ttfts) if ttfts else None,
                 "max": max(ttfts) if ttfts else None},
        "tpot": {"mean": statistics.mean(tpots) if tpots else None,
                 "p50": statistics.median(tpots) if tpots else None,
                 "min": min(tpots) if tpots else None,
                 "max": max(tpots) if tpots else None},
        "total": {"mean": statistics.mean(totals),
                  "p50": statistics.median(totals),
                  "min": min(totals), "max": max(totals)},
        "output_tokens_mean": statistics.mean(outs),
    }


def run_concurrency(host, api_key, model, messages, max_tokens, n=4):
    """Run N concurrent requests to simulate parallel Claude Code sessions."""
    print(f"\n{'='*70}")
    print(f"Concurrency test: {n} parallel requests")
    print(f"  input ~{sum(approx_tokens(m['content']) for m in messages)} tokens, "
          f"max_output={max_tokens} tokens")
    print(f"{'='*70}")

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [
            pool.submit(stream_request, host, api_key, model, messages, max_tokens)
            for _ in range(n)
        ]
        results = [f.result() for f in as_completed(futures)]
    wall_ms = (time.perf_counter() - t_start) * 1000

    for i, r in enumerate(sorted(results, key=lambda x: x["total_ms"])):
        if r["error"]:
            print(f"  [{i+1}] ERROR: {r['error']}  ({r['total_ms']:.0f}ms)")
        else:
            print(f"  [{i+1}] TTFT={r['ttft_ms']:.0f}ms  "
                  f"TPOT={r['tpot_ms']:.1f}ms  "
                  f"out={r['output_tokens']} tok  "
                  f"total={r['total_ms']:.0f}ms")

    ok = [r for r in results if r["error"] is None]
    if ok:
        ttfts = [r["ttft_ms"] for r in ok]
        tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] is not None]
        print(f"\n  Wall time: {wall_ms:.0f}ms  ({len(ok)}/{n} ok)")
        print(f"  TTFT: mean={statistics.mean(ttfts):.0f}ms  "
              f"p50={statistics.median(ttfts):.0f}ms")
        if tpots:
            print(f"  TPOT: mean={statistics.mean(tpots):.1f}ms  "
                  f"p50={statistics.median(tpots):.1f}ms")
            print(f"  Aggregate decode throughput: "
                  f"{len(ok)*1000/statistics.mean(tpots):.1f} tok/s")


# --- Claude Code-like prompts ---

# Scenario 1: short Q&A (quick chat turn)
PROMPT_SHORT = [
    {"role": "user", "content": "What does the Python `collections.OrderedDict` do? Answer in 2 sentences."}
]

# Scenario 2: medium code review (typical Claude Code interaction)
PROMPT_CODE = [
    {"role": "user", "content": """Review this Python function for bugs and suggest improvements:

```python
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    result = [0, 1]
    for i in range(2, n):
        result.append(result[i-1] + result[i-2])
    return result

def group_by(items, key_fn):
    groups = {}
    for item in items:
        k = key_fn(item)
        if k not in groups:
            groups[k] = []
        groups[k].append(item)
    return groups
```

List the issues and show the fixed code."""}
]

# Scenario 3: long context — simulate a large file + edit instruction
LONG_FILE = "".join([
    f"def process_item_{i}(data):\n"
    f"    \"\"\"Process item {i} from the data stream.\"\"\"\n"
    f"    value = data.get('field_{i}', 0)\n"
    f"    if value > {i * 10}:\n"
    f"        return value * 2\n"
    f"    return value + {i}\n\n"
    for i in range(80)
])  # ~80 small functions, ~2000 tokens

PROMPT_LONG = [
    {"role": "user", "content": f"""Here is a module with many helper functions:

```python
{LONG_FILE}
```

Refactor this module: add a generic `process_items(data, indices)` dispatcher,
add type hints, and write a short docstring for the module. Show only the new
module-level code, no per-function changes needed."""}
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://21.151.225.144:30011")
    p.add_argument("--api-key", default="sk-46faecc9d0bc4dcd9db6a15c73ae91c8")
    p.add_argument("--model", default="glm-5.2")
    p.add_argument("--rounds", type=int, default=5,
                   help="rounds per scenario")
    p.add_argument("--concurrency", type=int, default=4,
                   help="parallel requests in concurrency test")
    p.add_argument("--skip-long", action="store_true",
                   help="skip the long-context scenario")
    p.add_argument("--skip-concurrency", action="store_true",
                   help="skip the concurrency test")
    args = p.parse_args()

    print(f"SGLang 1P1D Benchmark — Claude Code scenario")
    print(f"Host: {args.host}")
    print(f"Model: {args.model}")
    print(f"Rounds per scenario: {args.rounds}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = []

    # Warmup — 1 short request to prime RDMA connections / JIT
    print("\n>>> Warmup request (priming RDMA + JIT)...")
    w = stream_request(args.host, args.api_key, args.model,
                       PROMPT_SHORT, max_tokens=8, timeout=300)
    if w["error"]:
        print(f"    Warmup FAILED: {w['error']}")
        print("    Aborting benchmark — server not ready.")
        sys.exit(1)
    print(f"    Warmup OK: TTFT={w['ttft_ms']:.0f}ms  "
          f"out={w['output_tokens']} tok  total={w['total_ms']:.0f}ms")

    # Scenario 1: short Q&A
    r = run_scenario("Short Q&A (quick chat turn)",
                     args.host, args.api_key, args.model,
                     PROMPT_SHORT, max_tokens=64, rounds=args.rounds)
    if r:
        all_results.append(r)

    # Scenario 2: code review (typical Claude Code)
    r = run_scenario("Code review (medium prompt, medium output)",
                     args.host, args.api_key, args.model,
                     PROMPT_CODE, max_tokens=256, rounds=args.rounds)
    if r:
        all_results.append(r)

    # Scenario 3: long context
    if not args.skip_long:
        r = run_scenario("Long context refactor (~2k token prompt)",
                         args.host, args.api_key, args.model,
                         PROMPT_LONG, max_tokens=384, rounds=args.rounds)
        if r:
            all_results.append(r)

    # Concurrency test
    if not args.skip_concurrency:
        run_concurrency(args.host, args.api_key, args.model,
                        PROMPT_CODE, max_tokens=256, n=args.concurrency)

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Scenario':<45} {'TTFT mean':>10} {'TPOT mean':>10}")
    print("-" * 70)
    for r in all_results:
        ttft = f"{r['ttft']['mean']:.0f}ms" if r['ttft']['mean'] else "n/a"
        tpot = f"{r['tpot']['mean']:.1f}ms" if r['tpot']['mean'] else "n/a"
        print(f"{r['scenario']:<45} {ttft:>10} {tpot:>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
