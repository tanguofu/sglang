#!/usr/bin/env python3
"""
Standalone benchmark for GLM-5.2 SGLang on MI308X.
No sglang.bench_serving dependency — uses only requests + asyncio.

Usage:
  python3 bench_standalone.py --base-url http://21.151.225.152:30000 --label v3-w1
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"


def make_payload(input_len: int, output_len: int) -> dict:
    """Build a chat completion payload with approximately input_len tokens."""
    # GLM tokenizer: ~1 token per ~3.5 chars for English, ~1 token per ~1.5 chars for Chinese.
    # Use a repeat pattern that tokenizes close to input_len tokens.
    word = "hello world "
    # Each "hello world " is ~3-4 tokens. Scale to hit input_len.
    repeats = max(1, int(input_len / 3))
    content = (word * repeats)[: input_len * 4]
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": output_len,
        "temperature": 0,
        "stream": False,
    }


def send_one(base_url: str, payload: dict, timeout: float = 300.0) -> dict:
    """Send one request and return timing + usage info."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            return {"ok": False, "dt": dt, "status": r.status_code, "body": r.text[:200]}
        data = r.json()
        usage = data.get("usage", {})
        # Estimate TTFT from server-timing if present, else leave None.
        return {
            "ok": True,
            "dt": dt,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": usage.get("reasoning_tokens", 0),
        }
    except Exception as e:
        return {"ok": False, "dt": time.perf_counter() - t0, "error": str(e)}


def run_bench(
    base_url: str,
    label: str,
    name: str,
    num_prompts: int,
    request_rate: float,
    input_len: int,
    output_len: int,
) -> dict:
    """Run a single benchmark scenario with fixed-rate concurrent requests."""
    print(f"\n[{label}] {name}: num_prompts={num_prompts}, rate={request_rate}/s, "
          f"input={input_len}, output={output_len}")
    interval = 1.0 / request_rate if request_rate > 0 else 0

    payload = make_payload(input_len, output_len)

    results = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(num_prompts, 1)) as ex:
        futures = []
        for i in range(num_prompts):
            futures.append(ex.submit(send_one, base_url, payload))
            if interval > 0 and i < num_prompts - 1:
                time.sleep(interval)
        for f in futures:
            results.append(f.result())
    total_dt = time.perf_counter() - t_start

    ok_results = [r for r in results if r.get("ok")]
    fail_results = [r for r in results if not r.get("ok")]

    if not ok_results:
        print(f"  ALL FAILED ({len(fail_results)} requests)")
        for r in fail_results[:3]:
            print(f"    {r}")
        return {"name": name, "success": 0, "total": num_prompts}

    latencies = [r["dt"] for r in ok_results]
    prompt_tokens = [r["prompt_tokens"] for r in ok_results]
    completion_tokens = [r["completion_tokens"] for r in ok_results]
    total_completion = sum(completion_tokens)
    total_prompt = sum(prompt_tokens)

    # Throughput: total completion tokens / total wall time
    throughput = total_completion / total_dt if total_dt > 0 else 0
    # Mean latency
    mean_lat = statistics.mean(latencies)
    median_lat = statistics.median(latencies)
    p90_lat = sorted(latencies)[int(len(latencies) * 0.9)] if len(latencies) >= 10 else max(latencies)
    p99_lat = sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) >= 100 else max(latencies)

    # Output token/s per request (gen throughput per request)
    per_req_gen_tps = [c / r["dt"] for r, c in zip(ok_results, completion_tokens) if r["dt"] > 0]
    mean_gen_tps = statistics.mean(per_req_gen_tps) if per_req_gen_tps else 0

    print(f"  Success: {len(ok_results)}/{num_prompts}, Fail: {len(fail_results)}")
    print(f"  Total wall time: {total_dt:.2f}s")
    print(f"  Total prompt tokens: {total_prompt}, completion tokens: {total_completion}")
    print(f"  Aggregate throughput: {throughput:.2f} completion token/s")
    print(f"  Mean per-request gen throughput: {mean_gen_tps:.2f} token/s")
    print(f"  Latency (s): mean={mean_lat:.2f}, median={median_lat:.2f}, "
          f"p90={p90_lat:.2f}, p99={p99_lat:.2f}")
    if fail_results:
        print(f"  Failures (first 3):")
        for r in fail_results[:3]:
            print(f"    {r}")

    return {
        "name": name,
        "success": len(ok_results),
        "total": num_prompts,
        "wall_time_s": total_dt,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "throughput_token_s": throughput,
        "mean_gen_tps": mean_gen_tps,
        "latency_mean_s": mean_lat,
        "latency_median_s": median_lat,
        "latency_p90_s": p90_lat,
        "latency_p99_s": p99_lat,
    }


def snapshot_metrics(base_url: str) -> str:
    """Grab EAGLE/spec/hicache metrics from /metrics."""
    try:
        r = requests.get(f"{base_url}/metrics", timeout=10)
        if r.status_code != 200:
            return f"(metrics status {r.status_code})"
        lines = r.text.splitlines()
        keep = []
        keywords = ["sglang:spec", "sglang:max_total", "sglang:hicache",
                    "sglang:eagle", "sglang:gen_throughput", "sglang:decode"]
        for line in lines:
            if line.startswith("#"):
                continue
            if any(kw in line for kw in keywords):
                keep.append(line)
        return "\n".join(keep) if keep else "(no matching metrics)"
    except Exception as e:
        return f"(metrics error: {e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://21.151.225.152:30000")
    ap.add_argument("--label", default="v3-w1")
    ap.add_argument("--outdir", default="/tmp/bench-results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"============================================")
    print(f"  Standalone Benchmark — {args.label}")
    print(f"  URL: {args.base_url}")
    print(f"  Model: {MODEL}")
    print(f"  EAGLE: steps=3, draft_tokens=4, eagle_topk=1")
    print(f"============================================")

    # Sanity check
    print("\n[0/5] Health check...")
    try:
        r = requests.get(f"{args.base_url}/health", timeout=5)
        print(f"  /health: {r.status_code}")
        if r.status_code != 200:
            print("  Health check failed, aborting.")
            sys.exit(1)
    except Exception as e:
        print(f"  Health check error: {e}, aborting.")
        sys.exit(1)

    all_results = []

    # [1] short_c32: 32 input, 256 output, 32 prompts, rate 8
    all_results.append(run_bench(
        args.base_url, args.label, "short_c32",
        num_prompts=32, request_rate=8, input_len=32, output_len=256,
    ))

    # [2] short_c128: 128 input, 256 output, 32 prompts, rate 8
    all_results.append(run_bench(
        args.base_url, args.label, "short_c128",
        num_prompts=32, request_rate=8, input_len=128, output_len=256,
    ))

    # [3] mid_c2048: 2048 input, 256 output, 16 prompts, rate 4
    #     — the original EAGLE coredump trigger
    all_results.append(run_bench(
        args.base_url, args.label, "mid_c2048",
        num_prompts=16, request_rate=4, input_len=2048, output_len=256,
    ))

    # [4] long_c8192: 8192 input, 256 output, 8 prompts, rate 2
    #     — stress test beyond the original 2048 trigger
    all_results.append(run_bench(
        args.base_url, args.label, "long_c8192",
        num_prompts=8, request_rate=2, input_len=8192, output_len=256,
    ))

    # [5] Metrics snapshot
    print(f"\n[5/5] EAGLE/HiCache metrics snapshot...")
    metrics = snapshot_metrics(args.base_url)
    print(metrics)

    # Save results
    out_file = os.path.join(args.outdir, f"{args.label}.json")
    with open(out_file, "w") as f:
        json.dump({"label": args.label, "results": all_results, "metrics": metrics}, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary table
    print(f"\n============================================")
    print(f"  Summary — {args.label}")
    print(f"============================================")
    print(f"{'Scenario':<15} {'Success':<10} {'Wall(s)':<10} {'Tput(tok/s)':<14} "
          f"{'Gen(req/s)':<12} {'Mean(s)':<10} {'P90(s)':<10}")
    for r in all_results:
        print(f"{r['name']:<15} {r['success']}/{r['total']:<8} "
              f"{r.get('wall_time_s', 0):<10.2f} "
              f"{r.get('throughput_token_s', 0):<14.2f} "
              f"{r.get('mean_gen_tps', 0):<12.2f} "
              f"{r.get('latency_mean_s', 0):<10.2f} "
              f"{r.get('latency_p90_s', 0):<10.2f}")


if __name__ == "__main__":
    main()
