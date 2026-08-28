#!/usr/bin/env python3
"""Latency benchmark for GLM-5.2 on MI308X.

Measures TTFT (Time To First Token), ITL (Inter-Token Latency),
E2E latency, and output throughput via streaming requests.

Usage:
    python 308x_latency_test.py
    python 308x_latency_test.py --rounds 5 --input-tokens 0,2048,8192 --output-tokens 128,512
"""
import argparse
import json
import statistics
import time

import requests

API = "http://127.0.0.1:30000"
KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"
LOG = "/tmp/latency_test.log"
RESULTS_FILE = "/tmp/latency_results.json"


def log(msg):
    line = "[{}] {}".format(time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def wait_health(max_wait=900):
    for i in range(max_wait // 5):
        try:
            r = requests.get(f"{API}/health", timeout=5)
            if r.status_code == 200:
                log(f"Health OK after {i*5}s")
                time.sleep(10)
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def tokenize(text):
    try:
        r = requests.post(
            f"{API}/tokenize",
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "prompt": text},
            timeout=30,
        )
        if r.status_code == 200:
            d = r.json()
            return d.get("count", d.get("len", 0))
    except Exception:
        pass
    return 0


def gen_prompt(target_tokens):
    """Generate a prompt with approximately target_tokens tokens."""
    if target_tokens <= 0:
        return "What is 2+2? Answer with just the number."
    base = "The quick brown fox jumps over the lazy dog. This is a test. "
    cnt = tokenize(base)
    ratio = len(base) / cnt if cnt and cnt > 0 else 4.0
    target_chars = int(target_tokens * ratio)
    reps = target_chars // len(base) + 1
    text = (base * reps)[:target_chars]
    return text + "\n\nWhat is 2+2? Answer with just the number."


def run_one(input_tokens, output_tokens):
    """Run a single streaming request and collect latency metrics."""
    prompt = gen_prompt(input_tokens)
    actual_in = tokenize(prompt)

    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    t_start = time.perf_counter()
    ttft = None
    token_times = []
    content_parts = []
    reasoning_parts = []
    usage = {}

    try:
        r = requests.post(
            f"{API}/v1/chat/completions",
            headers=headers,
            json=body,
            stream=True,
            timeout=300,
        )
        if r.status_code != 200:
            log(f"  HTTP {r.status_code}: {r.text[:300]}")
            return None

        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])
            if not choices:
                if chunk.get("usage"):
                    usage = chunk["usage"]
                continue
            delta = choices[0].get("delta", {})

            content = delta.get("content")
            reasoning = delta.get("reasoning_content")

            if content:
                if ttft is None:
                    ttft = time.perf_counter() - t_start
                token_times.append(time.perf_counter())
                content_parts.append(content)
            if reasoning:
                if ttft is None:
                    ttft = time.perf_counter() - t_start
                token_times.append(time.perf_counter())
                reasoning_parts.append(reasoning)

            if chunk.get("usage"):
                usage = chunk["usage"]

    except Exception as e:
        elapsed = time.perf_counter() - t_start
        log(f"  Exception ({elapsed:.3f}s): {e}")
        return None

    t_end = time.perf_counter()
    e2e = t_end - t_start

    # ITL: intervals between consecutive token arrivals
    itls = []
    for i in range(1, len(token_times)):
        itls.append(token_times[i] - token_times[i - 1])

    out_text = "".join(content_parts)
    reasoning_text = "".join(reasoning_parts)
    completion_tokens = usage.get("completion_tokens", len(token_times))
    prompt_tokens = usage.get("prompt_tokens", actual_in)

    return {
        "input_tokens_target": input_tokens,
        "input_tokens_actual": actual_in,
        "prompt_tokens": prompt_tokens,
        "output_tokens_target": output_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": (ttft * 1000) if ttft is not None else None,
        "e2e_ms": e2e * 1000,
        "itl_mean_ms": (statistics.mean(itls) * 1000) if itls else None,
        "itl_median_ms": (statistics.median(itls) * 1000) if itls else None,
        "itl_p90_ms": (statistics.quantiles(itls, n=10)[8] * 1000) if len(itls) >= 10 else None,
        "itl_p99_ms": (statistics.quantiles(itls, n=100)[98] * 1000) if len(itls) >= 100 else None,
        "throughput_tok_s": (completion_tokens / e2e) if e2e > 0 and completion_tokens else None,
        "output_preview": out_text[:80],
        "has_reasoning": len(reasoning_text) > 0,
        "reasoning_len": len(reasoning_text),
    }


def fmt_ms(v):
    return f"{v:.1f}" if v is not None else "N/A"


def run_config(input_tokens, output_tokens, rounds):
    log(f"\n{'='*60}")
    log(f"Config: input={input_tokens} tokens, output={output_tokens} tokens, rounds={rounds}")
    log(f"{'='*60}")

    results = []
    for i in range(rounds):
        log(f"  Round {i+1}/{rounds} ...")
        res = run_one(input_tokens, output_tokens)
        if res is None:
            log(f"  Round {i+1}: FAILED")
            continue
        log(
            f"  Round {i+1}: TTFT={fmt_ms(res['ttft_ms'])}ms "
            f"E2E={fmt_ms(res['e2e_ms'])}ms "
            f"ITL_mean={fmt_ms(res['itl_mean_ms'])}ms "
            f"TPS={fmt_ms(res['throughput_tok_s'])} "
            f"out_tok={res['completion_tokens']}"
        )
        results.append(res)

    if not results:
        log("  All rounds failed for this config.")
        return []

    # Aggregate
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    e2es = [r["e2e_ms"] for r in results]
    itl_means = [r["itl_mean_ms"] for r in results if r["itl_mean_ms"] is not None]
    tps = [r["throughput_tok_s"] for r in results if r["throughput_tok_s"] is not None]

    log(f"\n  --- Summary (input={input_tokens}, output={output_tokens}) ---")
    if ttfts:
        log(f"  TTFT  mean={fmt_ms(statistics.mean(ttfts))}  median={fmt_ms(statistics.median(ttfts))}  "
            f"p90={fmt_ms(statistics.quantiles(ttfts, n=10)[8] if len(ttfts) >= 10 else max(ttfts))}  ms")
    if e2es:
        log(f"  E2E   mean={fmt_ms(statistics.mean(e2es))}  median={fmt_ms(statistics.median(e2es))}  ms")
    if itl_means:
        log(f"  ITL   mean={fmt_ms(statistics.mean(itl_means))}  median={fmt_ms(statistics.median(itl_means))}  ms")
    if tps:
        log(f"  TPS   mean={fmt_ms(statistics.mean(tps))}  median={fmt_ms(statistics.median(tps))}  tok/s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark for GLM-5.2")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per config (default: 3)")
    parser.add_argument(
        "--input-tokens",
        type=str,
        default="0,512,2048,8192,32768",
        help="Comma-separated input token counts (default: 0,512,2048,8192,32768)",
    )
    parser.add_argument(
        "--output-tokens",
        type=str,
        default="128,512",
        help="Comma-separated output token counts (default: 128,512)",
    )
    args = parser.parse_args()

    open(LOG, "w").close()
    log("Starting latency benchmark")
    log(f"API={API}  MODEL={MODEL}")

    if not wait_health():
        log("FATAL: Server not ready at " + API)
        return

    input_list = [int(x) for x in args.input_tokens.split(",")]
    output_list = [int(x) for x in args.output_tokens.split(",")]

    all_results = []
    for out_tok in output_list:
        for in_tok in input_list:
            res = run_config(in_tok, out_tok, args.rounds)
            all_results.extend(res)

    # Save JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"\nResults saved to {RESULTS_FILE}")
    log("=== LATENCY BENCHMARK DONE ===")


if __name__ == "__main__":
    main()
