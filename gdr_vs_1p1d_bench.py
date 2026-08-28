#!/usr/bin/env python3
"""Compare gdr vs 1p1d HTTPRoute performance for GLM-5.2 PD.

GLM-5.2 is a reasoning model: it emits reasoning_content (thinking) then content.
This script tracks both for accurate TTFT / ITL / throughput.

Usage:
    python gdr_vs_1p1d_bench.py
    python gdr_vs_1p1d_bench.py --rounds 3 --input-tokens 0,512,4096 --output-tokens 256,1024
"""
import argparse
import json
import statistics
import time

import requests

ENDPOINTS = {
    "gdr":  {
        "url": "http://glm52-gdr.jmpti.woa.com",
        "key": "",  # gdr needs no key
    },
    "1p1d": {
        "url": "http://glm52-pd-1p1d.jmpti.woa.com",
        "key": "sk-46faecc9d0bc4dcd9db6a15c73ae91c8",
    },
}
MODEL = "glm-5.2"
RESULTS_FILE = "/tmp/gdr_vs_1p1d_results.json"
LOG_FILE = "/tmp/gdr_vs_1p1d_bench.log"


def log(msg):
    line = "[{}] {}".format(time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def fmt_ms(v):
    return f"{v:.1f}" if v is not None else "N/A"


def fmt_tps(v):
    return f"{v:.1f}" if v is not None else "N/A"


def check_health(base, key):
    headers = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        r = requests.get(f"{base}/health", headers=headers, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def gen_prompt(target_tokens):
    if target_tokens <= 0:
        return "What is 2+2? Answer with just the number."
    base = "The quick brown fox jumps over the lazy dog. This is a test. "
    ratio = 4.0
    target_chars = int(target_tokens * ratio)
    reps = target_chars // len(base) + 1
    text = (base * reps)[:target_chars]
    return text + "\n\nWhat is 2+2? Answer with just the number."


def run_one(base, key, input_tokens, output_tokens):
    prompt = gen_prompt(input_tokens)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
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
    reasoning_parts = []
    content_parts = []
    usage = {}

    try:
        r = requests.post(
            f"{base}/v1/chat/completions",
            headers=headers,
            json=body,
            stream=True,
            timeout=600,
        )
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}

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

            reasoning = delta.get("reasoning_content")
            content = delta.get("content")

            # Track any token arrival (reasoning or content)
            if reasoning or content:
                if ttft is None:
                    ttft = time.perf_counter() - t_start
                token_times.append(time.perf_counter())
            if reasoning:
                reasoning_parts.append(reasoning)
            if content:
                content_parts.append(content)

            if chunk.get("usage"):
                usage = chunk["usage"]

    except Exception as e:
        elapsed = time.perf_counter() - t_start
        return {"error": f"Exception ({elapsed:.3f}s): {e}"}

    e2e = time.perf_counter() - t_start
    itls = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]

    completion_tokens = usage.get("completion_tokens", len(token_times))
    reasoning_tokens = usage.get("reasoning_tokens", 0)
    content_tokens = completion_tokens - reasoning_tokens

    def pct(lst, p):
        if len(lst) < 2:
            return None
        idx = min(int(len(lst) * p / 100), len(lst) - 1)
        return sorted(lst)[idx]

    return {
        "ttft_ms": (ttft * 1000) if ttft is not None else None,
        "e2e_ms": e2e * 1000,
        "itl_mean_ms": (statistics.mean(itls) * 1000) if itls else None,
        "itl_median_ms": (statistics.median(itls) * 1000) if itls else None,
        "itl_p90_ms": (pct(itls, 90) * 1000) if itls else None,
        "itl_p99_ms": (pct(itls, 99) * 1000) if itls else None,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "content_tokens": content_tokens,
        "throughput_tok_s": (completion_tokens / e2e) if e2e > 0 and completion_tokens else None,
        "content_preview": "".join(content_parts)[:60],
    }


def run_config(name, base, key, input_tokens, output_tokens, rounds):
    log(f"  [{name}] in={input_tokens} out={output_tokens} rounds={rounds}")
    results = []
    for i in range(rounds):
        res = run_one(base, key, input_tokens, output_tokens)
        if "error" in res:
            log(f"    R{i+1}: FAIL {res['error']}")
            continue
        results.append(res)
        log(
            f"    R{i+1}: TTFT={fmt_ms(res['ttft_ms'])}ms "
            f"E2E={fmt_ms(res['e2e_ms'])}ms "
            f"ITL={fmt_ms(res['itl_mean_ms'])}ms "
            f"TPS={fmt_tps(res['throughput_tok_s'])} "
            f"tok={res['completion_tokens']}(r={res['reasoning_tokens']},c={res['content_tokens']})"
        )
    if not results:
        return None

    def agg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        if not vals:
            return None
        return {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
        }

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "rounds": len(results),
        "ttft_ms": agg("ttft_ms"),
        "e2e_ms": agg("e2e_ms"),
        "itl_mean_ms": agg("itl_mean_ms"),
        "itl_p99_ms": agg("itl_p99_ms"),
        "throughput_tok_s": agg("throughput_tok_s"),
        "reasoning_tokens": agg("reasoning_tokens"),
        "content_tokens": agg("content_tokens"),
    }


def print_comparison(configs):
    log(f"\n{'='*120}")
    log("COMPARISON: gdr vs 1p1d  (mean / median)")
    log(f"{'='*120}")
    log(f"{'Config':<22} {'TTFT(ms)':<18} {'E2E(ms)':<18} {'ITL_mean(ms)':<18} {'ITL_p99(ms)':<18} {'TPS(tok/s)':<14}")
    log("-" * 120)

    for cfg in configs:
        for ep in ["gdr", "1p1d"]:
            d = cfg["results"].get(ep)
            if not d:
                log(f"  {ep:<6} in={cfg['in']:>6} out={cfg['out']:>4}  FAILED")
                continue

            def fa(a):
                if not a:
                    return "N/A"
                return f"{a['mean']:.0f}/{a['median']:.0f}"

            label = f"{ep} in={cfg['in']} out={cfg['out']}"
            log(
                f"  {label:<20} "
                f"{fa(d['ttft_ms']):<18} "
                f"{fa(d['e2e_ms']):<18} "
                f"{fa(d['itl_mean_ms']):<18} "
                f"{fa(d['itl_p99_ms']):<18} "
                f"{fa(d['throughput_tok_s']):<14}"
            )
        log("")


def main():
    parser = argparse.ArgumentParser(description="Compare gdr vs 1p1d performance")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--input-tokens", default="0,512,2048,8192")
    parser.add_argument("--output-tokens", default="256,1024")
    args = parser.parse_args()

    open(LOG_FILE, "w").close()
    log("Starting gdr vs 1p1d benchmark")

    for name, ep in ENDPOINTS.items():
        ok = check_health(ep["url"], ep["key"])
        log(f"  {name} ({ep['url']}): {'OK' if ok else 'UNREACHABLE'}")
        if not ok:
            log(f"FATAL: {name} not reachable, aborting.")
            return

    input_list = [int(x) for x in args.input_tokens.split(",")]
    output_list = [int(x) for x in args.output_tokens.split(",")]

    all_configs = []
    for out_tok in output_list:
        for in_tok in input_list:
            cfg = {"in": in_tok, "out": out_tok, "results": {}}
            for name, ep in ENDPOINTS.items():
                cfg["results"][name] = run_config(
                    name, ep["url"], ep["key"], in_tok, out_tok, args.rounds
                )
            all_configs.append(cfg)

    print_comparison(all_configs)

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_configs, f, indent=2, ensure_ascii=False)
    log(f"\nResults saved to {RESULTS_FILE}")
    log("=== BENCHMARK DONE ===")


if __name__ == "__main__":
    main()
