#!/usr/bin/env python3
"""
End-to-end latency bottleneck benchmark for GLM-5.2 2tp8 deployment.

Measures at 3 hops:
  1. Gateway (https://glm52-2tp8.jmpti.woa.com) — full E2E
  2. Router-direct (kubectl exec into router pod, localhost:30001)
  3. Worker-direct (kubectl exec into worker pod, localhost:30000)

For each hop, measures:
  - TTFT (time to first SSE byte)
  - Total latency
  - Output token count (from SSE deltas)
  - Throughput (tokens/sec during decode phase)

Two prompt profiles:
  - Short prompt, short output (prefill-light, decode-light) — isolates fixed overhead
  - Long prompt, long output (prefill-heavy, decode-heavy) — realistic workload
"""
import argparse
import json
import re
import subprocess
import sys
import time
from typing import Optional

API_KEY = "${ANTHROPIC_AUTH_TOKEN}"
GATEWAY = "https://glm52-2tp8.jmpti.woa.com"
ROUTER_POD = "sglang-glm52-2tp8-router-55787586bf-xjw7x"
WORKER1_POD = "sglang-glm52-2tp8-sglang-0"
WORKER2_POD = "sglang-glm52-2tp8-w2-sglang-0"

# Prompt profiles
PROMPTS = {
    "short": {
        "input": "What is 2+2? Answer with just the number.",
        "max_tokens": 1500,
        "desc": "short prompt + reasoning output (prefill-light)",
    },
    "long": {
        "input": "Please write a detailed 500-word essay about the history of computing, from Charles Babbage's analytical engine through modern GPUs and AI accelerators. Cover key milestones including vacuum tubes, transistors, integrated circuits, microprocessors, and parallel computing architectures.",
        "max_tokens": 3000,
        "desc": "long prompt + long output (prefill+decode heavy)",
    },
    "code": {
        "input": "Write a Python function that implements binary search on a sorted list. Include type hints, docstring, and handle edge cases like empty lists and duplicate values. Then show 3 test cases.",
        "max_tokens": 3000,
        "desc": "code generation (medium prompt + medium output)",
    },
}


def run_curl(url: str, payload: dict, use_exec: Optional[str] = None,
             exec_pod: Optional[str] = None) -> dict:
    """Run a streaming curl request, return timing + token info.

    use_exec: if "router", kubectl exec into ROUTER_POD and curl localhost:30001
              if "worker1"/"worker2", kubectl exec into that worker pod and curl localhost:30000
    """
    headers = [
        "-H", "Authorization: Bearer " + API_KEY,
        "-H", "Content-Type: application/json",
    ]
    data = json.dumps(payload)

    if use_exec == "router":
        cmd = ["kubectl", "exec", "-n", "kube-system", ROUTER_POD, "--",
               "curl", "-s", "-N", "--max-time", "120",
               "-w", "\n__CURL_META__\nhttp_code=%{http_code}\ntime_total=%{time_total}\ntime_starttransfer=%{time_starttransfer}\nsize_download=%{size_download}\n"]
        cmd += headers + ["-X", "POST", "http://localhost:30001/v1/responses", "-d", data]
    elif use_exec == "worker1":
        cmd = ["kubectl", "exec", "-n", "kube-system", WORKER1_POD, "--",
               "curl", "-s", "-N", "--max-time", "120",
               "-w", "\n__CURL_META__\nhttp_code=%{http_code}\ntime_total=%{time_total}\ntime_starttransfer=%{time_starttransfer}\nsize_download=%{size_download}\n"]
        cmd += headers + ["-X", "POST", "http://localhost:30000/v1/responses", "-d", data]
    elif use_exec == "worker2":
        cmd = ["kubectl", "exec", "-n", "kube-system", WORKER2_POD, "--",
               "curl", "-s", "-N", "--max-time", "120",
               "-w", "\n__CURL_META__\nhttp_code=%{http_code}\ntime_total=%{time_total}\ntime_starttransfer=%{time_starttransfer}\nsize_download=%{size_download}\n"]
        cmd += headers + ["-X", "POST", "http://localhost:30000/v1/responses", "-d", data]
    else:  # gateway
        cmd = ["curl", "-s", "-N", "--max-time", "120",
               "-w", "\n__CURL_META__\nhttp_code=%{http_code}\ntime_total=%{time_total}\ntime_starttransfer=%{time_starttransfer}\nsize_download=%{size_download}\n"]
        cmd += headers + ["-X", "POST", url + "/v1/responses", "-d", data]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=130)
    wall_time = time.time() - start

    output = result.stdout
    # Split SSE stream from curl meta
    meta_match = re.search(r"__CURL_META__\n(.*)$", output, re.DOTALL)
    meta = {}
    if meta_match:
        meta_block = meta_match.group(1)
        for line in meta_block.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k] = v
        sse_output = output[:meta_match.start()]
    else:
        sse_output = output

    # Parse SSE for token counts and timing
    reasoning_tokens = 0
    output_tokens = 0
    first_byte_time = None
    first_delta_time = None
    last_delta_time = None

    for line in sse_output.split("\n"):
        if not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        etype = ev.get("type", "")
        if first_byte_time is None and etype:
            first_byte_time = time.time()
        if "delta" in ev and etype in ("response.reasoning_text.delta", "response.output_text.delta"):
            if first_delta_time is None:
                first_delta_time = time.time()
            last_delta_time = time.time()
            delta = ev.get("delta", "")
            if etype == "response.reasoning_text.delta":
                reasoning_tokens += len(delta) // 4  # rough estimate
            else:
                output_tokens += len(delta) // 4

    return {
        "http_code": int(meta.get("http_code", 0)),
        "ttft_curl": float(meta.get("time_starttransfer", 0)),  # curl TTFB
        "time_total": float(meta.get("time_total", 0)),
        "wall_time": wall_time,
        "reasoning_tokens_est": reasoning_tokens,
        "output_tokens_est": output_tokens,
        "total_tokens_est": reasoning_tokens + output_tokens,
        "sse_bytes": int(meta.get("size_download", 0)),
    }


def bench_hop(hop_name: str, prompt_key: str, use_exec: Optional[str] = None,
              url: str = GATEWAY, repeats: int = 3) -> list:
    """Run benchmark for one hop + one prompt profile."""
    prompt = PROMPTS[prompt_key]
    payload = {
        "model": "glm-5.2",
        "input": prompt["input"],
        "stream": True,
        "max_output_tokens": prompt["max_tokens"],
    }
    results = []
    for i in range(repeats):
        r = run_curl(url, payload, use_exec=use_exec)
        r["hop"] = hop_name
        r["prompt"] = prompt_key
        r["run"] = i + 1
        results.append(r)
        print(f"  [{hop_name}/{prompt_key}] run {i+1}: HTTP {r['http_code']} "
              f"TTFT={r['ttft_curl']:.2f}s total={r['time_total']:.2f}s "
              f"tokens~{r['total_tokens_est']} ({r['reasoning_tokens_est']}r+{r['output_tokens_est']}o)")
        time.sleep(0.5)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3, help="repeats per hop/prompt")
    parser.add_argument("--hops", nargs="+", default=["gateway", "router", "worker1", "worker2"],
                        choices=["gateway", "router", "worker1", "worker2"])
    parser.add_argument("--prompts", nargs="+", default=["short", "long", "code"],
                        choices=["short", "long", "code"])
    parser.add_argument("--out", default="/tmp/bench_bottleneck.json")
    args = parser.parse_args()

    all_results = []
    for prompt_key in args.prompts:
        print(f"\n=== Prompt: {prompt_key} ({PROMPTS[prompt_key]['desc']}) ===")
        for hop in args.hops:
            print(f"--- Hop: {hop} ---")
            if hop == "gateway":
                rs = bench_hop("gateway", prompt_key, use_exec=None, repeats=args.repeats)
            elif hop == "router":
                rs = bench_hop("router", prompt_key, use_exec="router", repeats=args.repeats)
            elif hop == "worker1":
                rs = bench_hop("worker1", prompt_key, use_exec="worker1", repeats=args.repeats)
            elif hop == "worker2":
                rs = bench_hop("worker2", prompt_key, use_exec="worker2", repeats=args.repeats)
            all_results.extend(rs)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY (averages)")
    print("=" * 100)
    print(f"{'hop':<12} {'prompt':<8} {'ttft(s)':<10} {'total(s)':<10} {'tokens':<10} {'tok/s':<10} {'overhead(s)':<12}")
    print("-" * 100)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r["hop"], r["prompt"])].append(r)

    # Compute gateway overhead per prompt
    for (hop, prompt_key), rs in sorted(grouped.items()):
        valid = [r for r in rs if r["http_code"] == 200]
        if not valid:
            print(f"{hop:<12} {prompt_key:<8} ALL FAILED")
            continue
        avg_ttft = sum(r["ttft_curl"] for r in valid) / len(valid)
        avg_total = sum(r["time_total"] for r in valid) / len(valid)
        avg_tokens = sum(r["total_tokens_est"] for r in valid) / len(valid)
        decode_time = avg_total - avg_ttft
        tps = avg_tokens / decode_time if decode_time > 0 else 0
        # Overhead = hop total - worker1 total (for same prompt)
        w1_key = ("worker1", prompt_key)
        if hop != "worker1" and w1_key in grouped:
            w1_valid = [r for r in grouped[w1_key] if r["http_code"] == 200]
            if w1_valid:
                w1_avg = sum(r["time_total"] for r in w1_valid) / len(w1_valid)
                overhead = avg_total - w1_avg
            else:
                overhead = 0
        else:
            overhead = 0
        print(f"{hop:<12} {prompt_key:<8} {avg_ttft:<10.2f} {avg_total:<10.2f} "
              f"{avg_tokens:<10.0f} {tps:<10.1f} {overhead:<+12.2f}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
