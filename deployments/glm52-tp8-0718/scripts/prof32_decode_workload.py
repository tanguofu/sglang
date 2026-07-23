#!/usr/bin/env python3
"""
Controlled decode-phase workload for sglang GLM-5.2 TP8 profiling.

Runs INSIDE the test32-sglang-0 pod (localhost:30000). Drives 8 concurrent
streaming decode requests with ~12K input tokens, temp 0.0, max_tokens 256,
for ~90s. A background thread snapshots /metrics every 15s to
/tmp/prof32-metrics-<idx>.txt so the host can collect them afterward.

Designed to pair with py-spy on the rank-0 scheduler process and with the
sibling agent's GPU rocprof run. Decode-only emphasis: long shared prefix
(prefix-cacheable) + short max_tokens so the run is dominated by decode
forward passes (and EAGLE verify) rather than prefill.
"""
import json
import os
import sys
import threading
import time
import urllib.request

BASE = "http://localhost:30000"
API_KEY = os.environ.get("API_KEY", "sk-46faecc9d0bc4dcd9db6a15c73ae91c8")
MODEL = os.environ.get("MODEL", "glm-5.2")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
DURATION_S = float(os.environ.get("DURATION_S", "90"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
TARGET_INPUT_TOKENS = int(os.environ.get("TARGET_INPUT_TOKENS", "12000"))
METRICS_INTERVAL = float(os.environ.get("METRICS_INTERVAL", "15"))

# A varied, prefix-cacheable block (~270 tokens). Repeated to reach the target
# input length. The shared header makes the prefix cacheable across requests.
HEADER = (
    "You are a senior SRE agent. You have been investigating a production "
    "incident on the GLM-5.2 inference cluster. Below is the accumulated "
    "context from prior turns. Continue the investigation and answer the "
    "final question concisely.\n\n"
)
BLOCK = (
    "--- turn {i:04d} ---\n"
    "user: analyze the following system metrics and propose a remediation plan.\n"
    "assistant: I will review the metrics, identify the bottleneck, and propose "
    "a step-by-step remediation. Let me gather the relevant telemetry first.\n"
    "tool_call: query_prometheus(metric='sglang:num_requests_running', range='5m')\n"
    "tool_result: peak=18, p95=15, mean=11.2 across both workers; worker-1 "
    "carries 0.88 of the load under cache_aware routing.\n"
    "tool_call: query_prometheus(metric='sglang:cache_hit_rate', range='5m')\n"
    "tool_result: worker-1=0.62, worker-2=0.04; prefix cache is concentrated "
    "on worker-1 because the shared system prompt only resides there.\n"
    "assistant: the imbalance is caused by cache_aware sticking to the worker "
    "that holds the prefix. The balance thresholds should trigger a spill once "
    "the active-request gap exceeds the configured abs/rel thresholds.\n"
)
TAIL = (
    "\n\nFinal question: based on the full transcript above, in one sentence, "
    "what is the single highest-impact action to restore load balance?"
)


def build_prompt(target_tokens: int) -> str:
    per_block_tokens = 270
    n = max(1, target_tokens // per_block_tokens)
    blocks = "\n".join(BLOCK.format(i=i) for i in range(n))
    return HEADER + blocks + TAIL


PROMPT = build_prompt(TARGET_INPUT_TOKENS)

stop_flag = threading.Event()
stats_lock = threading.Lock()
stats = {
    "completed": 0,
    "failed": 0,
    "total_output_tokens": 0,
    "ttfts": [],
    "itls": [],
    "wall_times": [],
}


def fetch_metrics(idx: int):
    """Snapshot /metrics to a file for later host-side collection."""
    try:
        req = urllib.request.Request(f"{BASE}/metrics")
        with urllib.request.urlopen(req, timeout=15) as r:
            text = r.read().decode("utf-8", "replace")
        path = f"/tmp/prof32-metrics-{idx:02d}.txt"
        with open(path, "w") as f:
            f.write(text)
    except Exception as e:
        sys.stderr.write(f"[metrics {idx}] error: {e}\n")


def metrics_loop():
    idx = 0
    fetch_metrics(idx)
    idx += 1
    while not stop_flag.is_set():
        if stop_flag.wait(METRICS_INTERVAL):
            break
        fetch_metrics(idx)
        idx += 1
    fetch_metrics(idx)  # final


def stream_one(req_id: int):
    """One streaming decode request. Returns per-request timing."""
    payload = json.dumps({
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/v1/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    last_t = None
    itls = []
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue
                now = time.perf_counter()
                choices = evt.get("choices", [])
                if not choices:
                    continue
                # v1/completions streaming: choices[0].text carries the delta
                delta = choices[0].get("text", "")
                if delta:
                    if ttft is None:
                        ttft = now - t0
                    n_tokens += 1
                    if last_t is not None:
                        itls.append(now - last_t)
                    last_t = now
    except Exception as e:
        with stats_lock:
            stats["failed"] += 1
        sys.stderr.write(f"[req {req_id}] error: {e}\n")
        return
    wall = time.perf_counter() - t0
    with stats_lock:
        stats["completed"] += 1
        stats["total_output_tokens"] += n_tokens
        if ttft is not None:
            stats["ttfts"].append(ttft)
        stats["itls"].extend(itls)
        stats["wall_times"].append(wall)


def worker(loop_id: int):
    """Loop sending requests until the global duration expires."""
    rid = loop_id * 1000
    while not stop_flag.is_set():
        stream_one(rid)
        rid += 1


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    print(f"prompt chars={len(PROMPT)} (~{len(PROMPT)//4} tokens)", flush=True)
    print(f"concurrency={CONCURRENCY} duration={DURATION_S}s "
          f"max_tokens={MAX_TOKENS}", flush=True)

    mthread = threading.Thread(target=metrics_loop, daemon=True)
    mthread.start()

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(CONCURRENCY)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()

    time.sleep(DURATION_S)
    stop_flag.set()
    for t in threads:
        t.join(timeout=300)

    wall = time.perf_counter() - t0
    mthread.join(timeout=30)

    with stats_lock:
        ttfts = list(stats["ttfts"])
        itls = list(stats["itls"])
        walls = list(stats["wall_times"])
        completed = stats["completed"]
        failed = stats["failed"]
        total_out = stats["total_output_tokens"]

    decode_tput = total_out / wall if wall > 0 else 0.0
    summary = {
        "wall_clock_s": round(wall, 2),
        "concurrency": CONCURRENCY,
        "completed": completed,
        "failed": failed,
        "total_output_tokens": total_out,
        "aggregate_decode_tput_tps": round(decode_tput, 2),
        "ttft_s": {
            "p50": round(pct(ttfts, 50), 4) if ttfts else None,
            "p95": round(pct(ttfts, 95), 4) if ttfts else None,
            "mean": round(sum(ttfts) / len(ttfts), 4) if ttfts else None,
        },
        "itl_s": {
            "p50": round(pct(itls, 50), 5) if itls else None,
            "p95": round(pct(itls, 95), 5) if itls else None,
            "mean": round(sum(itls) / len(itls), 5) if itls else None,
        },
        "per_req_wall_s": {
            "mean": round(sum(walls) / len(walls), 3) if walls else None,
        },
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    with open("/tmp/prof32-workload-summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
