#!/usr/bin/env python3
"""
Long agent-context concurrency stress test for GLM-5.2 2tp8 deployment.

Scenario: simulate a long-running agent session — a large context window
(multi-turn conversation history + tool transcripts) sent concurrently
through the router. This stresses:

  - Prefill throughput on long prompts (the dominant cost for agent context)
  - Router load balancing under concurrent long-context requests
  - cache_aware policy behavior (does it stick to one worker or rebalance?)

For each concurrent request we measure:
  - TTFT  (time to first SSE delta — reasoning starts)
  - TTOT  (time to first output_text delta — answer starts)
  - total wall-clock latency
  - output token count (reasoning + output)
  - decode throughput (tokens/sec after TTFT)
  - HTTP status / error

We also snapshot each worker's sglang metrics before and after the run to
compute the per-worker request/token delta — the real load-distribution signal
(the router only exposes an aggregate selection counter).

All requests go through `kubectl exec` into the router pod (localhost:30001),
matching the existing bench_bottleneck.py pattern and avoiding raw external
curl from the host.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
if not TOKEN:
    sys.exit("ERROR: ANTHROPIC_AUTH_TOKEN not set in host environment")

KUBE_NS = "kube-system"
ROUTER_POD = "sglang-glm52-2tp8-router-c58759dc6-lc89f"
WORKER1_POD = "sglang-glm52-2tp8-sglang-0"
WORKER2_POD = "sglang-glm52-2tp8-w2-sglang-0"
ROUTER_PORT = "30001"
MODEL = "glm-5.2"

# A long agent context: a synthetic multi-turn transcript with tool calls and
# results. Repeated blocks give a realistic long-prefill profile (~target tokens).
CONTEXT_BLOCK = (
    "user: I need you to analyze the following system metrics and propose a "
    "remediation plan. Here is the transcript from the last investigation turn.\n"
    "assistant: Understood. I will review the metrics, identify the bottleneck, "
    "and propose a step-by-step remediation. Let me first gather the relevant "
    "telemetry from the monitoring stack.\n"
    "tool_call: query_prometheus(metric='sglang:num_requests_running', range='5m')\n"
    "tool_result: peak=18, p95=15, mean=11.2 across both workers; worker-1 carries "
    "0.88 of the load under cache_aware routing.\n"
    "tool_call: query_prometheus(metric='sglang:cache_hit_rate', range='5m')\n"
    "tool_result: worker-1=0.62, worker-2=0.04; prefix cache is concentrated on "
    "worker-1 because the shared system prompt and conversation prefix only "
    "resides there.\n"
    "assistant: The imbalance is caused by cache_aware sticking to the worker that "
    "holds the prefix. The balance thresholds should trigger a spill once the "
    "active-request gap exceeds the configured abs/rel thresholds. I will verify "
    "the threshold settings and the resulting selection distribution next.\n"
)


def build_long_context(target_tokens: int) -> str:
    """Build a long agent-style context by repeating varied blocks.

    Targets ~target_tokens of prefill. Each block is ~120 tokens; we pad the
    block index so repeated blocks are not byte-identical (avoids trivial
    dedup) while still being prefix-cacheable up to the shared header.
    """
    header = (
        "You are a senior SRE agent. You have been investigating a production "
        "incident on the GLM-5.2 inference cluster. Below is the accumulated "
        "context from prior turns. Continue the investigation and answer the "
        "final question concisely.\n\n"
    )
    blocks = []
    i = 0
    # Measured ~270 tokens per block (block is ~1080 chars).
    per_block_tokens = 270
    n_blocks = max(1, target_tokens // per_block_tokens)
    for i in range(n_blocks):
        blocks.append(f"--- turn {i:04d} ---\n" + CONTEXT_BLOCK)
    tail = (
        "\n\nFinal question: based on the full transcript above, in one sentence, "
        "what is the single highest-impact action to restore load balance?"
    )
    return header + "\n".join(blocks) + tail


def snapshot_worker_metrics() -> dict:
    """Snapshot per-worker request/token counters from sglang /metrics."""
    out = {}
    for name, pod in (("w1", WORKER1_POD), ("w2", WORKER2_POD)):
        try:
            r = subprocess.run(
                ["kubectl", "exec", "-n", KUBE_NS, pod, "--",
                 "curl", "-s", "--max-time", "10",
                 "http://localhost:30000/metrics"],
                capture_output=True, text=True, timeout=20,
            )
            text = r.stdout
            m = {}
            for key in ("num_requests_total", "prompt_tokens_total",
                        "generation_tokens_total", "cached_tokens_total"):
                mm = re.search(rf'sglang:{key}\{{[^}}]*\}}\s+([0-9.eE+]+)', text)
                if mm:
                    m[key] = float(mm.group(1))
            # running requests
            mr = re.search(r'sglang:num_requests_running\{[^}]*\}\s+([0-9.]+)', text)
            if mr:
                m["num_requests_running"] = float(mr.group(1))
            out[name] = m
        except Exception as e:
            out[name] = {"error": str(e)}
    return out


def snapshot_router_metrics() -> dict:
    """Snapshot router selection / active-request counters."""
    try:
        r = subprocess.run(
            ["kubectl", "exec", "-n", KUBE_NS, ROUTER_POD, "--",
             "curl", "-s", "--max-time", "10",
             "http://localhost:29000/metrics"],
            capture_output=True, text=True, timeout=20,
        )
        text = r.stdout
        out = {}
        ms = re.search(r'smg_worker_selection_total\{[^}]*\}\s+([0-9.]+)', text)
        if ms:
            out["selection_total"] = float(ms.group(1))
        mr = re.search(r'smg_router_requests_total\{[^}]*\}\s+([0-9.]+)', text)
        if mr:
            out["router_requests_total"] = float(mr.group(1))
        # per-worker active
        for w, ip in (("w1", "172"), ("w2", "152")):
            ma = re.search(
                rf'smg_worker_requests_active\{{worker="http://21\.151\.225\.{ip}:30000"\}}\s+([0-9.]+)',
                text)
            if ma:
                out[f"active_{w}"] = float(ma.group(1))
        return out
    except Exception as e:
        return {"error": str(e)}


def stream_one(prompt: str, max_tokens: int, req_id: int) -> dict:
    """Send one streaming /v1/responses request via router pod, parse SSE timing."""
    payload = json.dumps({
        "model": MODEL,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "stream": True,
    })
    # Token is injected on the host side (pod has no ANTHROPIC_AUTH_TOKEN env).
    # Pass payload via stdin to avoid arg-length / quoting issues.
    cmd = [
        "kubectl", "exec", "-n", KUBE_NS, ROUTER_POD, "-i", "--",
        "sh", "-c",
        "curl -sS -N --max-time 300 "
        f"-H 'Authorization: Bearer {TOKEN}' "
        "-H 'Content-Type: application/json' "
        "-X POST http://localhost:30001/v1/responses "
        "-d @- -w '\\n__CURL_META__\\nhttp_code=%{http_code}\\n"
        "time_total=%{time_total}\\n"
        "time_starttransfer=%{time_starttransfer}\\n"
        "size_download=%{size_download}\\n'",
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    try:
        proc.stdin.write(payload)
        proc.stdin.close()
    except BrokenPipeError:
        pass

    # Real-time SSE parse: timestamp the first reasoning delta (true TTFT) and
    # the first output delta (TTOT), and record inter-delta times for ITL.
    first_reason_t = None      # true TTFT (first reasoning token)
    first_output_t = None      # TTOT (first output token)
    last_delta_t = None
    n_reason = 0
    n_output = 0
    itl_times = []
    meta = {}
    http_code = "?"
    timed_out = False
    deadline = t0 + 320
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.rstrip("\n")
        if line.startswith("__CURL_META__"):
            for ml in proc.stdout:
                ml = ml.strip()
                if "=" in ml:
                    k, v = ml.split("=", 1)
                    meta[k] = v
            break
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            continue
        try:
            evt = json.loads(data)
        except json.JSONDecodeError:
            continue
        etype = evt.get("type", "")
        now = time.perf_counter()
        if etype == "response.reasoning_text.delta" and evt.get("delta"):
            if first_reason_t is None:
                first_reason_t = now
            n_reason += 1
            if last_delta_t:
                itl_times.append(now - last_delta_t)
            last_delta_t = now
        elif etype == "response.output_text.delta" and evt.get("delta"):
            if first_output_t is None:
                first_output_t = now
            n_output += 1
            if last_delta_t:
                itl_times.append(now - last_delta_t)
            last_delta_t = now
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    wall = time.perf_counter() - t0

    http_code = meta.get("http_code", "?")
    n_total = n_reason + n_output
    # True TTFT = time to first reasoning token (reasoning model emits reasoning
    # first). Falls back to first output token if no reasoning.
    ttft = None
    if first_reason_t:
        ttft = first_reason_t - t0
    elif first_output_t:
        ttft = first_output_t - t0
    # decode throughput: tokens produced after TTFT, over (wall - ttft)
    if ttft and n_total > 0:
        decode_secs = max(1e-6, wall - ttft)
        tput = n_total / decode_secs
    else:
        tput = 0.0

    return {
        "req_id": req_id,
        "http_code": http_code,
        "ttft": ttft,
        "latency": float(meta.get("time_total", wall)),
        "n_reason": n_reason,
        "n_output": n_output,
        "n_total": n_total,
        "throughput": tput,
        "size_download": meta.get("size_download"),
        "ok": http_code == "200",
        "error": stderr.strip()[:200] if http_code != "200" else None,
    }


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


def measure_background_rate(duration_s: float = 20.0) -> dict:
    """Sample worker counters over a quiet window to estimate background rate.

    Background traffic on this shared cluster goes to w1 under cache_aware, so
    w2 delta is a clean signal for our rebalanced requests. w1 delta needs this
    baseline subtracted to isolate our traffic.
    """
    a = snapshot_worker_metrics()
    time.sleep(duration_s)
    b = snapshot_worker_metrics()
    rate = {}
    for w in ("w1", "w2"):
        da, db = a.get(w, {}), b.get(w, {})
        r = {}
        for k in ("num_requests_total", "prompt_tokens_total",
                  "generation_tokens_total"):
            if k in da and k in db:
                r[k + "_per_s"] = (db[k] - da[k]) / duration_s
        rate[w] = r
    return rate


def run(concurrency: int, n_req: int, ctx_tokens: int,
        max_tokens: int) -> dict:
    prompt = build_long_context(ctx_tokens)
    print(f"\n=== run: concurrency={concurrency} n_req={n_req} "
          f"ctx_tokens~={ctx_tokens} max_tokens={max_tokens} ===", flush=True)
    print(f"prompt chars={len(prompt)} (~{len(prompt)//4} tokens)", flush=True)

    bg = measure_background_rate(15.0)
    print(f"  background rate: {bg}", flush=True)

    w_before = snapshot_worker_metrics()
    r_before = snapshot_router_metrics()
    t0 = time.perf_counter()

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(stream_one, prompt, max_tokens, i) for i in range(n_req)]
        for fu in as_completed(futs):
            r = fu.result()
            results.append(r)
            tag = "OK" if r.get("ok") else f"FAIL({r.get('http_code')})"
            ttft = f"{r['ttft']:.2f}s" if r.get("ttft") else "n/a"
            print(f"  req {r['req_id']:02d}: {tag} ttft={ttft} "
                  f"lat={r['latency']:.2f}s tok={r['n_total']} "
                  f"tput={r['throughput']:.1f}t/s", flush=True)

    wall = time.perf_counter() - t0
    w_after = snapshot_worker_metrics()
    r_after = snapshot_router_metrics()

    # compute worker deltas
    w_delta = {}
    for w in ("w1", "w2"):
        b = w_before.get(w, {})
        a = w_after.get(w, {})
        d = {}
        for k in ("num_requests_total", "prompt_tokens_total",
                  "generation_tokens_total", "cached_tokens_total"):
            if k in b and k in a:
                d[k] = a[k] - b[k]
        w_delta[w] = d

    oks = [r for r in results if r.get("ok")]
    fails = [r for r in results if not r.get("ok")]
    ttfts = [r["ttft"] for r in oks if r.get("ttft")]
    lats = [r["latency"] for r in oks]
    tputs = [r["throughput"] for r in oks if r["throughput"] > 0]
    toks = [r["n_total"] for r in oks]

    # load split by prompt tokens
    w1_pt = w_delta.get("w1", {}).get("prompt_tokens_total", 0)
    w2_pt = w_delta.get("w2", {}).get("prompt_tokens_total", 0)
    total_pt = w1_pt + w2_pt

    summary = {
        "concurrency": concurrency,
        "n_req": n_req,
        "ctx_tokens_target": ctx_tokens,
        "max_tokens": max_tokens,
        "wall_clock_s": round(wall, 2),
        "n_ok": len(oks),
        "n_fail": len(fails),
        "ttft_s": {
            "p50": round(pct(ttfts, 50), 3) if ttfts else None,
            "p95": round(pct(ttfts, 95), 3) if ttfts else None,
            "mean": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
        },
        "latency_s": {
            "p50": round(pct(lats, 50), 3) if lats else None,
            "p95": round(pct(lats, 95), 3) if lats else None,
            "mean": round(sum(lats) / len(lats), 3) if lats else None,
        },
        "throughput_tps": {
            "mean": round(sum(tputs) / len(tputs), 2) if tputs else None,
            "p50": round(pct(tputs, 50), 2) if tputs else None,
        },
        "output_tokens": {
            "mean": round(sum(toks) / len(toks), 1) if toks else None,
            "total": sum(toks),
        },
        "worker_delta": w_delta,
        "load_split_by_prompt_tokens": {
            "w1": round(w1_pt, 0),
            "w2": round(w2_pt, 0),
            "w1_pct": round(100 * w1_pt / total_pt, 1) if total_pt else None,
            "w2_pct": round(100 * w2_pt / total_pt, 1) if total_pt else None,
        },
        "router_delta": {
            "selection": (r_after.get("selection_total", 0) -
                          r_before.get("selection_total", 0)),
            "requests": (r_after.get("router_requests_total", 0) -
                         r_before.get("router_requests_total", 0)),
        },
        "background_rate": bg,
        "background_corrected_w1_reqs": (
            w_delta.get("w1", {}).get("num_requests_total", 0) -
            bg.get("w1", {}).get("num_requests_total_per_s", 0) * wall
        ),
        "failures": [{"req_id": r["req_id"], "http_code": r.get("http_code"),
                      "error": r.get("error")} for r in fails],
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--n-per-level", type=int, default=8,
                    help="requests per concurrency level")
    ap.add_argument("--ctx-tokens", type=int, default=12000,
                    help="target prefill tokens per request")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="max output tokens per request")
    ap.add_argument("--out", default="results/long-agent-context-bench.json")
    args = ap.parse_args()

    all_results = []
    for c in args.concurrency:
        n = min(args.n_per_level, max(c, args.n_per_level))
        # for low concurrency, run n_per_level requests; for high, scale a bit
        n = args.n_per_level
        s = run(c, n, args.ctx_tokens, args.max_tokens)
        all_results.append(s)
        print(f"\n--- summary c={c} ---")
        print(json.dumps(s, indent=2, ensure_ascii=False))

    out = {
        "ts": "captured",
        "ctx_tokens_target": args.ctx_tokens,
        "max_tokens": args.max_tokens,
        "n_per_level": args.n_per_level,
        "runs": all_results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n=== written to {args.out} ===")


if __name__ == "__main__":
    main()
