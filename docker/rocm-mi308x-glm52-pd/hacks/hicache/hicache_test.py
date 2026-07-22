#!/usr/bin/env python3
"""hicache repeated-prefix validation.

Sends the SAME ~12K-token prompt N times sequentially. After each request,
captures hicache_host_used_tokens + cache_hit_rate + TTFT. Used to prove
whether write_back activates L2 (host) under low load, and whether
write_through_selective does. Run inside the worker pod via kubectl exec.
"""
import argparse
import json
import time
import urllib.error
import urllib.request

URL = "http://localhost:30000/v1/chat/completions"
METRICS = "http://localhost:30000/metrics"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"

BLOCK = (
    "user: Analyze the following system metrics and propose a remediation plan. "
    "Here is the transcript from the last investigation turn.\n"
    "assistant: Understood. I will review the metrics, identify the bottleneck, "
    "and propose a step-by-step remediation. Let me gather telemetry.\n"
    "tool_call: query_prometheus(metric='sglang:num_requests_running', range='5m')\n"
    "tool_result: peak=18, p95=15, mean=11.2 across workers; worker-1 carries "
    "0.88 of the load under cache_aware routing.\n"
    "tool_call: query_prometheus(metric='sglang:cache_hit_rate', range='5m')\n"
    "tool_result: worker-1=0.62, worker-2=0.04; prefix cache concentrated on "
    "worker-1 because the shared system prompt only resides there.\n"
    "assistant: The imbalance is caused by cache_aware sticking to the worker "
    "that holds the prefix. Balance thresholds should trigger a spill once the "
    "active-request gap exceeds the configured abs/rel thresholds.\n"
)


def build_context(target_tokens):
    header = (
        "You are a senior SRE agent investigating a production incident on the "
        "GLM-5.2 inference cluster. Below is accumulated context from prior "
        "turns. Continue the investigation and answer the final question "
        "concisely.\n\n"
    )
    per_block = 270
    n = max(1, target_tokens // per_block)
    blocks = "\n".join(f"--- turn {i:04d} ---\n{BLOCK}" for i in range(n))
    tail = (
        "\n\nFinal question: based on the full transcript above, in one sentence, "
        "what is the single highest-impact action to restore load balance?"
    )
    return header + blocks + tail


def fetch_metrics():
    req = urllib.request.Request(METRICS, headers={
        "Authorization": f"Bearer {API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception as e:
        return {"err": str(e)[:120]}
    out = {}
    for line in text.splitlines():
        if not line.startswith("sglang:") or "{" not in line:
            continue
        name = line.split("{", 1)[0]
        if " " in line:
            val = line.rsplit(" ", 1)[-1]
        else:
            val = ""
        if name in ("sglang:hicache_host_used_tokens",
                    "sglang:hicache_host_total_tokens",
                    "sglang:cache_hit_rate",
                    "sglang:hicache_gpu_used_tokens"):
            try:
                out[name] = float(val)
            except ValueError:
                pass
    return out


def one_request(prompt, max_tokens):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        URL, data=payload, method="POST",
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = 0.0
    tok = 0
    ok = False
    err = None
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            first = None
            for raw in resp:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if first is None:
                    ch0 = evt.get("choices")
                    if ch0 and (ch0[0].get("delta", {}).get("content")
                                or ch0[0].get("delta", {}).get("reasoning_content")):
                        first = time.perf_counter()
                        ttft = first - t0
                u = evt.get("usage")
                if u and u.get("completion_tokens") is not None:
                    tok = (u.get("completion_tokens", 0) or 0) + (u.get("reasoning_tokens", 0) or 0)
            ok = True
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}"
    except Exception as e:
        err = type(e).__name__ + ":" + str(e)[:100]
    wall = time.perf_counter() - t0
    return {"ok": ok, "ttft": ttft, "tok": tok, "wall": wall, "err": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-tokens", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    prompt = build_context(args.ctx_tokens)
    print(f"hicache_test: ctx~={args.ctx_tokens} max_tokens={args.max_tokens} "
          f"repeats={args.repeats} prompt_chars={len(prompt)}", flush=True)

    # Baseline metrics before any request
    m0 = fetch_metrics()
    print(f"[pre] host_used={m0.get('sglang:hicache_host_used_tokens')} "
          f"gpu_used={m0.get('sglang:hicache_gpu_used_tokens')} "
          f"cache_hit_rate={m0.get('sglang:cache_hit_rate')}", flush=True)

    for i in range(args.repeats):
        r = one_request(prompt, args.max_tokens)
        m = fetch_metrics()
        print(f"[rep {i+1}/{args.repeats}] ok={r['ok']} tok={r['tok']} "
              f"ttft={r['ttft']:.3f}s wall={r['wall']:.2f}s "
              f"host_used={m.get('sglang:hicache_host_used_tokens')} "
              f"gpu_used={m.get('sglang:hicache_gpu_used_tokens')} "
              f"cache_hit_rate={m.get('sglang:cache_hit_rate')} "
              f"err={r['err']}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
