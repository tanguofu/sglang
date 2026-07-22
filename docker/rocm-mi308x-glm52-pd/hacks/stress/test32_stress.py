#!/usr/bin/env python3
"""EAGLE+TP8 deadlock stress test — direct to worker localhost:30000.

Sustained concurrent long-context streaming load to try to reproduce the
TP8 collective deadlock (sglang PR #31478). Prints one progress line per
wave. Run inside the worker pod via kubectl exec.
"""
import argparse
import json
import socket
import sys
import threading
import time
import urllib.request
import urllib.error

URL = "http://localhost:30000/v1/chat/completions"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"

# A long agent-style context block (~270 tokens). Repeated to reach target.
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


def one_request(prompt, max_tokens, req_id, results, lock):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        URL, data=payload, method="POST",
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    rec = {"req_id": req_id, "ok": False, "http": None, "tok": 0,
           "lat": 0.0, "err": None}
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            rec["http"] = resp.status
            ntok = 0
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
                    first = time.perf_counter()
                ch = evt.get("choices")
                if ch:
                    delta = ch[0].get("delta", {})
                    c = delta.get("content")
                    if c:
                        ntok += 1
                    r = delta.get("reasoning_content")
                    if r:
                        ntok += 1
            rec["tok"] = ntok
            rec["ok"] = (resp.status == 200)
            rec["lat"] = time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        rec["http"] = e.code
        rec["err"] = f"HTTPError {e.code}"
        rec["lat"] = time.perf_counter() - t0
    except Exception as e:
        rec["err"] = type(e).__name__ + ":" + str(e)[:120]
        rec["lat"] = time.perf_counter() - t0
    with lock:
        results.append(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=28)
    ap.add_argument("--ctx-tokens", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--duration", type=int, default=1100,
                    help="total seconds to keep loading")
    ap.add_argument("--wave-gap", type=float, default=0.5)
    args = ap.parse_args()

    prompt = build_context(args.ctx_tokens)
    print(f"stress: url={URL} concurrency={args.concurrency} "
          f"ctx~={args.ctx_tokens} max_tokens={args.max_tokens} "
          f"duration={args.duration}s prompt_chars={len(prompt)}", flush=True)

    deadline = time.perf_counter() + args.duration
    wave = 0
    total_ok = 0
    total_fail = 0
    total_tok = 0
    while time.perf_counter() < deadline:
        wave += 1
        results = []
        lock = threading.Lock()
        threads = []
        t0 = time.perf_counter()
        for i in range(args.concurrency):
            t = threading.Thread(
                target=one_request,
                args=(prompt, args.max_tokens, i, results, lock))
            t.start()
            threads.append(t)
        # Wait for all in this wave (cap wait at 320s)
        for t in threads:
            t.join(timeout=320)
        wave_dur = time.perf_counter() - t0
        ok = sum(1 for r in results if r.get("ok"))
        fail = len(results) - ok
        toks = sum(r.get("tok", 0) for r in results)
        total_ok += ok
        total_fail += fail
        total_tok += toks
        errs = {}
        for r in results:
            if not r.get("ok") and r.get("err"):
                errs[r["err"]] = errs.get(r["err"], 0) + 1
        print(f"[wave {wave:03d} t={time.perf_counter()-deadline+args.duration:.0f}s] "
              f"ok={ok} fail={fail} tok={toks} dur={wave_dur:.1f}s "
              f"cum_ok={total_ok} cum_fail={total_fail} errs={errs}", flush=True)
        # brief gap between waves
        time.sleep(args.wave_gap)
    print(f"DONE total_ok={total_ok} total_fail={total_fail} total_tok={total_tok}",
          flush=True)


if __name__ == "__main__":
    main()
