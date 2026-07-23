#!/usr/bin/env python3
"""
Controlled decode-only workload for rocprof profiling of tune19-sglang-0.

Run INSIDE the pod via kubectl exec, hitting localhost:30000 directly (no
router). Sends a fixed decode-heavy load: N concurrent streaming requests,
~12K input tokens, max_tokens 256, temperature 0, identical prompt/seed.
Loops for a wall-clock duration so the decode phase is sustained while
rocprof is attached.

Stdlib only (urllib + threading) so it runs without extra deps in the pod.

Usage:
    python3 prof19_decode_workload.py --duration 60 --concurrency 8 \
        --ctx-tokens 12000 --max-tokens 256
"""
import argparse
import json
import sys
import threading
import time
import urllib.request

API = "http://localhost:30000/v1/chat/completions"
KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"

# ~270 tokens per block (measured). Repeated varied blocks give a long,
# prefix-cacheable prefill (~target tokens) without byte-identical dedup.
BLOCK = (
    "user: Analyze the following system metrics and propose a remediation plan. "
    "Here is the transcript from the last investigation turn.\n"
    "assistant: Understood. I will review the metrics, identify the bottleneck, "
    "and propose a step-by-step remediation. Let me first gather the relevant "
    "telemetry from the monitoring stack.\n"
    "tool_call: query_prometheus(metric='sglang:num_requests_running', range='5m')\n"
    "tool_result: peak=18, p95=15, mean=11.2 across both workers; worker-1 carries "
    "0.88 of the load under cache_aware routing.\n"
    "tool_call: query_prometheus(metric='sglang:cache_hit_rate', range='5m')\n"
    "tool_result: worker-1=0.62, worker-2=0.04; prefix cache is concentrated on "
    "worker-1 because the shared system prompt only resides there.\n"
    "assistant: The imbalance is caused by cache_aware sticking to the worker "
    "that holds the prefix. The balance thresholds should trigger a spill once "
    "the active-request gap exceeds the configured abs/rel thresholds.\n"
)


def build_prompt(target_tokens: int) -> str:
    header = (
        "You are a senior SRE agent. You have been investigating a production "
        "incident on the GLM-5.2 inference cluster. Below is the accumulated "
        "context from prior turns. Continue the investigation and answer the "
        "final question concisely.\n\n"
    )
    per_block = 270
    n = max(1, target_tokens // per_block)
    blocks = [f"--- turn {i:04d} ---\n" + BLOCK for i in range(n)]
    tail = (
        "\n\nFinal question: based on the full transcript above, in one sentence, "
        "what is the single highest-impact action to restore load balance?"
    )
    return header + "\n".join(blocks) + tail


def stream_one(prompt, max_tokens, stats, lock, sem):
    """One streaming request. Holds the concurrency semaphore for its lifetime."""
    try:
        payload = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            API, data=payload, method="POST",
            headers={"Authorization": f"Bearer {KEY}",
                     "Content-Type": "application/json"},
        )
        n_tok = 0
        ok = False
        err = None
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        evt = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    ch = evt.get("choices", [{}])[0]
                    delta = ch.get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content"):
                        n_tok += 1
                    if ch.get("finish_reason"):
                        ok = True
        except Exception as e:
            err = str(e)[:200]
        dt = time.perf_counter() - t0
        with lock:
            stats["done"] += 1
            stats["tokens"] += n_tok
            stats["ok"] += 1 if ok else 0
            stats["fail"] += 0 if ok else 1
            if err:
                stats["errors"].append(err)
        sys.stderr.write(
            f"  req done: tok={n_tok} dt={dt:.2f}s ok={ok}"
            f"{' err=' + err if err else ''}\n"
        )
        sys.stderr.flush()
    finally:
        sem.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--ctx-tokens", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    prompt = build_prompt(args.ctx_tokens)
    print(f"prompt chars={len(prompt)} (~{len(prompt)//4} tokens)", flush=True)
    print(f"duration={args.duration}s concurrency={args.concurrency} "
          f"max_tokens={args.max_tokens}", flush=True)

    stats = {"done": 0, "tokens": 0, "ok": 0, "fail": 0, "errors": []}
    lock = threading.Lock()
    deadline = time.perf_counter() + args.duration
    sem = threading.Semaphore(args.concurrency)

    # Producer: while before deadline, acquire a slot (blocks at concurrency),
    # then spawn a detached worker thread that releases the slot on completion.
    def producer():
        while time.perf_counter() < deadline:
            sem.acquire()
            if time.perf_counter() >= deadline:
                sem.release()
                break
            t = threading.Thread(
                target=stream_one,
                args=(prompt, args.max_tokens, stats, lock, sem),
                daemon=True,
            )
            t.start()

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()
    producer_thread.join()

    # Drain in-flight requests so the decode tail is captured.
    for _ in range(args.concurrency):
        sem.acquire()
    print("\n=== workload done ===", flush=True)
    print(json.dumps({
        "duration_s": args.duration,
        "concurrency": args.concurrency,
        "requests_completed": stats["done"],
        "tokens_decoded": stats["tokens"],
        "ok": stats["ok"],
        "fail": stats["fail"],
        "decode_tokens_per_s": round(stats["tokens"] / args.duration, 1),
        "errors": stats["errors"][:5],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
