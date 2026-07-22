#!/usr/bin/env python3
"""Benchmark GLM-5.2 1tp8 with v17 config — protocol comparison & concurrency."""
import subprocess, json, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://glm52-1tp8.jmpti.woa.com"
KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"

def curl(url, payload, stream=False):
    cmd = [
        "curl", "-s", "-w", "\n%{http_code}\n%{time_total}",
        "-X", "POST",
        "-H", f"Authorization: Bearer {KEY}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload),
        f"{BASE}{url}"
    ]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    wall = time.time() - start
    parts = result.stdout.rsplit("\n", 2)
    if len(parts) >= 3:
        body, code, curl_time = parts[0], parts[1], parts[2]
        try:
            d = json.loads(body)
            usage = d.get("usage", {})
            return {
                "http": code,
                "wall": wall,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": usage.get("reasoning_tokens", 0),
            }
        except:
            return {"http": code, "wall": wall, "error": body[:200]}
    return {"http": "ERR", "wall": wall, "error": result.stdout[:200]}

def bench_single(label, url, payload):
    r = curl(url, payload)
    tpot = r["wall"] / max(r["completion_tokens"], 1) if "completion_tokens" in r else 0
    print(f"  {label:45s} | HTTP={r['http']} | wall={r['wall']:.2f}s | "
          f"tokens={r.get('completion_tokens',0)} | tpot={tpot:.3f}s | "
          f"reasoning={r.get('reasoning_tokens',0)}")
    return r

def bench_concurrent(label, url, payload, n):
    results = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(curl, url, payload) for _ in range(n)]
        start = time.time()
        for f in as_completed(futures):
            results.append(f.result())
        wall = time.time() - start
    ok = [r for r in results if r.get("http") == "200"]
    total_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    print(f"  {label:45s} | N={n} | ok={len(ok)}/{n} | wall={wall:.2f}s | "
          f"tokens={total_tokens} | tput={total_tokens/wall:.1f} tok/s | "
          f"avg_lat={sum(r['wall'] for r in ok)/max(len(ok),1):.2f}s")
    return results

print("=" * 100)
print("GLM-5.2 1tp8 — v17 Config Benchmark")
print("=" * 100)

# --- Single request tests ---
print("\n--- Single Request Tests ---")
short = {"model":"glm-5.2","messages":[{"role":"user","content":"What is 2+3? Just the number."}],"max_tokens":100,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
medium = {"model":"glm-5.2","messages":[{"role":"user","content":"Explain how a hash map works in 3 sentences."}],"max_tokens":300,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
long_prompt = "Summarize the key concepts of distributed systems. " * 50
longp = {"model":"glm-5.2","messages":[{"role":"user","content":long_prompt}],"max_tokens":500,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}

for label, url, payload in [
    ("chat/completions (short, no-think)", "/v1/chat/completions", short),
    ("chat/completions (medium, no-think)", "/v1/chat/completions", medium),
    ("chat/completions (~500tok prompt, no-think)", "/v1/chat/completions", longp),
    ("/v1/messages (short, no-think)", "/v1/messages", short),
    ("/v1/responses (short, no-think)", "/v1/responses", {"model":"glm-5.2","input":"What is 2+3? Just the number.","max_output_tokens":100,"temperature":0}),
]:
    bench_single(label, url, payload)

# --- Reasoning tests ---
print("\n--- Reasoning Tests ---")
for effort in ["low", "high", "max"]:
    payload = {"model":"glm-5.2","messages":[{"role":"user","content":"What is 15*17? Show reasoning."}],"max_tokens":600,"temperature":0,"reasoning_effort":effort}
    bench_single(f"chat/completions (reasoning={effort})", "/v1/chat/completions", payload)

# --- Concurrency tests ---
print("\n--- Concurrency Tests (no-think, short) ---")
for n in [1, 5, 10, 20]:
    bench_concurrent(f"chat/completions concurrent={n}", "/v1/chat/completions", short, n)

print("\n--- Concurrency: Protocol Comparison (N=10) ---")
for label, url, payload in [
    ("/v1/chat/completions x10", "/v1/chat/completions", short),
    ("/v1/messages x10", "/v1/messages", short),
    ("/v1/responses x10", "/v1/responses", {"model":"glm-5.2","input":"What is 2+3? Just the number.","max_output_tokens":100,"temperature":0}),
]:
    bench_concurrent(label, url, payload, 10)

# --- Medium prompt concurrency ---
print("\n--- Concurrency: Medium Prompt (N=10) ---")
bench_concurrent("chat/completions medium x10", "/v1/chat/completions", medium, 10)

print("\n" + "=" * 100)
print("Benchmark complete.")
