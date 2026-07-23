#!/usr/bin/env python3
"""Benchmark Codex (Responses API streaming) latency through the gateway.

Measures:
- TTFT (time to first token)
- ITL (inter-token latency, mean)
- Total latency
- Tokens generated

Runs both via curl (raw Responses API) and via codex CLI (end-to-end).
"""
import json, subprocess, time, statistics, sys, re

GATEWAY = "https://glm52-2tp8.jmpti.woa.com"
TOKEN = "${ANTHROPIC_AUTH_TOKEN}"
MODEL = "glm-5.2"

def curl_stream(prompt, max_tokens=200):
    """Stream a Responses API request, return timing + tokens."""
    body = json.dumps({
        "model": MODEL,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "stream": True,
    })
    start = time.perf_counter()
    proc = subprocess.Popen([
        "/usr/bin/curl", "-sS", "-N", "--max-time", "120",
        f"{GATEWAY}/v1/responses",
        "-H", f"Authorization: Bearer {TOKEN}",
        "-H", "Content-Type: application/json",
        "-d", body,
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    first_token_time = None
    token_times = []
    output_tokens = 0
    reasoning_tokens = 0
    final_usage = None

    for line in proc.stdout:
        line = line.rstrip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            evt = json.loads(data)
        except json.JSONDecodeError:
            continue
        etype = evt.get("type", "")
        now = time.perf_counter()
        if etype in ("response.output_text.delta", "response.reasoning_text.delta",
                     "response.function_call_arguments.delta"):
            text = evt.get("text", "") or evt.get("arguments", "")
            if text:
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now)
                if "reasoning" in etype:
                    reasoning_tokens += 1
                else:
                    output_tokens += 1
        elif etype == "response.completed":
            r = evt.get("response", {})
            final_usage = r.get("usage")

    proc.wait()
    total = time.perf_counter() - start

    ttft = (first_token_time - start) if first_token_time else None
    itls = [(token_times[i+1] - token_times[i]) for i in range(len(token_times)-1)] if len(token_times) > 1 else []
    return {
        "total_s": round(total, 3),
        "ttft_ms": round(ttft * 1000, 1) if ttft else None,
        "itl_mean_ms": round(statistics.mean(itls) * 1000, 1) if itls else None,
        "itl_p50_ms": round(statistics.median(itls) * 1000, 1) if itls else None,
        "itl_p95_ms": round(sorted(itls)[int(len(itls)*0.95)] * 1000, 1) if itls else None,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "usage": final_usage,
    }

def codex_cli(prompt):
    """Run codex exec CLI end-to-end, measure wall-clock."""
    start = time.perf_counter()
    proc = subprocess.run([
        "/opt/homebrew/bin/codex", "exec",
        "--dangerously-bypass-approvals-and-sandbox", "-",
    ], input=prompt, capture_output=True, text=True, timeout=180)
    total = time.perf_counter() - start
    # Extract last message (codex prints "PONG" etc at end)
    out = proc.stdout
    return {"total_s": round(total, 3), "stdout_tail": out[-200:], "rc": proc.returncode}

print("=" * 78)
print("Codex (Responses API) Latency Benchmark")
print(f"Gateway: {GATEWAY}")
print("=" * 78)

# Warmup
print("\nWarmup...")
r = curl_stream("hi", max_tokens=10)
print(f"  warmup: {r['total_s']}s, ttft={r['ttft_ms']}ms")

# Test scenarios
scenarios = [
    ("short (5 tok)",       "Reply with: hello world",                            20),
    ("medium (50 tok)",     "List 10 fruits, one per line.",                      150),
    ("long (200 tok)",      "Write a short paragraph (about 100 words) about the ocean.", 400),
    ("reasoning-heavy",     "If a train leaves Beijing at 3pm at 60km/h and another leaves Shanghai at 4pm at 80km/h heading toward Beijing (1318km apart), when do they meet? Explain.", 600),
    ("code generation",     "Write a Python function that checks if a string is a palindrome. Just the code, no explanation.", 400),
]

print(f"\n{'scenario':<22} {'total(s)':<10} {'ttft(ms)':<10} {'itl_mean':<10} {'itl_p95':<10} {'out_tok':<8} {'reason_tok':<10}")
print("-" * 90)
results = []
for name, prompt, max_tok in scenarios:
    r = curl_stream(prompt, max_tokens=max_tok)
    results.append((name, r))
    print(f"{name:<22} {r['total_s']:<10} {str(r['ttft_ms']):<10} {str(r['itl_mean_ms']):<10} {str(r['itl_p95_ms']):<10} {r['output_tokens']:<8} {r['reasoning_tokens']:<10}")

# Codex CLI end-to-end
print("\n--- Codex CLI end-to-end (wall clock) ---")
cli_scenarios = [
    ("cli short",  "Reply with exactly: PONG"),
    ("cli medium", "What is 7+5? Reply with just the number."),
    ("cli code",   "Write a one-line Python lambda to reverse a string. Just the code."),
]
for name, prompt in cli_scenarios:
    r = codex_cli(prompt)
    print(f"{name:<22} {r['total_s']}s  rc={r['rc']}  tail={r['stdout_tail'][-60:]!r}")

# Summary
print("\n" + "=" * 78)
print("Summary (Responses API streaming via curl)")
print("=" * 78)
totals = [r["total_s"] for _, r in results]
ttfts = [r["ttft_ms"] for _, r in results if r["ttft_ms"]]
itls = [r["itl_mean_ms"] for _, r in results if r["itl_mean_ms"]]
print(f"  total:   min={min(totals):.2f}s  max={max(totals):.2f}s  mean={statistics.mean(totals):.2f}s")
print(f"  TTFT:    min={min(ttfts):.0f}ms  max={max(ttfts):.0f}ms  mean={statistics.mean(ttfts):.0f}ms")
print(f"  ITL:     min={min(itls):.0f}ms  max={max(itls):.0f}ms  mean={statistics.mean(itls):.0f}ms")
