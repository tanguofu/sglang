#!/usr/bin/env python3
"""Benchmark Codex (Responses API streaming) — corrected for GLM-5.2 reasoning model.

GLM-5.2 always emits reasoning_text before output_text. We measure:
- TTFT (first reasoning token — what client perceives as "thinking started")
- TTOT (first output token — what client perceives as "answer started")
- ITL (inter-token latency across all deltas)
- Reasoning vs output token split
- Total wall clock
"""
import json, subprocess, time, statistics, sys

GATEWAY = "https://glm52-2tp8.jmpti.woa.com"
TOKEN = "${ANTHROPIC_AUTH_TOKEN}"
MODEL = "glm-5.2"

def stream_request(prompt, max_tokens=400):
    body = json.dumps({
        "model": MODEL,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "stream": True,
    })
    start = time.perf_counter()
    proc = subprocess.Popen([
        "/usr/bin/curl", "-sS", "-N", "--max-time", "180",
        f"{GATEWAY}/v1/responses",
        "-H", f"Authorization: Bearer {TOKEN}",
        "-H", "Content-Type: application/json",
        "-d", body,
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    first_reason_time = None
    first_output_time = None
    delta_times = []
    reasoning_tokens = 0
    output_tokens = 0
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
        if etype == "response.reasoning_text.delta":
            if evt.get("delta"):
                if first_reason_time is None:
                    first_reason_time = now
                delta_times.append(now)
                reasoning_tokens += 1
        elif etype == "response.output_text.delta":
            if evt.get("delta"):
                if first_output_time is None:
                    first_output_time = now
                delta_times.append(now)
                output_tokens += 1
        elif etype == "response.function_call_arguments.delta":
            if evt.get("delta"):
                if first_output_time is None:
                    first_output_time = now
                delta_times.append(now)
                output_tokens += 1
        elif etype == "response.completed":
            r = evt.get("response", {})
            final_usage = r.get("usage")

    proc.wait()
    total = time.perf_counter() - start

    itls = [(delta_times[i+1] - delta_times[i]) for i in range(len(delta_times)-1)] if len(delta_times) > 1 else []
    total_tokens = reasoning_tokens + output_tokens
    return {
        "total_s": round(total, 3),
        "ttft_reason_ms": round((first_reason_time - start) * 1000, 1) if first_reason_time else None,
        "ttft_output_ms": round((first_output_time - start) * 1000, 1) if first_output_time else None,
        "itl_mean_ms": round(statistics.mean(itls) * 1000, 1) if itls else None,
        "itl_p50_ms": round(statistics.median(itls) * 1000, 1) if itls else None,
        "itl_p95_ms": round(sorted(itls)[int(len(itls)*0.95)] * 1000, 1) if itls else None,
        "reason_tokens": reasoning_tokens,
        "output_tokens": output_tokens,
        "total_gen_tokens": total_tokens,
        "tps": round(total_tokens / total, 1) if total > 0 else None,
        "usage": final_usage,
    }

print("=" * 90)
print("Codex (Responses API) Latency Benchmark — GLM-5.2")
print(f"Gateway: {GATEWAY}")
print("=" * 90)

# Warmup
print("\nWarmup...", end=" ", flush=True)
r = stream_request("hi", max_tokens=30)
print(f"{r['total_s']}s")

scenarios = [
    ("short",            "Reply with: hello world",                                           100),
    ("medium",           "List 10 fruits, one per line.",                                     200),
    ("long",             "Write a short paragraph (about 80 words) about the ocean.",         400),
    ("reasoning",        "If a train leaves Beijing at 3pm at 60km/h and another leaves Shanghai at 4pm at 80km/h toward Beijing (1318km apart), when do they meet?", 800),
    ("code",             "Write a Python function that checks if a string is a palindrome. Just the code, no explanation.", 500),
    ("tool_use",         "What's the weather in Paris? Use the get_weather tool.",            400),
    ("multi_turn_short", "User: my name is bob\nAssistant: hi bob\nUser: what is my name?",   200),
]

print(f"\n{'scenario':<20} {'total(s)':<9} {'ttft_r':<9} {'ttft_o':<9} {'itl_mean':<9} {'itl_p95':<9} {'reason':<7} {'out':<6} {'tps':<6}")
print("-" * 100)
results = []
for name, prompt, max_tok in scenarios:
    r = stream_request(prompt, max_tokens=max_tok)
    results.append((name, r))
    print(f"{name:<20} {r['total_s']:<9} {str(r['ttft_reason_ms']):<9} {str(r['ttft_output_ms']):<9} {str(r['itl_mean_ms']):<9} {str(r['itl_p95_ms']):<9} {r['reason_tokens']:<7} {r['output_tokens']:<6} {str(r['tps']):<6}")

print("\n" + "=" * 90)
print("Summary")
print("=" * 90)
totals = [r["total_s"] for _, r in results]
ttfts_r = [r["ttft_reason_ms"] for _, r in results if r["ttft_reason_ms"]]
ttfts_o = [r["ttft_output_ms"] for _, r in results if r["ttft_output_ms"]]
itls = [r["itl_mean_ms"] for _, r in results if r["itl_mean_ms"]]
tps = [r["tps"] for _, r in results if r["tps"]]
print(f"  total:     min={min(totals):.2f}s  max={max(totals):.2f}s  mean={statistics.mean(totals):.2f}s")
print(f"  TTFT-reason: min={min(ttfts_r):.0f}ms  max={max(ttfts_r):.0f}ms  mean={statistics.mean(ttfts_r):.0f}ms")
print(f"  TTFT-output: min={min(ttfts_o):.0f}ms  max={max(ttfts_o):.0f}ms  mean={statistics.mean(ttfts_o):.0f}ms")
print(f"  ITL mean:  min={min(itls):.0f}ms  max={max(itls):.0f}ms  mean={statistics.mean(itls):.0f}ms")
print(f"  TPS:       min={min(tps):.0f}  max={max(tps):.0f}  mean={statistics.mean(tps):.0f} tok/s")
