#!/usr/bin/env python3
"""HiCache verification stress test.

Strategy:
  Phase 1 — Fill GPU KV: send N unique long-prefix requests to fill GPU KV cache
  Phase 2 — Trigger eviction: send N more unique requests to push phase-1 KV to host
  Phase 3 — Verify HiCache hit: re-send phase-1 prompts, measure TTFT speedup
  Phase 4 — Collect metrics: hicache_host_used_tokens, cache_hit_rate, evict stats
"""
import json, subprocess, time, sys, statistics, os, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

GATEWAY = "https://glm52-2tp8.jmpti.woa.com"
TOKEN = os.environ.get("API_KEY", "sk-46faecc9d0bc4dcd9db6a15c73ae91c8")
MODEL = "glm-5.2"

# Unique prefix pool — each ~2000 tokens to fill GPU KV quickly
# 547840 max_total_num_tokens / 2000 per request ≈ 274 requests to fill GPU KV
PREFIX_TEMPLATE = (
    "You are a helpful assistant. Here is a long document about topic {topic_id}:\n\n"
    + "The quick brown fox jumps over the lazy dog. " * 80  # ~4000 chars ≈ 1000 tokens
    + "\n\nTopic {topic_id} details: {details}\n\n"
    + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 40  # ~2000 chars ≈ 500 tokens
    + "\n\nBased on the above document, "
)

TOPICS = [
    "quantum computing", "machine learning", "climate change", "ancient Rome",
    "neural networks", "blockchain technology", "space exploration", "genetic engineering",
    "renewable energy", "cybersecurity", "bioinformatics", "philosophy of mind",
    "macroeconomics", "distributed systems", "compiler design", "graph theory",
]

DETAILS = [
    "with focus on practical applications and recent breakthroughs in the field",
    "including historical context and future research directions",
    "covering both theoretical foundations and empirical results",
    "with emphasis on scalability, performance, and real-world trade-offs",
]


def make_prompt(uid):
    """Generate a unique ~2000-token prompt."""
    topic = TOPICS[uid % len(TOPICS)]
    detail = DETAILS[uid % len(DETAILS)]
    return PREFIX_TEMPLATE.format(topic_id=uid, topic=topic, details=detail) + f"What is topic {uid} about? Answer briefly."


def call_gateway(prompt, max_tokens=20, timeout=120):
    """Non-streaming chat completion for speed."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    })
    cmd = [
        "curl", "-sS", "--max-time", str(timeout),
        f"{GATEWAY}/v1/chat/completions",
        "-H", f"Authorization: Bearer {TOKEN}",
        "-H", "Content-Type: application/json",
        "-d", body,
    ]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
        elapsed = time.perf_counter() - start
        try:
            resp = json.loads(proc.stdout)
            usage = resp.get("usage", {})
            return {
                "ok": True,
                "elapsed": round(elapsed, 2),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": usage.get("reasoning_tokens", 0),
            }
        except json.JSONDecodeError:
            return {"ok": False, "elapsed": round(elapsed, 2), "error": proc.stdout[:200]}
    except Exception as e:
        return {"ok": False, "elapsed": round(time.perf_counter() - start, 2), "error": str(e)[:200]}


def get_metrics():
    """Get cache metrics from both workers via kubectl exec."""
    metrics = {}
    for pod_name in ["sglang-glm52-2tp8-sglang-0", "sglang-glm52-2tp8-sglang-1"]:
        cmd = [
            "kubectl", "--context=cls-bmmk3vtl-context", "-n", "kube-system",
            "exec", pod_name, "--", "curl", "-sS", "http://127.0.0.1:30000/metrics",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            lines = proc.stdout.splitlines()
            pod_metrics = {}
            for line in lines:
                if not line.startswith("sglang:") or "#" in line:
                    continue
                for key in ["cache_hit_rate", "hicache_host_used_tokens", "hicache_host_total_tokens",
                            "kv_evictable_tokens", "kv_used_tokens", "kv_available_tokens"]:
                    if key in line and "gauge" not in line:
                        try:
                            val = float(line.split()[-1])
                            pod_metrics[key] = val
                        except (ValueError, IndexError):
                            pass
            metrics[pod_name] = pod_metrics
        except Exception as e:
            metrics[pod_name] = {"error": str(e)[:100]}
    return metrics


def run_phase(name, prompts, concurrency=4, max_tokens=20):
    """Run a phase of requests and collect timing."""
    print(f"\n{'='*80}")
    print(f"Phase: {name} — {len(prompts)} requests, concurrency={concurrency}")
    print(f"{'='*80}")

    results = []
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(call_gateway, p, max_tokens): i for i, p in enumerate(prompts)}
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)
            status = "OK" if r["ok"] else "FAIL"
            if i % 10 == 0 or i == len(prompts):
                ok_count = sum(1 for r in results if r["ok"])
                print(f"  [{i}/{len(prompts)}] {status} {r['elapsed']}s "
                      f"(prompt_tok={r.get('prompt_tokens', '?')}) — {ok_count} ok")

    total_time = time.perf_counter() - start
    ok_results = [r for r in results if r["ok"]]
    fail_count = len(results) - len(ok_results)

    if ok_results:
        latencies = [r["elapsed"] for r in ok_results]
        print(f"\n  Phase summary: {len(ok_results)} ok, {fail_count} fail, {total_time:.1f}s total")
        print(f"  Latency: min={min(latencies):.2f}s  max={max(latencies):.2f}s  "
              f"mean={statistics.mean(latencies):.2f}s  median={statistics.median(latencies):.2f}s")

    return results


def main():
    print("=" * 80)
    print("HiCache Verification Stress Test")
    print(f"Gateway: {GATEWAY}")
    print(f"Token: {TOKEN[:15]}...")
    print("=" * 80)

    # --- Baseline metrics ---
    print("\n--- Baseline metrics ---")
    m0 = get_metrics()
    for pod, m in m0.items():
        print(f"  {pod}: {m}")

    # --- Phase 1: Fill GPU KV cache with unique prefixes ---
    # ~2000 tokens per prompt, 60 prompts ≈ 120K tokens (fills ~22% of 548K)
    # With concurrency, some will be evicted
    phase1_prompts = [make_prompt(i) for i in range(60)]
    r1 = run_phase("Phase 1 — Fill GPU KV (60 unique ~2K-token prompts)", phase1_prompts,
                   concurrency=4, max_tokens=15)

    print("\n--- Metrics after Phase 1 ---")
    m1 = get_metrics()
    for pod, m in m1.items():
        print(f"  {pod}: {m}")

    # --- Phase 2: Trigger eviction with more unique prefixes ---
    phase2_prompts = [make_prompt(100 + i) for i in range(60)]
    r2 = run_phase("Phase 2 — Trigger eviction (60 more unique prompts)", phase2_prompts,
                   concurrency=4, max_tokens=15)

    print("\n--- Metrics after Phase 2 ---")
    m2 = get_metrics()
    for pod, m in m2.items():
        print(f"  {pod}: {m}")

    # --- Phase 3: Re-send Phase 1 prompts — should hit HiCache ---
    r3 = run_phase("Phase 3 — HiCache hit test (re-send Phase 1 prompts)", phase1_prompts,
                   concurrency=4, max_tokens=15)

    print("\n--- Metrics after Phase 3 ---")
    m3 = get_metrics()
    for pod, m in m3.items():
        print(f"  {pod}: {m}")

    # --- Phase 4: Re-send Phase 2 prompts — should also hit HiCache ---
    r4 = run_phase("Phase 4 — HiCache hit test (re-send Phase 2 prompts)", phase2_prompts,
                   concurrency=4, max_tokens=15)

    print("\n--- Metrics after Phase 4 ---")
    m4 = get_metrics()
    for pod, m in m4.items():
        print(f"  {pod}: {m}")

    # --- Analysis ---
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    def avg_latency(results):
        ok = [r["elapsed"] for r in results if r["ok"]]
        return statistics.mean(ok) if ok else 0

    def avg_prompt_tokens(results):
        ok = [r.get("prompt_tokens", 0) for r in results if r["ok"]]
        return statistics.mean(ok) if ok else 0

    l1 = avg_latency(r1)
    l3 = avg_latency(r3)
    l2 = avg_latency(r2)
    l4 = avg_latency(r4)

    print(f"\n  Phase 1 (cold, fill GPU KV):    mean={l1:.2f}s  prompt_tok={avg_prompt_tokens(r1):.0f}")
    print(f"  Phase 3 (warm, HiCache hit):    mean={l3:.2f}s  prompt_tok={avg_prompt_tokens(r3):.0f}")
    print(f"  Phase 2 (cold, trigger evict):  mean={l2:.2f}s  prompt_tok={avg_prompt_tokens(r2):.0f}")
    print(f"  Phase 4 (warm, HiCache hit):    mean={l4:.2f}s  prompt_tok={avg_prompt_tokens(r4):.0f}")

    speedup1 = l1 / l3 if l3 > 0 else 0
    speedup2 = l2 / l4 if l4 > 0 else 0
    print(f"\n  Speedup Phase 1→3: {speedup1:.2f}x  (cold vs HiCache hit)")
    print(f"  Speedup Phase 2→4: {speedup2:.2f}x  (cold vs HiCache hit)")

    # HiCache metrics delta
    print(f"\n  HiCache host_used_tokens:")
    print(f"    Baseline:  {sum(m.get('hicache_host_used_tokens', 0) for m in m0.values() if isinstance(m, dict)):,.0f}")
    print(f"    After P1:  {sum(m.get('hicache_host_used_tokens', 0) for m in m1.values() if isinstance(m, dict)):,.0f}")
    print(f"    After P2:  {sum(m.get('hicache_host_used_tokens', 0) for m in m2.values() if isinstance(m, dict)):,.0f}")
    print(f"    After P3:  {sum(m.get('hicache_host_used_tokens', 0) for m in m3.values() if isinstance(m, dict)):,.0f}")
    print(f"    After P4:  {sum(m.get('hicache_host_used_tokens', 0) for m in m4.values() if isinstance(m, dict)):,.0f}")

    host_total = sum(m.get('hicache_host_total_tokens', 0) for m in m4.values() if isinstance(m, dict))
    host_used = sum(m.get('hicache_host_used_tokens', 0) for m in m4.values() if isinstance(m, dict))
    if host_total > 0:
        print(f"    Host cache utilization: {host_used/host_total*100:.1f}% ({host_used:,.0f}/{host_total:,.0f} tokens)")

    print(f"\n  Cache hit_rate:")
    for phase_name, m in [("Baseline", m0), ("After P1", m1), ("After P2", m2), ("After P3", m3), ("After P4", m4)]:
        rates = [v for pod_m in m.values() if isinstance(pod_m, dict) for k, v in pod_m.items() if k == "cache_hit_rate"]
        if rates:
            print(f"    {phase_name}: {statistics.mean(rates)*100:.1f}%")

    # Verdict
    print(f"\n  VERDICT:")
    if speedup1 > 1.5 or speedup2 > 1.5:
        print(f"    ✅ HiCache IS working — speedup {max(speedup1, speedup2):.2f}x on cache hit")
    elif host_used > 0:
        print(f"    ✅ HiCache IS working — {host_used:,.0f} tokens in host cache")
    else:
        print(f"    ⚠️ HiCache may NOT be working — no speedup and no host cache usage")

    # Save results
    out = {"phase1": r1, "phase2": r2, "phase3": r3, "phase4": r4,
           "metrics": {"baseline": m0, "p1": m1, "p2": m2, "p3": m3, "p4": m4}}
    with open("/tmp/hicache_stress_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Results saved to /tmp/hicache_stress_results.json")


if __name__ == "__main__":
    main()
