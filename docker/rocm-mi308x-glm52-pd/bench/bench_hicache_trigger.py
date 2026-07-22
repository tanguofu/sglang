#!/usr/bin/env python3
"""Trigger HiCache by filling GPU KV cache with large unique prompts.
Generates prompts that fill ~10K tokens each, sends 100 of them to fill ~1M tokens total,
exceeding the 509K GPU KV cache capacity and forcing eviction to host cache.
"""
import json
import random
import subprocess
import time
import sys

API_URL = "http://glm52-2tp8.jmpti.woa.com/v1/chat/completions"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"

WORDS = [
    "system", "design", "architecture", "scalability", "latency", "throughput",
    "reliability", "consistency", "availability", "partition", "tolerance",
    "replication", "sharding", "indexing", "caching", "queueing", "batching",
    "streaming", "concurrency", "parallelism", "asynchronous", "synchronous",
    "distributed", "centralized", "decentralized", "federated", "hybrid",
    "microservice", "monolith", "serverless", "container", "orchestration",
    "kubernetes", "docker", "service", "mesh", "api", "gateway", "backend",
    "frontend", "database", "message", "queue", "event", "driven", "reactive",
]


def gen_long_prompt(idx):
    """Generate a ~8000 token unique prompt."""
    rng = random.Random(idx * 7919 + 31)
    lines = []
    for i in range(800):
        line = " ".join(rng.sample(WORDS, 15))
        lines.append(f"{i}. {line}")
    return f"Analyze the following system design scenario #{idx} in detail: " + " ".join(lines)


def get_hicache_metrics(pod):
    """Get HiCache metrics from a pod."""
    result = subprocess.run(
        ["kubectl", "exec", "-n", "kube-system", pod, "--", "curl", "-s",
         "http://localhost:30000/metrics"],
        capture_output=True, text=True, timeout=30
    )
    metrics = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        for key in ["hicache_host_used_tokens", "kv_used_tokens",
                     "kv_available_tokens", "kv_evictable_tokens",
                     "cached_tokens_total", "cache_hit_rate"]:
            if line.startswith(f"sglang:{key}"):
                try:
                    val = float(line.split()[-1])
                    metrics[key] = val
                except (ValueError, IndexError):
                    pass
    return metrics


def send_request(prompt, max_tokens=5):
    """Send a single request and return (success, output_tokens)."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    })
    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", API_URL,
             "-H", "Content-Type: application/json",
             "-H", f"Authorization: Bearer {API_KEY}",
             "-d", payload],
            capture_output=True, text=True, timeout=120
        )
        resp = json.loads(result.stdout)
        return "error" not in resp, resp.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        print(f"  ERROR: {e}")
        return False, 0


def main():
    print("=== HiCache Trigger Test (Python) ===")
    print()

    pod0 = "sglang-glm52-2tp8-sglang-0"
    pod1 = "sglang-glm52-2tp8-sglang-1"

    print("--- BEFORE metrics ---")
    m0_before = get_hicache_metrics(pod0)
    m1_before = get_hicache_metrics(pod1)
    print(f"Pod-0 (.152): {m0_before}")
    print(f"Pod-1 (.172): {m1_before}")
    print()

    # Send 100 large unique prompts to fill ~800K tokens (exceeds 509K GPU KV)
    num_requests = 100
    print(f"Sending {num_requests} large unique prompts (~8K tokens each, ~800K total)...")

    success = 0
    total_output = 0
    for i in range(1, num_requests + 1):
        prompt = gen_long_prompt(i)
        ok, out_tok = send_request(prompt, max_tokens=5)
        if ok:
            success += 1
            total_output += out_tok
        if i % 20 == 0:
            print(f"  Completed {i}/{num_requests} (success={success})")
            # Check intermediate metrics
            m0 = get_hicache_metrics(pod0)
            m1 = get_hicache_metrics(pod1)
            print(f"    Pod-0: kv_used={m0.get('kv_used_tokens', 0):.0f} "
                  f"kv_avail={m0.get('kv_available_tokens', 0):.0f} "
                  f"hicache_used={m0.get('hicache_host_used_tokens', 0):.0f} "
                  f"cached={m0.get('cached_tokens_total', 0):.0f}")
            print(f"    Pod-1: kv_used={m1.get('kv_used_tokens', 0):.0f} "
                  f"kv_avail={m1.get('kv_available_tokens', 0):.0f} "
                  f"hicache_used={m1.get('hicache_host_used_tokens', 0):.0f} "
                  f"cached={m1.get('cached_tokens_total', 0):.0f}")

    print()
    print(f"Total: {success}/{num_requests} success, {total_output} output tokens")
    print()

    print("--- AFTER metrics ---")
    m0_after = get_hicache_metrics(pod0)
    m1_after = get_hicache_metrics(pod1)
    print(f"Pod-0 (.152): {m0_after}")
    print(f"Pod-1 (.172): {m1_after}")
    print()

    # Delta
    print("--- DELTA ---")
    for key in ["hicache_host_used_tokens", "kv_used_tokens",
                "kv_available_tokens", "cached_tokens_total"]:
        d0 = m0_after.get(key, 0) - m0_before.get(key, 0)
        d1 = m1_after.get(key, 0) - m1_before.get(key, 0)
        print(f"  {key}: pod-0 {d0:+.0f}, pod-1 {d1:+.0f}")

    print()
    print("=== HiCache trigger test complete ===")


if __name__ == "__main__":
    main()
