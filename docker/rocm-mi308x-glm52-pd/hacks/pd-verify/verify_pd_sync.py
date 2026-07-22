#!/usr/bin/env python3
"""
Verify PD (Prefill-Decode) disaggregation correctness:
  1. KV transfer from prefill to decode completes successfully
  2. Long-context requests (where KV transfer matters more) produce correct output
  3. Multiple sequential requests verify consistency
  4. Verify no token corruption: ask factual questions with deterministic answers
"""

import json
import sys
import time
import urllib.request
import urllib.error

BASE_URL = "https://glm52-pd-1p1d.jmpti.woa.com"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"

def call_chat(messages, max_tokens=256, temperature=0.0):
    body = {
        "model": "glm52",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = BASE_URL + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=headers)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode())
            dt = time.time() - t0
            return resp.status, payload, dt
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"error": body[:500]}
        return e.code, payload, 0.0
    except Exception as e:
        return -1, {"error": str(e)}, 0.0

def check(cond, label):
    print(f"  [{'OK' if cond else 'FAIL'}] {label}")
    return bool(cond)

def test_short_request():
    """Short request - small KV transfer."""
    print("\n=== PD Test 1: Short request (small KV transfer) ===")
    msgs = [
        {"role": "user", "content": "What is the capital of France? Reply with just the city name."},
    ]
    status, resp, dt = call_chat(msgs, max_tokens=64)
    print(f"  HTTP {status} ({dt:.2f}s)")
    if status != 200:
        print(f"  Error: {resp}")
        return False
    content = resp["choices"][0]["message"]["content"]
    usage = resp["usage"]
    print(f"  content: {content!r}")
    print(f"  usage: {usage}")
    ok = check("paris" in content.lower(), f"answer contains 'paris'")
    ok &= check(usage["prompt_tokens"] > 0, "prompt_tokens > 0")
    ok &= check(usage["completion_tokens"] > 0, "completion_tokens > 0")
    return ok

def test_long_context_request():
    """Long context - significant KV transfer between prefill and decode."""
    print("\n=== PD Test 2: Long context request (significant KV transfer) ===")
    # Build a long context: repeat a story multiple times then ask a question
    story = """Once upon a time, in a galaxy far far away, there lived a programmer named Alice who worked on distributed inference systems. She spent her days debugging RDMA transfers, optimizing KV cache layouts, and ensuring that prefill and decode pods communicated correctly. Her favorite number was 42, and she always kept a rubber duck on her desk for debugging purposes."""
    long_context = (story + "\n\n") * 20  # ~8000 tokens of context
    msgs = [
        {"role": "system", "content": "You answer questions about the text provided."},
        {"role": "user", "content": long_context + "\n\nWhat is Alice's favorite number? Reply with just the number."},
    ]
    status, resp, dt = call_chat(msgs, max_tokens=64)
    print(f"  HTTP {status} ({dt:.2f}s)")
    if status != 200:
        print(f"  Error: {resp}")
        return False
    content = resp["choices"][0]["message"]["content"]
    usage = resp["usage"]
    print(f"  content: {content!r}")
    print(f"  usage: {usage}")
    # KV transfer correctness: even with long context, the model should retrieve the answer
    ok = check("42" in content, f"answer contains '42' (KV transfer correctness)")
    ok &= check(usage["prompt_tokens"] > 1000, f"large prompt (>1000 tokens): {usage['prompt_tokens']}")
    return ok

def test_multi_turn_consistency():
    """Multi-turn conversation - verifies KV state preserved correctly."""
    print("\n=== PD Test 3: Multi-turn consistency ===")
    msgs = [
        {"role": "user", "content": "My name is TestUser123 and I like blueberries. Remember this."},
    ]
    status, resp, dt = call_chat(msgs, max_tokens=64)
    if status != 200:
        print(f"  First turn failed: {resp}")
        return False
    print(f"  Turn 1: {dt:.2f}s, content: {resp['choices'][0]['message']['content'][:60]!r}")

    msgs.append({"role": "assistant", "content": resp["choices"][0]["message"]["content"]})
    msgs.append({"role": "user", "content": "What is my name and what do I like?"})
    status, resp, dt = call_chat(msgs, max_tokens=64)
    if status != 200:
        print(f"  Second turn failed: {resp}")
        return False
    content = resp["choices"][0]["message"]["content"]
    print(f"  Turn 2: {dt:.2f}s, content: {content[:100]!r}")
    ok = check("TestUser123" in content, "remembers name TestUser123")
    ok &= check("blue" in content.lower(), "remembers blueberries")
    return ok

def test_sequential_consistency():
    """Multiple identical sequential requests should give similar answers."""
    print("\n=== PD Test 4: Sequential consistency (same request x3) ===")
    msgs = [
        {"role": "user", "content": "Count from 1 to 5, comma-separated. Just the numbers."},
    ]
    answers = []
    for i in range(3):
        status, resp, dt = call_chat(msgs, max_tokens=64, temperature=0.0)
        if status != 200:
            print(f"  Request {i+1} failed: {resp}")
            return False
        content = resp["choices"][0]["message"]["content"]
        answers.append(content)
        print(f"  Run {i+1}: {dt:.2f}s, content: {content!r}")
    # All should contain 1, 2, 3, 4, 5
    ok = True
    for i, ans in enumerate(answers):
        ok &= check(all(f"{n}" in ans for n in [1,2,3,4,5]), f"run {i+1} contains 1-5")
    return ok

def test_concurrent_pd_stress():
    """Concurrent requests - stress PD coordination."""
    print("\n=== PD Test 5: Concurrent PD stress (3 parallel requests) ===")
    import concurrent.futures
    prompts = [
        "What is 2+2? Brief answer.",
        "What is 3+3? Brief answer.",
        "What is 4+4? Brief answer.",
    ]
    expected = ["4", "6", "8"]
    results = [None] * 3

    def worker(idx, prompt):
        msgs = [{"role": "user", "content": prompt}]
        status, resp, dt = call_chat(msgs, max_tokens=64)
        return idx, status, resp, dt

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker, i, p) for i, p in enumerate(prompts)]
        for f in concurrent.futures.as_completed(futures):
            idx, status, resp, dt = f.result()
            results[idx] = (status, resp, dt)
            print(f"  Request {idx+1}: HTTP {status} ({dt:.2f}s)")

    ok = True
    for i, (status, resp, dt) in enumerate(results):
        if status != 200:
            print(f"  FAIL: request {i+1} status {status}")
            ok = False
            continue
        content = resp["choices"][0]["message"]["content"]
        ok &= check(expected[i] in content, f"request {i+1} contains expected answer '{expected[i]}'")
    return ok

def main():
    print(f"Endpoint: {BASE_URL}")
    print(f"Verifying PD disaggregation sync correctness")
    results = {}
    results["short_request"] = test_short_request()
    results["long_context"] = test_long_context_request()
    results["multi_turn"] = test_multi_turn_consistency()
    results["sequential"] = test_sequential_consistency()
    results["concurrent"] = test_concurrent_pd_stress()

    print("\n" + "=" * 50)
    print("PD SYNC VERIFICATION SUMMARY")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k:30s} {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)

if __name__ == "__main__":
    main()
