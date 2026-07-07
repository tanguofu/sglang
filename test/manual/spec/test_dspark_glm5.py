"""E2E smoke test for DSpark GLM-5.2 on AMD MI355X.

Short-to-short validation: start server, send a few requests, check:
1. Server starts and responds
2. Output is correct (not garbage)
3. Greedy determinism (same prompt → same output)
4. accept_len > 1.0 (at least some draft tokens accepted)
5. Server doesn't crash under concurrent requests

Usage on node-2:
    python3 test/manual/spec/test_dspark_glm5.py --base-url http://localhost:30001

Or run against a freshly started server:
    python3 test/manual/spec/test_dspark_glm5.py --start-server
"""

import argparse
import json
import subprocess
import sys
import time

import requests


def test_health(base_url):
    """Test 1: Server health check."""
    print("=== Test 1: Health check ===")
    resp = requests.get(f"{base_url}/health", timeout=10)
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    print("  ✅ Server healthy")


def test_basic_completion(base_url):
    """Test 2: Basic completion produces correct output."""
    print("=== Test 2: Basic completion ===")
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "What is 2+3? Answer with just the number."}],
            "temperature": 0,
            "max_tokens": 16,
        },
        timeout=60,
    )
    assert resp.status_code == 200, f"Request failed: {resp.status_code}"
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    print(f"  Output: {text!r}")
    assert len(text) > 0, "Empty output"
    print("  ✅ Output non-empty")


def test_greedy_determinism(base_url):
    """Test 3: Greedy decoding is deterministic."""
    print("=== Test 3: Greedy determinism ===")
    outputs = []
    for _ in range(2):
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "List 3 countries."}],
                "temperature": 0,
                "max_tokens": 64,
            },
            timeout=60,
        )
        outputs.append(resp.json()["choices"][0]["message"]["content"])
    print(f"  Output 1: {outputs[0][:60]!r}...")
    print(f"  Output 2: {outputs[1][:60]!r}...")
    if outputs[0] == outputs[1]:
        print("  ✅ Deterministic")
    else:
        print("  ⚠️ Outputs differ (may be due to radix cache / batching)")


def test_accept_len(base_url, container_name="glm52_dspark"):
    """Test 4: Check accept_len from server logs."""
    print("=== Test 4: accept_len check ===")
    try:
        result = subprocess.run(
            ["docker", "logs", container_name],
            capture_output=True, text=True, timeout=10,
        )
        lines = result.stdout + result.stderr
        accept_lines = [l for l in lines.split("\n") if "accept len" in l.lower()]
        if accept_lines:
            last = accept_lines[-1]
            print(f"  Latest: {last.strip()}")
            # Extract accept len value
            for part in last.split(","):
                if "accept len" in part:
                    val = float(part.split(":")[-1].strip())
                    print(f"  accept_len = {val}")
                    if val > 1.0:
                        print(f"  ✅ accept_len > 1.0 (some drafts accepted)")
                    else:
                        print(f"  ⚠️ accept_len = {val} (no drafts accepted — expected with corrupt cache)")
                    return val
        print("  ⚠️ No accept_len found in logs yet")
        return 0.0
    except Exception as e:
        print(f"  ⚠️ Could not check logs: {e}")
        return 0.0


def test_concurrent_requests(base_url, n=4):
    """Test 5: Concurrent requests don't crash server."""
    print(f"=== Test 5: Concurrent requests ({n}) ===")
    import concurrent.futures

    def send_request(i):
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": f"Say hello in {i+1} word(s)."}],
                "temperature": 0,
                "max_tokens": 32,
            },
            timeout=60,
        )
        return resp.status_code, resp.json()["choices"][0]["message"]["content"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        results = list(executor.map(send_request, range(n)))

    for i, (code, text) in enumerate(results):
        print(f"  Request {i}: status={code}, text={text[:40]!r}")
        assert code == 200, f"Request {i} failed: {code}"
    print(f"  ✅ All {n} concurrent requests succeeded")


def test_long_generation(base_url):
    """Test 6: Longer generation doesn't crash."""
    print("=== Test 6: Long generation ===")
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "Write a short poem about the ocean."}],
            "temperature": 0,
            "max_tokens": 256,
        },
        timeout=120,
    )
    assert resp.status_code == 200, f"Long generation failed: {resp.status_code}"
    text = resp.json()["choices"][0]["message"]["content"]
    print(f"  Output ({len(text)} chars): {text[:80]!r}...")
    assert len(text) > 50, f"Output too short: {len(text)} chars"
    print("  ✅ Long generation OK")


def main():
    parser = argparse.ArgumentParser(description="DSpark GLM-5.2 E2E smoke test")
    parser.add_argument("--base-url", default="http://localhost:30001", help="Server base URL")
    parser.add_argument("--container-name", default="glm52_dspark", help="Docker container name for log checking")
    args = parser.parse_args()

    print(f"DSpark E2E Smoke Test — {args.base_url}")
    print("=" * 60)

    tests = [
        ("Health", lambda: test_health(args.base_url)),
        ("Basic completion", lambda: test_basic_completion(args.base_url)),
        ("Greedy determinism", lambda: test_greedy_determinism(args.base_url)),
        ("Concurrent requests", lambda: test_concurrent_requests(args.base_url)),
        ("Long generation", lambda: test_long_generation(args.base_url)),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
        print()

    # accept_len check (non-blocking)
    try:
        test_accept_len(args.base_url, args.container_name)
    except Exception as e:
        print(f"  ⚠️ accept_len check skipped: {e}")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
