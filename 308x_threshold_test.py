#!/usr/bin/env python3
"""Quick threshold test: find the exact crash point."""
import requests
import json
import time
import sys

API = "http://127.0.0.1:30000"
KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"
LOG = "/tmp/threshold_test.log"

def log(msg):
    line = "[{}] {}".format(time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def tokenize(text):
    try:
        r = requests.post(f"{API}/tokenize",
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "prompt": text}, timeout=30)
        if r.status_code == 200:
            d = r.json()
            return d.get("count", d.get("len", 0))
    except:
        pass
    return 0

def gen_tokens(target):
    """Generate text with approximately target tokens."""
    base = "The quick brown fox jumps over the lazy dog. This is a test. "
    # Calibrate
    cnt = tokenize(base)
    if cnt and cnt > 0:
        ratio = len(base) / cnt
    else:
        ratio = 4.0
    target_chars = int(target * ratio)
    reps = target_chars // len(base) + 1
    text = (base * reps)[:target_chars]
    return text + " What is 2+2? Answer with just the number."

def test_tokens(name, target_tokens, max_tokens=10):
    log(f"\n=== {name}: target={target_tokens} tokens ===")
    text = gen_tokens(target_tokens)
    actual = tokenize(text)
    log(f"Generated {len(text)} chars, actual tokens: {actual}")

    start = time.time()
    try:
        r = requests.post(f"{API}/v1/chat/completions",
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": text}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=120)
        elapsed = time.time() - start
        log(f"HTTP {r.status_code} ({elapsed:.2f}s)")
        if r.status_code == 200:
            d = r.json()
            usage = d.get("usage", {})
            log(f"Usage: {usage}")
            return True
        else:
            log(f"Error: {r.text[:300]}")
            return False
    except Exception as e:
        elapsed = time.time() - start
        log(f"Exception ({elapsed:.2f}s): {e}")
        return False

def wait_health():
    for i in range(120):
        try:
            r = requests.get(f"{API}/health", timeout=5)
            if r.status_code == 200:
                log(f"Health OK after {i*5}s")
                time.sleep(10)
                return True
        except:
            pass
        time.sleep(5)
    return False

def main():
    open(LOG, "w").close()
    log("Starting threshold test")

    if not wait_health():
        log("FATAL: Server not ready")
        return

    # Test below DSA threshold (2048)
    test_tokens("1K", 1000)
    test_tokens("2K-below", 2000)

    # Test at DSA threshold
    test_tokens("2K-at", 2048)

    # Test just above DSA threshold
    test_tokens("2K-above", 2050)
    test_tokens("2.5K", 2500)

    # If server still alive, test 3K and 4K
    test_tokens("3K", 3000)
    test_tokens("4K", 4096)

    log("\n=== THRESHOLD TEST DONE ===")

if __name__ == "__main__":
    main()
