#!/usr/bin/env python3
"""Targeted debug test to isolate the server crash issue.
Tests different request types after health check passes.
"""
import requests
import json
import time
import sys

API = "http://127.0.0.1:30000"
KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"
LOG = "/tmp/debug_test2.log"

def log(msg):
    line = "[{}] {}".format(time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def wait_health(max_wait=900):
    for i in range(max_wait // 5):
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

def test(name, prompt, max_tokens=50, stream=False, endpoint="chat", extra=None):
    log(f"\n=== {name} ===")
    log(f"Prompt: {len(prompt)} chars, max_tokens={max_tokens}, stream={stream}, endpoint={endpoint}")

    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }
    if extra:
        body.update(extra)

    url = f"{API}/v1/chat/completions" if endpoint == "chat" else f"{API}/v1/completions"
    if endpoint == "completion":
        body = {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": stream,
        }

    start = time.time()
    try:
        if stream:
            r = requests.post(url, headers=headers, json=body, stream=True, timeout=120)
            log(f"HTTP {r.status_code} ({time.time()-start:.2f}s)")
            if r.status_code != 200:
                log(f"Error: {r.text[:300]}")
                return False
            content = ""
            for line in r.iter_lines():
                if line:
                    s = line.decode("utf-8", errors="replace")
                    if s.startswith("data: "):
                        d = s[6:]
                        if d.strip() == "[DONE]":
                            break
                        try:
                            j = json.loads(d)
                            delta = j.get("choices", [{}])[0].get("delta", {})
                            c = delta.get("content", "") or ""
                            if c:
                                content += c
                        except:
                            pass
            elapsed = time.time() - start
            log(f"Stream done ({elapsed:.2f}s): {content[:100]}")
            return True
        else:
            r = requests.post(url, headers=headers, json=body, timeout=120)
            elapsed = time.time() - start
            log(f"HTTP {r.status_code} ({elapsed:.2f}s)")
            if r.status_code == 200:
                d = r.json()
                c = d.get("choices", [{}])[0]
                msg = c.get("message", c.get("text", ""))
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    reasoning = msg.get("reasoning_content", "")
                else:
                    content = str(msg)
                    reasoning = ""
                log(f"Content: {content[:100]}")
                if reasoning:
                    log(f"Reasoning: {reasoning[:100]}")
                log(f"Usage: {d.get('usage', {})}")
                return True
            else:
                log(f"Error: {r.text[:300]}")
                return False
    except Exception as e:
        elapsed = time.time() - start
        log(f"Exception ({elapsed:.2f}s): {e}")
        return False

def main():
    open(LOG, "w").close()
    log("Starting targeted debug test")

    if not wait_health():
        log("FATAL: Server not ready")
        return

    base = "The quick brown fox jumps over the lazy dog. This is a test. "

    # Test 1: Warmup-like (should work)
    test("1. Warmup-like (short, 5 tokens, non-stream)", "Say hello", max_tokens=5)

    # Test 2: Short prompt, more tokens, non-stream
    test("2. Short 50tok non-stream", "Say hello", max_tokens=50)

    # Test 3: Short prompt, streaming
    test("3. Short 5tok stream", "Say hello", max_tokens=5, stream=True)

    # Test 4: Short prompt, 50 tokens, streaming
    test("4. Short 50tok stream", "Say hello", max_tokens=50, stream=True)

    # Test 5: Medium prompt (~500 tokens), non-stream
    med = base * 100 + "Say hello"
    test("5. Medium 50tok non-stream", med, max_tokens=50)

    # Test 6: Medium prompt, streaming
    test("6. Medium 50tok stream", med, max_tokens=50, stream=True)

    # Test 7: 4K prompt, 5 tokens, non-stream
    large = base * 800 + "Say hello"
    test("7. 4K 5tok non-stream", large, max_tokens=5)

    # Test 8: 4K prompt, 50 tokens, non-stream
    test("8. 4K 50tok non-stream", large, max_tokens=50)

    # Test 9: 4K prompt, 5 tokens, streaming
    test("9. 4K 5tok stream", large, max_tokens=5, stream=True)

    # Test 10: Raw completion endpoint (no chat template)
    test("10. Completion short", "Say hello", max_tokens=50, endpoint="completion")

    # Test 11: Chat with thinking disabled
    test("11. Short no-thinking", "Say hello", max_tokens=50,
         extra={"chat_template_kwargs": {"enable_thinking": False}})

    # Test 12: 4K with thinking disabled
    test("12. 4K no-thinking", large, max_tokens=50,
         extra={"chat_template_kwargs": {"enable_thinking": False}})

    log("\n=== ALL TESTS DONE ===")

if __name__ == "__main__":
    main()
