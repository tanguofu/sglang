#!/usr/bin/env python3
"""Debug test to isolate the 4K crash issue."""
import requests
import json
import time
import sys

API = "http://127.0.0.1:30000"
KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"

def test_request(name, prompt, max_tokens=256, stream=False, extra_params=None):
    print(f"\n=== {name} ===")
    print(f"Prompt length: {len(prompt)} chars")

    params = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": stream,
    }
    if extra_params:
        params.update(extra_params)

    start = time.time()
    try:
        if stream:
            r = requests.post(
                f"{API}/v1/chat/completions",
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                json=params,
                stream=True,
                timeout=120,
            )
            print(f"HTTP {r.status_code}")
            if r.status_code != 200:
                print(f"Error: {r.text[:300]}")
                return False
            content = ""
            reasoning = ""
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
                            content += delta.get("content", "") or ""
                            reasoning += delta.get("reasoning_content", "") or ""
                        except:
                            pass
            elapsed = time.time() - start
            print(f"Time: {elapsed:.2f}s")
            print(f"Content: {content[:100]}")
            print(f"Reasoning: {reasoning[:100]}")
            return True
        else:
            r = requests.post(
                f"{API}/v1/chat/completions",
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                json=params,
                timeout=120,
            )
            elapsed = time.time() - start
            print(f"HTTP {r.status_code} ({elapsed:.2f}s)")
            if r.status_code == 200:
                d = r.json()
                c = d.get("choices", [{}])[0]
                print(f"Content: {c.get('message', {}).get('content', '')[:100]}")
                print(f"Reasoning: {c.get('message', {}).get('reasoning_content', '')[:100]}")
                print(f"Usage: {d.get('usage', {})}")
                return True
            else:
                print(f"Error: {r.text[:300]}")
                return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

# Test 1: Short prompt, non-streaming
test_request("Short non-stream", "What is 2+2? Answer with just the number.", max_tokens=50)

# Test 2: Short prompt, streaming
test_request("Short stream", "What is 2+2? Answer with just the number.", max_tokens=50, stream=True)

# Test 3: Medium prompt (~500 tokens), non-streaming
base = "The quick brown fox jumps over the lazy dog. This is a test of the long context capability. "
medium_prompt = base * 100 + " What is 2+2? Answer with just the number."
test_request("Medium non-stream", medium_prompt, max_tokens=256)

# Test 4: Medium prompt, streaming
test_request("Medium stream", medium_prompt, max_tokens=256, stream=True)

# Test 5: Medium prompt with thinking disabled
test_request("Medium no-thinking", medium_prompt, max_tokens=256, extra_params={"chat_template_kwargs": {"enable_thinking": False}})

# Test 6: 4K prompt, non-streaming, no thinking
large_prompt = base * 800 + " What is 2+2? Answer with just the number."
test_request("4K non-stream no-thinking", large_prompt, max_tokens=256, extra_params={"chat_template_kwargs": {"enable_thinking": False}})

# Test 7: 4K prompt, streaming, no thinking
test_request("4K stream no-thinking", large_prompt, max_tokens=256, stream=True, extra_params={"chat_template_kwargs": {"enable_thinking": False}})

print("\n=== DONE ===")
