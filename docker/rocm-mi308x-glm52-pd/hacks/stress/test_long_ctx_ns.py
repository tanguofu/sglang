#!/usr/bin/env python3
"""Test long context — non-streaming with longer timeout."""
import subprocess, json, time

BASE = "https://glm52-1tp8.jmpti.woa.com"
KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"

def curl_ns(url, payload, timeout=600):
    cmd = [
        "curl", "-s", "-w", "\n%{http_code}\n%{time_total}",
        "--max-time", str(timeout),
        "-X", "POST",
        "-H", f"Authorization: Bearer {KEY}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload),
        f"{BASE}{url}"
    ]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+10)
    wall = time.time() - start
    parts = result.stdout.rsplit("\n", 2)
    if len(parts) >= 3:
        body, code, curl_time = parts[0], parts[1], parts[2]
        try:
            d = json.loads(body)
            usage = d.get("usage", {})
            content = d.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"http": code, "wall": float(curl_time),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "content": content}
        except:
            return {"http": code, "wall": float(curl_time), "error": body[:200]}
    return {"http": "ERR", "wall": wall, "error": result.stdout[:200]}

print("--- Long Context Non-Streaming (no-think, timeout=600s) ---")
for n_repeats in [50, 100, 200, 300, 500]:
    prompt = 'The quick brown fox jumps over the lazy dog. ' * n_repeats
    payload = {
        'model': 'glm-5.2',
        'messages': [{'role': 'user', 'content': f'How many times does "fox" appear? Just the number.\n{prompt}'}],
        'max_tokens': 20,
        'temperature': 0,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    r = curl_ns("/v1/chat/completions", payload, timeout=600)
    if r.get("http") == "200":
        print(f'  repeats={n_repeats:5d} | prompt_toks={r["prompt_tokens"]:6d} | HTTP=200 | '
              f'wall={r["wall"]:.2f}s | answer={r["content"][:30]}')
    else:
        print(f'  repeats={n_repeats:5d} | HTTP={r["http"]} | wall={r["wall"]:.2f}s | error={r.get("error","")[:100]}')
