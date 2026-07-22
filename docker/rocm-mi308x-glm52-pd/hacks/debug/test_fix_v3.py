#!/usr/bin/env python3
"""Full test suite for fix-eagle-coredump-v3 deployment."""
import subprocess, json, time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://glm52-1tp8.jmpti.woa.com"
KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
PASS = 0
FAIL = 0
RESULTS = []

def curl(url, payload, timeout=300):
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
            return {"http": code, "wall": float(curl_time), "data": d}
        except:
            return {"http": code, "wall": float(curl_time), "error": body[:200]}
    return {"http": "ERR", "wall": wall, "error": result.stdout[:200]}

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        RESULTS.append(f"  PASS | {name} | {detail}")
        print(f"  PASS | {name} | {detail}")
    else:
        FAIL += 1
        RESULTS.append(f"  FAIL | {name} | {detail}")
        print(f"  FAIL | {name} | {detail}")

print("=" * 100)
print("GLM-5.2 1tp8 — fix-eagle-coredump-v3 Full Test Suite")
print("=" * 100)

# --- 1. Basic functionality ---
print("\n--- 1. Basic Functionality ---")
r = curl("/v1/chat/completions", {"model":"glm-5.2","messages":[{"role":"user","content":"What is 2+3? Just the number."}],"max_tokens":100,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}})
d = r.get("data", {})
content = d.get("choices",[{}])[0].get("message",{}).get("content","")
check("chat basic (2+3=5)", content.strip() == "5", f'answer="{content}", wall={r["wall"]:.2f}s')

r = curl("/v1/messages", {"model":"glm-5.2","messages":[{"role":"user","content":"What is 7*8? Just the number."}],"max_tokens":100,"temperature":0})
d = r.get("data", {})
content_blocks = d.get("content", [])
content = "".join(b.get("text","") for b in content_blocks if b.get("type")=="text")
check("/v1/messages basic (7*8=56)", "56" in content, f'answer="{content[:50]}", wall={r["wall"]:.2f}s')

r = curl("/v1/responses", {"model":"glm-5.2","input":"What is 3+4? Just the number.","max_output_tokens":100,"temperature":0})
d = r.get("data", {})
outputs = d.get("output", [])
content = "".join(c.get("text","") for o in outputs for c in o.get("content",[]) if c.get("type") in ("output_text","text"))
check("/v1/responses basic (3+4=7)", "7" in content, f'answer="{content[:50]}", wall={r["wall"]:.2f}s')

# --- 2. Streaming ---
print("\n--- 2. Streaming ---")
cmd = ["curl", "-s", "-N", "--max-time", "60", "-X", "POST", "-H", f"Authorization: Bearer {KEY}", "-H", "Content-Type: application/json", "-d", json.dumps({"model":"glm-5.2","messages":[{"role":"user","content":"Count from 1 to 5."}],"max_tokens":50,"temperature":0,"chat_template_kwargs":{"enable_thinking":False},"stream":True}), f"{BASE}/v1/chat/completions"]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=70)
chunks = [l for l in result.stdout.split("\n") if l.startswith("data: ") and "[DONE]" not in l]
check("streaming produces chunks", len(chunks) > 2, f"chunks={len(chunks)}")

# --- 3. Tool calls ---
print("\n--- 3. Tool Calls ---")
r = curl("/v1/chat/completions", {"model":"glm-5.2","messages":[{"role":"user","content":"What is the weather in Tokyo? Use the tool."}],"max_tokens":300,"temperature":0,"chat_template_kwargs":{"enable_thinking":False},"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]})
d = r.get("data", {})
tool_calls = d.get("choices",[{}])[0].get("message",{}).get("tool_calls",[])
check("tool_calls works", tool_calls and tool_calls[0]["function"]["name"] == "get_weather", f'tool={tool_calls[0]["function"]["name"] if tool_calls else "none"}, wall={r["wall"]:.2f}s')

# --- 4. Reasoning ---
print("\n--- 4. Reasoning ---")
for effort in ["low", "high", "max"]:
    r = curl("/v1/chat/completions", {"model":"glm-5.2","messages":[{"role":"user","content":"What is 15*17?"}],"max_tokens":500,"temperature":0,"reasoning_effort":effort})
    d = r.get("data", {})
    reasoning = d.get("choices",[{}])[0].get("message",{}).get("reasoning_content","")
    content = d.get("choices",[{}])[0].get("message",{}).get("content","")
    check(f"reasoning_effort={effort}", "255" in content or "255" in reasoning, f'content="{content[:30]}", reasoning_len={len(reasoning)}, wall={r["wall"]:.2f}s')

# --- 5. Long context ---
print("\n--- 5. Long Context ---")
for n_repeats in [10, 50, 100, 200, 500]:
    prompt = 'The quick brown fox jumps over the lazy dog. ' * n_repeats
    r = curl("/v1/chat/completions", {"model":"glm-5.2","messages":[{"role":"user","content":f'How many times does "fox" appear? Just the number.\n{prompt}'}],"max_tokens":20,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}, timeout=600)
    d = r.get("data", {})
    content = d.get("choices",[{}])[0].get("message",{}).get("content","")
    prompt_toks = d.get("usage",{}).get("prompt_tokens",0)
    expected = n_repeats
    check(f"long context repeats={n_repeats} (expect {expected})", content.strip() == str(expected), f'answer="{content[:20]}", prompt_toks={prompt_toks}, wall={r["wall"]:.2f}s')

# --- 6. Concurrency ---
print("\n--- 6. Concurrency (N=10, short no-think) ---")
payload = {"model":"glm-5.2","messages":[{"role":"user","content":"What is 2+3? Just the number."}],"max_tokens":10,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
with ThreadPoolExecutor(max_workers=10) as pool:
    futures = [pool.submit(curl, "/v1/chat/completions", payload) for _ in range(10)]
    start = time.time()
    results = [f.result() for f in as_completed(futures)]
    wall = time.time() - start
ok = [r for r in results if r.get("http") == "200"]
check("concurrent=10 all succeed", len(ok) == 10, f"ok={len(ok)}/10, wall={wall:.2f}s, tput={len(ok)/wall:.1f} req/s")

# --- Summary ---
print("\n" + "=" * 100)
print(f"RESULTS: {PASS} PASS, {FAIL} FAIL")
print("=" * 100)
sys.exit(0 if FAIL == 0 else 1)
