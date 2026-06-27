#!/usr/bin/env python3
import json
import sys
import time
import urllib.request

ROUTER = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
SCHEME = sys.argv[2] if len(sys.argv) > 2 else "unknown"


def get(path, timeout=30):
    req = urllib.request.Request(f"{ROUTER}{path}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def post(path, data, timeout=180):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{ROUTER}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


print(f"=== Smoke test for {SCHEME} @ {ROUTER} ===")
get("/health")
print("[OK] /health")
models = json.loads(get("/v1/models"))
model_id = models["data"][0]["id"]
print(f"[OK] /v1/models: {model_id}")

text = ""
for attempt in range(3):
    time.sleep(2 if attempt else 5)
    t0 = time.time()
    gen = post(
        "/generate",
        {"text": "Hello", "sampling_params": {"temperature": 0, "max_new_tokens": 16}},
    )
    text = gen.get("text", "") if isinstance(gen, dict) else ""
    print(f"[try {attempt+1}] /generate in {time.time()-t0:.1f}s: {text[:80]!r}")
    if text:
        break

if not text:
    print("[WARN] /generate empty after retries; checking chat path")

t0 = time.time()
resp = post(
    "/v1/chat/completions",
    {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with exactly: PD_OK"}],
        "max_tokens": 32,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
t1 = time.time()
content = resp["choices"][0]["message"].get("content", "") or ""
print(f"[OK] chat in {t1-t0:.1f}s: {content[:80]!r}")
assert len(content) > 0, "empty chat output"
print(f"=== {SCHEME} SMOKE PASS ===")
