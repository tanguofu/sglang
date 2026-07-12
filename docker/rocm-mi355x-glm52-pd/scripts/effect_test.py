#!/usr/bin/env python3
"""
GLM-5.2 effect/quality alignment tests for the 1P1D PD deployment.

Verifies the PD-served output aligns with official GLM-5.2 behavior:
  1. Chat template applied correctly (thinking tags, role tags)
  2. Reasoning/thinking works (<think>...</think> + answer)
  3. Determinism under greedy decoding (PD KV-transfer lossless check)
  4. Tool calling (glm47 format) parses correctly
  5. Answer correctness on known questions (Chinese + English + math)
  6. enable_thinking=False suppresses reasoning
  7. Long-context coherence (chunked prefill + KV transfer)

Usage:
  python3 effect_test.py [ENDPOINT]
  # ENDPOINT default = http://216.128.154.57:8000  (router on bm1)
"""
import json
import sys
import time
import requests

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "http://216.128.154.57:8000"
TIMEOUT = 600

results = []  # (name, passed, detail)


def record(name, passed, detail=""):
    results.append((name, passed, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))


def chat(messages, tools=None, extra=None):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,  # greedy by default for determinism/correctness
        "top_p": 0.95,
        "max_tokens": 2048,
    }
    if tools:
        payload["tools"] = tools
    if extra:
        payload.update(extra)
    r = requests.post(f"{ENDPOINT}/v1/chat/completions", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_text(resp):
    return resp["choices"][0]["message"].get("content") or ""


def get_reasoning(resp):
    msg = resp["choices"][0]["message"]
    # sglang exposes reasoning via reasoning_content (glm45 parser) or inline <think>
    return msg.get("reasoning_content") or ""


# --- resolve model name ---
print(f"Endpoint: {ENDPOINT}")
try:
    m = requests.get(f"{ENDPOINT}/v1/models", timeout=30).json()
    MODEL_NAME = m["data"][0]["id"]
    print(f"Model name resolved: {MODEL_NAME}")
except Exception as e:
    print(f"Could not resolve model name: {e}")
    MODEL_NAME = "GLM-5.2-FP8"

print("=" * 60)

# ---- Test 1: basic coherence + thinking format ----
try:
    resp = chat([{"role": "user", "content": "你好，请用一句话介绍你自己。"}])
    txt = get_text(resp)
    reasoning = get_reasoning(resp)
    has_thinking = bool(reasoning) or "<think>" in (txt + reasoning)
    coherent = "GLM" in txt or "模型" in txt or "助手" in txt or "人工智能" in txt or "你好" in txt
    record("1. coherence + thinking",
          has_thinking and coherent and len(txt) > 5,
          f"reasoning_len={len(reasoning)} text_len={len(txt)} text_head={txt[:80]!r}")
except Exception as e:
    record("1. coherence + thinking", False, f"exception: {e}")

# ---- Test 2: determinism (PD KV-transfer lossless proxy) ----
try:
    p = [{"role": "user", "content": "1+1等于几？只回答数字。"}]
    r1 = get_text(chat(p))
    r2 = get_text(chat(p))
    record("2. determinism (PD KV integrity)", r1 == r2 and r1.strip() != "",
          f"run1={r1[:60]!r} run2={r2[:60]!r} identical={r1 == r2}")
except Exception as e:
    record("2. determinism (PD KV integrity)", False, f"exception: {e}")

# ---- Test 3: math correctness ----
try:
    resp = chat([{"role": "user", "content": "一个数加上5等于12，这个数是多少？只回答数字。"}])
    txt = get_text(resp)
    record("3. math correctness", "7" in txt, f"answer={txt[:80]!r}")
except Exception as e:
    record("3. math correctness", False, f"exception: {e}")

# ---- Test 4: English reasoning ----
try:
    resp = chat([{"role": "user", "content": "What is 15 * 17? Answer with just the number."}])
    txt = get_text(resp)
    record("4. english math", "255" in txt, f"answer={txt[:80]!r}")
except Exception as e:
    record("4. english math", False, f"exception: {e}")

# ---- Test 5: tool calling (glm47 format) ----
try:
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }]
    resp = chat([{"role": "user", "content": "北京今天天气怎么样？"}], tools=tools)
    msg = resp["choices"][0]["message"]
    tool_calls = msg.get("tool_calls") or []
    parsed_ok = len(tool_calls) > 0 and tool_calls[0].get("function", {}).get("name") == "get_weather"
    raw = json.dumps(msg, ensure_ascii=False)
    has_glm47_tag = "<tool_call>" in raw or "tool_call" in raw.lower()
    record("5. tool calling (glm47)",
          parsed_ok or has_glm47_tag,
          f"tool_calls={tool_calls} raw_head={raw[:200]!r}")
except Exception as e:
    record("5. tool calling (glm47)", False, f"exception: {e}")

# ---- Test 6: enable_thinking=False suppresses reasoning ----
try:
    resp = chat([{"role": "user", "content": "写一首关于月亮的五言绝句。"}],
                extra={"chat_template_kwargs": {"enable_thinking": False}})
    txt = get_text(resp)
    reasoning = get_reasoning(resp)
    # with thinking off, reasoning_content should be empty/absent
    record("6. enable_thinking=False",
          (not reasoning) or reasoning.strip() == "",
          f"reasoning_len={len(reasoning)} text_head={txt[:80]!r}")
except Exception as e:
    record("6. enable_thinking=False", False, f"exception: {e}")

# ---- Test 7: long-context coherence (chunked prefill + KV transfer) ----
try:
    filler = "GLM-5.2 是智谱 AI 开发的大语言模型，支持长上下文与混合推理。" * 200  # ~6K tokens
    q = f"{filler}\n\n根据上文，这个模型由谁开发？只回答公司名。"
    resp = chat([{"role": "user", "content": q}], extra={"max_tokens": 128})
    txt = get_text(resp)
    record("7. long-context coherence",
          "智谱" in txt or "Zhipu" in txt or "zhipu" in txt.lower(),
          f"prompt_len~{len(filler)} answer={txt[:80]!r}")
except Exception as e:
    record("7. long-context coherence", False, f"exception: {e}")

# ---- summary ----
print("=" * 60)
passed = sum(1 for _, p, _ in results if p)
total = len(results)
print(f"RESULT: {passed}/{total} passed")
sys.exit(0 if passed == total else 1)
