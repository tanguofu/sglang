#!/usr/bin/env python3
"""
Verify router API format support for:
  1. OpenAI /v1/chat/completions
  2. Codex /v1/responses
  3. Claude /v1/messages

Checks both request acceptance and response schema compliance.
"""

import json
import sys
import time
import urllib.request
import urllib.error

BASE_URL = "https://glm52-pd-1p1d.jmpti.woa.com"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"

def call(method, path, body):
    url = BASE_URL + path
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = resp.read().decode()
            dt = time.time() - t0
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                parsed = payload
            return resp.status, parsed, dt
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        return e.code, parsed, 0.0
    except Exception as e:
        return -1, str(e), 0.0

def check(cond, label):
    print(f"  [{'OK' if cond else 'FAIL'}] {label}")
    return bool(cond)

def test_chat_completions():
    print("\n=== Test 1: OpenAI /v1/chat/completions ===")
    body = {
        "model": "glm52",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer directly, no reasoning."},
            {"role": "user", "content": "What is 2+2? Answer briefly."},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }
    status, resp, dt = call("POST", "/v1/chat/completions", body)
    print(f"  HTTP status: {status} ({dt:.2f}s)")
    if status != 200:
        print(f"  Response: {json.dumps(resp)[:300]}")
        return False
    ok = True
    ok &= check(isinstance(resp, dict), "response is object")
    ok &= check(resp.get("object") == "chat.completion", f"object='chat.completion' (got {resp.get('object')!r})")
    ok &= check("id" in resp, "has id")
    ok &= check("created" in resp, "has created")
    ok &= check("model" in resp, "has model")
    ok &= check("choices" in resp and len(resp["choices"]) > 0, "has non-empty choices")
    if resp.get("choices"):
        ch = resp["choices"][0]
        ok &= check(ch.get("index") == 0, "choices[0].index == 0")
        ok &= check(ch.get("finish_reason") in ("stop", "length", "tool_calls"), f"finish_reason valid (got {ch.get('finish_reason')!r})")
        ok &= check("message" in ch, "choice has message")
        if ch.get("message"):
            msg = ch["message"]
            ok &= check(msg.get("role") == "assistant", "message.role == assistant")
            # For reasoning models, content may be empty if reasoning_content is used
            has_content = isinstance(msg.get("content"), str) and len(msg["content"]) > 0
            has_reasoning = isinstance(msg.get("reasoning_content"), str) and len(msg["reasoning_content"]) > 0
            ok &= check(has_content or has_reasoning, f"has content or reasoning_content (content={bool(has_content)}, reasoning={bool(has_reasoning)})")
            if has_reasoning:
                print(f"  reasoning_content (first 100): {msg['reasoning_content'][:100]!r}")
    ok &= check("usage" in resp, "has usage")
    if resp.get("usage"):
        u = resp["usage"]
        ok &= check("prompt_tokens" in u, "usage.prompt_tokens")
        ok &= check("completion_tokens" in u, "usage.completion_tokens")
        ok &= check("total_tokens" in u, "usage.total_tokens")
        print(f"  usage: {u}")
    if resp.get("choices"):
        content = resp["choices"][0]["message"].get("content", "")
        print(f"  content (first 100): {content[:100]!r}")
    return ok

def test_responses_codex():
    print("\n=== Test 2: Codex /v1/responses ===")
    body = {
        "model": "glm52",
        "input": "What is 2+2? Answer briefly.",
        "max_output_tokens": 512,
        "temperature": 0.0,
    }
    status, resp, dt = call("POST", "/v1/responses", body)
    print(f"  HTTP status: {status} ({dt:.2f}s)")
    if status != 200:
        print(f"  Response: {json.dumps(resp)[:400]}")
        return False
    ok = True
    ok &= check(isinstance(resp, dict), "response is object")
    ok &= check(resp.get("object") == "response", f"object='response' (got {resp.get('object')!r})")
    ok &= check("id" in resp, "has id")
    ok &= check("created_at" in resp, "has created_at")
    ok &= check("model" in resp, "has model")
    ok &= check("output" in resp and isinstance(resp["output"], list), "output is list")
    if resp.get("output"):
        found_text = False
        found_reasoning = False
        for item in resp["output"]:
            if item.get("type") == "message":
                ok &= check(item.get("role") in ("assistant", "user"), f"output message role (got {item.get('role')!r})")
                ok &= check("content" in item, "output message has content")
                if item.get("content"):
                    for c in item["content"]:
                        if c.get("type") == "output_text" and c.get("text"):
                            found_text = True
            elif item.get("type") == "reasoning":
                found_reasoning = True
                ok &= check("content" in item and isinstance(item["content"], list), "reasoning has content list")
        # Either text or reasoning should be present
        ok &= check(found_text or found_reasoning, f"found output_text or reasoning (text={found_text}, reasoning={found_reasoning})")
    ok &= check("usage" in resp, "has usage")
    if resp.get("usage"):
        u = resp["usage"]
        # Codex usage may use input_tokens/output_tokens OR prompt_tokens/completion_tokens
        ok &= check("input_tokens" in u or "prompt_tokens" in u, f"usage has input/prompt tokens (keys={list(u.keys())})")
        ok &= check("output_tokens" in u or "completion_tokens" in u, f"usage has output/completion tokens (keys={list(u.keys())})")
        print(f"  usage: {u}")
    print(f"  output types: {[o.get('type') for o in resp.get('output', [])]}")
    return ok

def test_messages_claude():
    print("\n=== Test 3: Claude /v1/messages ===")
    body = {
        "model": "glm52",
        "max_tokens": 512,
        "system": "You are a helpful assistant. Answer directly, no reasoning.",
        "messages": [
            {"role": "user", "content": "What is 2+2? Answer briefly."},
        ],
        "temperature": 0.0,
    }
    status, resp, dt = call("POST", "/v1/messages", body)
    print(f"  HTTP status: {status} ({dt:.2f}s)")
    if status != 200:
        print(f"  Response: {json.dumps(resp)[:400]}")
        return False
    ok = True
    ok &= check(isinstance(resp, dict), "response is object")
    ok &= check(resp.get("type") == "message", f"type='message' (got {resp.get('type')!r})")
    ok &= check(resp.get("role") == "assistant", f"role == assistant (got {resp.get('role')!r})")
    ok &= check("id" in resp, "has id")
    ok &= check("model" in resp, "has model")
    ok &= check("content" in resp and isinstance(resp["content"], list), "content is list")
    if resp.get("content"):
        text_found = False
        thinking_found = False
        for block in resp["content"]:
            if block.get("type") == "text":
                text_found = True
                ok &= check(isinstance(block.get("text"), str) and len(block["text"]) > 0, "text block has non-empty text")
            elif block.get("type") == "thinking":
                thinking_found = True
                ok &= check(isinstance(block.get("thinking"), str), "thinking block has thinking string")
        ok &= check(text_found or thinking_found, f"found text or thinking block (text={text_found}, thinking={thinking_found})")
    ok &= check("stop_reason" in resp, "has stop_reason")
    # stop_sequence may not be present (optional)
    ok &= check("usage" in resp, "has usage")
    if resp.get("usage"):
        u = resp["usage"]
        ok &= check("input_tokens" in u, "usage.input_tokens")
        ok &= check("output_tokens" in u, "usage.output_tokens")
        print(f"  usage: {u}")
    print(f"  content types: {[b.get('type') for b in resp.get('content', [])]}")
    print(f"  stop_reason: {resp.get('stop_reason')!r}")
    return ok

def test_chat_streaming():
    print("\n=== Test 4: OpenAI /v1/chat/completions streaming ===")
    body = {
        "model": "glm52",
        "messages": [{"role": "user", "content": "Say hello three times."}],
        "max_tokens": 32,
        "stream": True,
    }
    url = BASE_URL + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            chunks = 0
            saw_done = False
            for raw in resp:
                line = raw.decode().strip()
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(payload)
                        if chunk.get("object") == "chat.completion.chunk" and chunk.get("choices"):
                            chunks += 1
                    except json.JSONDecodeError:
                        pass
            print(f"  HTTP status: 200, chunks={chunks}, saw_done={saw_done}")
            return chunks > 0 and saw_done
    except urllib.error.HTTPError as e:
        print(f"  HTTP error: {e.code} {e.read().decode()[:200]}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_responses_streaming():
    print("\n=== Test 5: Codex /v1/responses streaming ===")
    body = {
        "model": "glm52",
        "input": "Say hello three times.",
        "max_output_tokens": 32,
        "stream": True,
    }
    url = BASE_URL + "/v1/responses"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            events = {}
            for raw in resp:
                line = raw.decode().strip()
                if line.startswith("event: "):
                    ev = line[7:].strip()
                    events[ev] = events.get(ev, 0) + 1
                elif line.startswith("data: "):
                    pass
            print(f"  HTTP status: 200, events: {events}")
            # Codex streaming response events typically include response.created, response.output_text.delta, response.completed
            return "response.created" in events and "response.completed" in events
    except urllib.error.HTTPError as e:
        print(f"  HTTP error: {e.code} {e.read().decode()[:200]}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_messages_streaming():
    print("\n=== Test 6: Claude /v1/messages streaming ===")
    body = {
        "model": "glm52",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Say hello three times."}],
        "stream": True,
    }
    url = BASE_URL + "/v1/messages"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            events = {}
            for raw in resp:
                line = raw.decode().strip()
                if line.startswith("event: "):
                    ev = line[7:].strip()
                    events[ev] = events.get(ev, 0) + 1
            print(f"  HTTP status: 200, events: {events}")
            # Claude SSE events: message_start, content_block_start, content_block_delta, content_block_stop, message_delta, message_stop
            return "message_start" in events and "message_stop" in events
    except urllib.error.HTTPError as e:
        print(f"  HTTP error: {e.code} {e.read().decode()[:200]}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    print(f"Endpoint: {BASE_URL}")
    print(f"Router image: mirrors.tencent.com/ti-platform/sglang-glm52-308x-pd-router:api-fix-0720")
    results = {}
    results["chat_completions"] = test_chat_completions()
    results["responses"] = test_responses_codex()
    results["messages"] = test_messages_claude()
    results["chat_streaming"] = test_chat_streaming()
    results["responses_streaming"] = test_responses_streaming()
    results["messages_streaming"] = test_messages_streaming()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for k, v in results.items():
        status = "PASS" if v else "FAIL"
        print(f"  {k:30s} {status}")
    sys.exit(0 if all(results.values()) else 1)

if __name__ == "__main__":
    main()
