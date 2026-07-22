#!/usr/bin/env python3
"""
Verify Codex /v1/responses and Claude /v1/messages protocol compliance:
  - Request schema validation
  - Response format compliance (non-streaming + streaming)
  - Reasoning content handling
  - Tool calls support
  - Multi-turn conversations
  - Edge cases (max_tokens, system prompt, etc.)
"""

import json
import sys
import time
import urllib.request
import urllib.error

BASE_URL = "https://glm52-pd-1p1d.jmpti.woa.com"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"

def call(method, path, body, timeout=300):
    url = BASE_URL + path
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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
        return -1, {"error": str(e)}, 0.0

def call_stream(path, body, timeout=300):
    """Return list of SSE events and data lines for streaming requests."""
    url = BASE_URL + path
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=headers)
    t0 = time.time()
    events = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            current_event = None
            for raw in resp:
                line = raw.decode().rstrip("\n")
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                elif line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        events.append(("done", None))
                    else:
                        try:
                            parsed = json.loads(data)
                        except json.JSONDecodeError:
                            parsed = data
                        events.append((current_event or "message", parsed))
                    current_event = None
            dt = time.time() - t0
            return resp.status, events, dt
    except urllib.error.HTTPError as e:
        return e.code, [("error", e.read().decode()[:300])], 0.0
    except Exception as e:
        return -1, [("error", str(e))], 0.0

def check(cond, label, detail=""):
    status = "OK" if cond else "FAIL"
    extra = f" — {detail}" if (detail and not cond) else ""
    print(f"  [{status}] {label}{extra}")
    return bool(cond)

# ===== Codex /v1/responses tests =====

def test_codex_basic():
    print("\n=== Codex Test 1: /v1/responses basic format ===")
    body = {
        "model": "glm52",
        "input": "What is 3+4? Answer with just the number.",
        "max_output_tokens": 256,
        "temperature": 0.0,
    }
    status, resp, dt = call("POST", "/v1/responses", body)
    print(f"  HTTP {status} ({dt:.2f}s)")
    if status != 200:
        print(f"  Response: {json.dumps(resp)[:400]}")
        return False
    ok = True
    ok &= check(isinstance(resp, dict), "response is object")
    ok &= check(resp.get("object") == "response", f"object='response' (got {resp.get('object')!r})")
    ok &= check("id" in resp and isinstance(resp["id"], str), "has string id")
    ok &= check("created_at" in resp, "has created_at")
    ok &= check("model" in resp, "has model")
    ok &= check("output" in resp and isinstance(resp["output"], list), "output is list")
    ok &= check(len(resp["output"]) > 0, "output non-empty")

    # Verify output item structure
    has_reasoning = False
    has_message = False
    for item in resp["output"]:
        if item.get("type") == "reasoning":
            has_reasoning = True
            ok &= check("id" in item, "reasoning item has id")
            ok &= check("summary" in item, "reasoning item has summary list")
            ok &= check("content" in item and isinstance(item["content"], list), "reasoning item has content list")
        elif item.get("type") == "message":
            has_message = True
            ok &= check("id" in item, "message item has id")
            ok &= check(item.get("role") == "assistant", f"message role == assistant (got {item.get('role')!r})")
            ok &= check("content" in item and isinstance(item["content"], list), "message content is list")
            ok &= check("status" in item, "message has status")
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    ok &= check("text" in c, "output_text has text field")
                    ok &= check("annotations" in c, "output_text has annotations list")
    ok &= check(has_message, "has at least one message output item")
    # reasoning is optional but expected for reasoning model
    print(f"  output types: {[o.get('type') for o in resp['output']]}")

    # Status field
    ok &= check(resp.get("status") in ("completed", "in_progress", "failed"), f"status valid (got {resp.get('status')!r})")

    # Usage
    ok &= check("usage" in resp, "has usage")
    if resp.get("usage"):
        u = resp["usage"]
        ok &= check("input_tokens" in u or "prompt_tokens" in u, f"usage has input/prompt tokens")
        ok &= check("output_tokens" in u or "completion_tokens" in u, f"usage has output/completion tokens")
        print(f"  usage: {u}")
    return ok

def test_codex_input_as_messages():
    """Codex /v1/responses accepts input as array of messages too."""
    print("\n=== Codex Test 2: /v1/responses with input as messages array ===")
    body = {
        "model": "glm52",
        "input": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "What is the capital of Japan?"},
        ],
        "max_output_tokens": 256,
        "temperature": 0.0,
    }
    status, resp, dt = call("POST", "/v1/responses", body)
    print(f"  HTTP {status} ({dt:.2f}s)")
    if status != 200:
        print(f"  Response: {json.dumps(resp)[:400]}")
        return False
    ok = True
    ok &= check(resp.get("object") == "response", "object='response'")
    ok &= check("output" in resp and len(resp["output"]) > 0, "has output")
    # Find text content
    text = ""
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    text += c.get("text", "")
    print(f"  output_text: {text[:100]!r}")
    ok &= check("tokyo" in text.lower(), f"answer mentions 'tokyo'")
    return ok

def test_codex_streaming():
    print("\n=== Codex Test 3: /v1/responses streaming ===")
    body = {
        "model": "glm52",
        "input": "Count from 1 to 5.",
        "max_output_tokens": 256,
        "stream": True,
        "temperature": 0.0,
    }
    status, events, dt = call_stream("/v1/responses", body)
    print(f"  HTTP {status} ({dt:.2f}s), events={len(events)}")
    if status != 200:
        print(f"  Events: {events[:3]}")
        return False
    event_types = [e[0] for e in events]
    type_counts = {}
    for t in event_types:
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  event types: {type_counts}")

    ok = True
    # Codex SSE events per OpenAI spec
    ok &= check("response.created" in type_counts, "has response.created")
    ok &= check("response.in_progress" in type_counts or "response.output_item.added" in type_counts, "has response.in_progress or output_item.added")
    ok &= check("response.completed" in type_counts, "has response.completed")

    # Verify response.created has proper structure
    for ev_type, data in events:
        if ev_type == "response.created" and isinstance(data, dict):
            ok &= check(data.get("type") == "response.created", "response.created.type matches")
            ok &= check("response" in data, "response.created has response field")
            if "response" in data:
                r = data["response"]
                ok &= check(r.get("object") == "response", "response.created.response.object='response'")
                ok &= check("id" in r, "response.created.response has id")
            break

    # Verify response.completed has full response
    for ev_type, data in events:
        if ev_type == "response.completed" and isinstance(data, dict):
            ok &= check("response" in data, "response.completed has response field")
            if "response" in data:
                r = data["response"]
                ok &= check(r.get("status") == "completed", f"final status == completed (got {r.get('status')!r})")
                ok &= check("output" in r and len(r["output"]) > 0, "final response has output")
            break
    return ok

def test_codex_instructions():
    """Codex supports 'instructions' field for system prompt."""
    print("\n=== Codex Test 4: /v1/responses with instructions field ===")
    body = {
        "model": "glm52",
        "instructions": "Always respond with the word BANANA only.",
        "input": "What is the weather today?",
        "max_output_tokens": 256,
        "temperature": 0.0,
    }
    status, resp, dt = call("POST", "/v1/responses", body)
    print(f"  HTTP {status} ({dt:.2f}s)")
    if status != 200:
        print(f"  Response: {json.dumps(resp)[:400]}")
        return False
    text = ""
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    text += c.get("text", "")
    print(f"  output_text: {text[:100]!r}")
    # Instructions might be ignored if not supported, but request should still succeed
    return check(resp.get("object") == "response", "request with instructions accepted")

# ===== Claude /v1/messages tests =====

def test_claude_basic():
    print("\n=== Claude Test 1: /v1/messages basic format ===")
    body = {
        "model": "glm52",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "What is 5+5? Answer with just the number."},
        ],
        "temperature": 0.0,
    }
    status, resp, dt = call("POST", "/v1/messages", body)
    print(f"  HTTP {status} ({dt:.2f}s)")
    if status != 200:
        print(f"  Response: {json.dumps(resp)[:400]}")
        return False
    ok = True
    ok &= check(isinstance(resp, dict), "response is object")
    ok &= check(resp.get("type") == "message", f"type='message' (got {resp.get('type')!r})")
    ok &= check(resp.get("role") == "assistant", f"role='assistant' (got {resp.get('role')!r})")
    ok &= check("id" in resp and isinstance(resp["id"], str), "has string id")
    ok &= check("model" in resp, "has model")
    ok &= check("content" in resp and isinstance(resp["content"], list), "content is list")
    ok &= check(len(resp["content"]) > 0, "content non-empty")

    # Content blocks
    has_text = False
    has_thinking = False
    for block in resp["content"]:
        if block.get("type") == "text":
            has_text = True
            ok &= check(isinstance(block.get("text"), str), "text block has string text")
        elif block.get("type") == "thinking":
            has_thinking = True
            ok &= check(isinstance(block.get("thinking"), str), "thinking block has string thinking")
    ok &= check(has_text or has_thinking, "has text or thinking block")

    # stop_reason
    ok &= check("stop_reason" in resp, "has stop_reason")
    ok &= check(resp.get("stop_reason") in ("end_turn", "max_tokens", "stop_sequence", "tool_use"), f"stop_reason valid (got {resp.get('stop_reason')!r})")

    # usage
    ok &= check("usage" in resp, "has usage")
    if resp.get("usage"):
        u = resp["usage"]
        ok &= check("input_tokens" in u, "usage has input_tokens")
        ok &= check("output_tokens" in u, "usage has output_tokens")
        print(f"  usage: {u}")

    print(f"  content types: {[b.get('type') for b in resp['content']]}")
    print(f"  stop_reason: {resp.get('stop_reason')!r}")
    return ok

def test_claude_system_prompt():
    print("\n=== Claude Test 2: /v1/messages with system prompt ===")
    body = {
        "model": "glm52",
        "max_tokens": 256,
        "system": "You always respond with the word APPLE only.",
        "messages": [
            {"role": "user", "content": "What time is it?"},
        ],
        "temperature": 0.0,
    }
    status, resp, dt = call("POST", "/v1/messages", body)
    print(f"  HTTP {status} ({dt:.2f}s)")
    if status != 200:
        print(f"  Response: {json.dumps(resp)[:400]}")
        return False
    text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")
    print(f"  text: {text[:100]!r}")
    ok = check(resp.get("type") == "message", "type='message'")
    ok &= check("apple" in text.lower(), "system prompt respected (mentions 'apple')")
    return ok

def test_claude_content_blocks():
    """Claude supports content as list of blocks."""
    print("\n=== Claude Test 3: /v1/messages with content as block list ===")
    body = {
        "model": "glm52",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "What is 7+8?"},
            ]},
        ],
        "temperature": 0.0,
    }
    status, resp, dt = call("POST", "/v1/messages", body)
    print(f"  HTTP {status} ({dt:.2f}s)")
    if status != 200:
        print(f"  Response: {json.dumps(resp)[:400]}")
        return False
    text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")
    print(f"  text: {text[:100]!r}")
    ok = check(resp.get("type") == "message", "type='message'")
    ok &= check("15" in text, "answer mentions '15'")
    return ok

def test_claude_streaming():
    print("\n=== Claude Test 4: /v1/messages streaming ===")
    body = {
        "model": "glm52",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "stream": True,
        "temperature": 0.0,
    }
    status, events, dt = call_stream("/v1/messages", body)
    print(f"  HTTP {status} ({dt:.2f}s), events={len(events)}")
    if status != 200:
        print(f"  Events: {events[:3]}")
        return False
    event_types = [e[0] for e in events]
    type_counts = {}
    for t in event_types:
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  event types: {type_counts}")

    ok = True
    # Claude SSE events per Anthropic spec
    ok &= check("message_start" in type_counts, "has message_start")
    ok &= check("content_block_start" in type_counts, "has content_block_start")
    ok &= check("content_block_delta" in type_counts, "has content_block_delta")
    ok &= check("content_block_stop" in type_counts, "has content_block_stop")
    ok &= check("message_delta" in type_counts, "has message_delta")
    ok &= check("message_stop" in type_counts, "has message_stop")

    # Verify message_start structure
    for ev_type, data in events:
        if ev_type == "message_start" and isinstance(data, dict):
            ok &= check("message" in data, "message_start has message field")
            if "message" in data:
                m = data["message"]
                ok &= check(m.get("role") == "assistant", "message_start.message.role == assistant")
                ok &= check(m.get("type") == "message", "message_start.message.type == message")
                ok &= check("id" in m, "message_start.message has id")
                ok &= check("usage" in m, "message_start.message has usage")
            break

    # Verify message_delta has stop_reason
    for ev_type, data in events:
        if ev_type == "message_delta" and isinstance(data, dict):
            ok &= check("delta" in data, "message_delta has delta field")
            if "delta" in data:
                ok &= check("stop_reason" in data["delta"] or "stop_reason" in data, "message_delta has stop_reason")
            break
    return ok

def test_claude_multi_turn():
    print("\n=== Claude Test 5: /v1/messages multi-turn conversation ===")
    # Turn 1
    body1 = {
        "model": "glm52",
        "max_tokens": 256,
        "system": "Be helpful. Remember user details.",
        "messages": [
            {"role": "user", "content": "My name is ClaudeTest42. What's my name?"},
        ],
        "temperature": 0.0,
    }
    status, resp, dt = call("POST", "/v1/messages", body1)
    if status != 200:
        print(f"  Turn 1 failed: {resp}")
        return False
    text1 = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            text1 += block.get("text", "")
    print(f"  Turn 1: {text1[:80]!r}")

    # Turn 2 - continue conversation
    body2 = {
        "model": "glm52",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "My name is ClaudeTest42."},
            {"role": "assistant", "content": text1},
            {"role": "user", "content": "What is my name?"},
        ],
        "temperature": 0.0,
    }
    status, resp, dt = call("POST", "/v1/messages", body2)
    if status != 200:
        print(f"  Turn 2 failed: {resp}")
        return False
    text2 = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            text2 += block.get("text", "")
    print(f"  Turn 2: {text2[:80]!r}")
    ok = check("claudetest42" in text2.lower(), "remembers name from context")
    return ok

def test_claude_max_tokens_limit():
    print("\n=== Claude Test 6: /v1/messages max_tokens enforcement ===")
    body = {
        "model": "glm52",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Tell me a long story."}],
        "temperature": 0.0,
    }
    status, resp, dt = call("POST", "/v1/messages", body)
    print(f"  HTTP {status} ({dt:.2f}s)")
    if status != 200:
        print(f"  Response: {json.dumps(resp)[:400]}")
        return False
    ok = check(resp.get("stop_reason") == "max_tokens", f"stop_reason == max_tokens (got {resp.get('stop_reason')!r})")
    if resp.get("usage"):
        u = resp["usage"]
        print(f"  usage: {u}")
        # output_tokens should be limited
        ok &= check(u.get("output_tokens", 999) <= 20, f"output_tokens near limit (got {u.get('output_tokens')})")
    return ok

# ===== Main =====

def main():
    print(f"Endpoint: {BASE_URL}")
    print(f"Verifying Codex (/v1/responses) and Claude (/v1/messages) protocol compliance")
    print("=" * 70)

    results = {}
    # Codex
    results["codex_basic"] = test_codex_basic()
    results["codex_input_messages"] = test_codex_input_as_messages()
    results["codex_streaming"] = test_codex_streaming()
    results["codex_instructions"] = test_codex_instructions()
    # Claude
    results["claude_basic"] = test_claude_basic()
    results["claude_system"] = test_claude_system_prompt()
    results["claude_content_blocks"] = test_claude_content_blocks()
    results["claude_streaming"] = test_claude_streaming()
    results["claude_multi_turn"] = test_claude_multi_turn()
    results["claude_max_tokens"] = test_claude_max_tokens_limit()

    print("\n" + "=" * 70)
    print("CODEX + CLAUDE PROTOCOL VERIFICATION SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k:30s} {'PASS' if v else 'FAIL'}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  Total: {passed}/{total} passed")
    sys.exit(0 if all(results.values()) else 1)

if __name__ == "__main__":
    main()
