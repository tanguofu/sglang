#!/usr/bin/env python3
"""Test reasoning_effort levels and context length on glm52-1tp8."""
import json
import subprocess
import time
import sys

API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
BASE = "https://glm52-1tp8.jmpti.woa.com"


def curl_post(endpoint, payload, timeout=120):
    """POST and return (status_code, total_time, body)."""
    cmd = [
        "curl", "-s",
        "-w", "\n%{http_code}|%{time_total}",
        "-X", "POST",
        f"{BASE}{endpoint}",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {API_KEY}",
        "-d", json.dumps(payload),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = result.stdout.strip()
        # Split last line as metadata
        lines = output.rsplit("\n", 1)
        if len(lines) == 2:
            body, meta = lines
        else:
            body, meta = "", lines[0]
        parts = meta.split("|")
        status = int(parts[0]) if parts and parts[0].isdigit() else 0
        total = float(parts[1]) if len(parts) > 1 else 0
        return status, total, body
    except subprocess.TimeoutExpired:
        return 0, timeout, "TIMEOUT"
    except Exception as e:
        return 0, 0, str(e)


def test_reasoning_effort():
    """Test different reasoning_effort levels on /v1/chat/completions."""
    print("=" * 70)
    print("  REASONING EFFORT TEST: none / low / high / max")
    print("=" * 70)

    prompt = "What is 15*37? Show your reasoning."

    for effort in ["none", "low", "high", "max"]:
        payload = {
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "reasoning_effort": effort,
        }
        print(f"\n  --- reasoning_effort: {effort} ---")
        status, total, body = curl_post("/v1/chat/completions", payload)

        if status != 200:
            print(f"    HTTP {status} | {total:.3f}s | body: {body[:200]}")
            continue

        try:
            d = json.loads(body)
            u = d.get("usage", {})
            c = d["choices"][0]["message"]
            rc = c.get("reasoning_content", "") or ""
            content = c.get("content", "") or ""
            print(f"    HTTP {status} | {total:.3f}s | finish={d['choices'][0]['finish_reason']}")
            print(f"    tokens: prompt={u.get('prompt_tokens','?')} comp={u.get('completion_tokens','?')} reason={u.get('reasoning_tokens','?')}")
            print(f"    reasoning_content: {'YES' if rc else 'NO'} ({len(rc)} chars)")
            if rc:
                print(f"    reasoning preview: {rc[:150]}...")
            print(f"    content: {content[:150]}")
        except Exception as e:
            print(f"    parse error: {e}")
            print(f"    raw: {body[:300]}")

    # Also test on /v1/responses
    print("\n" + "=" * 70)
    print("  REASONING EFFORT on /v1/responses")
    print("=" * 70)

    for effort in ["none", "high", "max"]:
        payload = {
            "model": "glm-5.2",
            "input": prompt,
            "max_output_tokens": 512,
            "reasoning": {"effort": effort},
        }
        print(f"\n  --- reasoning.effort: {effort} ---")
        status, total, body = curl_post("/v1/responses", payload)

        if status != 200:
            print(f"    HTTP {status} | {total:.3f}s | body: {body[:200]}")
            continue

        try:
            d = json.loads(body)
            u = d.get("usage", {})
            outputs = d.get("output", [])
            reasoning_text = ""
            content_text = ""
            for o in outputs:
                if o.get("type") == "reasoning":
                    for c in o.get("content", []):
                        reasoning_text += c.get("text", "")
                elif o.get("type") == "message":
                    for c in o.get("content", []):
                        content_text += c.get("text", "")
            print(f"    HTTP {status} | {total:.3f}s | status={d.get('status','?')}")
            print(f"    tokens: prompt={u.get('prompt_tokens','?')} comp={u.get('completion_tokens','?')} reason={u.get('reasoning_tokens','?')}")
            print(f"    reasoning: {'YES' if reasoning_text else 'NO'} ({len(reasoning_text)} chars)")
            if reasoning_text:
                print(f"    reasoning preview: {reasoning_text[:150]}...")
            print(f"    content: {content_text[:150]}")
        except Exception as e:
            print(f"    parse error: {e}")
            print(f"    raw: {body[:300]}")


def test_context_length():
    """Test context length limits with progressively longer prompts."""
    print("\n" + "=" * 70)
    print("  CONTEXT LENGTH TEST")
    print("=" * 70)

    # Server config: context_length=524288, max_total_num_tokens=490240
    print(f"  Server config: context_length=524288, max_total_num_tokens=490240")
    print(f"  max_req_input_len=490234")
    print()

    # Test with progressively longer prompts (using token estimation ~0.75 words per token)
    for num_repeats in [100, 500, 1000, 2000]:
        text = "The quick brown fox jumps over the lazy dog. " * num_repeats
        est_tokens = int(len(text.split()) / 0.75)

        payload = {
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": f"Summarize in 1 sentence: {text}"}],
            "max_tokens": 64,
            "stream": True,
        }
        print(f"  --- ~{est_tokens} tokens (repeats={num_repeats}) ---")

        # Use streaming to avoid timeout
        tmpfile = "/tmp/ctx_test_output.txt"
        cmd = [
            "curl", "-s", "-o", tmpfile,
            "-w", "%{http_code}|%{time_total}|%{time_starttransfer}",
            "-X", "POST",
            f"{BASE}/v1/chat/completions",
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {API_KEY}",
            "-d", json.dumps(payload),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            parts = result.stdout.strip().split("|")
            status = int(parts[0]) if parts and parts[0].isdigit() else 0
            total = float(parts[1]) if len(parts) > 1 else 0
            ttfb = float(parts[2]) if len(parts) > 2 else 0

            if status == 200:
                # Read the streaming output
                with open(tmpfile, "r") as f:
                    stream_data = f.read()
                chunks = stream_data.count("\ndata:")
                # Try to get usage from last chunk
                usage_line = ""
                for line in stream_data.split("\n"):
                    if "usage" in line and line.startswith("data: "):
                        usage_line = line[6:]
                if usage_line:
                    try:
                        d = json.loads(usage_line)
                        u = d.get("usage", {})
                        print(f"    HTTP {status} | {total:.1f}s | TTFB {ttfb:.1f}s | chunks={chunks}")
                        print(f"    tokens: prompt={u.get('prompt_tokens','?')} comp={u.get('completion_tokens','?')}")
                    except:
                        print(f"    HTTP {status} | {total:.1f}s | TTFB {ttfb:.1f}s | chunks={chunks}")
                else:
                    print(f"    HTTP {status} | {total:.1f}s | TTFB {ttfb:.1f}s | chunks={chunks}")
            else:
                # Read error
                try:
                    with open(tmpfile, "r") as f:
                        err = f.read()[:200]
                except:
                    err = "(no body)"
                print(f"    HTTP {status} | {total:.1f}s | error: {err}")
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT after 300s")
        except Exception as e:
            print(f"    ERROR: {e}")


def test_chat_template_kwargs():
    """Test enable_thinking via chat_template_kwargs."""
    print("\n" + "=" * 70)
    print("  CHAT TEMPLATE KWARGS: enable_thinking=true/false")
    print("=" * 70)

    prompt = "What is 15*37?"

    for enable in [True, False]:
        payload = {
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": enable},
        }
        print(f"\n  --- enable_thinking: {enable} ---")
        status, total, body = curl_post("/v1/chat/completions", payload)

        if status != 200:
            print(f"    HTTP {status} | {total:.3f}s | body: {body[:200]}")
            continue

        try:
            d = json.loads(body)
            u = d.get("usage", {})
            c = d["choices"][0]["message"]
            rc = c.get("reasoning_content", "") or ""
            print(f"    HTTP {status} | {total:.3f}s | finish={d['choices'][0]['finish_reason']}")
            print(f"    tokens: comp={u.get('completion_tokens','?')} reason={u.get('reasoning_tokens','?')}")
            print(f"    reasoning_content: {'YES' if rc else 'NO'} ({len(rc)} chars)")
            print(f"    content: {(c.get('content','') or '')[:150]}")
        except Exception as e:
            print(f"    parse error: {e}")
            print(f"    raw: {body[:300]}")


if __name__ == "__main__":
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    for i in range(120):
        status, _, _ = curl_post("/health", {}, timeout=10)
        # /health might not accept POST; use GET
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"{BASE}/health"],
            capture_output=True, text=True, timeout=10
        )
        if result.stdout.strip() == "200":
            print(f"  Server ready after {i*5}s")
            break
        time.sleep(5)
    else:
        print("  Server not ready after 600s, continuing anyway...")

    test_reasoning_effort()
    test_chat_template_kwargs()
    test_context_length()

    print("\n" + "=" * 70)
    print("  ALL TESTS COMPLETE")
    print("=" * 70)
