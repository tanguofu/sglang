#!/usr/bin/env python3
"""Robust effort test — re-run each effort level 3 times."""
import json, subprocess, time

GATEWAY = "https://glm52-2tp8.jmpti.woa.com"
TOKEN = "${ANTHROPIC_AUTH_TOKEN}"
PROMPT = "Write a Python function to check if a string is a palindrome. Just the code, no explanation."

def test_effort(effort, max_tokens=600):
    body = json.dumps({
        "model": "glm-5.2",
        "input": PROMPT,
        "max_output_tokens": max_tokens,
        "reasoning": {"effort": effort},
        "stream": True,
    })
    start = time.perf_counter()
    proc = subprocess.Popen([
        "/usr/bin/curl", "-sS", "-N", "--max-time", "120",
        f"{GATEWAY}/v1/responses",
        "-H", f"Authorization: Bearer {TOKEN}",
        "-H", "Content-Type: application/json",
        "-d", body,
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    reason_count = 0
    output_count = 0
    error = None
    final_usage = None

    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("event: "):
            etype = line[7:]
            if etype == "response.reasoning_text.delta":
                reason_count += 1
            elif etype == "response.output_text.delta":
                output_count += 1
        elif line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                evt = json.loads(data)
                if "error" in evt and isinstance(evt["error"], (dict, str)):
                    error = evt["error"]
                elif evt.get("type") == "response.completed":
                    final_usage = evt.get("response", {}).get("usage")
            except json.JSONDecodeError:
                pass

    proc.wait()
    total = time.perf_counter() - start
    return {
        "total_s": round(total, 2),
        "reason_deltas": reason_count,
        "output_deltas": output_count,
        "error": error,
        "usage": final_usage,
    }

print(f"Prompt: {PROMPT[:60]}...")
print(f"{'effort':<10} {'run':<5} {'total(s)':<10} {'reason':<8} {'output':<8} {'error/usage'}")
print("-" * 80)

for effort in ["low", "medium", "high"]:
    for run in range(1, 4):
        r = test_effort(effort)
        err_or_usage = ""
        if r["error"]:
            err_or_usage = f"ERROR: {str(r['error'])[:60]}"
        elif r["usage"]:
            u = r["usage"]
            err_or_usage = f"in={u.get('input_tokens')} out={u.get('output_tokens')}"
        print(f"{effort:<10} {run:<5} {r['total_s']:<10} {r['reason_deltas']:<8} {r['output_deltas']:<8} {err_or_usage}")
        time.sleep(1)  # avoid hammering
