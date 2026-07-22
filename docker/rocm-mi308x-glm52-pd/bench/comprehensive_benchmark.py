#!/usr/bin/env python3
"""Comprehensive benchmark for GLM-5.2 2tp8 merged chart deployment.

Runs three suites:
  1. Effectiveness eval (eval_glm52_v2.py) — 28 cases covering reasoning/math/code/debug/instruction/knowledge/tool_use/long_context
  2. Performance latency benchmark (bench_codex_v2.py) — TTFT/ITL/TPS across 7 scenarios
  3. tool_choice专项测试 — 18 cases testing tool_choice modes (auto/none/required/function) + ThinkingConfig adaptive

Usage: python3 /tmp/comprehensive_benchmark.py
"""
import json, subprocess, time, statistics, sys, os, re

GATEWAY = "https://glm52-2tp8.jmpti.woa.com"
TOKEN = "${ANTHROPIC_AUTH_TOKEN}"
MODEL = "glm-5.2"

def curl_post(path, body, timeout=180, stream=False):
    """POST to gateway, return parsed JSON (or raw lines if stream)."""
    cmd = [
        "/usr/bin/curl", "-sS", "--max-time", str(timeout),
        f"{GATEWAY}{path}",
        "-H", f"Authorization: Bearer {TOKEN}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(body) if not isinstance(body, str) else body,
    ]
    if stream:
        cmd.insert(2, "-N")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    if result.returncode != 0:
        return {"error": f"curl failed (rc={result.returncode}): {result.stderr[:200]}"}  # keep short
    return result.stdout

def call_chat(prompt, max_tokens=1500, tools=None, tool_choice=None, thinking=None):
    """Chat Completions API call."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if thinking:
        body["thinking"] = thinking
    out = curl_post("/v1/chat/completions", body, timeout=180)
    try:
        return json.loads(out)
    except Exception:
        return {"error": f"non-JSON: {out[:200]}"}  # keep short

def call_responses(prompt, max_tokens=1500, tools=None, tool_choice=None, thinking=None):
    """Responses API call."""
    body = {
        "model": MODEL,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        resp_tools = []
        for t in tools:
            resp_tools.append({
                "type": "function",
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "parameters": t["function"].get("parameters", {"type": "object"}),
            })
        body["tools"] = resp_tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if thinking:
        body["thinking"] = thinking
    out = curl_post("/v1/responses", body, timeout=180)
    try:
        return json.loads(out)
    except Exception:
        return {"error": f"non-JSON: {out[:200]}"}  # keep short

def stream_request(prompt, max_tokens=400):
    """Streaming Responses API for latency measurement."""
    body = json.dumps({
        "model": MODEL,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "stream": True,
    })
    start = time.perf_counter()
    proc = subprocess.Popen([
        "/usr/bin/curl", "-sS", "-N", "--max-time", "180",
        f"{GATEWAY}/v1/responses",
        "-H", f"Authorization: Bearer {TOKEN}",
        "-H", "Content-Type: application/json",
        "-d", body,
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    first_reason_time = None
    first_output_time = None
    itls = []
    last_delta_time = None
    reasoning_tokens = 0
    output_tokens = 0
    final_usage = None

    for line in proc.stdout:
        line = line.rstrip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            evt = json.loads(data)
        except json.JSONDecodeError:
            continue
        etype = evt.get("type", "")
        if etype == "response.output_text.delta":
            delta = evt.get("delta", "")
            now = time.perf_counter()
            if first_output_time is None:
                first_output_time = now
            if last_delta_time:
                itls.append(now - last_delta_time)
            last_delta_time = now
            output_tokens += 1
        elif etype == "response.reasoning_text.delta":
            delta = evt.get("delta", "")
            now = time.perf_counter()
            if first_reason_time is None:
                first_reason_time = now
            if last_delta_time:
                itls.append(now - last_delta_time)
            last_delta_time = now
            reasoning_tokens += 1
        elif etype == "response.completed":
            final_usage = evt.get("response", {}).get("usage", {})

    proc.wait()
    total = time.perf_counter() - start
    total_tokens = reasoning_tokens + output_tokens
    return {
        "total_s": round(total, 3),
        "ttft_reason_ms": round((first_reason_time - start) * 1000, 1) if first_reason_time else None,
        "ttft_output_ms": round((first_output_time - start) * 1000, 1) if first_output_time else None,
        "itl_mean_ms": round(statistics.mean(itls) * 1000, 1) if itls else None,
        "itl_p50_ms": round(statistics.median(itls) * 1000, 1) if itls else None,
        "itl_p95_ms": round(sorted(itls)[int(len(itls)*0.95)] * 1000, 1) if itls else None,
        "reason_tokens": reasoning_tokens,
        "output_tokens": output_tokens,
        "total_gen_tokens": total_tokens,
        "tps": round(total_tokens / total, 1) if total > 0 else None,
        "usage": final_usage,
    }

# ============================================================
# Suite 1: Effectiveness eval (abbreviated — key cases)
# ============================================================
EVAL_CASES = [
    {"id": "R1", "category": "reasoning", "prompt": "All cats are mammals. Tom is a cat. What is Tom? Answer in one word.", "expect_regex": r"mammal", "max_tokens": 1500},
    {"id": "R2", "category": "reasoning", "prompt": "If A > B and B > C, is A > C? Answer yes or no with one sentence explaining why.", "expect_regex": r"yes|true", "max_tokens": 1500},
    {"id": "R3", "category": "reasoning", "prompt": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Just the number.", "expect_regex": r"\b9\b", "max_tokens": 1500},
    {"id": "M1", "category": "math", "prompt": "What is 123 * 456? Just the number.", "expect_regex": r"56088", "max_tokens": 1500},
    {"id": "M2", "category": "math", "prompt": "If a train travels at 60 km/h for 2.5 hours, how far does it go? Just the number with units.", "expect_regex": r"150", "max_tokens": 1500},
    {"id": "M3", "category": "math", "prompt": "Solve for x: 2x + 5 = 17. Just the value of x.", "expect_regex": r"\b6\b", "max_tokens": 1500},
    {"id": "C1", "category": "code", "prompt": "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome (case-insensitive, ignoring non-alphanumeric). Just the code.", "expect_regex": r"def\s+is_palindrome", "max_tokens": 2500},
    {"id": "C2", "category": "code", "prompt": "Write a Python function `fib(n)` that returns the n-th Fibonacci number iteratively. Just the code.", "expect_regex": r"def\s+fib", "max_tokens": 2500},
    {"id": "D1", "category": "debug", "prompt": "Find the bug in this Python code:\n\n```python\ndef count_words(text):\n    words = text.split()\n    count = 0\n    for word in words:\n        count = count + 1\n        return count\n```\n\nWhat's the bug? Give the fix in one sentence.", "expect_regex": r"return|indent", "max_tokens": 2000},
    {"id": "I1", "category": "instruction", "prompt": "List exactly 3 fruits, one per line, numbered 1 to 3. Nothing else.", "expect_regex": r"1[.)]\s+\w+\s*2[.)]\s+\w+\s*3[.)]\s+\w+", "max_tokens": 1500},
    {"id": "I3", "category": "instruction", "prompt": "Output a JSON object with exactly two keys: 'name' (string 'GLM') and 'version' (number 5.2). Nothing else.", "expect_regex": r'"name"\s*:\s*"GLM"\s*,\s*"version"\s*:\s*5\.2', "max_tokens": 1500},
    {"id": "K1", "category": "knowledge", "prompt": "What is the capital of Japan? One word.", "expect_regex": r"Tokyo", "max_tokens": 1500},
    {"id": "K2", "category": "knowledge", "prompt": "Who wrote the play 'Romeo and Juliet'? One name.", "expect_regex": r"Shakespeare", "max_tokens": 1500},
]

# ============================================================
# Suite 3: tool_choice tests (18 cases)
# ============================================================
WEATHER_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

CALC_TOOL = [{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform a math calculation",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}]

TOOL_CHOICE_CASES = [
    # --- Chat Completions API: tool_choice variants ---
    {"id": "TC01", "api": "chat", "desc": "chat auto (tool present)", "prompt": "What's the weather in Paris?",
     "tools": WEATHER_TOOL, "tool_choice": "auto",
     "expect": "tool_call", "expect_field": "tool_calls"},
    {"id": "TC02", "api": "chat", "desc": "chat none (tool present)", "prompt": "What's the weather in Paris?",
     "tools": WEATHER_TOOL, "tool_choice": "none",
     "expect": "no_tool_call", "expect_field": "tool_calls"},
    {"id": "TC03", "api": "chat", "desc": "chat required (tool present)", "prompt": "What's the weather in Paris?",
     "tools": WEATHER_TOOL, "tool_choice": "required",
     "expect": "tool_call", "expect_field": "tool_calls"},
    {"id": "TC04", "api": "chat", "desc": "chat required (no tools — should NOT 400)", "prompt": "Hello, how are you?",
     "tools": None, "tool_choice": "required",
     "expect": "no_400", "expect_field": "error"},
    {"id": "TC05", "api": "chat", "desc": "chat function-specific", "prompt": "What's the weather in Paris?",
     "tools": WEATHER_TOOL, "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
     "expect": "tool_call", "expect_field": "tool_calls"},
    {"id": "TC06", "api": "chat", "desc": "chat auto (no tools, no tool_choice)", "prompt": "Say hello in 3 words.",
     "tools": None, "tool_choice": None,
     "expect": "no_400", "expect_field": "error"},
    {"id": "TC07", "api": "chat", "desc": "chat none (no tools)", "prompt": "Say hello in 3 words.",
     "tools": None, "tool_choice": "none",
     "expect": "no_400", "expect_field": "error"},
    # --- Responses API: tool_choice variants ---
    {"id": "TC08", "api": "responses", "desc": "responses auto (tool present)", "prompt": "What's the weather in Paris?",
     "tools": WEATHER_TOOL, "tool_choice": "auto",
     "expect": "tool_call", "expect_field": "output"},
    {"id": "TC09", "api": "responses", "desc": "responses none (tool present)", "prompt": "What's the weather in Paris?",
     "tools": WEATHER_TOOL, "tool_choice": "none",
     "expect": "no_tool_call", "expect_field": "output"},
    {"id": "TC10", "api": "responses", "desc": "responses required (tool present)", "prompt": "What's the weather in Paris?",
     "tools": WEATHER_TOOL, "tool_choice": "required",
     "expect": "tool_call", "expect_field": "output"},
    {"id": "TC11", "api": "responses", "desc": "responses required (no tools — should NOT 400)", "prompt": "Hello, how are you?",
     "tools": None, "tool_choice": "required",
     "expect": "no_400", "expect_field": "error"},
    {"id": "TC12", "api": "responses", "desc": "responses auto (no tools)", "prompt": "Say hello in 3 words.",
     "tools": None, "tool_choice": "auto",
     "expect": "no_400", "expect_field": "error"},
    # --- ThinkingConfig adaptive ---
    {"id": "TC13", "api": "chat", "desc": "chat thinking enabled", "prompt": "What is 2+2? Answer briefly.",
     "tools": None, "tool_choice": None, "thinking": {"type": "enabled", "budget_tokens": 1024},
     "expect": "no_400", "expect_field": "error"},
    {"id": "TC14", "api": "chat", "desc": "chat thinking disabled", "prompt": "What is 2+2? Answer briefly.",
     "tools": None, "tool_choice": None, "thinking": {"type": "disabled"},
     "expect": "no_400", "expect_field": "error"},
    {"id": "TC15", "api": "chat", "desc": "chat thinking adaptive (Claude 4.7+)", "prompt": "What is 2+2? Answer briefly.",
     "tools": None, "tool_choice": None, "thinking": {"type": "adaptive"},
     "expect": "no_400", "expect_field": "error"},
    {"id": "TC16", "api": "responses", "desc": "responses thinking enabled", "prompt": "What is 2+2? Answer briefly.",
     "tools": None, "tool_choice": None, "thinking": {"type": "enabled", "budget_tokens": 1024},
     "expect": "no_400", "expect_field": "error"},
    {"id": "TC17", "api": "responses", "desc": "responses thinking adaptive", "prompt": "What is 2+2? Answer briefly.",
     "tools": None, "tool_choice": None, "thinking": {"type": "adaptive"},
     "expect": "no_400", "expect_field": "error"},
    {"id": "TC18", "api": "responses", "desc": "responses tool_choice=any (legacy)", "prompt": "What's the weather in Paris?",
     "tools": WEATHER_TOOL, "tool_choice": "any",
     "expect": "tool_call_or_no_400", "expect_field": "output"},
]

# ============================================================
# Run suites
# ============================================================
def run_suite1_eval():
    """Suite 1: Effectiveness eval."""
    print("\n" + "=" * 80)
    print("Suite 1: Effectiveness Eval (13 key cases)")
    print("=" * 80)
    passed = 0
    failed = 0
    results = []
    for case in EVAL_CASES:
        cid = case["id"]
        prompt = case["prompt"]
        expect = case["expect_regex"]
        max_tok = case["max_tokens"]
        t0 = time.time()
        resp = call_responses(prompt, max_tokens=max_tok)
        latency = round(time.time() - t0, 2)
        # Extract output text
        output_text = ""
        if "output" in resp and isinstance(resp["output"], list):
            for item in resp["output"]:
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            output_text += c.get("text", "")
        elif "error" in resp:
            output_text = ""
        match = bool(re.search(expect, output_text, re.IGNORECASE))
        status = "PASS" if match else "FAIL"
        if match:
            passed += 1
        else:
            failed += 1
        snippet = output_text[:80].replace("\n", " ") if output_text else resp.get("error", "(empty)")[:80]
        print(f"  [{status}] {cid:<5} {case['category']:<14} {latency:>5}s  expect={expect[:25]:<25}  got={snippet}")
        results.append({"id": cid, "status": status, "latency_s": latency})
    print(f"\n  Suite 1 summary: {passed}/{len(EVAL_CASES)} PASS ({passed*100//len(EVAL_CASES)}%)")
    return {"passed": passed, "failed": failed, "total": len(EVAL_CASES), "results": results}

def run_suite2_perf():
    """Suite 2: Performance latency benchmark."""
    print("\n" + "=" * 80)
    print("Suite 2: Performance Latency Benchmark (7 scenarios)")
    print("=" * 80)
    print("\nWarmup...", end=" ", flush=True)
    r = stream_request("hi", max_tokens=30)
    print(f"{r['total_s']}s")
    scenarios = [
        ("short",            "Reply with: hello world",                                           100),
        ("medium",           "List 10 fruits, one per line.",                                     200),
        ("long",             "Write a short paragraph (about 80 words) about the ocean.",         400),
        ("reasoning",        "If a train leaves Beijing at 3pm at 60km/h and another leaves Shanghai at 4pm at 80km/h toward Beijing (1318km apart), when do they meet?", 800),
        ("code",             "Write a Python function that checks if a string is a palindrome. Just the code, no explanation.", 500),
        ("tool_use",         "What's the weather in Paris? Use the get_weather tool.",            400),
        ("multi_turn_short", "User: my name is bob\nAssistant: hi bob\nUser: what is my name?",   200),
    ]
    print(f"\n{'scenario':<20} {'total(s)':<9} {'ttft_r':<9} {'ttft_o':<9} {'itl_mean':<9} {'itl_p95':<9} {'reason':<7} {'out':<6} {'tps':<6}")
    print("-" * 100)
    results = []
    for name, prompt, max_tok in scenarios:
        r = stream_request(prompt, max_tokens=max_tok)
        results.append((name, r))
        print(f"{name:<20} {r['total_s']:<9} {str(r['ttft_reason_ms']):<9} {str(r['ttft_output_ms']):<9} {str(r['itl_mean_ms']):<9} {str(r['itl_p95_ms']):<9} {r['reason_tokens']:<7} {r['output_tokens']:<6} {str(r['tps']):<6}")
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    totals = [r["total_s"] for _, r in results]
    ttfts_r = [r["ttft_reason_ms"] for _, r in results if r["ttft_reason_ms"]]
    ttfts_o = [r["ttft_output_ms"] for _, r in results if r["ttft_output_ms"]]
    itls = [r["itl_mean_ms"] for _, r in results if r["itl_mean_ms"]]
    tps = [r["tps"] for _, r in results if r["tps"]]
    print(f"  total:       min={min(totals):.2f}s  max={max(totals):.2f}s  mean={statistics.mean(totals):.2f}s")
    print(f"  TTFT-reason: min={min(ttfts_r):.0f}ms  max={max(ttfts_r):.0f}ms  mean={statistics.mean(ttfts_r):.0f}ms")
    print(f"  TTFT-output: min={min(ttfts_o):.0f}ms  max={max(ttfts_o):.0f}ms  mean={statistics.mean(ttfts_o):.0f}ms")
    print(f"  ITL mean:    min={min(itls):.0f}ms  max={max(itls):.0f}ms  mean={statistics.mean(itls):.0f}ms")
    print(f"  TPS:         min={min(tps):.0f}  max={max(tps):.0f}  mean={statistics.mean(tps):.0f} tok/s")
    return {"results": [{"scenario": n, **r} for n, r in results]}

def run_suite3_tool_choice():
    """Suite 3: tool_choice + ThinkingConfig tests."""
    print("\n" + "=" * 80)
    print("Suite 3: tool_choice + ThinkingConfig (18 cases)")
    print("=" * 80)
    passed = 0
    failed = 0
    results = []
    for case in TOOL_CHOICE_CASES:
        cid = case["id"]
        api = case["api"]
        desc = case["desc"]
        prompt = case["prompt"]
        tools = case["tools"]
        tc = case["tool_choice"]
        thinking = case.get("thinking")
        expect = case["expect"]
        t0 = time.time()
        if api == "chat":
            resp = call_chat(prompt, max_tokens=800, tools=tools, tool_choice=tc, thinking=thinking)
        else:
            resp = call_responses(prompt, max_tokens=800, tools=tools, tool_choice=tc, thinking=thinking)
        latency = round(time.time() - t0, 2)
        # Evaluate
        has_error = "error" in resp or (isinstance(resp.get("error"), str))
        has_tool_call = False
        if api == "chat":
            tc_list = resp.get("choices", [{}])[0].get("message", {}).get("tool_calls")
            has_tool_call = bool(tc_list)
        elif api == "responses":
            out = resp.get("output", [])
            for item in out:
                if item.get("type") == "function_call":
                    has_tool_call = True
                    break
        # Determine pass/fail
        ok = False
        if expect == "tool_call":
            ok = has_tool_call and not has_error
        elif expect == "no_tool_call":
            ok = not has_tool_call and not has_error
        elif expect == "no_400":
            # Ensure not a 400 error
            err = resp.get("error", "")
            if isinstance(err, dict):
                err = err.get("message", "")
            ok = not has_error or "400" not in str(err)
        elif expect == "tool_call_or_no_400":
            err = resp.get("error", "")
            if isinstance(err, dict):
                err = err.get("message", "")
            ok = has_tool_call or "400" not in str(err)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        err_snippet = ""
        if has_error:
            err_val = resp.get("error", "")
            if isinstance(err_val, dict):
                err_snippet = err_val.get("message", str(err_val))[:60]
            else:
                err_snippet = str(err_val)[:60]
        detail = f"tool_call={has_tool_call} error={err_snippet}" if has_error else f"tool_call={has_tool_call}"
        print(f"  [{status}] {cid:<5} {api:<10} {desc:<42} {latency:>5}s  {detail}")
        results.append({"id": cid, "status": status, "latency_s": latency, "detail": detail})
    print(f"\n  Suite 3 summary: {passed}/{len(TOOL_CHOICE_CASES)} PASS ({passed*100//len(TOOL_CHOICE_CASES)}%)")
    return {"passed": passed, "failed": failed, "total": len(TOOL_CHOICE_CASES), "results": results}

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("GLM-5.2 2tp8 Merged Chart — Comprehensive Benchmark")
    print(f"Gateway: {GATEWAY}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    # Health check first
    print("\nHealth check...", end=" ", flush=True)
    health = subprocess.run(["/usr/bin/curl", "-sS", "--max-time", "10", f"{GATEWAY}/health"],
                            capture_output=True, text=True)
    if health.returncode == 0:
        print(f"OK ({health.stdout.strip()[:50]})")
    else:
        print(f"FAILED: {health.stderr[:100]}")
        sys.exit(1)
    s1 = run_suite1_eval()
    s2 = run_suite2_perf()
    s3 = run_suite3_tool_choice()
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"  Suite 1 (Effectiveness): {s1['passed']}/{s1['total']} PASS")
    print(f"  Suite 2 (Performance):   {len(s2['results'])} scenarios completed")
    print(f"  Suite 3 (tool_choice):   {s3['passed']}/{s3['total']} PASS")
    total_pass = s1["passed"] + s3["passed"]
    total_cases = s1["total"] + s3["total"]
    print(f"  Total: {total_pass}/{total_cases} PASS ({total_pass*100//total_cases}%)")
    print("=" * 80)
    # Save full results
    out_file = f"/tmp/benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump({"suite1_eval": s1, "suite2_perf": s2, "suite3_tool_choice": s3}, f, indent=2)
    print(f"\nFull results saved to: {out_file}")
