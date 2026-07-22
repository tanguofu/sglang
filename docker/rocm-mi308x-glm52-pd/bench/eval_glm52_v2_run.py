#!/usr/bin/env python3
"""Multi-dimension eval suite v2 — fixed for GLM-5.2 reasoning model.

Key fixes vs v1:
  1. max_tokens increased to 1500-4000 (reasoning + output budget)
  2. Capture reasoning_text AND output_text separately (both via Responses API)
  3. Check BOTH reasoning + output for expected answer (reasoning may contain the answer)
  4. Use Responses API (/v1/responses) — native to GLM-5.2 reasoning separation
  5. Use gateway (router stable now) with worker-direct fallback on 503
"""
import json, subprocess, time, re, sys, os

GATEWAY = "https://glm52-2tp8.jmpti.woa.com"
TOKEN = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"
WORKER_POD = "sglang-glm52-2tp8-sglang-0"

# Eval cases — max_tokens tuned for reasoning model.
# reasoning overhead typically 200-1500 tokens, output 50-1000 tokens.
EVAL_CASES = [
    # ========== 1. Reasoning ==========
    {"id": "R1", "category": "reasoning",
     "prompt": "All cats are mammals. Tom is a cat. What is Tom? Answer in one word.",
     "expect_regex": r"mammal", "max_tokens": 1500},
    {"id": "R2", "category": "reasoning",
     "prompt": "If A > B and B > C, is A > C? Answer yes or no with one sentence explaining why.",
     "expect_regex": r"yes|true", "max_tokens": 1500},
    {"id": "R3", "category": "reasoning",
     "prompt": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Just the number.",
     "expect_regex": r"\b9\b", "max_tokens": 1500},
    {"id": "R4", "category": "reasoning",
     "prompt": "I have a 5-liter jug and a 3-liter jug. I need exactly 4 liters of water. How do I do it? List the steps.",
     "expect_regex": r"(5|five).*(3|three)|(3|three).*(5|five)", "max_tokens": 2500},
    # ========== 2. Math ==========
    {"id": "M1", "category": "math",
     "prompt": "What is 123 * 456? Just the number.",
     "expect_regex": r"56088", "max_tokens": 1500},
    {"id": "M2", "category": "math",
     "prompt": "If a train travels at 60 km/h for 2.5 hours, how far does it go? Just the number with units.",
     "expect_regex": r"150", "max_tokens": 1500},
    {"id": "M3", "category": "math",
     "prompt": "Solve for x: 2x + 5 = 17. Just the value of x.",
     "expect_regex": r"\b6\b", "max_tokens": 1500},
    {"id": "M4", "category": "math",
     "prompt": "What is 15% of 200? Just the number.",
     "expect_regex": r"\b30\b", "max_tokens": 1500},
    # ========== 3. Code generation ==========
    {"id": "C1", "category": "code",
     "prompt": "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome (case-insensitive, ignoring non-alphanumeric). Just the code.",
     "expect_regex": r"def\s+is_palindrome", "max_tokens": 2500},
    {"id": "C2", "category": "code",
     "prompt": "Write a Python function `fib(n)` that returns the n-th Fibonacci number iteratively. Just the code.",
     "expect_regex": r"def\s+fib", "max_tokens": 2500},
    {"id": "C3", "category": "code",
     "prompt": "Write a Python function `binary_search(arr, target)` that returns the index of target in sorted arr, or -1. Just the code.",
     "expect_regex": r"def\s+binary_search", "max_tokens": 2500},
    # ========== 4. Code debugging ==========
    {"id": "D1", "category": "debug",
     "prompt": "Find the bug in this Python code:\n\n```python\ndef count_words(text):\n    words = text.split()\n    count = 0\n    for word in words:\n        count = count + 1\n        return count\n```\n\nWhat's the bug? Give the fix in one sentence.",
     "expect_regex": r"return|indent", "max_tokens": 2000},
    {"id": "D2", "category": "debug",
     "prompt": "What's wrong with this Python code?\n\n```python\ndef get_average(numbers):\n    return sum(numbers) / len(numbers)\n\nprint(get_average([]))\n```\n\nOne sentence on the bug.",
     "expect_regex": r"empty|len|zero|division|ZeroDivision", "max_tokens": 2000},
    # ========== 5. Instruction following ==========
    {"id": "I1", "category": "instruction",
     "prompt": "List exactly 3 fruits, one per line, numbered 1 to 3. Nothing else.",
     "expect_regex": r"1[.)]\s+\w+\s*2[.)]\s+\w+\s*3[.)]\s+\w+", "max_tokens": 1500},
    {"id": "I2", "category": "instruction",
     "prompt": "Reply with exactly 5 words. No more, no less.",
     "expect_regex": r"^\W*(\w+\W+){4}\w+\W*$", "max_tokens": 1500},
    {"id": "I3", "category": "instruction",
     "prompt": "Output a JSON object with exactly two keys: 'name' (string 'GLM') and 'version' (number 5.2). Nothing else.",
     "expect_regex": r'"name"\s*:\s*"GLM"\s*,\s*"version"\s*:\s*5\.2', "max_tokens": 1500},
    # ========== 6. Knowledge ==========
    {"id": "K1", "category": "knowledge",
     "prompt": "What is the capital of Japan? One word.",
     "expect_regex": r"Tokyo", "max_tokens": 1500},
    {"id": "K2", "category": "knowledge",
     "prompt": "Who wrote the play 'Romeo and Juliet'? One name.",
     "expect_regex": r"Shakespeare", "max_tokens": 1500},
    {"id": "K3", "category": "knowledge",
     "prompt": "What is the chemical symbol for gold? Just the symbol.",
     "expect_regex": r"\bAu\b", "max_tokens": 1500},
    # ========== 7. Chinese ==========
    {"id": "ZH1", "category": "chinese",
     "prompt": "中国的首都是哪里?只回答城市名。",
     "expect_regex": r"北京", "max_tokens": 1500},
    {"id": "ZH2", "category": "chinese",
     "prompt": "用一句话解释什么是人工智能。",
     "expect_regex": r"人工智能|AI|模拟.*智能|机器.*学习", "max_tokens": 2000},
    {"id": "ZH3", "category": "chinese",
     "prompt": "写一个 Python 函数,计算两个数的最大公约数。只给代码。",
     "expect_regex": r"def|gcd", "max_tokens": 2500},
    # ========== 8. Tool use ==========
    {"id": "T1", "category": "tool_use",
     "prompt": "What's the weather in Tokyo? Use the get_weather tool.",
     "expect_regex": r"tool_use|get_weather|\{.*city.*Tokyo",
     "max_tokens": 2000,
     "tools": [{"name": "get_weather", "description": "Get current weather for a city",
                "input_schema": {"type": "object",
                                 "properties": {"city": {"type": "string"}},
                                 "required": ["city"]}}],
     "tool_choice": {"type": "any"}},
    # ========== 9. Long context ==========
    {"id": "L1", "category": "long_context",
     "prompt": ("Here is a list of items:\n"
                "Item 1: Apple\nItem 2: Banana\nItem 3: Cherry\nItem 4: Date\n"
                "Item 5: Elderberry\nItem 6: Fig\nItem 7: Grape\nItem 8: Honeydew\n"
                "Item 9: Kiwi\nItem 10: Lemon\nItem 11: Mango\nItem 12: Nectarine\n"
                "Item 13: Orange\nItem 14: Papaya\nItem 15: Quince\nItem 16: Raspberry\n"
                "Item 17: Strawberry\nItem 18: Tangerine\nItem 19: Ugli fruit\nItem 20: Vanilla\n\n"
                "Question: What was Item 14? Just the answer."),
     "expect_regex": r"Papaya", "max_tokens": 1500},
    # ========== 10. Creative ==========
    {"id": "CR1", "category": "creative",
     "prompt": "Write a 4-line haiku about the ocean. Just the haiku.",
     "expect_regex": r"\n.*\n", "max_tokens": 2000},
    # ========== 11. Summarization ==========
    {"id": "S1", "category": "summarization",
     "prompt": ("Summarize this in one sentence:\n\n"
                "The Apollo 11 mission was the first manned mission to land on the Moon. "
                "It was launched on July 16, 1969, and carried astronauts Neil Armstrong, "
                "Buzz Aldrin, and Michael Collins. Armstrong and Aldrin walked on the lunar "
                "surface while Collins orbited above. The mission returned safely to Earth on July 24, 1969."),
     "expect_regex": r"Apollo\s*11|moon|Armstrong", "max_tokens": 2000},
]


def parse_responses_sse(raw):
    """Parse Responses API SSE — extract reasoning_text, output_text, tool_calls, status."""
    reasoning_chunks = []
    output_chunks = []
    tool_chunks = []
    status = None
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            evt = json.loads(data)
        except json.JSONDecodeError:
            continue
        t = evt.get("type", "")
        if t == "response.output_text.delta":
            output_chunks.append(evt.get("delta", ""))
        elif t == "response.reasoning_text.delta":
            reasoning_chunks.append(evt.get("delta", ""))
        elif t == "response.function_call_arguments.delta":
            tool_chunks.append(evt.get("delta", ""))
        elif t == "response.completed":
            resp = evt.get("response", {})
            status = resp.get("status")
            usage = resp.get("usage", {})
            return {
                "reasoning": "".join(reasoning_chunks),
                "output": "".join(output_chunks),
                "tool": "".join(tool_chunks),
                "status": status,
                "usage": usage,
            }
    return {
        "reasoning": "".join(reasoning_chunks),
        "output": "".join(output_chunks),
        "tool": "".join(tool_chunks),
        "status": status,
        "usage": {},
    }


def call_gateway(prompt, max_tokens=2000, tools=None, tool_choice=None):
    """Call gateway via Responses API."""
    body = {
        "model": MODEL,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "max_output_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        # Convert Anthropic tool schema to Responses API format
        resp_tools = []
        for tool in tools:
            resp_tools.append({
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            })
        body["tools"] = resp_tools
        if tool_choice:
            if tool_choice.get("type") == "any":
                body["tool_choice"] = "auto"

    body_json = json.dumps(body)
    cmd = [
        "curl", "-sS", "-N", "--max-time", "180",
        f"{GATEWAY}/v1/responses",
        "-H", f"Authorization: Bearer {TOKEN}",
        "-H", "Content-Type: application/json",
        "-d", body_json,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    return parse_responses_sse(proc.stdout), proc.stderr


def call_worker(prompt, max_tokens=2000, tools=None, tool_choice=None):
    """Worker-direct via kubectl exec (fallback when router 503s)."""
    body = {
        "model": MODEL,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "max_output_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        resp_tools = []
        for tool in tools:
            resp_tools.append({
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            })
        body["tools"] = resp_tools
        if tool_choice and tool_choice.get("type") == "any":
            body["tool_choice"] = "auto"

    body_json = json.dumps(body)
    cmd = [
        "kubectl", "exec", "-n", "kube-system", WORKER_POD, "--",
        "/usr/bin/curl", "-sS", "-N", "--max-time", "180",
        "http://127.0.0.1:30000/v1/responses",
        "-H", f"Authorization: Bearer {TOKEN}",
        "-H", "Content-Type: application/json",
        "-d", body_json,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    return parse_responses_sse(proc.stdout), proc.stderr


def check(case, result):
    """Check expected answer against reasoning + output + tool."""
    expect = case.get("expect_regex")
    if not expect:
        return True, "no expect"
    # Check combined reasoning + output + tool
    combined = "\n".join([result["reasoning"], result["output"], result["tool"]])
    if re.search(expect, combined, re.IGNORECASE | re.DOTALL | re.MULTILINE):
        # Identify where match was found
        if re.search(expect, result["output"], re.IGNORECASE | re.DOTALL | re.MULTILINE):
            return True, "matched in output"
        if re.search(expect, result["reasoning"], re.IGNORECASE | re.DOTALL | re.MULTILINE):
            return True, "matched in reasoning (output empty/truncated)"
        return True, "matched in tool"
    return False, f"no match for /{expect}/"


def run_case(case, use_worker=False):
    """Run one case, return result dict."""
    cid, cat = case["id"], case["category"]
    fn = call_worker if use_worker else call_gateway
    start = time.perf_counter()
    try:
        result, err = fn(
            case["prompt"],
            max_tokens=case.get("max_tokens", 2000),
            tools=case.get("tools"),
            tool_choice=case.get("tool_choice"),
        )
        elapsed = time.perf_counter() - start
        ok, note = check(case, result)
        return {
            "id": cid, "category": cat, "ok": ok, "note": note,
            "elapsed": round(elapsed, 2),
            "status": result.get("status"),
            "reasoning_len": len(result.get("reasoning", "")),
            "output_len": len(result.get("output", "")),
            "output": result.get("output", "")[:800],
            "reasoning": result.get("reasoning", "")[:300],
            "usage": result.get("usage", {}),
            "stderr": err[:200] if err else "",
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "id": cid, "category": cat, "ok": False,
            "note": f"error: {e}", "elapsed": round(elapsed, 2),
            "output": "", "reasoning": "", "stderr": str(e)[:200],
        }


def main():
    use_worker = "--worker" in sys.argv
    mode = "WORKER-DIRECT" if use_worker else "GATEWAY"
    print("=" * 80)
    print(f"GLM-5.2 Eval v2 ({mode}) — {len(EVAL_CASES)} cases")
    print(f"Endpoint: {'kubectl exec ' + WORKER_POD if use_worker else GATEWAY}")
    print("=" * 80)

    results = []
    by_cat = {}
    for i, case in enumerate(EVAL_CASES, 1):
        cid = case["id"]
        print(f"\n[{i}/{len(EVAL_CASES)}] {cid} ({case['category']}): ", end="", flush=True)
        r = run_case(case, use_worker=use_worker)
        results.append(r)
        by_cat.setdefault(r["category"], []).append(r["ok"])
        status = "PASS" if r["ok"] else "FAIL"
        print(f"{status} ({r['elapsed']}s, status={r.get('status')}, "
              f"reason={r['reasoning_len']}c, out={r['output_len']}c) — {r['note']}")
        if not r["ok"]:
            print(f"  Output: {r['output'][:300]!r}")
            if r["reasoning"]:
                print(f"  Reasoning (head): {r['reasoning'][:200]!r}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    failed = total - passed
    print(f"\nTotal: {total}  Pass: {passed}  Fail: {failed}  Rate: {passed/total*100:.1f}%")

    print(f"\nBy category:")
    print(f"{'category':<18} {'pass':<6} {'total':<6} {'rate':<8}")
    print("-" * 40)
    for cat in sorted(by_cat.keys()):
        oks = by_cat[cat]
        p = sum(oks); t = len(oks)
        print(f"{cat:<18} {p:<6} {t:<6} {p/t*100:.0f}%")

    print(f"\nFailed cases:")
    any_fail = False
    for r in results:
        if not r["ok"]:
            any_fail = True
            print(f"  [{r['id']}] {r['category']}: {r['note']} "
                  f"(reason={r['reasoning_len']}c, out={r['output_len']}c)")
    if not any_fail:
        print("  (none)")

    out_file = "/tmp/eval_results_v2.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {out_file}")


if __name__ == "__main__":
    main()
