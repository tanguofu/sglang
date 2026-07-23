#!/usr/bin/env python3
"""Multi-dimension eval suite for current GLM-5.2 deployment.

Covers:
 1. Reasoning (logic, multi-step)
 2. Math (arithmetic, word problems, algebra)
 3. Code generation (Python, JS)
 4. Code debugging (find bugs)
 5. Instruction following (format constraints)
 6. Knowledge (factual Q&A)
 7. Chinese language
 8. Tool use (function calling)
 9. Long context (within context, not 512K)
10. Creative writing
11. Summarization

Each case has:
- prompt
- expected (substring or regex or callable check)
- category
- max_tokens

Run via Messages API streaming through the gateway.
"""
import json, subprocess, time, re, sys

GATEWAY = "https://glm52-2tp8.jmpti.woa.com"
TOKEN = "${ANTHROPIC_AUTH_TOKEN}"
MODEL = "glm-5.2"

EVAL_CASES = [
    # ========== 1. Reasoning ==========
    {
        "id": "R1",
        "category": "reasoning",
        "prompt": "All cats are mammals. Tom is a cat. What is Tom? Answer in one word.",
        "expect_regex": r"mammal",
        "max_tokens": 100,
    },
    {
        "id": "R2",
        "category": "reasoning",
        "prompt": "If A > B and B > C, is A > C? Answer yes or no with one sentence explaining why.",
        "expect_regex": r"yes|true",
        "max_tokens": 100,
    },
    {
        "id": "R3",
        "category": "reasoning",
        "prompt": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Just the number.",
        "expect_regex": r"\b9\b",
        "max_tokens": 100,
    },
    {
        "id": "R4",
        "category": "reasoning",
        "prompt": "I have a 5-liter jug and a 3-liter jug. I need exactly 4 liters of water. How do I do it? List the steps.",
        "expect_regex": r"(5|five).*(3|three)|(3|three).*(5|five)",
        "max_tokens": 500,
    },
    # ========== 2. Math ==========
    {
        "id": "M1",
        "category": "math",
        "prompt": "What is 123 * 456? Just the number.",
        "expect_regex": r"56088",
        "max_tokens": 100,
    },
    {
        "id": "M2",
        "category": "math",
        "prompt": "If a train travels at 60 km/h for 2.5 hours, how far does it go? Just the number with units.",
        "expect_regex": r"150",
        "max_tokens": 100,
    },
    {
        "id": "M3",
        "category": "math",
        "prompt": "Solve for x: 2x + 5 = 17. Just the value of x.",
        "expect_regex": r"\b6\b",
        "max_tokens": 100,
    },
    {
        "id": "M4",
        "category": "math",
        "prompt": "What is 15% of 200? Just the number.",
        "expect_regex": r"\b30\b",
        "max_tokens": 100,
    },
    # ========== 3. Code generation ==========
    {
        "id": "C1",
        "category": "code",
        "prompt": "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome (case-insensitive, ignoring non-alphanumeric). Just the code.",
        "expect_regex": r"def\s+is_palindrome|def is_palindrome",
        "max_tokens": 400,
    },
    {
        "id": "C2",
        "category": "code",
        "prompt": "Write a Python function `fib(n)` that returns the n-th Fibonacci number iteratively. Just the code.",
        "expect_regex": r"def\s+fib",
        "max_tokens": 400,
    },
    {
        "id": "C3",
        "category": "code",
        "prompt": "Write a Python function `binary_search(arr, target)` that returns the index of target in sorted arr, or -1. Just the code.",
        "expect_regex": r"def\s+binary_search",
        "max_tokens": 500,
    },
    # ========== 4. Code debugging ==========
    {
        "id": "D1",
        "category": "debug",
        "prompt": "Find the bug in this Python code:\n\n```python\ndef count_words(text):\n    words = text.split()\n    count = 0\n    for word in words:\n        count = count + 1\n        return count\n```\n\nWhat's the bug? Give the fix in one sentence.",
        "expect_regex": r"return|indent",
        "max_tokens": 200,
    },
    {
        "id": "D2",
        "category": "debug",
        "prompt": "What's wrong with this Python code?\n\n```python\ndef get_average(numbers):\n    return sum(numbers) / len(numbers)\n\nprint(get_average([]))\n```\n\nOne sentence on the bug.",
        "expect_regex": r"empty|len|zero|division",
        "max_tokens": 200,
    },
    # ========== 5. Instruction following ==========
    {
        "id": "I1",
        "category": "instruction",
        "prompt": "List exactly 3 fruits, one per line, numbered 1 to 3. Nothing else.",
        "expect_regex": r"1\.\s+\w+\n2\.\s+\w+\n3\.\s+\w+",
        "max_tokens": 100,
    },
    {
        "id": "I2",
        "category": "instruction",
        "prompt": "Reply with exactly 5 words. No more, no less.",
        "expect_regex": r"^\W*(\w+\W+){4}\w+\W*$",
        "max_tokens": 50,
    },
    {
        "id": "I3",
        "category": "instruction",
        "prompt": "Output a JSON object with exactly two keys: 'name' (string 'GLM') and 'version' (number 5.2). Nothing else.",
        "expect_regex": r'\{\s*"name"\s*:\s*"GLM"\s*,\s*"version"\s*:\s*5\.2\s*\}',
        "max_tokens": 100,
    },
    # ========== 6. Knowledge ==========
    {
        "id": "K1",
        "category": "knowledge",
        "prompt": "What is the capital of Japan? One word.",
        "expect_regex": r"Tokyo",
        "max_tokens": 50,
    },
    {
        "id": "K2",
        "category": "knowledge",
        "prompt": "Who wrote the play 'Romeo and Juliet'? One name.",
        "expect_regex": r"Shakespeare",
        "max_tokens": 50,
    },
    {
        "id": "K3",
        "category": "knowledge",
        "prompt": "What is the chemical symbol for gold? Just the symbol.",
        "expect_regex": r"\bAu\b",
        "max_tokens": 50,
    },
    # ========== 7. Chinese ==========
    {
        "id": "ZH1",
        "category": "chinese",
        "prompt": "中国的首都是哪里?只回答城市名。",
        "expect_regex": r"北京",
        "max_tokens": 50,
    },
    {
        "id": "ZH2",
        "category": "chinese",
        "prompt": "用一句话解释什么是人工智能。",
        "expect_regex": r"人工智能|AI|模拟.*智能|机器.*学习",
        "max_tokens": 200,
    },
    {
        "id": "ZH3",
        "category": "chinese",
        "prompt": "写一个 Python 函数,计算两个数的最大公约数。只给代码。",
        "expect_regex": r"def|gcd",
        "max_tokens": 300,
    },
    # ========== 8. Tool use ==========
    {
        "id": "T1",
        "category": "tool_use",
        "prompt": "What's the weather in Tokyo? Use the get_weather tool.",
        "expect_regex": r"tool_use|get_weather",
        "max_tokens": 200,
        "tools": [{
            "name": "get_weather",
            "description": "Get current weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }],
        "tool_choice": {"type": "any"},
    },
    # ========== 9. Long context (moderate) ==========
    {
        "id": "L1",
        "category": "long_context",
        "prompt": ("Here is a list of items:\n"
                   "Item 1: Apple\nItem 2: Banana\nItem 3: Cherry\nItem 4: Date\n"
                   "Item 5: Elderberry\nItem 6: Fig\nItem 7: Grape\nItem 8: Honeydew\n"
                   "Item 9: Kiwi\nItem 10: Lemon\nItem 11: Mango\nItem 12: Nectarine\n"
                   "Item 13: Orange\nItem 14: Papaya\nItem 15: Quince\nItem 16: Raspberry\n"
                   "Item 17: Strawberry\nItem 18: Tangerine\nItem 19: Ugli fruit\nItem 20: Vanilla\n\n"
                   "Question: What was Item 14? Just the answer."),
        "expect_regex": r"Papaya",
        "max_tokens": 50,
    },
    # ========== 10. Creative ==========
    {
        "id": "CR1",
        "category": "creative",
        "prompt": "Write a 4-line haiku about the ocean. Just the haiku.",
        "expect_regex": r"\n.*\n.*\n",
        "max_tokens": 100,
    },
    # ========== 11. Summarization ==========
    {
        "id": "S1",
        "category": "summarization",
        "prompt": ("Summarize this in one sentence:\n\n"
                   "The Apollo 11 mission was the first manned mission to land on the Moon. "
                   "It was launched on July 16, 1969, and carried astronauts Neil Armstrong, "
                   "Buzz Aldrin, and Michael Collins. Armstrong and Aldrin walked on the lunar "
                   "surface while Collins orbited above. The mission returned safely to Earth on July 24, 1969."),
        "expect_regex": r"Apollo\s*11|moo[nnd]|Armstrong",
        "max_tokens": 150,
    },
]

def call_messages(prompt, max_tokens=300, tools=None, tool_choice=None):
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    if tools:
        body["tools"] = tools
    if tool_choice:
        body["tool_choice"] = tool_choice

    proc = subprocess.Popen([
        "/usr/bin/curl", "-sS", "-N", "--max-time", "120",
        f"{GATEWAY}/v1/messages",
        "-H", f"Authorization: Bearer {TOKEN}",
        "-H", "anthropic-version: 2023-06-01",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(body),
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    text_chunks = []
    tool_calls = []
    stop_reason = None
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
        t = evt.get("type", "")
        if t == "content_block_delta":
            delta = evt.get("delta", {})
            if delta.get("type") == "text_delta":
                text_chunks.append(delta.get("text", ""))
            elif delta.get("type") == "input_json_delta":
                tool_calls.append(delta.get("partial_json", ""))
        elif t == "message_delta":
            stop_reason = evt.get("delta", {}).get("stop_reason")

    proc.wait()
    return "".join(text_chunks), "".join(tool_calls), stop_reason

def check_output(case, text, tool_text):
    expect = case.get("expect_regex")
    if not expect:
        return True, "no expect"
    pattern = re.compile(expect, re.IGNORECASE | re.DOTALL | re.MULTILINE)
    full = text + "\n" + tool_text
    if pattern.search(full):
        return True, "matched"
    return False, f"no match for /{expect}/"

# Run eval
print("=" * 80)
print(f"GLM-5.2 Deployment Eval — {len(EVAL_CASES)} cases")
print(f"Gateway: {GATEWAY}")
print("=" * 80)

results = []
by_category = {}
for case in EVAL_CASES:
    cid = case["id"]
    cat = case["category"]
    print(f"\n[{cid}] {cat}: ", end="", flush=True)
    start = time.perf_counter()
    try:
        text, tool_text, stop = call_messages(
            case["prompt"],
            max_tokens=case.get("max_tokens", 300),
            tools=case.get("tools"),
            tool_choice=case.get("tool_choice"),
        )
        elapsed = time.perf_counter() - start
        ok, note = check_output(case, text, tool_text)
        results.append({"id": cid, "category": cat, "ok": ok, "note": note,
                        "elapsed": elapsed, "stop_reason": stop,
                        "text": text, "tool_text": tool_text})
        by_category.setdefault(cat, []).append(ok)
        status = "PASS" if ok else "FAIL"
        print(f"{status} ({elapsed:.1f}s, stop={stop}) — {note}")
        if not ok:
            print(f"  Output: {text[:300]!r}")
            if tool_text:
                print(f"  Tool:   {tool_text[:200]!r}")
    except Exception as e:
        elapsed = time.perf_counter() - start
        results.append({"id": cid, "category": cat, "ok": False, "note": f"error: {e}",
                        "elapsed": elapsed, "text": "", "tool_text": ""})
        by_category.setdefault(cat, []).append(False)
        print(f"ERROR ({elapsed:.1f}s) — {e}")

# Summary
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
for cat in sorted(by_category.keys()):
    oks = by_category[cat]
    p = sum(oks)
    t = len(oks)
    print(f"{cat:<18} {p:<6} {t:<6} {p/t*100:.0f}%")

print(f"\nFailed cases:")
for r in results:
    if not r["ok"]:
        print(f"  [{r['id']}] {r['category']}: {r['note']}")

# Save full results
with open("/tmp/eval_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nFull results saved to /tmp/eval_results.json")
