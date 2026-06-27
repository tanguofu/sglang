#!/usr/bin/env python3
"""Accuracy test v3: uses /v1/chat/completions API with proper chat template.
Tests coding, math, and knowledge accuracy."""
import requests, re, json, time

URL = "http://localhost:30000/v1/chat/completions"
MODEL = "/data/models/GLM-5.2-FP8"

def send(user_msg, max_tokens=512, temperature=0.0):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": user_msg}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }
    r = requests.post(URL, json=payload, timeout=120)
    if r.status_code == 200:
        data = r.json()
        return data["choices"][0]["message"]["content"]
    return f"ERROR: {r.status_code} {r.text[:200]}"

def extract_num(text):
    # Try to find "answer is X" or "#### X" or last number
    m = re.search(r"(?:answer is|####|=)\s*\$?\s*([-+]?\d*\.?\d+)", text, re.I)
    if m:
        try: return float(m.group(1))
        except: pass
    numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if numbers:
        try: return float(numbers[-1])
        except: return None
    return None

# ============================================================
# MATH TESTS (GSM8K-style)
# ============================================================
math_tests = [
    ("math_1", "A store sells apples at $2 each. If you buy 5 apples, how much do you pay? Show your work and give the final answer.", 10),
    ("math_2", "Tom has 3 boxes, each containing 4 pencils. How many pencils does Tom have in total? Show your work and give the final answer.", 12),
    ("math_3", "A train travels 60 miles in 1 hour. How far will it travel in 3.5 hours? Show your work and give the final answer.", 210),
    ("math_4", "If 15% of 200 students play basketball, how many students play basketball? Show your work and give the final answer.", 30),
    ("math_5", "A rectangle has length 8 and width 5. What is its area? Show your work and give the final answer.", 40),
    ("math_6", "If x + 5 = 12, what is x? Show your work and give the final answer.", 7),
    ("math_7", "A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices are left? Show your work and give the final answer.", 2),
    ("math_8", "What is 25% of 80? Show your work and give the final answer.", 20),
    ("math_9", "If a shirt costs $25 and is on sale for 20% off, what is the sale price? Show your work and give the final answer.", 20),
    ("math_10", "A car uses 3 gallons of gas per hour. How many gallons does it use in 4.5 hours? Show your work and give the final answer.", 13.5),
    ("math_11", "What is 7 * 8? Give the final answer.", 56),
    ("math_12", "If a triangle has angles of 60 and 80 degrees, what is the third angle? Show your work and give the final answer.", 40),
    ("math_13", "Solve: 3x - 6 = 15. What is x? Show your work and give the final answer.", 7),
    ("math_14", "A book has 240 pages. If you read 30 pages per day, how many days to finish? Show your work and give the final answer.", 8),
    ("math_15", "What is the square root of 144? Give the final answer.", 12),
]

# ============================================================
# CODING TESTS (HumanEval-style)
# ============================================================
coding_tests = [
    ("coding_1", "Write a Python function that returns the sum of two numbers. Only output the function code.\n\ndef add(a, b):", lambda r: "return a + b" in r or "return a+b" in r),
    ("coding_2", "Write a Python function that reverses a string. Only output the function code.\n\ndef reverse_string(s):", lambda r: "s[::-1]" in r),
    ("coding_3", "Write a Python function that checks if a number is even. Only output the function code.\n\ndef is_even(n):", lambda r: "n % 2 == 0" in r or "n%2==0" in r),
    ("coding_4", "Write a Python function that returns the factorial of n using recursion. Only output the function code.\n\ndef factorial(n):", lambda r: "factorial(n - 1)" in r or "factorial(n-1)" in r),
    ("coding_5", "Write a Python function that finds the maximum element in a list. Only output the function code.\n\ndef find_max(lst):", lambda r: "max(lst)" in r or "max(" in r),
    ("coding_6", "Write a Python function that checks if a string is a palindrome. Only output the function code.\n\ndef is_palindrome(s):", lambda r: "s[::-1]" in r or "reversed" in r),
    ("coding_7", "Write a Python function that returns the n-th Fibonacci number using recursion. Only output the function code.\n\ndef fibonacci(n):", lambda r: "fibonacci(n - 1)" in r or "fibonacci(n-1)" in r),
    ("coding_8", "Write a Python function that counts vowels in a string. Only output the function code.\n\ndef count_vowels(s):", lambda r: "aeiou" in r.lower() or "AEIOU" in r),
    ("coding_9", "Write a Python function that removes duplicates from a list. Only output the function code.\n\ndef remove_duplicates(lst):", lambda r: "set(" in r or "seen" in r or "dict" in r),
    ("coding_10", "Write a Python function that sorts a list in ascending order. Only output the function code.\n\ndef sort_list(lst):", lambda r: "sorted(lst)" in r or "sort()" in r or "sorted(" in r),
    ("coding_11", "Write a Python function that returns the length of a string. Only output the function code.\n\ndef string_length(s):", lambda r: "len(s)" in r or "len(" in r),
    ("coding_12", "Write a Python function that converts Celsius to Fahrenheit: F = C * 9/5 + 32. Only output the function code.\n\ndef celsius_to_fahrenheit(c):", lambda r: "9" in r and "5" in r and "32" in r),
    ("coding_13", "Write a Python function that checks if a number is prime. Only output the function code.\n\ndef is_prime(n):", lambda r: "range(2" in r or "range(1" in r or "sqrt" in r),
    ("coding_14", "Write a Python function that returns the sum of a list. Only output the function code.\n\ndef sum_list(lst):", lambda r: "sum(lst)" in r or "sum(" in r),
    ("coding_15", "Write a Python function that converts a string to uppercase. Only output the function code.\n\ndef to_upper(s):", lambda r: "upper()" in r),
]

# ============================================================
# KNOWLEDGE TESTS
# ============================================================
know_tests = [
    ("know_1", "What is the capital of France? Answer with just the city name.", lambda r: "paris" in r.lower()),
    ("know_2", "What is 2 + 2? Answer with just the number.", lambda r: r.strip().startswith("4")),
    ("know_3", "What programming language is Django written in? Answer with just the language name.", lambda r: "python" in r.lower()),
    ("know_4", "What does CPU stand for? Answer briefly.", lambda r: "central processing unit" in r.lower()),
    ("know_5", "What is the chemical symbol for water? Answer with just the symbol.", lambda r: "h2o" in r.lower()),
    ("know_6", "What is the largest planet in our solar system? Answer with just the name.", lambda r: "jupiter" in r.lower()),
    ("know_7", "Who wrote the play Romeo and Juliet? Answer with just the name.", lambda r: "shakespeare" in r.lower()),
    ("know_8", "What is the freezing point of water in Celsius? Answer with just the number.", lambda r: "0" in r.strip().split(".")[0]),
    ("know_9", "How many continents are there on Earth? Answer with just the number.", lambda r: "7" in r.strip().split("\n")[0]),
    ("know_10", "What gas do plants absorb from the atmosphere? Answer with just the gas name.", lambda r: "carbon dioxide" in r.lower() or "co2" in r.lower()),
    ("know_11", "What is the speed of light in vacuum approximately (in km/s)? Answer with just the number.", lambda r: "299792" in r.replace(",","").replace(" ","") or "300000" in r.replace(",","").replace(" ","") or "3e5" in r.lower()),
    ("know_12", "Who painted the Mona Lisa? Answer with just the name.", lambda r: "davinci" in r.lower().replace(" ","") or "da vinci" in r.lower()),
    ("know_13", "What is the largest ocean on Earth? Answer with just the name.", lambda r: "pacific" in r.lower()),
    ("know_14", "In what year did World War II end? Answer with just the year.", lambda r: "1945" in r),
    ("know_15", "What is the powerhouse of the cell? Answer briefly.", lambda r: "mitochondria" in r.lower()),
]

# Run tests
print("=" * 60)
print("MATH TESTS (GSM8K-style, 15 questions)")
print("=" * 60)
math_pass = 0
for tid, prompt, expected in math_tests:
    resp = send(prompt, max_tokens=512)
    got = extract_num(resp)
    passed = got is not None and abs(got - expected) < 0.5
    if passed: math_pass += 1
    status = "PASS" if passed else "FAIL"
    print(f"  {tid}: {status} (expected={expected}, got={got})")
    if not passed:
        print(f"    Response: {repr(resp[:300])}")

print(f"\n  MATH: {math_pass}/{len(math_tests)} ({100*math_pass/len(math_tests):.0f}%)")

print("\n" + "=" * 60)
print("CODING TESTS (HumanEval-style, 15 questions)")
print("=" * 60)
coding_pass = 0
for tid, prompt, check in coding_tests:
    resp = send(prompt, max_tokens=256)
    passed = check(resp)
    if passed: coding_pass += 1
    status = "PASS" if passed else "FAIL"
    print(f"  {tid}: {status}")
    if not passed:
        print(f"    Response: {repr(resp[:300])}")

print(f"\n  CODING: {coding_pass}/{len(coding_tests)} ({100*coding_pass/len(coding_tests):.0f}%)")

print("\n" + "=" * 60)
print("KNOWLEDGE TESTS (15 questions)")
print("=" * 60)
know_pass = 0
for tid, prompt, check in know_tests:
    resp = send(prompt, max_tokens=64)
    passed = check(resp)
    if passed: know_pass += 1
    status = "PASS" if passed else "FAIL"
    print(f"  {tid}: {status}")
    if not passed:
        print(f"    Response: {repr(resp[:200])}")

print(f"\n  KNOWLEDGE: {know_pass}/{len(know_tests)} ({100*know_pass/len(know_tests):.0f}%)")

total = math_pass + coding_pass + know_pass
total_tests = len(math_tests) + len(coding_tests) + len(know_tests)
print(f"\n{'=' * 60}")
print(f"OVERALL: {total}/{total_tests} ({100*total/total_tests:.0f}%)")
print(f"{'=' * 60}")
