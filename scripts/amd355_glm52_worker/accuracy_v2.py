#!/usr/bin/env python3
"""Accuracy test v2: uses raw generate API with proper prompts."""
import requests, re, json

URL = "http://localhost:30000/generate"

def send(prompt, max_tokens=256):
    payload = {"text": prompt, "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0}}
    r = requests.post(URL, json=payload, timeout=120)
    return r.json().get("text", "")

def extract_num(text):
    numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if numbers:
        try: return float(numbers[-1])
        except: return None
    return None

# Math tests with clear single-answer prompts
math_tests = [
    ("math_1", "A store sells apples at $2 each. If you buy 5 apples, how much do you pay? Give only the number.", 10),
    ("math_2", "Tom has 3 boxes, each containing 4 pencils. How many pencils does Tom have in total? Give only the number.", 12),
    ("math_3", "A train travels 60 miles in 1 hour. How far will it travel in 3.5 hours? Give only the number.", 210),
    ("math_4", "If 15% of 200 students play basketball, how many students play basketball? Give only the number.", 30),
    ("math_5", "A rectangle has length 8 and width 5. What is its area? Give only the number.", 40),
    ("math_6", "If x + 5 = 12, what is x? Give only the number.", 7),
    ("math_7", "A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices are left? Give only the number.", 2),
    ("math_8", "What is 25% of 80? Give only the number.", 20),
    ("math_9", "If a shirt costs $25 and is on sale for 20% off, what is the sale price? Give only the number.", 20),
    ("math_10", "A car uses 3 gallons of gas per hour. How many gallons does it use in 4.5 hours? Give only the number.", 13.5),
]

# Coding tests
coding_tests = [
    ("coding_1", "Write a Python function that returns the sum of two numbers.\n\ndef add(a, b):", lambda r: "return a + b" in r or "return a+b" in r),
    ("coding_2", "Write a Python function that reverses a string.\n\ndef reverse_string(s):", lambda r: "s[::-1]" in r),
    ("coding_3", "Write a Python function that checks if a number is even.\n\ndef is_even(n):", lambda r: "n % 2 == 0" in r or "n%2==0" in r),
    ("coding_4", "Write a Python function that returns the factorial of n using recursion.\n\ndef factorial(n):", lambda r: "factorial(n - 1)" in r or "factorial(n-1)" in r),
    ("coding_5", "Write a Python function that finds the maximum element in a list.\n\ndef find_max(lst):", lambda r: "max(lst)" in r or "max(" in r),
    ("coding_6", "Write a Python function that checks if a string is a palindrome.\n\ndef is_palindrome(s):", lambda r: "s[::-1]" in r or "reversed" in r),
    ("coding_7", "Write a Python function that returns the n-th Fibonacci number using recursion.\n\ndef fibonacci(n):", lambda r: "fibonacci(n - 1)" in r or "fibonacci(n-1)" in r),
    ("coding_8", "Write a Python function that counts vowels in a string.\n\ndef count_vowels(s):", lambda r: "aeiou" in r.lower() or "AEIOU" in r),
    ("coding_9", "Write a Python function that removes duplicates from a list.\n\ndef remove_duplicates(lst):", lambda r: "set(" in r or "seen" in r or "dict" in r),
    ("coding_10", "Write a Python function that sorts a list in ascending order.\n\ndef sort_list(lst):", lambda r: "sorted(lst)" in r or "sort()" in r or "sorted(" in r),
]

# Knowledge tests
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
]

print("=" * 60)
print("MATH TESTS (GSM8K-style)")
print("=" * 60)
math_pass = 0
for tid, prompt, expected in math_tests:
    resp = send(prompt, max_tokens=64)
    got = extract_num(resp)
    passed = got is not None and abs(got - expected) < 0.01
    if passed: math_pass += 1
    status = "PASS" if passed else "FAIL"
    print(f"  {tid}: {status} (expected={expected}, got={got})")
    if not passed:
        print(f"    Response: {repr(resp[:200])}")

print(f"\n  MATH: {math_pass}/{len(math_tests)} ({100*math_pass/len(math_tests):.0f}%)")

print("\n" + "=" * 60)
print("CODING TESTS (HumanEval-style)")
print("=" * 60)
coding_pass = 0
for tid, prompt, check in coding_tests:
    resp = send(prompt, max_tokens=128)
    passed = check(resp)
    if passed: coding_pass += 1
    status = "PASS" if passed else "FAIL"
    print(f"  {tid}: {status}")
    if not passed:
        print(f"    Response: {repr(resp[:200])}")

print(f"\n  CODING: {coding_pass}/{len(coding_tests)} ({100*coding_pass/len(coding_tests):.0f}%)")

print("\n" + "=" * 60)
print("KNOWLEDGE TESTS")
print("=" * 60)
know_pass = 0
for tid, prompt, check in know_tests:
    resp = send(prompt, max_tokens=32)
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
