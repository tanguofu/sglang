#!/usr/bin/env python3
"""Accuracy test for GLM-5.2 FP8 on optimized worker.
Tests: coding (HumanEval-style), math (GSM8K-style), and reasoning.
Compares worker output against known-correct answers."""
import json
import re
import time
import requests
import concurrent.futures

URL = "http://localhost:30000/generate"

# ============================================================
# Test 1: Coding problems (HumanEval-style, deterministic answers)
# ============================================================
CODING_TESTS = [
    {
        "id": "coding_1",
        "prompt": "Write a Python function that returns the sum of two numbers. Only output the function code, no explanation.\n\ndef add(a, b):",
        "check": lambda resp: "return a + b" in resp or "return a+b" in resp,
        "max_tokens": 64,
    },
    {
        "id": "coding_2",
        "prompt": "Write a Python function that reverses a string. Only output the function code.\n\ndef reverse_string(s):",
        "check": lambda resp: "return s[::-1]" in resp or "return s[::-1]" in resp,
        "max_tokens": 64,
    },
    {
        "id": "coding_3",
        "prompt": "Write a Python function that checks if a number is even. Only output the function code.\n\ndef is_even(n):",
        "check": lambda resp: "return n % 2 == 0" in resp or "return n%2==0" in resp,
        "max_tokens": 64,
    },
    {
        "id": "coding_4",
        "prompt": "Write a Python function that returns the factorial of n using recursion. Only output the function code.\n\ndef factorial(n):",
        "check": lambda resp: "return n * factorial(n - 1)" in resp or "return n * factorial(n-1)" in resp,
        "max_tokens": 128,
    },
    {
        "id": "coding_5",
        "prompt": "Write a Python function that finds the maximum element in a list. Only output the function code.\n\ndef find_max(lst):",
        "check": lambda resp: "return max(lst)" in resp or "max(" in resp,
        "max_tokens": 64,
    },
    {
        "id": "coding_6",
        "prompt": "Write a Python function that merges two sorted lists into one sorted list. Only output the function code.\n\ndef merge_sorted(a, b):",
        "check": lambda resp: "sorted" in resp or "heapq" in resp or "merge" in resp.lower(),
        "max_tokens": 128,
    },
    {
        "id": "coding_7",
        "prompt": "Write a Python function that checks if a string is a palindrome. Only output the function code.\n\ndef is_palindrome(s):",
        "check": lambda resp: "s[::-1]" in resp or "reversed" in resp,
        "max_tokens": 64,
    },
    {
        "id": "coding_8",
        "prompt": "Write a Python function that returns the n-th Fibonacci number. Only output the function code.\n\ndef fibonacci(n):",
        "check": lambda resp: "fibonacci(n-1)" in resp or "fibonacci(n - 1)" in resp or "fib(n-1)" in resp,
        "max_tokens": 128,
    },
    {
        "id": "coding_9",
        "prompt": "Write a Python function that counts the number of vowels in a string. Only output the function code.\n\ndef count_vowels(s):",
        "check": lambda resp: "aeiou" in resp or "AEIOU" in resp or "aeiouAEIOU" in resp,
        "max_tokens": 128,
    },
    {
        "id": "coding_10",
        "prompt": "Write a Python function that removes duplicates from a list while preserving order. Only output the function code.\n\ndef remove_duplicates(lst):",
        "check": lambda resp: "set(" in resp or "dict" in resp or "seen" in resp,
        "max_tokens": 128,
    },
]

# ============================================================
# Test 2: Math problems (GSM8K-style, deterministic answers)
# ============================================================
MATH_TESTS = [
    {
        "id": "math_1",
        "prompt": "Question: A store sells apples at $2 each. If you buy 5 apples, how much do you pay?\nAnswer:",
        "answer": 10,
        "max_tokens": 128,
    },
    {
        "id": "math_2",
        "prompt": "Question: Tom has 3 boxes, each containing 4 pencils. How many pencils does Tom have in total?\nAnswer:",
        "answer": 12,
        "max_tokens": 128,
    },
    {
        "id": "math_3",
        "prompt": "Question: A train travels 60 miles in 1 hour. How far will it travel in 3.5 hours?\nAnswer:",
        "answer": 210,
        "max_tokens": 128,
    },
    {
        "id": "math_4",
        "prompt": "Question: If 15% of 200 students play basketball, how many students play basketball?\nAnswer:",
        "answer": 30,
        "max_tokens": 128,
    },
    {
        "id": "math_5",
        "prompt": "Question: A rectangle has length 8 and width 5. What is its area?\nAnswer:",
        "answer": 40,
        "max_tokens": 128,
    },
    {
        "id": "math_6",
        "prompt": "Question: If x + 5 = 12, what is x?\nAnswer:",
        "answer": 7,
        "max_tokens": 64,
    },
    {
        "id": "math_7",
        "prompt": "Question: A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices are left?\nAnswer:",
        "answer": 2,
        "max_tokens": 128,
    },
    {
        "id": "math_8",
        "prompt": "Question: What is 25% of 80?\nAnswer:",
        "answer": 20,
        "max_tokens": 64,
    },
    {
        "id": "math_9",
        "prompt": "Question: If a shirt costs $25 and is on sale for 20% off, what is the sale price?\nAnswer:",
        "answer": 20,
        "max_tokens": 128,
    },
    {
        "id": "math_10",
        "prompt": "Question: A car uses 3 gallons of gas per hour. How many gallons does it use in 4.5 hours?\nAnswer:",
        "answer": 13.5,
        "max_tokens": 128,
    },
]

# ============================================================
# Test 3: Knowledge / reasoning (deterministic factual answers)
# ============================================================
KNOWLEDGE_TESTS = [
    {
        "id": "know_1",
        "prompt": "What is the capital of France? Answer with just the city name.",
        "check": lambda resp: "paris" in resp.lower(),
        "max_tokens": 32,
    },
    {
        "id": "know_2",
        "prompt": "What is 2 + 2? Answer with just the number.",
        "check": lambda resp: "4" in resp.strip().split("\n")[0],
        "max_tokens": 16,
    },
    {
        "id": "know_3",
        "prompt": "What programming language is Django written in? Answer with just the language name.",
        "check": lambda resp: "python" in resp.lower(),
        "max_tokens": 16,
    },
    {
        "id": "know_4",
        "prompt": "What does CPU stand for? Answer briefly.",
        "check": lambda resp: "central processing unit" in resp.lower(),
        "max_tokens": 32,
    },
    {
        "id": "know_5",
        "prompt": "What is the chemical symbol for water? Answer with just the symbol.",
        "check": lambda resp: "h2o" in resp.lower() or "H₂O" in resp,
        "max_tokens": 16,
    },
]

def send_request(prompt, max_tokens=128, temperature=0.0):
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
        },
    }
    try:
        r = requests.post(URL, json=payload, timeout=120)
        if r.status_code == 200:
            return r.json().get("text", "")
        return f"ERROR: {r.status_code}"
    except Exception as e:
        return f"ERROR: {e}"

def extract_number(text):
    numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if numbers:
        try:
            return float(numbers[-1])
        except:
            return None
    return None

def run_tests():
    results = {"coding": [], "math": [], "knowledge": []}
    
    # --- Coding tests ---
    print("\n" + "="*60)
    print("CODING TESTS (HumanEval-style)")
    print("="*60)
    for t in CODING_TESTS:
        resp = send_request(t["prompt"], t["max_tokens"])
        passed = t["check"](resp)
        results["coding"].append({"id": t["id"], "passed": passed, "response": resp[:200]})
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {t['id']}: {status}")
        if not passed:
            print(f"    Response: {resp[:150]}")
    
    # --- Math tests ---
    print("\n" + "="*60)
    print("MATH TESTS (GSM8K-style)")
    print("="*60)
    for t in MATH_TESTS:
        resp = send_request(t["prompt"], t["max_tokens"])
        extracted = extract_number(resp)
        passed = extracted is not None and abs(extracted - t["answer"]) < 0.01
        results["math"].append({"id": t["id"], "passed": passed, "expected": t["answer"], "got": extracted, "response": resp[:200]})
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {t['id']}: {status} (expected={t['answer']}, got={extracted})")
        if not passed:
            print(f"    Response: {resp[:150]}")
    
    # --- Knowledge tests ---
    print("\n" + "="*60)
    print("KNOWLEDGE TESTS")
    print("="*60)
    for t in KNOWLEDGE_TESTS:
        resp = send_request(t["prompt"], t["max_tokens"])
        passed = t["check"](resp)
        results["knowledge"].append({"id": t["id"], "passed": passed, "response": resp[:200]})
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {t['id']}: {status}")
        if not passed:
            print(f"    Response: {resp[:150]}")
    
    # --- Summary ---
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for category, tests in results.items():
        passed = sum(1 for t in tests if t["passed"])
        total = len(tests)
        print(f"  {category.upper()}: {passed}/{total} ({100*passed/total:.0f}%)")
    
    total_passed = sum(1 for cat in results.values() for t in cat if t["passed"])
    total_tests = sum(len(cat) for cat in results.values())
    print(f"\n  OVERALL: {total_passed}/{total_tests} ({100*total_passed/total_tests:.0f}%)")
    
    return results

if __name__ == "__main__":
    run_tests()
