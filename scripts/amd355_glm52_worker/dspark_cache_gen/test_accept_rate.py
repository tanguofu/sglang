#!/usr/bin/env python3
"""Test DSpark accept_rate by sending requests to a DSPARK sglang server.

Usage: python3 test_accept_rate.py <sglang_url> [num_requests]

Reports the accept_rate metric from the server's response metadata.
"""
import requests, sys, time, json

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:30000"
num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 20

prompts = [
    "Write a Python function to reverse a string.",
    "Explain how quicksort works.",
    "What is the time complexity of binary search?",
    "Write a SQL query to find the second highest salary.",
    "Implement a binary tree in Python.",
    "Explain the difference between TCP and UDP.",
    "Write a function to check if a string is a palindrome.",
    "How does garbage collection work in Python?",
    "Write a regex to match email addresses.",
    "Explain the CAP theorem.",
] * (num_requests // 10 + 1)

print(f"Testing accept_rate against {url} ({num_requests} requests)")
print()

accept_rates = []
for i in range(num_requests):
    try:
        resp = requests.post(f"{url}/generate", json={
            "text": prompts[i],
            "sampling_params": {"max_new_tokens": 128, "temperature": 0},
        }, timeout=120)
        data = resp.json()
        meta = data.get("meta_info", {})
        # DSpark reports accept metrics in meta_info
        accept_len = meta.get("spec_accept_len") or meta.get("accept_len")
        num_draft = meta.get("spec_num_draft_tokens") or meta.get("num_draft_tokens")
        completion_tokens = meta.get("completion_tokens", 0)

        if accept_len is not None and num_draft and num_draft > 0:
            rate = accept_len / num_draft
            accept_rates.append(rate)
            print(f"  req {i}: accept_len={accept_len} num_draft={num_draft} rate={rate:.2f} tokens={completion_tokens}")
        else:
            # Try to find accept info in other fields
            print(f"  req {i}: tokens={completion_tokens} meta_keys={list(meta.keys())[:8]}")
    except Exception as e:
        print(f"  req {i}: ERROR — {e}")

if accept_rates:
    avg = sum(accept_rates) / len(accept_rates)
    print(f"\n=== RESULTS ===")
    print(f"Average accept_rate: {avg:.2f} (over {len(accept_rates)} requests)")
    print(f"Min: {min(accept_rates):.2f}  Max: {max(accept_rates):.2f}")
else:
    print("\nNo accept_rate data found. Check server configuration.")
