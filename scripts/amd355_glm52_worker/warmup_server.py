#!/usr/bin/env python3
"""Warmup script: pre-run inference to JIT-compile AITER kernels and warm CUDA graphs.
Run after server is healthy to eliminate cold-start penalty."""
import requests
import time
import sys

URL = "http://localhost:30000/generate"

def warmup_request(input_text, max_tokens=32, concurrency=1):
    """Send a single warmup request."""
    payload = {
        "text": input_text,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
    }
    try:
        r = requests.post(URL, json=payload, timeout=120)
        return r.status_code == 200
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def main():
    print("[WARMUP] Starting warmup sequence...")
    
    # Phase 1: Single request with short input (warms up decode CUDA graph for bs=1)
    print("[WARMUP] Phase 1: Short input, bs=1 decode...")
    warmup_request("Hello", max_tokens=16)
    print("  Done")
    
    # Phase 2: Medium input (warms up prefill path + AITER kernels for medium shapes)
    print("[WARMUP] Phase 2: Medium input (512 tokens)...")
    text = "The quick brown fox jumps over the lazy dog. " * 64  # ~512 tokens
    warmup_request(text, max_tokens=16)
    print("  Done")
    
    # Phase 3: Long input (warms up prefill path for 4K context + DSA indexer)
    print("[WARMUP] Phase 3: Long input (4096 tokens)...")
    text = "The quick brown fox jumps over the lazy dog. " * 512  # ~4096 tokens
    warmup_request(text, max_tokens=16)
    print("  Done")
    
    # Phase 4: Batch of 32 short requests (warms up decode CUDA graph for bs=32)
    print("[WARMUP] Phase 4: Batch of 32 short requests...")
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(warmup_request, f"Test {i}", max_tokens=16) for i in range(32)]
        concurrent.futures.wait(futures)
    print("  Done")
    
    # Phase 5: Batch of 32 medium requests (warms up prefill + decode for batch)
    print("[WARMUP] Phase 5: Batch of 32 medium requests...")
    text = "The quick brown fox jumps over the lazy dog. " * 128  # ~1024 tokens
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(warmup_request, text, max_tokens=16) for i in range(32)]
        concurrent.futures.wait(futures)
    print("  Done")
    
    print("[WARMUP] Warmup complete! Server is now warm.")

if __name__ == "__main__":
    main()
