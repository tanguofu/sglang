#!/usr/bin/env python3
"""
Thorough long context garbled output test.
Use higher max_tokens and better garbled detection.
"""
import json
import time
import re
import requests
from collections import Counter

ROUTER_1P1D = "http://sglang-1p1d-router.kube-system:30001"
ROUTER_2TP8 = "http://sglang-glm52-2tp8-router.kube-system:30080"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

FILLER = "The quick brown fox jumps over the lazy dog. "


def build_prompt(target_tokens, question="What is 2 + 3? Reply with just the number."):
    reps = max(1, target_tokens // 11)
    context = FILLER * reps
    return (f"Below is a long document for you to analyze:\n\n{context}\n\n"
            f"Document ends here. Now please answer: {question}")


def detect_garbled(text):
    """Better garbled detection."""
    if not text or not text.strip():
        return ["empty"]
    signs = []
    # 1. Repetition: check for n-gram repetition
    words = text.split()
    if len(words) > 8:
        # Check if any 3-word sequence repeats >3 times
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        common = Counter(trigrams).most_common(1)
        if common and common[0][1] > 3:
            signs.append(f"repetition ({common[0][1]}x: {' '.join(common[0][0])})")
    # 2. Char-level repetition
    if len(text) > 20:
        # Check if any 10-char substring repeats >2 times
        chunks = [text[i:i+10] for i in range(0, len(text) - 10, 5)]
        chunk_counts = Counter(chunks)
        for chunk, count in chunk_counts.items():
            if count > 2 and len(chunk.strip()) > 3:
                signs.append(f"char-rep ({count}x: {repr(chunk)})")
                break
    # 3. Non-printable chars
    weird = sum(1 for c in text if ord(c) > 0xFFFF or (ord(c) < 32 and c not in "\n\t"))
    if weird > 3:
        signs.append(f"{weird} non-printable")
    # 4. Mixed CJK + latin incoherently (for this English-only test)
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if cjk > 5 and not any(w in text.lower() for w in ["chinese", "中文", "汉字"]):
        signs.append(f"{cjk} unexpected CJK chars")
    # 5. Markdown artifact spam
    if text.count("**") > 4 and len(text) < 200:
        signs.append("markdown spam")
    return signs or ["NONE"]


def test(url, label, input_tokens, max_tokens=300):
    print(f"\n[{label}] {input_tokens//1000}k input, max_tokens={max_tokens}...")
    prompt = build_prompt(input_tokens)
    payload = {
        "model": "glm-5.2" if "1p1d" in label else "unknown",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{url}/v1/chat/completions", json=payload,
                          headers=HEADERS, timeout=600)
        elapsed = time.perf_counter() - t0
        if r.status_code != 200:
            print(f"  ERROR: HTTP {r.status_code}: {r.text[:200]}")
            return {"label": label, "input": input_tokens, "error": f"HTTP {r.status_code}"}
        data = r.json()
        content = data["choices"][0]["message"].get("content", "")
        reasoning = data["choices"][0]["message"].get("reasoning_content", "")
        usage = data.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)

        content_garbled = detect_garbled(content)
        reasoning_garbled = detect_garbled(reasoning)

        print(f"  Time: {elapsed:.1f}s, prompt_tok={pt}, output_tok={ct}")
        print(f"  Content ({len(content)} chars): {repr(content[:300])}")
        print(f"  Content garbled: {content_garbled}")
        if reasoning:
            print(f"  Reasoning ({len(reasoning)} chars): {repr(reasoning[:300])}")
            print(f"  Reasoning garbled: {reasoning_garbled}")

        return {
            "label": label, "input": input_tokens, "elapsed": elapsed,
            "prompt_tokens": pt, "completion_tokens": ct,
            "content": content[:500], "content_garbled": content_garbled,
            "reasoning": reasoning[:500], "reasoning_garbled": reasoning_garbled,
        }
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return {"label": label, "input": input_tokens, "error": str(e)}


def main():
    results = []
    # Test with higher max_tokens for clearer output
    for input_tokens in [80000, 120000]:
        for label, url in [("1p1d", ROUTER_1P1D), ("2tp8", ROUTER_2TP8)]:
            r = test(url, label, input_tokens, max_tokens=300)
            results.append(r)
            time.sleep(3)

    print(f"\n{'=' * 70}")
    print(f"  GARBLED OUTPUT SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Test':<16} {'Content':>10} {'Reasoning':>10} {'Time':>8}")
    print("-" * 50)
    for r in results:
        label = f"{r['label']}_{r['input']//1000}k"
        if "error" in r:
            print(f"{label:<16} {'ERROR':>10} {'-':>10} {'-':>8}")
        else:
            cg = "GARBLED" if r["content_garbled"] != ["NONE"] else "ok"
            rg = "GARBLED" if r["reasoning_garbled"] != ["NONE"] and r["reasoning_garbled"] != ["empty"] else "ok"
            print(f"{label:<16} {cg:>10} {rg:>10} {r.get('elapsed',0):>7.1f}s")

    # Save
    with open("/tmp/bench-compare/long_context_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: /tmp/bench-compare/long_context_results.json")


if __name__ == "__main__":
    main()
