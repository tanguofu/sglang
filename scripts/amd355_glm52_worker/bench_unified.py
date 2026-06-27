#!/usr/bin/env python3
"""Concurrent benchmark client for GLM-5.2 MTP validation on AMD355.

Runs the 6 suites from glm-5.2-amd355-benchmark.md against the sglang server
at http://127.0.0.1:30000/v1/chat/completions, measures aggregate prompt/
completion throughput + latency, and prints a results table.

Suite definitions (prompt_len is approximate target in tokens):
  short_c32      : short prompt,  concurrency 32, max_tokens 128
  short_c128     : short prompt,  concurrency 128, max_tokens 128
  mid_c32        : ~2k prompt,    concurrency 32, max_tokens 512
  prefill16k_c32 : ~16k prompt,   concurrency 32, max_tokens 32
  prefill64k_c4  : ~64k prompt,   concurrency 4,  max_tokens 32
  prefill128k_c1 : ~128k prompt,  concurrency 1,  max_tokens 32
"""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from transformers import AutoTokenizer

BASE_URL = "http://127.0.0.1:30000"
MODEL_PATH = "/data/models/GLM-5.2-FP8"

# (suite, prompt_len, concurrency, max_tokens, num_requests)
# num_requests == concurrency so every wave is one full concurrent wave
# (matches the benchmark doc's per-suite shape: a single saturated wave).
SUITES = [
    ("short_c32",      128,   32,  128, 32),
    ("short_c128",     128,  128,  128, 128),
    ("mid_c32",        2048,  32,  512, 32),
    ("prefill16k_c32", 16384, 32,   32, 32),
    ("prefill64k_c4",  65536,  4,   32,  4),
    ("prefill128k_c1", 131072, 1,   32,  1),
]

FILLER = ("The quick brown fox jumps over the lazy dog. "
          "Inference systems benefit from speculative decoding when the draft "
          "model is cheap and the target is memory-bandwidth bound. ")


def build_prompt(tokenizer, target_tokens):
    """Build a user-message string that tokenizes to ~target_tokens (O(n)).

    For short prompts we use a continuation task that sustains generation so
    completion_tokens ~= max_tokens (cleaner decode throughput). For long
    prompts we replicate filler text to the target length."""
    if target_tokens <= 128:
        return ("Continue the following story with detailed events, dialogue, "
                "and description. Keep writing without ending the story. "
                "Story so far: In a quiet workshop, a robot named Piebo picked "
                "up a brush for the first time and hesitated. ")
    # Encode filler once to find its token count, then replicate.
    ftok = len(tokenizer.encode(FILLER, add_special_tokens=False))
    copies = target_tokens // ftok + 2
    ids = tokenizer.encode(FILLER * copies, add_special_tokens=False)
    if len(ids) > target_tokens:
        ids = ids[:target_tokens]
    return tokenizer.decode(ids, clean_up_tokenization_spaces=True)


def one_request(args, prompt, max_tokens):
    suite, _plen, _conc, _mt, _n = args
    payload = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    t0 = time.time()
    try:
        r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=600)
        dt = time.time() - t0
        if r.status_code != 200:
            return {"ok": False, "err": f"HTTP {r.status_code}: {r.text[:200]}", "dt": dt}
        j = r.json()
        usage = j.get("usage", {})
        return {
            "ok": True,
            "dt": dt,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
    except Exception as e:
        return {"ok": False, "err": str(e)[:200], "dt": time.time() - t0}


def run_suite(args):
    suite, prompt_len, conc, max_tokens, n = args
    print(f"\n=== {suite} | prompt~{prompt_len}tok c={conc} max_tokens={max_tokens} n={n} ===", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    prompt = build_prompt(tok, prompt_len)
    actual_ptok = len(tok.encode(prompt, add_special_tokens=False))
    print(f"  prompt built: {actual_ptok} tokens", flush=True)

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(one_request, args, prompt, max_tokens) for _ in range(n)]
        for f in as_completed(futs):
            results.append(f.result())
    wall = time.time() - t_start

    ok = [r for r in results if r.get("ok")]
    errs = [r for r in results if not r.get("ok")]
    lats = [r["dt"] for r in ok]
    tot_ptok = sum(r["prompt_tokens"] for r in ok)
    tot_ctok = sum(r["completion_tokens"] for r in ok)
    row = {
        "suite": suite,
        "n": n,
        "ok": len(ok),
        "errors": len(errs),
        "wall_s": round(wall, 3),
        "prompt_tok_s": round(tot_ptok / wall, 1) if wall > 0 else 0,
        "completion_tok_s": round(tot_ctok / wall, 1) if wall > 0 else 0,
        "avg_latency_s": round(statistics.mean(lats), 3) if lats else 0,
        "p95_latency_s": round(sorted(lats)[int(0.95 * len(lats)) - 1], 3) if len(lats) >= 2 else (round(lats[0], 3) if lats else 0),
        "prompt_tokens_actual": actual_ptok,
        "first_err": errs[0]["err"] if errs else "",
    }
    print(f"  -> ok={row['ok']}/{n} err={row['errors']} wall={row['wall_s']}s "
          f"prompt_tok/s={row['prompt_tok_s']} completion_tok/s={row['completion_tok_s']} "
          f"avg={row['avg_latency_s']}s p95={row['p95_latency_s']}s", flush=True)
    if errs:
        print(f"  first error: {errs[0]['err']}", flush=True)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suites", default="all", help="comma list or 'all'")
    p.add_argument("--out", default="/data/bench_mtp_cons_results.json")
    a = p.parse_args()

    # quick health check
    try:
        r = requests.get(f"{BASE_URL}/v1/models", timeout=10)
        print("health /v1/models:", r.status_code, flush=True)
    except Exception as e:
        print("health check failed:", e, flush=True)

    sel = SUITES if a.suites == "all" else [s for s in SUITES if s[0] in a.suites.split(",")]
    rows = []
    for s in sel:
        try:
            rows.append(run_suite(s))
        except Exception as e:
            print(f"  SUITE {s[0]} CRASHED: {e}", flush=True)
            rows.append({"suite": s[0], "error": str(e)})

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'suite':<20}{'ok':>5}{'err':>5}{'prompt/s':>12}{'compl/s':>12}{'avg_s':>9}{'p95_s':>9}", flush=True)
    for r in rows:
        if "error" in r:
            print(f"{r['suite']:<20}ERROR: {r['error']}", flush=True)
            continue
        print(f"{r['suite']:<20}{r['ok']:>5}{r['errors']:>5}"
              f"{r['prompt_tok_s']:>12}{r['completion_tok_s']:>12}"
              f"{r['avg_latency_s']:>9}{r['p95_latency_s']:>9}", flush=True)

    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nresults written to {a.out}", flush=True)


if __name__ == "__main__":
    main()
