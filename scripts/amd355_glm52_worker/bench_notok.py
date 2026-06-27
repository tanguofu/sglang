#!/usr/bin/env python3
import json, statistics, time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

BASE_URL = "http://127.0.0.1:30000"
MODEL = "/data/models/GLM-5.2-FP8"
SHORT_PROMPT = ("Continue the following story with detailed events, dialogue, "
    "and description. Keep writing without ending the story. "
    "Story so far: In a quiet workshop, a robot named Piebo picked "
    "up a brush for the first time and hesitated. ")
FILLER = ("The quick brown fox jumps over the lazy dog. "
    "Inference systems benefit from speculative decoding when the draft "
    "model is cheap and the target is memory-bandwidth bound. " * 200)
SUITES = [
    ("short_c32",  SHORT_PROMPT, 32, 128, 32),
    ("short_c128", SHORT_PROMPT, 128, 128, 128),
    ("mid_c32",    FILLER[:8000], 32, 512, 32),
]

def one_request(prompt, max_tokens):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.7, "stream": False}
    t0 = time.time()
    try:
        r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=600)
        dt = time.time() - t0
        if r.status_code != 200:
            return {"ok": False, "err": "HTTP %d: %s" % (r.status_code, r.text[:200]), "dt": dt}
        j = r.json()
        u = j.get("usage", {})
        return {"ok": True, "dt": dt,
                "prompt_tokens": u.get("prompt_tokens", 0),
                "completion_tokens": u.get("completion_tokens", 0)}
    except Exception as e:
        return {"ok": False, "err": str(e), "dt": time.time() - t0}

def run_suite(suite):
    name, prompt, conc, max_tokens, n = suite
    print("\n=== %s | c=%d max_tokens=%d n=%d ===" % (name, conc, max_tokens, n), flush=True)
    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(one_request, prompt, max_tokens) for _ in range(n)]
        for f in as_completed(futs):
            results.append(f.result())
    wall = time.time() - t_start
    ok = [r for r in results if r.get("ok")]
    errs = [r for r in results if not r.get("ok")]
    lats = [r["dt"] for r in ok]
    tot_ptok = sum(r["prompt_tokens"] for r in ok)
    tot_ctok = sum(r["completion_tokens"] for r in ok)
    row = {"suite": name, "n": n, "ok": len(ok), "errors": len(errs),
           "wall_s": round(wall, 3),
           "prompt_tok_s": round(tot_ptok / wall, 1) if wall > 0 else 0,
           "completion_tok_s": round(tot_ctok / wall, 1) if wall > 0 else 0,
           "avg_latency_s": round(statistics.mean(lats), 3) if lats else 0,
           "p95_latency_s": round(sorted(lats)[int(0.95 * len(lats)) - 1], 3) if len(lats) >= 2 else (round(lats[0], 3) if lats else 0)}
    print("  -> ok=%d/%d err=%d wall=%.3fs prompt_tok/s=%.1f completion_tok/s=%.1f avg=%.3fs p95=%.3fs" % (
        row["ok"], n, row["errors"], row["wall_s"], row["prompt_tok_s"], row["completion_tok_s"], row["avg_latency_s"], row["p95_latency_s"]), flush=True)
    if errs:
        print("  first error: %s" % errs[0]["err"], flush=True)
    return row

if __name__ == "__main__":
    try:
        r = requests.get(f"{BASE_URL}/v1/models", timeout=10)
        print("health:", r.status_code, flush=True)
    except Exception as e:
        print("health failed:", e, flush=True)
    rows = [run_suite(s) for s in SUITES]
    print("\n=== SUMMARY ===")
    print("%-20s %4s %4s %10s %10s %8s %8s" % ("suite", "ok", "err", "prompt/s", "compl/s", "avg_s", "p95_s"))
    for r in rows:
        print("%-20s %4d %4d %10.1f %10.1f %8.3f %8.3f" % (r["suite"], r["ok"], r["errors"], r["prompt_tok_s"], r["completion_tok_s"], r["avg_latency_s"], r["p95_latency_s"]))
    out = sys.argv[1] if len(sys.argv) > 1 else "/data/bench_results.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print("\nresults written to %s" % out)
