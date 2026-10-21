#!/usr/bin/env python3
"""Compile the per-batch-size DSV4.1 kernels before traffic arrives.

TileLang keys its kernel cache on the generated prim_func, so the DSA prefill
attention kernel is rebuilt whenever the batch shape changes.  On gfx942 that
rebuild takes ~70 s per new batch size, and in this image the compiled artifact
does not reach the on-disk cache (the process keeps it in memory only), so a
freshly started server pays it again on the first request of every batch size.

Driving one tiny request through each target batch size moves that cost into
startup, where a readinessProbe can hold traffic back.

Usage: warmup_ds41.py --port 30000 --model DeepSeek-V4.1-Flash
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

BATCHES = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128)
PROMPT = "Warm up: reply with one word."


def one_request(url, model, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 4,
        "temperature": 0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            json.load(resp)
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def warm_batch(url, model, batch, timeout):
    ok = [0]
    lock = threading.Lock()

    def task():
        good = one_request(url, model, timeout)
        with lock:
            ok[0] += int(good)

    threads = [threading.Thread(target=task) for _ in range(batch)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return ok[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--batches", default=",".join(str(b) for b in BATCHES))
    args = ap.parse_args()

    url = "http://%s:%d/v1/chat/completions" % (args.host, args.port)
    batches = [int(x) for x in args.batches.split(",") if x.strip()]

    failures = 0
    for batch in batches:
        start = time.time()
        ok = warm_batch(url, args.model, batch, args.timeout)
        elapsed = time.time() - start
        status = "ok" if ok == batch else "PARTIAL %d/%d" % (ok, batch)
        if ok != batch:
            failures += 1
        print(
            "warmup bs=%-3d %7.2fs  %s" % (batch, elapsed, status),
            flush=True,
        )

    if failures:
        # A partial warm-up still leaves the server usable; report and exit 0 so
        # the readiness gate opens rather than hanging the rollout.
        print("warmup finished with %d incomplete batch sizes" % failures, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
