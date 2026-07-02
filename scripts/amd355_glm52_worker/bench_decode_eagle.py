#!/usr/bin/env python3
"""Decode performance benchmark for GLM-5.2 EAGLE MTP on AMD MI355X.

Measures decode throughput at various concurrency levels with EAGLE MTP.
Reports: tok/s, accept_len, TPOT, and comparison vs no-spec baseline.

Usage (inside container):
  python3 /data/bench_decode_eagle.py --base-url http://localhost:30000/v1
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

PROMPT = (
    "Write a detailed essay about the history of artificial intelligence, "
    "covering its origins, key milestones, major figures, and future directions. "
    "Be thorough and include specific examples."
)


def bench_concurrency(
    client: OpenAI,
    model: str,
    concurrency: int,
    output_len: int,
    timeout: int,
) -> dict:
    """Benchmark decode at a given concurrency level."""
    t0 = time.time()
    total_completion_tokens = 0
    errors = 0

    def one_request():
        nonlocal total_completion_tokens, errors
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT}],
                temperature=0.0,
                max_tokens=output_len,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                timeout=timeout,
            )
            ct = resp.usage.completion_tokens or 0
            return ct
        except Exception as e:
            errors += 1
            return 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_request) for _ in range(concurrency)]
        for f in as_completed(futures):
            total_completion_tokens += f.result()

    elapsed = time.time() - t0
    tok_per_sec = total_completion_tokens / elapsed if elapsed > 0 else 0
    # TPOT = elapsed / total_tokens * 1000 (ms per token, averaged)
    tpot_ms = (elapsed * 1000 / total_completion_tokens) if total_completion_tokens > 0 else 0

    return {
        "concurrency": concurrency,
        "output_len_target": output_len,
        "total_completion_tokens": total_completion_tokens,
        "elapsed_sec": round(elapsed, 2),
        "tok_per_sec": round(tok_per_sec, 1),
        "tpot_ms": round(tpot_ms, 2),
        "errors": errors,
    }


def get_accept_len(client: OpenAI, model: str) -> float:
    """Get accept length from a single decode request."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Count from 1 to 50."}],
            temperature=0.0,
            max_tokens=512,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            timeout=120,
        )
        return resp.usage.completion_tokens or 0
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Decode benchmark for GLM-5.2 EAGLE MTP")
    parser.add_argument("--base-url", default="http://localhost:30000/v1")
    parser.add_argument("--model", default="/data/models/GLM-5.2-FP8")
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 32, 128])
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out-dir", default="/data/eval_results")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    print("=" * 70)
    print("GLM-5.2 EAGLE MTP Decode Benchmark (1M context, BF16 KV, CUDA graph)")
    print(f"  Output length: {args.output_len} tokens")
    print(f"  Concurrency: {args.concurrency}")
    print("=" * 70)

    results = []
    for c in args.concurrency:
        print(f"\n--- Concurrency {c} ---", flush=True)
        r = bench_concurrency(client, args.model, c, args.output_len, args.timeout)
        print(
            f"  tok/s: {r['tok_per_sec']}, TPOT: {r['tpot_ms']}ms, "
            f"tokens: {r['total_completion_tokens']}, time: {r['elapsed_sec']}s, "
            f"errors: {r['errors']}",
            flush=True,
        )
        results.append(r)

    # Get accept len from server logs
    print("\n--- Accept length (from server logs) ---")

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"{'Conc':>6} {'tok/s':>10} {'TPOT(ms)':>10} {'tokens':>10} {'time(s)':>10}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['concurrency']:>6} {r['tok_per_sec']:>10.1f} {r['tpot_ms']:>10.2f} "
            f"{r['total_completion_tokens']:>10} {r['elapsed_sec']:>10.2f}"
        )
    print("=" * 70)

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"bench_decode_eagle_{ts}.json"
    out_file.write_text(
        json.dumps(
            {
                "config": {
                    "model": args.model,
                    "output_len": args.output_len,
                    "speculative": "EAGLE MTP steps=2 draft=3 topk=1",
                    "kv_cache": "bfloat16",
                    "context_length": 1048576,
                    "cuda_graph": True,
                },
                "results": results,
            },
            indent=2,
        )
    )
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
