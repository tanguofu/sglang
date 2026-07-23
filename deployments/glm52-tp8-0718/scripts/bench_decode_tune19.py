#!/usr/bin/env python3
"""
Decode-heavy fixed load for tune19 aiter GEMM tuning BEFORE/AFTER comparison.

Hits a single sglang worker directly (no router) at /v1/chat/completions with
streaming. Sends N concurrent requests, each with a long prompt (~ctx_tokens)
and a short max_tokens budget so the run is decode-dominated. Reports aggregate
generation throughput (tok/s) and wall time.

Usage:
  python3 bench_decode_tune19.py --endpoint http://21.234.170.19:30000 \
      --api-key sk-... --concurrency 8 --ctx-tokens 12000 --max-tokens 256 \
      --duration 180 --out results/tune19-before.json
"""
import argparse
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import urllib.request


def build_prompt(ctx_tokens: int) -> str:
    header = (
        "You are a senior SRE agent. Below is the accumulated context from prior "
        "investigation turns. Read it, then answer the final question concisely.\n\n"
    )
    block = (
        "user: Analyze the following system metrics and propose a remediation plan. "
        "Here is the transcript from the last investigation turn.\n"
        "assistant: Understood. I will review the metrics, identify the bottleneck, "
        "and propose a step-by-step remediation. Let me gather the relevant telemetry.\n"
        "tool_call: query_prometheus(metric='sglang:num_requests_running', range='5m')\n"
        "tool_result: peak=18, p95=15, mean=11.2; worker-1 carries 0.88 of the load.\n"
        "assistant: The imbalance is caused by cache_aware sticking to the worker that "
        "holds the prefix. I will verify the threshold settings and the selection "
        "distribution next.\n"
    )
    per_block_tokens = 270
    n_blocks = max(1, ctx_tokens // per_block_tokens)
    body = "\n".join(f"--- turn {i:04d} ---\n{block}" for i in range(n_blocks))
    tail = (
        "\n\nFinal question: in one sentence, what is the single highest-impact action "
        "to restore load balance?"
    )
    return header + body + tail


def stream_one(endpoint, api_key, prompt, max_tokens, req_id, stop_evt):
    payload = json.dumps({
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }).encode()
    url = f"{endpoint}/v1/chat/completions"
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    t0 = time.perf_counter()
    n_tokens = 0
    http_code = 0
    err = None
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            http_code = resp.getcode()
            for raw in resp:
                if stop_evt.is_set():
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue
                ch = evt.get("choices")
                if not ch:
                    continue
                delta = ch[0].get("delta", {})
                # GLM-5.2 is a reasoning model: count both reasoning_content and
                # content deltas — both are decode output tokens.
                content = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if content:
                    n_tokens += 1
    except Exception as e:
        err = str(e)[:200]
    wall = time.perf_counter() - t0
    return {"req_id": req_id, "http_code": http_code, "n_tokens": n_tokens,
            "wall": wall, "ok": http_code == 200 and err is None, "err": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://21.234.170.19:30000")
    ap.add_argument("--api-key", default="sk-46faecc9d0bc4dcd9db6a15c73ae91c8")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--ctx-tokens", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--duration", type=int, default=180, help="wall budget seconds")
    ap.add_argument("--out", default="results/tune19-bench.json")
    args = ap.parse_args()

    prompt = build_prompt(args.ctx_tokens)
    print(f"prompt chars={len(prompt)} (~{len(prompt)//4} tokens)", flush=True)
    print(f"concurrency={args.concurrency} max_tokens={args.max_tokens} "
          f"duration={args.duration}s", flush=True)

    stop_evt = threading.Event()
    results = []
    req_id = 0
    t0 = time.perf_counter()

    def submit(ex, rid):
        return ex.submit(stream_one, args.endpoint, args.api_key, prompt,
                         args.max_tokens, rid, stop_evt)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [submit(ex, i) for i in range(args.concurrency)]
        while time.perf_counter() - t0 < args.duration:
            # top up to keep concurrency saturated
            done = [f for f in futs if f.done()]
            if done:
                for f in done:
                    futs.remove(f)
                    r = f.result()
                    results.append(r)
                    print(f"  req {r['req_id']:03d}: {'OK' if r['ok'] else 'FAIL'} "
                          f"tok={r['n_tokens']} wall={r['wall']:.1f}s "
                          f"tput={r['n_tokens']/max(r['wall'],1e-6):.1f}t/s", flush=True)
                    req_id += 1
                    futs.append(submit(ex, req_id))
            time.sleep(0.2)
        stop_evt.set()
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"  req {r['req_id']:03d}: {'OK' if r['ok'] else 'FAIL'} "
                  f"tok={r['n_tokens']} wall={r['wall']:.1f}s", flush=True)

    wall = time.perf_counter() - t0
    oks = [r for r in results if r["ok"]]
    total_tok = sum(r["n_tokens"] for r in oks)
    gen_throughput = total_tok / wall if wall > 0 else 0
    per_req_tput = [r["n_tokens"] / max(r["wall"], 1e-6) for r in oks if r["n_tokens"] > 0]

    summary = {
        "endpoint": args.endpoint,
        "concurrency": args.concurrency,
        "ctx_tokens_target": args.ctx_tokens,
        "max_tokens": args.max_tokens,
        "duration_budget_s": args.duration,
        "wall_clock_s": round(wall, 2),
        "n_requests": len(results),
        "n_ok": len(oks),
        "n_fail": len(results) - len(oks),
        "total_output_tokens": total_tok,
        "aggregate_gen_throughput_tps": round(gen_throughput, 2),
        "per_request_throughput_tps": {
            "mean": round(sum(per_req_tput) / len(per_req_tput), 2) if per_req_tput else None,
            "max": round(max(per_req_tput), 2) if per_req_tput else None,
        },
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
