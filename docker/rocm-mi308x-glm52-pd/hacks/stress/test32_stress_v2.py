#!/usr/bin/env python3
"""EAGLE A/B + hicache stress client — direct to worker localhost:30000.

Fixed-load streaming client for the EAGLE sweep and the hicache
repeated-prefix validation. Reports per-wave and aggregate: ok/fail,
generated tokens, mean/p50/p95 wall latency. Run inside the worker pod
via kubectl exec. Identical prompt + temperature 0 across all configs.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request

URL = "http://localhost:30000/v1/chat/completions"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"

BLOCK = (
    "user: Analyze the following system metrics and propose a remediation plan. "
    "Here is the transcript from the last investigation turn.\n"
    "assistant: Understood. I will review the metrics, identify the bottleneck, "
    "and propose a step-by-step remediation. Let me gather telemetry.\n"
    "tool_call: query_prometheus(metric='sglang:num_requests_running', range='5m')\n"
    "tool_result: peak=18, p95=15, mean=11.2 across workers; worker-1 carries "
    "0.88 of the load under cache_aware routing.\n"
    "tool_call: query_prometheus(metric='sglang:cache_hit_rate', range='5m')\n"
    "tool_result: worker-1=0.62, worker-2=0.04; prefix cache concentrated on "
    "worker-1 because the shared system prompt only resides there.\n"
    "assistant: The imbalance is caused by cache_aware sticking to the worker "
    "that holds the prefix. Balance thresholds should trigger a spill once the "
    "active-request gap exceeds the configured abs/rel thresholds.\n"
)


def build_context(target_tokens):
    header = (
        "You are a senior SRE agent investigating a production incident on the "
        "GLM-5.2 inference cluster. Below is accumulated context from prior "
        "turns. Continue the investigation and answer the final question "
        "concisely.\n\n"
    )
    per_block = 270
    n = max(1, target_tokens // per_block)
    blocks = "\n".join(f"--- turn {i:04d} ---\n{BLOCK}" for i in range(n))
    tail = (
        "\n\nFinal question: based on the full transcript above, in one sentence, "
        "what is the single highest-impact action to restore load balance?"
    )
    return header + blocks + tail


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def one_request(prompt, max_tokens, req_id, results, lock):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        URL, data=payload, method="POST",
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    rec = {"req_id": req_id, "ok": False, "http": None, "tok": 0,
           "lat": 0.0, "ttft": 0.0, "err": None}
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            rec["http"] = resp.status
            ntok = 0
            first = None
            for raw in resp:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if first is None:
                    ch0 = evt.get("choices")
                    if ch0 and (ch0[0].get("delta", {}).get("content")
                                or ch0[0].get("delta", {}).get("reasoning_content")):
                        first = time.perf_counter()
                        rec["ttft"] = first - t0
                # Prefer authoritative usage from the final usage event.
                u = evt.get("usage")
                if u and u.get("completion_tokens") is not None:
                    ct = u.get("completion_tokens", 0)
                    rt = u.get("reasoning_tokens", 0) or 0
                    # Total decoded = content + reasoning (engine decodes both).
                    ntok = ct + rt
                else:
                    ch = evt.get("choices")
                    if ch and not ntok:
                        # Fallback: count non-empty delta chunks only if no usage.
                        delta = ch[0].get("delta", {})
                        if delta.get("content") or delta.get("reasoning_content"):
                            ntok += 1
            rec["tok"] = ntok
            rec["ok"] = (resp.status == 200)
            rec["lat"] = time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        rec["http"] = e.code
        rec["err"] = f"HTTPError {e.code}"
        rec["lat"] = time.perf_counter() - t0
    except Exception as e:
        rec["err"] = type(e).__name__ + ":" + str(e)[:120]
        rec["lat"] = time.perf_counter() - t0
    with lock:
        results.append(rec)


def run_wave(prompt, max_tokens, concurrency):
    results = []
    lock = threading.Lock()
    threads = []
    t0 = time.perf_counter()
    for i in range(concurrency):
        t = threading.Thread(target=one_request,
                             args=(prompt, max_tokens, i, results, lock))
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=320)
    wave_dur = time.perf_counter() - t0
    return results, wave_dur


def summarize(results):
    ok = [r for r in results if r.get("ok")]
    lats = sorted(r["lat"] for r in ok)
    ttfts = sorted(r["ttft"] for r in ok if r["ttft"] > 0)
    return {
        "ok": len(ok),
        "fail": len(results) - len(ok),
        "tok": sum(r.get("tok", 0) for r in ok),
        "mean_lat": statistics.mean(lats) if lats else 0.0,
        "p50_lat": pct(lats, 50),
        "p95_lat": pct(lats, 95),
        "mean_ttft": statistics.mean(ttfts) if ttfts else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=28)
    ap.add_argument("--ctx-tokens", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--duration", type=int, default=180,
                    help="total seconds to keep loading")
    ap.add_argument("--wave-gap", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=0,
                    help="if >0, run exactly this many sequential requests (for "
                         "hicache repeated-prefix test) instead of duration mode")
    args = ap.parse_args()

    prompt = build_context(args.ctx_tokens)
    print(f"stress: url={URL} concurrency={args.concurrency} "
          f"ctx~={args.ctx_tokens} max_tokens={args.max_tokens} "
          f"duration={args.duration}s repeats={args.repeats} "
          f"prompt_chars={len(prompt)}", flush=True)

    if args.repeats > 0:
        # Sequential repeated-prefix mode (same prompt N times in a row)
        all_ok = []
        for i in range(args.repeats):
            results, wd = run_wave(prompt, args.max_tokens, 1)
            s = summarize(results)
            all_ok.extend(r for r in results if r.get("ok"))
            print(f"[rep {i+1}/{args.repeats}] ok={s['ok']} fail={s['fail']} "
                  f"tok={s['tok']} lat={s['mean_lat']:.2f}s "
                  f"ttft={s['mean_ttft']:.3f}s dur={wd:.2f}s "
                  f"err={results[0].get('err') if not s['ok'] else ''}", flush=True)
        lats = sorted(r["lat"] for r in all_ok)
        print(f"DONE repeats total_ok={len(all_ok)} total_tok="
              f"{sum(r.get('tok',0) for r in all_ok)} "
              f"mean_lat={statistics.mean(lats) if lats else 0:.2f} "
              f"mean_ttft={statistics.mean([r['ttft'] for r in all_ok if r['ttft']>0]) if any(r['ttft']>0 for r in all_ok) else 0:.3f}",
              flush=True)
        return

    deadline = time.perf_counter() + args.duration
    wave = 0
    total_ok = 0
    total_fail = 0
    total_tok = 0
    all_lats = []
    all_ttfts = []
    t_start = time.perf_counter()
    while time.perf_counter() < deadline:
        wave += 1
        results, wave_dur = run_wave(prompt, args.max_tokens, args.concurrency)
        s = summarize(results)
        total_ok += s["ok"]
        total_fail += s["fail"]
        total_tok += s["tok"]
        all_lats.extend(r["lat"] for r in results if r.get("ok"))
        all_ttfts.extend(r["ttft"] for r in results if r.get("ok") and r["ttft"] > 0)
        errs = {}
        for r in results:
            if not r.get("ok") and r.get("err"):
                errs[r["err"]] = errs.get(r["err"], 0) + 1
        print(f"[wave {wave:03d} t={time.perf_counter()-t_start:.0f}s] "
              f"ok={s['ok']} fail={s['fail']} tok={s['tok']} "
              f"mean_lat={s['mean_lat']:.2f} p50={s['p50_lat']:.2f} "
              f"p95={s['p95_lat']:.2f} ttft={s['mean_ttft']:.3f} "
              f"dur={wave_dur:.1f}s cum_ok={total_ok} cum_fail={total_fail} "
              f"errs={errs}", flush=True)
        time.sleep(args.wave_gap)

    elapsed = time.perf_counter() - t_start
    all_lats.sort()
    all_ttfts.sort()
    agg_tp = total_tok / elapsed if elapsed > 0 else 0.0
    print(f"DONE elapsed={elapsed:.1f}s total_ok={total_ok} total_fail={total_fail} "
          f"total_tok={total_tok} agg_gen_throughput={agg_tp:.1f}tok/s "
          f"mean_lat={statistics.mean(all_lats) if all_lats else 0:.2f} "
          f"p50_lat={pct(all_lats,50):.2f} p95_lat={pct(all_lats,95):.2f} "
          f"mean_ttft={statistics.mean(all_ttfts) if all_ttfts else 0:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
