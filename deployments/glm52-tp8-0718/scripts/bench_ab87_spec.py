#!/usr/bin/env python3
"""
EAGLE speculative-decoding A/B sweep load generator for the ab87 worker.

Sends a fixed, deterministic decode-heavy load against a single sglang worker
(http://<node>:30000) so every EAGLE config is measured identically. Load is
issued via `kubectl exec` into the worker pod itself (curl against localhost),
matching the existing bench pattern and avoiding external-network routing
questions.

Per config we record:
  - spec_accept_rate, spec_accept_length  (from /metrics, before & after)
  - aggregate gen_throughput (tok/s)        (client-side: total out tokens / wall)
  - server-side gen tok/s                  (generation_tokens_total delta / wall)
  - mean wall time per request, n_ok / n_fail

Usage:
  python3 bench_ab87_spec.py --pod ab87-sglang-0 --duration 180 \
      --concurrency 8 --ctx-tokens 12000 --max-tokens 256 \
      --out results/ab87-C0.json --tag C0
"""
import argparse
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

KUBE_NS = "kube-system"
MODEL = "glm-5.2"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"

# Deterministic long agent context (~target_tokens of prefill). Repeated varied
# blocks; identical across configs so prefill cost is held constant.
CONTEXT_BLOCK = (
    "user: I need you to analyze the following system metrics and propose a "
    "remediation plan. Here is the transcript from the last investigation turn.\n"
    "assistant: Understood. I will review the metrics, identify the bottleneck, "
    "and propose a step-by-step remediation. Let me first gather the relevant "
    "telemetry from the monitoring stack.\n"
    "tool_call: query_prometheus(metric='sglang:num_requests_running', range='5m')\n"
    "tool_result: peak=18, p95=15, mean=11.2 across both workers; worker-1 carries "
    "0.88 of the load under cache_aware routing.\n"
    "tool_call: query_prometheus(metric='sglang:cache_hit_rate', range='5m')\n"
    "tool_result: worker-1=0.62, worker-2=0.04; prefix cache is concentrated on "
    "worker-1 because the shared system prompt and conversation prefix only "
    "resides there.\n"
    "assistant: The imbalance is caused by cache_aware sticking to the worker that "
    "holds the prefix. The balance thresholds should trigger a spill once the "
    "active-request gap exceeds the configured abs/rel thresholds. I will verify "
    "the threshold settings and the resulting selection distribution next.\n"
)


def build_long_context(target_tokens: int) -> str:
    header = (
        "You are a senior SRE agent. You have been investigating a production "
        "incident on the GLM-5.2 inference cluster. Below is the accumulated "
        "context from prior turns. Continue the investigation and answer the "
        "final question concisely.\n\n"
    )
    per_block_tokens = 270
    n_blocks = max(1, target_tokens // per_block_tokens)
    blocks = [f"--- turn {i:04d} ---\n" + CONTEXT_BLOCK for i in range(n_blocks)]
    tail = (
        "\n\nFinal question: based on the full transcript above, in one sentence, "
        "what is the single highest-impact action to restore load balance?"
    )
    return header + "\n".join(blocks) + tail


def kubectl_exec(pod: str, cmd: str, stdin_data: str | None = None,
                 timeout: int = 320) -> str:
    """Run a command inside the worker pod via kubectl exec -i."""
    full = ["kubectl", "exec", "-n", KUBE_NS, pod, "-i", "--", "sh", "-c", cmd]
    proc = subprocess.run(full, input=stdin_data, capture_output=True,
                          text=True, timeout=timeout)
    return proc.stdout


def get_metrics(pod: str) -> dict:
    """Snapshot spec + throughput counters from /metrics."""
    text = kubectl_exec(pod, "curl -s --max-time 10 http://localhost:30000/metrics",
                        timeout=20)
    out = {}
    # Spec metrics: capture every line mentioning spec/accept/eagle for discovery.
    spec_lines = {}
    for line in text.splitlines():
        if ("spec" in line or "accept" in line or "eagle" in line) and " " in line:
            m = re.match(r'(sglang:[\w:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)', line)
            if m:
                name = m.group(1)
                # keep last value per bare name (gauge form)
                spec_lines[name] = float(m.group(3))
    out["spec"] = spec_lines
    # throughput counters
    for key in ("generation_tokens_total", "prompt_tokens_total",
                "num_requests_total", "gen_throughput"):
        m = re.search(rf'sglang:{key}\{{[^}}]*\}}\s+([0-9.eE+]+)', text)
        if not m:
            m = re.search(rf'sglang:{key}\s+([0-9.eE+]+)', text)
        if m:
            out[key] = float(m.group(1))
    return out


def stream_one(pod: str, prompt: str, max_tokens: int, req_id: int) -> dict:
    """One streaming chat/completions request via kubectl exec + curl."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "ignore_eos": False,
    })
    cmd = (
        "curl -sS -N --max-time 300 "
        f"-H 'Authorization: Bearer {API_KEY}' "
        "-H 'Content-Type: application/json' "
        "-X POST http://localhost:30000/v1/chat/completions "
        "-d @- -w '\\n__CURL_META__\\nhttp_code=%{http_code}\\n"
        "time_total=%{time_total}\\n"
        "time_starttransfer=%{time_starttransfer}\\n"
        "size_download=%{size_download}\\n'"
    )
    t0 = time.perf_counter()
    raw = kubectl_exec(pod, cmd, stdin_data=payload, timeout=320)
    wall = time.perf_counter() - t0

    meta = {}
    n_output = 0
    first_token_t = None
    # parse SSE
    meta_section = False
    for line in raw.splitlines():
        if line.startswith("__CURL_META__"):
            meta_section = True
            continue
        if meta_section:
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k] = v
            continue
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            continue
        try:
            evt = json.loads(data)
        except json.JSONDecodeError:
            continue
        # usage chunk (final)
        u = evt.get("usage")
        if u and u.get("completion_tokens") is not None:
            n_output = max(n_output, int(u["completion_tokens"]))
        ch = evt.get("choices") or []
        if ch:
            delta = ch[0].get("delta", {})
            if delta.get("content"):
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                # fallback token count if usage absent
                if not n_output:
                    n_output += 1

    http_code = meta.get("http_code", "?")
    ttft = (first_token_t - t0) if first_token_t else None
    if ttft and n_output > 0:
        decode_secs = max(1e-6, float(meta.get("time_total", wall)) - ttft)
        tput = n_output / decode_secs
    else:
        tput = 0.0
    return {
        "req_id": req_id,
        "http_code": http_code,
        "ttft": ttft,
        "wall": float(meta.get("time_total", wall)),
        "n_output": n_output,
        "throughput": tput,
        "ok": http_code == "200",
    }


def run(pod: str, concurrency: int, duration_s: int, ctx_tokens: int,
        max_tokens: int, tag: str) -> dict:
    prompt = build_long_context(ctx_tokens)
    print(f"\n=== [{tag}] concurrency={concurrency} duration={duration_s}s "
          f"ctx~={ctx_tokens}tok max_tokens={max_tokens} ===", flush=True)
    print(f"  prompt chars={len(prompt)} (~{len(prompt)//4} tokens)", flush=True)

    m_before = get_metrics(pod)
    t_start = time.perf_counter()
    deadline = t_start + duration_s

    results = []
    lock = threading.Lock()
    req_counter = [0]

    def slot():
        local = []
        while time.perf_counter() < deadline:
            with lock:
                req_counter[0] += 1
                rid = req_counter[0]
            r = stream_one(pod, prompt, max_tokens, rid)
            local.append(r)
            if not r["ok"]:
                print(f"  req {rid}: FAIL http={r['http_code']}", flush=True)
        return local

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(slot) for _ in range(concurrency)]
        for fu in futs:
            results.extend(fu.result())

    wall = time.perf_counter() - t_start
    m_after = get_metrics(pod)

    oks = [r for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]
    walls = [r["wall"] for r in oks]
    ttfts = [r["ttft"] for r in oks if r["ttft"]]
    tputs = [r["throughput"] for r in oks if r["throughput"] > 0]
    total_out = sum(r["n_output"] for r in oks)

    # server-side gen tok/s from counter delta
    gen_delta = (m_after.get("generation_tokens_total", 0) -
                 m_before.get("generation_tokens_total", 0))
    server_gen_tps = gen_delta / wall if wall else 0.0

    summary = {
        "tag": tag,
        "pod": pod,
        "concurrency": concurrency,
        "duration_s": duration_s,
        "ctx_tokens_target": ctx_tokens,
        "max_tokens": max_tokens,
        "wall_clock_s": round(wall, 2),
        "n_ok": len(oks),
        "n_fail": len(fails),
        "total_output_tokens": total_out,
        "aggregate_gen_throughput_tps": round(total_out / wall, 2) if wall else 0,
        "server_gen_throughput_tps": round(server_gen_tps, 2),
        "mean_wall_s": round(sum(walls) / len(walls), 3) if walls else None,
        "mean_ttft_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
        "mean_per_req_decode_tps": round(sum(tputs) / len(tputs), 2) if tputs else None,
        "spec_metrics_before": m_before.get("spec", {}),
        "spec_metrics_after": m_after.get("spec", {}),
        "failures": [{"req_id": r["req_id"], "http_code": r["http_code"]}
                     for r in fails],
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", default="ab87-sglang-0")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--duration", type=int, default=180)
    ap.add_argument("--ctx-tokens", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tag", default="C0")
    ap.add_argument("--out", default="results/ab87-spec.json")
    args = ap.parse_args()

    s = run(args.pod, args.concurrency, args.duration, args.ctx_tokens,
            args.max_tokens, args.tag)
    print(json.dumps(s, indent=2, ensure_ascii=False))
    with open(args.out, "w") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    print(f"\n=== written to {args.out} ===", flush=True)


if __name__ == "__main__":
    main()
