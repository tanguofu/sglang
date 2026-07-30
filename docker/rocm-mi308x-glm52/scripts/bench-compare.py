#!/usr/bin/env python3
"""
Self-contained benchmark comparing 1p1d PD vs 2tp8 throughput.
Uses async HTTP requests — no sglang internals needed.
"""
import asyncio
import json
import time
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx not found, installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

ROUTER_1P1D = "http://sglang-1p1d-router.kube-system:30001"
ROUTER_2TP8 = "http://sglang-glm52-2tp8-router.kube-system:30080"
MODEL_1P1D = "glm-5.2"
MODEL_2TP8 = "unknown"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"

OUTDIR = Path("/tmp/bench-compare")
OUTDIR.mkdir(exist_ok=True)

WORKLOADS = [
    # (name, input_len, output_len, num_prompts, concurrency)
    ("short_c8",    32,   256,  32, 8),
    ("short_c16",   32,   256,  32, 16),
    ("medium_c8",   128,  256,  32, 8),
    ("long_c8",     2048, 256,  16, 8),
    ("decode_c8",   32,   512,  32, 8),
]

# Prompt template — repeated to hit target input length
PROMPT_TEMPLATE = "Here is a question: {text}\nPlease answer concisely."

# ~4 chars per token, so to get ~N input tokens we need ~4N chars of text
FILLER = "The quick brown fox jumps over the lazy dog. "  # ~45 chars ~ 11 tokens


def build_prompt(input_len):
    """Build a prompt of approximately input_len tokens."""
    target_chars = input_len * 4
    reps = max(1, target_chars // len(FILLER))
    text = (FILLER * reps)[:target_chars]
    return PROMPT_TEMPLATE.format(text=text)


async def send_request(client, url, model, prompt, output_len, idx):
    """Send a single chat completion request and measure metrics."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len,
        "temperature": 0.1,
        "stream": False,
    }
    t_start = time.perf_counter()
    try:
        resp = await client.post(f"{url}/v1/chat/completions",
                                  json=payload, headers=headers, timeout=300)
        t_end = time.perf_counter()
        if resp.status_code != 200:
            return {"idx": idx, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        data = resp.json()
        completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        e2e_ms = (t_end - t_start) * 1000
        # TTFT approximation: for non-streaming, we can't measure true TTFT
        # Use E2E / output_tokens as ITL proxy
        itl_ms = e2e_ms / max(completion_tokens, 1)
        return {
            "idx": idx,
            "e2e_ms": e2e_ms,
            "ttft_ms": e2e_ms,  # non-streaming, so E2E ≈ TTFT
            "itl_ms": itl_ms,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": completion_tokens + prompt_tokens,
        }
    except Exception as e:
        return {"idx": idx, "error": str(e)}


async def run_workload(url, model, name, input_len, output_len, num_prompts, concurrency):
    """Run a single workload with given concurrency."""
    prompt = build_prompt(input_len)
    print(f"  {name}: in={input_len}, out={output_len}, "
          f"prompts={num_prompts}, conc={concurrency}")

    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(300.0)) as client:
        t0 = time.perf_counter()
        sem = asyncio.Semaphore(concurrency)

        async def bounded_send(idx):
            async with sem:
                return await send_request(client, url, model, prompt, output_len, idx)

        tasks = [bounded_send(i) for i in range(num_prompts)]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

    # Aggregate metrics
    ok = [r for r in results if "error" not in r]
    err = [r for r in results if "error" in r]
    if not ok:
        print(f"    -> ALL FAILED ({len(err)} errors)")
        return {"name": name, "error": f"{len(err)} failures", "errors": [e["error"] for e in err[:3]]}

    total_completion = sum(r["completion_tokens"] for r in ok)
    total_tokens = sum(r["total_tokens"] for r in ok)
    e2e_list = [r["e2e_ms"] for r in ok]
    itl_list = [r["itl_ms"] for r in ok]

    metrics = {
        "name": name,
        "input": input_len,
        "output": output_len,
        "num_prompts": num_prompts,
        "concurrency": concurrency,
        "elapsed_s": elapsed,
        "successful": len(ok),
        "failed": len(err),
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "output_throughput": total_completion / elapsed,  # tok/s
        "total_throughput": total_tokens / elapsed,  # tok/s
        "mean_e2e_ms": sum(e2e_list) / len(e2e_list),
        "p50_e2e_ms": sorted(e2e_list)[len(e2e_list) // 2],
        "p99_e2e_ms": sorted(e2e_list)[int(len(e2e_list) * 0.99)] if len(e2e_list) > 1 else e2e_list[-1],
        "mean_itl_ms": sum(itl_list) / len(itl_list),
        "mean_ttft_ms": sum(r["ttft_ms"] for r in ok) / len(ok),
    }
    print(f"    -> out={metrics['output_throughput']:.1f} tok/s, "
          f"total={metrics['total_throughput']:.1f} tok/s, "
          f"e2e={metrics['mean_e2e_ms']:.0f}ms, "
          f"ok={len(ok)}/{len(results)}")
    return metrics


async def main():
    results = {"1p1d": [], "2tp8": []}

    for label, url, model in [("1p1d", ROUTER_1P1D, MODEL_1P1D),
                               ("2tp8", ROUTER_2TP8, MODEL_2TP8)]:
        print(f"\n{'=' * 60}")
        print(f"  Benchmarking {label} ({url})")
        print(f"{'=' * 60}")
        for name, in_len, out_len, num, conc in WORKLOADS:
            full_name = f"{label}_{name}"
            try:
                m = await run_workload(url, model, full_name, in_len, out_len, num, conc)
                m["deployment"] = label
                results[label].append(m)
            except Exception as e:
                print(f"    -> ERROR: {e}")
                results[label].append({"name": full_name, "error": str(e)})
            # Cool-down between workloads
            await asyncio.sleep(3)

    # Summary comparison
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON SUMMARY (output throughput, tok/s)")
    print(f"{'=' * 70}")
    print(f"{'Workload':<16} {'1p1d':>10} {'2tp8':>10} {'Diff':>10} {'Winner':>8}")
    print("-" * 70)
    for i, w in enumerate(WORKLOADS):
        wname = w[0]
        m1 = results["1p1d"][i] if i < len(results["1p1d"]) else {}
        m2 = results["2tp8"][i] if i < len(results["2tp8"]) else {}
        t1 = m1.get("output_throughput", 0) or 0
        t2 = m2.get("output_throughput", 0) or 0
        diff = t1 - t2
        winner = "1p1d" if diff > 0.5 else "2tp8" if diff < -0.5 else "tie"
        print(f"{wname:<16} {t1:>10.2f} {t2:>10.2f} {diff:>+10.2f} {winner:>8}")

    print(f"\n{'Workload':<16} {'1p1d tot':>10} {'2tp8 tot':>10} {'Diff':>10}")
    print("-" * 50)
    for i, w in enumerate(WORKLOADS):
        wname = w[0]
        m1 = results["1p1d"][i] if i < len(results["1p1d"]) else {}
        m2 = results["2tp8"][i] if i < len(results["2tp8"]) else {}
        t1 = m1.get("total_throughput", 0) or 0
        t2 = m2.get("total_throughput", 0) or 0
        print(f"{wname:<16} {t1:>10.2f} {t2:>10.2f} {t1-t2:>+10.2f}")

    print(f"\n{'Workload':<16} {'1p1d E2E':>10} {'2tp8 E2E':>10} {'Diff':>10}")
    print("-" * 50)
    for i, w in enumerate(WORKLOADS):
        wname = w[0]
        m1 = results["1p1d"][i] if i < len(results["1p1d"]) else {}
        m2 = results["2tp8"][i] if i < len(results["2tp8"]) else {}
        t1 = m1.get("mean_e2e_ms", 0) or 0
        t2 = m2.get("mean_e2e_ms", 0) or 0
        print(f"{wname:<16} {t1:>8.0f}ms {t2:>8.0f}ms {t1-t2:>+8.0f}ms")

    (OUTDIR / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results: {OUTDIR}/summary.json")


if __name__ == "__main__":
    asyncio.run(main())
