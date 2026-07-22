#!/usr/bin/env python3
"""PD performance benchmark: measures TTFT, TPOT, throughput at various concurrency levels."""
import argparse
import asyncio
import json
import time
import statistics
import httpx

async def send_request(client, url, api_key, model, prompt, max_tokens, idx):
    """Send a single request and return timing stats."""
    start = time.perf_counter()
    first_token_time = None
    tokens_received = 0

    try:
        async with client.stream("POST", url, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        }, timeout=300) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk.get("choices"):
                        delta = chunk["choices"][0].get("delta", {})
                        # GLM-5.2 streams reasoning_content first, then content
                        text = delta.get("content") or delta.get("reasoning_content")
                        if text:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            tokens_received += 1
                except json.JSONDecodeError:
                    pass

    except Exception as e:
        end = time.perf_counter()
        return {"idx": idx, "error": str(e), "total_ms": (end - start) * 1000}

    end = time.perf_counter()
    total_ms = (end - start) * 1000
    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
    tpot_ms = (end - first_token_time) * 1000 / max(1, tokens_received - 1) if first_token_time and tokens_received > 1 else 0

    return {
        "idx": idx,
        "total_ms": total_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "tokens": tokens_received,
        "total_tokens": tokens_received,
    }

async def run_benchmark(url, api_key, model, prompt, max_tokens, concurrency, num_requests):
    """Run benchmark at given concurrency."""
    print(f"\n--- Concurrency={concurrency}, Requests={num_requests}, max_tokens={max_tokens} ---")

    sem = asyncio.Semaphore(concurrency)
    async def bounded_send(client, idx):
        async with sem:
            return await send_request(client, url, api_key, model, prompt, max_tokens, idx)

    async with httpx.AsyncClient() as client:
        tasks = [bounded_send(client, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)

    errors = [r for r in results if "error" in r]
    success = [r for r in results if "error" not in r]

    if success:
        ttfts = [r["ttft_ms"] for r in success if r["ttft_ms"] > 0]
        tpots = [r["tpot_ms"] for r in success if r["tpot_ms"] > 0]
        totals = [r["total_ms"] for r in success]
        all_tokens = sum(r["tokens"] for r in success)

        total_wall = max(totals) if totals else 0
        throughput = all_tokens / (total_wall / 1000) if total_wall > 0 else 0

        print(f"  Success: {len(success)}/{num_requests}, Errors: {len(errors)}")
        if ttfts:
            print(f"  TTFT  (ms): p50={statistics.median(ttfts):.0f} avg={statistics.mean(ttfts):.0f} p99={max(ttfts):.0f}")
        if tpots:
            print(f"  TPOT  (ms): p50={statistics.median(tpots):.0f} avg={statistics.mean(tpots):.0f} p99={max(tpots):.0f}")
        print(f"  Total (ms): p50={statistics.median(totals):.0f} avg={statistics.mean(totals):.0f} max={max(totals):.0f}")
        print(f"  Tokens: {all_tokens}, Wall time: {total_wall:.0f}ms, Throughput: {throughput:.1f} tok/s")

    if errors:
        for e in errors[:3]:
            print(f"  ERROR req {e['idx']}: {e['error'][:100]}")

    return success, errors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://21.234.170.159:30001")
    parser.add_argument("--api-key", default="sk-46faecc9d0bc4dcd9db6a15c73ae91c8")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--prompt", default="Explain what machine learning is in 3 sentences.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--num-requests", type=int, default=None)
    args = parser.parse_args()

    url = f"{args.url}/v1/chat/completions"
    print(f"=== GLM-5.2 PD Benchmark ===")
    print(f"URL: {url}")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")

    for conc in args.concurrency:
        num_req = args.num_requests or conc
        asyncio.run(run_benchmark(url, args.api_key, args.model, args.prompt, args.max_tokens, conc, num_req))

if __name__ == "__main__":
    main()
