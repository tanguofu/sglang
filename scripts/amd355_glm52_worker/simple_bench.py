#!/usr/bin/env python3
import asyncio, aiohttp, json, time, sys, argparse, numpy as np
async def send_request(session, url, prompt, max_tokens, input_len, enable_thinking=True):
    payload = {"model": "/data/models/GLM-5.2-FP8", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.7, "stream": True}
    if not enable_thinking: payload["chat_template_kwargs"] = {"enable_thinking": False}
    ttft = None; tokens = 0; start = time.perf_counter()
    try:
        async with session.post(f"{url}/v1/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=600)) as resp:
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    if data.get("choices"):
                        delta = data["choices"][0].get("delta", {})
                        if delta.get("content") or delta.get("reasoning_content"):
                            if ttft is None: ttft = time.perf_counter() - start
                            tokens += 1
    except Exception as e: return {"error": str(e), "tokens": tokens, "latency": time.perf_counter() - start}
    latency = time.perf_counter() - start
    return {"ttft": ttft, "tokens": tokens, "latency": latency, "tpot": (latency - (ttft or 0)) / max(tokens - 1, 1) * 1000 if tokens > 1 else 0}
async def run_bench(url, concurrency, max_tokens, input_len, num_requests, enable_thinking=True):
    if input_len > 0: prompt = ("The quick brown fox jumps over the lazy dog. " * (input_len // 10 + 1))[:input_len * 4]
    else: prompt = "Write a short poem about the sea."
    sem = asyncio.Semaphore(concurrency)
    async def bounded(session):
        async with sem: return await send_request(session, url, prompt, max_tokens, input_len, enable_thinking)
    async with aiohttp.ClientSession() as session:
        start = time.perf_counter()
        results = await asyncio.gather(*[bounded(session) for _ in range(num_requests)])
        elapsed = time.perf_counter() - start
    ok = [r for r in results if "error" not in r and r["tokens"] > 0]
    total_tokens = sum(r["tokens"] for r in ok)
    ttfts = [r["ttft"] for r in ok if r["ttft"]]; tpots = [r["tpot"] for r in ok if r["tpot"] > 0]; lats = [r["latency"] for r in ok]
    thinking = "thinking" if enable_thinking else "no-thinking"
    print(f"\n{'='*60}\nc={concurrency}, in={input_len}, out={max_tokens}, reqs={num_requests}, {thinking}\n{'='*60}")
    print(f"OK: {len(ok)}/{num_requests}, Elapsed: {elapsed:.2f}s, Output: {total_tokens/elapsed:.1f} tok/s ({total_tokens} tok)")
    if input_len > 0: print(f"Input: {input_len*len(ok)/elapsed:.1f} tok/s, Req: {len(ok)/elapsed:.2f} req/s")
    else: print(f"Req: {len(ok)/elapsed:.2f} req/s")
    if ttfts: print(f"TTFT: mean={np.mean(ttfts)*1000:.0f}ms, p99={np.percentile(ttfts,99)*1000:.0f}ms")
    if tpots: print(f"TPOT: mean={np.mean(tpots):.1f}ms, p50={np.percentile(tpots,50):.1f}ms, p99={np.percentile(tpots,99):.1f}ms")
    if lats: print(f"Lat: mean={np.mean(lats)*1000:.0f}ms, p99={np.percentile(lats,99)*1000:.0f}ms")
    print()
async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:30000"); p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=128); p.add_argument("--input-len", type=int, default=0)
    p.add_argument("--num-requests", type=int, default=None); p.add_argument("--no-thinking", action="store_true")
    a = p.parse_args(); await run_bench(a.url, a.concurrency, a.max_tokens, a.input_len, a.num_requests or a.concurrency, not a.no_thinking)
if __name__ == "__main__": asyncio.run(main())
