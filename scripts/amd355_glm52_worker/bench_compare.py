#!/usr/bin/env python3
import asyncio, json, time, sys, urllib.request, urllib.error

API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"
URL = "http://localhost:30000/v1/chat/completions"

def send_request(prompt, max_tokens):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(URL, data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}, method="POST")
    start = time.monotonic()
    resp = urllib.request.urlopen(req, timeout=600)
    body = json.loads(resp.read())
    elapsed = time.monotonic() - start
    usage = body.get("usage", {})
    return elapsed, usage.get("completion_tokens", 0), usage.get("prompt_tokens", 0)

def bench(conc, input_tokens, max_tokens, num_warmup=1):
    if input_tokens > 0:
        prompt = "The history of science is a fascinating subject. " * (input_tokens * 4 // 50 + 1)
        prompt = prompt[:input_tokens * 4]
    else:
        prompt = "Write a detailed analysis of quantum computing."

    # warmup
    for _ in range(num_warmup):
        send_request(prompt, max_tokens)

    async def run_concurrent():
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, send_request, prompt, max_tokens) for _ in range(conc)]
        start = time.monotonic()
        results = await asyncio.gather(*tasks)
        total = time.monotonic() - start
        return results, total

    results, total = asyncio.run(run_concurrent())
    total_tok = sum(r[1] for r in results)
    prompt_tok = sum(r[2] for r in results)
    output_tok_s = total_tok / total if total > 0 else 0
    total_tok_s = (total_tok + prompt_tok) / total if total > 0 else 0
    req_s = len(results) / total if total > 0 else 0
    avg_lat = sum(r[0] for r in results) / len(results) * 1000
    return {
        "conc": conc, "input": input_tokens, "output": max_tokens,
        "output_tok_s": round(output_tok_s, 1),
        "total_tok_s": round(total_tok_s, 1),
        "req_s": round(req_s, 2),
        "avg_lat_ms": round(avg_lat, 0),
        "elapsed": round(total, 2),
    }

if __name__ == "__main__":
    tests = [
        # (conc, input_tokens, max_tokens, label)
        (1, 0, 1024, "decode_short"),
        (8, 0, 1024, "decode_short_c8"),
        (1, 2048, 1024, "decode_2k"),
        (8, 2048, 1024, "decode_2k_c8"),
        (1, 0, 256, "qa_thinking"),
        (4, 4096, 256, "medium_ctx_c4"),
    ]
    print(f"{'test':<20} {'conc':>4} {'in':>5} {'out':>4} {'tok/s':>8} {'req/s':>6} {'lat_ms':>8}")
    print("-" * 60)
    for conc, inp, out, label in tests:
        r = bench(conc, inp, out)
        print(f"{label:<20} {r['conc']:>4} {r['input']:>5} {r['output']:>4} {r['output_tok_s']:>8.1f} {r['req_s']:>6.2f} {r['avg_lat_ms']:>8.0f}")
        sys.stdout.flush()
