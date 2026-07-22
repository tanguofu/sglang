import subprocess
import json
import time
import concurrent.futures

URL = "https://glm52-pd-1p1d.jmpti.woa.com/v1/chat/completions"
HEADERS = ["-H", "Authorization: Bearer sk-46faecc9d0bc4dcd9db6a15c73ae91c8", "-H", "Content-Type: application/json"]
DATA = '{"model":"glm-5.2","messages":[{"role":"user","content":"Tell me a story about a brave knight on a quest"}],"max_tokens":256}'

def run_request(i):
    start = time.time()
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", URL] + HEADERS + ["-d", DATA, "--max-time", "120", "-o", f"/tmp/bench_req_{i}.json", "-w", "%{http_code}"],
        capture_output=True, text=True
    )
    elapsed = time.time() - start
    http_code = result.stdout.strip() if result.stdout else "000"
    tokens = 0
    try:
        with open(f"/tmp/bench_req_{i}.json") as f:
            d = json.load(f)
        tokens = d.get('usage', {}).get('completion_tokens', 0)
    except:
        pass
    return (i, elapsed, http_code, tokens)

for conc in [1, 2, 4, 8]:
    print(f"\n=== Concurrency={conc} (256 tokens each) ===")
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as executor:
        futures = [executor.submit(run_request, i) for i in range(conc)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    wall = time.time() - start
    
    results.sort(key=lambda x: x[0])
    total_tokens = 0
    ok = 0
    for i, elapsed, code, tokens in results:
        print(f"  req{i}: {elapsed:.2f}s http={code} tokens={tokens}")
        if code == "200":
            total_tokens += tokens
            ok += 1
    
    throughput = total_tokens / wall if wall > 0 else 0
    avg_latency = sum(r[1] for r in results) / len(results)
    print(f"  Summary: wall={wall:.2f}s ok={ok}/{conc} total_tokens={total_tokens} avg_latency={avg_latency:.2f}s aggregate_throughput={throughput:.1f} t/s")
