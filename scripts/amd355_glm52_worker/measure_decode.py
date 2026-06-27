import requests, time, json

URL = "http://localhost:30000/generate"
payload = {
    "text": "Write a long essay about AI. " * 200,
    "sampling_params": {"max_new_tokens": 64, "temperature": 0.0, "ignore_eos": True}
}
# Warmup
requests.post(URL, json=payload, timeout=60)

# Timed run
t0 = time.perf_counter()
r = requests.post(URL, json=payload, timeout=60)
t1 = time.perf_counter()
data = r.json()
meta = data.get("meta_info", {})
ct = meta.get("completion_tokens", 1)
pt = meta.get("prompt_tokens", 0)
e2e = meta.get("e2e_latency", 0)
ttft = meta.get("ttft", 0)
print(f"Total time: {(t1-t0)*1000:.1f}ms")
print(f"Output tokens: {ct}")
print(f"Input tokens: {pt}")
print(f"TPOT: {e2e / max(ct, 1) * 1000:.1f}ms")
print(f"TTFT: {ttft * 1000:.1f}ms")
print(f"Gen throughput: {ct / (t1-t0):.1f} tok/s")
