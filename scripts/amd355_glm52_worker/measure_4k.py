import requests, time
URL = "http://localhost:30000/generate"
text = "The quick brown fox. " * 512
payload = {"text": text, "sampling_params": {"max_new_tokens": 64, "temperature": 0.0, "ignore_eos": True}}
requests.post(URL, json=payload, timeout=60)
t0 = time.perf_counter()
r = requests.post(URL, json=payload, timeout=60)
t1 = time.perf_counter()
meta = r.json().get("meta_info", {})
ct = meta.get("completion_tokens", 1)
pt = meta.get("prompt_tokens", 0)
print(f"Input: {pt} tok, Output: {ct} tok")
print(f"TPOT: {meta.get('e2e_latency',0)/max(ct,1)*1000:.1f}ms")
print(f"TTFT: {meta.get('ttft',0)*1000:.1f}ms")
print(f"Throughput: {ct/(t1-t0):.1f} tok/s")
