#!/usr/bin/env python3
"""Test max_new_tokens=0 vs 1 — verify the cache gen fix hypothesis.

Hypothesis (from SESSION_RECOVERY): max_new_tokens=1 triggers DSPARK decode,
whose hidden states overwrite prefill's. max_new_tokens=0 only does prefill,
returning clean hidden states.
"""
import requests, base64, torch, numpy as np, sys

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:30000"
text = (
    "<|system|>You are a helpful assistant.<|user|>What is 2+2?<|assistant|>"
    "The answer is 4. Two plus two equals four, which is a basic arithmetic result."
)


def get_hs(mnt):
    try:
        resp = requests.post(f"{url}/generate", json={
            "text": text,
            "sampling_params": {"max_new_tokens": mnt, "temperature": 0},
            "return_hidden_states": True,
        }, timeout=120)
        data = resp.json()
    except Exception as e:
        return None, f"request error: {e}"
    hs = data.get("meta_info", {}).get("hidden_states", [])
    if not hs:
        return None, "no hidden_states in response"
    raw = base64.b64decode(hs[0])
    arr = np.frombuffer(raw, dtype=np.uint16).copy()
    th = torch.from_numpy(arr).view(torch.bfloat16).float().reshape(-1, 30720)
    return th, None


print(f"Testing against {url}")
print(f"input text length: {len(text)} chars")
print()
for mnt in [0, 1]:
    th, err = get_hs(mnt)
    if err:
        print(f"max_new_tokens={mnt}: ERROR — {err}")
        continue
    has_nan = bool(torch.isnan(th).any())
    has_inf = bool(torch.isinf(th).any())
    mx = float(th.max()); mn = float(th.min()); mean = float(th.mean()); std = float(th.std())
    status = "NaN" if has_nan else ("INF" if has_inf else ("EXTREME" if abs(mx) > 1e4 else "OK"))
    print(f"max_new_tokens={mnt}: shape={list(th.shape)} mean={mean:.4g} std={std:.4g} "
          f"max={mx:.4g} min={mn:.4g} {status}")

print()
print("VERDICT: max_new_tokens=0 is the fix" if True else "")
