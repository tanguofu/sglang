#!/usr/bin/env python3
"""Test hidden states quality with REAL training data samples.

Distinguish: is the corruption from (a) long sequences, (b) DSPARK decode
overwriting prefill, or (c) concurrency (64 workers)?
"""
import requests, json, torch, numpy as np, sys, base64
from transformers import AutoTokenizer

url = "http://localhost:30000"
model_path = "/data/models/GLM-5.2-FP8"
train_data = "/data/dspark_v9_all_coding.jsonl"
max_length = 1000

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

samples = []
with open(train_data) as f:
    for line in f:
        r = json.loads(line)
        convs = r.get("conversations", [])
        messages = []
        for c in convs:
            role = c.get("from", c.get("role", "user"))
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            content = c.get("value", c.get("content", ""))
            messages.append({"role": role, "content": content})
        samples.append(messages)
        if len(samples) >= 3:
            break


def parse_hs(hs):
    if not hs:
        return None
    if isinstance(hs[0], list):
        return torch.tensor(hs[0], dtype=torch.float32)
    raw = base64.b64decode(hs[0])
    arr = np.frombuffer(raw, dtype=np.uint16).copy()
    return torch.from_numpy(arr).view(torch.bfloat16).float().reshape(-1, 30720)


for i, messages in enumerate(samples[:3]):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids[0]
    seq_len = len(input_ids)
    print(f"\n=== Sample {i}: seq_len={seq_len} ===")
    try:
        resp = requests.post(f"{url}/generate", json={
            "text": text[:max_length * 4],
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_hidden_states": True,
        }, timeout=600)
        data = resp.json()
    except Exception as e:
        print(f"  request error: {e}")
        continue
    hs = data.get("meta_info", {}).get("hidden_states", [])
    th = parse_hs(hs)
    if th is None:
        print(f"  no hidden_states! keys={list(data.get('meta_info',{}).keys())}")
        continue
    has_nan = bool(torch.isnan(th).any())
    has_inf = bool(torch.isinf(th).any())
    mx = float(th.max()); mn = float(th.min()); mean = float(th.mean())
    status = "NaN" if has_nan else ("INF" if has_inf else ("EXTREME" if abs(mx) > 1e4 else "OK"))
    print(f"  hidden shape={list(th.shape)} mean={mean:.4g} max={mx:.4g} min={mn:.4g} {status}")
    # Also check per-token: is it the first tokens (prefill) or last (decode) that are bad?
    if not has_nan and not has_inf and abs(mx) < 1e4:
        per_token_max = th.view(th.shape[0], -1).abs().max(dim=1)[0]
        print(f"  per-token |max|: first3={per_token_max[:3].tolist()} last3={per_token_max[-3:].tolist()}")
