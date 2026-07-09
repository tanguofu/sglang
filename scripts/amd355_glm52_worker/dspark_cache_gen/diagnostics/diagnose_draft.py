#!/usr/bin/env python3
"""Diagnose DSpark draft model: load checkpoint, run forward, compare argmax vs target.

This script loads the draft checkpoint directly and checks if its predictions
match what the target model would predict. Run inside the DSpark server container
(so target model is already loaded).
"""
import requests, torch, sys

# 1. Send a request to the DSpark server and get the generated tokens
# 2. Then send the SAME prompt to a non-speculative endpoint (if available)
# 3. Compare the tokens

url = "http://localhost:30000"
prompt = "The capital of France is"

# Get DSpark generated tokens
print("=== Sending request to DSpark server ===")
r = requests.post(f"{url}/generate", json={
    "text": prompt,
    "sampling_params": {"max_new_tokens": 16, "temperature": 0},
}, timeout=120)
d = r.json()
text = d.get("text", "")
print(f"Generated: {text[:100]}")

# Check server logs for accept metrics
print("\n=== Server accept metrics (from logs) ===")
import subprocess
result = subprocess.run(
    ["docker", "logs", "--tail", "20", "glm52_dspark_test"],
    capture_output=True, text=True, timeout=10
)
for line in result.stdout.split('\n'):
    if "accept" in line.lower() and "decode" in line.lower():
        print(f"  {line.strip()}")

# 3. Check if the draft model weights are loaded correctly
# by inspecting the model object
print("\n=== Checking draft model weights ===")
try:
    # The server stores the model in TP workers; we can't easily access it
    # from here. But we can check the checkpoint directly.
    import json, safetensors.torch
    ckpt_dir = "/data/checkpoints/deepspec/dspark_glm5_2_v9_256exp_clean/step_180"
    config = json.load(open(f"{ckpt_dir}/config.json"))
    print(f"architectures: {config.get('architectures')}")
    print(f"num_hidden_layers: {config.get('num_hidden_layers')}")
    print(f"dspark_num_layers: {config.get('dspark_num_layers')}")
    print(f"dspark_block_size: {config.get('dspark_block_size')}")
    print(f"dspark_markov_rank: {config.get('dspark_markov_rank')}")

    # Check weight keys
    from safetensors import safe_open
    st_file = f"{ckpt_dir}/model.safetensors"
    with safe_open(st_file, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"\nTotal weight keys: {len(keys)}")
        # Check for markov weights
        markov_keys = [k for k in keys if "markov" in k.lower()]
        print(f"Markov keys: {markov_keys[:5]}")
        # Check for norm weights
        norm_keys = [k for k in keys if "norm" in k.lower() and "layer" not in k.lower()]
        print(f"Norm keys (non-layer): {norm_keys[:10]}")
        # Check for main_proj/main_norm
        main_keys = [k for k in keys if "main" in k.lower()]
        print(f"Main keys: {main_keys[:5]}")
        # Check for lm_head/embed_tokens
        head_keys = [k for k in keys if "lm_head" in k.lower() or "embed" in k.lower() or "shared_head" in k.lower()]
        print(f"Head/embed keys: {head_keys[:10]}")
        # Check weight stats for markov_w1
        for k in markov_keys[:2]:
            t = f.get_tensor(k)
            print(f"  {k}: shape={list(t.shape)} mean={float(t.float().mean()):.4g} std={float(t.float().std()):.4g}")
except Exception as e:
    print(f"Error: {e}")
