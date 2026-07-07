#!/usr/bin/env python3
"""Check DSpark target cache data quality — detect NaN/Inf/extreme hidden states."""
import struct, json, os, sys, torch
import numpy as np

base = sys.argv[1] if len(sys.argv) > 1 else "/data/dspark_target_cache_v9_coding_clean/partial_0"
n_check = int(sys.argv[2]) if len(sys.argv) > 2 else 10

manifest = json.load(open(f"{base}/manifest.json"))
hidden_size = manifest["hidden_size"]
num_layers = len(manifest["target_layer_ids"])
idx_struct = struct.Struct("<QIIQQQQQ")
idx_bytes = open(f"{base}/samples.idx", "rb").read()
n = len(idx_bytes) // idx_struct.size
print(f"base={base}")
print(f"num_samples in idx: {n}, hidden_size={hidden_size}, num_layers={num_layers}")

shard_files = {s["shard_id"]: open(f"{base}/{s['file_name']}", "rb") for s in manifest["shards"]}

stats = {"nan": 0, "inf": 0, "extreme": 0, "ok": 0}
for i in range(min(n_check, n)):
    entry = idx_struct.unpack(idx_bytes[i * idx_struct.size:(i + 1) * idx_struct.size])
    sample_id, shard_id, seq_len, ii_off, am_off, lm_off, th_off, tlh_off = entry
    fh = shard_files[shard_id]
    th_nbytes = seq_len * num_layers * hidden_size * 2
    fh.seek(th_off)
    raw = fh.read(th_nbytes)
    arr = np.frombuffer(raw, dtype=np.uint16).copy()
    th = torch.from_numpy(arr).view(torch.bfloat16).float().reshape(seq_len, num_layers, hidden_size)
    has_nan = bool(torch.isnan(th).any())
    has_inf = bool(torch.isinf(th).any())
    mx = float(th.max()); mn = float(th.min()); mean = float(th.mean())
    if has_nan:
        status, stats["nan"] = "NaN", stats["nan"] + 1
    elif has_inf:
        status, stats["inf"] = "INF", stats["inf"] + 1
    elif abs(mx) > 1e4 or abs(mn) > 1e4:
        status, stats["extreme"] = "EXTREME", stats["extreme"] + 1
    else:
        status, stats["ok"] = "OK", stats["ok"] + 1
    print(f"  sample {sample_id}: seq_len={seq_len} mean={mean:.4g} max={mx:.4g} min={mn:.4g} {status}")

print(f"\nStats of first {min(n_check, n)}: {stats}")
print("VERDICT:", "CORRUPT" if stats["ok"] < min(n_check, n) else "CLEAN")
