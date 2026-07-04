#!/usr/bin/env python3
"""Generate AITER BF16 tuned config for K=6144 on 0702 image.
Only adds N=32 and N=160 (missing shapes). Does NOT touch N=128/256
to avoid cross-config duplicates with glm47/minimax configs.
Benchmark-proven: torch native is optimal for N=32 and N=160."""
import csv

CONFIG_FILE = "/sgl-workspace/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv"

with open(CONFIG_FILE, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

existing = set()
for row in rows:
    try:
        M, N, K = int(row[2]), int(row[3]), int(row[4])
    except (IndexError, ValueError):
        continue
    existing.add((M, N, K))

def make_torch_row(M, N, K):
    return [
        "gfx950", "256", str(M), str(N), str(K),
        "False", "torch.bfloat16", "torch.bfloat16",
        "False", "False", "torch", "0", "0", "0",
        "native", "0.0", "0.0", "0.0"
    ]

added = 0
for M in range(1, 50001):
    for N in [32, 160]:
        if (M, N, 6144) in existing:
            continue
        rows.append(make_torch_row(M, N, 6144))
        existing.add((M, N, 6144))
        added += 1

with open(CONFIG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"[DONE] Added {added} entries (N=32+N=160 only). Total: {len(rows)}")
