#!/usr/bin/env python3
"""Generate dense a8w8 blockscale tuned config for GLM5.
Source: glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv
Covers M from 1 to 65536 for N=128, 2624, 3072, 6144 at K=6144."""
import csv

CONFIG_FILE = "/sgl-workspace/aiter/aiter/configs/model_configs/glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"

with open(CONFIG_FILE, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

existing = set()
templates = {}
for row in rows:
    try:
        M, N, K = int(row[2]), int(row[3]), int(row[4])
    except (IndexError, ValueError):
        continue
    existing.add((M, N, K))
    if K == 6144:
        templates[(M, N)] = row

target_N_values = [128, 2624, 3072, 6144]

# Also use templates from the merged config (dsv3 etc) if available
import os
for alt_path in [
    "/sgl-workspace/aiter/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_dsv3.csv",
    "/sgl-workspace/aiter/aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3.5_397b.csv",
]:
    if os.path.exists(alt_path):
        with open(alt_path) as af:
            areader = csv.reader(af)
            try:
                aheader = next(areader)
            except StopIteration:
                continue
            for row in areader:
                try:
                    M, N, K = int(row[2]), int(row[3]), int(row[4])
                except (IndexError, ValueError):
                    continue
                if K == 6144 and N in target_N_values:
                    if (M, N) not in templates:
                        templates[(M, N)] = row

added = 0
for M in range(1, 65537):
    for N in target_N_values:
        if (M, N, 6144) in existing:
            continue
        n_templates = {m: r for (m, n), r in templates.items() if n == N}
        if not n_templates:
            n_templates = {m: r for (m, n), r in templates.items() if n == 128}
        if not n_templates:
            continue
        nearest_M = min(n_templates.keys(), key=lambda x: abs(x - M))
        template = n_templates[nearest_M]
        new_row = template.copy()
        new_row[2] = str(M)
        new_row[3] = str(N)
        rows.append(new_row)
        added += 1

with open(CONFIG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"[DONE] Added {added} a8w8 entries. Total: {len(rows)}")
