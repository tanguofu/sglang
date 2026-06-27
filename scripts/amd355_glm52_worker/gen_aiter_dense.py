#!/usr/bin/env python3
"""Generate dense AITER BF16 tuned config entries for K=6144.
Covers M from 1 to 50000 in steps of 1 for N=32 and N=256."""
import csv

CONFIG_FILE = "/sgl-workspace/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv"

with open(CONFIG_FILE, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

# Collect existing entries and templates
existing = set()
templates = {}  # (M, N) -> row
for row in rows:
    try:
        M, N, K = int(row[2]), int(row[3]), int(row[4])
    except (IndexError, ValueError):
        continue
    existing.add((M, N, K))
    if K == 6144:
        templates[(M, N)] = row

# Template entries: use M=256 for small, M=16384 for large
tmpl_small_32 = templates.get((256, 32))
tmpl_small_256 = templates.get((256, 256))
tmpl_large_32 = templates.get((16384, 32))
tmpl_large_256 = templates.get((16384, 256))

added = 0
for M in range(1, 50001):
    for N, tmpl_small, tmpl_large in [
        (32, tmpl_small_32, tmpl_large_32),
        (256, tmpl_small_256, tmpl_large_256),
    ]:
        if (M, N, 6144) in existing:
            continue
        tmpl = tmpl_small if M < 1000 else tmpl_large
        if tmpl is None:
            tmpl = tmpl_small or tmpl_large
        if tmpl is None:
            continue
        new_row = tmpl.copy()
        new_row[2] = str(M)
        new_row[3] = str(N)
        rows.append(new_row)
        added += 1

with open(CONFIG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"[DONE] Added {added} entries. Total: {len(rows)}")
