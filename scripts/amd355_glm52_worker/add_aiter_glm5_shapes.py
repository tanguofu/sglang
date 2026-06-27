#!/usr/bin/env python3
"""Add missing BF16 GEMM tuned shapes for K=6144 to GLM5 AITER config.
Missing M values: 264, 288, 312, 336, 360, 384, 434, 1953, 3648
For N=32 and N=256."""
import csv

CONFIG_FILE = "/sgl-workspace/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv"

with open(CONFIG_FILE, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

existing = set()
for row in rows:
    if row[4] == "6144":
        existing.add((int(row[2]), int(row[3])))

templates = {}
for row in rows:
    if row[4] == "6144":
        M, N = int(row[2]), int(row[3])
        templates[(M, N)] = row

missing_M = [264, 288, 312, 336, 360, 384, 434, 1953, 3648]
added = 0

for M in missing_M:
    for N in [32, 256]:
        if (M, N) in existing:
            continue
        if M < 1000:
            template = templates.get((256, N))
        else:
            template = templates.get((16384, N))
        if template is None:
            print(f"  SKIP: No template for M={M}, N={N}")
            continue
        new_row = template.copy()
        new_row[2] = str(M)
        rows.append(new_row)
        added += 1
        print(f"  Added: M={M}, N={N}, K=6144, libtype={new_row[10]}")

with open(CONFIG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"\n[DONE] Added {added} entries to glm5_bf16_tuned_gemm.csv. Total: {len(rows)}")
