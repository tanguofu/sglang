#!/usr/bin/env python3
"""Add missing BF16 GEMM tuned shapes for K=6144 to AITER config.
Missing M values: 264, 288, 312, 336, 360, 384, 434, 1953, 3648
For N=32 and N=256."""
import csv

CONFIG_FILE = "/tmp/aiter_configs/bf16_tuned_gemm.csv"

# Read existing entries
with open(CONFIG_FILE, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

# Find existing entries for K=6144
existing = {}
for row in rows:
    if row[4] == "6144":  # K=6144
        M, N = int(row[2]), int(row[3])
        existing[(M, N)] = row

# Template entries (from nearest available M)
# M=256 for small batches, M=16384 for large batches
template_small = {
    32: existing.get((256, 32)),
    256: existing.get((256, 256)),
}
template_large = {
    32: existing.get((16384, 32)),
    256: existing.get((16384, 256)),
}

missing_M = [264, 288, 312, 336, 360, 384, 434, 1953, 3648]
added = 0

for M in missing_M:
    for N in [32, 256]:
        if (M, N) in existing:
            continue
        # Choose template based on M size
        if M < 1000:
            template = template_small[N]
        else:
            template = template_large[N]
        
        if template is None:
            print(f"  SKIP: No template for M={M}, N={N}")
            continue
        
        # Create new entry with modified M
        new_row = template.copy()
        new_row[2] = str(M)  # Update M value
        rows.append(new_row)
        added += 1
        print(f"  Added: M={M}, N={N}, K=6144, libtype={new_row[10]}, kernel={new_row[14][:40]}...")

# Write back
with open(CONFIG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"\n[DONE] Added {added} entries. Total entries: {len(rows)}")
