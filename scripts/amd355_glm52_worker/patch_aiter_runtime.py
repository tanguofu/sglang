#!/usr/bin/env python3
"""Patch runtime AITER config with all missing BF16 GEMM shapes.
Reads missing shapes from docker logs, finds nearest template, appends entries."""
import csv, re, subprocess, sys

CONFIG_FILE = "/tmp/aiter_configs/bf16_tuned_gemm.csv"

# Read existing entries
with open(CONFIG_FILE, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

# Build lookup: (M, N) -> row, and collect existing M values for K=6144
existing = set()
templates = {}  # (M, N) -> row
all_M_for_6144 = {}  # M -> row (for N=32 and N=256)
for row in rows:
    try:
        M, N, K = int(row[2]), int(row[3]), int(row[4])
    except (IndexError, ValueError):
        continue
    existing.add((M, N, K))
    if K == 6144:
        all_M_for_6144.setdefault(N, {})[M] = row

# Get missing shapes from docker logs
result = subprocess.run(
    ["docker", "logs", "sglang_final", "2>&1"],
    capture_output=True, text=True, shell=True
)
logs = result.stdout + result.stderr

missing_shapes = set()
for m in re.finditer(r"shape is M:(\d+), N:(\d+), K:(\d+).*not found tuned config", logs):
    M, N, K = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if (M, N, K) not in existing:
        missing_shapes.add((M, N, K))

print(f"[INFO] Found {len(missing_shapes)} missing shapes")

# For each missing shape, find nearest template and create entry
added = 0
for M, N, K in sorted(missing_shapes):
    if K != 6144:
        # Skip non-BF16 shapes (a8w8 etc) for now
        continue
    
    # Find nearest M in templates for this N
    n_templates = all_M_for_6144.get(N, {})
    if not n_templates:
        # Try N=256 as fallback for other N values
        n_templates = all_M_for_6144.get(256, {})
    if not n_templates:
        n_templates = all_M_for_6144.get(32, {})
    
    if not n_templates:
        print(f"  SKIP: No template for M={M}, N={N}, K={K}")
        continue
    
    # Find nearest M
    nearest_M = min(n_templates.keys(), key=lambda x: abs(x - M))
    template = n_templates[nearest_M]
    
    new_row = template.copy()
    new_row[2] = str(M)  # Update M
    if N != int(new_row[3]):
        new_row[3] = str(N)  # Update N if different
    rows.append(new_row)
    added += 1

# Write back
with open(CONFIG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"[DONE] Added {added} entries. Total: {len(rows)}")
