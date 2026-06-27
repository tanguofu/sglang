#!/usr/bin/env python3
"""Patch runtime AITER config with all missing BF16 GEMM shapes.
Reads missing shapes from /tmp/missing_shapes.txt, finds nearest template, appends."""
import csv, re, sys

CONFIG_FILE = "/tmp/aiter_configs/bf16_tuned_gemm.csv"
MISSING_FILE = "/tmp/missing_shapes.txt"

# Read existing entries
with open(CONFIG_FILE, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

# Build lookup
existing = set()
templates_by_N = {}  # N -> {M -> row}
for row in rows:
    try:
        M, N, K = int(row[2]), int(row[3]), int(row[4])
    except (IndexError, ValueError):
        continue
    existing.add((M, N, K))
    if K == 6144:
        templates_by_N.setdefault(N, {})[M] = row

# Read missing shapes
missing_shapes = set()
with open(MISSING_FILE) as f:
    for line in f:
        m = re.match(r"M:(\d+), N:(\d+), K:(\d+)", line.strip())
        if m:
            M, N, K = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if (M, N, K) not in existing:
                missing_shapes.add((M, N, K))

print(f"[INFO] Found {len(missing_shapes)} missing shapes to add")

# For each missing shape, find nearest template
added = 0
for M, N, K in sorted(missing_shapes):
    if K != 6144:
        continue
    
    # Find nearest M in templates for this N
    n_templates = templates_by_N.get(N, {})
    if not n_templates:
        # Fallback: use N=256 templates
        n_templates = templates_by_N.get(256, {})
    if not n_templates:
        n_templates = templates_by_N.get(32, {})
    
    if not n_templates:
        print(f"  SKIP: No template for M={M}, N={N}, K={K}")
        continue
    
    nearest_M = min(n_templates.keys(), key=lambda x: abs(x - M))
    template = n_templates[nearest_M]
    
    new_row = template.copy()
    new_row[2] = str(M)
    new_row[3] = str(N)
    rows.append(new_row)
    added += 1

# Write back
with open(CONFIG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"[DONE] Added {added} entries. Total: {len(rows)}")
