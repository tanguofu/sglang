#!/usr/bin/env python3
"""Generate AITER tuned GEMM configs for GLM-5.2 on MI355X (gfx950).

Extracted from patch_0706_unified.py Gen1+Gen2. Run inside the Docker image
where /sgl-workspace/aiter/ already exists with base configs.

Gen1: AITER BF16 tuned config (adds M=1..50000, N=32+160, K=6144)
Gen2: A8W8 blockscale tuned config (adds M=1..65536, N=128+2624+3072+6144, K=6144)

Usage: python3 /gen_aiter_configs.py
Idempotent: safe to run multiple times.
"""
from __future__ import annotations
import csv, os

AITER_CONFIGS = "/sgl-workspace/aiter/aiter/configs/model_configs"


def gen_bf16_config():
    print("="*60)
    print("Gen1: AITER BF16 tuned config")
    print("="*60)

    config_file = os.path.join(AITER_CONFIGS, "glm5_bf16_tuned_gemm.csv")
    with open(config_file, "r") as f:
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

    with open(config_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[DONE] Added {added} BF16 entries (N=32+N=160). Total: {len(rows)}")


def gen_a8w8_config():
    print("="*60)
    print("Gen2: A8W8 blockscale tuned config")
    print("="*60)

    config_file = os.path.join(
        AITER_CONFIGS, "glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
    )
    with open(config_file, "r") as f:
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
    for alt_path in [
        os.path.join(
            AITER_CONFIGS, "a8w8_blockscale_bpreshuffle_tuned_gemm_dsv3.csv"
        ),
        "/sgl-workspace/aiter/aiter/configs/"
        "a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3.5_397b.csv",
    ]:
        if os.path.exists(alt_path):
            with open(alt_path) as af:
                areader = csv.reader(af)
                try:
                    next(areader)
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
            n_templates = {
                m: r for (m, n), r in templates.items() if n == N
            }
            if not n_templates:
                n_templates = {
                    m: r for (m, n), r in templates.items() if n == 128
                }
            if not n_templates:
                continue
            nearest_M = min(n_templates.keys(), key=lambda x: abs(x - M))
            template = n_templates[nearest_M]
            new_row = template.copy()
            new_row[2] = str(M)
            new_row[3] = str(N)
            rows.append(new_row)
            added += 1

    with open(config_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[DONE] Added {added} a8w8 entries. Total: {len(rows)}")


def main():
    print("="*60)
    print("AITER Config Generation for GLM-5.2 on MI355X")
    print("="*60)
    gen_bf16_config()
    gen_a8w8_config()
    print("="*60)
    print("[DONE] All AITER configs generated")
    print("="*60)


if __name__ == "__main__":
    main()
