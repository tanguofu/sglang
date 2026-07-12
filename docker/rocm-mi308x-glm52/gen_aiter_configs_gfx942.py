#!/usr/bin/env python3
"""Generate AITER tuned GEMM/fMoE configs for GLM-5.2 on MI308X (gfx942).

Adapted from docker/rocm-mi355x-glm52-0708/gen_aiter_configs.py (gfx950).

The gfx942 config is 100x smaller than gfx950 (1,482 vs 100K+ GEMM entries).
This script fills the critical gaps:

Gen1: A8W8 GEMM torch-fallback entries for missing MoE shapes:
  - Stage 1 (g1u1): (M, N=4096, K=6144) — ZERO entries currently
  - Router:         (M, N=256,  K=6144) — ZERO entries currently
  Stage 2 (M, N=6144, K=2048) already has full coverage — skipped.

Gen2: fMoE entries for missing CUDA graph batch sizes:
  - token=24,40,48,56,72,80,96 (bs=3,5,6,7,9,10,12)
  Uses nearest-neighbor template matching from existing tuned entries.

Usage: python3 gen_aiter_configs_gfx942.py
Idempotent: safe to run multiple times — only adds missing entries.
"""
from __future__ import annotations

import csv
import os
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GFX = "gfx942"
CU_NUM = 80  # MI308X CUs per GPU (304 total, 80 used for tuning)

PATCHES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "patches"
)
GEMM_CONFIG = os.path.join(PATCHES_DIR, "glm5_1_gemm_gfx942.csv")
FMOE_CONFIG = os.path.join(PATCHES_DIR, "glm5_1_fmoe_gfx942.csv")

# GLM-5.2 MoE shapes:
#   Stage 1 (gate+up, g1u1): hidden=6144 -> 2*inter=4096, so (M, N=4096, K=6144)
#   Stage 2 (down):           inter=2048 -> hidden=6144,   so (M, N=6144, K=2048)
#   Router:                   hidden=6144 -> num_experts,  so (M, N=256,  K=6144)
GEMM_TARGET_SHAPES = [
    (4096, 6144),   # Stage 1 MoE — ZERO entries currently
    (256, 6144),    # Router GEMM  — ZERO entries currently
    # (6144, 2048),  # Stage 2 MoE — already fully covered, skip
]

# M range: 1..16512 covers all decode and prefill batch sizes.
# CUDA graph bs: 1,2,3,4,5,6,7,8,9,10,12,16 -> token=bs*8 -> M=8..128
# Prefill can go up to 32768 tokens, but MoE padding rounds to 16512 max.
GEMM_MAX_M = 16512

# fMoE token values needed for CUDA graph batch sizes (token = bs * 8):
#   bs=1->8, 2->16, 3->24, 4->32, 5->40, 6->48, 7->56,
#   8->64, 9->72, 10->80, 12->96, 16->128
FMOE_NEEDED_TOKENS = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 128]

# GLM-5.2 fMoE parameters (constant across all entries)
FMOE_MODEL_DIM = 6144
FMOE_INTER_DIM = 2048
FMOE_EXPERT = 32
FMOE_TOPK = 8
FMOE_ACT_TYPE = "ActivationType.Silu"
FMOE_DTYPE = "torch.bfloat16"
FMOE_Q_DTYPE_A = "torch.float8_e4m3fnuz"
FMOE_Q_DTYPE_W = "torch.float8_e4m3fnuz"
FMOE_Q_TYPE = "QuantType.per_1x128"
FMOE_USE_G1U1 = 1
FMOE_DOWEIGHT_STAGE1 = 0


# ---------------------------------------------------------------------------
# Gen1: A8W8 GEMM torch-fallback entries
# ---------------------------------------------------------------------------

def gen_gemm_config():
    print("=" * 60)
    print("Gen1: A8W8 GEMM torch-fallback for missing gfx942 shapes")
    print("=" * 60)

    if not os.path.exists(GEMM_CONFIG):
        print(f"[ERROR] Config not found: {GEMM_CONFIG}")
        return

    with open(GEMM_CONFIG, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Build existing (M, N, K) set
    existing: set[tuple[int, int, int]] = set()
    for row in rows:
        try:
            m, n, k = int(row[2]), int(row[3]), int(row[4])
        except (IndexError, ValueError):
            continue
        existing.add((m, n, k))

    added = 0
    for n_val, k_val in GEMM_TARGET_SHAPES:
        missing_count = 0
        for m_val in range(1, GEMM_MAX_M + 1):
            if (m_val, n_val, k_val) in existing:
                continue
            # torch fallback: libtype=torch, kernelId=0, splitK=0,
            # us=0, kernelName=native, tflops=0.0, bw=0.0, errRatio=0.0
            row = [
                GFX, str(CU_NUM), str(m_val), str(n_val), str(k_val),
                "torch", "0", "0", "0", "native", "0.0", "0.0", "0.0",
            ]
            rows.append(row)
            existing.add((m_val, n_val, k_val))
            added += 1
            missing_count += 1
        print(
            f"  (N={n_val}, K={k_val}): added {missing_count} torch-fallback entries"
        )

    with open(GEMM_CONFIG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[DONE] Added {added} GEMM entries. Total: {len(rows)}")


# ---------------------------------------------------------------------------
# Gen2: fMoE entries for missing CUDA graph batch sizes
# ---------------------------------------------------------------------------

def gen_fmoe_config():
    print("=" * 60)
    print("Gen2: fMoE entries for missing CUDA graph batch sizes")
    print("=" * 60)

    if not os.path.exists(FMOE_CONFIG):
        print(f"[ERROR] Config not found: {FMOE_CONFIG}")
        return

    with open(FMOE_CONFIG, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Build existing token set and template map
    existing_tokens: set[int] = set()
    templates: dict[int, list[str]] = {}
    for row in rows:
        try:
            token = int(row[1])
        except (IndexError, ValueError):
            continue
        existing_tokens.add(token)
        templates[token] = row

    added = 0
    for token in FMOE_NEEDED_TOKENS:
        if token in existing_tokens:
            print(f"  token={token}: already present, skipping")
            continue

        # Find nearest existing token for template matching
        if not templates:
            print(f"  token={token}: no templates available, skipping")
            continue
        nearest = min(templates.keys(), key=lambda t: abs(t - token))
        template = templates[nearest].copy()

        # Override token value; keep kernel config from nearest template.
        # The fMoE kernel is parameterized by token count at runtime, so
        # the same kernel config works for different token values.
        template[1] = str(token)

        rows.append(template)
        templates[token] = template
        existing_tokens.add(token)
        added += 1
        print(f"  token={token}: added (template from token={nearest})")

    with open(FMOE_CONFIG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[DONE] Added {added} fMoE entries. Total: {len(rows)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("AITER Config Generation for GLM-5.2 on MI308X (gfx942)")
    print("=" * 60)
    gen_gemm_config()
    gen_fmoe_config()
    print("=" * 60)
    print("[DONE] All AITER configs generated")
    print("=" * 60)


if __name__ == "__main__":
    main()
