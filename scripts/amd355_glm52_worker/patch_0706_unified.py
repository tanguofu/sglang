#!/usr/bin/env python3
"""
Unified GLM-5.2 ROCm/MI355X patch bundle for SGLang 0706 image.

All patches and config generators merged into one self-contained script:
  01-05: HIP/DSA/MTP/AITER enablement (from patch_sglang_glm52_rocm_all)
  06a:   Supplement v4 (view->reshape, cos_sin_cache, dual_stream threshold)
  06b:   D2H sync elimination (dsa_backend + frozen_kv_mtp_worker)
  06c:   Draft extend CUDA graph enablement for HIP
  Gen1:  AITER BF16 tuned config generation (N=32, N=160)
  Gen2:  A8W8 blockscale tuned config generation

Usage: python3 /data/patch_0706_unified.py
Idempotent: safe to run multiple times.
"""
from __future__ import annotations
import os, sys, re, csv, tempfile, subprocess
from pathlib import Path

SGLANG_SRC = "/sgl-workspace/sglang/python/sglang/srt"
AITER_CONFIGS = "/sgl-workspace/aiter/aiter/configs/model_configs"
errors = []

# ============================================================
# 01-05: Run the existing patch_sglang_glm52_rocm_all.py
# ============================================================
def run_bundle_01_05():
    """Execute the original 5-patch bundle (01-05)."""
    bundle = "/data/patch_sglang_glm52_rocm_all.py"
    if not os.path.exists(bundle):
        print("[ERROR] patch_sglang_glm52_rocm_all.py not found")
        errors.append("01-05 bundle missing")
        return
    print("\n" + "="*60)
    print("Running 01-05: patch_sglang_glm52_rocm_all.py")
    print("="*60)
    rc = subprocess.call([sys.executable, bundle])
    if rc != 0:
        errors.append("01-05 bundle failed")

# ============================================================
# 06a: Supplement v4
# ============================================================
def patch_06a_supplement_v4():
    print("\n" + "="*60)
    print("Running 06a: Supplement v4")
    print("="*60)
    script = "/data/patch_0706_supplement_v4.py"
    if os.path.exists(script):
        rc = subprocess.call([sys.executable, script])
        if rc != 0:
            errors.append("06a supplement_v4 failed")
    else:
        print("[SKIP] patch_0706_supplement_v4.py not found")

# ============================================================
# 06b: D2H sync elimination
# ============================================================
def patch_06b_d2h_sync():
    print("\n" + "="*60)
    print("Running 06b: D2H sync elimination")
    print("="*60)

    # Patch 1: dsa_backend.py
    dsa_path = os.path.join(SGLANG_SRC, "layers/attention/dsa_backend.py")
    with open(dsa_path, "r") as f:
        content = f.read()

    old_block = """        if forward_batch.seq_lens_cpu is not None:
            max_seqlen_k = int(
                forward_batch.seq_lens_cpu.max().item() + draft_token_num
            )
        else:
            # needs_cpu_seq_lens=False nulls the host mirror for spec-v2 relay
            # batches; graph replay uses the static page-table width, so only this
            # eager (e.g. over-capture-bs) fallback needs a length here.
            max_seqlen_k = int(forward_batch.seq_lens.max().item()) + draft_token_num"""

    new_block = """        if forward_batch.seq_lens_cpu is not None:
            max_seqlen_k = int(
                forward_batch.seq_lens_cpu.max().item() + draft_token_num
            )
        elif forward_batch.seq_lens_sum is not None:
            # Avoid D2H sync: use pre-computed sum as upper bound for max.
            # For bs=1, sum == max (exact); for bs>1, sum > max (safe over-estimate).
            max_seqlen_k = forward_batch.seq_lens_sum + draft_token_num
        else:
            # needs_cpu_seq_lens=False nulls the host mirror for spec-v2 relay
            # batches; graph replay uses the static page-table width, so only this
            # eager (e.g. over-capture-bs) fallback needs a length here.
            max_seqlen_k = int(forward_batch.seq_lens.max().item()) + draft_token_num"""

    if old_block in content:
        content = content.replace(old_block, new_block)
        with open(dsa_path, "w") as f:
            f.write(content)
        print("[OK] Patched dsa_backend.py - eliminated GPU .item() D2H sync")
    else:
        print("[SKIP] dsa_backend.py - pattern not found (may already be patched)")

    # Patch 2: frozen_kv_mtp_worker_v2.py
    mtp_path = os.path.join(SGLANG_SRC, "speculative/frozen_kv_mtp_worker_v2.py")
    with open(mtp_path, "r") as f:
        content = f.read()

    old_line = "        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()"
    new_line = """        if batch.seq_lens_cpu is not None:
            batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()
        else:
            batch.seq_lens_sum = torch.sum(batch.seq_lens).item()"""

    if old_line in content:
        content = content.replace(old_line, new_line, 1)
        with open(mtp_path, "w") as f:
            f.write(content)
        print("[OK] Patched frozen_kv_mtp_worker_v2.py - use seq_lens_cpu for sum")
    else:
        print("[SKIP] frozen_kv_mtp_worker_v2.py - pattern not found (may already be patched)")

# ============================================================
# 06c: Draft extend CUDA graph for HIP
# ============================================================
def patch_06c_draft_extend_graph():
    print("\n" + "="*60)
    print("Running 06c: Draft extend CUDA graph for HIP")
    print("="*60)

    path = os.path.join(SGLANG_SRC, "speculative/eagle_worker_v2.py")
    with open(path, "r") as f:
        content = f.read()

    # Patch 1: Add DSA to graph_supported_backend_types for HIP
    old1 = """        if _is_cuda or _is_musa:
            # DSA is CUDA-only; import lazily so non-CUDA builds don't pull in
            # deep_gemm and the rest of the sparse-attention stack at import time.
            from sglang.srt.layers.attention.dsa_backend import (
                DeepseekSparseAttnBackend,
            )

            graph_supported_backend_types.append(DeepseekSparseAttnBackend)"""

    new1 = """        if _is_cuda or _is_musa or _is_hip:
            # DSA is CUDA-only; import lazily so non-CUDA builds don't pull in
            # deep_gemm and the rest of the sparse-attention stack at import time.
            from sglang.srt.layers.attention.dsa_backend import (
                DeepseekSparseAttnBackend,
            )

            graph_supported_backend_types.append(DeepseekSparseAttnBackend)"""

    if old1 in content:
        content = content.replace(old1, new1)
        print("[OK] Patched graph_supported_backend_types to include HIP")
    else:
        print("[SKIP] graph_supported_backend_types pattern not found")

    # Patch 2: Enable supports_cuda_draft_extend_graph for HIP
    old2 = """        supports_cuda_draft_extend_graph = (
            _is_cuda or _is_musa
        ) and graph_supported_backend"""

    new2 = """        supports_cuda_draft_extend_graph = (
            _is_cuda or _is_musa or _is_hip
        ) and graph_supported_backend"""

    if old2 in content:
        content = content.replace(old2, new2)
        print("[OK] Patched supports_cuda_draft_extend_graph to include HIP")
    else:
        print("[SKIP] supports_cuda_draft_extend_graph pattern not found")

    with open(path, "w") as f:
        f.write(content)

# ============================================================
# Gen1: AITER BF16 tuned config
# ============================================================
def gen_bf16_config():
    print("\n" + "="*60)
    print("Running Gen1: AITER BF16 tuned config")
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

# ============================================================
# Gen2: A8W8 blockscale tuned config
# ============================================================
def gen_a8w8_config():
    print("\n" + "="*60)
    print("Running Gen2: A8W8 blockscale tuned config")
    print("="*60)

    config_file = os.path.join(AITER_CONFIGS, "glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv")
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
        os.path.join(AITER_CONFIGS, "a8w8_blockscale_bpreshuffle_tuned_gemm_dsv3.csv"),
        "/sgl-workspace/aiter/aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3.5_397b.csv",
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

    with open(config_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[DONE] Added {added} a8w8 entries. Total: {len(rows)}")

# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("Unified GLM-5.2 ROCm 0706 Patch Script")
    print("="*60)

    run_bundle_01_05()
    patch_06a_supplement_v4()
    patch_06b_d2h_sync()
    patch_06c_draft_extend_graph()
    gen_bf16_config()
    gen_a8w8_config()

    print("\n" + "="*60)
    if errors:
        print(f"[FAILED] {len(errors)} step(s) failed: {', '.join(errors)}")
        sys.exit(1)
    else:
        print("[DONE] All patches and config generation completed successfully")
    print("="*60)

if __name__ == "__main__":
    main()
