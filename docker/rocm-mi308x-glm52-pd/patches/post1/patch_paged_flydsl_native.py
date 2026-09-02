#!/usr/bin/env python3
"""Replace the CUDA-graph-unsafe ragged FlyDSL gather path with native paged MFMA.

v0831-dp8ep8-mtp-paged-flydsl-v3 still gathers KV into a ragged buffer and then
calls flydsl_fp8_mqa_logits(ks, ke). Decode CUDA-graph capture leaves
indexer_k_start_end=None, so get_indexer_kvcache_range() unpacks None.

This script is idempotent and safe to run at container start:
  1. Overlay flydsl kernel files if a source dir is provided.
  2. Swap the old SGLANG_DSA_PAGED_FLYDSL branch for flydsl_fp8_paged_mqa_logits.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


OLD_BRANCH_START = "        if _is_hip and get_bool_env_var(\"SGLANG_DSA_PAGED_FLYDSL\"):\n"
OLD_IMPORT = "                flydsl_fp8_mqa_logits,\n"
NEW_BRANCH = '''        if (
            _is_hip
            and get_bool_env_var("SGLANG_DSA_PAGED_FLYDSL")
            and not _use_aiter_preshuffle
            and page_size == 1
        ):
            from aiter.ops.flydsl.kernels.fp8_mqa_logits import (
                flydsl_fp8_paged_mqa_logits,
            )

            if next_n == 4:
                paged_flydsl_variant = "mfma_r4_w4"
            elif next_n == 2:
                paged_flydsl_variant = "mfma_r2_w4"
            else:
                paged_flydsl_variant = "mfma_r1_w4"

            kv_flat = kv_cache_fp8.reshape(-1)
            if _is_fp8_fnuz:
                kv_flat = kv_flat.view(torch.float8_e4m3fnuz)
            else:
                kv_flat = kv_flat.view(torch.float8_e4m3fn)
            kv_scales = kv_flat.view(torch.float32)[
                (head_dim_with_sf - 4) // 4 :: head_dim_with_sf // 4
            ]
            seqlens_for_flydsl = (
                seqlens_32.reshape(-1) if seqlens_32.dim() == 2 else seqlens_32
            )
            rows = q_fp8[:q_offset].shape[0]
            tables = block_tables
            if tables.shape[0] != rows:
                if tables.shape[0] == 0 or rows % tables.shape[0] != 0:
                    raise ValueError(
                        f"paged FlyDSL block table rows mismatch: tables={tables.shape[0]}, q_rows={rows}"
                    )
                tables = tables.repeat_interleave(rows // tables.shape[0], dim=0)
            if (
                seqlens_for_flydsl.shape[0] != rows
            ):
                if seqlens_for_flydsl.shape[0] == 0 or rows % seqlens_for_flydsl.shape[0] != 0:
                    raise ValueError(
                        f"paged FlyDSL context length rows mismatch: context_lens={seqlens_for_flydsl.shape[0]}, q_rows={rows}"
                    )
                seqlens_for_flydsl = seqlens_for_flydsl.repeat_interleave(
                    rows // seqlens_for_flydsl.shape[0]
                )
            logits = flydsl_fp8_paged_mqa_logits(
                q_fp8[:q_offset],
                kv_flat,
                kv_scales,
                weights[:q_offset],
                seqlens_for_flydsl,
                tables,
                max_seq_len,
                variant=paged_flydsl_variant,
            )
            row_starts = None
        elif self.paged_mqa_logits_backend.is_aiter():
'''


def overlay_flydsl(src_dir: Path) -> None:
    kernel_src = src_dir / "flydsl_fp8_mqa_logits.py"
    init_src = src_dir / "flydsl_init.py"
    kernel_dst = Path(
        "/sgl-workspace/aiter/aiter/ops/flydsl/kernels/fp8_mqa_logits.py"
    )
    init_dst = Path("/sgl-workspace/aiter/aiter/ops/flydsl/__init__.py")
    if kernel_src.exists():
        shutil.copy2(kernel_src, kernel_dst)
        print(f"[ok] overlay {kernel_dst}")
    if init_src.exists():
        shutil.copy2(init_src, init_dst)
        print(f"[ok] overlay {init_dst}")
    kernel = kernel_dst.read_text()
    if "def flydsl_fp8_paged_mqa_logits" not in kernel:
        raise SystemExit("flydsl overlay missing flydsl_fp8_paged_mqa_logits")


def replace_old_branch(indexer: Path) -> str:
    src = indexer.read_text()
    if "flydsl_fp8_paged_mqa_logits" in src:
        return "already-native"
    start = src.find(OLD_BRANCH_START)
    if start < 0 or OLD_IMPORT not in src[start : start + 400]:
        raise SystemExit(
            "native paged FlyDSL branch not found and old ragged branch absent; "
            "refusing to silently fall back to AITER"
        )
    end = src.find("        elif self.paged_mqa_logits_backend.is_aiter():\n", start)
    if end < 0:
        raise SystemExit("aiter elif after old FlyDSL branch not found")
    src = src[:start] + NEW_BRANCH + src[end + len("        elif self.paged_mqa_logits_backend.is_aiter():\n") :]
    if src.count("flydsl_fp8_paged_mqa_logits") < 1:
        raise SystemExit("native paged FlyDSL replacement failed")
    if "ks, ke = metadata.get_indexer_kvcache_range()" in src[start : start + 1200]:
        raise SystemExit("ragged ks/ke unpack still present in FlyDSL branch")
    indexer.write_text(src)
    return "replaced-ragged"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indexer",
        default="/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py",
    )
    parser.add_argument("--flydsl-src", default="")
    args = parser.parse_args()
    indexer = Path(args.indexer)
    src = indexer.read_text()
    needs_replace = (
        "flydsl_fp8_paged_mqa_logits" not in src
        and OLD_BRANCH_START in src
        and OLD_IMPORT in src
    )
    kernel_dst = Path(
        "/sgl-workspace/aiter/aiter/ops/flydsl/kernels/fp8_mqa_logits.py"
    )
    kernel_has_paged = (
        kernel_dst.exists()
        and "def flydsl_fp8_paged_mqa_logits" in kernel_dst.read_text()
    )
    if args.flydsl_src and (needs_replace or not kernel_has_paged):
        overlay_flydsl(Path(args.flydsl_src))
    status = replace_old_branch(indexer)
    print(f"[ok] paged FlyDSL native patch: {status}")


if __name__ == "__main__":
    main()
