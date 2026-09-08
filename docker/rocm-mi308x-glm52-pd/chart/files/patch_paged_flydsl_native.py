#!/usr/bin/env python3
"""Overlay the paged FlyDSL kernel and enable it in sglang."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indexer",
        default="/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py",
    )
    parser.add_argument("--flydsl-src", default="")
    parser.add_argument(
        "--enable-script",
        default="/opt/aiter-scripts/enable_paged_flydsl.py",
    )
    args = parser.parse_args()

    indexer = Path(args.indexer)
    enable_script = Path(args.enable_script)
    if not enable_script.exists():
        raise SystemExit(f"enable script missing: {enable_script}")

    kernel_dst = Path(
        "/sgl-workspace/aiter/aiter/ops/flydsl/kernels/fp8_mqa_logits.py"
    )
    kernel_has_paged = (
        kernel_dst.exists()
        and "def flydsl_fp8_paged_mqa_logits" in kernel_dst.read_text()
    )
    needs_enable = "flydsl_fp8_paged_mqa_logits" not in indexer.read_text()
    if args.flydsl_src and (needs_enable or not kernel_has_paged):
        overlay_flydsl(Path(args.flydsl_src))

    subprocess.run(
        [sys.executable, str(enable_script), str(indexer)],
        check=True,
    )


if __name__ == "__main__":
    main()
