#!/usr/bin/env python3
"""
Driver run inside an sglang pod to tune the MISSING bf16 GEMM shapes
observed in 2tp8 prod logs: N=160, K=6144 (shared-expert / projection GEMM).

The existing /etc/aiter-configs/bf16_tuned_gemm.csv has 200 gfx942 rows but
ZERO rows for N=160, so every M in {4..5955} falls back to the untuned torch
default. This driver writes an untune CSV, invokes the aiter a16w16 tuner
(/sgl-workspace/aiter/csrc/gemm_a16w16/gemm_a16w16_tune.py), and prints the
resulting gfx942 CSV rows ready to merge into the ConfigMap / bake into image.

Run with:
  kubectl exec <pod> -- python3 /tmp/tune_n160_driver.py <mode>
    mode=probe  -> only print padded-M mapping + untune CSV, no tuning
    mode=tune   -> tune padded-M buckets, print CSV rows
"""
import os
import sys
import subprocess

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("SGLANG_USE_AITER", "1")

import torch  # noqa: E402
import aiter  # noqa: E402
from aiter.jit.utils.chip_info import get_cu_num, get_gfx  # noqa: E402
from aiter.ops.gemm_op_common import get_padded_m  # noqa: E402

GFX = get_gfx()
CU = get_cu_num()
print(f"[driver] gfx={GFX} cu_num={CU} devices={torch.cuda.device_count()}", flush=True)
if GFX != "gfx942":
    print(f"[driver] WARN: expected gfx942, got {GFX}", flush=True)

TUNER = "/sgl-workspace/aiter/csrc/gemm_a16w16/gemm_a16w16_tune.py"

# Observed missing M values from prod logs (N=160, K=6144), deduped & sorted.
OBSERVED_M = [
    4, 5, 6, 7, 8, 9, 12, 16, 24, 32, 43, 48, 60, 66, 72, 78, 87, 88, 90,
    96, 98, 573, 721, 5955,
]
# Dense small-M decode range (covers bs*steps*draft_tokens combos up to bs=48).
SMALL_M = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
N = 160
K = 6144
DTYPE = "torch.bfloat16"
OTYPE = "torch.bfloat16"

UNTUNE_CSV = "/tmp/tune_n160_untune.csv"
TUNE_OUT_CSV = "/tmp/tune_n160_tuned.csv"


def probe():
    """Print padded-M mapping for all observed + small M, write untune CSV."""
    print(f"[probe] N={N} K={K}", flush=True)
    print(f"{'rawM':>8} {'pad0':>8} {'pad1':>8}", flush=True)
    seen = set()
    for M in sorted(set(OBSERVED_M + SMALL_M)):
        try:
            p0 = get_padded_m(M, N, K, 0)
            p1 = get_padded_m(M, N, K, 1)
        except Exception as e:
            p0 = f"ERR:{e}"
            p1 = ""
        print(f"{M:>8} {str(p0):>8} {str(p1):>8}", flush=True)
        for p in (p0, p1):
            if isinstance(p, int):
                seen.add(p)
    buckets = sorted(seen)
    print(f"[probe] unique padded-M buckets to tune: {buckets}", flush=True)
    print(f"[probe] total shapes to tune: {len(buckets)}", flush=True)

    # Write untune CSV in the format the tuner expects.
    header = "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle\n"
    lines = [header]
    for M in buckets:
        lines.append(
            f"{GFX},{CU},{M},{N},{K},False,{DTYPE},{OTYPE},False,False\n"
        )
    with open(UNTUNE_CSV, "w") as f:
        f.writelines(lines)
    print(f"[probe] wrote {len(buckets)} shapes to {UNTUNE_CSV}", flush=True)
    print(f"[probe] untune CSV content:", flush=True)
    print("".join(lines), flush=True)
    return buckets


def tune():
    """Run the aiter a16w16 tuner, print resulting gfx942 CSV rows."""
    buckets = probe()
    if not buckets:
        print("[tune] no buckets to tune, abort", flush=True)
        return

    # Restrict to backends that work on gfx942. asm is gfx950-only; opus/torch/
    # skinny/triton are the gfx942 candidates.
    libtypes = os.environ.get("TUNE_N160_LIBTYPES", "torch,skinny,triton,opus")
    cmd = [
        sys.executable, TUNER,
        "-i", UNTUNE_CSV,
        "-o", TUNE_OUT_CSV,
        "--libtype", libtypes,
        "--mp", "1",
        "--warmup", "5",
        "--iters", "101",
        "--errRatio", "0.05",
    ]
    print(f"[tune] running tuner: {' '.join(cmd)}", flush=True)
    print(f"[tune] this takes ~{len(buckets) * 3}-{len(buckets) * 8} sec "
          f"({len(buckets)} shapes)", flush=True)
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = "0"
    env["SGLANG_USE_AITER"] = "1"
    p = subprocess.run(cmd, env=env, text=True)
    print(f"[tune] tuner exit={p.returncode}", flush=True)

    if os.path.exists(TUNE_OUT_CSV):
        print("=== TUNED CSV ROWS (gfx942) ===", flush=True)
        with open(TUNE_OUT_CSV) as f:
            print(f.read(), flush=True)
    else:
        print(f"[tune] NO output csv produced at {TUNE_OUT_CSV}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if mode == "tune":
        tune()
    else:
        probe()
        print("[driver] probe-only mode (use 'tune' to actually tune)", flush=True)
