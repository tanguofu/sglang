#!/usr/bin/env python3
"""
Driver run inside the tune19-sglang-0 pod to:
  1. discover get_padded_m buckets for observed decode shapes,
  2. write an untune CSV (gfx942 rows) for the missing (N=256/32, K=6144) shapes,
  3. invoke the aiter a16w16 GEMM tuner on 1 GPU,
  4. print the resulting gfx942 rows so they can be merged.

Run with:
  kubectl exec tune19-sglang-0 -- python3 /tmp/tune19_gfx942_driver.py <mode>
    mode=padded   -> tune padded-M buckets for broad coverage
    mode=exact    -> tune exact observed raw M
    mode=probe    -> only print padded-M mapping, no tuning
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
print(f"[driver] gfx={GFX} cu_num={CU} devices={torch.cuda.device_count()}")

# Observed missing shapes from prod logs: K=6144, N in {256, 32}, bf16->bf16,
# bias=False, scaleAB=False, bpreshuffle=False. Raw M values seen:
OBSERVED_M = [38, 553, 739, 1594, 8575, 10564, 12869, 15768]
# Dense small-M decode range (covers bs*steps*draft_tokens combos):
SMALL_M = [1, 2, 4, 8, 16, 24, 30, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
N_LIST = [256, 32]
K = 6144
DTYPE = "torch.bfloat16"
OTYPE = "torch.bfloat16"


def probe():
    print(f"[probe] gfx={GFX} cu={CU}")
    print(f"{'rawM':>8} {'N':>4} {'K':>5} {'pad0':>8} {'pad1':>8}")
    seen = set()
    for N in N_LIST:
        for M in OBSERVED_M + SMALL_M:
            try:
                p0 = get_padded_m(M, N, K, 0)
                p1 = get_padded_m(M, N, K, 1)
            except Exception as e:
                p0 = f"ERR:{e}"
                p1 = ""
            print(f"{M:>8} {N:>4} {K:>5} {str(p0):>8} {str(p1):>8}")
            for p in (p0, p1):
                if isinstance(p, int):
                    seen.add((N, p))
    print(f"[probe] unique (N, paddedM) buckets: {sorted(seen)}")
    return seen


def write_untune_csv(path, m_values_by_N):
    header = "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle\n"
    lines = [header]
    for N, ms in m_values_by_N.items():
        for M in sorted(set(ms)):
            lines.append(
                f"{GFX},{CU},{M},{N},{K},False,{DTYPE},{OTYPE},False,False\n"
            )
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"[driver] wrote {len(lines)-1} shapes to {path}")
    print("".join(lines))


def run_tuner(untune_csv, tune_out_csv):
    tuner = "/sgl-workspace/aiter/csrc/gemm_a16w16/gemm_a16w16_tune.py"
    # dtype/outdtype come from the untune CSV (torch.bfloat16). Restrict to
    # backends that work on gfx942 without extra flags; asm is what gfx950 uses.
    libtypes = os.environ.get("TUNE19_LIBTYPES", "asm,torch,skinny,triton,opus")
    cmd = [
        sys.executable, tuner,
        "-i", untune_csv,
        "-o", tune_out_csv,
        "--libtype", libtypes,
        "--mp", "1",
        "--warmup", "5",
        "--iters", "101",
        "--errRatio", "0.05",
    ]
    print(f"[driver] running tuner: {' '.join(cmd)}")
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = "0"
    env["SGLANG_USE_AITER"] = "1"
    p = subprocess.run(cmd, env=env, text=True)
    print(f"[driver] tuner exit={p.returncode}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if mode == "probe":
        probe()
        return
    untune_csv = "/tmp/tune19_untune.csv"
    tune_out_csv = "/tmp/tune19_gfx942_tuned.csv"
    if mode == "padded":
        buckets = probe()
        m_by_N = {}
        for N, pM in buckets:
            m_by_N.setdefault(N, []).append(pM)
        # also include exact observed M for guaranteed gl=None hits
        for N in N_LIST:
            m_by_N.setdefault(N, []).extend(OBSERVED_M)
        write_untune_csv(untune_csv, m_by_N)
    elif mode == "exact":
        m_by_N = {N: list(OBSERVED_M) + list(SMALL_M) for N in N_LIST}
        write_untune_csv(untune_csv, m_by_N)
    else:
        print(f"unknown mode {mode}")
        return
    run_tuner(untune_csv, tune_out_csv)
    # print resulting gfx942 rows
    if os.path.exists(tune_out_csv):
        print(f"[driver] === tuned rows ({tune_out_csv}) ===")
        with open(tune_out_csv) as f:
            print(f.read())
    else:
        print(f"[driver] NO output csv produced at {tune_out_csv}")


if __name__ == "__main__":
    main()
