"""Grouped fp8 block-scaled GEMM for DS4.1 MoE on gfx942 (MI308X).

One CTA owns one (expert, N tile) and walks that expert's *real* rows in
BLOCK_M-sized subtiles, streaming the expert's weight tile once per subtile.
The subtile count comes from a runtime ``nrows[e]``, so nothing is capped by a
compile-time constant and no row can be dropped: an expert with 400 rows and one
with 20 rows are both covered exactly.

Numerics follow the reference the MoE path already uses: fp8 products
accumulated in fp32, with the [128, 128] block scales applied once per K block
(``acc += partial * a_s[m, kk] * b_s[n, kk]``). No split-K, no atomics, fixed
K order, so a row is bitwise identical alone or in any batch.

Status: correct, measured in a microbenchmark, NOT wired into any MoE runner,
and NOT validated end-to-end. On this platform every microbench win so far has
failed to transfer, so nothing here may be claimed as a speedup until an
end-to-end run says so.

Measured on gfx942 (2026-10-27, tilelang a55a8230), E=48 with 256 rows per
expert, idle MI308X (80 CU), same-process interleaved repeats. Read
/tmp/poc/ds41/B5_RESULT.md before spending time on this file -- it corrects the
numbers that used to be quoted here and explains why the design premise
("A resident in registers", "built for decode thin-M") does not hold.

  * numerics: matches an fp32 reference to rel 1.4e-7 - 2.7e-7 (fp32 rounding),
    for uniform 256-row experts and for uneven cases (100/20/64/3 and
    400/20/256/128 rows) at BLOCK_M 64 and 128 -- no row is ever dropped;
  * best config (BLOCK_M=128 BLOCK_N=128 BLOCK_K=128 k_pack=2 stages=2
    threads=256 grid_subtile=True), gate_up N=4608 K=5120 (579.8 GFLOP):
    5.51 ms -> 105 TFLOPS; down N=5120 K=2304 (289.9 GFLOP): 2.41 ms ->
    120 TFLOPS. Layer total 7.92 ms.
  * triton fused_moe with the config sglang actually deploys (BLOCK_SIZE_M=64
    BLOCK_SIZE_N=128 GROUP_M=32 warps=4 stages=2): 4.66 + 2.93 = 7.59 ms.
    Best triton config found (BLOCK_SIZE_M=256 BLOCK_SIZE_N=128 GROUP_M=8
    warps=8 stages=2): 3.75 + 2.49 = 6.24 ms.

So on the whole MoE layer this is 1.04x vs the deployed triton config (a wash)
and 1.27x vs the best triton config. It is 1.47x slower on gate_up but 1.03x
FASTER on the down projection. It loses worst where it was meant to win:
triton is 2.0x faster at 16 rows/expert and 1.95x at 32.

Hardware ceilings for reading those percentages (measured, see B5_RESULT.md):
bf16 ~235 TFLOPS (hipBLASLt reaches 205), fp8 ~470 TFLOPS (pure-MFMA loop
reaches 393). The older "bf16 peak 90.6 / fp8 peak 22.6" figures are wrong --
they were one bad triton config, not a ceiling, and they made 158-183 TFLOPS
look impossible when it is not.

Why it loses (measured, not guessed): it emits the real fp8 MFMA
(__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8) and its pure-MFMA issue rate
matches triton's (388 vs 393 TF at 128x128), so the instruction is not the
problem. A is NOT register-resident -- it is in LDS via alloc_shared, and at
256 rows/expert 1.31 MB could not fit in registers or LDS anyway. What it
re-reads is B: runtime is exactly linear in the subtile count (2.33/4.29/8.09 ms
at 1/2/4 subtiles) because every subtile re-streams the expert weight tile. The
generated HIP also declares ~360 registers/thread under launch_bounds(256) and
indexes a_scale with the dynamic k variable, so it spills, and it runs 1 CTA/CU
at 4 wavefronts with 2 stages and two __syncthreads() per k iteration. Do not
adopt for speed; closing the gap needs tilelang HIP backend changes, not edits
to this file.
"""

import functools

import torch

FP8_DTYPE = "float8_e4m3fnuz"  # gfx942 is fnuz
ACC_DTYPE = "float32"

# LDS on gfx942 is 64 KB per CU. fp8 shared tiles cost
# (BLOCK_M + BLOCK_N) * BLOCK_K bytes per stage, so BLOCK_M=128 BLOCK_N=128
# BLOCK_K=128 is exactly the largest tile that still gets 2 stages.
DEFAULT_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 128,
    "k_pack": 2,
    "num_stages": 2,
    "threads": 256,
    # Put the row subtile in the grid instead of looping over it inside the CTA.
    # More CTAs (better latency hiding) at the cost of re-reading the weights
    # once per subtile either way. Benchmarked slower than the serial loop.
    "grid_subtile": False,
}


def _jit(E, R, N, K, config):
    import tilelang
    import tilelang.language as T

    BM = config["BLOCK_M"]
    BN = config["BLOCK_N"]
    BK = config["BLOCK_K"]
    k_pack = config["k_pack"]
    stages = config["num_stages"]
    threads = config["threads"]
    grid_subtile = config["grid_subtile"]
    KS = K // 128
    NSUB = R // BM
    # BLOCK_K is the copy/MFMA granularity while the scale is per 128 K, so 128
    # gives exactly one promotion per scale block (same reduction order as the
    # triton kernel) and 64 gives two. 256 is deliberately unsupported: one
    # promotion per 256-K would have to fold two different scale blocks
    # together, and (BLOCK_M + BLOCK_N) * 256 bytes per stage already blows the
    # 64 KB LDS at every tile size that is worth running.
    assert BK in (64, 128), BK
    assert R % BM == 0, (R, BM)

    @tilelang.jit(out_idx=[-1])
    def kernel(E: int):
        @T.prim_func
        def main(
            A: T.Tensor((E, R, K), FP8_DTYPE),
            B: T.Tensor((E, N, K), FP8_DTYPE),
            As: T.Tensor((E, R, KS), ACC_DTYPE),
            Bs: T.Tensor((E, N // 128, KS), ACC_DTYPE),
            nrows: T.Tensor((E,), "int32"),
            C: T.Tensor((E, R, N), ACC_DTYPE),
        ):
            if grid_subtile:
                with T.Kernel(E, NSUB, T.ceildiv(N, BN), threads=threads) as (e, s, bn):
                    A_shared = T.alloc_shared((BM, BK), FP8_DTYPE)
                    B_shared = T.alloc_shared((BN, BK), FP8_DTYPE)
                    partial = T.alloc_fragment((BM, BN), ACC_DTYPE)
                    acc = T.alloc_fragment((BM, BN), ACC_DTYPE)
                    a_scale = T.alloc_fragment((BM, KS), ACC_DTYPE)
                    b_scale = T.alloc_fragment((KS,), ACC_DTYPE)

                    for kk in T.Parallel(KS):
                        b_scale[kk] = Bs[e, bn * BN // 128, kk]
                    for i, kk in T.Parallel(BM, KS):
                        a_scale[i, kk] = As[e, s * BM + i, kk]

                    T.clear(acc)
                    for k in T.Pipelined(K // BK, num_stages=stages):
                        T.copy(B[e, bn * BN, k * BK], B_shared)
                        T.copy(A[e, s * BM, k * BK], A_shared)
                        T.clear(partial)
                        T.gemm(A_shared, B_shared, partial, transpose_B=True, k_pack=k_pack)
                        for i, j in T.Parallel(BM, BN):
                            acc[i, j] += (
                                partial[i, j]
                                * a_scale[i, k * BK // 128]
                                * b_scale[k * BK // 128]
                            )
                    T.copy(acc, C[e, s * BM, bn * BN])
            else:
                with T.Kernel(E, T.ceildiv(N, BN), threads=threads) as (e, bn):
                    A_shared = T.alloc_shared((BM, BK), FP8_DTYPE)
                    B_shared = T.alloc_shared((BN, BK), FP8_DTYPE)
                    partial = T.alloc_fragment((BM, BN), ACC_DTYPE)
                    acc = T.alloc_fragment((BM, BN), ACC_DTYPE)
                    a_scale = T.alloc_fragment((BM, KS), ACC_DTYPE)
                    b_scale = T.alloc_fragment((KS,), ACC_DTYPE)

                    for kk in T.Parallel(KS):
                        b_scale[kk] = Bs[e, bn * BN // 128, kk]

                    # Derived at runtime, so every expert's real rows are covered
                    # however uneven the routing is.
                    for s in T.serial((nrows[e] + BM - 1) // BM):
                        for i, kk in T.Parallel(BM, KS):
                            a_scale[i, kk] = As[e, s * BM + i, kk]
                        T.clear(acc)
                        for k in T.Pipelined(K // BK, num_stages=stages):
                            T.copy(B[e, bn * BN, k * BK], B_shared)
                            T.copy(A[e, s * BM, k * BK], A_shared)
                            T.clear(partial)
                            T.gemm(A_shared, B_shared, partial, transpose_B=True,
                                   k_pack=k_pack)
                            for i, j in T.Parallel(BM, BN):
                                acc[i, j] += (
                                    partial[i, j]
                                    * a_scale[i, k * BK // 128]
                                    * b_scale[k * BK // 128]
                                )
                        T.copy(acc, C[e, s * BM, bn * BN])

        return main

    return kernel(E)


@functools.lru_cache(maxsize=32)
def _cached(E: int, R: int, N: int, K: int, **config):
    cfg = {**DEFAULT_CONFIG, **config}
    return _jit(E, R, N, K, cfg)


def thin_grouped_fp8_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    nrows: torch.Tensor,
    **config,
) -> torch.Tensor:
    """``A[e, :nrows[e]] @ B[e].T`` with [128, 128] block scales, grouped.

    A     [E, R, K] fp8   - per-expert rows, R padded up to a multiple of BLOCK_M
    B     [E, N, K] fp8   - expert weights
    As    [E, R, K/128] fp32, Bs [E, N/128, K/128] fp32
    nrows [E] int32       - real row count per expert; rows >= nrows[e] are
                            computed from padded input and must be discarded by
                            the caller, exactly like the triton path does.
    returns [E, R, N] fp32
    """
    E, R, K = A.shape
    N = B.shape[1]
    assert K % 128 == 0 and N % 128 == 0, (K, N)
    cfg = {**DEFAULT_CONFIG, **config}
    assert R % cfg["BLOCK_M"] == 0, (R, cfg["BLOCK_M"])
    return _cached(E, R, N, K, **config)(A, B, As, Bs, nrows)
