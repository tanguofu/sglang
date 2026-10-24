"""Thin-M grouped fp8 GEMM for DS4.1 MoE on gfx942.

Decode on MI308X routes ~192 rows (batch x top_k) over 48 local experts, i.e.
a handful of rows per expert. The triton path pads every hit expert up to
BLOCK_SIZE_M (64), so it runs on ~3072 rows and re-reads each expert's weight
tile for every padded block. This kernel instead puts one CTA on each
(expert, N tile), streams that expert's weight tile exactly once, and loops
over the expert's real rows, so the cost is the weight stream and nothing else.

Numerics follow the reference the MoE path already uses: fp8 products
accumulated in fp32, with the [128, 128] block scales applied once per K block
(``acc += partial * a_s[m, kk] * b_s[n, kk]``). No split-K, no atomics, fixed
K order, so a row is bitwise identical alone or in any batch.

Status: measured, NOT wired into any MoE runner. Do not call it from a
production path until an end-to-end run confirms the win -- on this platform
every microbench win has failed to transfer (triton BLOCK_M=16: -16% standalone
but -7% end-to-end; triton BLOCK_N=64/warps=4: -17% standalone, -9% end-to-end).

Verified on gfx942 (2026-10-24, tilelang a55a8230), gate_up E=48 N=4608 K=5120
(1.13 GB of weights, so 1.0 TB/s ~= 1.13 ms):
  * numerics: matches an fp32 reference to rel 2.4e-7 (fp32 rounding) and is
    bit-identical across M=1/2/4 and under CUDA graph;
  * R=16 (rows per expert): 1.054 ms -> 1.07 TB/s, vs triton's 1.378 ms (0.82
    TB/s). BUT R is a compile-time constant here, so an expert with more than R
    rows silently loses rows -- production needs the multi-subtile variant;
  * R=64 (safe for any expert occupancy): 1.876 ms -> 0.60 TB/s, i.e. 36%
    SLOWER than triton.
  * pure weight streaming ceiling with no GEMM: 0.670 ms (1.69 TB/s).
"""

import functools

import torch

FP8_DTYPE = "float8_e4m3fnuz"  # gfx942 is fnuz
ACC_DTYPE = "float32"

# Measured best of the LDS-safe configs at R=16. BLOCK_N=128 lands at 0.73 TB/s
# and BLOCK_N=256 does not fit in the 64 KB LDS at all.
DEFAULT_CONFIG = {"BLOCK_N": 64, "BLOCK_K": 128, "k_pack": 2, "num_stages": 2, "threads": 256}


def _jit(R, N, K, config):
    import tilelang
    import tilelang.language as T

    BN = config["BLOCK_N"]
    BK = config["BLOCK_K"]
    k_pack = config["k_pack"]
    stages = config["num_stages"]
    threads = config["threads"]
    KS = K // 128
    NB = N // 128

    @tilelang.jit(out_idx=[-1])
    def kernel(E: int):
        @T.prim_func
        def main(
            A: T.Tensor((E, R, K), FP8_DTYPE),
            B: T.Tensor((E, N, K), FP8_DTYPE),
            As: T.Tensor((E, R, KS), ACC_DTYPE),
            Bs: T.Tensor((E, NB, KS), ACC_DTYPE),
            C: T.Tensor((E, R, N), ACC_DTYPE),
        ):
            with T.Kernel(E, T.ceildiv(N, BN), threads=threads) as (e, bn):
                A_shared = T.alloc_shared((R, BK), FP8_DTYPE)
                B_shared = T.alloc_shared((BN, BK), FP8_DTYPE)
                partial = T.alloc_fragment((R, BN), ACC_DTYPE)
                acc = T.alloc_fragment((R, BN), ACC_DTYPE)
                a_scale = T.alloc_fragment((R, KS), ACC_DTYPE)
                b_scale = T.alloc_fragment((KS,), ACC_DTYPE)

                for kk in T.Parallel(KS):
                    b_scale[kk] = Bs[e, bn * BN // 128, kk]
                for i, kk in T.Parallel(R, KS):
                    a_scale[i, kk] = As[e, i, kk]

                T.clear(acc)
                for k in T.Pipelined(K // BK, num_stages=stages):
                    T.copy(B[e, bn * BN, k * BK], B_shared)
                    T.copy(A[e, 0, k * BK], A_shared)
                    T.clear(partial)
                    T.gemm(A_shared, B_shared, partial, transpose_B=True, k_pack=k_pack)
                    for i, j in T.Parallel(R, BN):
                        acc[i, j] += partial[i, j] * a_scale[i, k] * b_scale[k]
                T.copy(acc, C[e, 0, bn * BN])

        return main

    return kernel


@functools.lru_cache(maxsize=8)
def _cached(E: int, R: int, N: int, K: int, **config):
    cfg = {**DEFAULT_CONFIG, **config}
    return _jit(R, N, K, cfg)(E)


def thin_grouped_fp8_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    **config,
) -> torch.Tensor:
    """``A[e] @ B[e].T`` with [128, 128] block scales, for the thin-M decode case.

    A  [E, R, K] fp8  - each expert's rows, R == BLOCK_M (16) padded rows
    B  [E, N, K] fp8  - expert weights
    As [E, R, K/128] fp32, Bs [E, N/128, K/128] fp32
    returns [E, R, N] fp32
    """
    E, R, K = A.shape
    N = B.shape[1]
    assert K % 128 == 0 and N % 128 == 0, (K, N)
    return _cached(E, R, N, K, **config)(A, B, As, Bs)
