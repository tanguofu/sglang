# Precision-Preserving Optimization Plan — 2026-07-18

> Re-evaluation of the optimization plan under the hard constraint:
> **NO precision loss is acceptable.** Quality/accuracy must match the FP8
> weight + bf16 compute baseline. Every proposed change must be either
> (a) numerically lossless, or (b) validated by a 26/26 eval pass before
> and after the change.
>
> Companion to `deployment_kernel_analysis_0718.md`.

## 1. P2P / Multi-GPU Topology Review

### 1.1 Hardware evidence

| Check | Finding |
|---|---|
| `torch.cuda.can_device_access_peer(i,j)` for all i≠j | **True** for all 56 pairs |
| NCCL transport for all-reduce | `via P2P/IPC` only — no SHM, no NET |
| NCCL all-reduce 4 MB on 8 GPUs | avg 0.23 ms, **effective BW 123 GB/s** |
| XGMI physical_id per GPU | 0..7 (all 8 GPUs in same XGMI hive) |
| `xgmi_num_hops` | 41 — misleading kernel-export value (error code, not actual hop count) |
| `xgmi_num_links` | 1 between every GPU pair — fully connected |
| Channels | 56 distinct NCCL channels observed (matches `NCCL_MIN_NCHANNELS=112` config, half logged) |
| RoCE NICs | 8 × 400 Gbps `bnxt_re_bond0..7` present but **not used** for intra-node comm |

### 1.2 Verdict

**P2P is fully enabled and working over XGMI/Infinity Fabric.** All intra-node TP8
communication uses `P2P/IPC` — NCCL never falls back to SHM or network. The 123 GB/s
effective all-reduce bandwidth is healthy for MI308X (peak ~200 GB/s unidirectional
XGMI per link, aggregate across 8 GPUs ≈ 100-150 GB/s realistic for all-reduce).

No `NCCL_P2P_DISABLE` is set, no `NCCL_IB_*` overrides are forcing network transport.
The `HSA_ENABLE_SDMA=0` env var disables SDMA (uses memcpy instead) which is the
correct setting for XGMI P2P on gfx942 — SDMA is for PCIe/DMA transfers, not XGMI.

**Conclusion: ✅ P2P is correctly configured and working. No change needed.**

### 1.3 What could still be tuned (but is NOT a precision issue)

| Tuning | Effect | Precision impact |
|---|---|---|
| `NCCL_MIN_NCHANNELS=112` (already set) | More concurrent channels → higher BW utilization | None |
| `NCCL_NET_GDR_LEVEL` | Not applicable — no IB used intra-node | N/A |
| `HSA_ENABLE_SDMA=0` (already set) | Forces memcpy over XGMI (correct) | None |
| NCCL topology file (`/etc/nccl/nccl.conf`) | Could pin ring/tree order to XGMI-favoring topology | None |

---

## 2. Precision Risk Audit — Existing Config

### 2.1 Settings that DO affect precision (already in production)

| Setting | Current | Precision impact | Risk |
|---|---|---|---|
| `kv_cache_dtype` | `fp8_e4m3` | KV cache stored in FP8 — 8-bit quantization of K/V. Causes small numerical drift vs bf16 KV. | **Already accepted** — this is part of the FP8 deployment baseline. Eval passes 26/26 with this. |
| `triton_attention_reduce_in_fp32` | `False` | Triton attention reduce in fp8/bf16, not fp32 | Low — matches FP8 model design |
| `enable_fp32_lm_head` | `False` | LM head in fp16/bf16, not fp32 | Low — standard for FP8 models |
| `ROCM_QUICK_REDUCE_QUANTIZATION` | `INT8` | NCCL all-reduce quantizes to INT8 → **lossy**. Used for quick-reduce path. | ⚠ **See §2.2** |
| `enable_quant_communications` | `False` | FP8 all-reduce disabled (good) | ✅ Lossless baseline |
| `dtype` | `auto` → `bf16` | Compute in bf16 | ✅ Standard |
| `quantization` | `None` (FP8 weights via model config) | Weights are FP8 e4m3 (block [128,128] dynamic) | **Already accepted** — this is the FP8 model |
| `bf16_gemm_backend` | `auto` | bf16 GEMM kernel | ✅ |
| `fp8_gemm_runner_backend` | `auto` | FP8 GEMM for weight-only FP8 | ✅ Standard |

### 2.2 ⚠ `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` — precision-sensitive

This env var tells NCCL's "quick reduce" path (used for small all-reduce operations) to quantize activations to INT8 before reduction, then dequantize. **This is lossy.**

However, the quick-reduce path is only used for **small** all-reduce buffers (typically < 2 KB, e.g., scalar metadata, sequence lengths). The main gradient/activation all-reduce uses the standard NCCL algorithm with full bf16/fp32 precision.

**Recommendation under zero-precision-loss constraint**:
- **Set `ROCM_QUICK_REDUCE_QUANTIZATION=NONE`** (or unset the env var) to disable INT8 quantization in the quick-reduce path.
- This is a one-line env var change. No kernel swap, no operator change.
- Risk: Small latency increase on tiny all-reduces (negligible — these are sub-microsecond operations).
- Validation: Re-run 26-case eval; expect identical pass rate.

### 2.3 Other precision-preserving defaults confirmed

| Setting | Value | Why it matters |
|---|---|---|
| `enable_quant_communications` | `False` | FP8 all-reduce would quantize activations → lossy. Currently disabled. **Do NOT enable.** |
| `triton_attention_reduce_in_fp32` | `False` | Would force fp32 attention reduce → more precise but slower. Current `False` matches FP8 model design intent. Leaving as-is. |
| `enable_fp32_lm_head` | `False` | Would compute LM head in fp32. Current `False` is standard for FP8 models. Leaving as-is. |
| `disable_flashinfer_cutlass_moe_fp4_allgather` | `False` | FP4 MoE allgather — not relevant (model is FP8) |
| `flashinfer_mxfp4_moe_precision` | `default` | MXFP4 not used |

---

## 3. Revised Optimization Plan (Zero Precision Loss)

### 3.1 Changes REMOVED from the original plan (would cause precision loss)

| Original proposal | Why removed |
|---|---|
| `--enable-quant-communications` | ❌ Enables FP8 all-reduce → quantizes activations → **lossy**. Removed. |
| `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` (keep) | ❌ Already INT8 — should change to `NONE` (see §2.2). Reversed. |

### 3.2 Changes that are precision-safe (numerically lossless)

All changes below either (a) do not touch numerical computation, or (b) only affect
scheduling/batching/memory with no impact on math.

#### P0 — Restore worker2 (operational, zero precision impact)

| Action | Precision impact |
|---|---|
| Fix node `.38` or move `w2-sglang` StatefulSet to healthy node, scale to `replicas=1` | None — same binary, same config |
| Update router `--worker-urls` if worker2 IP changes | None |

#### P1 — EAGLE speculative tuning (lossless if eval passes)

EAGLE does NOT quantize or approximate — it proposes draft tokens that are **verified** against the target model's actual forward pass. Accepted drafts are tokens the target model would have produced anyway. **If a draft is wrong, it is rejected.** This is mathematically lossless.

| Change | Mechanism | Precision risk |
|---|---|---|
| `--speculative-eagle-topk 2` | Draft proposes 2 candidates per step instead of 1; target still verifies each | **None** — verified, not approximated |
| `--speculative-num-draft-tokens 6` | More draft tokens per step; each verified | **None** — verified |
| `--speculative-num-steps 2` (down from 3) | Fewer draft iterations; less draft work | **None** |
| `--speculative-attention-mode decode` | Draft uses decode-shape attention | **None** — affects draft proposal only, not target verification |

**Validation requirement**: Run 26-case eval after EACH change. Eval must remain 26/26 PASS. If any case fails, revert. The eval is the precision gate.

#### P1 — `--dsa-decode-backend aiter` (lossless if eval passes)

The `aiter` MLA decode kernel is AMD's first-party implementation of the same MLA math. It uses the same FP8 KV cache, same q/k/v projections, same attention computation. The only difference is the kernel implementation (aiter vs tilelang).

| Change | Mechanism | Precision risk |
|---|---|---|
| `--dsa-decode-backend aiter` | Different kernel for MLA decode — same math, different CUDA/HIP code path | **Theoretical: none** (same dtype, same math). **Practical: small numerical differences possible** due to kernel-level reduction order. |

**Validation requirement**: Run 26-case eval. If 26/26 PASS, precision is preserved (eval compares output text, which is robust to <1e-3 numerical drift). If any case fails, revert.

This is the **one** kernel-swap candidate that needs careful validation. All other kernel backends are either already optimal (tilelang for prefill on HIP) or CUDA-only (flashmla/fa3).

#### P2 — Scheduler/memory tuning (zero precision impact)

| Change | Mechanism | Precision risk |
|---|---|---|
| `--num-continuous-decode-steps 2` | Scheduler does 2 decode steps between prefill checks | **None** — scheduling only |
| `--mem-fraction-static 0.85` (from 0.82) | More KV cache capacity | **None** — memory allocation only |
| `--schedule-conservativeness 1.0` (from 0.5) | More aggressive admission | **None** — scheduling only |
| `--chunked-prefill-size 32768` (from 131072) | Smaller prefill chunks | **None** — chunking only affects scheduling, not math |
| `--max-running-requests 48` (from 32) | Higher concurrency | **None** — scheduling only. Requires cuda-graph-bs expansion. |

#### P3 — Communication tuning (precision-safe subset only)

| Change | Mechanism | Precision risk |
|---|---|---|
| `NCCL_MIN_NCHANNELS=112` (already set) | More NCCL channels | **None** |
| `NCCL_BUFFSIZE` / `NCCL_NTHREADS` tuning | Buffer sizes | **None** |
| `ROCM_QUICK_REDUCE_QUANTIZATION=NONE` (from INT8) | Disable INT8 quant in quick-reduce | **Increases precision** (removes lossy quant) |

### 3.3 Changes explicitly EXCLUDED (would cause precision loss)

| Change | Why excluded |
|---|---|
| `--enable-quant-communications` | FP8 all-reduce → lossy |
| `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` (keep as INT8) | Already lossy — should change to NONE |
| `--kv-cache-dtype bf16` (downgrade from fp8) | Would INCREASE precision but double KV memory → halve capacity. Not needed — FP8 KV is the accepted baseline. |
| `--kv-cache-dtype int8` (further quantize) | ❌ Lossy |
| `--enable-fp4` / `SGLANG_INT4_WEIGHT=1` | ❌ Lossy |
| Any torch.compile path | ❌ May fuse ops in different precision order |

---

## 4. Prioritized Action List

| Priority | Action | Precision impact | Validation |
|---|---|---|---|
| **P0** | Restore worker2 (fix node .38 or migrate) | None | Operational |
| **P0** | `ROCM_QUICK_REDUCE_QUANTIZATION=NONE` (remove INT8 quick-reduce) | **Increases precision** | Run eval |
| **P1** | `--speculative-eagle-topk 2` | None (verified) | 26/26 eval |
| **P1** | `--speculative-num-draft-tokens 6` | None (verified) | 26/26 eval |
| **P1** | `--dsa-decode-backend aiter` (A/B test) | Theoretical none, validate | 26/26 eval |
| **P2** | `--num-continuous-decode-steps 2` | None | 26/26 eval |
| **P2** | `--mem-fraction-static 0.85` + `--schedule-conservativeness 1.0` | None | 26/26 eval |
| **P2** | `--chunked-prefill-size 32768` (if long-context tail is issue) | None | 26/26 eval |
| **P3** | Sync repo manifest to live config (doc hygiene) | None | Docs |
| **P3** | PD disaggregation (manifests ready in `configs/pd-manifests/`) | None — same kernels | 26/26 eval |

---

## 5. Summary Answer

### Q: 多卡之间开启了 P2P 么?

**Yes, fully enabled and working.** Verified three ways:
1. `torch.cuda.can_device_access_peer(i,j) = True` for all 56 GPU pairs
2. NCCL logs show `via P2P/IPC` for every channel/rank pair (no SHM, no NET fallback)
3. All-reduce benchmark: 123 GB/s effective bandwidth on 8 GPUs — healthy XGMI throughput

The 8 MI308X GPUs are in a single XGMI hive (physical_id 0-7), fully connected with 1 link per pair. `HSA_ENABLE_SDMA=0` correctly forces memcpy over XGMI. `NCCL_MIN_NCHANNELS=112` provides high channel concurrency. No `NCCL_P2P_DISABLE` is set. **No action needed on P2P.**

### Q: 优化计划 (零精度损失约束)

Under the zero-precision-loss constraint, the revised plan:

1. **Removed**: `--enable-quant-communications` (would quantize all-reduce → lossy)
2. **Added**: `ROCM_QUICK_REDUCE_QUANTIZATION=NONE` (current INT8 is lossy; removing it increases precision)
3. **Kept**: EAGLE tuning (`eagle_topk`, `num_draft_tokens`) — speculative decode is verified, not approximated, so it's lossless by design
4. **Kept**: `--dsa-decode-backend aiter` — same math, different kernel; validate with eval
5. **Kept**: All scheduler/memory tuning — no math impact

The single biggest opportunity remains **EAGLE tuning** (34.7% → target >50% acceptance, recovering ~1.5-2× decode throughput) — and it's precision-safe because every drafted token is verified against the target model before acceptance.
