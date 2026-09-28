# Findings

## Current DFlash2 state
- Canary decode image: `20260927r`
- Prefill image: `20260927l`
- Root cause of TP7 hang: `window_kv_indptr[-1].item()` forced GPU scalar readback.
- Fix: compute window length on CPU and avoid `.item()`.
- Additional fix: replace host-to-device `torch.tensor(...)` with GPU-side `torch.full` in DSA target-verify metadata.

## Model compatibility
- `glm53-flash`: `Glm5Next`, hidden size 4096.
- `glm53-fp8`: `GlmMoeDsa`, hidden size 6144.
- Current DFlash2 draft model hidden size: 4096.
- DFlash2 borrows the target model's `lm_head`, so draft hidden size must match target hidden size.

## Production `glm53-fp8` state
- Decode uses `NEXTN` with 2 steps and 4 draft tokens, DSA + tilelang, and CUDA graph.
- Production image `v0924-src-mooncake-fix` already contains the internal ROCm/DSA patch set.
- Runtime optimizations already enabled:
  - `SGLANG_OPT_USE_TOPK_V2=1`
  - `SGLANG_ROCM_DSA_INDEXER_PROJECTION_FUSION=1`
  - `SGLANG_ROCM_USE_CK_GEMM_GFX942=1`
  - AITER tuned GEMM/FMOE configs
  - Fused DSA metadata on HIP
  - Absorb no-op skip and target-verify q-cat optimization
  - QuickReduce saturation fix
  - gfx942 CU-count fix
- `SGLANG_PD_HOST_STAGING=0`; host-staging code exists in the image but is disabled.

## Portability classification
1. **Direct merge**: `dsa_backend.py` target-verify metadata fix that replaces host-to-device `torch.tensor(...)` with GPU-side `torch.full`. This path is active for `glm53-fp8` NEXTN target verify.
2. **Conditional merge**: Mooncake host staging with selective host-to-GPU copy. Production already has an older host-staging implementation behind a disabled switch; benchmark before enabling.
3. **No current effect**: DSA KPool ROCm support. `glm53-fp8` has `index_kpool=1`, so the KPool path is inactive.
4. **Not applicable**: Triton sliding-window no-`.item()` fix; `glm53-fp8` uses DSA, not Triton sliding window.
5. **Not applicable**: DFlash disaggregation support in `spec_info.py`; `glm53-fp8` uses NEXTN.
6. **Blocked**: DFlash2 draft model reuse. Hidden size 4096 cannot directly pair with `glm53-fp8` hidden size 6144 because DFlash2 borrows the target `lm_head`.

## Recommended merge order
1. Cherry-pick only the DSA target-verify `torch.full` fix into `glm53-fp8`.
2. Benchmark host staging on canary; enable only if it improves 200K TTFT/TFOT without regression.
3. Keep KPool, Triton no-item, and DFlash disaggregation changes out of the `glm53-fp8` port.
