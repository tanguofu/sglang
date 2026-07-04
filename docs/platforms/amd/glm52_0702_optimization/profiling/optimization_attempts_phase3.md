# Optimization Attempts — Phase 3 (2026-07-04)

## Summary

Two kernel-level optimizations were attempted based on profiling analysis.
Both were reverted due to negative net impact on MTP accept rate.

## Attempt 1: Last Layer AllReduce Fusion

**Goal**: Enable AITER allreduce fusion for the last decoder layer and draft model layer.

**Root Cause**: `should_fuse_mlp_allreduce_with_next_layer` returns False for
`is_last_layer=True`, forcing non-fused allreduce (AITER 2stage, 485us avg)
instead of fused 1stage (12us avg).

**Changes**:
- `communicator.py`: Removed `(not self.is_last_layer)` condition
- `deepseek_v2.py`: Added deferred allreduce handling in final RMSNorm via
  `forward_with_allreduce_fusion`

**Result**: REVERTED
- Accept rate: 76-82% → 11-23%
- Throughput: 178 tok/s → 83 tok/s (-53%)
- Draft model CUDA graph capture doesn't handle final norm fusion correctly

## Attempt 2: MoE Preshuffle On

**Goal**: Set `is_shuffled=True` on FP8 MoE weights after `shuffle_weight`.

**Root Cause**: FP8 MoE weight processing shuffles weights via `shuffle_weight((16,16))`
but never sets `is_shuffled=True`. AITER uses `preshuffle_off` kernel with warning.

**Changes**:
- `fp8.py`: Added `is_shuffled=True` after `shuffle_weight` in both
  `_is_fp8_fnuz` and `elif _use_aiter` paths

**Result**: REVERTED
- AITER correctly loads `preshuffle_on` kernel
- Output correctness verified (2+2=4, Paris, 10*10=100)
- Accept rate: 76-82% → 42-51%
- Throughput: 178 tok/s → 134 tok/s (-25%)
- Per-forward throughput slightly improved (56.3 vs 54.4 forward/s)
- But MTP accept rate drop offsets MoE GEMM improvement

## Key Findings

1. MTP is highly sensitive to MoE kernel numerical precision changes
2. Draft model (NextN) `is_last_layer=True` prevents allreduce fusion
3. AITER preshuffle_off was "accidentally correct" on shuffled weights
4. Per-forward throughput can improve while overall throughput decreases
   due to MTP accept rate sensitivity

## Next Steps

1. GPU utilization improvement (48.6% → target 70%+)
2. Draft model-specific allreduce fusion (bypass is_last_layer check)
3. Partial logits to avoid 2stage allreduce
4. AITER C++ 1stage/2stage threshold investigation
