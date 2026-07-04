# Next Optimization Opportunities

## Tier 1: High-Yield (>10%)

### ① Fix SBO (Single Batch Overlap) for HIP

**Expected gain**: 5-10% TPOT reduction (long context decode)

**Current blocker**: `alt_stream` is `None` in `deepseek_v2.py:1080` (`_pre_combine_hook`). The v6 patch only fixed `glm4_moe.py:1063` but the parent class `DeepseekV2MoEModel` still has `if _is_cuda else None` for `alt_stream` creation, and inner MLP layers receive `alt_stream=None`.

**Fix approach**:
1. Patch `deepseek_v2.py` to create `alt_stream = torch.cuda.Stream()` on HIP (not just CUDA)
2. Ensure `alt_stream` is propagated to all inner `DeepseekV2DecoderLayer` → `DeepseekV2MLP` → MoE layers
3. Verify `alt_stream` is not `None` before `.wait_stream()` calls in `_pre_combine_hook`

**Risk**: Medium — requires careful propagation through the model hierarchy.

### ② NSA topk Reuse Across MTP Draft Steps

**Expected gain**: 5-10% draft compute reduction

**Current state**: `eagle_topk=1`, satisfies reuse condition. Kunlun芯 branch has implemented this (commit `b5d71a58b`).

**Fix approach**:
1. Cherry-pick the NSA topk reuse code from Kunlun芯 branch
2. Set `index_share_for_mtp_iteration: true` in model `config.json`
3. First draft step's `topk_indices` are reused by subsequent steps, skipping repeated indexer computation

**Risk**: Low — code cherry-pick + config patch.

### ③ MTP steps 3→4 (if accept rate holds)

**Expected gain**: ~10-15% additional decode throughput

**Current state**: steps=3, accept len=3.17-3.29 (72-80% accept rate). Theoretical max for steps=4 is 5.

**Verification needed**: Check if accept rate drops significantly at steps=4. If accept len ≥ 3.8, net gain is positive.

**Risk**: Medium — draft VRAM increases, accept rate may drop.

---

## Tier 2: Medium-Yield (5-10%)

### ④ EPLB with Expert Parallel

**Expected gain**: 3-5% (low concurrency); larger at high concurrency

**Current blocker**: EPLB requires `ep_size > 1`. Current deployment is TP-only.

**Fix approach**: Switch to TP+EP hybrid (e.g., tp_size=4, ep_size=2) to enable EPLB. This requires:
1. Verify model supports TP+EP combination
2. Adjust NCCL configuration for EP groups
3. Enable `--enable-eplb --eplb-rebalance-num-iterations 100 --eplb-min-rebalancing-utilization-threshold 0.7`

**Risk**: Medium — TP+EP topology change requires testing.

### ⑤ Prefill CUDA Graph Size Tuning

**Expected gain**: 2-5% prefill latency

**Current state**: `--cuda-graph-bs-prefill 4 8 16 32`. Actual prefill batch sizes may not match.

**Fix approach**: Profile actual prefill batch sizes and align CG capture to match. Add sizes like 1, 2, 6 if common.

**Risk**: Low — parameter change only.

### ⑥ Chunked Prefill Size Tuning

**Expected gain**: <2%

**Current state**: `--chunked-prefill-size 32768`. 90% prefill hit rate >90%, median new-token=384.

**Fix approach**: Reduce to 8192 for small-increment coding assistant pattern. Most prefill is cache-hit + small computation.

**Risk**: Low — parameter change only.

---

## Tier 3: Low-Yield (<5%) or Research

### ⑦ HiCache (L2 Cache)

**Expected gain**: <3%

**Current state**: Radix cache hit rate 99.8%, KV pool only 10.8% used. HiCache L2 only helps when KV pool is near full and eviction triggers.

**Decision**: Not worth enabling now. Revisit when concurrency increases and KV pool approaches capacity.

### ⑧ AITER GEMM Expansion

**Expected gain**: ~0%

**Current state**: BF16 tuned GEMM has 101,183 rows, FP8 MoE tuned has 2,177 rows. "not found tuned config" warnings = 0. Already fully covered.

### ⑨ Schedule Conservativeness Micro-tuning

**Expected gain**: <2%

**Current state**: 0.5. At low concurrency, scheduler has almost no contention, so this parameter has minimal impact.

### ⑩ Custom Triton Kernel for DSA Indexer (N=32, N=160)

**Expected gain**: 5-10% (if faster than torch native)

**Current state**: torch native is fastest for N=32/160. A custom Triton kernel optimized for these specific shapes could potentially be faster.

**Risk**: High — requires kernel development and benchmarking.

### ⑪ FP8 MoE Preshuffle

**Current warning**: `tuned config found but is_shuffled=False. Tuned kernels are optimized for preshuffled weights (preshuffle_on). Running with preshuffle_off may produce incorrect results.`

**Fix approach**: Enable preshuffle during weight loading. This may improve MoE kernel performance and fix correctness warnings.

**Risk**: Medium — requires weight preprocessing.

---

## Summary Priority Matrix

| Rank | Optimization | Expected Gain | Difficulty | Action |
|---|---|---|---|---|
| 1 | Fix SBO for HIP | 5-10% | Medium | Patch `deepseek_v2.py` alt_stream propagation |
| 2 | NSA topk reuse | 5-10% | Low | Cherry-pick + config |
| 3 | MTP steps 3→4 | 10-15% | Medium | Test accept rate |
| 4 | EPLB with TP+EP | 3-5% | Medium | Topology change |
| 5 | Prefill CG tuning | 2-5% | Low | Profile + adjust |
| 6 | FP8 preshuffle | Unknown | Medium | Weight preprocessing |
| 7 | Custom Triton N=32/160 | 5-10% | High | Kernel development |

Items 1-3 combined could yield **15-35% additional decode throughput** on top of current results.
