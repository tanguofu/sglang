# Timeline Analysis & CUDA Graph Optimization — Phase 4 (2026-07-04)

## Steady-State Timeline Analysis (after warmup)

### GPU Utilization
- Time span: 1305.4ms
- GPU busy: 1026.2ms (78.6%)
- GPU idle: 279.2ms (21.4%)
- Kernel count: 141,587
- Graph launches: 120 (avg 10.15ms per step)

### #1 Bottleneck: hipMemcpyWithStream — 904ms
- 121 calls, avg 7.4ms per call
- GPU actual copy: only 6.6ms (2239 gpu_memcpy events, avg 2.9us)
- Root cause: Synchronization barrier — CPU blocks waiting for GPU to finish
- Source: AMD copy_to_cpu runs synchronously on forward stream
- NVIDIA uses separate copy_stream (async, overlapped)

### #2: Non-graph hipLaunchKernel — 101ms
- 8,566 calls, avg 3.6us (top 5: 5.5-27.5ms JIT compilation)
- Draft model operations not captured in CUDA graph

### #3: Graph launch overhead — 26.7ms
- 120 calls, avg 222.8us (efficient)

### Non-graph Operation Patterns
| Before kernel | After kernel | Gaps | Total |
|---|---|---|---|
| index_elementwise | _get_last_loc_safe | 60 | 25.1ms |
| _fused_append_shared | opus_moe_sorting | 59 | 10.5ms |
| _gather_rows | assign_draft_cache_locs | 41 | 6.1ms |
| CatArrayBatchedCopy | main_kernel (attn) | 43 | 4.5ms |
| _batched_gemm_a8w8 | _fused_qk_rope | 29 | 3.1ms |

## Optimization Attempt 3: Copy Stream for HIP — REVERTED
- Removed _is_hip special case to use copy_stream on AMD
- Result: Accept rate 76% → 32-53%, throughput 178 → 125 tok/s
- Root cause: MTP race condition with copy stream
- Reverted

## Kernel Performance Assessment
- All kernel selections already optimized (AITER tuned, rocBLAS tuned)
- MoE: 31.3%, AllReduce: 21.8%, Attention: 9.7%, GEMM: 9.8%, Quant: 9.6%
- Main bottleneck is scheduling overhead, not individual kernel speed

## Key Conclusion
The hipMemcpyWithStream synchronization barrier (904ms) is the primary
optimization target, but it cannot be fixed with copy_stream due to MTP
incompatibility. Future work should focus on:
1. MTP-compatible async D2H transfer
2. Extending CUDA graph to capture draft model operations
3. Kernel fusion to reduce non-graph kernel launches
