# GLM-5.2 MI355X Decode Profile & Optimization Summary (0706 Image)

## Profile Results (torch profiler, 3 decode steps, TP=0)

### Total kernel time: 34.1ms (3 steps)

### Kernel breakdown by category:
| Category | Time (ms) | % | Calls |
|----------|-----------|---|-------|
| Communication (reduce/allgather) | 12.87 | 37.8% | 174 |
| MoE GEMM (fmoe) | 9.86 | 29.0% | 388 |
| CK GEMM (blockscale) | 2.51 | 7.4% | 260 |
| Quantization | 2.36 | 6.9% | 519 |
| Elementwise/Copy | 1.94 | 5.7% | 460 |
| Other (main_kernel, softmax) | 2.21 | 6.5% | 229 |
| Other GEMM | 1.13 | 3.3% | 118 |
| RMSNorm | 0.69 | 2.0% | 156 |
| Attention/DSA | 0.49 | 1.5% | 130 |

### Step breakdown:
- **Prefill** (Step 1): 26.7ms kernel, 28.5ms wall (2155 kernels)
- **Draft model** (Step 5): 0.5ms kernel, 4.3ms wall (42 kernels) — CPU/launch overhead inflated by profiling
- **Target verify** (Step 6): 2.3ms kernel, 3.8ms wall (126 kernels) — 61% communication
- **Target decode** (Step 7): 4.0ms kernel, 5.4ms wall (84 kernels) — 90% communication

### Top kernels:
1. `aiter::fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_32x256` — 8.47ms (24.8%)
2. `aiter::reduce_scatter_cross_device_store` — 6.20ms (18.2%)
3. `aiter::cross_device_reduce_1stage` — 3.33ms (9.8%)
4. `ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle` — 2.44ms (7.2%)
5. `aiter::allreduce_fusion_kernel_1stage` — 1.71ms (5.0%)

## Optimization Attempts

### Opt 1: AITER GEMM tuning — NEGLIGIBLE
- 16 bf16 GEMM fallbacks (M=1489/1493, N=256, K=6144) at 0.024ms each = 0.4ms total
- AITER already returns `torch` as solution for these shapes
- Not worth tuning

### Opt 2: Fused free path PR (commits 20da85b + 47220e4d) — N/A
- Worker uses DSA backend with `hybrid_swa=False`
- SWA allocator not in use, fused free path PR targets SWA allocator
- Not applicable

### Opt 3a: num_continuous_decode_steps=2 + scheduler_recv_interval=4 — NO IMPROVEMENT
- decode_short c1: 169.2 (baseline 170.5, -0.8%)
- decode_short c8: 914.4 (baseline 921.4, -0.8%)
- decode_2k c8: 703.7 (baseline 817.3, -13.9%)
- medium_ctx c4: 411.0 (baseline 490.1, -16.1%)
- Reverted

### Opt 3b: SGLANG_ROCM_USE_MULTI_STREAM=1 — SIGNIFICANT REGRESSION
- Enables alt_stream for MoE to overlap shared expert with all-reduce
- decode_short c1: 130.8 (baseline 170.5, -23.3%)
- decode_2k c1: 124.5 (baseline 161.3, -22.8%)
- qa_thinking c1: 121.8 (baseline 164.7, -26.0%)
- Stream synchronization overhead > overlap benefit at low concurrency
- Reverted

## Key Findings
1. **Communication is the fundamental bottleneck** (90% of decode kernel time)
2. **AITER AllReduce Fusion is already enabled** — custom all-reduce is optimized
3. **CUDA graph captures most compute** — communication kernels are outside graph
4. **Low-concurrency (c1) is communication-bound** — TP=8 all-reduce dominates
5. **Multi-stream overlap hurts** — synchronization overhead > overlap benefit
6. **MoE GEMM is fast** — not a bottleneck
7. **Attention is very small** — DSA tilelang backend is efficient

## Baseline Performance (3/4/1 + skip-warmup, after 2x warmup)
| Test | Worker (0706) |
|------|---------------|
| decode_short c1 | 170.5 |
| decode_short c8 | 921.4 |
| decode_2k c1 | 161.3 |
| decode_2k c8 | 817.3 |
| qa_thinking c1 | 164.7 |
| medium_ctx c4 | 490.1 |

MTP: accept_rate=0.762, accept_length=3.29, decode_tps=183 (median)
