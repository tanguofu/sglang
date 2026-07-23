# GLM-5.2 2tp8 Deployment & Kernel Operator Analysis — 2026-07-18

> Snapshot of the running production (`glm52-2tp8.jmpti.woa.com`) deployment and a
> per-operator review of every kernel selection on AMD MI308X (gfx942).
> Source: live `get_server_info` + `metrics` from `sglang-glm52-2tp8-sglang-0`,
> pod env, manifest in `configs/pd-manifests/sglang-glm52-2tp8-manifest.yaml`,
> custom patches in `configs/sglang-custom-patches/`.

## 1. Hardware

| Component | Value |
|---|---|
| GPU | AMD Instinct MI308X (gfx942, 80 CUs, 1850 MHz, 192 GB HBM3) ×8 |
| CPU | AMD EPYC 9K84 96-Core ×2 sockets (NUMA 0/1) |
| Memory | ~2.1 TB host RAM |
| Topology | 8 GPUs, single node, SPX partition, NPS1 |
| Interconnect | Infinity Fabric (no NVLink/PCIe-only) |

## 2. Current Topology (live, observed 2026-07-18)

```
                aiagent.qq.com  (external HTTPS)
                         │
                  ┌──────▼──────┐
                  │  Istio EG   │  envoy LB 21.162.215.14:80
                  └──────┬──────┘
                         │  HTTPRoute glm52-2tp8.jmpti.woa.com
                         │
                  ┌──────▼──────┐
                  │   router    │  sglang_router, cache_aware
                  │  .38:30001  │  (hostNetwork, hostPort)
                  └──┬──────┬───┘
            ┌────────┘      └────────┐
            ▼                        ▼
     worker1 .103:30000       worker2 .38:30000
     (Running, TP8)            (StatefulSet replicas=0)
```

### 2.1 Critical issues found

| # | Severity | Issue | Evidence |
|---|---|---|---|
| 1 | **HIGH** | Node `21.234.170.38` is `NotReady,SchedulingDisabled` — likely MI308X SMC/MEC firmware bug (see `project_mi308x_gpu_coredump_firmware.md`). | `kubectl get nodes` shows `.38 NotReady`; `kubectl exec` to router pod on `.38` times out |
| 2 | **HIGH** | Production worker2 StatefulSet `sglang-glm52-2tp8-w2-sglang` is `replicas=0` — only **1 of 2** workers is actually serving. | `kubectl get statefulset ... w2-sglang` → `0/0` |
| 3 | **MED** | Router still has `--worker-urls http://21.234.170.38:30000` configured, so any prefix-cache affinity miss to `.38` will fail. | Router pod args inspect |
| 4 | **LOW** | Manifest in repo (`mem-fraction-static 0.88`, `chunked-prefill-size 32768`, `max-prefill-tokens 32768`, `watchdog-timeout 3600`) diverges from live config (`0.82`, `131072`, `131072`, `1200`). | Diff between `configs/pd-manifests/.../manifest.yaml` and live StatefulSet |
| 5 | **INFO** | VRAM usage is 93-94% per GPU at idle (worker loaded). With `mem-fraction-static=0.82` the KV pool owns most of the allocation. | `rocm-smi` shows 93% VRAM |

### 2.2 Implication

The "production" deployment is currently running **single-worker**. The router is
still configured to send traffic to `.38`, but `.38` is unreachable. For any
benchmark, capacity planning, or operator-tuning decision made below, this
must be kept in mind: the current state is **degraded**, not the intended 2-worker topology.

---

## 3. Model Architecture (from `config.json`)

| Field | Value | Note |
|---|---|---|
| `architectures` | `GlmMoeDsaForCausalLM` | DeepSeek-V3 family with DSA |
| `hidden_size` | 6144 | |
| `num_hidden_layers` | 78 | |
| `num_attention_heads` | 64 | MLA — q heads |
| `num_key_value_heads` | 64 | MLA —kv heads |
| `head_dim` / `qk_nope_head_dim` | 192 / 192 | |
| `qk_rope_head_dim` | 64 | decoupled RoPE |
| `kv_lora_rank` | 512 | MLA compression |
| `n_routed_experts` | 256 | DeepSeek-MoE |
| `n_shared_experts` | 1 | |
| `num_experts_per_tok` | 8 | top-8 of 256 |
| `topk_method` | `noaux_tc` | no-auxiliary-loss top-K |
| `moe_intermediate_size` | 2048 | |
| `intermediate_size` | 12288 | dense layers (first 3) |
| `first_k_dense_replace` | 3 | first 3 layers are dense |
| `num_nextn_predict_layers` | 1 | MTP draft layer (EAGLE/NEXTN) |
| `index_topk` | 2048 | DSA sparse attention top-K |
| `index_n_heads` | 32 | DSA indexer heads |
| `index_head_dim` | 128 | |
| `indexer_types` | `full`/`shared` pattern | every 4th layer is `full` |
| `quantization_config` | FP8 e4m3 dynamic, block [128,128] | weight-only FP8 |

### 3.1 What this implies for kernel selection

- **MLA + DSA** → attention is sparse. Standard `flash_attn` is wrong; need `dsa` backend.
- **256-expert MoE** → MoE kernel must handle 256-way routing efficiently. On TP8 (no EP), each rank holds 256 experts → 32 experts/GPU. Weight memory dominates.
- **MTP layer** → EAGLE/NEXTN speculative decoding shares the embedding+lm_head with target.
- **FP8 weights** → need FP8 GEMM kernel (deep_gemm / aiter / triton).

---

## 4. Kernel / Operator Selection (LIVE)

Pulled from running pod's `/get_server_info`. Items marked ⚠ warrant review.

### 4.1 Attention

| Setting | Live Value | Verdict | Discussion |
|---|---|---|---|
| `attention_backend` | `dsa` | ✅ correct | Auto-selected by `is_deepseek_dsa()` — required for `GlmMoeDsaForCausalLM` |
| `dsa_prefill_backend` | `tilelang` | ✅ correct for gfx942 | On AMD HIP, `tilelang` is the only supported DSA prefill impl that handles MLA+DSA. Alternatives `flashmla_sparse`/`flashmla_kv`/`fa3` are CUDA-only (`is_cuda() and not _is_hip` gates). `aiter` is available but only via explicit `--dsa-prefill-backend aiter` and is currently less stable on gfx942 for the sparse path. |
| `dsa_decode_backend` | `tilelang` | ✅ correct | Same reasoning — decode path also routes through `_forward_tilelang` for HIP. `flashmla_kv`/`fa3`/`flashmla_sparse` are CUDA-only. The `aiter` decode path (`_forward_aiter`) is available but only via explicit opt-in. ⚠ See §4.1.1 below — `aiter` decode may be worth benchmarking. |
| `dsa_paged_mqa_logits_backend` | `auto` | ✅ correct | Resolves to `deep_gemm` on CUDA, falls back to Triton on HIP. Used for the DSA top-K prefilter. |
| `dsa_topk_backend` | `sgl-kernel` | ✅ correct | `sgl-kernel` provides the fused top-K transform used by the indexer. The `SGLANG_OPT_USE_TOPK_V2=false` env var correctly disables the v2 folded top-K plan that fails to JIT-compile on gfx942 (documented in pod env comment). |
| `enable_dsa_prefill_context_parallel` | `False` | ✅ correct for single-node | `attn_cp_size=1`, no CP. CP only helps with very long prompts ≥32K that don't fit in a single forward; current `chunked-prefill-size=131072` already handles 128K-token chunks. |
| `enable_deepseek_v4_fp4_indexer` | `False` | ✅ correct | Model is FP8, not FP4. The V4 indexer is for DeepSeek-V4 FP4 models. |

#### 4.1.1 Decode backend: `tilelang` vs `aiter` — worth benchmarking

The code path for `dsa_decode_impl == "aiter"` calls `aiter.mla_decode_fwd`, which is AMD's first-party tuned MLA decode kernel. It may outperform `tilelang` on gfx942 for the sparse MLA decode shape (q_len=1, kv_lora_rank=512, num_kv_heads=64). Currently `tilelang` is the default; **no head-to-head benchmark has been done**.

**Recommendation**: short A/B test with `--dsa-decode-backend aiter` on a non-prod worker. Measure `gen_throughput` and `per_stage_req_latency` decode distribution. Risk: JIT compile differences.

### 4.2 Speculative Decoding (EAGLE / NEXTN)

| Setting | Live Value | Verdict |
|---|---|---|
| `speculative_algorithm` | `EAGLE` (parsed from `--speculative-algorithm NEXTN`) | ✅ correct |
| `speculative_num_steps` | `3` | ⚠ review |
| `speculative_num_draft_tokens` | `4` | ⚠ review |
| `speculative_eagle_topk` | `1` | ⚠ review |
| `speculative_attention_mode` | `prefill` | ✅ correct |
| `speculative_moe_runner_backend` | `auto` | ✅ correct |

#### 4.2.1 Live acceptance metrics (from `sglang:spec_*`)

| Metric | Value |
|---|---|
| `spec_verify_calls_total` | 41,145 |
| `spec_accept_rate` | **34.7%** |
| `spec_accept_length` | **2.04** (accepted drafts + bonus token per forward) |
| `spec_num_steps` | 3 |
| `spec_num_draft_tokens` | 4 |

#### 4.2.2 Interpretation

- With `num_steps=3, num_draft_tokens=4, eagle_topk=1`, the draft proposes up to 4 tokens per forward.
- Effective decode speedup = `spec_accept_length / 1.0` ≈ **2.04×** (vs theoretical 4× if all drafts accepted).
- `accept_rate = 34.7%` means of the 4 drafted tokens, ~1.4 are accepted on average — adding the bonus token gives `2.04` mean accepted length.
- This is **substantially below the theoretical 4× ceiling**. The spec decode is providing ~2× decode speedup, not 4×.

#### 4.2.3 Recommendations

| Tuning | Hypothesis | How to validate |
|---|---|---|
| Increase `speculative_eagle_topk` from `1` → `2` | More candidates per step → higher acceptance, but more verify work | A/B with `--speculative-eagle-topk 2`; watch `spec_accept_length` and `gen_throughput` |
| Increase `speculative_num_draft_tokens` from `4` → `6` or `8` | More drafts per step → if acceptance stays >50%, net win | Same — watch accept rate doesn't drop |
| Decrease `speculative_num_steps` from `3` → `2` | Lower draft latency per step; current low acceptance suggests 3 steps is too aggressive | A/B; if `accept_length` drops <1.5, revert |
| Try `--speculative-attention-mode decode` | Uses decode-shape attention for draft (lower latency per step) | A/B |

**Most likely root cause of low acceptance**: GLM-5.2's reasoning CoT is highly diverse — the MTP draft layer trained on the model's distribution may not predict reasoning tokens well. Acceptance on factual/output tokens may be much higher than on reasoning tokens. Worth checking `spec_accept_rate` filtered by output type if possible.

### 4.3 MoE

| Setting | Live Value | Verdict |
|---|---|---|
| `ep_size` | `1` | ⚠ worth review |
| `moe_runner_backend` | `auto` | ✅ correct |
| `moe_a2a_backend` | `none` | ✅ correct (matches EP=1) |
| `deepep_mode` | `auto` | ✅ |
| `enable_fused_moe_sum_all_reduce` | `False` | ✅ correct for TP-only |
| `disable_shared_experts_fusion` | `False` | ✅ correct |
| `enforce_shared_experts_fusion` | `False` | ✅ |
| `n_routed_experts` (model) | 256 | — |
| `num_experts_per_tok` (model) | 8 | — |

#### 4.3.1 EP=1 (TP-only MoE) — correct for current topology

With EP=1, all 256 experts are replicated across 8 GPUs (32 experts/GPU). Each token's top-8 experts may live on any GPU, requiring all-to-all of activations. With `moe_a2a_backend=none`, this is handled via NCCL all-to-all (default `enable_aiter_allreduce_fusion=True` for the all-reduce portion).

For a 2-worker deployment, EP is **not** beneficial — EP needs ≥2 workers to shard experts across. The current TP=8 within a single worker is correct.

**If scaling to 4+ workers in future**: consider EP across workers (DeepEP). For now, ✅ correct.

#### 4.3.2 MoE runner backend

`moe_runner_backend=auto` resolves to:
- `aiter` on HIP (AMD's fused MoE kernel, optimized for gfx942)
- `deep_gemm`/`triton` on CUDA

For FP8 weights with block-quant `[128,128]`, the `aiter` MoE kernel supports fused gate+up+down with FP8 GEMM. The env `SGLANG_USE_AITER=1` confirms this path is active.

✅ Correct.

### 4.4 Memory & KV Cache

| Setting | Live Value | Verdict |
|---|---|---|
| `kv_cache_dtype` | `fp8_e4m3` | ✅ correct — FP8 KV halves memory vs bf16, doubles capacity |
| `mem_fraction_static` | `0.82` | ⚠ review — see §4.4.1 |
| `page_size` | `64` | ✅ correct (DSA backend assumes page_size=1 for sparse indices, but uses 64 for dense allocations) |
| `max_total_num_tokens` (computed) | 740,352 per rank | 740K tokens of KV capacity — plenty for current load (peak usage observed = 1%) |
| `enable_hierarchical_cache` | `True` | ✅ correct |
| `hicache_ratio` | `1.0` | ✅ — same size host pool as GPU pool |
| `hicache_io_backend` | `direct` | ✅ — direct memcpy, no io_uring on gfx942 |
| `hicache_mem_layout` | `page_first_direct` | ✅ — page-major layout for direct transfer |
| `hicache_write_policy` | `write_through` | ✅ — write to GPU + host simultaneously |
| `enable_page_major_kv_layout` | `False` | ✅ — `page_first_direct` already covers the hicache path |
| `disable_radix_cache` | `False` | ✅ — radix cache enabled for prefix reuse |

#### 4.4.1 `mem_fraction_static=0.82` — review recommended

`rocm-smi` shows VRAM at 93-94% at idle. The 0.82 static fraction leaves ~18% for activations + intermediates + cuda-graph capture. With `max-running-requests=32` and `chunked-prefill-size=131072`, peak activation memory can spike:

- Decode activation: ~32 × 6144 × fp8 ≈ 200 MB / layer
- Prefill activation: 131072 × 6144 × fp8 ≈ 800 MB / layer (worst case single chunk)
- CUDA graph capture for decode BS 16: ~16 × 78 layers × intermediates

**Concern**: At 93% VRAM idle, peak may push to 100% → OOM. The pod has not OOM'd in 10h, so the headroom is sufficient for current load, but adding max-running-requests higher or larger prefill chunks could trigger OOM.

**Recommendation**: Keep `0.82` unless increasing batch capacity. If capacity needs to grow, consider `0.85` only after measuring peak activation memory under load.

### 4.5 Scheduler

| Setting | Live Value | Verdict |
|---|---|---|
| `schedule_policy` | `fcfs` | ✅ correct — first-come-first-served, no priority preemption |
| `schedule_conservativeness` | `0.5` | ⚠ review — see §4.5.1 |
| `retraction_policy` | `length` | ✅ correct — evict longest-sequence first under pressure |
| `max_running_requests` | `32` | ✅ matches cuda-graph-bs-decode max |
| `prefill_max_requests` | `32` | ✅ |
| `max_prefill_tokens` | `131072` | ✅ matches chunked-prefill-size |
| `chunked_prefill_size` | `131072` | ⚠ review — see §4.5.2 |
| `disable_overlap_schedule` | `False` | ✅ — overlap scheduler enabled (CPU scheduling overlaps with GPU compute) |
| `num_continuous_decode_steps` | `1` | ⚠ review — see §4.5.3 |
| `enable_two_batch_overlap` | `False` | ✅ correct — two-batch overlap is for EP, not TP-only |

#### 4.5.1 `schedule_conservativeness=0.5` — conservative

Lower conservativeness = scheduler is more cautious about admitting new prefills. At `0.5`, it reserves more memory headroom than the default `1.0`. Combined with `mem_fraction_static=0.82` and 93% VRAM usage, this is the right setting to avoid OOM.

If VRAM headroom improves (e.g., via `0.85`), `schedule_conservativeness` can go back to `1.0` for more aggressive batching.

#### 4.5.2 `chunked_prefill_size=131072` — very large

131K-token chunks are 4× larger than the manifest's documented 32768. This means a single prefill can consume up to 131K tokens of KV cache in one forward. With `max_total_num_tokens=740K`, that's ~18% of pool per chunk.

**Pros**: Fewer chunks for long prompts → less scheduling overhead, better prefill throughput.

**Cons**: A single 131K-token prefill blocks the scheduler from admitting other prefills. Under mixed prefill+decode workload, this can stall decodes.

**Empirical**: prefill_forward avg = 358 ms, p50 ≈ 124 ms, p99 ≈ 5.9 s. The p99 tail is driven by long-context prefills (>32K input tokens).

**Recommendation**: For current workload (codex/chat, mostly <8K prompts), keep 131072. If long-context (>32K) prompts become common, consider lowering to 32768 to reduce tail latency.

#### 4.5.3 `num_continuous_decode_steps=1` — review

Currently the scheduler does 1 decode step between prefill checks. Increasing to 2-3 can improve decode throughput by reducing scheduler overhead, but may delay new prefills.

**Recommendation**: Try `num_continuous_decode_steps=2` under concurrent decode-heavy load. Watch TTFT for new prefills.

### 4.6 CUDA Graphs

| Setting | Live Value | Verdict |
|---|---|---|
| `cuda_graph_backend_decode` | `full` (resolved from default) | ✅ correct — full graph capture for decode |
| `cuda_graph_backend_prefill` | `breakable` | ✅ correct — breakable allows variable-length prefill |
| `cuda_graph_bs_decode` | `[1,2,3,4,5,6,7,8,9,10,12,16]` | ✅ covers 1-16 with common sizes |
| `cuda_graph_bs_prefill` | `[4,8,16,32]` | ✅ covers common prefill batch sizes |
| `cuda_graph_max_bs_decode` | `16` | ✅ matches max-running-requests / 2 for spec decode |
| `cuda_graph_tc_compiler` | `eager` (resolved) | ✅ correct for HIP — avoids torch.compile issues |

#### 4.6.1 Decode BS ceiling

`cuda_graph_max_bs_decode=16` means any decode batch >16 falls back to eager. With `max_running_requests=32` and spec decode (each request uses 1 target + 1 draft slot), effective batch can reach 16 target + 16 draft = 32. The graph captures up to BS=16, beyond which eager mode kicks in.

**Empirical**: Under 20 concurrent, mid-flight `num_running_reqs=11` per worker — well within graph coverage.

✅ Correct.

### 4.7 Communication

| Setting | Live Value | Verdict |
|---|---|---|
| `enable_aiter_allreduce_fusion` | `True` | ✅ correct — aiter fused all-reduce on HIP |
| `enable_flashinfer_allreduce_fusion` | `False` | ✅ — flashinfer is CUDA-only |
| `enable_quant_communications` | `False` | ⚠ review |
| `enable_symm_mem` | `False` | ✅ — not needed for TP-only |
| `enable_torch_symm_mem` | `False` | ✅ |
| `enable_mscclpp` | `False` | ✅ — mscclpp is CUDA-only |
| `disable_custom_all_reduce` | `False` | ✅ — allows aiter all-reduce |
| `enable_nccl_nvls` | `False` | ✅ — NVLS is NVIDIA-only |
| `NCCL_MIN_NCHANNELS` | `112` | ✅ tuned for 8-GPU all-reduce on MI308X |
| `NCCL_CUMEM_ENABLE` | `0` | ✅ — disable NCCL registered memory on AMD |
| `ROCM_QUICK_REDUCE_QUANTIZATION` | `INT8` | ✅ — INT8 quantized all-reduce for bandwidth efficiency |

#### 4.7.1 `enable_quant_communications=False` — review

Quantized communications (FP8 all-reduce) would reduce NCCL bandwidth usage by 2×. On MI308X with Infinity Fabric (~200 GB/s GPU-GPU), this could improve TP8 all-reduce latency.

**Recommendation**: A/B test `--enable-quant-communications` under concurrent load. Risk: numerical accuracy — verify eval results don't regress.

### 4.8 Other Operators

| Setting | Live Value | Verdict |
|---|---|---|
| `sampling_backend` | `pytorch` | ✅ — `torch_sampling` is the most portable on HIP |
| `grammar_backend` | `xgrammar` | ✅ — xgrammar is the only supported backend |
| `mamba_backend` | `triton` | N/A — model has no Mamba/SSM layers |
| `linear_attn_backend` | `triton` | N/A — model has no linear attention layers |
| `enable_fused_qk_norm_rope` | `True` | ✅ — fused QK-norm + RoPE saves a kernel launch |
| `enable_mixed_chunk` | `False` | ✅ — forced off by EAGLE mutex (`speculative_hook.py:354`) |
| `enable_torch_compile` | `False` | ✅ — torch.compile has issues on gfx942 |
| `triton_attention_num_kv_splits` | `16` | ✅ — used by triton decode fallback |
| `triton_attention_reduce_in_fp32` | `False` | ✅ — fp8 reduce is fine for fp8 KV |

### 4.9 SGLang Environment Variables (pod)

| Variable | Value | Verdict |
|---|---|---|
| `SGLANG_USE_AITER` | `1` | ✅ — enables AMD aiter kernels |
| `SGLANG_USE_ROCM700A` | `1` | ✅ — enables ROCm 700A addressing |
| `SGLANG_ROCM_FUSED_DECODE_MLA` | `1` | ✅ — fused MLA decode kernel |
| `SGLANG_ROCM_DISABLE_LINEARQUANT` | `0` | ✅ — linear quant enabled |
| `SGLANG_MOE_PADDING` | `1` | ✅ — pad MoE inputs for aligned GEMM |
| `SGLANG_INT4_WEIGHT` | `0` | ✅ — model is FP8, not INT4 |
| `SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM` | `1` | ✅ — dual-stream DeepSeek-V2 prefill-compute overlap |
| `SGLANG_OPT_USE_TOPK_V2` | `false` | ✅ — disable v2 top-K JIT kernel (compile failure on gfx942) |
| `SGLANG_DISABLE_CUDNN_CHECK` | `1` | ✅ — skip cuDNN check on ROCm |
| `SGLANG_SET_CPU_AFFINITY` | `1` | ✅ — pin CPU cores for scheduler threads |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | `1` | ✅ — allow context > model's max_position (1048576 > 524288) |
| `HIP_FORCE_DEV_KERNARG` | `1` | ✅ — device-side kernel arg allocation |
| `HSA_NO_SCRATCH_RECLAIM` | `1` | ✅ — avoid scratch reclaim overhead |
| `HSA_ENABLE_SDMA` | `0` | ✅ — disable SDMA (use memcpy) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | ✅ — reduce fragmentation |
| `PYTORCH_ROCM_ARCH` | `gfx942` | ✅ — target arch |

---

## 5. Operator-Level Latency Evidence

### 5.1 Prefill latency distribution (2339 samples, tp_rank=0)

| Percentile | Latency |
|---|---|
| p50 | ≤ 124 ms |
| p75 | ≤ 124 ms |
| p90 | ≤ 327 ms |
| p95 | ≤ 1.39 s |
| p99 | ≤ 5.91 s |
| avg | 358 ms |

**Distribution**:
- 88.3% of prefills ≤ 124 ms (short prompts, <2K tokens)
- 4.0% in 124-327 ms (medium, 2-8K)
- 4.0% in 327 ms - 1.4 s (long, 8-32K)
- 3.4% in 1.4 s - 6 s (very long, 32K-128K)
- 0.3% > 6 s (extreme, likely 128K chunked prefills)

**Interpretation**: The DSA prefill kernel (`tilelang`) is performing well for the common case. The tail is dominated by long-context prefills hitting the 131K chunk size.

### 5.2 Decode throughput

From concurrent benchmark (20 reqs):
- Worker1 (.103) mid-flight: `gen_throughput=3.69 tok/s` per request × 11 active = **40.6 tok/s aggregate**
- Worker2 (.38 was unavailable — no data)
- Post-test: `gen_throughput=376.7 tok/s` (worker1, batch draining)

With EAGLE `spec_accept_length=2.04`, effective per-request decode = ~2.04 tokens per forward step. At ~93 tok/s single-stream (from bottleneck benchmark) and batch scheduling, aggregate scales to ~370 tok/s — consistent.

### 5.3 EAGLE effectiveness

| Metric | Value | Interpretation |
|---|---|---|
| Theoretical max speedup | 4× (4 draft tokens accepted + 1 bonus) | Upper bound |
| Measured accept_length | 2.04 | ~2× speedup — 51% of theoretical |
| Measured accept_rate | 34.7% | Of 4 drafted, ~1.4 accepted |
| Single-stream decode | 93.7 tok/s | With EAGLE |
| Without EAGLE (estimated) | ~46 tok/s | 93.7 / 2.04 |

EAGLE is delivering real value (~2× decode speedup) but leaving ~2× on the table vs theoretical max.

---

## 6. Summary of Recommendations

### 6.1 Immediate (operational) — fix degraded state

| Priority | Action | Effort |
|---|---|---|
| P0 | **Bring worker2 back online** — either fix node `.38` (firmware update) or move `w2-sglang` StatefulSet to a healthy node and scale to `replicas=1` | Infra team |
| P0 | **Update router worker-urls** if worker2 IP changes | kubectl patch |
| P1 | **Sync repo manifest to live config** — update `configs/pd-manifests/sglang-glm52-2tp8-manifest.yaml` to reflect `mem-fraction-static=0.82`, `chunked-prefill-size=131072`, `max-prefill-tokens=131072`, `watchdog-timeout=1200` | docs |

### 6.2 Operator tuning (short-term, low risk)

| Priority | Setting change | Hypothesis | Validation |
|---|---|---|---|
| P1 | `--dsa-decode-backend aiter` | AMD's first-party MLA decode kernel may beat tilelang on gfx942 | A/B on test env, measure `gen_throughput` and decode latency p50/p99 |
| P1 | `--speculative-eagle-topk 2` | More candidates → higher acceptance | Watch `spec_accept_length`; if >2.5, keep |
| P2 | `--speculative-num-draft-tokens 6` | More drafts → more accepted tokens per step | Watch `spec_accept_rate`; if >50%, keep |
| P2 | `--num-continuous-decode-steps 2` | Less scheduler overhead between decode steps | A/B under concurrent decode load |
| P3 | `--enable-quant-communications` | FP8 all-reduce → 2× less NCCL bandwidth | Verify eval still passes |

### 6.3 Operator tuning (medium-term, needs benchmarking)

| Priority | Setting change | Reason |
|---|---|---|
| P3 | `--mem-fraction-static 0.85` + `--schedule-conservativeness 1.0` | More KV capacity + aggressive admission — only if VRAM headroom confirmed |
| P3 | `--chunked-prefill-size 32768` (revert to manifest value) | Reduce p99 prefill tail from 5.9s — only if long-context prompts dominate |
| P4 | PD disaggregation (already in `configs/pd-manifests/`) | Decouple prefill from decode for better SLO |

### 6.4 Explicitly NOT recommended

| Change | Why NOT |
|---|---|
| `--enable-mixed-chunk` | Mutex with EAGLE (`speculative_hook.py:354`) — loses 2× decode speedup |
| `--enable-deepseek-v4-fp4-indexer` | Model is FP8, not FP4 |
| `--enable-torch-compile` | Unstable on gfx942 |
| `--ep-size >1` within single worker | EP needs multi-node; TP-only is correct for 8-GPU single node |
| `--moe-a2a-backend deepep` | DeepEP is for EP>1 across nodes |
| `SGLANG_OPT_USE_TOPK_V2=true` | JIT compile failure on gfx942 (dpsk_v4 topk_v2 macro errors) |
| `--dsa-prefill-backend flashmla_*` | CUDA-only, gated by `is_cuda() and not _is_hip` |
| `--dsa-prefill-backend fa3` | CUDA-only |

---

## 7. Verdict: Is the current operator selection "most reasonable"?

**Yes, with caveats.**

The kernel selections are **correct for the hardware (gfx942 / MI308X) and model architecture (GlmMoeDsaForCausalLM FP8)**. Every CUDA-only alternative is properly gated off. The AMD-specific paths (aiter, tilelang, ROCm env vars) are all enabled. The DSA sparse attention backend is auto-selected and using the only viable backend on HIP (tilelang).

**However, three things are not optimal:**

1. **EAGLE is underperforming**: 34.7% acceptance / 2.04× speedup vs 4× theoretical. This is the single biggest decode-throughput opportunity. Tuning `eagle_topk` and `num_draft_tokens` could potentially unlock another 1.5-2× decode.

2. **DSA decode backend untested against `aiter`**: `tilelang` is the default, but `aiter.mla_decode_fwd` is AMD's first-party kernel specifically tuned for MLA on gfx942. No head-to-head benchmark exists.

3. **Deployment is degraded**: Worker2 is offline (node `.38` NotReady). All benchmark numbers reflect single-worker operation, which understates capacity and may bias operator-tuning decisions (e.g., EAGLE acceptance may differ under cross-worker batching).

**Recommendation priority order**:
1. Restore worker2 (P0 infra)
2. A/B test `--dsa-decode-backend aiter` (P1, low risk, potentially large decode win)
3. A/B test `--speculative-eagle-topk 2` and `--speculative-num-draft-tokens 6` (P1, low risk, directly targets the 2× gap)
4. Sync repo manifest to live config (P1, doc hygiene)
5. Benchmark `--enable-quant-communications` (P2)
6. Evaluate `--num-continuous-decode-steps 2` (P2)
