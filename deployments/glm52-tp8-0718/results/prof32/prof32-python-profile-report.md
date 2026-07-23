# prof32 — sglang GLM-5.2 TP8 Python/sglang-internal decode profile

**Node:** node-21.234.170.32 (IP 21.234.170.32) · **Pod:** `test32-sglang-0` · **Rank profiled:** TP0 scheduler, pid 760
**Date:** 2026-07-20 · **Layer:** Python + sglang-internal (pairs with sibling agent's GPU rocprof on node 19)

## Setup
- **Workload** (`scripts/prof32_decode_workload.py`, run inside the pod): 8 concurrent streaming
  `/v1/completions`, ~9.6K input tokens (prefix-cacheable shared header), `max_tokens=256`,
  `temperature=0.0`, 90s. `/metrics` snapshotted every 15s to `/tmp/prof32-metrics-NN.txt`.
- **py-spy** attached to pid 760 (scheduler_TP0): clean run 50 Hz × 60 s = **2967 samples, 0 errors**
  (`prof32-pyspy.svg`); raw folded stacks 50 Hz × 40 s = **1988 samples, 0 errors**
  (`prof32-pyspy-raw.txt`); native run 100 Hz × 60 s = 5201 samples but **fell behind 23 s —
  unreliable, kept only for cross-reference** (`prof32-pyspy-native.svg`).
- All artifacts persisted to `deployments/glm52-tp8-0718/results/prof32/`.
  (`/data` was read-only on the host; used the repo results dir instead.)

## 1. sglang built-in metrics under decode load

Timing-related metrics exposed (gauges/counters/histograms, tp_rank=0 unless noted):

| metric | value (decode load) | note |
|---|---|---|
| `gen_throughput` (tp0) | up to 118.3 tok/s | engine instantaneous |
| `inter_token_latency_seconds` mean | **~77 ms** | engine-level ITL (all traffic) |
| `time_to_first_token_seconds` mean | 1112 ms | cumulative since startup (long-context prefills) |
| `per_stage_req_latency_seconds{stage=prefill_forward}` | 471 s / 675 reqs ≈ 0.70 s/req | **only prefill_forward + request_process stages exposed** |
| `per_stage_req_latency_seconds{stage=request_process}` | ~0.002 s/req | negligible |
| `spec_accept_rate` | **0.463** | EAGLE accept rate |
| `spec_accept_length` | **2.85** | tokens accepted per verify call |
| `spec_num_steps` / `spec_num_draft_tokens` | 4 / 5 | C4 winner config |
| `spec_verify_calls_total` | ~23.6k cumulative | |
| `is_cuda_graph` | **0** | last forward was NOT a cuda graph |
| `cuda_graph_passes_total` | 396, **delta=0 during workload** | **no graph replays during decode** |
| `num_running_reqs` | 6–8 | |
| `num_queue_reqs` | 0 | no queueing |
| `kv_used_tokens` | ~9.7k | light KV usage |

**Per-phase breakdown sglang exposes:** only `prefill_forward` and `request_process`. There is
**no `decode_forward` or `verify` stage** in `per_stage_req_latency_seconds`, so sglang's own
metrics cannot split decode vs verify vs schedule time. `inter_token_latency_seconds` is the only
decode-side timing signal (mean ~77 ms; workload-measured ITL p50 130 ms / p95 414 ms).

## 2. py-spy flamegraph — top functions by self-time (clean 50 Hz run)

Phase breakdown by leaf self-time:

| phase | self% |
|---|---|
| attention-launch (DSA `init_forward_metadata` + tilelang) | **39.1%** |
| aiter/tilelang JIT dispatch + torch-op overhead (in "other") | ~25% |
| model-forward (deepseek_v2 / norms / linear) | 7.8% |
| MoE-routing (`moe_sorting`, `fused_moe_1stage`) | 5.0% |
| torch-overhead (`torch/cuda`, `_lazy_init`, `current_stream`) | 4.2% |
| allreduce/comm | 3.1% |
| IPC/tokenizer | 1.2% |
| scheduler (`run_batch`, `event_loop`) | 0.7% |
| EAGLE-verify (`eagle_worker_v2`) | 0.4% |
| sampling | 0.3% |

Top 12 functions by self-time:

| self% | samples | function (file) |
|---|---|---|
| **24.65%** | 490 | `init_forward_metadata` (dsa_backend.py) |
| 4.88% | 97 | `wrapper` (aiter jit core.py) |
| 4.58% | 91 | `__call__` (torch/_ops.py) |
| 4.23% | 84 | `lambda_forward` (aiter adapter.py) |
| 3.37% | 67 | `torch_to_aiter_pybind` (aiter dtypes.py) |
| 3.32% | 66 | `forward_absorb_core` (forward_mla.py) |
| 2.87% | 57 | `forward_absorb_prepare` (forward_mla.py) |
| 2.46% | 49 | `gemm_a8w8_blockscale` (gemm_a8w8_blockscale.py) |
| 1.56% | 31 | `__call__` (aiter driver.py) |
| 1.46% | 29 | `per_group_quant_hip` (quant.py) |
| 1.41% | 28 | `all_reduce` (distributed_c10d.py) |
| 1.41% | 28 | `run` (aiter jit.py) |

**Dominant phase: attention-launch** — specifically the DSA backend's per-forward Python metadata
preparation. The EAGLE-verify *Python* frame itself is cheap (0.4%); the cost is the attention
metadata + aiter JIT dispatch that every verify/draft forward triggers.

## 3. sglang profiling flags discovered

- `SGLANG_PROFILE_V2` env var + `/start_profile` `/stop_profile` HTTP endpoints → drives
  `torch.profiler` (CUDA/CPU activity) with `profile_by_stage` (separate traces for EXTEND vs DECODE).
  This is **GPU-kernel-level** (overlaps the sibling agent's rocprof) — not used here, since the
  task is the Python layer.
- `--enable_profile_cuda_graph` server arg (cuda-graph capture profiling only).
- No env flag exposes a per-phase **Python** time breakdown; `per_stage_req_latency_seconds` does
  not cover decode/verify. py-spy was the right tool and attached cleanly (ptrace permitted).

## 4. #1 Python-layer hotspot + optimization candidates

**#1 hotspot: `DSA init_forward_metadata` (dsa_backend.py) — 24.65% of Python self-time.**

Root cause: under EAGLE (NEXTN steps=4, draft=4, topk=1) the decode loop runs the **target-verify
and draft-extend forwards in eager mode**, not via cuda-graph replay. Evidence:
- py-spy stacks show `eagle_worker_v2.forward_batch_generation → model_runner.forward →
  _forward_raw → eager_runner.execute → _execute_extend` on every decode sample.
- `is_cuda_graph=0` and `cuda_graph_passes_total` delta=0 during the workload — zero graph replays.
- Each eager forward re-runs the full `init_forward_metadata` Python path (tensor aranges,
  `repeat_interleave` for the page table, `seqlens_expand_triton`, `compute_cu_seqlens`, several
  `.item()`/`.cpu()` host syncs) plus per-op aiter/tilelang JIT dispatch overhead.

With `spec_accept_length≈2.85`, each accepted token costs ~1 verify + draft forwards, all eager —
so this Python overhead is paid ~once per ~2.85 tokens and dominates the non-GK time.

**Optimization candidates:**
1. **Capture cuda graphs for the EAGLE target-verify / draft-extend paths** (or extend
   `init_forward_metadata_out_graph` + `init_forward_metadata_replay_cuda_graph_from_precomputed`
   to the verify/draft modes). The decode cuda-graph path already exists for plain decode; the
   verify/draft eager path is the gap. This would eliminate the per-forward
   `init_forward_metadata` + aiter JIT dispatch cost — the single biggest win.
2. **Cache/specialize `init_forward_metadata` for the fixed-shape verify batch.** The target-verify
   batch shape is static (bs × `speculative_num_draft_tokens`); the `torch.arange`,
   `torch.repeat_interleave`, `compute_cu_seqlens`, and `seqlens_expand_triton` outputs could be
   precomputed once and reused, avoiding per-forward Python tensor construction and the
   `.item()` host syncs.
3. **Reduce aiter/tilelang per-call dispatch overhead** (`wrapper`/`__call__`/`torch_to_aiter_pybind`/
   `compute_cache_key`/`deepcopy` ≈ 20–25% combined): the JIT cache-key computation and dtype
   pybind conversion run on every MoE/GEMM/quant call. Caching the resolved callable by
   (shape,dtype) and skipping `deepcopy`/`compute_cache_key` on hot paths would cut launch overhead
   for the MoE dispatch (the prior hypothesis: MoE dispatch / allreduce / attention — confirmed here
   as Python-side dispatch, not the bf16 GEMM kernel).

## 5. Does the Python layer reveal overhead the GPU profile won't see?

**Yes — this is the main value of the Python-level pass.** The GPU rocprof profile (sibling agent)
measures kernel execution time. It will **not** see:
- The **DSA `init_forward_metadata` Python metadata prep** (24.65%) — pure CPU between kernel
  launches, including `.item()`/`.cpu()` host syncs that stall the pipeline.
- The **aiter/tilelang JIT dispatch + `compute_cache_key` + `deepcopy`** (~20–25%) — Python-side
  kernel-selection overhead per MoE/GEMM/quant call.
- The **eager-vs-cuda-graph gap**: rocprof sees the same kernels either way, but only the Python
  profile reveals that decode is running eager (no graph replay) and paying full launch overhead
  per forward. `is_cuda_graph=0` + `cuda_graph_passes_total` delta=0 is the smoking gun.
- `torch.cuda` lazy-init / `current_stream` / `__getattr__` micro-overhead (~4%) per op.

The GPU profile will show attention/MoE/allreduce kernel time; the Python profile shows that
**between those kernels, ~40% of wall time is DSA metadata prep + aiter JIT dispatch**, and that
**EAGLE verify/draft forwards bypass cuda graphs entirely**. The two layers are complementary:
the decode bottleneck is not the bf16 GEMM kernel (prior finding) nor raw kernel execution — it
is Python-side attention-metadata + eager-launch overhead amplified by EAGLE's many eager forwards.

## 6. Blockers / faithfulness notes
- **py-spy attached cleanly** (ptrace permitted in the container) — no permission blocker.
- **Native py-spy (100 Hz `--native`) fell 23 s behind** → results inaccurate; the clean
  Python-only 50 Hz run is the authoritative one. Native C-extension time is therefore
  under-attributed; the aiter/tilelang C call overhead is if anything *larger* than reported.
- **sglang exposes no decode/verify timing stage** in `per_stage_req_latency_seconds` (only
  `prefill_forward` + `request_process`), so the prefill/decode/verify/ schedule split cannot be
  read from metrics alone — py-spy was required to get it.
- `/data` was read-only on the host; artifacts persisted to
  `deployments/glm52-tp8-0718/results/prof32/` instead.
- Worker `test32-sglang-0` left RUNNING READY 1/1. Only `prof32-`-prefixed files created in the pod
  (`/tmp/prof32-*`); no production pods touched, no `prof32-` k8s resources needed.

## Artifacts
- `prof32-pyspy.svg` — clean flamegraph (50 Hz, 60 s, 2967 samples)
- `prof32-pyspy-raw.txt` — folded stacks (aggregated for the tables above)
- `prof32-pyspy-native.svg` — native run (kept for cross-ref, fell behind)
- `prof32-metrics-00..06.txt` — `/metrics` snapshots
- `prof32-workload{,2,3}.log` — workload summaries
- `scripts/prof32_decode_workload.py`, `scripts/prof32_aggregate_pyspy.py`
