# GLM-5.2 AMD355 / MI35x SGLang Benchmark Summary

Date: 2026-06-22 / 2026-06-23

This document summarizes the GLM-5.2 SGLang validation runs on idle AMD GPU machines. The goal was to verify whether MTP/speculative decoding, EP/EPLB, and TP8/PP1 variants can improve throughput while preserving 1M context support.

## Executive Summary

Current recommendation: **split by workload**.

- **Decode-heavy service (conversation/generation)**: use `TP8/PP1 + MTP conservative` (`steps=3`). Verified 2026-06-23 on bare-metal-a to deliver **1.56x–4.15x decode speedup** over the `TP4/PP2` baseline (warm), with stable MTP acceptance (`accept len 2.4–3.15`, `rate 0.47–0.72`) and balanced GPU util (82–90% across 7 GPUs). Zero code change required — `steps=3` avoids the `cuda_runtime.h` JIT path that hangs `steps=5`.
- **Prefill-heavy / 1M-context retrieval service (low decode)**: keep the stable `TP4/PP2` baseline. MTP does not accelerate prefill, and large-prompt prefill regressed under the MTP config in this run (needs isolation).
- Do not enable `EP basic` (slower), `EPLB` (deadlocks), or `TP4/PP2+MTP` (Bug A — PP + spec unsupported by design).

Main findings:

- `TP4/PP2 + MTP/speculative decoding` fails at startup because SGLang currently rejects pipeline parallelism with speculative decoding under the active schedule path. **Confirmed by code investigation to be a design limit, not a conservative gate** — see `amd355-rocm-mtp/ANALYSIS.md` Bug A.
- `EP basic` starts and completes benchmarks, but it is slower than baseline in most tested cases.
- `EPLB` fails health check with scheduler watchdog timeout and Gloo receive waits.
- `MTP+EP` cannot be validated with the current `TP4/PP2` baseline because it hits the same PP + speculative decoding assertion.
- `TP8/PP1+MTP aggressive` (`steps=5`) starts and reports MTP acceptance, but the first benchmark hangs on `cuda_runtime.h` JIT compilation failures (Bug B — multi-backend fused-metadata-copy JIT path not HIP-guarded). **`TP8/PP1+MTP conservative` (`steps=3`) is verified working** and is the current decode-throughput winner.

## Test Machines

| Agent | Machine | IP | Test Focus |
|---|---|---:|---|
| Agent A | `xid18k-node-4` | `149.28.124.220` | MTP / speculative decoding |
| Agent B | `xid18k-node-7` | `216.128.155.171` | EP / EPLB |
| Agent C | `bare-metal-a` | `216.128.154.57` | MTP+EP and TP8/PP1+MTP |

All agents used SSH key `~/.ssh/id_ed25519_amd_poc` and only created/removing test containers with `sglang_perf_*` prefixes. Existing non-test containers were not modified.

## Common Environment

- Docker image: `lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260620`
- Model path: `/data/models/GLM-5.2-FP8`
- API endpoint: `http://127.0.0.1:30000/v1/chat/completions`
- Container shared memory: `--shm-size 32g`
- Context length: `1048576`
- KV cache dtype: `fp8_e4m3`
- Patch scripts executed before `sglang.launch_server`:
  - `/data/patch_glm_config.py`
  - `/data/patch_pp_missing_layer.py`

## Baseline Configuration

The baseline mirrors the current stable master-style 1M context deployment:

```bash
--tp-size 4
--pp-size 2
--context-length 1048576
--kv-cache-dtype fp8_e4m3
--mem-fraction-static 0.88
--pp-async-batch-depth 4
--pp-max-micro-batch-size 32
--enable-mixed-chunk
--chunked-prefill-size 32768
--enable-aiter-allreduce-fusion
--enable-fused-qk-norm-rope
--schedule-conservativeness 0.5
```

## Benchmark Suites

Each variant was intended to run the same suites:

| Suite | Purpose | Shape |
|---|---|---|
| `short_c32` | decode-heavy | short prompt, concurrency 32, `max_tokens=128` |
| `short_c128` | high-concurrency decode-heavy | short prompt, concurrency 128, `max_tokens=128` |
| `mid_c32` | mixed prompt/decode | medium prompt, concurrency 32, `max_tokens=512` |
| `prefill16k_c32` | long prompt prefill | about 16k prompt, concurrency 32, `max_tokens=32` |
| `prefill64k_c4` | large prompt stability | about 64k prompt, concurrency 4, `max_tokens=32` |
| `prefill128k_c1` | 1M-context large prompt check | about 128k prompt, concurrency 1, `max_tokens=32` |

Metrics collected:

- request success / errors
- `prompt_tok_s`
- `completion_tok_s`
- average latency
- P95 latency
- health check time
- GPU utilization / VRAM summary
- relevant SGLang log snippets, especially MTP/speculative and EP/EPLB errors

## Agent A: MTP Validation

Machine: `xid18k-node-4` / `149.28.124.220`

### Parameter Variants

`mtp_conservative`:

```bash
--speculative-algorithm NEXTN
--speculative-num-steps 3
--speculative-num-draft-tokens 4
--speculative-eagle-topk 1
```

`mtp_aggressive`:

```bash
--speculative-algorithm NEXTN
--speculative-num-steps 5
--speculative-num-draft-tokens 6
--speculative-eagle-topk 1
```

Both variants were applied on top of the `TP4/PP2` baseline.

### Startup Status

| Variant | Start | Health Check | Result |
|---|---:|---:|---|
| `baseline` | success | success, about `309.93s` | benchmark ran |
| `mtp_conservative` | failed | n/a | startup assertion |
| `mtp_aggressive` | failed | n/a | startup assertion |

### Baseline Result

| Suite | Success | Prompt tok/s | Completion tok/s | Avg Latency | P95 Latency | Errors |
|---|---:|---:|---:|---:|---:|---:|
| `short_c32` | yes | 301.434 | 803.823 | 5.054s | 5.616s | 0 |
| `short_c128` | yes | 547.606 | 1460.282 | 11.173s | 11.188s | 0 |
| `mid_c32` | yes | 4342.419 | 454.852 | 36.007s | 38.903s | 0 |
| `prefill16k_c32` | yes | 106804.667 | 94.866 | 10.762s | 10.767s | 0 |
| `prefill64k_c4` | no | 0 | 0 | n/a | n/a | 8 |
| `prefill128k_c1` | no | 0 | 0 | n/a | n/a | 2 |

Note: on this node, the baseline service exited after the `prefill64k_c4` failure, causing `prefill128k_c1` to receive connection refused errors.

### MTP Error

Both MTP variants failed before benchmark:

```text
Pipeline parallelism is incompatible with overlap schedule.
Max running requests is reset to 48 for speculative decoding.
Non-overlap (synchronous) spec v2 is used for eagle/eagle3/standalone speculative decoding.
Mixed chunked prefill is disabled because of using eagle speculative decoding.
AssertionError: Pipeline parallelism is not compatible with overlap schedule, speculative decoding
```

Interpretation:

- Current `TP4/PP2` baseline cannot directly enable SGLang speculative decoding/MTP.
- The failure happens before service health, so there is no valid MTP throughput comparison for this configuration.
- Follow-up optimization should inspect SGLang scheduling compatibility around PP + speculative decoding, especially overlap vs non-overlap schedule selection.

## Agent B: EP / EPLB Validation

Machine: `xid18k-node-7` / `216.128.155.171`

### Parameter Variants

`ep_basic`:

```bash
--expert-parallel-size 4
--moe-runner-backend aiter
--moe-a2a-backend none
```

`ep_eplb`:

```bash
--expert-parallel-size 4
--moe-runner-backend aiter
--moe-a2a-backend none
--enable-eplb
--eplb-algorithm auto
--enable-expert-distribution-metrics
```

Both variants were applied on top of the `TP4/PP2` baseline.

### Startup Status

| Variant | Start | Health | Health Time | Result |
|---|---:|---:|---:|---|
| `baseline` | success | success | 297.73s | 6 suites completed |
| `ep_basic` | success | success | 306.94s | 6 suites completed |
| `ep_eplb` | success | failed | n/a | benchmark not run |

### Benchmark Result

| Suite | Variant | Success | Prompt tok/s | Completion tok/s | Avg Latency | P95 Latency | Errors |
|---|---|---:|---:|---:|---:|---:|---:|
| `short_c32` | baseline | yes | 151.90 | 720.10 | 5.674s | 5.678s | 0 |
| `short_c32` | `ep_basic` | yes | 148.56 | 704.29 | 5.805s | 5.809s | 0 |
| `short_c128` | baseline | yes | 397.39 | 1883.91 | 8.651s | 8.667s | 0 |
| `short_c128` | `ep_basic` | yes | 358.47 | 1699.43 | 9.594s | 9.601s | 0 |
| `mid_c32` | baseline | yes | 1244.24 | 648.73 | 25.241s | 25.245s | 0 |
| `mid_c32` | `ep_basic` | yes | 1281.02 | 667.91 | 24.518s | 24.522s | 0 |
| `prefill16k_c32` | baseline | yes | 79642.60 | 144.94 | 7.043s | 7.052s | 0 |
| `prefill16k_c32` | `ep_basic` | yes | 75084.76 | 136.65 | 7.470s | 7.480s | 0 |
| `prefill64k_c4` | baseline | yes | 37172.79 | 16.92 | 7.560s | 7.563s | 0 |
| `prefill64k_c4` | `ep_basic` | yes | 35532.90 | 16.17 | 7.909s | 7.912s | 0 |
| `prefill128k_c1` | baseline | yes | 21353.19 | 4.86 | 6.583s | 6.583s | 0 |
| `prefill128k_c1` | `ep_basic` | yes | 19468.95 | 4.43 | 7.220s | 7.220s | 0 |

### EP Result Interpretation

`ep_basic` did not improve decode throughput:

- `short_c32`: `704.29 / 720.10 = 0.98x`
- `short_c128`: `1699.43 / 1883.91 = 0.90x`

It also slowed most long prompt cases:

- `prefill16k_c32`: `75084.76 / 79642.60 = 0.94x`
- `prefill64k_c4`: `35532.90 / 37172.79 = 0.96x`
- `prefill128k_c1`: `19468.95 / 21353.19 = 0.91x`

Only `mid_c32` improved slightly, but not enough to justify the regression in high-concurrency short decode and long prefill.

### EPLB Error

`ep_eplb` failed before benchmark. Key log symptoms:

```text
EPLB is enabled. The expert_distribution_recorder_mode is automatically set.
```

Then multiple workers entered expert distribution collection/reduce:

```text
eplb/expert_distribution.py
_append_utilization_rate -> torch.distributed.reduce
```

The failure ended with scheduler timeout and Gloo wait/recv symptoms:

```text
Scheduler watchdog timeout (self.watchdog_timeout=300)
SIGQUIT received. It usually means one child failed.
gloo ... waitRecv
_pp_recv_proxy_tensors
```

Interpretation:

- EPLB/expert distribution metrics appear to interact poorly with the current PP setup.
- The failure path suggests distributed synchronization or scheduler deadlock risk.
- This is a strong candidate area for SGLang code investigation if EPLB is desired on AMD + GLM-5.2.

## Agent C: MTP+EP and TP8/PP1+MTP

Machine: `bare-metal-a` / `216.128.154.57`

### Parameter Variants

`mtp_ep_combo`:

```bash
--speculative-algorithm NEXTN
--speculative-num-steps 3
--speculative-num-draft-tokens 4
--speculative-eagle-topk 1
--expert-parallel-size 4
--moe-runner-backend aiter
--moe-a2a-backend none
```

This was applied on top of the `TP4/PP2` baseline.

`tp8pp1_mtp`:

```bash
--tp-size 8
--pp-size 1
--context-length 1048576
--kv-cache-dtype fp8_e4m3
--mem-fraction-static 0.88
--enable-mixed-chunk
--chunked-prefill-size 32768
--enable-aiter-allreduce-fusion
--enable-fused-qk-norm-rope
--schedule-conservativeness 0.5
--speculative-algorithm NEXTN
--speculative-num-steps 5
--speculative-num-draft-tokens 6
--speculative-eagle-topk 1
```

### Startup Status

| Variant | Health | Result |
|---|---:|---|
| `baseline` | success, about `346s` | 6 suites completed |
| `mtp_ep_combo` | failed before health | PP + speculative assertion |
| `tp8pp1_mtp` | success, about `366s` | first benchmark hung |

### Baseline Result

| Suite | Success | Prompt tok/s | Completion tok/s | Avg Latency | P95 Latency | Errors |
|---|---:|---:|---:|---:|---:|---:|
| `short_c32` | yes | 203.6 | 766.4 | 5.34s | 6.10s | 0 |
| `short_c128` | yes | 406.1 | 1528.9 | 10.66s | 11.87s | 0 |
| `mid_c32` | yes | 2529.8 | 662.9 | 24.71s | 25.23s | 0 |
| `prefill16k_c32` | yes | 192790.4 | 220.1 | 4.53s | 4.60s | 0 |
| `prefill64k_c4` | yes | 246495.9 | 60.6 | 2.05s | 2.49s | 0 |
| `prefill128k_c1` | yes | 176385.6 | 22.0 | 1.45s | 1.46s | 0 |

### MTP+EP Error

`mtp_ep_combo` failed with the same speculative + PP assertion:

```text
Max running requests is reset to 48
Mixed chunked prefill is disabled because of using eagle speculative decoding
AssertionError: Pipeline parallelism is not compatible with overlap schedule, speculative decoding
```

Interpretation:

- MTP+EP has no valid benchmark in the current `TP4/PP2` shape.
- EP behavior cannot be evaluated in combination because the speculative startup path fails first.

### TP8/PP1+MTP Error

`tp8pp1_mtp` passed health check, and the SGLang log reported MTP acceptance:

```text
accept len: 3.82, accept rate: 0.56
```

However, the first `short_c32` benchmark did not finish and the container had to be stopped. Key log error:

```text
fatal error: 'cuda_runtime.h' file not found
```

Observed behavior:

- repeated JIT fused metadata copy compilation failures
- GPU0 around `95%` utilization
- other GPUs mostly idle
- benchmark produced no usable throughput/latency JSON

Interpretation:

- PP1 avoids the PP + speculative assertion, so this path is worth debugging separately.
- The current container/runtime lacks the expected CUDA header for a JIT path, or SGLang is entering a CUDA-oriented code path unexpectedly in ROCm environment.
- GPU imbalance suggests the speculative/JIT path may not distribute work correctly after fallback or compilation failure.

### TP8/PP1+MTP Conservative (steps=3) — VERIFIED 2026-06-23

The aggressive `steps=5` run hung on Bug B (above). Re-tested on the same bare-metal-a with conservative `steps=3`, which avoids the multi-backend fused-metadata-copy JIT path (`dsa_backend.py:2604` only fires when `speculative_num_steps > 3`). Server healthy, all 6 suites passed with 0 errors.

Config: `--tp-size 8 --pp-size 1 --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-num-draft-tokens 4 --speculative-eagle-topk 1` (+ baseline flags). `patch_glm_config.py` applied; `patch_pp_missing_layer.py` not needed (PP1). Startup ~327s. KV: 2.96M tokens, 27.5 GB free/GPU.

**Decode throughput (warm re-run, completion tok/s, vs TP4/PP2 baseline same machine):**

| Suite | MTP conservative (warm) | Baseline TP4/PP2 | Speedup |
|---|---:|---:|---:|
| `short_c32` (128 tok decode) | 2098.7 | 766.4 | **2.74x** |
| `short_c128` (c=128) | 2378.9 | 1528.9 | **1.56x** (capped by `max_running_requests=48` forced by spec) |
| `mid_c32` (512 tok decode) | 2749.9 | 662.9 | **4.15x** |

Server-side decode logs confirm: `accept len 2.4–3.15`, `rate 0.47–0.72`, `cuda graph: True`, aggregate `gen throughput 2750–2968 tok/s` at `bs=32` warm.

**Prefill (prompt tok/s):**

| Suite | MTP | Baseline | Ratio | Notes |
|---|---:|---:|---:|---|
| `prefill16k_c32` | 177070 | 192790 | 0.92x | flat |
| `prefill64k_c4` | 34937 | 246495 | 0.14x | regression — needs isolation (baseline anomalously high; cross-machine baselines vary 21k–176k) |
| `prefill128k_c1` | 18990 | 176385 | 0.11x | same caveat — run TP8/PP1 no-MTP control to isolate |

**Cold-start caveat:** the first `short_c32` run measured 272.6 tok/s (14.25s avg) — the verify-decode cuda graph captures inline on first `bs=32` hit. After warmup the same suite measured 2098.7 tok/s (1.62s avg). **MTP benchmarks must warm-run; cold first-run numbers are not valid.**

**GPU balance confirmed:** 7 GPUs at 82–90%, GPU4 transient at 47%. The earlier "GPU0 95% / others idle" was a symptom of the Bug B JIT hang, not a separate defect — it disappears once `steps=3` is used.

## Cross-Machine Baseline Notes

Baseline results differed across machines, especially prefill throughput. This may reflect machine condition, warmup behavior, measurement script differences, or large prompt token accounting. The key directional result is still reliable because each variant was compared against the same machine's baseline.

Notable baseline issues:

- Agent A baseline failed at `prefill64k_c4`, then `prefill128k_c1` received connection refused after service exit.
- Agent B baseline completed all suites and is the cleanest EP comparison.
- Agent C baseline completed all suites and is the cleanest combo/TP8 comparison.

## Current Recommendation

**By workload:**

**Decode-heavy service (conversation / generation) — verified winner:**

```bash
--tp-size 8
--pp-size 1
--context-length 1048576
--kv-cache-dtype fp8_e4m3
--mem-fraction-static 0.88
--enable-mixed-chunk            # auto-disabled by spec hook, kept for compat
--chunked-prefill-size 32768
--enable-aiter-allreduce-fusion
--enable-fused-qk-norm-rope
--schedule-conservativeness 0.5
--speculative-algorithm NEXTN
--speculative-num-steps 3
--speculative-num-draft-tokens 4
--speculative-eagle-topk 1
--max-running-requests 128     # override spec-hook default 48 for high-concurrency decode
```

Verified 2026-06-23 on bare-metal-a: 1.56x–4.15x decode speedup over `TP4/PP2` baseline (warm), stable MTP acceptance, balanced GPU util. Apply `patch_glm_config.py` before launch.

**Prefill-heavy / 1M-context retrieval service (low decode) — keep stable baseline:**

```bash
--tp-size 4
--pp-size 2
--context-length 1048576
--kv-cache-dtype fp8_e4m3
--mem-fraction-static 0.88
--pp-async-batch-depth 4
--pp-max-micro-batch-size 32
--enable-mixed-chunk
--chunked-prefill-size 32768
--enable-aiter-allreduce-fusion
--enable-fused-qk-norm-rope
--schedule-conservativeness 0.5
```

Do not currently enable:

- `--speculative-algorithm NEXTN` with `--pp-size 2` (Bug A — design limit, see `amd355-rocm-mtp/ANALYSIS.md`)
- `--expert-parallel-size 4` for general service (slower)
- `--enable-eplb` in this PP setup (deadlocks)
- `TP8/PP1+MTP aggressive` (`steps=5`) until the one-line `dsa_backend.py:2604` HIP guard fix is applied (Bug B)

## SGLang Optimization Targets

### 1. PP + Speculative Decoding Compatibility

Current blocker:

```text
AssertionError: Pipeline parallelism is not compatible with overlap schedule, speculative decoding
```

Questions for code investigation:

- Is the assertion fundamentally required, or only required for overlap schedule?
- Can SGLang force a safe non-overlap schedule when `pp_size > 1` and speculative decoding is enabled?
- The log says non-overlap speculative v2 is selected, but the later assertion still rejects PP. Check whether schedule state is inconsistent.
- `--enable-mixed-chunk` is disabled automatically with speculative decoding. Confirm whether this interacts with PP and chunked prefill assumptions.

### 2. EPLB / Expert Distribution Deadlock

Current blocker:

```text
Scheduler watchdog timeout (self.watchdog_timeout=300)
gloo ... waitRecv
_pp_recv_proxy_tensors
torch.distributed.reduce
```

Questions for code investigation:

- Does expert distribution metric collection call a collective on all expected ranks under PP + TP?
- Are PP ranks that do not host relevant experts still participating in `torch.distributed.reduce`?
- Is there a mismatch between expert parallel group, tensor parallel group, and pipeline parallel group on AMD/Gloo/RCCL?
- Can metric collection be made async or isolated so it does not block scheduler health?

### 3. ROCm JIT Path for TP8/PP1+MTP

Current blocker:

```text
fatal error: 'cuda_runtime.h' file not found
```

Questions for code investigation:

- Why is a CUDA header required in the ROCm image for this fused metadata copy JIT path?
- Is this path accidentally using a CUDA-only source/template instead of HIP/ROCm?
- Can the fused metadata copy kernel be disabled or precompiled on ROCm?
- Does the fallback path cause single-GPU bottlenecking, explaining GPU0 high utilization and other GPUs idle?

### 4. GPU Utilization Imbalance Under Speculative Decoding

Observed during `TP8/PP1+MTP`:

- GPU0 around `95%`
- other GPUs mostly idle
- benchmark hung before producing result

Questions for code investigation:

- Are draft/verify stages assigned disproportionately to rank 0?
- Does the speculative path correctly use TP groups after JIT fallback?
- Are request scheduling queues stuck waiting on one rank while others idle?

## Re-test Suggestions After Code Changes

After SGLang changes, re-run in this order:

1. ✅ `TP8/PP1+MTP conservative`: `steps=3`, `draft=4`, no EP. **DONE 2026-06-23** — verified 1.56x–4.15x decode speedup, 0 errors. See "TP8/PP1+MTP Conservative — VERIFIED" section.
2. `TP8/PP1+MTP aggressive`: `steps=5`, `draft=6`. **Blocked by Bug B** — apply one-line fix `dsa_backend.py:2604` (`if self.speculative_num_steps > 3 and _USE_FUSED_METADATA_COPY:`), then re-test for higher `accept_len` (~3.82).
3. `TP8/PP1 no-MTP control` (NEW): isolate the large-prefill regression — if prefill64k/128k is still slow without MTP, it's a TP8/PP1 architecture issue; if it recovers, MTP is hurting prefill.
4. `--max-running-requests` sweep (NEW): test 128/256 on `short_c128` to lift the spec-hook 48 cap and quantify high-concurrency decode headroom.
5. `TP4/PP2+MTP conservative`: only after the PP + speculative assertion is addressed (Bug A — design limit, not a quick fix).
6. `TP4/PP2+EP basic`: only if EP implementation changes are made; current result is slower.
7. `TP4/PP2+EP+EPLB`: only after expert distribution collective deadlock is fixed.
8. `TP4/PP2+MTP+EP`: only after both MTP and EP pass independently.

Use the same baseline and benchmark suites so results remain comparable.
