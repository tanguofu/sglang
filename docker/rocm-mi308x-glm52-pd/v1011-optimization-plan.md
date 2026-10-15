# v1011 Optimization Plan

Baseline: `sglang-glm52-308x:v1010-src-toolargs` (helm rev92, 4 workers, PD 2P2D).
Scope of this document: what to port, in what order, and how to prove it is safe.

## 0. Two facts that must not be forgotten

1. **Source branch and chart branch are different.**
   - Image source: `mi308x-1p1d-optimize` (this is what ti-builder
     `/data/sglang-build/sglang-unified` builds, HEAD `2354569318`).
   - Chart: **`glm53-perf-20`**. Rendering with the `mi308x-1p1d-optimize` chart
     re-introduces the AITER OPUS JIT block that production does not run.
     Always render + diff against the live manifest before any `helm upgrade`.
2. **PD image changes that touch KV must roll all 4 workers together.**
   Rolling `decode-1` alone with `v1009` produced ~50% garbage output and no error.
   `env`-only changes may be canaried one node at a time; image changes may not.

## 1. Corrections to the previous plan

| Previous item | Verdict | Evidence |
|---|---|---|
| P1: "TOPK_V2 drift, pod rebuild silently loses fused topk" | **Not a defect** | `server_args.py:5293` does `envs.SGLANG_OPT_USE_TOPK_V2.set(False)` for `is_hip() and is_deepseek_dsa()`. GLM-5.x is DSA, so the env is inert regardless of its value. Align it as hygiene during the v1011 rollout; do not spend a change window on it. |
| "Port 3 small diffs" | **Wrong** | `git apply --check` fails for all five candidates. Every one needs manual conflict resolution. |
| "Use the optimize worktree chart" | **Wrong** | See §0.1. |

## 2. Baseline first, and how to measure

Nothing below is accepted without the baseline in `§7`. The measurement rules that
the older plans violated:

- cold and warm are reported **separately** (they differ by ~200x)
- every phase reports the cache hit rate; TTFT without a hit rate is meaningless
- n>=3, report median / min / max, never a single sample
- correctness gate runs **before** any perf number is trusted
- operator-level gains are judged from **kernel profile `dur`**, not end-to-end TTFT
  (run noise is ±7%, so end-to-end is only a no-regression gate)

## 3. Port backlog (all verified missing from `mi308x-1p1d-optimize`)

| # | Commit | What | Applies to us? | Conflict / dependency |
|---|---|---|---|---|
| 1 | `790551c382` (#35872) | Skip full-vocab softmax in EAGLE `topk==1` draft on ROCm | **Yes, highest confidence.** Live decode is `speculative_algorithm=EAGLE`, `eagle_topk=1`, `steps=2`; `spec_info.py:301` selects `EAGLEWorkerV2`, exactly this patch's target | `eagle_worker_v2.py` conflicts; needs new file `python/sglang/kernels/ops/speculative/topk1.py` (absent in our tree) |
| 2 | `db32673b78` (#41258) | GLM DSA NextN draft declares its own shared-expert fusion architecture | **Yes.** `glm4_moe.py:1448` declares it for the main model, but the NextN class at `:1451` inherits the DeepSeek name, which never matches the gate, so the draft shared expert is never fused | `test_glm_nextn_moe_ptpc.py` does not exist here. Second hunk depends on `should_apply_glm_nextn_moe_ptpc`, also absent. Take the first hunk, or port the helper too |
| 3 | `37114afe20` (#42982) | Saturate FP8 KV commits instead of reproducing e4m3fnuz NaN overflow | **Yes, defensive correctness.** We run `--kv-cache-dtype fp8_e4m3` | `cache_move.py:57`, `fp8_kernel`, `memory_pool.py` conflict. **Touches KV values → all-4 rollout** |
| 4 | `461717cd48` | Check each tool schema once, not per request | Only pays off for large tool sets (81 tools ~150ms). Our codex traffic carries few tools | Conflicts at `serving_chat.py:125`, i.e. our own tool-args patch. Manual merge |
| 5 | `aa0ce20201` (#41725) | GLM-5.2 decode-shaped MoE/MLA tiles, split speculative softmax | **Unproven for us.** Conflict is in `fused_moe_triton_kernels.py`, but our profile shows decode MoE runs the aiter `fmoe_bf16_blockscaleFp8_g1u1_vs_ps_silu_64x256` path, not triton | Verify the path before spending effort |

## 4. Not in scope for v1011

- `--dsa-prefill-backend aiter` — A/B done, CUDA crash, ruled out.
- FlyDSL `mfma_r4_w4` override — micro-benchmark says it is the fastest variant;
  aiter's auto would pick `mfma_r2_w2`, which is 39% slower. The override protects us.
- `--hicache-size 200` — deprioritized. Host pool is at 40% (144: 769,920/1,909,696),
  not the 99.5% seen on 10-08. Still open, but the binding constraint is that
  `host_hit` and `storage_hit` are both **0** on both prefills: L2/L3 are write-only.
- Prefill CK dense GEMM, MTP `steps=3`, paged FlyDSL decode — previously ruled out.

## 5. Execution order

1. Delete junk pods (`sglang-temp-router-baseline/canary`, `sglang-gdr-*`,
   `debug-priv-a`). Zero risk, independent of helm. Confirm ownership first.
2. v1010 baseline bench + **decode-side kernel profile** (§7, §8).
3. Port #1 -> #2 -> #3 in that order, on `mi308x-1p1d-optimize`.
   #4 optionally; #5 only after path verification.
4. Build `v1011-src-ports` on ti-builder, push.
5. Roll **all 4 workers** (OnDelete, one pod at a time, drain first), because #3
   touches KV. Align `SGLANG_OPT_USE_TOPK_V2` in the same upgrade.
6. 2h observation window, then re-run the same bench and compare against §7.

## 6. Gates

**L0 effectiveness**
- 4 workers on the same tag; `grep -c FIX(tool-args-tolerant)` == 1 in each container
- no `WARNING: ... patch failed` in startup logs
- #1: `topk1` kernel present in the decode profile
- #2: draft `shared_experts_fusion_disable_reason` no longer reported for the NextN draft
- env really reached the pod and was not overridden by `server_args`

**L1 correctness (blocking, roll back on failure)**
- manual PD matrix: 2 prefill x 2 decode, all OK, no garbage
- router x12 `17*23` -> 12/12
- 200K needle x3 all correct
- 6 concurrent arithmetic all correct
- bad / empty `tool_calls.arguments` replay -> 200, not 400, no hang

**L2 benefit**
- profile: target kernel `dur` before/after (only meaningful at >=5% share)
- cold TTFT no regression (>= -2%), warm TTFT and hit rate no regression
- decode tok/s and accept length at 300K context

**L3 stability (2h)**
- zero restarts, router zero 4xx/5xx, all four `/health` 200

**Rollback**: `helm upgrade` back to `v1010-src-toolargs` using the `glm53-perf-20`
chart, then delete pods. Keep the old tag published.

## 7. Baseline measured on v1010 (2026-10-12)

Router path, n=3, prompts sized by an inaccurate chars/token estimate so the
labels below are the *actual* token counts reported by `usage.prompt_tokens`:

| Metric | v1010 |
|---|---|
| cold 15K TTFT | 12.18s (min 4.42 / max 61.81) — **unusable, see §9** |
| cold 28K TTFT | 20.02s (min 10.71 / max 25.41) |
| cold 60K TTFT | 58.16s (min 18.39 / max 58.21) |
| warm 28K TTFT | 0.43s (0.34 / 0.35 after the first miss at 26.08s) |
| 200K needle | **4/4 correct**, 70.7 / 226.0 / 70.9 / 71.0s |
| concurrent 6 arithmetic | 6/6 OK, wall 28.8s |
| hit rate | 144 = 86.3% ; 152 = 0.0% (152 rebuilt for the aiter A/B, cache cold) |

Two harness bugs were found and fixed while producing this table; do not reuse the
earlier scripts:

- the needle check truncated `reasoning_content` to 60 chars, so the code, which
  the model states later in its reasoning, was never seen. Result was a false 0/3.
- the chars/token calibration was 2.9; the real value for this filler text is
  **6.1**. Every earlier "32K/64K/128K" figure in older notes is actually about
  half that many tokens.

## 8. Decode profile measured (2026-10-12, decode-0 / 132, 10 steps)

| kernel | share | count |
|---|---|---|
| `aiter::fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_16x128_flat_pf3` (MoE) | 15.6% | 85 |
| `main_kernel` (tilelang DSA decode attention) | 8.8% | 194 |
| `aiter::allreduce_fusion_kernel_1stage` | 7.9% | 174 |
| CK GEMMs (`Cijk_*`, four shapes) | ~13.5% | — |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 5.3% | 366 |
| `__amd_rocclr_fillBufferAligned` (memset) | 4.5% | 297 |
| **`cunn_SoftMaxForwardGmem`** ← #35872 target | 3.4% | 19 |
| **`reduce_kernel<ArgMaxOps>`** ← #35872 target | 2.9% | 10 |
| `aiter::kn_entry_2c_sbhd_cached_indirect_inplace` (MLA) | 2.6% | 128 |
| **`_fused_append_shared_experts_kernel`** ← #41258 target | 1.1% | 75 |

Numeric gates this gives us:

- **#35872 = 6.3%** of decode kernel time (softmax 3.4% + argmax 2.9%). Above the
  5% bar, so it is worth doing and worth measuring.
- **#41258 = ~1.1%**. Small; do it because the diff is 11 lines, not because it moves
  throughput.
- **#41725 is confirmed irrelevant**: decode MoE runs the aiter `fmoe_bf16_a16_...`
  kernel, not the triton path that commit patches.
- New observation: 297 memset calls in 10 steps (~30/step, 4.5%). Worth a separate
  look; nothing in the current backlog targets it.

## 9. Blocking finding: prefill-0 (node 144) is 3.25x slower, and it is hardware

Pinning the prefill (manual PD, same payload, same decode) removes the routing
variable and shows the router-path variance was never noise:

```
prefill-0 144   54.55s  (53.99 / 55.05)  -> 1,089 tok/s
prefill-1 152   16.81s  (16.81 / 16.87)  -> 3,532 tok/s
```

Within one prefill the spread is under 1%. The 3.25x gap is real and constant.

Cause: **node 144 GPU[0] is thermally faulted.**

```
144: GPU[0] junction 100 C, memory 77 C, sclk 1383 MHz under load drops to 1251 MHz
     GPU[1..7]         49-53 C, sclk 1437-1443 MHz
152: all GPUs          51-55 C
```

A single GPU at 100 C while its seven siblings sit at 50 C points at that card's
cooling (fan / cold plate / thermal pad), not node airflow. Under TP8 every step
synchronises through all-reduce, so one throttled rank slows the whole group.

This also retroactively explains the historic `prefill-0` crashes
(`Failed to CUDA calloc 6291456 bytes` at the sampler all-reduce) and part of the
144-to-172 RDMA flakiness attributed to the transport.

**Do not tune software on 144 until this is resolved** — every A/B that lands on
144 will read 3x worse and will be misattributed. Recommended actions, in order:

1. Drain prefill-0 and inspect / re-seat GPU 0 on node 144 (or RMA the node).
2. Until then, treat 152's numbers as the software baseline and record 144 as
   degraded rather than comparing the two.
3. Re-run §7/§8 after the fix; the numbers will change.

Caveat: the 13% clock deficit does not by itself account for 225% slower prefill,
so thermal throttling is the prime suspect but may not be the whole story. A second
contributor is that 144's host pool reached 85% (1,627,456 / 1,909,696) during the
bench versus 56% on 152, so eviction pressure is also in play. Confirm by comparing
the two prefills again once GPU 0 is healthy.

## 10. What v1011 actually delivered (rolled out 2026-10-12, helm rev93)

Built `v1011-src-ports` from `mi308x-1p1d-optimize` @ `cf00030e30`, containing
#35872 and #41258. Rolled decode-1 -> decode-0 -> prefill-0 -> prefill-1, one pod
at a time, then `helm upgrade` with the `glm53-perf-20` chart so values is the
source of truth again (rev93, 6 `workerImage` lines, no pod restarted by helm).

### Effect, measured at the kernel level

| kernel | v1010 (132) | v1011 (172) |
|---|---|---|
| `cunn_SoftMaxForwardGmem` (full-vocab draft softmax) | 3.4%, x19 | **absent** |
| `reduce_kernel<ArgMaxOps>` | 2.9%, x10 | 3.0%, x10 |
| `_fused_append_shared_experts_kernel` | 1.1%, x75 | 1.5%, x94 |

- **#35872 worked**: the draft's full-vocab softmax is gone.
- **#41258 did not**: the shared-expert append kernel is still there, and its count
  went up. Porting only the architecture declaration is not enough; the upstream
  commit's second hunk (and the `should_apply_glm_nextn_moe_ptpc` helper that this
  tree lacks) is required. Treat this one as unfinished, not done.

### Effect, measured end to end

Pinned A/B (prefill=152, 28K context, max_tokens=384, completion capped at 384 so
every run does identical work, requests interleaved between the two decodes):

| run | 132 | 172 | pooled median |
|---|---|---|---|
| v1010 / v1010 (n=10) | 12.21s | 12.29s | **12.215s** |
| v1010 vs v1011 (n=8 each) | 12.04s | 12.11s | — |
| v1011 / v1011 (n=16) | 12.20s | 11.98s | 12.055s |
| v1011 / v1011 final (n=16) | 12.14s | 12.02s | 12.08s |

**End-to-end gain: about -1.2%, i.e. inside the noise band.** The kernel-level win
is real but decode is only ~4s of the 12s wall at this shape, so 3.4% of decode
kernels is ~1% of the request.

This is the expected outcome and it is worth stating plainly: a correct change can
be visible in the profile and invisible end to end. Do not claim a throughput
improvement from this release.

### Lossless verification (all green)

| gate | result |
|---|---|
| L0 markers in every container | `draft_topk1_argmax_only` x2, NextN fusion decl x1, tool-args x1 |
| L0 startup | no errors; only benign `Ignore import error ... sarashina2_vision` and AITER JIT cache-publish notes |
| L1 PD matrix 2x2, x2 each | **8/8 OK**, no garbage |
| L1 router x12 | **12/12** |
| L1 200K needle x2 | **2/2 correct** (127.6s / 70.9s) |
| L1 bad / empty tool arguments | 6/6 HTTP 200, 0.4-0.6s, no hang |
| L3 | 4 pods, **0 restarts**; router 30min **0 errors / 200 requests** |
| L3 values alignment | `helm get values` shows 6x `v1011-src-ports`, rev93 deployed |

### Deliberately not shipped

#42982 (FP8 KV saturation). No throughput benefit, changes KV values, and would
have forced the whole rollout into the "touches KV encoding" risk class. It needs
its own release with its own correctness run.

## 11. Stability stress test (2026-10-12, after the v1011 rollout)

Through the production router path. Mixed workload: 60% short turns (128 out),
30% 8K context (256 out), 10% 60K context (128 out). Concurrency 8, 8 minutes.

```
requests=1057   errors=0 (0.0%)   wall=490s   qps=2.16
wall  p50=2.35s  p90=3.55s  p95=4.05s  p99=33.74s  max=142.95s
ttft  p50=0.45s  p90=0.76s  p95=0.92s  p99=30.71s  max=142.47s
prompt_tokens=8,262,546   completion_tokens=137,899   agg_out=281.2 tok/s
short n=633 p50=2.19s p95=3.00s      long n=424 p50=2.92s p95=4.44s
```

Stability markers, all clean:

- 4 worker pods, **0 restarts**; router **0 errors** over 1117 requests
- `hicache_dropped_tokens_total` = 0 on both prefills
- hit rate after warm-up: 144 = 96.3%, 152 = 72.5%
- **host_hit still 0 on both.** L2 remains write-only. `storage_hit` finally moved
  off zero on 152 (59,328) but is negligible.
- host pool usage is now low: 144 = 67,200 / 1,909,696 (3.5%),
  152 = 906,240 / 1,911,040 (47%). This is well below the 99.5% seen on 10-08, which
  further deprioritises `--hicache-size`.

Two things the test exposed:

1. **A long tail**: p99 is 33.7s and max is 143s against a p50 of 2.35s. During the
   run, prefill-0 (144) showed `num_queue_reqs=3` while prefill-1 (152) sat at 0.
   `cache_aware` keeps pinning cached prefixes to 144, which is the thermally
   degraded node, so the requests most likely to be cheap are being sent to the
   slowest instance.
2. **Warm-up dominates short runs**: only 44 requests completed in the first 180s,
   then ~200/min. Any benchmark shorter than a few minutes on a freshly restarted
   prefill measures cache warm-up, not steady state.

## 12. Revised plan (what is actually worth doing next)

| # | Item | Why | Effort |
|---|---|---|---|
| 1 | Fix or remove node 144 GPU 0 | 3.25x on half of prefill capacity. Larger than everything else combined | hardware |
| 2 | Stop `cache_aware` pinning to the degraded node | Would cut the p99 tail without touching hardware. Options: drain 144 from the router pool until GPU 0 is fixed, or weight it down | config, needs a window |
| 3 | Finish #41258 (port the `should_apply_glm_nextn_moe_ptpc` helper too) | Currently a no-op; ~1.1% of decode kernels | small |
| 4 | Ship #42982 on its own | Correctness hardening for FP8 KV; no throughput effect | small + own release |
| 5 | Investigate `__amd_rocclr_fillBufferAligned` | 4.5% of decode kernel time, ~30 memsets per step, nothing in the backlog targets it | unknown |
| 6 | ~~Make L2 readable, or turn it off~~ | **Withdrawn.** L2 works; see §13 | — |
| 7 | ~~`--hicache-size`~~ | **Withdrawn.** Current 85.8 GB pool survives 2x eviction pressure; enlarging carries a documented RDMA registration risk. See §13 | — |

Do not restart prefills casually while 144 is degraded: every restart clears that
instance's cache and, for prefill-1, leaves the service running on the slow node
alone for ~20 minutes.

## 13. Post-hardware-fix state, and two withdrawn items

Node 144 was repaired on 2026-10-15 (GPU 0 back to 47 C, in line with the other
seven). Both prefills are now identical:

| scenario | 144 | 152 |
|---|---|---|
| 60K cold prefill | 16.82s / 3,531 tok/s | 16.77s / 3,541 tok/s |
| 200K cold prefill | 66.17s / 2,990 tok/s | 66.35s / 2,982 tok/s |
| device hit | — | 0.63s |
| L2 hit after eviction | — | 2.21s (vs 16.75s cold) |

Stress after the fix: 1535 requests, 0 errors, qps 3.13, p50 2.34s, **p99 5.85s**
(was 33.74s), max 24.57s, agg_out 403 tok/s.

### L2 is not broken - earlier conclusion was wrong

`host_hit` was 0 in every earlier measurement, which I read as "L2 is write-only,
we pay backup cost for a layer we never read". That was a misreading.
`hiradix_cache.match_prefix` only counts L2 when the *device* node has been
evicted:

```python
while last_node.evicted:
    host_hit_length += len(last_node.host_value)
```

Under light traffic the device cache holds everything (65-96% device hit), so L2
is legitimately never consulted. A direct probe proves it works:

```
probe A cold        16.75s
probe A again        0.63s   (device hit)
22 unique 60K fillers (406s) to evict A
probe A again        2.21s   host_hit delta = 59,328   <- L2 returned the prefix
```

### `--hicache-size` is not needed

Doubling the eviction pressure (44 fillers, ~2.6M tokens vs a 1.91M-token pool)
still returned A in 2.30s. The 85.8 GB pool is adequate. Enlarging it has a
documented RDMA memory-registration failure mode, so with no demonstrated benefit
and a real risk, the change is withdrawn.

### Other items confirmed to have no headroom

- `SGLANG_DSA_USE_TOPK_V2_PACKED=1` is **already set** on both prefills.
- `SGLANG_DSA_TOPK_BROADCAST` needs PyNCCL *during CUDA graph capture*; HIP CUDA
  graph capture has already bitten us once, so it is not worth the risk.
- The "superlinear long-context degradation" was an artifact of the wrong
  chars/token calibration. Measured properly, 60K -> 200K is only **-15.8%**
  (3,541 -> 2,982 tok/s), consistent with DSA's indexer cost growing with context.

### What is genuinely left

Only small items: finish #41258 (whose effect could not be confirmed from the
profile, because `_fused_append_shared_experts_kernel` lives on the triton MoE
path while our decode runs aiter), ship #42982 for FP8 KV robustness, and look at
`__amd_rocclr_fillBufferAligned` (4.5% of decode kernels, ~30 memsets/step).
None of these is expected to move throughput by more than a few percent.
