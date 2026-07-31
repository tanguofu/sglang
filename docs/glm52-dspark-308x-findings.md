# GLM-5.2 DSpark on MI308X (gfx942) — Official PR Port Findings

**Date**: 2026-07-31
**Branch**: `feat/glm52-dspark-308x-official`
**Base**: upstream/main (937c77cf50) + PR #31047 + #31260 merged

## Goal
Port official GLM-5.2 DSpark PRs (#31047 config/block correctness + #31260 ROCm enablement) to MI308X (gfx942), verify DSpark can achieve significant decode speedup.

## What was done
1. Merged PR #31047 (14 files, +885/-67) and #31260 (20 files, +1111/-108) onto upstream/main
2. Resolved 1 merge conflict: `dsa_topk_backend.py` — kept `should_use_topk_v2()` call but added `not _is_hip` to the method definition (satisfies #31260's HIP-disable intent while keeping #31047's encapsulation)
3. Built image `mirrors.tencent.com/ti-platform/sglang-glm52-dspark-308x:latest` based on `lmsysorg/sglang-rocm:v0.5.15.post1-rocm720-mi30x-20260718`
4. Deployed to groupb node-21.151.225.172 (MI308X, gfx942), RedHat draft checkpoint downloaded
5. Verified correctness: DSpark server starts, requests return correct output, AR/AL metrics produced

## Checkpoint
- Target: `zai-org/GLM-5.2-FP8` (already on groupb)
- Draft: `RedHatAI/GLM-5.2-speculator.dspark` (schema B, dense draft, 3 layers, block_size=8, vanilla markov)
- Downloaded to `/data/model/glm52-dspark-redhat` on node-21.151.225.172

## Results
- DSpark draft model loads: `DSparkDraftModel`, gamma=8, verify_num_draft_tokens=9, VanillaMarkov
- DSpark target-verify runs on HIP via ragged top-k path
- **accept_len: 2.58–3.15 (avg ~2.8)**
- **accept_rate: 0.20–0.27**
- gen throughput: 9.59–11.69 token/s (CUDA graph disabled)

## 2 bugs found and fixed

### Bug 1: `dsa_indexer.py:1147` assert fires on target-verify (fixed)
PR #31260 routes HIP target-verify through `_get_topk_ragged`, but the assert at line 1147 still required `extend_seq_lens_cpu is not None`. target-verify batches don't populate `extend_seq_lens_cpu`, so the assert fired on every DSpark target-verify forward.

**Fix**: Relax assert to only require `seq_lens_cpu` when `is_target_verify()`. Downstream code uses `indexer_seq_lens_cpu` from metadata, not `extend_seq_lens_cpu`, so safe.

Commit: `d6326dd3c5 fix(dsa_indexer): relax extend_seq_lens_cpu assert for target_verify on HIP`

### Bug 2: CUDA graph capture crash (workaround only, NOT properly fixed)
`_get_topk_ragged` → `metadata.get_indexer_kvcache_range()` returns None during CUDA graph capture. Root cause: #31260 routes HIP target-verify through `_get_topk_ragged`, but during CUDA graph capture `forward_mode` is decode (not target_verify), so `_cal_indexer_k_start_end` returns None → `indexer_k_start_end` is None → unpack fails.

**Workaround**: `--disable-cuda-graph` in start_server.sh. This makes DSpark work but kills performance (9-11 token/s vs production ~500+ token/s with CUDA graph).

**Proper fix needed**: Either (a) populate `indexer_k_start_end` during capture, or (b) guard `_get_topk_ragged` against None `indexer_k_start_end` during capture, or (c) skip ragged path during capture and use paged. This is the main blocker for production.

## gfx942 vs gfx950
- upstream/main already merged 6/14 of 0708-opt patches (JIT imports, DSA indexer fusion, dual stream, alt_stream, fp8 is_shuffled)
- Remaining 8 patches (6 perf + 2 correctness) not needed for DSpark correctness
- topk v2 disabled on all HIP via `not _is_hip` (avoids gfx942 topk_v2 crash)
- No gfx950 hardcoding in ported code — uses generic `_is_hip`

## vs EAGLE MTP baseline
| | DSpark (this work) | EAGLE MTP (existing) | Official MI350 |
|---|---|---|---|
| accept_len | ~2.8 | 2.82 | ~5.17 |
| CUDA graph | disabled | enabled | enabled |
| 1M context | not verified | verified | not verified |
| AIME | not tested | 100% | not tested |
| Production ready | no | yes | no (S4 not started) |

## Conclusion
DSpark works on MI308X but **does not achieve significant speedup** yet:
1. Must disable CUDA graph → performance far below production
2. accept_len ~2.8 comparable to EAGLE MTP, no advantage
3. Official MI350's AL ~5.17 depends on S1b/S2 perf stack (not in any PR, in tanth47 fork staging)
4. RedHat checkpoint is workload-sensitive (code workload may be worse)

## Next steps for production (if pursued)
1. Fix CUDA graph capture for DSpark target-verify (proper fix, not --disable-cuda-graph)
2. Port S1b/S2 perf stack (SPS/STS/overlap scheduling) from tanth47 fork staging
3. Compare throughput vs EAGLE MTP with CUDA graph enabled
4. Test on code workload (RedHat checkpoint workload-sensitive)

## Files
- `docker/rocm-mi308x-glm52/Dockerfile` — image build (DSpark verify, no 0708-opt verify)
- `docker/rocm-mi308x-glm52/start_server.sh` — DSpark params, --disable-cuda-graph
- `docker/rocm-mi308x-glm52/dspark-deploy.yaml` — StatefulSet, nodeSelector .172, toleration sglang-1pd-B-group
- `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` — assert fix (commit d6326dd3c5)
- `python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py` — merge conflict resolution
