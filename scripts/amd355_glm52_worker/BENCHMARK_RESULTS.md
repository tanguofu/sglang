# Worker 0706 vs Master 0702 Benchmark Results

## Date: 2026-07-06

## Configuration
- **Worker**: AMD MI355X, 0706 image (`lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260706`)
- **Master**: AMD MI355X, 0702 image (`lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260702`)
- **Model**: GLM-5.2-FP8, TP=8, MTP steps=3, draft_tokens=4, eagle_topk=1
- **Patches**: patch_sglang_glm52_rocm_all.py + patch_0706_supplement_v4.py

## Key Fix: cos_sin_cache Alignment (v3/v4)
The 0706 image used `self.rotary_emb.cos_sin_cache` directly at 2 usage sites
(lines ~2009, ~2125) instead of `self._indexer_cos_sin_cache` (pre-processed to
float32 + 2D). This caused:
1. decode_short c8 to drop from ~910 to ~390 (2.3x regression)
2. Lower MTP accept rate due to numerical precision differences

Fix: Changed usage sites to `self._indexer_cos_sin_cache` + unconditional
`_cos_sin_cache_val` storage in `__init__` (matching master 0702).

## Benchmark Results (after proper warmup)

| Test | Master (0702) | Worker v4 (0706) | Worker vs Master |
|------|-------------|-------------------|------------------|
| decode_short c1 | 166.2 | 170.5 | **102.6%** ✅ |
| decode_short c8 | 945.4 | 921.4 | 97.5% |
| decode_2k c1 | 163.6 | 161.3 | 98.6% |
| decode_2k c8 | 794.6 | 817.3 | **102.9%** ✅ |
| qa_thinking c1 | 178.5 | 164.7 | 92.3% |
| medium_ctx c4 | 500.5 | 490.1 | 97.9% |

## MTP Metrics
| Metric | Master | Worker |
|--------|--------|--------|
| spec_accept_rate | 0.692 | 0.60-0.65 |
| spec_accept_length | 3.075 | 2.75-2.86 |

## Quality Evaluation
- **Exact match**: 8/8 (100%) — worker outputs identical to master
- **Test categories**: Math, Code, Science, Network, Creative, Logic, Summary, Translation

## Warmup Note
First benchmark run after restart shows low decode_short c8 (~390) due to
insufficient CUDA graph warmup. Second run shows full performance (~910-920).
Recommend 2+ warmup iterations before benchmarking.

## Remaining Gaps
1. MTP accept rate 0.60-0.65 vs 0.692 — likely 0706 image kernel selection differences
2. qa_thinking c1 at 92.3% — short generation with thinking tokens
3. All gaps are within 8% of master performance

## Patches Applied (v4 supplement)
1. S1: DUAL_STREAM_TOKEN_THRESHOLD = 1024 (was 0 on HIP)
2. S2: cos_sin_cache init unconditional (matches master)
3. S3: cos_sin_cache usage site 1 → _indexer_cos_sin_cache
4. S4: cos_sin_cache usage site 2 → _indexer_cos_sin_cache
5. S5: metadata None guard for breakable graph
