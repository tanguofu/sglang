# MTP 3/4/1 vs 2/3/1 Benchmark Comparison

**Date**: 2026-07-05
**Hardware**: 8× AMD MI355X (309GB VRAM each), AMD EPYC 9575F 64-Core
**Docker Image**: `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260702`
**Model**: GLM-5.2-FP8

## MTP Configurations

| Config | `--speculative-num-steps` | `--speculative-num-draft-tokens` | `--speculative-eagle-topk` |
|-------|--------------------------|----------------------------------|---------------------------|
| 3/4/1 (optimized) | 3 | 4 | 1 |
| 2/3/1 (upstream default) | 2 | 3 | 1 |

## Benchmark Results

| Test | Concurrency | Input | Output | Master (3/4/1) tok/s | Node-1 (2/3/1) tok/s | Diff |
|------|------------|-------|--------|----------------------|----------------------|------|
| decode_short | 1 | 0 | 1024 | 167.6 | 148.7 | +13% |
| decode_short | 8 | 0 | 1024 | 939.0 | 370.7 | +153% |
| decode_2k | 1 | 2048 | 1024 | 180.9 | 146.9 | +23% |
| decode_2k | 8 | 2048 | 1024 | 769.4 | 251.9 | +205% |
| qa_thinking | 1 | 0 | 256 | 173.2 | 147.7 | +17% |
| medium_ctx | 4 | 4096 | 256 | 514.6 | 432.6 | +19% |

## MTP Runtime Metrics

### Master (3/4/1)

| Metric | Range |
|--------|-------|
| accept rate | 0.58-0.78 |
| accept len | 2.75-3.35 |
| DSA fused store JIT | compile fail → fallback (expected) |

### Node-1 (2/3/1)

| Metric | Range |
|--------|-------|
| accept rate | 0.80-0.97 |
| accept len | 2.60-2.95 |
| DSA fused store JIT | compile fail → fallback (expected) |

## Conclusion

- **3/4/1 outperforms 2/3/1 across all scenarios** (+13% to +205%)
- High concurrency gains are dramatic: decode C=8 +153%, decode_2k C=8 +205%
- 2/3/1 has higher per-step accept rate (0.80-0.97 vs 0.58-0.78) but fewer steps means fewer tokens accepted per round
- 3/4/1 accepts ~3 tokens/round vs ~2.7 tokens/round for 2/3/1, resulting in higher throughput
- **Recommendation**: Use 3/4/1 as the default MTP configuration

## DSA Fused Store JIT Note

`patch_fused_store_fp8_convert.py` must NOT be added to the CMD. Although it fixes the JIT compile error in `fused_store_index_cache.cuh`, the `__hip_cvt_float2_to_fp8x2()` conversion produces incorrect FP8 values, corrupting KV cache and causing MTP accept rate to drop from ~0.7 to ~0.05. Without the patch, the kernel fails to compile and falls back to a correct path.
