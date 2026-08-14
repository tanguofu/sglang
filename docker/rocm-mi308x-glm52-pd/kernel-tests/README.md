# gfx942 kernel unit tests

Self-contained pytest files for the GLM-5.2 MI308X hot path. No full model.

| File | Kernel | Shard |
|---|---|---|
| `test_fp8_mqa_logits.py` | FlyDSL + Triton fp8 MQA logits | A / 152 |
| `test_fused_store_index_cache.py` | DSA fused store (HIP, not skipped) | A / 152 |
| `test_rmsnorm.py` | Aiter RMSNorm | A / 152 |
| `test_aiter_gemm_bf16.py` | Aiter tuned GEMM bf16 | B / 172 |
| `test_aiter_fmoe.py` | Aiter MoE GEMM / fmoe symbol | B / 172 |
| `test_rope.py` | Aiter RoPE | B / 172 |

```bash
KERNEL_TEST_SHARD=a ./run_all.sh   # or b / all
```

Jobs use the live A-group image so they can run while the new source-build image compiles. After `pd-l3-20260814` is ready, change the Job image and re-run.
