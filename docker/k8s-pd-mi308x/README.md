# GLM-5.2 PD (1p1d) MI308X Deployment Configs

Optimized PD (Prefill-Decode disaggregation) Kubernetes configs for GLM-5.2 on AMD MI308X with bnxt_re RDMA.

## Files

| File | Description |
|------|-------------|
| `pd-decode.yaml` | Decode pod spec (Batch 1 optimized) |
| `pd-prefill.yaml` | Prefill pod spec |
| `pd-router.yaml` | PD router pod spec |
| `values-1p1d-batch1.yaml` | Helm values for glm52-pd chart |

## Batch 1 Optimizations (2026-07-25)

| Parameter | Old | New | Effect |
|-----------|-----|-----|--------|
| `cuda-graph-max-bs-decode` | 16 | 32 | Eliminates eager fallback at conc>16 |
| `cuda-graph-bs-decode` | 1-16 | + 20 24 32 | Extended graph coverage |
| `max-running-requests` | 128 | 32 | Align with CUDA graph capacity |
| `schedule-conservativeness` | 1.0 | 0.5 | Smoother batch management |
| `speculative-num-steps` | 3→2 | 3 (restored) | accept_length 2.58→2.97 (+15%) |
| `num-continuous-decode-steps` | 1 | 2 | Reduce scheduler overhead |
| `optimistic-prefill-attempts` | 0 | 1 | Overlap transfer with prefill |
| `mem-fraction-static` (prefill) | 0.85 | 0.90 | More KV cache capacity |

## Benchmark Results

Peak throughput: **524.5 tok/s** @ conc=32 (vs initial 444.1, +18%)

| Conc | Initial tok/s | Batch1 tok/s | vs 1tp8 |
|------|--------------|--------------|---------|
| 1 | 64.2 | 62.3 | 1.20x |
| 4 | 195.9 | 200.9 | 1.02x |
| 8 | 121.3 | 332.6 | 0.93x |
| 16 | 356.5 | 496.1 | 1.01x |
| 32 | 444.1 | 524.5 | 1.13x |
| 64 | N/A | 507.1 | 0.90x |

## Base Version

Branch `308x-pd-v0516` is based on upstream SGLang **v0.5.16** (tag `fdebc938f`, released 2026-07-25) with custom patches cherry-picked on top:

| Commit | Description |
|--------|-------------|
| `5a0d1cff7` | host staging fallback (conn.py), /v1/responses routing (pd_router.rs), vendored openai-protocol |
| `8dcae59eb` | Batch 1 deployment configs (this directory) |

### v0.5.16 new fixes (vs old base bbb5702a3)

These fixes are NEW in v0.5.16 relative to the previous base (`bbb5702a3`, Jul 14). Verified via `git merge-base --is-ancestor`:

**GLM-5.2 / MTP:**
| PR | Description |
|----|-------------|
| #30839 | GLM-5.2 MTP IndexShare stability across PD + CUDA graph replay |
| #30992 | GLM-5.2 MTP index sharing with prefill CP |
| #28416 | [GLM5] FlashInfer TRT-LLM MoE direct write (MoE perf) |
| #30506 | [AMD] Disable DSA fused top-k v2 on ROCm for GLM-5.2 / DSv3.2 |
| #31577 | GLM5.2 Cookbook LayerSplit docs |

**AMD / ROCm / gfx942:**
| PR | Description |
|----|-------------|
| #26852 | [AMD] Reuse fused FP8 KV cache write on aiter prefill/decode |
| #29508 | [AMD] Fix quickreduce acc error in cudagraph mode |
| #31688 | [AMD] Fix ROCm fused KV and KDA paths |
| #31675 | [AMD] Fix DeepSeek MLA prefill shape mismatch on HIP eager fallback |
| #31368 | [AMD][PD] Fix early-send cached-prefix KV racing prefill forward on mori |
| #30940 | [AMD] Gate TP4 oproj/qkv CK block-FP8 GEMM shapes to Triton |

**Speculative decoding (EAGLE/MTP):**
| PR | Description |
|----|-------------|
| #30947/#30948 | [EAGLE] Fuse topk=1 draft postprocess + TP vocab-parallel embedding |
| #31614 | Fix multi_layer_eagle rotate_input_ids kernel registration |
| #31620 | Replace torch.multinomial with native torch ops in rejection sampling |
| #32254 | Fix inkling multi-layer MTP draft extend cuda graph |

**PD / Disaggregation:**
| PR | Description |
|----|-------------|
| #31306 | [PD] Fix send_multipart blocking after prefill failure |
| #31584 | Fix KV Manager Disaggregation Heartbeat Checker |

### Already in old base (unchanged)

- `optimistic_prefill_attempts` server arg (PR #30951 rename) — already in `bbb5702a3`
- Optimistic prefill inflight-queue hang fix (PR #31075) — already in `bbb5702a3`

### Fork-only custom patches (NOT in upstream)

- #29421: DSA Cache Layer Split under Prefill CP (`8e54517f0` on fork only) — requires `--enable-prefill-cp --cp-strategy interleave --enable-dsa-cache-layer-split`; NOT yet enabled in deployment

## Image Build

### SGLang image (decode + prefill)

```bash
docker build \
  --build-arg BRANCH_TYPE=local \
  --build-arg GPU_ARCH=gfx942 \
  --build-arg BUILD_TYPE=srt \
  --build-arg SETUPTOOLS_SCM_PRETEND_VERSION=0.5.16 \
  -t mirrors.tencent.com/ti-platform/sglang-glm52-308x:v0516-batch1 \
  -f docker/rocm.Dockerfile .
```

### PD Router image

```bash
docker build \
  -t mirrors.tencent.com/ti-platform/sglang-glm52-308x-pd-router:v0516-batch1 \
  -f docker/sgl-router.Dockerfile .
```

## Known Incompatibilities

- `--enable-mixed-chunk`: incompatible with eagle speculative decoding
- `--enable-hierarchical-cache`: incompatible with PD decode mode (forces disable-radix-cache)

## Related

- iWiki: https://iwiki.woa.com/p/4027588922
- Repo: https://github.com/tanguofu/sglang
- Branch: `308x-pd-v0516` (based on v0.5.16)
- PR link: https://github.com/tanguofu/sglang/pull/new/308x-pd-v0516
