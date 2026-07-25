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

## Known Incompatibilities

- `--enable-mixed-chunk`: incompatible with eagle speculative decoding
- `--enable-hierarchical-cache`: incompatible with PD decode mode (forces disable-radix-cache)

## Related

- iWiki: https://iwiki.woa.com/p/4027588922
- Repo: https://github.com/tanguofu/sglang
- Commit: 6f4a77b (code patches: conn.py, pd_router.rs, Cargo.toml)
