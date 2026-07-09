# GLM-5.2-FP8 SGLang Worker — amd-355-master Snapshot

Exact reproduction of the **amd-355-master** (216.128.153.58) production
deployment as of **2026-07-09**. All runtime patches are baked into the
Docker image at build time — no runtime patching needed.

## What This Captures

| Item | Value |
|------|-------|
| Source machine | 216.128.153.58 (amd-355-master) |
| Snapshot date | 2026-07-09 |
| Base image | `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260706` |
| AITER version | `9127c94a1` (2026-06-25, base image — NOT upgraded) |
| Container name on master | `guofu-PROD-glm5.2-DONT-TOUCH` |
| Launch script reference | `/data/start_worker_0706_no_fused.sh` |

## Patches Applied (all baked in at build time)

### 01-05 Bundle (`patch_sglang_glm52_rocm_all.py`)
- **dsa_indexer.py**: JIT imports on HIP, DSA indexer fusion, AITER 3-tuple
  extraction, FP8 dtype fix, cos_sin_cache, k_norm f32, assert relaxed
- **deepseek_v2.py**: Dual stream threshold on HIP
- **deepseek_nextn.py**: alt_stream on HIP
- **transformers.py**: `mla_kv_a_proj` → `replicate` TP style
- **dsa/utils.py**: `is_graph_dsa_split_op_surface` on HIP
- **dsa_backend.py**: `view` → `reshape` for non-contiguous tensors
- **radix_attention.py**: Disable `_pcg_mha_companion` swap
- **JIT kernel `.cuh`/`.cu`**: `cuda_fp8.h` → `hip/hip_fp8.h`
- **fp8.py**: `is_shuffled = True` after `shuffle_weight`
- **elementwise.py**: FP8 `uint8` buffer view fix for HIP
- **dsa_indexer.py**: `target_verify` metadata None guard
- **fused_store_index_cache.cuh**: ROCm FP8 pack via `deepseek_v4::fp8`

### 06a Supplement v4 (`patch_0706_supplement_v4.py`)
- **dsa_indexer.py**: `DUAL_STREAM_TOKEN_THRESHOLD = 1024` (unconditional),
  `cos_sin_cache` init + usage fixes, metadata None guard
- **dsa_backend.py**: `view` → `reshape` (3 locations in `forward_extend`)

### 06b D2H Sync Elimination (inline in `patch_0706_unified.py`)
- **dsa_backend.py**: Use `seq_lens_sum` as upper bound (avoids GPU `.item()`)
- **frozen_kv_mtp_worker_v2.py**: Use `seq_lens_cpu` for sum

### 06c Draft Extend CUDA Graph for HIP (inline in `patch_0706_unified.py`)
- **eagle_worker_v2.py**: Enable DSA backend + `supports_cuda_draft_extend_graph` on HIP

### 06d Disable DSA Fused-Store on ROCm
- **fused_store_index_cache.py**: `can_use_dsa_fused_store` → `return False`
  - Long-context correctness guard: A/B testing showed output corruption
    and MTP accept-rate collapse when fused-store writes index K cache.
  - Also applied via `disable_dsa_fused_store.py` (redundant but faithful
    to master's runtime command sequence).

### AITER Config Generation (inline in `patch_0706_unified.py`)
- **glm5_bf16_tuned_gemm.csv**: M=1..50000, N=32+160, K=6144
- **glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv**: M=1..65536, N=128+2624+3072+6144, K=6144

## Launch Command (exact match to master)

```
python3 -m sglang.launch_server \
    --model-path /data/models/GLM-5.2-FP8 \
    --model-impl sglang \
    --served-model-name glm-5.2 \
    --api-key sk-46faecc9d0bc4dcd9db6a15c73ae91c8 \
    --tp-size 8 --pp-size 1 --trust-remote-code \
    --host 0.0.0.0 --port 30000 \
    --context-length 1048576 \
    --tool-call-parser glm47 --reasoning-parser glm45 \
    --mem-fraction-static 0.88 \
    --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 12 16 \
    --cuda-graph-max-bs-decode 16 \
    --enable-aiter-allreduce-fusion --enable-mixed-chunk \
    --chunked-prefill-size 32768 \
    --enable-fused-qk-norm-rope \
    --schedule-conservativeness 0.5 \
    --prefill-max-requests 32 --max-prefill-tokens 32768 \
    --kv-cache-dtype fp8_e4m3 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
    --speculative-eagle-topk 1 \
    --cuda-graph-backend-prefill breakable \
    --max-running-requests 32 \
    --cuda-graph-bs-prefill 4 8 16 32 \
    --enable-metrics --skip-server-warmup \
    --watchdog-timeout 3600 --log-level info
```

## Environment Variables (exact match to master)

```
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NCCL_DEBUG=INFO
HSA_ENABLE_SDMA=0
HIP_FORCE_DEV_KERNARG=1
HSA_NO_SCRATCH_RECLAIM=1
NCCL_CUMEM_ENABLE=0
NCCL_MIN_NCHANNELS=112
NCCL_NVLS_ENABLE=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTORCH_ROCM_ARCH=gfx950
ROCM_QUICK_REDUCE_QUANTIZATION=INT8
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
SGLANG_DISABLE_CUDNN_CHECK=1
SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM=1
SGLANG_INT4_WEIGHT=0
SGLANG_MOE_PADDING=1
SGLANG_ROCM_DISABLE_LINEARQUANT=0
SGLANG_ROCM_FUSED_DECODE_MLA=1
SGLANG_SET_CPU_AFFINITY=1
SGLANG_USE_AITER=1
SGLANG_USE_ROCM700A=1
```

## Build

From the repository root (branch `amd355-master-snapshot`):

```bash
docker build -f docker/rocm-mi355x-glm52-master-snapshot/Dockerfile \
  -t sglang-glm52-master-snapshot:latest .
```

## Run

```bash
docker run -d \
  --name sglang_master_snapshot \
  --privileged \
  --network host \
  --shm-size 32g \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /data:/data \
  sglang-glm52-master-snapshot:latest
```

Override defaults via `-e`:

```bash
docker run -d ... \
  -e PORT=30001 \
  -e MODEL_PATH=/data/models/GLM-5.2-FP8 \
  -e API_KEY=your-key \
  sglang-glm52-master-snapshot:latest
```

## Differences from `docker/rocm-mi355x-glm52-0708/`

| Aspect | 0708-opt | master-snapshot (this) |
|--------|----------|----------------------|
| Patching | Pre-patched source COPY | Runtime scripts RUN at build time |
| AITER | Upgraded to `origin/main` (45458be72) | Base image `9127c94a1` (not upgraded) |
| 06d (disable fused store) | Not included | Included (correctness guard) |
| NCCL_DEBUG | Not set | `INFO` (matches master) |
| NCCL_MIN_NCHANNELS | Not set | `112` (matches master) |
| Verified on GPU? | No (never ran) | Yes (master ran this config 20+ hours) |

## Files

```
docker/rocm-mi355x-glm52-master-snapshot/
├── Dockerfile              # Build-time patching, exact env vars
├── start_server.sh         # Exact launch command (entrypoint)
├── README.md              # This file
├── container-inspect.json  # Full docker inspect of master container (reference)
└── patches/
    ├── patch_0706_unified.py           # Main bundle (01-05 + 06a-d + Gen1/Gen2)
    ├── patch_sglang_glm52_rocm_all.py  # 01-05 bundle (called by unified)
    ├── patch_0706_supplement_v4.py     # 06a supplement (called by unified)
    └── disable_dsa_fused_store.py      # 06d standalone (redundant with unified's 06d)
```

## Reproduction Notes

- The `container-inspect.json` is the full `docker inspect` output of the
  master container, saved for reference and auditing.
- `patch_glm_config.py` is NOT included — the 0706 base image already has
  the `qk_rope_head_dim` fix (confirmed in patch_sglang_glm52_rocm_all.py header).
- The master container was started with `start_worker_0706_no_fused.sh`,
  which adds `disable_dsa_fused_store.py` to the command (the `start_worker_0706.sh`
  variant does not). Both are functionally equivalent because `patch_0706_unified.py`
  already includes 06d inline.
