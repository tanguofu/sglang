# GLM-5.2-FP8 SGLang Worker for AMD MI355X (0708-opt)

Pre-patched Docker image for running GLM-5.2-FP8 on AMD MI355X (gfx950)
with SGLang. All code patches are baked into the image — **no runtime
patching needed**.

## What Changed vs. 0706 Runtime-Patch Workflow

| Aspect | 0706 (old) | 0708-opt (new) |
|--------|-----------|----------------|
| Patching | `python3 /data/patch_0706_unified.py` at container start | Patches baked into image at build time |
| Env vars | Set via `docker run -e` flags (20+ flags) | Pre-set via `ENV` in Dockerfile |
| Startup | Inline `bash -c "patch && exec ..."` | `/start_server.sh` entrypoint |
| Reproducibility | Depends on `/data` patch files existing | Fully self-contained image |
| Start time | ~30s patching + server init | Server init only |

## Base Image

```
lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260706
```

## Patches Applied (16 files, all baked in)

### 01-05 Bundle (HIP/DSA/MTP/AITER enablement)
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

### 06a Supplement v4
- **dsa_indexer.py**: `DUAL_STREAM_TOKEN_THRESHOLD = 1024` (unconditional),
  `cos_sin_cache` init + usage fixes, metadata None guard

### 06b D2H Sync Elimination
- **dsa_backend.py**: Use `seq_lens_sum` as upper bound (avoids GPU `.item()`)
- **frozen_kv_mtp_worker_v2.py**: Use `seq_lens_cpu` for sum

### 06c Draft Extend CUDA Graph for HIP
- **eagle_worker_v2.py**: Enable DSA backend + `supports_cuda_draft_extend_graph` on HIP

### AITER Config Generation
- **glm5_bf16_tuned_gemm.csv**: M=1..50000, N=32+160, K=6144
- **glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv**: M=1..65536, N=128+2624+3072+6144, K=6144

## Build

From the repository root (branch `0708-opt`):

```bash
docker build -f docker/rocm-mi355x-glm52-0708/Dockerfile \
  -t sglang-glm52-0708:latest .
```

## Run

```bash
docker run -d \
  --name sglang_0708_worker \
  --privileged \
  --network host \
  --shm-size 32g \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /data:/data \
  sglang-glm52-0708:latest
```

### Override defaults

```bash
docker run -d ... \
  -e PORT=30001 \
  -e MODEL_PATH=/data/models/GLM-5.2-FP8 \
  -e API_KEY=your-key \
  sglang-glm52-0708:latest
```

## Environment Variables (pre-set in Dockerfile)

| Variable | Value |
|----------|-------|
| `HIP_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` |
| `NCCL_DEBUG` | `INFO` |
| `HSA_ENABLE_SDMA` | `0` |
| `HIP_FORCE_DEV_KERNARG` | `1` |
| `HSA_NO_SCRATCH_RECLAIM` | `1` |
| `NCCL_CUMEM_ENABLE` | `0` |
| `NCCL_MIN_NCHANNELS` | `112` |
| `NCCL_NVLS_ENABLE` | `0` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` |
| `PYTORCH_ROCM_ARCH` | `gfx950` |
| `ROCM_QUICK_REDUCE_QUANTIZATION` | `INT8` |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | `1` |
| `SGLANG_DISABLE_CUDNN_CHECK` | `1` |
| `SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM` | `1` |
| `SGLANG_INT4_WEIGHT` | `0` |
| `SGLANG_MOE_PADDING` | `1` |
| `SGLANG_ROCM_DISABLE_LINEARQUANT` | `0` |
| `SGLANG_ROCM_FUSED_DECODE_MLA` | `1` |
| `SGLANG_SET_CPU_AFFINITY` | `1` |
| `SGLANG_USE_AITER` | `1` |
| `SGLANG_USE_ROCM700A` | `1` |

## Server Launch Command (in start_server.sh)

```
python3 -m sglang.launch_server \
  --model-path /data/models/GLM-5.2-FP8 \
  --model-impl sglang \
  --served-model-name glm-5.2 \
  --api-key sk-... \
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

## Verification

To verify the image without starting the server:

```bash
# Start a sleep container
docker run -d --name sglang_0708_verify \
  --entrypoint sleep \
  sglang-glm52-0708:latest infinity

# Check patched files
docker exec sglang_0708_verify grep -c "_is_cuda or _is_hip" \
  /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py

# Check env vars
docker exec sglang_0708_verify env | grep SGLANG

# Check startup script
docker exec sglang_0708_verify cat /start_server.sh

# Check AITER configs
docker exec sglang_0708_verify wc -l \
  /sgl-workspace/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv

docker rm -f sglang_0708_verify
```
