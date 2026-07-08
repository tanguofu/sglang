---
name: sglang-glm52-deploy
description: Deploy GLM-5.2-FP8 SGLang on AMD MI355X with pre-patched Docker images. Use when building, updating, verifying, or rolling out a new sglang/aiter version. CRITICAL: amd-355-master is PRODUCTION — never auto-update; always verify on amd-355-worker first, then deliver as a Docker image for manual rollout.
---

# SGLang GLM-5.2 Deploy on AMD MI355X

## ⚠️ CRITICAL SAFETY RULES

1. **`amd-355-master` is PRODUCTION.** NEVER auto-update, NEVER auto-deploy, NEVER auto-restart containers on it without explicit user confirmation.
2. **Always verify on `amd-355-worker` first.** Every code change, patch, or AITER upgrade must be built, started, and smoke-tested on `amd-355-worker` before it touches master.
3. **Deliver updates as Docker images.** The user manually pulls and restarts on master. We do NOT push images or run containers on master automatically.
4. **Never delete or modify `/data` on any machine.** Model weights and configs live there.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Local Dev (macOS)                                       │
│  - sglang repo (branch 0708-opt)                         │
│  - Apply patches to source → commit → push to GitHub     │
│  - Dockerfile in docker/rocm-mi355x-glm52-0708/          │
└──────────────────────┬──────────────────────────────────┘
                       │ git push
                       ▼
┌─────────────────────────────────────────────────────────┐
│  GitHub: tanguofu/sglang (branch 0708-opt)              │
└──────────────────────┬──────────────────────────────────┘
                       │ git clone/pull
                       ▼
┌─────────────────────────────────────────────────────────┐
│  amd-355-worker (TEST/STAGING)                          │
│  - 8× AMD MI355X (gfx950)                                │
│  - git pull → docker build → start container             │
│  - Run smoke tests, benchmarks, quality eval             │
│  - If PASS → tag image for production                    │
└──────────────────────┬──────────────────────────────────┘
                       │ docker save / docker load (manual)
                       ▼
┌─────────────────────────────────────────────────────────┐
│  amd-355-master (PRODUCTION) ⚠️                          │
│  - 8× AMD MI355X (gfx950)                                │
│  - User manually: docker load → docker run                │
│  - NEVER auto-update without user confirmation           │
└─────────────────────────────────────────────────────────┘
```

## Machine Access

| Machine | SSH Host | IP | Role | GPU |
|---------|----------|----|-----|-----|
| amd-355-master | `amd-355-master` | 216.128.153.58 | **PRODUCTION** ⚠️ | 8× MI355X |
| amd-355-worker | `amd-355-worker` | 144.202.61.0 | TEST/STAGING | 8× MI355X |

SSH config is in `~/.ssh/config`. Both use `~/.ssh/id_ed25519_amd_poc`.

## Key Files

| File | Purpose |
|------|---------|
| `docker/rocm-mi355x-glm52-0708/Dockerfile` | Main build file (FROM official 0706 image) |
| `docker/rocm-mi355x-glm52-0708/start_server.sh` | Entrypoint (server launch command) |
| `docker/rocm-mi355x-glm52-0708/gen_aiter_configs.py` | AITER BF16 + A8W8 tuned GEMM config generation |
| `docker/rocm-mi355x-glm52-0708/README.md` | Full documentation |
| `scripts/amd355_glm52_worker/` | Patch scripts, benchmark scripts, eval scripts |
| `scripts/amd355_glm52_worker/start_worker_0706.sh` | Original runtime-patch launch script (reference) |

## Workflow: Code Update → Verify → Deploy

### Phase 1: Local Development

1. **Apply patches to local sglang source** (if new patches needed):
   ```bash
   # Patches are already baked into 0708-opt branch.
   # To add new patches, adapt from scripts/amd355_glm52_worker/patch_*.py
   # Replace /sgl-workspace/sglang with local repo root.
   ```

2. **Commit and push**:
   ```bash
   git add -A
   git commit -m "feat: <description>"
   git push origin 0708-opt
   ```

3. **Update Dockerfile if needed** (e.g., AITER version, env vars, patches).

### Phase 2: Build on amd-355-worker (TEST)

```bash
# SSH to worker
ssh amd-355-worker

# Clone or update repo
cd /root
if [ ! -d sglang-0708 ]; then
  git clone --depth 1 --branch 0708-opt https://github.com/tanguofu/sglang.git sglang-0708
else
  cd sglang-0708 && git pull origin 0708-opt
fi

# Build image
cd /root/sglang-0708
docker build -f docker/rocm-mi355x-glm52-0708/Dockerfile -t sglang-glm52-0708:test .

# Start test container
docker run -d \
  --name sglang_0708_test \
  --privileged --network host --shm-size 32g \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /data:/data \
  sglang-glm52-0708:test

# Check logs
docker logs -f sglang_0708_test
```

### Phase 3: Verify on amd-355-worker

```bash
# 1. Wait for server ready
until curl -s http://localhost:30000/health | grep -q "ok"; do sleep 5; done

# 2. Smoke test
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-46faecc9d0bc4dcd9db6a15c73ae91c8" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'

# 3. Benchmark (optional)
docker exec sglang_0708_test python3 /data/bench_decode_eagle.py

# 4. Quality eval (optional)
docker exec sglang_0708_test python3 /data/eval_aime25.py

# 5. Verify patches
docker exec sglang_0708_test grep -c "_is_cuda or _is_hip" \
  /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py

# 6. Verify AITER version
docker exec sglang_0708_test cat /aiter_version.txt

# 7. Verify env vars
docker exec sglang_0708_test env | grep SGLANG | sort
```

### Phase 4: Tag and Export Image for Production

```bash
# On amd-355-worker, after verification passes:
docker tag sglang-glm52-0708:test sglang-glm52-0708:prod-$(date +%Y%m%d)

# Export image (large, ~108GB — use compression)
docker save sglang-glm52-0708:prod-$(date +%Y%m%d) | gzip > /data/sglang-glm52-0708-prod-$(date +%Y%m%d).tar.gz

# Or transfer directly to master (user must confirm):
# docker save sglang-glm52-0708:prod-$(date +%Y%m%d) | ssh amd-355-master "docker load"
```

### Phase 5: Manual Deploy on amd-355-master (USER ONLY)

```bash
# ⚠️ USER MUST RUN THESE MANUALLY — never automate

# On amd-355-master:
# 1. Load the new image
docker load < /data/sglang-glm52-0708-prod-YYYYMMDD.tar.gz

# 2. Stop old container
docker stop sglang_0706_worker  # or current container name
# docker rm sglang_0706_worker  # optional

# 3. Start new container
docker run -d \
  --name sglang_0708_worker \
  --privileged --network host --shm-size 32g \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /data:/data \
  sglang-glm52-0708:prod-YYYYMMDD

# 4. Verify
docker logs -f sglang_0708_worker
curl -s http://localhost:30000/health
```

## What's Baked Into the Image

### SGLang Patches (16 files, all pre-applied)

| Patch | File | Change |
|------|------|--------|
| 01 (1.1-1.12) | `dsa_indexer.py` | JIT imports, DSA fusion, AITER tuple extraction, k_norm f32, cos_sin_cache, assert relaxed |
| 02 | `fp8.py` | `is_shuffled=True` after shuffle_weight |
| 03 | `elementwise.py` | FP8 uint8 buffer view fix for HIP |
| 04 | `dsa_indexer.py` | target_verify metadata None guard |
| 05 | `fused_store_index_cache.cuh` | ROCm FP8 pack via deepseek_v4::fp8 |
| 06a | `dsa_indexer.py` + `dsa_backend.py` | DUAL_STREAM_TOKEN_THRESHOLD=1024, cos_sin_cache, view→reshape |
| 06b | `dsa_backend.py` + `frozen_kv_mtp_worker_v2.py` | D2H sync elimination (seq_lens_sum) |
| 06c | `eagle_worker_v2.py` | Draft extend CUDA graph on HIP |
| 2.1 | `deepseek_v2.py` | Dual stream on HIP |
| 3.0-3.1 | `deepseek_nextn.py` | alt_stream on HIP |
| 4.1 | `transformers.py` | mla_kv_a_proj → replicate TP style |
| 5b | `dsa/utils.py` | is_graph_dsa_split_op_surface on HIP |
| 5c | `dsa_backend.py` | view → reshape (3 locations) |
| 5d | `radix_attention.py` | Disable _pcg_mha_companion swap |
| 6.1 | JIT `.cuh` (4 files) | cuda_fp8.h → hip/hip_fp8.h |

### AITER

- Upgraded to latest `origin/main` (commit `45458be72`)
- Includes GLM-5.2 tuned configs, OPUS RMSNorm, gfx950 MLA/MoE optimizations
- `.so` files from base image (222); new kernels JIT-compile at runtime
- BF16 tuned config: 100,087 lines; A8W8 tuned config: 262,176 lines

### Environment Variables (21, pre-set via ENV)

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

### Server Launch Command (in start_server.sh)

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

## Verification Checklist

Before deploying to master, verify on worker:

- [ ] Docker build succeeds (all 16 patches verified in build)
- [ ] AITER version is latest origin/main (`cat /aiter_version.txt`)
- [ ] Container starts and server becomes healthy (`/health` returns ok)
- [ ] Smoke test: chat completion returns valid response
- [ ] Patch patterns present (grep for key patterns)
- [ ] Env vars correct (21 vars, all match)
- [ ] AITER configs present (BF16 + A8W8 CSV files)
- [ ] Startup command matches reference (start_server.sh)
- [ ] Benchmark: decode throughput within expected range
- [ ] Quality eval: AIME/other benchmarks pass

## Rollback

If production has issues after deploy:

```bash
# On amd-355-master (USER runs manually):
docker stop sglang_0708_worker
docker start sglang_0706_worker  # restart old container
```

Old containers are not auto-removed. Keep at least one previous-version container stopped but available for rollback.

## Common Issues

### AITER upgrade fails in Docker build

`PREBUILD_KERNELS=1` needs `rocminfo` (GPU access), unavailable in Docker build.
**Fix**: Skip PREBUILD_KERNELS. Keep base image .so files. New kernels JIT-compile at runtime.

### CK submodule fetch fails

CK remote has force-pushed branches. `git submodule update --recursive` fails.
**Fix**: Use `git fetch --no-recurse-submodules`. Skip CK submodule update (base image CK is compatible).

### `import aiter` fails without GPU

AITER's `__init__.py` calls `rocminfo` to detect GPU arch.
**Fix**: Don't `import aiter` in Dockerfile RUN steps. Verify via .so file count instead.

### git checkout fails (local modifications)

Base image has local modifications to AITER config CSVs from runtime patching.
**Fix**: Use `git checkout -f origin/main` (force).
