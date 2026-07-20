# GLM-5.2-FP8 on AMD MI308X (gfx942) — Complete Reproduction Guide

This guide walks through reproducing the GLM-5.2-FP8 SGLang deployment on AMD
MI308X from a clean slate: building the worker image, preparing Kubernetes
nodes, deploying via Helm, and verifying the EAGLE speculative decoding fix.

## Table of Contents

- [Background](#background)
- [Prerequisites](#prerequisites)
- [Step 1: Build the Worker Image](#step-1-build-the-worker-image)
- [Step 2: Prepare Kubernetes Nodes](#step-2-prepare-kubernetes-nodes)
- [Step 3: Deploy via Helm](#step-3-deploy-via-helm)
- [Step 4: Post-Deploy Verification](#step-4-post-deploy-verification)
- [Step 5: Benchmark](#step-5-benchmark)
- [EAGLE Coredump Fix Summary](#eagle-coredump-fix-summary)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Validation Results (2026-07-20)](#validation-results-2026-07-20)

---

## Background

The original deployment (`xgmi-opt-0716d` image, based on sglang `b76dd0be`
from 2026-07-10) suffered an 8-rank GPU coredump whenever EAGLE speculative
decoding ran after a >1024-token prefill. Root cause was three ROCm-specific
issues in the new DSA indexer code path, plus missing upstream EAGLE/CUDA
graph fixes. This branch (`fix/eagle-decode-coredump-mi308x`) fixes all three
issues and backports 6 critical upstream commits + 1 local patch (PR #31478).

See [EAGLE Coredump Fix Summary](#eagle-coredump-fix-summary) for the full
root cause analysis.

---

## Prerequisites

### Hardware

- 2x AMD MI308X nodes (8x gfx942 GPUs each, 192GB VRAM/GPU, ~2TB host RAM)
- Broadcom BCM57608 RDMA NICs (for PD test variants, optional)

### Software

- **OS**: TencentOS Server 4.4 (kernel 6.6) — use custom image
  `img-ebtth3fd` (ap-zhongwei) which has amdgpu 6.16.13 + ROCm 7.2.4
  pre-installed. See `project_tos44_hccpa1_custom_image.md` memory for details.
- **ROCm**: 7.2.4 (hip-runtime, miopen, rccl, rocfft, rocsparse, rocrand)
- **Kubernetes**: TKE cluster with AMD GPU device plugin
- **Docker**: Installed on build machine
- **Helm**: 3.x
- **kubectl**: Configured for target cluster

### Model Weights

GLM-5.2-FP8 weights must be present at `/data/model/glm52-fp8/` on each
worker node (hostPath mount). Download from HuggingFace
`THUDM/GLM-5.2-FP8` (or internal mirror).

### Container Registry

The image is pushed to `mirrors.tencent.com/ti-platform/`. Build machine must
have:
- `/etc/hosts` entry: `30.163.240.137 mirrors.tencent.com`
- Docker login credentials for `mirrors.tencent.com`

### Repository

```bash
git clone https://github.com/tanguofu/sglang.git
cd sglang
git checkout fix/eagle-decode-coredump-mi308x
```

This branch is based on `origin/main` @ `50c118704a` (2026-07-20) with a
single squashed commit containing all fixes.

---

## Step 1: Build the Worker Image

### 1.1 Build

From the repo root:

```bash
docker build -f docker/rocm-mi308x-glm52/Dockerfile \
  -t mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3 .
```

**Base image**: `lmsysorg/sglang-rocm:v0.5.15.post1-rocm720-mi30x-20260718`
(~15GB pull if not cached).

**Build-time assertions**: The Dockerfile verifies all 6 upstream fix markers
+ PR #31478 are present in the source. Build fails if any patch is missing:

```
All EAGLE coredump fixes verified: PR #31478 + 78dc581518 + 7a973c03a0 + fc1e3797b7 + cce5fe7696 + 942bf04ef9
```

**Build time**: ~27s if base image is cached (only COPY + verify steps).
First build with base image pull: ~15-20 min depending on bandwidth.

### 1.2 Push to Registry

```bash
docker push mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3
```

### 1.3 Verify Image

```bash
# Check digest
docker inspect mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3 \
  --format '{{.Id}}'

# Quick smoke (runs entrypoint, should start sglang server)
docker run --rm --entrypoint echo \
  mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3 \
  "Image OK"
```

### 1.4 (Optional) Build the PD Router Image

For PD (prefill-decode) disaggregated deployment with `/v1/messages` support,
a separate router image is needed. This is only required if you deploy the
PD test manifests in `pd-test-gz-rdma/`. The standard 2tp8 deployment uses
the worker image for the router.

```bash
# See docker/rocm-mi308x-glm52-pd/ for PD router Dockerfile
# Tag: mirrors.tencent.com/ti-platform/sglang-glm52-308x-pd-router:messages-0717c
```

---

## Step 2: Prepare Kubernetes Nodes

### 2.1 Verify GPU and Driver Health

On each worker node (`.152` and `.172` in our deployment):

```bash
# 8 GPUs healthy
rocm-smi --showproductname --showhealth

# RDMA (optional, for PD test)
ibstat | grep -E "CA type|Port state"

# Driver version
cat /sys/module/amdgpu/version  # should be 6.16.13
```

### 2.2 Label Nodes

```bash
# Primary worker (hosts router + gateway)
kubectl label node node-21.151.225.152 \
  accelerator=amd-gpu \
  sglang-model=ready \
  --overwrite

# Secondary worker
kubectl label node node-21.151.225.172 \
  accelerator=amd-gpu \
  sglang-model=ready \
  --overwrite
```

### 2.3 Add Taints (Optional, for Dedication)

If nodes are dedicated to sglang (recommended for production):

```bash
kubectl taint node node-21.151.225.152 dedicated=sglang-2tp8:NoSchedule
kubectl taint node node-21.151.225.172 dedicated=sglang-2tp8:NoSchedule
```

The chart values include tolerations for `dedicated=sglang-2tp8:NoSchedule`
plus a broad `operator: Exists` fallback.

### 2.4 Verify Model Weights

```bash
ssh node-21.151.225.152 'ls -la /data/model/glm52-fp8/*.safetensors | wc -l'
# Should show expected number of shard files

ssh node-21.151.225.172 'ls -la /data/model/glm52-fp8/*.safetensors | wc -l'
```

### 2.5 Verify Host RAM Available

The deployment uses ~1.4TB host RAM (HiCache + DSA indexer). Nodes have
~2TB total. Verify at least 1.6TB free before deploying:

```bash
ssh node-21.151.225.152 'free -g | awk "/^Mem:/{print \$4\" GB free\"}"'
# Should show >1600 GB free
```

---

## Step 3: Deploy via Helm

### 3.1 Deploy Worker 1 (Primary — hosts router + gateway)

```bash
helm install sglang-glm52-2tp8 \
  docker/rocm-mi308x-glm52/chart/ \
  -n kube-system \
  -f docker/rocm-mi308x-glm52/chart/values-glm52-2tp8.yaml
```

This creates:
- StatefulSet `sglang-glm52-2tp8-sglang-0` on node `.152`
- Service `sglang-glm52-2tp8-sglang` (port 30000)
- Router Deployment `sglang-glm52-2tp8-router` (sgl-model-gateway, port 30001)
- HTTPRoute `glm52-2tp8.jmpti.woa.com` → router

### 3.2 Deploy Worker 2 (Secondary — worker only)

```bash
helm install sglang-glm52-2tp8-w2 \
  docker/rocm-mi308x-glm52/chart/ \
  -n kube-system \
  -f docker/rocm-mi308x-glm52/chart/values-glm52-2tp8-w2.yaml
```

This creates:
- StatefulSet `sglang-glm52-2tp8-w2-sglang-0` on node `.172`
- Service `sglang-glm52-2tp8-w2-sglang` (port 30000)

### 3.3 Monitor Startup

Startup takes 3-5 minutes (CUDA graph capture + HiCache allocation):

```bash
# Watch W1
kubectl logs -n kube-system sglang-glm52-2tp8-sglang-0 -f

# Watch W2 (separate terminal)
kubectl logs -n kube-system sglang-glm52-2tp8-w2-sglang-0 -f
```

**Expected startup sequence** (look for these markers in logs):

```
# 1. Model loading
Loading model from /data/model/glm52-fp8...
Model loaded in 45.2s

# 2. CUDA graph capture (decode)
Capturing CUDA graphs for decode (bs=1,2,3,4,5,6,7,8,9,10,12,16)...
Capturing EAGLE draft decode CUDA graph...
Decode CUDA graph captured in 67.8s

# 3. CUDA graph capture (prefill, tc_piecewise)
Capturing tc_piecewise prefill CUDA graphs (bs=4,8,16,32)...
Prefill CUDA graph captured in 23.4s

# 4. HiCache allocation
Allocating HiCache host pool: ratio=2, 83.22 GB/rank, 692.16 GB total
HiCache host pool allocated in 12.3s

# 5. DSA indexer allocation
Allocating DSA indexer host memory: 56.89 GB/rank, 455.12 GB total

# 6. Server ready
Uvicorn running on http://0.0.0.0:30000
```

### 3.4 Verify Pods Running

```bash
kubectl get pod -n kube-system -l app=sglang -o wide
```

Expected output:
```
NAME                              READY   STATUS    RESTARTS   AGE
sglang-glm52-2tp8-sglang-0        1/1     Running   0          5m
sglang-glm52-2tp8-w2-sglang-0     1/1     Running   0          5m
sglang-glm52-2tp8-router-xxxxx    1/1     Running   0          5m
```

If `RESTARTS > 0`, check [Troubleshooting](#troubleshooting).

### 3.5 (Optional) Helm Upgrade

When updating the image or config:

```bash
helm upgrade sglang-glm52-2tp8 \
  docker/rocm-mi308x-glm52/chart/ \
  -n kube-system \
  -f docker/rocm-mi308x-glm52/chart/values-glm52-2tp8.yaml

helm upgrade sglang-glm52-2tp8-w2 \
  docker/rocm-mi308x-glm52/chart/ \
  -n kube-system \
  -f docker/rocm-mi308x-glm52/chart/values-glm52-2tp8-w2.yaml
```

**Important**: After `helm upgrade`, the router Deployment may need a manual
`kubectl patch` to remove `kubernetes.io/hostname` from `nodeSelector`
(strategic merge doesn't delete keys). The router tolerations are captured in
`values-glm52-2tp8.yaml` so they survive upgrades.

---

## Step 4: Post-Deploy Verification

### 4.1 Health Check

```bash
# W1 direct
curl -s http://21.151.225.152:30000/health
# Expected: ok

# W2 direct
curl -s http://21.151.225.172:30000/health
# Expected: ok

# Via router (load-balanced)
curl -s http://21.151.225.152:30001/health
# Expected: ok
```

### 4.2 Smoke Test

```bash
curl -s http://21.151.225.152:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-46faecc9d0bc4dcd9db6a15c73ae91c8" \
  -d '{
    "model": "glm-5.2",
    "messages": [{"role": "user", "content": "What is 2+3? Answer with just the number."}],
    "max_tokens": 16,
    "temperature": 0
  }' | python3 -m json.tool
```

Expected: response with `content: "5"` and non-zero `usage.completion_tokens`.

### 4.3 EAGLE Coredump Regression Test

This is the critical test — the original bug triggered on 2048-token prefills:

```bash
# Generate a ~2K token prompt
LONG_TEXT=$(python3 -c 'print("The quick brown fox jumps over the lazy dog. " * 150)')

curl -s http://21.151.225.152:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-46faecc9d0bc4dcd9db6a15c73ae91c8" \
  -d "$(python3 -c "
import json
text = 'The quick brown fox jumps over the lazy dog. ' * 150
print(json.dumps({
    'model': 'glm-5.2',
    'messages': [{'role': 'user', 'content': text + 'What animal is mentioned? Answer in one word.'}],
    'max_tokens': 50,
    'temperature': 0
}))
")" | python3 -m json.tool
```

**Expected**: HTTP 200 with valid response (e.g., `"fox"`). No coredump.

**If coredump occurs**: Check `kubectl logs` for `NameError`,
`AssertionError`, or `SIGKILL` — see [Troubleshooting](#troubleshooting).

### 4.4 Stress Test (4 Concurrent 2048-token Prefills)

```bash
for i in 1 2 3 4; do
  curl -s http://21.151.225.152:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-46faecc9d0bc4dcd9db6a15c73ae91c8" \
    -d "$(python3 -c "
import json
text = 'The quick brown fox jumps over the lazy dog. ' * 150
print(json.dumps({
    'model': 'glm-5.2',
    'messages': [{'role': 'user', 'content': text + 'Summarize in one sentence.'}],
    'max_tokens': 256,
    'temperature': 0
}))
")" > /tmp/stress_$i.json &
done
wait
echo "All 4 requests completed"
# Verify all returned valid JSON
for i in 1 2 3 4; do
  python3 -c "import json; json.load(open('/tmp/stress_$i.json'))" && echo "  ✓ Request $i OK"
done
```

### 4.5 Run the Verification Script

```bash
# From repo root, against W1
bash docker/rocm-mi308x-glm52/scripts/verify-v14.sh 21.151.225.152

# Against W2
bash docker/rocm-mi308x-glm52/scripts/verify-v14.sh 21.151.225.172
```

This runs: health check → metrics snapshot → smoke test → long context test →
MTP accept rate check.

### 4.6 Check Metrics

```bash
curl -s http://21.151.225.152:30000/metrics | grep -E "sglang:(spec_|hicache_|max_total)"
```

Key metrics to verify:
- `sglang:spec_accept_rate` > 0.5 (healthy EAGLE)
- `sglang:spec_accept_length` > 2.0 (good draft acceptance)
- `sglang:max_total_num_tokens` > 900000 (HiCache enabled)
- `sglang:hicache_host_total_tokens` > 1.8M (host cache populated)

---

## Step 5: Benchmark

### 5.1 Using the Included Benchmark Script

```bash
# Against W1 (requires sglang.bench_serving — run from inside the container
# or from a machine with sglang installed)
bash docker/rocm-mi308x-glm52/scripts/benchmark-v14.sh 21.151.225.152 w1
bash docker/rocm-mi308x-glm52/scripts/benchmark-v14.sh 21.151.225.172 w2
```

**Note**: `sglang.bench_serving` may fail with `ModuleNotFoundError` in some
container environments due to namespace package path mismatch. If this
happens, use the standalone benchmark script below.

### 5.2 Standalone Benchmark (No sglang Dependency)

If `sglang.bench_serving` fails, use a standalone script with `requests` +
`ThreadPoolExecutor`:

```python
# /tmp/bench_standalone.py — see benchmark section for full script
# Runs 4 scenarios: short_c32, short_c128, mid_c2048, long_c8192
# Reports: success rate, aggregate throughput, per-request gen throughput,
#          latency percentiles, EAGLE metrics
```

Run from any pod with Python + `requests` (e.g., the router pod):

```bash
kubectl cp /tmp/bench_standalone.py kube-system/<router-pod>:/tmp/bench.py
kubectl exec -n kube-system <router-pod> -- python3 /tmp/bench.py --host http://21.151.225.152:30000
kubectl exec -n kube-system <router-pod> -- python3 /tmp/bench.py --host http://21.151.225.172:30000
```

### 5.3 Expected Benchmark Results

See [Validation Results (2026-07-20)](#validation-results-2026-07-20) for the
reference numbers achieved with `fix-eagle-coredump-v3`.

---

## EAGLE Coredump Fix Summary

### Symptom

After a prefill of >1024 tokens, EAGLE speculative decode triggered an 8-rank
GPU coredump on MI308X (gfx942). Prefill itself succeeded (HTTP 200), then the
decode phase crashed all 8 GPUs simultaneously.

### Root Cause

The container was based on sglang `b76dd0be` (2026-07-10), which was 380
commits behind upstream `origin/main` (`b3570a4531`, 2026-07-20). The missing
commits included 6 critical EAGLE/CUDA graph fixes. Additionally, 3
ROCm-specific issues were discovered during the fix:

### Issue 1: NameError — `logits_head_gate_graph` not defined

**File**: `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`

`scale_head_gate_graph` and `logits_head_gate_graph` were defined inside an
`if _is_cuda:` block. On ROCm/HIP where `_is_cuda=False`, these functions
were never defined. Call sites at lines 1959 and 1968 referenced them
unqualified, causing `NameError` on first EAGLE decode after a >1024-token
prefill (which triggered breakable prefill CUDA graph capture warmup).

**Fix**: Hoist both function definitions (and their `_fake_impl` helpers) to
module level. Only CUDA-specific imports and
`broadcast_indexer_topk_from_rank0_` remain inside the `if _is_cuda:` block.

```python
# Before (broken):
if _is_cuda:
    def scale_head_gate_graph(...): ...
    def logits_head_gate_graph(...): ...

# After (fixed):
def _scale_head_gate_graph_fake_impl(...): ...
@register_custom_op(fake_impl=_scale_head_gate_graph_fake_impl)
def scale_head_gate_graph(...): ...
def _logits_head_gate_graph_fake_impl(...): ...
@register_custom_op(fake_impl=_logits_head_gate_graph_fake_impl)
def logits_head_gate_graph(...): ...

if _is_cuda:  # only CUDA-specific stuff remains
    @register_custom_op(mutates_args=["topk_indices"])
    @register_split_op()
    def broadcast_indexer_topk_from_rank0_(topk_indices): ...
```

### Issue 2: AssertionError — BCG split-op dispatch not selected on ROCm

**File**: `python/sglang/srt/layers/attention/dsa/dsa_indexer.py:2061`

After fixing Issue 1, breakable CUDA graph capture hit:
```
AssertionError: Internal error: in-graph DSA prefill must go through the
graph DSA split-op dispatch
```

Root cause: `is_graph_dsa_split_op_surface()` (`dsa/utils.py:103`) gates
split-op dispatch with `is_cuda()`, which returns `False` on ROCm. The
dispatch is never selected, but the assertion still fires.

Upstream `default_prefill_backend()` (`cuda_graph_config.py:101`) explicitly
returns `tc_piecewise` for non-CUDA platforms:
> "BCG is the prefill default on CUDA only; other platforms (HIP/NPU/...)
> keep tc_piecewise until BCG is validated there."

**Fix**: Switch `cudaGraphBackendPrefill` from `breakable` to `tc_piecewise`
in `chart/values.yaml` and `start_server.sh`.

### Issue 3: Host Memory OOM — DSA indexer + HiCache exceed node RAM

After fixing Issue 2, the new DSA indexer host memory allocation caused
`SIGKILL` (exit code -9) on Rank 5 scheduler.

The new upstream DSA indexer allocates 56.89 GB/rank host memory for
`page_first_direct` layout (455 GB total). Combined with HiCache at
`hicacheRatio=4` (248 GB/rank = 2 TB total), total host memory was ~2.4 TB,
exceeding node RAM (~2 TB).

**Fix**: Reduce `hicacheRatio` from 4 to 2 (HiCache 124 GB/rank + DSA
19 GB/rank × 8 = ~1.4 TB, well within node RAM).

### Missing Upstream Fixes (6 commits)

All included in the v3 image, verified by Dockerfile build-time assertions:

| Commit | Description |
|--------|-------------|
| `78dc581518` | Stabilize GLM-5.2 MTP IndexShare across PD and CUDA graph replay |
| `7a973c03a0` | Stamp capture-time num_tokens_per_req in multi-layer EAGLE |
| `fc1e3797b7` | Split capture width from num_tokens_per_req and gate replay |
| `cce5fe7696` | Move WAR barrier right after each run_batch launch |
| `942bf04ef9` | Add SGLANG_FORCE_COARSE_WAR_BARRIER opt-in |
| `7e229e2a81` | Support GLM-5.2 MTP index sharing with prefill CP |

### Local Patch (PR #31478, not yet merged upstream)

Adds TP broadcast in the EAGLE greedy branch to prevent collective deadlock
under large prefill + EAGLE overlap scheduling.

---

## Configuration Reference

### Key Server Parameters (`values.yaml` + `values-glm52-2tp8.yaml`)

| Parameter | Default | 2tp8 Override | Description |
|-----------|---------|---------------|-------------|
| `sglang.tpSize` | 8 | 8 | Tensor parallel (all 8 GPUs) |
| `sglang.contextLength` | 1048576 | 524288 | 512K (reduced for stability) |
| `sglang.memFractionStatic` | 0.88 | 0.75 | OOM fix: leaves ~46GB/rank activation |
| `sglang.chunkedPrefillSize` | 32768 | 16384 | OOM fix: smaller peak activation |
| `sglang.prefillMaxRequests` | 32 | 4 | OOM fix: bound concurrent prefills |
| `sglang.scheduleConservativeness` | 0.5 | 1.0 | OOM fix: conservative scheduling |
| `sglang.kvCacheDtype` | fp8_e4m3 | fp8_e4m3 | FP8 KV cache |
| `sglang.cudaGraphBackendPrefill` | tc_piecewise | tc_piecewise | **NOT breakable** (ROCm) |
| `sglang.hicacheRatio` | 2.0 | 2 | HiCache 2× GPU KV pool |
| `sglang.hicacheWritePolicy` | write_through | write_back | Fixes host load-back |
| `sglang.hicacheMemLayout` | page_first_direct | page_first_direct | DSA indexer layout |
| `sglang.speculativeAlgorithm` | NEXTN | NEXTN | EAGLE MTP |
| `sglang.speculativeNumSteps` | 3 | 3 | MTP steps |
| `sglang.speculativeNumDraftTokens` | 4 | 4 | Draft tokens per step |
| `sglang.speculativeEagleTopk` | 1 | 1 | EAGLE top-k |
| `sglang.watchdogTimeout` | 3600 | 1200 | Faster fail on stuck detokenizer |
| `sglang.numaNode` | "0 0 0 0 1 1 1 1" | (same) | NUMA binding per rank |

### Key Environment Variables (Dockerfile)

| Variable | Value | Description |
|----------|-------|-------------|
| `HIP_VISIBLE_DEVICES` | 0,1,2,3,4,5,6,7 | All 8 GPUs |
| `NCCL_DEBUG` | WARN | Suppress INFO spam (slows health probe) |
| `NCCL_MIN_NCHANNELS` | 80 | RCCL 2.27.7 hard cap (higher = clamped + spam) |
| `HSA_ENABLE_SDMA` | 0 | Correct for MI308X P2P/XGMI |
| `PYTORCH_ROCM_ARCH` | gfx942 | MI308X architecture |
| `ROCM_QUICK_REDUCE_QUANTIZATION` | NONE | Zero precision loss (was INT8) |
| `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` | false | /health returns 200 directly (no prefill) |
| `SGLANG_FORCE_COARSE_WAR_BARRIER` | true | Stable EAGLE overlap on ROCm |
| `SGLANG_USE_AITER` | 1 | AMD AITER kernels |
| `SGLANG_USE_ROCM700A` | 1 | ROCm 7.0A features |
| `SGLANG_ROCM_FUSED_DECODE_MLA` | 1 | Fused decode MLA |
| `SGLANG_SET_CPU_AFFINITY` | 1 | CPU pinning |
| `SGLANG_MOE_PADDING` | 1 | MoE padding optimization |

### Helm Values Files

| File | Release | Topology | Notes |
|------|---------|----------|-------|
| `values.yaml` | — | chart defaults | 1M context, ratio=2, write_through |
| `values-glm52-2tp8.yaml` | `sglang-glm52-2tp8` | GZ, .152 | Primary: worker + router + gateway |
| `values-glm52-2tp8-w2.yaml` | `sglang-glm52-2tp8-w2` | GZ, .172 | Secondary: worker only |
| `values-glm52-test.yaml` | `sglang-glm52-test` | GZ test, 144+132 | Legacy test env |
| `values-glm52-test-w2.yaml` | `sglang-glm52-test-w2` | GZ test, 132 | Legacy test worker |
| `values-prod.yaml` | `sglang-glm52-prod` | ZW prod | Planned |
| `values-test.yaml` | `sglang-glm52-test` | Legacy | Superseded by values-glm52-test.yaml |

---

## Troubleshooting

### NameError: `logits_head_gate_graph` is not defined

**Cause**: The hoist fix is missing. You're running an old image
(`fix-eagle-coredump` v1 or v2).

**Fix**: Use `fix-eagle-coredump-v3` or later. Verify with:
```bash
docker run --rm --entrypoint python3 \
  mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3 \
  -c "import sglang.srt.layers.attention.dsa.dsa_indexer; print('OK')"
```

### AssertionError: in-graph DSA prefill must go through split-op dispatch

**Cause**: `cudaGraphBackendPrefill` is set to `breakable` on ROCm. BCG is
CUDA-only per upstream `default_prefill_backend()`.

**Fix**: Set `cudaGraphBackendPrefill: tc_piecewise` in values.yaml. Already
fixed in this branch's `values.yaml` and `start_server.sh`.

### SIGKILL (exit code -9) during startup

**Cause**: Host memory OOM. DSA indexer (57 GB/rank) + HiCache (ratio=4 →
248 GB/rank) exceeds node RAM.

**Fix**: Reduce `hicacheRatio` to 2 (or lower). Verify host RAM:
```bash
ssh <node> 'free -g | awk "/^Mem:/{print \$4\" GB free\"}"'
# Must be > 1600 GB for ratio=2
```

### NCCL error: unhandled cuda error

**Cause**: GPU state corruption from previous crash. The amdgpu kernel may
be in a bad state.

**Fix**: Delete the pod to force a clean restart:
```bash
kubectl delete pod -n kube-system <pod-name> --grace-period=0 --force
```
If the error persists, the node itself needs a reboot to reset GPU state.

### hipIpcGetMemHandle failed: invalid argument

**Cause**: Same as NCCL error — GPU state corruption from previous crash.

**Fix**: Delete pod. If persistent, reboot node.

### Pod stuck in CrashLoopBackOff

1. Check logs: `kubectl logs -n kube-system <pod-name> --previous`
2. Common causes:
   - Model weights missing at `/data/model/glm52-fp8/`
   - Node not labeled `sglang-model=ready`
   - Insufficient host RAM (see SIGKILL above)
   - GPU device plugin not running on node

### Router 503 / Connection refused

1. Check router pod: `kubectl get pod -n kube-system -l app=sglang-router`
2. Check router logs: `kubectl logs -n kube-system <router-pod>`
3. Verify worker URLs in values match actual node IPs
4. Check HTTPRoute: `kubectl get httproute -n ti-cloud`
5. If router was upgraded via helm, may need manual patch to fix
   `nodeSelector` (strategic merge doesn't delete keys):
   ```bash
   kubectl patch deploy -n kube-system <router-deploy> --type=json \
     -p='[{"op":"remove","path":"/spec/template/spec/nodeSelector/kubernetes.io~1hostname"}]'
   ```

### EAGLE accept rate < 0.5

**Cause**: May indicate EAGLE model weights not loaded, or reasoning tokens
are hard to predict.

**Fix**:
1. Verify `--speculative-algorithm NEXTN` is in launch args
2. Check EAGLE model weights present in model directory
3. Try `eagle_topk=2` (was 1) for slightly better acceptance
4. Reasoning tokens inherently have lower accept rate (~60-85% is normal)

### Health probe failing (timeout)

**Cause**: `/health` was running prefill 64 tokens, causing GPU wake-up delay
of 10-30s when GPUs are in low-power state.

**Fix**: Set `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false` in Dockerfile
(already set in v3). This makes `/health` return 200 directly without
touching the GPU.

---

## Validation Results (2026-07-20)

### Deployment

| Component | W1 (.152) | W2 (.172) |
|-----------|-----------|-----------|
| Helm release | `sglang-glm52-2tp8` rev 31 | `sglang-glm52-2tp8-w2` rev 27 |
| Image tag | `fix-eagle-coredump-v3` | `fix-eagle-coredump-v3` |
| Image digest | `sha256:416eb7f8...` | `sha256:416eb7f8...` |
| Status | 1/1 Running, 0 restarts | 1/1 Running, 0 restarts |
| Prefill backend | tc_piecewise | tc_piecewise |
| HiCache ratio | 2 | 2 |
| HiCache write policy | write_back | write_back |

### Benchmark (4 scenarios, 100% success on both workers)

| Scenario | W1 (.152) | W2 (.172) |
|----------|-----------|-----------|
| short_c32 (in=32, out=256, n=32, rate=8) | 153.00 tok/s | 146.31 tok/s |
| short_c128 (in=128, out=256, n=32, rate=8) | 242.40 tok/s | 251.87 tok/s |
| **mid_c2048** (in=2048, out=256, n=16, rate=4) | 166.47 tok/s | 184.06 tok/s |
| long_c8192 (in=8192, out=256, n=8, rate=2) | 108.21 tok/s | 75.60 tok/s |

**mid_c2048 is the original coredump trigger scenario — now 100% stable.**

### EAGLE Metrics

| Metric | W1 (.152) | W2 (.172) |
|--------|-----------|-----------|
| `sglang:spec_accept_rate` | 0.60 | 0.858 |
| `sglang:spec_accept_length` | 2.80 | 3.575 |
| `sglang:spec_verify_calls_total` | 8060 | 11139 |
| `sglang:max_total_num_tokens` | 926080 | 926080 |
| `sglang:hicache_host_total_tokens` | 1.85M | 1.85M |

### Server-Side Decode Throughput (from logs)

- W1: peak 444.53 tok/s at concurrency 9 (accept rate 0.70-0.73)
- W2: peak 437.17 tok/s at concurrency 9 (accept rate 0.68-0.72)

### Memory Footprint

- GPU VRAM: ~160 GB used / 192 GB total per GPU (mem-fraction=0.75)
- Host RAM: ~819 GB total (HiCache 83 GB/rank + DSA 19 GB/rank × 8)
- Activation headroom: ~46 GB/GPU (18× improvement from OOM fix)

---

## File Map

```
docker/rocm-mi308x-glm52/
├── Dockerfile                    # Worker image (base + source + patches)
├── start_server.sh               # Entrypoint (all launch params)
├── REPRODUCE.md                  # This file
├── chart/
│   ├── Chart.yaml                # Helm chart metadata
│   ├── README.md                 # Chart quick reference
│   ├── values.yaml               # Chart defaults
│   ├── values-glm52-2tp8.yaml    # W1 (primary, .152)
│   ├── values-glm52-2tp8-w2.yaml # W2 (secondary, .172)
│   ├── values-glm52-test.yaml    # Legacy test env
│   ├── values-glm52-test-w2.yaml # Legacy test worker
│   ├── values-prod.yaml          # ZW prod (planned)
│   ├── values-test.yaml          # Legacy
│   └── templates/
│       ├── _helpers.tpl          # Label/selectors helpers
│       ├── sglang-statefulset.yaml
│       ├── sglang-service.yaml
│       ├── sglang-router.yaml     # sgl-model-gateway router
│       ├── sglang-httproute.yaml  # Envoy gateway HTTPRoute
│       └── llm-d-router.yaml      # llm-d EPP router (alternative)
├── patches/
│   ├── fp8_mqa_logits.py          # BLOCK_KV=64 for gfx942 (Triton fallback)
│   └── flydsl/
│       ├── __init__.py
│       └── fp8_mqa_logits.py      # FlyDSL kernel for gfx942
├── pd-test-gz-rdma/               # PD disaggregated test manifests
│   ├── pd-prefill-152.yaml
│   ├── pd-prefill-144.yaml
│   ├── pd-prefill-144-tcp.yaml
│   ├── pd-decode-132.yaml
│   ├── pd-decode-132-tcp.yaml
│   ├── pd-decode-172.yaml
│   ├── pd-router-152.yaml
│   ├── pd-router-172.yaml
│   ├── pd-router-172-tcp.yaml
│   └── test-correctness.sh
└── scripts/
    ├── benchmark-v14.sh           # 4-scenario benchmark
    └── verify-v14.sh              # Health + smoke + long context + MTP
```

---

## Git History

This branch (`fix/eagle-decode-coredump-mi308x`) contains a single squashed
commit on top of `origin/main` @ `50c118704a`:

```
39145f548d fix(eagle): comprehensive EAGLE decode coredump fix for MI308X GLM-5.2
50c118704a (origin/main, main) [diffusion] disagg: handle numpy arrays ...
```

The original 5 commits (in `sglang-offical-github` repo) were:

| Commit | Description |
|--------|-------------|
| `250019ef99` | Comprehensive EAGLE coredump fix (PR #31478 + docker/chart infra) |
| `cb91a13ef7` | fix(dsa): use self. prefix (WRONG — superseded) |
| `a9bc24365b` | fix(dsa): hoist scale/logits_head_gate_graph out of if _is_cuda block |
| `b4628cf86a` | fix(chart): use tc_piecewise prefill backend on ROCm (not breakable) |
| `50b9138541` | fix(chart): reduce hicacheRatio 4 to 2 for DSA indexer host memory |

### Image Version History

| Tag | Status | Issue |
|-----|--------|-------|
| `fix-eagle-coredump` | v1 | NameError (functions not defined on HIP) |
| `fix-eagle-coredump-v2` | v2 | AttributeError (wrong self. prefix fix) |
| **`fix-eagle-coredump-v3`** | **v3 (current)** | **hoist + tc_piecewise + ratio=2, all pass** |

---

## Contact

- Maintainer: guofutan
- Cluster: TKE `cls-bmmk3vtl` (GZ test), namespace `kube-system`
- Nodes: `node-21.151.225.152` (W1), `node-21.151.225.172` (W2)
- Gateway: `glm52-2tp8.jmpti.woa.com` → envoy LB → router → workers
