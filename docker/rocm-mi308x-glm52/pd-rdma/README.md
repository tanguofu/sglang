# GLM-5.2-FP8 PD RDMA Disaggregation on AMD MI308X

> **Complete deployment guide for RDMA-based Prefill-Decode disaggregation on AMD MI308X (gfx942) with SGLang + Mooncake.**
>
> This document records every bug encountered, root cause found, and fix applied. Following this guide step-by-step will reproduce the working deployment.

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Environment Prerequisites](#2-environment-prerequisites)
3. [Bug Fix History (Root Cause → Fix)](#3-bug-fix-history-root-cause--fix)
4. [Docker Image Build](#4-docker-image-build)
5. [Deployment](#5-deployment)
6. [Verification](#6-verification)
7. [File Reference](#7-file-reference)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Architecture Overview

```
                    External Request
                          │
                          ▼
                ┌─────────────────────┐
                │  PD Router (13002)  │  SGLang Rust Router
                │  --pd-disaggregation │  Routes to prefill, pairs with decode
                └──────┬───────┬──────┘
                       │       │
          ┌────────────┘       └────────────┐
          ▼                                  ▼
┌─────────────────────┐          ┌─────────────────────┐
│  Prefill (13000)    │          │  Decode (13000)     │
│  21.234.170.19      │          │  21.234.170.32      │
│  TP=8, no MTP       │  RDMA    │  TP=8, MTP=3       │
│  Bootstrap: 12999   │ ───────► │  Bootstrap: 12999   │
│  RDMA: bnxt_re_bond0│  KV xfer │  RDMA: bnxt_re_bond0│
│  29.198.252.130     │          │  29.198.252.190     │
└─────────────────────┘          └─────────────────────┘
```

**Key design decisions:**
- **RDMA with host staging**: GPU memory cannot be directly RDMA'd (kernel lacks P2PDMA). Instead, KV data is `hipMemcpy`'d D2H before RDMA send, and H2D after RDMA receive. The network transfer still uses RDMA (not TCP).
- **eth0 for Mooncake RPC**: Mooncake's P2P handshake and session management uses TCP on eth0 (within security group range). RDMA data transfer uses bnxt_re_bond0.
- **bond0 for all TP ranks**: All 8 TP ranks use `bnxt_re_bond0` (not bond0-7) to simplify RDMA endpoint setup and avoid P2P handshake failures.
- **Port range 10000-13000**: Security group `sg-d6yni7o7` allows TCP 10000-13000. Server=13000, bootstrap=12999, router=13002.

---

## 2. Environment Prerequisites

### 2.1 Hardware

| Component | Prefill Node | Decode Node |
|-----------|-------------|-------------|
| Node name | node-21.234.170.19 | node-21.234.170.32 |
| Instance | ins-dewy6nun | ins-qwn8n8ij |
| CPU | 384 cores | 384 cores |
| Memory | 2304GB | 2304GB |
| GPU | 8x MI308X (gfx942) | 8x MI308X (gfx942) |
| RDMA NIC | 8x bnxt_re bond (400Gb/s) | 8x bnxt_re bond (400Gb/s) |
| bond0 IP | 29.198.252.130/30 | 29.198.252.190/30 |
| eth0 IP | 21.234.170.19 | 21.234.170.32 |

### 2.2 Software

| Component | Version |
|-----------|---------|
| OS | TencentOS Server 3.1 (Final) |
| Kernel | 5.4.119-19.0009.60 |
| ROCm | 7.2.0 |
| SGLang | v0.5.14 (base image: `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi30x-20260710`) |
| Mooncake | Built from source (in base image) |
| K8s | v1.30.0-tke.20 |
| Container Runtime | containerd://1.6.9-tke.8 |

### 2.3 Model

- **Model**: GLM-5.2-FP8 (141 safetensors shards, ~108GB per GPU)
- **Path**: `/data/model/glm52-fp8` on both nodes
- **Quantization**: FP8 (e4m3) weights, FP8 KV cache

### 2.4 Security Group

Security group `sg-d6yni7o7` must allow:
- TCP:10000-13000 between ZW nodes (21.234.170.19 ↔ 21.234.170.32)
- This covers: server port (13000), bootstrap port (12999), router port (13002)

### 2.5 RDMA Devices

```bash
# Verify RDMA devices on both nodes
ls /sys/class/infiniband/
# Expected: bnxt_re_bond0 bnxt_re_bond1 ... bnxt_re_bond7

# Verify GID index 3 (RoCE v2 IPv4-routable)
cat /sys/class/infiniband/bnxt_re_bond0/ports/1/gids/3
# Expected: 0000:0000:0000:0000:0000:ffff:1dc6:fc82 (= 29.198.252.130)
```

### 2.6 Kernel P2PDMA Check

```bash
# Check if kernel supports P2PDMA (it does NOT on TencentOS 5.4)
grep -c pci_p2pdma /proc/kallsyms
# Expected: 0 (this is why we need host staging)

zcat /proc/config.gz 2>/dev/null | grep PCI_P2PDMA
# Expected: # CONFIG_PCI_P2PDMA is not set
```

---

## 3. Bug Fix History (Root Cause → Fix)

### Bug 1: `hipIpcOpenMemHandle error 17`

**Symptom**: Mooncake HIP transport fails to install on ROCm.

**Root cause**: Mooncake's HIP transport uses `hipIpcOpenMemHandle` for inter-node GPU memory sharing, which fails on ROCm (AMD's IPC implementation differs from NVIDIA's).

**Fix**: Patch Mooncake C++ source to wrap HIP transport install in an env var check:

```cpp
// In transfer_engine_impl.cpp, wrap the HIP transport install:
if (!std::getenv("MC_DISABLE_HIP_TRANSPORT") ||
    std::string(std::getenv("MC_DISABLE_HIP_TRANSPORT")) != "1") {
    Transport* hip_transport = multi_transports_->installTransport("hip", nullptr);
    // ...
} else {
    LOG(INFO) << "HIP transport disabled by MC_DISABLE_HIP_TRANSPORT=1";
}
```

**Env var**: `MC_DISABLE_HIP_TRANSPORT=1`

---

### Bug 2: Mooncake patch not compiled into .so

**Symptom**: After patching Mooncake source, the `MC_DISABLE_HIP_TRANSPORT` string is not found in the compiled `engine.cpython*.so`.

**Root cause**: Three issues:
1. The cmake target name is `engine` (not `mooncake-transfer-engine` or `mooncake-integration`)
2. `cmake --build` segfaults in `cmake_check_build_system` (memory issue in 16GB Colima VM)
3. Stale `.o`/`.a`/`.so` artifacts prevent recompilation

**Fix**: Use `make engine -j4` directly (not `cmake --build`), and delete stale artifacts:

```dockerfile
RUN python3 /tmp/patch_mooncake.py && \
    find /sgl-workspace/Mooncake/build -name "transfer_engine_impl.cpp.o" -delete && \
    find /sgl-workspace/Mooncake/build -name "transfer_engine_impl.cpp.o.d" -delete && \
    find /sgl-workspace/Mooncake/build -name "rdma_context.cpp.o" -delete && \
    find /sgl-workspace/Mooncake/build -name "rdma_context.cpp.o.d" -delete && \
    rm -f /sgl-workspace/Mooncake/build/mooncake-transfer-engine/src/libtransfer_engine.a && \
    rm -f /sgl-workspace/Mooncake/build/mooncake-integration/engine.cpython-310-x86_64-linux-gnu.so && \
    cd /sgl-workspace/Mooncake/build && make engine -j4 && \
    cp /sgl-workspace/Mooncake/build/mooncake-integration/engine.cpython-310-x86_64-linux-gnu.so \
       /opt/venv/lib/python3.10/site-packages/mooncake/engine.cpython-310-x86_64-linux-gnu.so
```

---

### Bug 3: RDMA KV transfer corrupts data

**Symptom**: PD inference returns garbage output or 500 errors.

**Root cause**: Kernel 5.4.119-19.0009.60 lacks `CONFIG_PCI_P2PDMA`. Mooncake's `isKernelDmabufSupported()` checks `/proc/kallsyms` for `pci_p2pdma` and `dma_buf_move_notify`; both return 0. Mooncake falls back to `ibv_reg_mr()` on GPU memory, which returns a valid MR but RDMA transfers silently corrupt data because GPU memory is not pinned for PCIe DMA.

**Fix**: Enable `SGLANG_PD_HOST_STAGING=1`. SGLang allocates host-side `ctypes.c_char` buffers, registers them with Mooncake RDMA (host memory is always RDMA-accessible), and `hipMemcpy`s D2H before send and H2D after receive.

---

### Bug 4: `send_kvcache()` uses GPU pointers (not host staging)

**Symptom**: `Memory region not registered by any active device(s)` error during KV transfer.

**Root cause**: SGLang's `send_kvcache()` (used for same-TP PD transfers) uses `self.kv_args.kv_data_ptrs` (GPU pointers) directly as RDMA source addresses. With host staging, only host buffers are registered with Mooncake RDMA, not GPU memory.

**Fix**: Added `_copy_gpu_to_host()` method and modified `send_kvcache()` to D2H copy before transfer and use host staging pointers:

```python
# In mooncake/conn.py, MooncakeKVManager:

def _copy_gpu_to_host(self):
    """Copy KV data from GPU to host staging buffers before PD transfer."""
    import ctypes
    hip_lib = ctypes.CDLL("libamdhip64.so")
    for gpu_ptr, host_ptr, length in zip(
        self._gpu_ptrs, self._host_staging_ptrs, self._host_staging_lens,
    ):
        ret = hip_lib.hipMemcpy(
            ctypes.c_void_p(int(host_ptr)),
            ctypes.c_void_p(int(gpu_ptr)),
            ctypes.c_size_t(length),
            ctypes.c_int(2),  # hipMemcpyDeviceToHost
        )

def send_kvcache(self, mooncake_session_id, prefill_kv_indices, dst_kv_ptrs, dst_kv_indices, executor):
    if hasattr(self, "_host_staging_ptrs") and self._host_staging_ptrs:
        self._copy_gpu_to_host()
        src_ptrs = self._host_staging_ptrs
    else:
        src_ptrs = self.kv_args.kv_data_ptrs
    return self._send_kvcache_generic(
        mooncake_session_id=mooncake_session_id,
        src_data_ptrs=src_ptrs,
        # ...
    )
```

---

### Bug 5: Decode sends GPU pointers via bootstrap

**Symptom**: `Failed to get segment descriptor for segment ... address 0x7f...` error.

**Root cause**: The decode's `_register_kv_args()` sends `self.kv_mgr.kv_args.kv_data_ptrs` (GPU pointers) to the prefill via the bootstrap server. The prefill uses these as RDMA write destinations, which must be registered host memory (not unregistered GPU memory).

**Fix**: Modified `_register_kv_args()` to send host staging pointers when host staging is enabled:

```python
# In mooncake/conn.py, MooncakeKVReceiver._register_kv_args():

def _register_kv_args(self):
    kv_data_ptrs = (
        self.kv_mgr._host_staging_ptrs
        if hasattr(self.kv_mgr, "_host_staging_ptrs") and self.kv_mgr._host_staging_ptrs
        else self.kv_mgr.kv_args.kv_data_ptrs
    )
    for bootstrap_info in self.bootstrap_infos:
        packed_kv_data_ptrs = b"".join(
            struct.pack("Q", ptr) for ptr in kv_data_ptrs
        )
```

---

### Bug 6: Mooncake P2P handshake failure (multi-bond)

**Symptom**: `SocketHandShakePlugin: failed to receive handshake message, malformed json format` error.

**Root cause**: When using 8 different bond devices (bond0-7) for 8 TP ranks, each rank's RPC server listens on bond0's IP while RDMA transport uses different bonds. The P2P handshake between ranks fails.

**Fix**: Use `bnxt_re_bond0` for all TP ranks:

```bash
--disaggregation-ib-device '{"0":"bnxt_re_bond0","1":"bnxt_re_bond0","2":"bnxt_re_bond0","3":"bnxt_re_bond0","4":"bnxt_re_bond0","5":"bnxt_re_bond0","6":"bnxt_re_bond0","7":"bnxt_re_bond0"}'
```

---

### Bug 7: Bootstrap port outside security group

**Symptom**: `Connection refused` when prefill tries to connect to decode's bootstrap server.

**Root cause**: Bootstrap port 8998 (default) and 13001 are outside the security group range (10000-13000).

**Fix**: Use port 12999 for bootstrap (within security group range):

```bash
ENV DISAGG_BOOTSTRAP_PORT=12999
```

---

### Bug 8: `--sleep-on-idle` deadlocks PD disaggregation

**Symptom**: PD disaggregation hangs — TP0 sleeps while TP1-7 enter NCCL broadcast.

**Root cause**: `--sleep-on-idle` causes TP0 to sleep when idle, but during PD bootstrap, all TP ranks must participate in NCCL communication simultaneously.

**Fix**: Remove `--sleep-on-idle` from PD start scripts. (Already removed in current scripts.)

---

## 4. Docker Image Build

### 4.1 Prerequisites

```bash
# Start Colima with x86_64 emulation (for building on arm64 Mac)
colima start --cpu 8 --memory 16 --disk 200

# Verify Docker
docker info
```

### 4.2 Build

```bash
# From the repository root:
bash docker/rocm-mi308x-glm52/pd-rdma/build-and-push.sh 0712-rdma10
```

### 4.3 Dockerfile Structure

Both Dockerfiles (`Dockerfile.prefill` and `Dockerfile.decode`) share the same structure:

```dockerfile
FROM lmsysorg/sglang-rocm:v0.5.14-rocm720-mi30x-20260710

# Step 1: Replace sglang source with pre-patched version (0711-opt HIP fixes)
COPY python/sglang/ /sgl-workspace/sglang/python/sglang/

# Step 1b: FlyDSL gfx942 kernel + Triton fallback
COPY docker/rocm-mi308x-glm52/patches/flydsl/fp8_mqa_logits.py /sgl-workspace/aiter/aiter/ops/flydsl/kernels/
COPY docker/rocm-mi308x-glm52/patches/flydsl/__init__.py /sgl-workspace/aiter/aiter/ops/flydsl/
COPY docker/rocm-mi308x-glm52/patches/fp8_mqa_logits.py /sgl-workspace/aiter/aiter/ops/triton/attention/

# Step 1c: Patch Mooncake C++ source and rebuild the Python extension .so
COPY docker/rocm-mi308x-glm52/patches/mooncake/patch_mooncake.py /tmp/patch_mooncake.py
RUN python3 /tmp/patch_mooncake.py && \
    find ... -name "*.o" -delete && \
    rm -f .../libtransfer_engine.a && \
    rm -f .../engine.cpython*.so && \
    cd /sgl-workspace/Mooncake/build && make engine -j4 && \
    cp .../engine.cpython*.so /opt/venv/lib/python3.10/site-packages/mooncake/

# Step 2: Environment variables
ENV MC_GID_INDEX=3
ENV MC_DISABLE_HIP_TRANSPORT=1
ENV SGLANG_PD_HOST_STAGING=1
ENV MOONCAKE_PROTOCOL=rdma
# ... (see Dockerfile for full list)

# Step 3: Entrypoint
COPY docker/rocm-mi308x-glm52/start_prefill.sh /start_server.sh
ENTRYPOINT ["/start_server.sh"]
```

### 4.4 Mooncake Patch Script

`patches/mooncake/patch_mooncake.py` applies two patches:

1. **MC_DISABLE_HIP_TRANSPORT** (in `transfer_engine_impl.cpp`): Wraps HIP transport install in env var check.
2. **MC_FORCE_DMABUF** (in `rdma_context.cpp`): Bypasses `isKernelDmabufSupported()` check to force dmabuf MR registration path (fallback when host staging is not used).

---

## 5. Deployment

### 5.1 Quick Deploy

```bash
# From the repository root:
bash docker/rocm-mi308x-glm52/pd-rdma/deploy.sh
```

This script:
1. Cleans up old pods and GPU processes
2. Deploys prefill and decode pods
3. Waits for both servers to be healthy
4. Deploys the router
5. Tests inference

### 5.2 Manual Deploy

```bash
# Step 1: Clean up
bash docker/rocm-mi308x-glm52/pd-rdma/clean-gpu.sh

# Step 2: Deploy prefill
kubectl apply -f docker/rocm-mi308x-glm52/pd-rdma/prefill.yaml

# Step 3: Deploy decode
kubectl apply -f docker/rocm-mi308x-glm52/pd-rdma/decode.yaml

# Step 4: Wait for both to be ready (takes ~3-4 minutes for model loading)
kubectl get pods -n kube-system -l "app in (pd-prefill-rdma, pd-decode-rdma)" -w

# Step 5: Deploy router
kubectl apply -f docker/rocm-mi308x-glm52/pd-rdma/router.yaml

# Step 6: Test
bash docker/rocm-mi308x-glm52/pd-rdma/test-inference.sh
```

### 5.3 Environment Variables (Baked into Image)

| Variable | Value | Purpose |
|----------|-------|---------|
| `MC_DISABLE_HIP_TRANSPORT` | `1` | Skip HIP transport (hipIpcOpenMemHandle fails on ROCm) |
| `MC_GID_INDEX` | `3` | RoCE v2 IPv4-routable GID index |
| `SGLANG_PD_HOST_STAGING` | `1` | Use host buffers for RDMA (kernel lacks P2PDMA) |
| `MC_TCP_ENABLE_CONNECTION_POOL` | `true` | Connection pooling for Mooncake TCP |
| `MOONCAKE_PROTOCOL` | `rdma` | Use RDMA for data transfer |
| `HIP_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | All 8 GPUs visible |
| `NCCL_DEBUG` | `INFO` | NCCL debug logging |
| `HSA_ENABLE_SDMA` | `0` | Disable SDMA (ROCm workaround) |
| `HIP_FORCE_DEV_KERNARG` | `1` | Force device kernarg (ROCm workaround) |
| `HSA_NO_SCRATCH_RECLAIM` | `1` | Disable scratch reclaim (ROCm workaround) |
| `NCCL_CUMEM_ENABLE` | `0` | Disable NCCL cumulative memory |
| `NCCL_MIN_NCHANNELS` | `112` | NCCL channel count |
| `PYTORCH_ROCM_ARCH` | `gfx942` | Target GPU architecture |
| `SGLANG_USE_AITER` | `1` | Use AITER kernels |
| `SGLANG_USE_ROCM700A` | `1` | Use ROCm 7.0.0A features |
| `SGLANG_OPT_USE_TOPK_V2` | `0` | Disable topk_v2 (GPU hang on gfx942) |

### 5.4 Runtime Env Vars (Set in Pod YAML)

| Variable | Prefill | Decode | Purpose |
|----------|---------|--------|---------|
| `SGLANG_HOST_IP` | `21.234.170.19` | `21.234.170.32` | Mooncake RPC listen address (eth0) |
| `MODEL_PATH` | `/data/model/glm52-fp8` | `/data/model/glm52-fp8` | Model checkpoint path |
| `PORT` | `13000` | `13000` | SGLang server port |
| `DISAGG_BOOTSTRAP_PORT` | `12999` | `12999` | Bootstrap server port |
| `API_KEY` | `sk-46fa...` | `sk-46fa...` | API key |

### 5.5 Start Script Parameters

**Prefill** (`start_prefill.sh`):
```bash
python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp-size 8 --pp-size 1 --trust-remote-code \
    --host 0.0.0.0 --port "$PORT" \
    --context-length 1048576 \
    --mem-fraction-static 0.90 \
    --kv-cache-dtype fp8_e4m3 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device '{"0":"bnxt_re_bond0",...all bond0...}' \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    --disable-overlap-schedule \
    --cuda-graph-backend-prefill breakable \
    --cuda-graph-bs-prefill 4 8 16 32 \
    --max-running-requests 128 \
    --enable-metrics --skip-server-warmup \
    --watchdog-timeout 3600 --log-level info
```

**Decode** (`start_decode.sh`):
```bash
python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp-size 8 --pp-size 1 --trust-remote-code \
    --host 0.0.0.0 --port "$PORT" \
    --context-length 1048576 \
    --mem-fraction-static 0.88 \
    --kv-cache-dtype fp8_e4m3 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
    --speculative-eagle-topk 1 \
    --cuda-graph-bs-decode 1 2 4 8 16 \
    --cuda-graph-max-bs-decode 16 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device '{"0":"bnxt_re_bond0",...all bond0...}' \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    --disable-overlap-schedule \
    --enable-metrics --skip-server-warmup \
    --watchdog-timeout 3600 --log-level info
```

**Router**:
```bash
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://21.234.170.19:13000 12999 \
    --decode http://21.234.170.32:13000 \
    --host 0.0.0.0 --port 13002 \
    --log-level info
```

---

## 6. Verification

### 6.1 Health Check

```bash
# Check all pods are running
kubectl get pods -n kube-system -l "app in (pd-prefill-rdma, pd-decode-rdma, pd-router-rdma)"

# Check health endpoints
kubectl exec pd-prefill-rdma -n kube-system -- python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:13000/health').read())"
kubectl exec pd-decode-rdma -n kube-system -- python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:13000/health').read())"
kubectl exec pd-router-rdma -n kube-system -- python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:13002/health').read())"
```

### 6.2 Log Verification

```bash
# Verify Mooncake patch is active (should see on all 8 TP ranks):
kubectl logs pd-prefill-rdma -n kube-system | grep "HIP transport disabled"
# Expected: "HIP transport disabled by MC_DISABLE_HIP_TRANSPORT=1"

# Verify host staging is active:
kubectl logs pd-prefill-rdma -n kube-system | grep "Host staging"
# Expected: "Host staging: registered 78 host buffers for KV data (total 48116809728 bytes)"

kubectl logs pd-decode-rdma -n kube-system | grep "Host staging"
# Expected: "Host staging: registered 79 host buffers for KV data (total 43220791296 bytes)"

# Verify RDMA transport:
kubectl logs pd-prefill-rdma -n kube-system | grep "Transfer Engine RPC"
# Expected: "Transfer Engine RPC using P2P handshake, listening on 21.234.170.19:..."
```

### 6.3 Inference Test

```bash
bash docker/rocm-mi308x-glm52/pd-rdma/test-inference.sh
```

Expected output:
```json
{
  "id": "...",
  "object": "chat.completion",
  "model": "glm-5.2",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Four.",
      "reasoning_content": "..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 5,
    "reasoning_tokens": 51
  }
}
```

---

## 7. File Reference

### 7.1 Deployment Files (`docker/rocm-mi308x-glm52/pd-rdma/`)

| File | Purpose |
|------|---------|
| `prefill.yaml` | Prefill pod YAML (node-21.234.170.19) |
| `decode.yaml` | Decode pod YAML (node-21.234.170.32) |
| `router.yaml` | Router pod YAML |
| `deploy.sh` | One-command deploy script |
| `clean-gpu.sh` | Clean up pods and GPU processes |
| `test-inference.sh` | Inference test script |
| `build-and-push.sh` | Build and push Docker images |
| `README.md` | This document |

### 7.2 Dockerfiles (`docker/rocm-mi308x-glm52/`)

| File | Purpose |
|------|---------|
| `Dockerfile.prefill` | Prefill image (no MTP, higher mem fraction) |
| `Dockerfile.decode` | Decode image (MTP enabled, cuda graph decode) |

### 7.3 Start Scripts (`docker/rocm-mi308x-glm52/`)

| File | Purpose |
|------|---------|
| `start_prefill.sh` | Prefill entrypoint |
| `start_decode.sh` | Decode entrypoint |

### 7.4 Patches (`docker/rocm-mi308x-glm52/patches/`)

| File | Purpose |
|------|---------|
| `mooncake/patch_mooncake.py` | Mooncake C++ source patcher (MC_DISABLE_HIP_TRANSPORT + MC_FORCE_DMABUF) |
| `flydsl/fp8_mqa_logits.py` | FlyDSL gfx942 FP8 MQA logits kernel |
| `flydsl/__init__.py` | FlyDSL module init |
| `fp8_mqa_logits.py` | Triton fallback for FP8 MQA logits |

### 7.5 SGLang Code Fixes (`python/sglang/srt/disaggregation/mooncake/conn.py`)

Three modifications to `MooncakeKVManager`:

1. **`_copy_gpu_to_host()`** (new method): D2H copy before RDMA transfer
2. **`send_kvcache()`**: Use host staging pointers as RDMA source
3. **`MooncakeKVReceiver._register_kv_args()`**: Send host staging pointers via bootstrap

---

## 8. Troubleshooting

### 8.1 "Memory region not registered by any active device(s)"

**Cause**: `send_kvcache()` is using GPU pointers instead of host staging pointers.

**Fix**: Ensure the SGLang code fix is applied (check `send_kvcache()` calls `_copy_gpu_to_host()` and uses `self._host_staging_ptrs`).

### 8.2 "Failed to get segment descriptor"

**Cause**: Decode is sending GPU pointers to prefill via bootstrap.

**Fix**: Ensure `_register_kv_args()` sends `_host_staging_ptrs` instead of `kv_data_ptrs`.

### 8.3 "Session ... is not alive"

**Cause**: Mooncake P2P handshake failed between ranks.

**Fix**: Use `bnxt_re_bond0` for all TP ranks (not bond0-7). Check that `SGLANG_HOST_IP` is set to eth0 IP (not bond0 IP).

### 8.4 "Connection refused" on bootstrap port

**Cause**: Bootstrap port is outside security group range.

**Fix**: Use port 12999 (within 10000-13000 range).

### 8.5 Server takes >5 minutes to start

**Normal**: Model loading takes ~100s, CUDA graph capture takes ~60s, host staging buffer allocation takes ~10s. Total startup time is ~3-4 minutes.

### 8.6 Decode pod restarts (CrashLoopBackOff)

**Cause**: Replacing `kv_data_ptrs` with host staging pointers broke the scheduler's KV cache access.

**Fix**: Do NOT replace `kv_data_ptrs` globally. Only send host staging pointers in `_register_kv_args()` (bootstrap exchange). The `kv_data_ptrs` must remain as GPU pointers for the scheduler.

### 8.7 Empty response content

**Cause**: `max_tokens` too small — model uses all tokens for reasoning (thinking mode).

**Fix**: Set `max_tokens` to at least 200 to allow reasoning + content generation.

---

## Git History

All fixes are on branch `308x-pd-rdma` with the following commits:

```
d8109f8e83 fix(pd): send host staging ptrs in bootstrap, not kv_data_ptrs
b6be517c5b fix(pd): replace kv_data_ptrs with host staging pointers for decode
7ed1964c86 fix(pd): add D2H copy to send_kvcache for host staging
b7b763fcc0 fix(pd): use bond0 for all TP ranks to fix Mooncake P2P handshake
38c4040c8b fix(pd): RDMA PD disaggregation with host staging on AMD MI308X
```

## Image Tags

| Tag | Description |
|-----|-------------|
| `0712-rdma10` | **Latest working** — all fixes applied |
| `0712-rdma9` | kv_data_ptrs replacement (broken) |
| `0712-rdma8` | send_kvcache D2H copy added |
| `0712-rdma7` | bond0-only for all TP ranks |
| `0712-rdma6` | SGLANG_PD_HOST_STAGING=1 + MC_FORCE_DMABUF |
