# GLM-5.2-FP8 on AMD MI308X (gfx942) — EAGLE Coredump 修复完整部署文档

> **维护者**: guofutan (谭国富)
> **创建时间**: 2026-07-20
> **分支**: `fix/eagle-decode-coredump-mi308x` (基于 `origin/main` @ `50c118704a`)
> **仓库**: [tanguofu/sglang](https://github.com/tanguofu/sglang/tree/fix/eagle-decode-coredump-mi308x)
> **镜像**: `mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3`
> **iWiki 文档**: https://iwiki.woa.com/p/4026586166

---

## 一、概述

### 1.1 背景

原部署 (`xgmi-opt-0716d` 镜像,基于 sglang `b76dd0be` 2026-07-10) 存在严重 bug:当 EAGLE 推测解码在 >1024 token prefill 后运行时,触发 8-rank GPU coredump。根因是 sglang 容器代码落后上游 380 个 commit,缺失多个关键 EAGLE/CUDA graph 修复,同时存在 3 个 ROCm 特有问题。

### 1.2 修复内容

- **3 个 ROCm 特有问题修复**: NameError (函数 hoist), AssertionError (tc_piecewise), Host OOM (hicacheRatio=2)
- **6 个上游 EAGLE 修复**: 78dc581518, 7a973c03a0, fc1e3797b7, cce5fe7696, 942bf04ef9, 7e229e2a81
- **1 个本地 patch**: PR #31478 (EAGLE greedy 分支 TP broadcast)

### 1.3 部署拓扑

| 组件 | 节点 | 角色 |
|------|------|------|
| W1 (sglang-glm52-2tp8) | node-21.151.225.152 | 主 worker (TP8) + router + gateway |
| W2 (sglang-glm52-2tp8-w2) | node-21.151.225.172 | 副 worker (TP8) |
| Router | (随 W1) | sgl-model-gateway, cache_aware 策略 |
| Gateway | envoy LB | `glm52-2tp8.jmpti.woa.com` → router |

### 1.4 硬件与软件栈

- **硬件**: 2x AMD MI308X 节点 (8x gfx942 GPU/节点,192GB VRAM/GPU,~2TB host RAM)
- **OS**: TencentOS Server 4.4, kernel 6.6
- **GPU 驱动**: amdgpu 6.16.13 DKMS + MOK 签名
- **ROCm**: 7.2.4 (hip-runtime, miopen, rccl, rocfft, rocsparse, rocrand)
- **K8s**: TKE cluster `cls-bmmk3vtl`, namespace `kube-system`
- **自定义镜像**: `img-ebtth3fd` (ap-zhongwei, 含完整驱动栈,跳过 2-3 小时安装)

---

## 二、镜像构建

### 2.1 镜像信息

| 字段 | 值 |
|------|-----|
| 镜像 | `mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3` |
| Base | `lmsysorg/sglang-rocm:v0.5.15.post1-rocm720-mi30x-20260718` |
| 分支 | `fix/eagle-decode-coredump-mi308x` (squashed commit `39145f548d`) |
| 构建耗时 | ~27s (base 已缓存),首次 ~15-20 min |

### 2.2 镜像版本演进

| Tag | 状态 | 问题 |
|-----|------|------|
| fix-eagle-coredump | v1 | NameError (函数未定义 on HIP) |
| fix-eagle-coredump-v2 | v2 | AttributeError (错误的 self. 修复) |
| **fix-eagle-coredump-v3** | **v3 (当前)** | **hoist + tc_piecewise + ratio=2,全部通过** |

### 2.3 构建命令

```bash
# Clone 分支
git clone https://github.com/tanguofu/sglang.git
cd sglang
git checkout fix/eagle-decode-coredump-mi308x

# 构建 (需要 Docker,base ~15GB)
docker build -f docker/rocm-mi308x-glm52/Dockerfile \
  -t mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3 .

# 推送 (需要 /etc/hosts: 30.163.240.137 mirrors.tencent.com)
docker push mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3

# 验证
docker run --rm --entrypoint python3 \
  mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3 \
  -c "import sglang.srt.layers.attention.dsa.dsa_indexer; print('OK')"
```

### 2.4 完整 Dockerfile

```dockerfile
# GLM-5.2-FP8 SGLang Worker for AMD MI308X (gfx942)
# Based on official mi30x ROCm image with upstream main + EAGLE coredump fixes.
#
# Build (from repo root):
#   docker build -f docker/rocm-mi308x-glm52/Dockerfile \
#     -t sglang-glm52-308x:fix-eagle-coredump .
#
# Run:
#   docker run -d --name sglang_308x \
#     --privileged --network host --shm-size 32g \
#     --device /dev/kfd --device /dev/dri --group-add video \
#     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
#     -v /data:/data \
#     sglang-glm52-308x:fix-eagle-coredump
#
# This branch (fix/eagle-decode-coredump-mi308x) is based on upstream main
# (b3570a4531, 2026-07-20) which contains 380 commits beyond the previous
# container base (b76dd0be, 2026-07-10), including critical EAGLE fixes:
#   - 78dc581518 Stabilize GLM-5.2 MTP IndexShare across PD and CUDA graph replay
#   - 7a973c03a0 Stamp capture-time num_tokens_per_req in multi-layer EAGLE
#   - fc1e3797b7 Split capture width from num_tokens_per_req and gate replay
#   - cce5fe7696 Move WAR barrier right after each run_batch launch
#   - 942bf04ef9 Add SGLANG_FORCE_COARSE_WAR_BARRIER opt-in
# Plus our local patch:
#   - PR #31478: TP broadcast in EAGLE greedy branch (not yet merged upstream)

FROM lmsysorg/sglang-rocm:v0.5.15.post1-rocm720-mi30x-20260718

LABEL maintainer="guofutan"
LABEL description="GLM-5.2-FP8 SGLang worker for AMD MI308X (gfx942) with EAGLE coredump fix"
LABEL sglang-base-image="v0.5.15.post1-rocm720-mi30x-20260718"
LABEL gpu-arch="gfx942"
LABEL branch="fix/eagle-decode-coredump-mi308x"
LABEL upstream-base="b3570a4531 (origin/main, 2026-07-20)"
LABEL local-patch="PR #31478 TP broadcast in EAGLE greedy branch"

# ============================================================
# Step 1: Replace sglang source with upstream main + local patches
# ============================================================
# The official image has sglang installed in editable mode pointing to
# /sgl-workspace/sglang/python/sglang/. We overwrite the source files
# with our fix branch version (upstream main + PR #31478).
COPY python/sglang/ /sgl-workspace/sglang/python/sglang/

# Verify the EAGLE coredump fix is present (fail build if missing)
RUN python3 -c "\
eagle = open('/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_utils.py').read(); \
assert 'PR #31478' in eagle, 'PATCH MISSING: PR #31478 TP broadcast in greedy branch'; \
assert 'tp_group.broadcast(predict, src=0)' in eagle, 'PATCH MISSING: TP broadcast call'; \
ml_runner = open('/sgl-workspace/sglang/python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py').read(); \
assert 'captured_req_width' in ml_runner, 'UPSTREAM FIX MISSING: fc1e3797b7 captured_req_width'; \
assert 'spec_info.num_tokens_per_req = self.captured_req_width' in ml_runner, 'UPSTREAM FIX MISSING: 7a973c03a0 stamp num_tokens_per_req at capture'; \
eagle_info = open('/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_info.py').read(); \
assert 'future_dsa_topk_indices_available' in eagle_info, 'UPSTREAM FIX MISSING: 78dc581518 dsa_topk_indices stabilization'; \
eagle_worker = open('/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py').read(); \
assert 'seed_dsa_topk_from_draft_extend' in eagle_worker, 'UPSTREAM FIX MISSING: 78dc581518 seed_dsa_topk guard'; \
sched = open('/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py').read(); \
assert 'SGLANG_FORCE_COARSE_WAR_BARRIER' in sched, 'UPSTREAM FIX MISSING: 942bf04ef9 coarse WAR barrier'; \
assert '_apply_war_barrier' in sched, 'UPSTREAM FIX MISSING: cce5fe7696 WAR barrier after run_batch'; \
print('All EAGLE coredump fixes verified: PR #31478 + 78dc581518 + 7a973c03a0 + fc1e3797b7 + cce5fe7696 + 942bf04ef9')"

# ============================================================
# Step 1b: Add FlyDSL gfx942 fp8 MQA logits kernel + patch Triton path
# MI308X (gfx942) has 64KB shared memory limit vs MI355X's 80KB+.
# ============================================================
COPY docker/rocm-mi308x-glm52/patches/flydsl/fp8_mqa_logits.py /sgl-workspace/aiter/aiter/ops/flydsl/kernels/fp8_mqa_logits.py
COPY docker/rocm-mi308x-glm52/patches/flydsl/__init__.py /sgl-workspace/aiter/aiter/ops/flydsl/__init__.py
COPY docker/rocm-mi308x-glm52/patches/fp8_mqa_logits.py /sgl-workspace/aiter/aiter/ops/triton/attention/fp8_mqa_logits.py

# ============================================================
# Step 2: Environment Variables (based on MI355X production config)
# Key change: PYTORCH_ROCM_ARCH=gfx942 (was gfx950 for MI355X)
# ============================================================
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV NCCL_DEBUG=WARN
ENV HSA_ENABLE_SDMA=0
ENV HIP_FORCE_DEV_KERNARG=1
ENV HSA_NO_SCRATCH_RECLAIM=1
ENV NCCL_CUMEM_ENABLE=0
ENV NCCL_MIN_NCHANNELS=80
ENV NCCL_NVLS_ENABLE=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PYTORCH_ROCM_ARCH=gfx942
ENV ROCM_QUICK_REDUCE_QUANTIZATION=NONE
ENV SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
ENV SGLANG_DISABLE_CUDNN_CHECK=1
ENV SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM=1
ENV SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false
ENV SGLANG_FORCE_COARSE_WAR_BARRIER=true
ENV SGLANG_INT4_WEIGHT=0
ENV SGLANG_MOE_PADDING=1
ENV SGLANG_ROCM_DISABLE_LINEARQUANT=0
ENV SGLANG_ROCM_FUSED_DECODE_MLA=1
ENV SGLANG_SET_CPU_AFFINITY=1
ENV SGLANG_USE_AITER=1
ENV SGLANG_USE_ROCM700A=1

# Runtime-overridable params
ENV MODEL_PATH=/data/model/glm52-fp8
ENV PORT=30000
ENV API_KEY=sk-46faecc9d0bc4dcd9db6a15c73ae91c8

# ============================================================
# Entrypoint
# ============================================================
COPY docker/rocm-mi308x-glm52/start_server.sh /start_server.sh
RUN chmod +x /start_server.sh

ENTRYPOINT ["/start_server.sh"]
```

### 2.5 完整环境变量表

| 变量 | 值 | 说明 |
|------|-----|------|
| `HIP_VISIBLE_DEVICES` | 0,1,2,3,4,5,6,7 | 全部 8 GPU |
| `NCCL_DEBUG` | WARN | 抑制 INFO 日志刷屏 (slows health probe) |
| `HSA_ENABLE_SDMA` | 0 | MI308X P2P/XGMI 正确设置 |
| `HIP_FORCE_DEV_KERNARG` | 1 | 强制 device kernarg |
| `HSA_NO_SCRATCH_RECLAIM` | 1 | 不回收 scratch 内存 |
| `NCCL_CUMEM_ENABLE` | 0 | 关闭 NCCL CUMEM |
| `NCCL_MIN_NCHANNELS` | 80 | RCCL 2.27.7 硬上限 (higher = clamped + spam) |
| `NCCL_NVLS_ENABLE` | 0 | 关闭 NVLS (ROCm 不需要) |
| `PYTORCH_CUDA_ALLOC_CONF` | expandable_segments:True | PyTorch 显存分配策略 |
| `PYTORCH_ROCM_ARCH` | gfx942 | MI308X 架构 |
| `ROCM_QUICK_REDUCE_QUANTIZATION` | NONE | 零精度损失 (原 INT8) |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | 1 | 允许覆盖 context length |
| `SGLANG_DISABLE_CUDNN_CHECK` | 1 | 关闭 cuDNN check |
| `SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM` | 1 | DSv2 双流 |
| `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` | false | /health 直接返回 200 (不跑 prefill) |
| `SGLANG_FORCE_COARSE_WAR_BARRIER` | true | 强制 coarse WAR barrier (EAGLE overlap 稳定性) |
| `SGLANG_INT4_WEIGHT` | 0 | 关闭 INT4 权重 |
| `SGLANG_MOE_PADDING` | 1 | MoE padding 优化 |
| `SGLANG_ROCM_DISABLE_LINEARQUANT` | 0 | 不禁用 linear quant |
| `SGLANG_ROCM_FUSED_DECODE_MLA` | 1 | Fused decode MLA |
| `SGLANG_SET_CPU_AFFINITY` | 1 | CPU pinning |
| `SGLANG_USE_AITER` | 1 | AMD AITER kernels |
| `SGLANG_USE_ROCM700A` | 1 | ROCm 7.0A features |
| `MODEL_PATH` | /data/model/glm52-fp8 | 模型权重路径 |
| `PORT` | 30000 | 服务端口 |
| `API_KEY` | sk-46faecc9d0bc4dcd9db6a15c73ae91c8 | API 密钥 |

---

## 三、启动命令

### 3.1 完整 start_server.sh

```bash
#!/usr/bin/env bash
# Entrypoint for GLM-5.2-FP8 SGLang worker on AMD MI308X (gfx942).
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/data/model/glm52-fp8}
API_KEY=${API_KEY:-sk-46faecc9d0bc4dcd9db6a15c73ae91c8}
PORT=${PORT:-30000}

echo "============================================"
echo " GLM-5.2-FP8 SGLang Worker (MI308X gfx942)"
echo "============================================"
echo " Model:  $MODEL_PATH"
echo " Port:   $PORT"
echo "============================================"

exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --model-impl sglang \
    --served-model-name glm-5.2 \
    --api-key "$API_KEY" \
    --tp-size 8 --pp-size 1 --trust-remote-code \
    --host 0.0.0.0 --port "$PORT" \
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
    --cuda-graph-backend-prefill tc_piecewise \
    --max-running-requests 32 \
    --cuda-graph-bs-prefill 4 8 16 32 \
    --enable-metrics --skip-server-warmup \
    --watchdog-timeout 3600 --log-level info
```

### 3.2 启动参数详解

| 参数 | 值 | 说明 |
|------|-----|------|
| `--model-path` | /data/model/glm52-fp8 | GLM-5.2 FP8 权重 |
| `--model-impl` | sglang | 使用 sglang 实现 |
| `--served-model-name` | glm-5.2 | API 返回的模型名 |
| `--tp-size` | 8 | 8 GPU 张量并行 |
| `--pp-size` | 1 | 流水并行 1 (单机) |
| `--context-length` | 1048576 | 1M context (生产用 524288) |
| `--tool-call-parser` | glm47 | 工具调用解析器 |
| `--reasoning-parser` | glm45 | 推理解析器 |
| `--mem-fraction-static` | 0.88 | 静态显存比例 (生产用 0.75) |
| `--cuda-graph-bs-decode` | 1 2 3 4 5 6 7 8 9 10 12 16 | decode CUDA graph batch sizes |
| `--cuda-graph-max-bs-decode` | 16 | decode CUDA graph 最大 batch |
| `--enable-aiter-allreduce-fusion` | (flag) | AITER all-reduce 融合 |
| `--enable-mixed-chunk` | (flag) | 混合 chunk prefill+decode |
| `--chunked-prefill-size` | 32768 | chunked prefill 大小 (生产用 16384) |
| `--enable-fused-qk-norm-rope` | (flag) | 融合 QK norm + RoPE |
| `--schedule-conservativeness` | 0.5 | 调度保守度 (生产用 1.0) |
| `--prefill-max-requests` | 32 | 最大并发 prefill (生产用 4) |
| `--max-prefill-tokens` | 32768 | 单批最大 prefill tokens |
| `--kv-cache-dtype` | fp8_e4m3 | FP8 KV cache |
| `--speculative-algorithm` | NEXTN | EAGLE MTP 推测解码 |
| `--speculative-num-steps` | 3 | MTP 步数 |
| `--speculative-num-draft-tokens` | 4 | 每 step draft tokens |
| `--speculative-eagle-topk` | 1 | EAGLE top-k |
| `--cuda-graph-backend-prefill` | **tc_piecewise** | **NOT breakable** (ROCm 关键) |
| `--max-running-requests` | 32 | 最大运行请求数 |
| `--cuda-graph-bs-prefill` | 4 8 16 32 | prefill CUDA graph batch sizes |
| `--enable-metrics` | (flag) | 启用 metrics |
| `--skip-server-warmup` | (flag) | 跳过 warmup |
| `--watchdog-timeout` | 3600 | 看门狗超时 (生产用 1200) |
| `--log-level` | info | 日志级别 |

**注意**: 以上是 start_server.sh 的默认值。**生产部署通过 helm values 覆盖**多个参数(见下方 4.3 节),实际启动命令由 StatefulSet 的 command/args 生成,包含 NUMA 绑定等额外参数。

### 3.3 生产实际启动命令 (由 helm 渲染)

```bash
# 安装 numactl (若缺失)
apt-get update -qq && apt-get install -y -qq numactl

# 实际启动 (由 helm chart 模板渲染)
exec python3 -m sglang.launch_server \
    --model-path /data/model/glm52-fp8 \
    --model-impl sglang \
    --served-model-name glm-5.2 \
    --api-key sk-46faecc9d0bc4dcd9db6a15c73ae91c8 \
    --tp-size 8 --pp-size 1 --trust-remote-code \
    --host 0.0.0.0 --port 30000 \
    --numa-node 0 0 0 0 1 1 1 1 \
    --context-length 524288 \
    --tool-call-parser glm47 --reasoning-parser glm45 \
    --mem-fraction-static 0.75 \
    --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 12 16 \
    --cuda-graph-max-bs-decode 16 \
    --enable-aiter-allreduce-fusion --enable-mixed-chunk \
    --chunked-prefill-size 16384 \
    --enable-fused-qk-norm-rope \
    --schedule-conservativeness 1.0 \
    --prefill-max-requests 4 --max-prefill-tokens 32768 \
    --kv-cache-dtype fp8_e4m3 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
    --speculative-eagle-topk 1 \
    --cuda-graph-backend-prefill tc_piecewise \
    --max-running-requests 32 \
    --cuda-graph-bs-prefill 4 8 16 32 \
    --enable-hierarchical-cache \
    --hicache-ratio 2 \
    --hicache-io-backend direct \
    --hicache-mem-layout page_first_direct \
    --hicache-write-policy write_back \
    --enable-metrics --skip-server-warmup \
    --watchdog-timeout 1200 --log-level info
```

---

## 四、Helm Chart 部署

### 4.1 Chart 结构

```
docker/rocm-mi308x-glm52/chart/
├── Chart.yaml                     # chart 元数据
├── values.yaml                    # chart 默认值
├── values-glm52-2tp8.yaml         # W1 (.152) 生产配置
├── values-glm52-2tp8-w2.yaml      # W2 (.172) 生产配置
├── values-glm52-test.yaml         # 测试环境
├── values-glm52-test-w2.yaml      # 测试 worker
├── values-prod.yaml               # 中卫生产 (规划)
├── values-test.yaml               # 旧测试
└── templates/
    ├── _helpers.tpl               # label/selector helpers
    ├── sglang-statefulset.yaml    # StatefulSet (worker)
    ├── sglang-service.yaml        # Service + Headless Service
    ├── sglang-router.yaml         # sgl-model-gateway router
    ├── sglang-httproute.yaml      # Envoy HTTPRoute
    └── llm-d-router.yaml          # llm-d EPP router (备选)
```

### 4.2 Chart.yaml

```yaml
apiVersion: v2
name: sglang-glm52-308x
description: SGLang GLM-5.2-FP8 inference service on AMD MI308X (gfx942) with envoy gateway
type: application
version: 0.1.0
appVersion: "0.5.14"
keywords:
  - sglang
  - glm-5.2
  - amd
  - mi308x
  - gfx942
  - rocm
  - llm
maintainers:
  - name: guofutan
```

### 4.3 values.yaml (chart 默认值,完整)

```yaml
# SGLang GLM-5.2 on AMD MI308X (gfx942) — Helm Chart Values

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
image: mirrors.tencent.com/ti-platform/sglang-glm52-308x
tag: latest
pullPolicy: IfNotPresent
imagePullSecret: tencent-registry

# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------
namespace: kube-system
replicas: 1
hostNetwork: true
port: 30000

# ---------------------------------------------------------------------------
# Node selection
# ---------------------------------------------------------------------------
nodeName: ""
nodeSelector:
  accelerator: amd-gpu
  sglang-model: ready

# Tolerations applied to the worker StatefulSet.
tolerations: []

# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
resources:
  requests:
    cpu: 360
    memory: 2100Gi
    amd.com/gpu: 8
  limits:
    amd.com/gpu: 8

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model:
  path: /data/model/glm52-fp8
  servedName: glm-5.2
  apiKey: sk-46faecc9d0bc4dcd9db6a15c73ae91c8

# ---------------------------------------------------------------------------
# Server launch parameters
# ---------------------------------------------------------------------------
sglang:
  tpSize: 8
  ppSize: 1
  contextLength: "1048576"
  # NUMA memory binding — TP0-3 → NUMA 0, TP4-7 → NUMA 1
  numaNode: "0 0 0 0 1 1 1 1"
  memFractionStatic: 0.88
  kvCacheDtype: fp8_e4m3
  chunkedPrefillSize: 32768
  maxRunningRequests: 32
  prefillMaxRequests: 32
  maxPrefillTokens: 32768
  scheduleConservativeness: 0.5
  cudaGraphBsDecode: "1 2 3 4 5 6 7 8 9 10 12 16"
  cudaGraphMaxBsDecode: 16
  cudaGraphBsPrefill: "4 8 16 32"
  # Prefill CUDA graph backend — tc_piecewise (NOT breakable).
  # BCG is CUDA-only per upstream default_prefill_backend().
  # is_graph_dsa_split_op_surface() gated by is_cuda(), returns False on ROCm.
  cudaGraphBackendPrefill: tc_piecewise
  speculativeAlgorithm: NEXTN
  speculativeNumSteps: 3
  speculativeNumDraftTokens: 4
  speculativeEagleTopk: 1
  enableHierarchicalCache: true
  hicacheRatio: 2.0
  hicacheIoBackend: direct
  hicacheMemLayout: page_first_direct
  hicacheWritePolicy: write_through
  enableAiterAllreduceFusion: true
  enableMixedChunk: true
  enableFusedQkNormRope: true
  enableMetrics: true
  skipServerWarmup: true
  watchdogTimeout: 3600
  logLevel: info
  toolCallParser: glm47
  reasoningParser: glm45

# ---------------------------------------------------------------------------
# Host paths
# ---------------------------------------------------------------------------
hostPaths:
  data: /data
  devKfd: /dev/kfd
  devDri: /dev/dri

shmSize: 32Gi

# ---------------------------------------------------------------------------
# SGLang Router (sgl-model-gateway) — DISABLED by default
# ---------------------------------------------------------------------------
router:
  enabled: false
  port: 30001
  policy: cache_aware
  workerUrls:
    - "http://21.151.225.144:30000"
    - "http://21.151.225.132:30000"
    - "http://21.151.225.172:30000"
    - "http://21.151.225.152:30000"
  cacheThreshold: 0.3
  balanceAbsThreshold: 64
  balanceRelThreshold: 1.5
  image: ""
  tag: ""
  nodeName: ""
  nodeSelector: {}
  tolerations: []

# ---------------------------------------------------------------------------
# llm-d Router — EPP + Envoy sidecar (备选)
# ---------------------------------------------------------------------------
llmDRouter:
  enabled: true
  port: 30001
  eppImage: mirrors.tencent.com/ti-platform/llm-d-router-endpoint-picker
  eppVersion: v0.9.0-amd64
  envoyImage: mirrors.tencent.com/ti-platform/envoy
  envoyVersion: distroless-v1.33.2-amd64
  nodeName: ""
  endpoints:
    - name: sglang-test-0
      address: 21.151.225.144
      port: 30000
    - name: sglang-test-1
      address: 21.151.225.132
      port: 30000
    - name: sglang-test-2
      address: 21.151.225.172
      port: 30000
    - name: sglang-test-3
      address: 21.151.225.152
      port: 30000

# ---------------------------------------------------------------------------
# Envoy Gateway — TEST environment (广州 GZ)
# ---------------------------------------------------------------------------
gateway:
  enabled: true
  gatewayName: eg-tke
  gatewayNamespace: ti-cloud
  hostname: glm52-308x-test.jmpti.woa.com
  lbAddress: 21.162.215.14

# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------
readinessProbe:
  httpGet:
    path: /health
    port: 30000
  initialDelaySeconds: 120
  periodSeconds: 30
  timeoutSeconds: 10
  failureThreshold: 10

livenessProbe:
  httpGet:
    path: /health
    port: 30000
  initialDelaySeconds: 600
  periodSeconds: 60
  timeoutSeconds: 10
  failureThreshold: 5
```

### 4.4 values-glm52-2tp8.yaml (W1 .152 生产配置,完整)

```yaml
# SGLang GLM-5.2 2tp8 deployment — 152 + 172 dual TP8 workers with cache_aware router
#
# Topology:
#   - sglang-glm52-2tp8    on node-21.151.225.152 (worker + router + gateway)
#   - sglang-glm52-2tp8-w2 on node-21.151.225.172 (worker only)
#
# History:
#   2026-07-18: OOM fix — memFraction 0.88→0.75, prefillMaxRequests 32→4,
#               chunkedPrefillSize 32768→16384, scheduleConservativeness 0.5→1.0
#   2026-07-19: HiCache fix — write_through→write_back, hicacheRatio 1→8
#   2026-07-19: hicacheRatio 8→4 — NUMA 1 OOM on node .152
#   2026-07-20: hicacheRatio 4→2 — new upstream DSA indexer allocates 56.89 GB/rank

# ---------------------------------------------------------------------------
# Image — fix-eagle-coredump build
# ---------------------------------------------------------------------------
image: mirrors.tencent.com/ti-platform/sglang-glm52-308x
tag: fix-eagle-coredump-v3

# ---------------------------------------------------------------------------
# Worker placement — pin to 152 (primary, hosts router + gateway)
# ---------------------------------------------------------------------------
nodeName: node-21.151.225.152
replicas: 1

# ---------------------------------------------------------------------------
# SGLang server parameters — OOM fix (2026-07-18) + HiCache fix (2026-07-19)
# ---------------------------------------------------------------------------
sglang:
  contextLength: "524288"          # 512K (chart default is 1M, reduced for stability)
  memFractionStatic: 0.75          # OOM fix: was 0.88, left ~2.5GB/rank activation room
  chunkedPrefillSize: 16384        # OOM fix: was 32768, smaller chunks = less peak activation
  prefillMaxRequests: 4            # OOM fix: was 32, limit concurrent prefills
  scheduleConservativeness: 1.0    # OOM fix: was 0.5, more conservative scheduling
  hicacheRatio: 2                  # HiCache: was 4, DSA indexer needs 57 GB/rank host RAM
  hicacheWritePolicy: write_back   # HiCache fix: fixes host load-back=0
  watchdogTimeout: 1200            # fail faster on stuck detokenizer

# ---------------------------------------------------------------------------
# SGLang Router (sgl-model-gateway with /v1/messages support)
# ---------------------------------------------------------------------------
router:
  enabled: true
  port: 30001
  policy: cache_aware
  workerUrls:
    - "http://21.151.225.152:30000"  # primary (this release)
    - "http://21.151.225.172:30000"  # secondary (sglang-glm52-2tp8-w2)
  cacheThreshold: 0.2
  balanceAbsThreshold: 1
  balanceRelThreshold: 1.2
  image: mirrors.tencent.com/ti-platform/sglang-glm52-308x-pd-router
  tag: messages-0717c
  tolerations:
    - key: dedicated
      operator: Equal
      value: sglang-2tp8
      effect: NoSchedule
    - operator: Exists

# ---------------------------------------------------------------------------
# llm-d Router — DISABLED, we use sgl-model-gateway
# ---------------------------------------------------------------------------
llmDRouter:
  enabled: false

# ---------------------------------------------------------------------------
# Envoy Gateway — 2tp8 hostname
# ---------------------------------------------------------------------------
gateway:
  enabled: true
  gatewayName: eg-tke
  gatewayNamespace: ti-cloud
  hostname: glm52-2tp8.jmpti.woa.com
  lbAddress: 21.162.215.14
```

### 4.5 values-glm52-2tp8-w2.yaml (W2 .172 生产配置,完整)

```yaml
# SGLang GLM-5.2 2tp8 deployment — worker 2 (no router, no gateway)
# Usage:
#   helm install sglang-glm52-2tp8-w2 .../chart/ -f values-glm52-2tp8-w2.yaml
#
# Topology:
#   - sglang-glm52-2tp8    on node-21.151.225.152 (worker + router + gateway)
#   - sglang-glm52-2tp8-w2 on node-21.151.225.172 (this file, worker only)

# ---------------------------------------------------------------------------
# Image — fix-eagle-coredump build (same as worker 1)
# ---------------------------------------------------------------------------
image: mirrors.tencent.com/ti-platform/sglang-glm52-308x
tag: fix-eagle-coredump-v3

# ---------------------------------------------------------------------------
# Worker placement — pin to 172 (secondary, worker only)
# ---------------------------------------------------------------------------
nodeName: node-21.151.225.172
replicas: 1

# ---------------------------------------------------------------------------
# SGLang server parameters — OOM fix (2026-07-18) + HiCache fix (2026-07-19)
# ---------------------------------------------------------------------------
sglang:
  contextLength: "524288"          # 512K
  memFractionStatic: 0.75          # OOM fix
  chunkedPrefillSize: 16384        # OOM fix
  prefillMaxRequests: 4            # OOM fix
  scheduleConservativeness: 1.0    # OOM fix
  hicacheRatio: 2                  # parity with w1
  hicacheWritePolicy: write_back   # HiCache fix
  watchdogTimeout: 1200

# ---------------------------------------------------------------------------
# Router — DISABLED on worker 2 (router lives on worker 1)
# ---------------------------------------------------------------------------
router:
  enabled: false

# ---------------------------------------------------------------------------
# llm-d Router — DISABLED
# ---------------------------------------------------------------------------
llmDRouter:
  enabled: false

# ---------------------------------------------------------------------------
# Envoy Gateway — DISABLED on worker 2
# ---------------------------------------------------------------------------
gateway:
  enabled: false
```

### 4.6 templates/_helpers.tpl (完整)

```yaml
{{/*
Common labels
*/}}
{{- define "sglang.labels" -}}
app.kubernetes.io/name: sglang-glm52-308x
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
app: sglang
accelerator: amd-gpu
{{- end -}}

{{/*
Selector labels — minimal labels for pod selection (immutable, kept empty)
*/}}
{{- define "sglang.selectorLabels" -}}
{{- end -}}

{{/*
Service selector labels — includes instance for per-release isolation
*/}}
{{- define "sglang.serviceSelectorLabels" -}}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Full image reference — worker
*/}}
{{- define "sglang.image" -}}
{{ .Values.image }}:{{ .Values.tag }}
{{- end -}}

{{/*
Router image — uses router.image/tag when set, otherwise worker image
*/}}
{{- define "sglang.router.image" -}}
{{- if and .Values.router.image .Values.router.tag -}}
{{ .Values.router.image }}:{{ .Values.router.tag }}
{{- else -}}
{{ include "sglang.image" . }}
{{- end -}}
{{- end -}}
```

### 4.7 templates/sglang-statefulset.yaml (完整)

```yaml
# SGLang LLM Inference Service — StatefulSet for AMD MI308X (gfx942)
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {{ .Release.Name }}-sglang
  namespace: {{ .Values.namespace }}
  labels:
    {{- include "sglang.labels" . | nindent 4 }}
spec:
  serviceName: {{ .Release.Name }}-sglang
  replicas: {{ .Values.replicas }}
  selector:
    matchLabels:
      {{- include "sglang.selectorLabels" . | nindent 6 }}
      app: sglang
  template:
    metadata:
      labels:
        {{- include "sglang.labels" . | nindent 8 }}
    spec:
      hostNetwork: {{ .Values.hostNetwork }}
      dnsPolicy: ClusterFirstWithHostNet
      {{- if .Values.nodeName }}
      nodeName: {{ .Values.nodeName | quote }}
      {{- else }}
      nodeSelector:
        {{- toYaml .Values.nodeSelector | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- if .Values.imagePullSecret }}
      imagePullSecrets:
        - name: {{ .Values.imagePullSecret }}
      {{- end }}
      terminationGracePeriodSeconds: 300
      containers:
        - name: sglang
          image: {{ include "sglang.image" . }}
          imagePullPolicy: {{ .Values.pullPolicy }}
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -euo pipefail
              MODEL_PATH=${MODEL_PATH:-/data/model/glm52-fp8}
              API_KEY=${API_KEY:-sk-46faecc9d0bc4dcd9db6a15c73ae91c8}
              PORT=${PORT:-30000}
              if ! command -v numactl &>/dev/null; then
                echo "Installing numactl..."
                apt-get update -qq && apt-get install -y -qq numactl >/dev/null 2>&1
              fi
              echo "=== GLM-5.2-FP8 SGLang (MI308X gfx942) ==="
              exec python3 -m sglang.launch_server \
                --model-path "$MODEL_PATH" \
                --model-impl sglang \
                --served-model-name {{ .Values.model.servedName }} \
                --api-key "$API_KEY" \
                --tp-size {{ .Values.sglang.tpSize }} --pp-size {{ .Values.sglang.ppSize }} --trust-remote-code \
                --host 0.0.0.0 --port "$PORT" \
                --numa-node {{ .Values.sglang.numaNode }} \
                --context-length {{ .Values.sglang.contextLength | quote }} \
                --tool-call-parser {{ .Values.sglang.toolCallParser }} --reasoning-parser {{ .Values.sglang.reasoningParser }} \
                --mem-fraction-static {{ .Values.sglang.memFractionStatic }} \
                --cuda-graph-bs-decode {{ .Values.sglang.cudaGraphBsDecode }} \
                --cuda-graph-max-bs-decode {{ .Values.sglang.cudaGraphMaxBsDecode }} \
                --enable-aiter-allreduce-fusion --enable-mixed-chunk \
                --chunked-prefill-size {{ .Values.sglang.chunkedPrefillSize }} \
                --enable-fused-qk-norm-rope \
                --schedule-conservativeness {{ .Values.sglang.scheduleConservativeness }} \
                --prefill-max-requests {{ .Values.sglang.prefillMaxRequests }} --max-prefill-tokens {{ .Values.sglang.maxPrefillTokens }} \
                --kv-cache-dtype {{ .Values.sglang.kvCacheDtype }} \
                {{- if ne .Values.sglang.speculativeAlgorithm "NONE" }}
                --speculative-algorithm {{ .Values.sglang.speculativeAlgorithm }} \
                --speculative-num-steps {{ .Values.sglang.speculativeNumSteps }} --speculative-num-draft-tokens {{ .Values.sglang.speculativeNumDraftTokens }} \
                --speculative-eagle-topk {{ .Values.sglang.speculativeEagleTopk }} \
                {{- end }}
                --cuda-graph-backend-prefill {{ .Values.sglang.cudaGraphBackendPrefill }} \
                --max-running-requests {{ .Values.sglang.maxRunningRequests }} \
                --cuda-graph-bs-prefill {{ .Values.sglang.cudaGraphBsPrefill }} \
                {{- if .Values.sglang.enableHierarchicalCache }}
                --enable-hierarchical-cache \
                --hicache-ratio {{ .Values.sglang.hicacheRatio }} \
                --hicache-io-backend {{ .Values.sglang.hicacheIoBackend }} \
                --hicache-mem-layout {{ .Values.sglang.hicacheMemLayout }} \
                --hicache-write-policy {{ .Values.sglang.hicacheWritePolicy }} \
                {{- end }}
                --enable-metrics --skip-server-warmup \
                --watchdog-timeout {{ .Values.sglang.watchdogTimeout }} --log-level {{ .Values.sglang.logLevel }}
          env:
            - name: MODEL_PATH
              value: {{ .Values.model.path | quote }}
            - name: PORT
              value: {{ .Values.port | quote }}
            - name: API_KEY
              value: {{ .Values.model.apiKey | quote }}
            # ROCm environment — 23 vars
            - name: HIP_VISIBLE_DEVICES
              value: "0,1,2,3,4,5,6,7"
            - name: NCCL_DEBUG
              value: "WARN"
            - name: HSA_ENABLE_SDMA
              value: "0"
            - name: HIP_FORCE_DEV_KERNARG
              value: "1"
            - name: HSA_NO_SCRATCH_RECLAIM
              value: "1"
            - name: NCCL_CUMEM_ENABLE
              value: "0"
            - name: NCCL_MIN_NCHANNELS
              value: "80"
            - name: NCCL_NVLS_ENABLE
              value: "0"
            - name: PYTORCH_CUDA_ALLOC_CONF
              value: "expandable_segments:True"
            - name: PYTORCH_ROCM_ARCH
              value: "gfx942"
            - name: ROCM_QUICK_REDUCE_QUANTIZATION
              value: "NONE"
            - name: SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN
              value: "1"
            - name: SGLANG_DISABLE_CUDNN_CHECK
              value: "1"
            - name: SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION
              value: "false"
            - name: SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM
              value: "1"
            - name: SGLANG_FORCE_COARSE_WAR_BARRIER
              value: "true"
            - name: SGLANG_INT4_WEIGHT
              value: "0"
            - name: SGLANG_MOE_PADDING
              value: "1"
            - name: SGLANG_ROCM_DISABLE_LINEARQUANT
              value: "0"
            - name: SGLANG_ROCM_FUSED_DECODE_MLA
              value: "1"
            - name: SGLANG_SET_CPU_AFFINITY
              value: "1"
            - name: SGLANG_USE_AITER
              value: "1"
            - name: SGLANG_USE_ROCM700A
              value: "1"
            - name: CUDA_ENABLE_USER_TRIGGERED_COREDUMP
              value: "1"
          securityContext:
            privileged: true
            capabilities:
              add:
                - SYS_PTRACE
              drop: ["ALL"]
            seccompProfile:
              type: Unconfined
          ports:
            - name: http
              containerPort: {{ .Values.port }}
              hostPort: {{ .Values.port }}
              protocol: TCP
          volumeMounts:
            - name: data
              mountPath: {{ .Values.hostPaths.data }}
            - name: dev-kfd
              mountPath: {{ .Values.hostPaths.devKfd }}
            - name: dev-dri
              mountPath: {{ .Values.hostPaths.devDri }}
            - name: shm
              mountPath: /dev/shm
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          readinessProbe:
            {{- toYaml .Values.readinessProbe | nindent 12 }}
          livenessProbe:
            {{- toYaml .Values.livenessProbe | nindent 12 }}
      volumes:
        - name: data
          hostPath:
            path: {{ .Values.hostPaths.data }}
            type: Directory
        - name: dev-kfd
          hostPath:
            path: {{ .Values.hostPaths.devKfd }}
            type: CharDevice
        - name: dev-dri
          hostPath:
            path: {{ .Values.hostPaths.devDri }}
            type: Directory
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: {{ .Values.shmSize }}
```

### 4.8 templates/sglang-router.yaml (完整)

```yaml
# SGLang Router — cache_aware prefix-aware load balancer
{{- if .Values.router.enabled }}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-router
  namespace: {{ .Values.namespace }}
  labels:
    {{- include "sglang.labels" . | nindent 4 }}
    app: sglang-router
spec:
  replicas: 1
  selector:
    matchLabels:
      {{- include "sglang.selectorLabels" . | nindent 6 }}
      app: sglang-router
  template:
    metadata:
      labels:
        {{- include "sglang.labels" . | nindent 8 }}
        app: sglang-router
    spec:
      hostNetwork: true
      dnsPolicy: ClusterFirstWithHostNet
      {{- if .Values.imagePullSecret }}
      imagePullSecrets:
        - name: {{ .Values.imagePullSecret }}
      {{- end }}
      {{- if .Values.router.nodeName }}
      nodeName: {{ .Values.router.nodeName | quote }}
      {{- else if .Values.router.nodeSelector }}
      nodeSelector:
        {{- toYaml .Values.router.nodeSelector | nindent 8 }}
      {{- else }}
      nodeSelector:
        accelerator: amd-gpu
      {{- end }}
      {{- with .Values.router.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      terminationGracePeriodSeconds: 30
      containers:
        - name: router
          image: {{ include "sglang.router.image" . }}
          imagePullPolicy: {{ .Values.pullPolicy }}
          command: ["python3", "-m", "sglang_router.launch_router"]
          args:
            - --worker-urls
            {{- range .Values.router.workerUrls }}
            - {{ . | quote }}
            {{- end }}
            - --policy
            - {{ .Values.router.policy | quote }}
            - --host
            - "0.0.0.0"
            - --port
            - {{ .Values.router.port | quote }}
            {{- if eq .Values.router.policy "cache_aware" }}
            - --cache-threshold
            - {{ .Values.router.cacheThreshold | quote }}
            - --balance-abs-threshold
            - {{ .Values.router.balanceAbsThreshold | quote }}
            - --balance-rel-threshold
            - {{ .Values.router.balanceRelThreshold | quote }}
            {{- end }}
          ports:
            - name: http
              containerPort: {{ .Values.router.port }}
              protocol: TCP
          resources:
            requests:
              cpu: 2
              memory: 4Gi
            limits:
              cpu: 4
              memory: 8Gi
          readinessProbe:
            httpGet:
              path: /health
              port: {{ .Values.router.port }}
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: {{ .Values.router.port }}
            initialDelaySeconds: 15
            periodSeconds: 30
            timeoutSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: {{ .Release.Name }}-router
  namespace: {{ .Values.namespace }}
  labels:
    {{- include "sglang.labels" . | nindent 4 }}
    app: sglang-router
spec:
  type: ClusterIP
  selector:
    {{- include "sglang.serviceSelectorLabels" . | nindent 4 }}
    app: sglang-router
  ports:
    - name: http
      port: {{ .Values.router.port }}
      targetPort: http
      protocol: TCP
{{- end }}
```

### 4.9 templates/sglang-service.yaml (完整)

```yaml
# SGLang Service — ClusterIP + Headless Service for StatefulSet
apiVersion: v1
kind: Service
metadata:
  name: {{ .Release.Name }}-sglang
  namespace: {{ .Values.namespace }}
  labels:
    {{- include "sglang.labels" . | nindent 4 }}
spec:
  type: ClusterIP
  selector:
    {{- include "sglang.serviceSelectorLabels" . | nindent 4 }}
    app: sglang
  ports:
    - name: http
      port: {{ .Values.port }}
      targetPort: http
      protocol: TCP
---
apiVersion: v1
kind: Service
metadata:
  name: {{ .Release.Name }}-sglang-headless
  namespace: {{ .Values.namespace }}
  labels:
    {{- include "sglang.labels" . | nindent 4 }}
spec:
  type: ClusterIP
  clusterIP: None
  selector:
    {{- include "sglang.serviceSelectorLabels" . | nindent 4 }}
    app: sglang
  ports:
    - name: http
      port: {{ .Values.port }}
      targetPort: http
      protocol: TCP
```

### 4.10 templates/sglang-httproute.yaml (完整)

```yaml
{{- if .Values.gateway.enabled }}
# HTTPRoute — envoy gateway forwarding to SGLang Router
# External: http://<hostname> → envoy LB → router → TP8 instances
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: {{ .Release.Name }}-sglang
  namespace: {{ .Values.namespace }}
  labels:
    {{- include "sglang.labels" . | nindent 4 }}
spec:
  hostnames:
    - {{ .Values.gateway.hostname | quote }}
  parentRefs:
    - group: gateway.networking.k8s.io
      kind: Gateway
      name: {{ .Values.gateway.gatewayName }}
      namespace: {{ .Values.gateway.gatewayNamespace }}
      sectionName: http
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /
      backendRefs:
        - group: ""
          kind: Service
          {{- if .Values.llmDRouter.enabled }}
          name: {{ .Release.Name }}-llm-d-router
          namespace: {{ .Values.namespace }}
          port: {{ .Values.llmDRouter.port }}
          {{- else if .Values.router.enabled }}
          name: {{ .Release.Name }}-router
          namespace: {{ .Values.namespace }}
          port: {{ .Values.router.port }}
          {{- else }}
          name: {{ .Release.Name }}-sglang
          namespace: {{ .Values.namespace }}
          port: {{ .Values.port }}
          {{- end }}
          weight: 1
{{- end }}
```

---

## 五、部署步骤 (从 0 开始)

### 5.1 节点准备

```bash
# 1. 验证 GPU 健康 (每节点)
rocm-smi --showproductname --showhealth  # 8 GPU healthy
cat /sys/module/amdgpu/version            # 6.16.13

# 2. Label 节点
kubectl label node node-21.151.225.152 \
  accelerator=amd-gpu sglang-model=ready --overwrite
kubectl label node node-21.151.225.172 \
  accelerator=amd-gpu sglang-model=ready --overwrite

# 3. (可选) 添加 taint (dedication)
kubectl taint node node-21.151.225.152 dedicated=sglang-2tp8:NoSchedule
kubectl taint node node-21.151.225.172 dedicated=sglang-2tp8:NoSchedule

# 4. 验证模型权重
ssh node-21.151.225.152 'ls /data/model/glm52-fp8/*.safetensors | wc -l'
ssh node-21.151.225.172 'ls /data/model/glm52-fp8/*.safetensors | wc -l'

# 5. 验证 host RAM (>1600 GB free)
ssh node-21.151.225.152 'free -g | awk "/^Mem:/{print \$4\" GB free\"}"'
```

### 5.2 镜像构建

```bash
# Clone 分支
git clone https://github.com/tanguofu/sglang.git
cd sglang
git checkout fix/eagle-decode-coredump-mi308x

# 构建
docker build -f docker/rocm-mi308x-glm52/Dockerfile \
  -t mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3 .

# 推送
docker push mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3
```

### 5.3 Helm 部署

```bash
# W1 (主 worker + router + gateway)
helm install sglang-glm52-2tp8 \
  docker/rocm-mi308x-glm52/chart/ -n kube-system \
  -f docker/rocm-mi308x-glm52/chart/values-glm52-2tp8.yaml

# W2 (副 worker)
helm install sglang-glm52-2tp8-w2 \
  docker/rocm-mi308x-glm52/chart/ -n kube-system \
  -f docker/rocm-mi308x-glm52/chart/values-glm52-2tp8-w2.yaml
```

### 5.4 监控启动 (3-5 分钟)

```bash
# W1 日志
kubectl logs -n kube-system sglang-glm52-2tp8-sglang-0 -f

# W2 日志
kubectl logs -n kube-system sglang-glm52-2tp8-w2-sglang-0 -f
```

**预期启动序列** (日志标记):
```
# 1. Model loading (45s)
Loading model from /data/model/glm52-fp8...

# 2. Decode CUDA graph (67s)
Capturing CUDA graphs for decode (bs=1,2,3,4,5,6,7,8,9,10,12,16)...
Capturing EAGLE draft decode CUDA graph...

# 3. Prefill CUDA graph (23s, tc_piecewise)
Capturing tc_piecewise prefill CUDA graphs (bs=4,8,16,32)...

# 4. HiCache (12s, 83 GB/rank)
Allocating HiCache host pool: ratio=2, 83.22 GB/rank...

# 5. DSA indexer (57 GB/rank)
Allocating DSA indexer host memory: 56.89 GB/rank, 455.12 GB total

# 6. Ready
Uvicorn running on http://0.0.0.0:30000
```

### 5.5 验证部署

```bash
# Pod 状态
kubectl get pod -n kube-system -l app=sglang -o wide
# 期望: 2 worker + 1 router, 1/1 Running, 0 restarts

# 健康检查
curl -s http://21.151.225.152:30000/health  # → ok
curl -s http://21.151.225.172:30000/health  # → ok

# Smoke test
curl -s http://21.151.225.152:30000/v1/chat/completions \
  -H "Authorization: Bearer sk-46faecc9d0bc4dcd9db6a15c73ae91c8" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"What is 2+3?"}],"max_tokens":16}'

# EAGLE coredump 回归测试 (原 bug 触发场景, 2048 prefill)
LONG_TEXT=$(python3 -c 'print("The quick brown fox jumps over the lazy dog. " * 150)')
curl -s http://21.151.225.152:30000/v1/chat/completions \
  -H "Authorization: Bearer sk-46faecc9d0bc4dcd9db6a15c73ae91c8" \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json
text = 'The quick brown fox jumps over the lazy dog. ' * 150
print(json.dumps({'model':'glm-5.2','messages':[{'role':'user','content':text+'What animal is mentioned?'}],'max_tokens':50}))
")"
# 期望: HTTP 200 + 有效响应, 无 coredump

# Metrics
curl -s http://21.151.225.152:30000/metrics | grep -E "sglang:(spec_|hicache_|max_total)"
# 期望:
#   sglang:spec_accept_rate > 0.5
#   sglang:max_total_num_tokens > 900000
#   sglang:hicache_host_total_tokens > 1.8M
```

### 5.6 Helm Upgrade (更新配置)

```bash
helm upgrade sglang-glm52-2tp8 \
  docker/rocm-mi308x-glm52/chart/ -n kube-system \
  -f docker/rocm-mi308x-glm52/chart/values-glm52-2tp8.yaml

helm upgrade sglang-glm52-2tp8-w2 \
  docker/rocm-mi308x-glm52/chart/ -n kube-system \
  -f docker/rocm-mi308x-glm52/chart/values-glm52-2tp8-w2.yaml
```

**注意**: helm upgrade 后 router Deployment 可能需要手动 patch 删除 `nodeSelector` 的 `kubernetes.io/hostname`:
```bash
kubectl patch deploy -n kube-system sglang-glm52-2tp8-router --type=json \
  -p='[{"op":"remove","path":"/spec/template/spec/nodeSelector/kubernetes.io~1hostname"}]'
```

---

## 六、代码修改 (核心修复)

### 6.1 dsa_indexer.py — hoist fix (NameError 修复)

**文件**: `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`

**问题**: `scale_head_gate_graph` 和 `logits_head_gate_graph` 定义在 `if _is_cuda:` 块内。ROCm 上 `_is_cuda=False`,函数从未定义。

**修复**: 将两个函数 (及其 `_fake_impl` helpers) 提升到模块级:

```python
# ============================================================
# 修复后的代码 (dsa_indexer.py:181-246)
# ============================================================

# FIX: Hoist scale_head_gate_graph and logits_head_gate_graph out of the
# `if _is_cuda:` block so they are defined on ROCm/HIP too. These functions
# only use torch.mm and arithmetic — no CUDA-specific kernels — so they are
# platform-agnostic. Previously they were only defined under `if _is_cuda:`,
# which caused NameError on ROCm when the breakable prefill CUDA graph
# warmup exercised the in_piecewise_or_breakable_cuda_graph path.
def _scale_head_gate_graph_fake_impl(
    weights_raw: torch.Tensor,
    n_heads_inv_sqrt: float,
    softmax_scale: float,
    q_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (weights_raw.shape[0], weights_raw.shape[1], q_scale.shape[-1]),
        dtype=torch.float32,
        device=weights_raw.device,
    )


# In-graph (PCG/BCG) head gate for the fused path
@register_custom_op(fake_impl=_scale_head_gate_graph_fake_impl)
def scale_head_gate_graph(
    weights_raw: torch.Tensor,
    n_heads_inv_sqrt: float,
    softmax_scale: float,
    q_scale: torch.Tensor,
) -> torch.Tensor:
    weights = weights_raw * n_heads_inv_sqrt
    return weights.unsqueeze(-1) * q_scale * softmax_scale


def _logits_head_gate_graph_fake_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    n_heads_inv_sqrt: float,
    softmax_scale: float,
    q_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0], q_scale.shape[-1]),
        dtype=torch.float32,
        device=x.device,
    )


# In-graph (PCG/BCG) head gate for the NON-prefill path
@register_custom_op(fake_impl=_logits_head_gate_graph_fake_impl)
def logits_head_gate_graph(
    x: torch.Tensor,
    weight: torch.Tensor,
    n_heads_inv_sqrt: float,
    softmax_scale: float,
    q_scale: torch.Tensor,
) -> torch.Tensor:
    out = torch.mm(x, weight.t(), out_dtype=torch.float32)
    weights = out * n_heads_inv_sqrt
    weights = weights.unsqueeze(-1) * q_scale * softmax_scale
    return weights


# Only CUDA-specific stuff remains inside `if _is_cuda:` block
if _is_cuda:
    @register_custom_op(mutates_args=["topk_indices"])
    @register_split_op()
    def broadcast_indexer_topk_from_rank0_(topk_indices: torch.Tensor) -> None:
        _broadcast_indexer_topk_from_rank0_impl(topk_indices)
```

**调用点** (dsa_indexer.py:~1959, ~1968) 使用裸名引用这些函数:
```python
# 在 DSA forward 路径中
weights = scale_head_gate_graph(weights_raw, n_heads_inv_sqrt, softmax_scale, q_scale)
# 和
logits = logits_head_gate_graph(x, weight, n_heads_inv_sqrt, softmax_scale, q_scale)
```

### 6.2 eagle_utils.py — PR #31478 (TP broadcast)

**文件**: `python/sglang/srt/speculative/eagle_utils.py`

**问题**: EAGLE greedy 分支 (HIP/ROCm, CPU, NPU, XPU 使用) 缺少 TP broadcast,导致各 rank argmax 结果可能不同,seq_lens 发散,next TP collective deadlock。

**修复**: 在 greedy 分支添加 TP broadcast (eagle_utils.py:665-680):

```python
# ============================================================
# 修复后的代码 (eagle_utils.py:665-680)
# ============================================================

        # FIX: PR #31478 — broadcast greedy results across TP ranks to prevent
        # per-rank argmax divergence causing EAGLE verify deadlock on ROCm when
        # --enable-aiter-allreduce-fusion makes all-reduce non-deterministic.
        # The non-greedy branch below already broadcasts; the greedy branch
        # (taken on HIP/ROCm and CPU/NPU/XPU) was missing this sync. When TP
        # ranks pick different tokens via argmax, seq_lens diverge and the next
        # TP collective deadlocks. Broadcast from rank 0 for consistency.
        tp_group = (
            get_parallel().attn_tp_group
            if is_dp_attention_enabled()
            else get_tp_group()
        )
        if tp_group.world_size > 1:
            tp_group.broadcast(predict, src=0)
            tp_group.broadcast(accept_index, src=0)
            tp_group.broadcast(num_correct_drafts, src=0)
```

### 6.3 chart values 修复 (AssertionError + OOM)

**修复 1**: `cudaGraphBackendPrefill: breakable` → `tc_piecewise` (values.yaml:90)
- 根因: BCG split-op dispatch 用 `is_cuda()` 门控,ROCm 上不选中但 assertion 触发
- 上游 `default_prefill_backend()` 明确: BCG 仅 CUDA 默认

**修复 2**: `hicacheRatio: 4` → `2` (values-glm52-2tp8.yaml:55)
- 根因: DSA indexer 56.89 GB/rank + HiCache ratio=4 (248 GB/rank) × 8 = ~2.4 TB > 节点 RAM
- ratio=2 → 124 GB/rank + 57 GB/rank × 8 = ~1.4 TB,安全

**修复 3**: OOM 修复 (values-glm52-2tp8.yaml)
- `memFractionStatic`: 0.88 → 0.75 (激活值空间 2.5GB → 46GB)
- `prefillMaxRequests`: 32 → 4
- `chunkedPrefillSize`: 32768 → 16384
- `scheduleConservativeness`: 0.5 → 1.0

---

## 七、问题根因分析

### 7.1 问题一: NameError (启动崩溃)

**现象**:
```
NameError: name 'logits_head_gate_graph' is not defined
```

**根因**: `scale_head_gate_graph` 和 `logits_head_gate_graph` 定义在 `if _is_cuda:` 块内 (dsa_indexer.py:173)。ROCm 上 `_is_cuda=False`,函数从未定义。调用点 (line 1959, 1968) 使用裸名引用,触发 NameError。

**触发条件**: breakable prefill CUDA graph 捕获 warmup 阶段,第一次 EAGLE decode after >1024-token prefill。

**修复**: hoist 到模块级 (commit `a9bc24365b`)。

### 7.2 问题二: AssertionError (hoist 修复后)

**现象**:
```
AssertionError: Internal error: in-graph DSA prefill must go through the graph DSA split-op dispatch
```

**根因**: `is_graph_dsa_split_op_surface()` (dsa/utils.py:103) 用 `is_cuda()` 门控 split-op dispatch。ROCm 上 `is_cuda()` 返回 False,dispatch 从不被选中,但 dsa_indexer.py:2061 的 assertion 仍触发。

**上游设计**: `default_prefill_backend()` (cuda_graph_config.py:101) 明确:
> "BCG (breakable) is the prefill default on CUDA only; other platforms (HIP/NPU/...) keep tc_piecewise until BCG is validated there."

**修复**: `cudaGraphBackendPrefill: breakable` → `tc_piecewise` (commit `b4628cf86a`)。

### 7.3 问题三: Host Memory OOM (tc_piecewise 修复后)

**现象**: Rank 5 scheduler 被 SIGKILL (exit code -9)。

**根因**: 新上游 DSA indexer 分配 56.89 GB/rank host 内存 (page_first_direct layout,455 GB total)。加上 HiCache ratio=4 (248 GB/rank = 2 TB total),总计 ~2.4 TB > 节点 RAM (~2 TB)。

**修复**: `hicacheRatio` 4 → 2 (commit `50b9138541`)。
- ratio=2: HiCache 124 GB/rank + DSA 19 GB/rank × 8 = ~1.4 TB,安全

### 7.4 缺失的上游修复 (6 个 commit)

| Commit | 描述 |
|--------|------|
| `78dc581518` | Stabilize GLM-5.2 MTP IndexShare across PD and CUDA graph replay |
| `7a973c03a0` | Stamp capture-time num_tokens_per_req in multi-layer EAGLE |
| `fc1e3797b7` | Split capture width from num_tokens_per_req and gate replay |
| `cce5fe7696` | Move WAR barrier right after each run_batch launch |
| `942bf04ef9` | Add SGLANG_FORCE_COARSE_WAR_BARRIER opt-in |
| `7e229e2a81` | Support GLM-5.2 MTP index sharing with prefill CP |

### 7.5 本地 Patch (PR #31478)

EAGLE greedy 分支添加 TP broadcast,防止大 prefill + EAGLE overlap 调度时 collective deadlock。尚未合并上游。

---

## 八、验证结果 (2026-07-20)

### 8.1 部署状态

| 组件 | W1 (.152) | W2 (.172) |
|------|-----------|-----------|
| Helm release | sglang-glm52-2tp8 rev 31 | sglang-glm52-2tp8-w2 rev 27 |
| Image | fix-eagle-coredump-v3 | fix-eagle-coredump-v3 |
| Digest | sha256:416eb7f8... | sha256:416eb7f8... |
| Status | 1/1 Running, 0 restarts | 1/1 Running, 0 restarts |
| Prefill backend | tc_piecewise | tc_piecewise |
| HiCache ratio | 2 | 2 |

### 8.2 Benchmark (4 场景,100% 成功)

| 场景 | W1 (.152) | W2 (.172) |
|------|-----------|-----------|
| short_c32 (in=32, out=256, n=32, rate=8) | 153.00 tok/s | 146.31 tok/s |
| short_c128 (in=128, out=256, n=32, rate=8) | 242.40 tok/s | 251.87 tok/s |
| **mid_c2048** (in=2048, out=256, n=16, rate=4, **原 coredump 触发**) | 166.47 tok/s | 184.06 tok/s |
| long_c8192 (in=8192, out=256, n=8, rate=2) | 108.21 tok/s | 75.60 tok/s |

### 8.3 EAGLE 推测解码指标

| 指标 | W1 (.152) | W2 (.172) |
|------|-----------|-----------|
| sglang:spec_accept_rate | 0.60 | 0.858 |
| sglang:spec_accept_length | 2.80 | 3.575 |
| sglang:spec_verify_calls_total | 8060 | 11139 |
| sglang:max_total_num_tokens | 926080 | 926080 |
| sglang:hicache_host_total_tokens | 1.85M | 1.85M |

### 8.4 内存占用

- **GPU VRAM**: ~160 GB used / 192 GB total per GPU (mem-fraction=0.75)
- **Host RAM**: ~819 GB total (HiCache 83 GB/rank + DSA 19 GB/rank × 8)
- **Activation headroom**: ~46 GB/GPU (OOM 修复后 18 倍提升)

---

## 九、故障排查

### 9.1 NameError: logits_head_gate_graph is not defined

**原因**: 缺少 hoist 修复,运行旧镜像 (v1/v2)。
**修复**: 使用 `fix-eagle-coredump-v3`。

### 9.2 AssertionError: in-graph DSA prefill must go through split-op dispatch

**原因**: `cudaGraphBackendPrefill=breakable`。BCG 是 CUDA-only。
**修复**: 设为 `tc_piecewise`。

### 9.3 SIGKILL (exit code -9) during startup

**原因**: Host memory OOM (DSA 57 GB/rank + HiCache ratio=4 超过节点 RAM)。
**修复**: `hicacheRatio` 降到 2。

### 9.4 NCCL error: unhandled cuda error

**原因**: 之前崩溃导致 GPU 状态损坏。
**修复**: `kubectl delete pod --grace-period=0 --force`,持续则 reboot 节点。

### 9.5 hipIpcGetMemHandle failed: invalid argument

**原因**: 同 NCCL error — GPU 状态损坏。
**修复**: 删除 pod,持续则 reboot 节点。

### 9.6 Router 503 / Connection refused

**修复**:
1. 检查 router pod: `kubectl get pod -n kube-system -l app=sglang-router`
2. helm upgrade 后 patch router nodeSelector:
   ```bash
   kubectl patch deploy -n kube-system <router-deploy> --type=json \
     -p='[{"op":"remove","path":"/spec/template/spec/nodeSelector/kubernetes.io~1hostname"}]'
   ```

### 9.7 Health probe timeout

**原因**: `/health` 运行 prefill 64 tokens,GPU 唤醒延迟 10-30s。
**修复**: `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false` (v3 已设)。

---

## 十、Git 历史

### 10.1 当前分支 (sglang fork, squashed)

```
62d98600dd  docs(rocm-mi308x-glm52): add complete reproduction guide and update README
39145f548d  fix(eagle): comprehensive EAGLE decode coredump fix for MI308X GLM-5.2
50c118704a  (origin/main, main) [diffusion] disagg: handle numpy arrays ...
```

### 10.2 原 5 个 commit (sglang-offical-github)

| Commit | 描述 |
|--------|------|
| `250019ef99` | 综合 EAGLE coredump 修复 (PR #31478 + docker/chart) |
| `cb91a13ef7` | fix(dsa): use self. prefix (错误,被覆盖) |
| `a9bc24365b` | fix(dsa): hoist scale/logits_head_gate_graph |
| `b4628cf86a` | fix(chart): tc_piecewise prefill backend on ROCm |
| `50b9138541` | fix(chart): reduce hicacheRatio 4 to 2 |

---

## 十一、参考链接

- **仓库**: [tanguofu/sglang](https://github.com/tanguofu/sglang) 分支 `fix/eagle-decode-coredump-mi308x`
- **上游**: [sgl-project/sglang](https://github.com/sgl-project/sglang)
- **基础镜像**: `lmsysorg/sglang-rocm:v0.5.15.post1-rocm720-mi30x-20260718`
- **PR #31478**: TP broadcast in EAGLE greedy branch (未合并上游)
- **集群**: TKE `cls-bmmk3vtl` (GZ test), namespace `kube-system`
- **Gateway**: `glm52-2tp8.jmpti.woa.com` → envoy LB → router → workers
- **iWiki**: https://iwiki.woa.com/p/4026586166

---

## 十二、联系

- **维护者**: guofutan (谭国富)
- **节点**: node-21.151.225.152 (W1), node-21.151.225.172 (W2)
- **最后更新**: 2026-07-20
