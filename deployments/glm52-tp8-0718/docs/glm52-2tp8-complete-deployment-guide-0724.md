# GLM-5.2 2tp8 完整部署方案 — MI308X gfx942

> **文档版本**: 2026-07-24 (Final, 生产环境基线)
> **分支**: `sglang-2tp8-0723` (github.com/tanguofu/sglang.git)
> **iWiki docid**: [4027591453](https://iwiki.woa.com/p/4027591453)
> **关联文档**: [4027539965](https://iwiki.woa.com/p/4027539965) (前序部署记录)

---

## 目录

1. [架构概览](#1-架构概览)
2. [硬件与驱动环境](#2-硬件与驱动环境)
3. [Docker 镜像构建](#3-docker-镜像构建)
4. [SGLang 启动命令与参数](#4-sglang-启动命令与参数)
5. [环境变量完整清单](#5-环境变量完整清单)
6. [Helm Chart 配置](#6-helm-chart-配置)
7. [Rust Router 原生路由](#7-rust-router-原生路由)
8. [Hack Patch 代码](#8-hack-patch-代码)
9. [ConfigMap 覆盖机制](#9-configmap-覆盖机制)
10. [Kubernetes 部署拓扑](#10-kubernetes-部署拓扑)
11. [Benchmark 性能数据](#11-benchmark-性能数据)
12. [故障排查指南](#12-故障排查指南)
13. [Git Commit 历史](#13-git-commit-历史)

---

## 1. 架构概览

```
                    ┌─────────────────────────────┐
                    │   Envoy Gateway (eg-tke)    │
                    │   glm52-2tp8.jmpti.woa.com  │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │  sgl-model-gateway (Rust)   │
                    │  Router (cache_aware)       │
                    │  node-21.151.225.144:30080  │
                    └──┬───────────────────────┬──┘
                       │                       │
            ┌──────────▼─────────┐  ┌──────────▼─────────┐
            │  STS pod-0         │  │  STS pod-1         │
            │  sglang (TP=8)     │  │  sglang (TP=8)     │
            │  node-.172:30000   │  │  node-.152:30000   │
            │  8× MI308X gfx942  │  │  8× MI308X gfx942  │
            └────────────────────┘  └────────────────────┘
```

**核心特性**:
- **2 个 worker pod**, 每个 pod 占据 1 个节点 (8 GPU), TP=8
- **EAGLE 投机解码** (NEXTN), 2.4-2.85x 接受长度
- **FP8 KV Cache** + **HiCache** (GPU+Host 两级缓存)
- **DSA 注意力后端** + **TileLang** prefill/decode
- **AiterCustomAllreduce** + **AITER AllReduce Fusion**
- **FlyDSL gfx942 fp8 MQA logits** 优化内核
- **Rust 原生路由** (消除 Python proxy), 支持 /v1/responses + /v1/messages + /v1/chat/completions

---

## 2. 硬件与驱动环境

### 2.1 GPU

| 属性 | 值 |
|---|---|
| GPU 型号 | AMD Instinct MI308X OAM |
| 架构 | gfx942 (CDNA 3) |
| 单卡显存 | 192 GB HBM3 |
| GPU 数量 | 8 卡/节点 |
| 互联 | XGMI (8 GPU 单 hive, P2P/IPC, 123 GB/s all-reduce) |

### 2.2 节点配置

| 节点 | IP | OS | Kernel | 角色 |
|---|---|---|---|---|
| node-21.151.225.144 | 21.151.225.144 | TencentOS 3.1 | 5.4.119-19.0009.60 | Router |
| node-21.151.225.152 | 21.151.225.152 | TencentOS 3.1 | 5.4.119-19.0009.60 | Worker (pod-1) |
| node-21.151.225.172 | 21.151.225.172 | TencentOS 3.1 | 5.4.119-19.0009.60 | Worker (pod-0) |

**节点资源**: 384 CPU, 2.2 TiB 内存, 8 GPU

### 2.3 容器内软件栈

| 组件 | 版本 |
|---|---|
| sglang | `0.5.15.post1.dev20260718+gd7b9425529` |
| PyTorch | `2.9.1+rocm7.2.0.git7e1940d4` |
| ROCm | `7.2.0` |
| HIP runtime | `7.2.26015-fc0010cf6a` |
| Triton | `3.6.0+git42270451` |
| Python | `3.10` |
| amdgpu-dkms | `6.16.13.30300000` |
| Ubuntu | `22.04.5 LTS (Jammy)` |

### 2.4 基础镜像

```
lmsysorg/sglang-rocm:v0.5.14-rocm720-mi30x-20260708
```

---

## 3. Docker 镜像构建

### 3.1 Dockerfile

**路径**: `docker/rocm-mi308x-glm52/Dockerfile`
**当前 tag**: `fix-eagle-coredump-v3`

```dockerfile
# GLM-5.2-FP8 SGLang Worker for AMD MI308X (gfx942)
FROM lmsysorg/sglang-rocm:v0.5.14-rocm720-mi30x-20260708

LABEL maintainer="guofutan"
LABEL description="GLM-5.2-FP8 SGLang worker for AMD MI308X (gfx942) with 0708-opt patches"
LABEL sglang-base-image="v0.5.14-rocm720-mi30x-20260708"
LABEL gpu-arch="gfx942"
LABEL branch="308x-glm52-opt"

# ============================================================
# Step 1: 替换 sglang 源码为预补丁版本 (0708-opt 分支)
# ============================================================
COPY python/sglang/ /sgl-workspace/sglang/python/sglang/

# 验证 16 个关键补丁已就位 (构建时失败即中止)
RUN python3 -c "\
src = open('/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py').read(); \
assert '_is_cuda or _is_hip' in src, 'PATCH MISSING: 1.1 JIT imports on HIP'; \
assert '(_is_cuda or _is_hip) and not envs.SGLANG_DISABLE_DSA_INDEXER_FUSION' in src, 'PATCH MISSING: 1.2 DSA indexer fusion on HIP'; \
assert '_k_norm_weight_f32' in src, 'PATCH MISSING: 1.6 k_norm f32 properties'; \
assert 'FIX(breakable-target-verify)' in src, 'PATCH MISSING: 04 target_verify metadata guard'; \
assert 'DUAL_STREAM_TOKEN_THRESHOLD = 1024' in src, 'PATCH MISSING: 06a S1 dual stream threshold'; \
src2 = open('/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py').read(); \
assert '(_is_cuda or _is_hip) and envs.SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM' in src2, 'PATCH MISSING: 2.1 dual stream on HIP'; \
src3 = open('/sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py').read(); \
assert 'is_hip() or envs.SGLANG_NPU_USE_MULTI_STREAM' in src3, 'PATCH MISSING: 3.1 alt_stream on HIP'; \
src4 = open('/sgl-workspace/sglang/python/sglang/srt/models/transformers.py').read(); \
assert 'mla_kv_a_proj' in src4, 'PATCH MISSING: 4.1 TP style'; \
src5 = open('/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py').read(); \
assert 'q.reshape(-1, layer.tp_q_head_num' in src5, 'PATCH MISSING: 5c view->reshape'; \
assert 'elif forward_batch.seq_lens_sum is not None' in src5, 'PATCH MISSING: 06b D2H sync elimination'; \
src6 = open('/sgl-workspace/sglang/python/sglang/srt/layers/radix_attention.py').read(); \
assert '# if _is_hip and not save_kv_cache' in src6, 'PATCH MISSING: 5d _pcg_mha_companion disabled'; \
src7 = open('/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8.py').read(); \
assert 'is_shuffled = True' in src7, 'PATCH MISSING: 02 fp8 is_shuffled'; \
src8 = open('/sgl-workspace/sglang/python/sglang/jit_kernel/dsv4/elementwise.py').read(); \
assert 'q_fp8_raw' in src8, 'PATCH MISSING: 03 FP8 view fix'; \
src9 = open('/sgl-workspace/sglang/python/sglang/jit_kernel/csrc/dsa/fused_store_index_cache.cuh').read(); \
assert 'USE_ROCM' in src9, 'PATCH MISSING: 05 fused_store ROCm FP8'; \
print('All 16 patches verified + 308X workarounds reverted OK')"

# ============================================================
# Step 1b: 添加 FlyDSL gfx942 fp8 MQA logits 内核
# MI308X 64KB 共享内存限制 vs MI355X 80KB+
# ============================================================
COPY docker/rocm-mi308x-glm52/patches/flydsl/fp8_mqa_logits.py /sgl-workspace/aiter/aiter/ops/flydsl/kernels/fp8_mqa_logits.py
COPY docker/rocm-mi308x-glm52/patches/flydsl/__init__.py /sgl-workspace/aiter/aiter/ops/flydsl/__init__.py
COPY docker/rocm-mi308x-glm52/patches/fp8_mqa_logits.py /sgl-workspace/aiter/aiter/ops/triton/attention/fp8_mqa_logits.py

# ============================================================
# Step 2: 环境变量 (21 个, 基于 MI355X 生产配置)
# ============================================================
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV NCCL_DEBUG=INFO
ENV HSA_ENABLE_SDMA=0
ENV HIP_FORCE_DEV_KERNARG=1
ENV HSA_NO_SCRATCH_RECLAIM=1
ENV NCCL_CUMEM_ENABLE=0
ENV NCCL_MIN_NCHANNELS=112
ENV NCCL_NVLS_ENABLE=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PYTORCH_ROCM_ARCH=gfx942
ENV ROCM_QUICK_REDUCE_QUANTIZATION=INT8
ENV SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
ENV SGLANG_DISABLE_CUDNN_CHECK=1
ENV SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM=1
ENV SGLANG_INT4_WEIGHT=0
ENV SGLANG_MOE_PADDING=1
ENV SGLANG_ROCM_DISABLE_LINEARQUANT=0
ENV SGLANG_ROCM_FUSED_DECODE_MLA=1
ENV SGLANG_SET_CPU_AFFINITY=1
ENV SGLANG_USE_AITER=1
ENV SGLANG_USE_ROCM700A=1

ENV MODEL_PATH=/data/model/glm52-fp8
ENV PORT=30000
ENV API_KEY=sk-46faecc9d0bc4dcd9db6a15c73ae91c8

COPY docker/rocm-mi308x-glm52/start_server.sh /start_server.sh
RUN chmod +x /start_server.sh

ENTRYPOINT ["/start_server.sh"]
```

### 3.2 16 个构建时验证的补丁

| # | 补丁 | 文件 | 作用 |
|---|---|---|---|
| 1.1 | JIT imports on HIP | dsa_indexer.py | HIP 平台 JIT 导入修复 |
| 1.2 | DSA indexer fusion on HIP | dsa_indexer.py | DSA 索引器融合 |
| 1.6 | k_norm f32 properties | dsa_indexer.py | k_norm 权重 f32 属性 |
| 04 | target_verify metadata guard | dsa_indexer.py | target_verify 元数据保护 |
| 06a | S1 dual stream threshold | dsa_indexer.py | 双流 token 阈值 1024 |
| 2.1 | dual stream on HIP | deepseek_v2.py | HIP 双流启用 |
| 3.1 | alt_stream on HIP | deepseek_nextn.py | HIP 备用流 |
| 4.1 | TP style | transformers.py | MLA TP 风格 |
| 5c | view->reshape | dsa_backend.py | view 改 reshape |
| 06b | D2H sync elimination | dsa_backend.py | D2H 同步消除 |
| 5d | _pcg_mha_companion disabled | radix_attention.py | 禁用 _pcg_mha_companion |
| 02 | fp8 is_shuffled | fp8.py | FP8 shuffle 模式 |
| 03 | FP8 view fix | elementwise.py | FP8 view 修复 |
| 05 | fused_store ROCm FP8 | fused_store_index_cache.cuh | ROCm FP8 融合存储 |

### 3.3 构建命令

```bash
cd /path/to/sglang-worktree-2tp8-0723

docker build -f docker/rocm-mi308x-glm52/Dockerfile \
  -t mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3 .

docker push mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3
```

---

## 4. SGLang 启动命令与参数

### 4.1 完整启动命令

```bash
exec python3 -m sglang.launch_server \
  --model-path /data/model/glm52-fp8 \
  --model-impl sglang \
  --served-model-name glm-5.2 \
  --api-key "$API_KEY" \
  --tp-size 8 --pp-size 1 --trust-remote-code \
  --host 0.0.0.0 --port 30000 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --context-length 524288 \
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
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-io-backend direct \
  --hicache-mem-layout page_first_direct \
  --hicache-write-policy write_through_selective \
  --enable-cache-report \
  --enable-metrics --skip-server-warmup \
  --watchdog-timeout 3600 --log-level info
```

### 4.2 参数详解

| 参数 | 值 | 说明 |
|---|---|---|
| `--model-path` | `/data/model/glm52-fp8` | FP8 量化模型路径 |
| `--model-impl` | `sglang` | SGLang 原生实现 |
| `--served-model-name` | `glm-5.2` | API 模型名 |
| `--tp-size` | `8` | 8 卡张量并行 |
| `--pp-size` | `1` | 无流水线并行 |
| `--numa-node` | `0 0 0 0 1 1 1 1` | NUMA 绑定 (TP0-3→NUMA0, TP4-7→NUMA1) |
| `--context-length` | `524288` | 512K 上下文 |
| `--tool-call-parser` | `glm47` | GLM-4.7 工具调用格式 |
| `--reasoning-parser` | `glm45` | GLM-4.5 推理格式 |
| `--mem-fraction-static` | `0.88` | 88% GPU 内存静态分配 |
| `--cuda-graph-bs-decode` | `1 2 3 4 5 6 7 8 9 10 12 16` | Decode CUDA Graph 批次 |
| `--cuda-graph-max-bs-decode` | `16` | Decode CG 最大批次 |
| `--enable-aiter-allreduce-fusion` | — | AITER AllReduce 融合 |
| `--enable-mixed-chunk` | — | 混合 prefill+decode 批处理 |
| `--chunked-prefill-size` | `32768` | 长序列分块 prefill |
| `--enable-fused-qk-norm-rope` | — | 融合 QK-norm + RoPE |
| `--schedule-conservativeness` | `0.5` | 激进调度 |
| `--prefill-max-requests` | `32` | 最大并发 prefill 请求 |
| `--max-prefill-tokens` | `32768` | 单次 prefill 最大 token |
| `--kv-cache-dtype` | `fp8_e4m3` | FP8 KV 缓存 |
| `--speculative-algorithm` | `NEXTN` | EAGLE 投机解码 |
| `--speculative-num-steps` | `3` | 投机步数 |
| `--speculative-num-draft-tokens` | `4` | 每步草稿 token 数 |
| `--speculative-eagle-topk` | `1` | EAGLE top-k (DSA 限制, 不能更高) |
| `--cuda-graph-backend-prefill` | `tc_piecewise` | DSA 索引器崩溃修复 |
| `--max-running-requests` | `32` | 最大并发运行请求 |
| `--cuda-graph-bs-prefill` | `4 8 16 32` | Prefill CUDA Graph 批次 |
| `--enable-hierarchical-cache` | — | 两级 KV 缓存 (GPU+Host) |
| `--hicache-ratio` | `2` | Host 缓存 = 2x GPU 缓存 |
| `--hicache-io-backend` | `direct` | Direct I/O |
| `--hicache-mem-layout` | `page_first_direct` | 页优先内存布局 |
| `--hicache-write-policy` | `write_through_selective` | 选择性写穿透 |
| `--enable-cache-report` | — | 返回 cached_tokens 给 router (cache_aware 路由依据) |
| `--enable-metrics` | — | Prometheus 指标 |
| `--skip-server-warmup` | — | 跳过预热 |
| `--watchdog-timeout` | `3600` | 1 小时看门狗 |
| `--log-level` | `info` | Info 日志级别 |

> **注意**: `page_size=64` (SGLang 默认, 未显式配置)。Radix cache 按 64 token 对齐匹配前缀, prompt < 64 token 不会命中 cache。Codex/agent 场景的 system prompt 通常 >> 64 token, cache 正常工作。

---

## 5. 环境变量完整清单

### 5.1 Dockerfile 内置 (21 个)

| # | 环境变量 | 值 | 说明 |
|---|---|---|---|
| 1 | `HIP_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | 暴露所有 8 GPU |
| 2 | `NCCL_DEBUG` | `INFO` (chart 覆盖为 `WARN`) | NCCL 日志级别 |
| 3 | `HSA_ENABLE_SDMA` | `0` | 禁用 SDMA (P2P/IPC 走 XGMI) |
| 4 | `HIP_FORCE_DEV_KERNARG` | `1` | 强制设备内核参数 |
| 5 | `HSA_NO_SCRATCH_RECLAIM` | `1` | 禁用 scratch 内存回收 |
| 6 | `NCCL_CUMEM_ENABLE` | `0` | 禁用 NCCL CUMEM |
| 7 | `NCCL_MIN_NCHANNELS` | `112` (chart 覆盖为 `80`) | RCCL 通道数 |
| 8 | `NCCL_NVLS_ENABLE` | `0` | 禁用 NVLink SHARP |
| 9 | `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | PyTorch 内存分配器 |
| 10 | `PYTORCH_ROCM_ARCH` | `gfx942` | ROCm GPU 架构 |
| 11 | `ROCM_QUICK_REDUCE_QUANTIZATION` | `INT8` (chart 覆盖为 `NONE`) | 零精度损失 |
| 12 | `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | `1` | 允许超过模型最大上下文 |
| 13 | `SGLANG_DISABLE_CUDNN_CHECK` | `1` | 跳过 cuDNN 检查 |
| 14 | `SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM` | `1` | DSv2 双流 |
| 15 | `SGLANG_INT4_WEIGHT` | `0` | 禁用 INT4 权重 |
| 16 | `SGLANG_MOE_PADDING` | `1` | MoE 张量填充 |
| 17 | `SGLANG_ROCM_DISABLE_LINEARQUANT` | `0` | 启用线性量化 |
| 18 | `SGLANG_ROCM_FUSED_DECODE_MLA` | `1` | 融合 decode MLA |
| 19 | `SGLANG_SET_CPU_AFFINITY` | `1` | 设置 CPU 亲和性 |
| 20 | `SGLANG_USE_AITER` | `1` | 使用 AITER 内核 |
| 21 | `SGLANG_USE_ROCM700A` | `1` | 使用 ROCm 7.0.0A 代码路径 |

### 5.2 Chart 追加 (8 个)

| # | 环境变量 | 值 | 说明 |
|---|---|---|---|
| 22 | `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` | `false` | /health 直接返回 200 (不走 prefill) |
| 23 | `SGLANG_OPT_USE_AITER_INDEXER` | `1` | 使用 AITER 索引器 (绕过 topk_v2 JIT) |
| 24 | `FLYDSL_FP8_MQA_LOGITS_VARIANT` | `mfma_r4_w4` | FlyDSL 内核变体 (4行/块, 4波/块) |
| 25 | `CUDA_ENABLE_USER_TRIGGERED_COREDUMP` | `1` | 捕获 GPU coredump |
| 26 | `AITER_CONFIG_GEMM_BF16` | `/etc/aiter-configs/bf16_tuned_gemm.csv` | 调优 GEMM 配置 (ConfigMap) |
| 27 | `MODEL_PATH` | `/data/model/glm52-fp8` | 模型路径 |
| 28 | `PORT` | `30000` | 监听端口 |
| 29 | `API_KEY` | `sk-****` | API 认证密钥 |

### 5.3 Chart 覆盖 Dockerfile 的变量

| 变量 | Dockerfile | Chart | 原因 |
|---|---|---|---|
| `NCCL_DEBUG` | `INFO` | `WARN` | 减少 NCCL 日志刷屏 |
| `NCCL_MIN_NCHANNELS` | `112` | `80` | RCCL 2.27.7 硬上限 |
| `ROCM_QUICK_REDUCE_QUANTIZATION` | `INT8` | `NONE` | 零精度损失 |

---

## 6. Helm Chart 配置

### 6.1 values-glm52-2tp8-merged.yaml

**路径**: `docker/rocm-mi308x-glm52/chart/values-glm52-2tp8-merged.yaml`

```yaml
image: mirrors.tencent.com/ti-platform/sglang-glm52-308x
tag: fix-eagle-coredump-v3
imagePullSecret: ""
imagePullSecrets:
  - name: tencent-registry
  - name: tencent-mirror-secret

namespace: kube-system
replicas: 2
hostNetwork: true
port: 30000

podAnnotations:
  eks.tke.cloud.tencent.com/cluster-ip-switch: cluster

nodeName: ""
nodeSelector:
  accelerator: amd-gpu
  sglang-model: ready
tolerations:
  - key: dedicated
    operator: Equal
    value: sglang-2tp8
    effect: NoSchedule

sglang:
  tpSize: 8
  ppSize: 1
  contextLength: "524288"
  numaNode: "0 0 0 0 1 1 1 1"
  memFractionStatic: 0.88
  kvCacheDtype: fp8_e4m3
  chunkedPrefillSize: 32768
  prefillMaxRequests: 32
  maxRunningRequests: 32
  maxPrefillTokens: 32768
  scheduleConservativeness: 0.5
  cudaGraphBsDecode: "1 2 3 4 5 6 7 8 9 10 12 16"
  cudaGraphMaxBsDecode: 16
  cudaGraphBsPrefill: "4 8 16 32"
  cudaGraphBackendPrefill: tc_piecewise
  speculativeAlgorithm: NEXTN
  speculativeNumSteps: 3
  speculativeNumDraftTokens: 4
  speculativeEagleTopk: 1
  enableHierarchicalCache: true
  hicacheRatio: 2.0
  hicacheIoBackend: direct
  hicacheMemLayout: page_first_direct
  hicacheWritePolicy: write_through_selective
  enableAiterAllreduceFusion: true
  enableMixedChunk: true
  enableFusedQkNormRope: true
  enableMetrics: true
  enableCacheReport: true             # 返回 cached_tokens 给 router, cache_aware 路由依据
  skipServerWarmup: true
  watchdogTimeout: 3600
  logLevel: info
  toolCallParser: glm47
  reasoningParser: glm45

rocmQuickReduceQuantization: "NONE"

router:
  enabled: true
  port: 30080
  policy: cache_aware
  workerUrls:
    - "http://21.151.225.152:30000"
    - "http://21.151.225.172:30000"
  cacheThreshold: 0.2
  balanceAbsThreshold: 1
  balanceRelThreshold: 1.2
  image: mirrors.tencent.com/ti-platform/sglang-glm52-308x-pd-router
  tag: pd-resp-msg-v1
  nodeName: node-21.151.225.144
  tolerations:
    - key: dedicated
      operator: Equal
      value: sglang-1pd-A-group
      effect: NoSchedule

gateway:
  enabled: true
  gatewayName: eg-tke
  gatewayNamespace: ti-cloud
  hostname: glm52-2tp8.jmpti.woa.com
  lbAddress: 21.162.215.14
  httpsSectionName: https-glm52-2tp8

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
  initialDelaySeconds: 2400
  periodSeconds: 60
  timeoutSeconds: 10
  failureThreshold: 5

eaglePatch:
  enabled: false

aitersTunedGemm:
  enabled: true
  configMapName: aiters-tuned-gemm
  # volume.optional: true (防止 ConfigMap 删除后 pod 卡在 ContainerCreating)
```

### 6.2 资源配置 (values.yaml 默认)

```yaml
resources:
  requests:
    cpu: 360
    memory: 2100Gi
    amd.com/gpu: 8
  limits:
    amd.com/gpu: 8

shmSize: 32Gi
```

### 6.3 部署命令

```bash
helm install sglang-glm52-2tp8 \
  docker/rocm-mi308x-glm52/chart/ \
  -f docker/rocm-mi308x-glm52/chart/values-glm52-2tp8-merged.yaml \
  -n kube-system
```

---

## 7. Rust Router 原生路由

### 7.1 架构

Router 使用 **sgl-model-gateway** (Rust), 通过 patched wheel 部署, 原生支持:
- `/v1/chat/completions` (OpenAI)
- `/v1/responses` (OpenAI Responses API)
- `/v1/messages` (Anthropic Messages API)
- `/health` 健康检查

**消除了 Python proxy 层**, 减少延迟和维护复杂度。

### 7.2 Entrypoint 脚本

**ConfigMap**: `sglang-glm52-2tp8-native-entrypoint`

```bash
#!/bin/bash
set -euo pipefail

WHEEL_CACHE="/wheel-cache/sglang_router-0.3.2-cp38-abi3-manylinux_2_34_x86_64.whl"

if [ -f "${WHEEL_CACHE}" ]; then
  echo "=== Installing patched router wheel ==="
  pip install --force-reinstall --no-deps "${WHEEL_CACHE}" 2>&1 | tail -3
  echo "=== Wheel installed successfully ==="
else
  echo "=== WARNING: Patched wheel not found, using base image router ==="
fi

echo "=== Starting SGLang Router (native, no proxy) ==="
exec python3 -m sglang_router.launch_router "$@"
```

### 7.3 Router 启动参数

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://21.151.225.152:30000 http://21.151.225.172:30000 \
  --policy cache_aware \
  --host 0.0.0.0 \
  --port 30080 \
  --cache-threshold 0.2 \
  --balance-abs-threshold 1 \
  --balance-rel-threshold 1.2
```

### 7.4 Router 调度策略

| 参数 | 值 | 说明 |
|---|---|---|
| `--policy` | `cache_aware` | 基于 prefix cache 的路由 |
| `--cache-threshold` | `0.2` | 20% prefix 匹配即路由到缓存节点 |
| `--balance-abs-threshold` | `1` | 绝对负载差 ≥1 触发重平衡 |
| `--balance-rel-threshold` | `1.2` | 相对负载比 ≥1.2 触发重平衡 |

### 7.5 Patched Wheel 修复内容

**文件**: `sglang_router-0.3.2-cp38-abi3-manylinux_2_34_x86_64.whl`

修复 4 个问题:
1. `unwrap_namespace_tools` — 解包 namespace 工具类型
2. `ensure_stream_default` — 默认 stream=false (修复 BUG 1)
3. `build_messages_routing_text` — /v1/messages 路由文本构建
4. `tool_choice sanitize` — 清理 tool_choice 参数

---

## 8. Hack Patch 代码

### 8.1 /v1/responses API 4 个 BUG 修复

**文件**: `python/sglang/srt/entrypoints/openai/serving_responses.py`

#### BUG 1: stream:null 导致 400 (第 189-192 行)

```python
# Fix BUG 1: Router may forward stream:null when the client omits the
# stream field. ChatCompletionRequest requires stream as bool.
if request.stream is None:
    request.stream = False
```

#### BUG 2: reasoning_tokens 始终为 0 (第 294-297, 377 行)

```python
# Fix BUG 2: Pass require_reasoning so the scheduler counts
# reasoning tokens. Without this, reasoning_tokens is always 0
# in the Responses API (the scheduler skips the counting path).
require_reasoning = self._is_thinking_enabled_for_request(request)
```

在 `GenerateReqInput(...)` 中添加:
```python
require_reasoning=require_reasoning,
```

#### BUG 3: 非流式 usage 格式错误 (第 634-651 行)

```python
# Convert usage from Chat Completions format (UsageInfo) to Responses
# API format (input_tokens/output_tokens/output_tokens_details)
response_dict = response.model_dump()
if response_dict.get("usage"):
    u = response_dict["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    response_dict["usage"] = {
        "input_tokens": u.get("prompt_tokens", 0),
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": u.get("completion_tokens", 0),
        "output_tokens_details": {
            "reasoning_tokens": u.get("reasoning_tokens", 0)
        },
        "total_tokens": u.get("total_tokens", 0),
    }
return ORJSONResponse(content=response_dict)
```

#### BUG 4: 工具调用 done 事件 call_id/name 错误 (第 2031-2066, 2236-2284 行)

存储 `tc_id`/`tc_name` 到 tool_call_states, 在 done 事件中使用:

```python
def _close_tool_call_state(tool_index: int):
    state = tool_call_states.get(tool_index)
    if state is None or state.get("done"):
        return []
    arguments = state["arguments"]
    completed_item = ResponseFunctionToolCall(
        arguments=arguments,
        call_id=state["call_id"],      # 使用存储的 call_id
        name=state["name"] or "",      # 使用存储的 name
        type="function_call",
        id=state["item_id"],
        status="completed",
    )
    # ... emit done events
```

### 8.2 context.py reasoning_tokens 累加

**文件**: `python/sglang/srt/entrypoints/context.py` (第 100-101 行)

```python
if "reasoning_tokens" in meta_info:
    self.num_reasoning_tokens += meta_info["reasoning_tokens"]
```

### 8.3 EAGLE PR #31478 (已烘焙到镜像)

**修复**: EAGLE verify 结果在 TP ranks 间广播, 防止 per-rank argmax 分歧导致 decode coredump。

**文件**: `python/sglang/srt/layers/speculative/eagle_utils.py`

### 8.4 FlyDSL gfx942 fp8 MQA logits 内核

**路径**: `docker/rocm-mi308x-glm52/patches/flydsl/fp8_mqa_logits.py`

MI308X 64KB 共享内存限制下, 使用 `mfma_r4_w4` 变体 (4行/块, 4波/块), 比 `mfma_r2_w4` 快 14-29%。

---

## 9. ConfigMap 覆盖机制

### 9.1 Responses API 修复 ConfigMap

**ConfigMap**: `sglang-glm52-2tp8-responses-fix`

| Key | 挂载路径 | 大小 |
|---|---|---|
| `serving_responses.py` | `/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py` | 103,925 字符 |
| `context.py` | `/sgl-workspace/sglang/python/sglang/srt/entrypoints/context.py` | 8,426 字符 |

通过 STS `volumeMounts` + `subPath` 覆盖镜像内文件, 无需重建镜像。

### 9.2 Aiter 调优 GEMM ConfigMap

**ConfigMap**: `aiters-tuned-gemm`

| Key | 挂载路径 | 说明 |
|---|---|---|
| `bf16_tuned_gemm.csv` | `/etc/aiter-configs/bf16_tuned_gemm.csv` | 2385 行, 16 个 gfx942 K=6144 N=256 调优 GEMM |

环境变量 `AITER_CONFIG_GEMM_BF16` 指向此文件。

### 9.3 Router Entrypoint ConfigMap

**ConfigMap**: `sglang-glm52-2tp8-native-entrypoint`

| Key | 挂载路径 | 说明 |
|---|---|---|
| `entrypoint-native.sh` | `/opt/entrypoint-native.sh` | Router 启动脚本 |

### 9.4 EAGLE Patch ConfigMap (遗留, 未执行)

**ConfigMap**: `sglang-glm52-2tp8-eagle-patch`

虽然 `eaglePatch.enabled=false`, ConfigMap 仍挂载但脚本不执行。PR #31478 已烘焙到镜像。

---

## 10. Kubernetes 部署拓扑

### 10.1 Pod 放置

| Pod | 节点 | IP | 角色 |
|---|---|---|---|
| `sglang-glm52-2tp8-sglang-0` | node-21.151.225.172 | 21.151.225.172 | Worker (TP=8) |
| `sglang-glm52-2tp8-sglang-1` | node-21.151.225.152 | 21.151.225.152 | Worker (TP=8) |
| `sglang-glm52-2tp8-router-676f895454-q2gv8` | node-21.151.225.144 | 21.151.225.144 | Router |

### 10.2 Services

| Service | ClusterIP | Port | 类型 |
|---|---|---|---|
| `sglang-glm52-2tp8-router` | 9.165.99.83 | 30080/TCP | ClusterIP |
| `sglang-glm52-2tp8-sglang` | 9.165.48.178 | 30000/TCP | ClusterIP |
| `sglang-glm52-2tp8-sglang-headless` | None | 30000/TCP | Headless |

### 10.3 HTTPRoute

| 主机名 | Gateway |
|---|---|
| `glm52-2tp8.jmpti.woa.com` | `eg-tke` (ti-cloud namespace) |

### 10.4 Taints/Tolerations

```
Worker nodes (.152, .172):
  taint: dedicated=sglang-2tp8:NoSchedule

Router node (.144):
  taint: dedicated=sglang-1pd-A-group:NoSchedule
```

### 10.5 关键 STS 配置

```yaml
spec:
  podManagementPolicy: Parallel
  replicas: 2
  terminationGracePeriodSeconds: 300
  template:
    spec:
      hostNetwork: true
      dnsPolicy: ClusterFirstWithHostNet
      securityContext:
        privileged: true
        capabilities:
          add: [SYS_PTRACE]
          drop: ["ALL"]
        seccompProfile:
          type: Unconfined
```

---

## 11. Benchmark 性能数据

### 11.1 环境 2tp8 vs 1tp8 对比 (2026-07-24)

| 部署 | Wall time | Comp 吞吐 | Total 吞吐 | 加速比 |
|---|---|---|---|---|
| **1tp8** (单 pod) | 12.27s | 162.9 tok/s | 193.9 tok/s | 1.0x (基准) |
| **2tp8** (双 pod) | 5.10s | **392.4 tok/s** | **466.9 tok/s** | **2.41x** |

### 11.2 Warm Benchmark 完整场景 (2026-07-24, 生产配置)

预热: 10 sequential + 10 concurrent 请求 (JIT 编译 + CUDA Graph 捕获完成)。

| 场景 | 并发 | 请求 | maxT | Wall(s) | Comp tok/s | Total tok/s | p50(s) | p90(s) | EAGLE accept | Cache hit | 负载均衡 |
|------|------|------|------|---------|-----------|------------|--------|--------|-------------|-----------|---------|
| S1 短输出 | 20 | 20 | 100 | 2.44 | **819.0** | 1015.6 | 2.25 | 2.34 | 77.8% / 3.33 | 0% | 9/11 (1.22x) |
| S2 长输出 | 10 | 10 | 500 | 9.52 | 525.2 | 568.3 | 9.31 | 9.52 | 64.0% / 2.92 | 0% | 4/6 (1.50x) |
| S3 长输入 | 10 | 10 | 200 | 3.73 | 536.2 | **1407.6** | 3.36 | 3.72 | **85.0%** / 3.55 | **98.5%** | 4/6 (1.50x) |
| S4 饱和 | 40 | 40 | 200 | 16.77 | 477.1 | 574.9 | 16.24 | 16.49 | 72.3% / 3.17 | 0% | 18/22 (1.22x) |
| S5 极限 | 60 | 60 | 150 | 16.89 | 532.9 | 678.6 | 15.11 | 15.32 | 71.2% / 3.14 | 0% | 27/33 (1.22x) |

**关键发现**:
- 峰值 **819 comp tok/s** @ 20 并发 (S1, 最佳工作点)
- S3 total tok/s 达 **1408** (受益于 98.5% radix cache 命中, prompt 几乎全复用)
- EAGLE accept rate 64-85%, accept length 2.92-3.55 (理论 4.0)
- 长上下文 EAGLE 表现最佳 (S3: 85% accept, 3.55 长度)
- 负载均衡始终 1.22-1.50x (健康, 符合 balance-rel=1.2)

### 11.3 EAGLE 指标 (warm)

| 指标 | Pod-0 | Pod-1 | 说明 |
|---|---|---|---|
| Accept rate | 73-83% | 64-83% | 长上下文最高 85% |
| Accept length | 3.13-3.48 | 2.91-3.62 | 理论 4.0 |
| Verify calls | 1156-1255 | 1448-1557 | 饱和场景 |

### 11.4 Radix Cache + cache_aware 路由验证

| 测试 | 结果 |
|------|------|
| Sequential shared-prefix (5 reqs) | 1st=3.38s (cold), 2nd+=0.49s → **~7x speedup** |
| 10 sequential shared-prefix (focused) | **100% 路由到 cached pod** (pod-1=10, pod-0=0) |
| 20 concurrent shared-prefix | Pod-1 维持 91% cache_hit_rate |
| 10 concurrent unique-prefix | 均衡分流 pod-0=4, pod-1=6 (1.50x ratio) |
| Unique-prefix (no cache) | 0% cache hit (符合预期) |

**cache_aware router 行为**:
- Shared-prefix 流量: 100% 路由到已有 cache 的 pod (`cached_tokens` 让 router 精准路由)
- Unique-prefix 流量: 1.22-1.50x 均衡比 (无缓存优势时保持负载均衡)

### 11.5 page_size=64 对 cache hit 的影响

| prompt 长度 | sequential 2nd+ cached_tokens | 结论 |
|------------|------------------------------|------|
| 23 tokens | 0 | < 64, page_aligned 后 = 0, 不缓存 |
| 55 tokens | 0 | < 64, 不缓存 |
| 89 tokens | 64 | ≥ 64, 匹配 1 个 page |

S1/S2/S4/S5 的 prompt < 64 token → cache hit=0 (设计行为, 非 bug)。
S3 prompt 3250 token → 98.5% hit。Codex/agent 场景 system prompt >> 64 token, cache 正常。

### 11.6 冷启动 vs 暖启动

| 阶段 | Wall time | 说明 |
|---|---|---|
| 冷启动 (JIT + CUDA Graph) | 28.9s | aiter JIT 编译 ~5min, CUDA Graph 捕获需 5-10 请求 |
| 暖启动 | 2.44s | 所有 JIT 编译完成, CUDA Graph 已捕获 (S1 场景) |

---

## 12. 故障排查指南

### 12.1 Pod 启动失败

```bash
# 检查 pod 状态
kubectl get pods -n kube-system | grep sglang-glm52-2tp8

# 查看启动日志
kubectl logs -n kube-system sglang-glm52-2tp8-sglang-0 --tail=100

# 常见问题:
# 1. GPU 被占用: kubectl exec -- rocm-smi
# 2. 模型文件缺失: kubectl exec -- ls -la /data/model/glm52-fp8/
# 3. NCCL 超时: 检查 NCCL_DEBUG=WARN 日志
```

### 12.2 EAGLE Coredump

```
症状: GPU coredump, NCCL 集体通信全挂
根因: EAGLE decode 阶段 amdgpu kernel soft lockup
修复: 使用 fix-eagle-coredump-v3 镜像 (PR #31478 已烘焙)
紧急: kubectl delete pod 强制重启, 或节点重启
```

### 12.3 Router 503

```
症状: Router 返回 503 Service Unavailable
根因: Worker pod 不健康 或 endpoints 为空
排查:
  kubectl get endpoints -n kube-system sglang-glm52-2tp8-sglang-headless
  kubectl exec -n kube-system deploy/sglang-glm52-2tp8-router -- curl -s http://21.151.225.152:30000/health
```

### 12.4 JIT 编译慢

```
症状: 首次请求 24-30s, 后续 2-3s
根因: aiter JIT 编译需 ~5min, CUDA Graph 捕获需 5-10 请求
解决: 等待暖启动完成, 或预热请求
```

### 12.5 流量不均衡

```
症状: 一个 pod 请求多, 另一个空闲
根因: cache_aware 路由粘滞到 warm cache pod
修复: 重启 pod 清除 cache bias, 或调整 balance-abs-threshold
```

### 12.6 HIP OOM

```
症状: HIP out of memory
根因: mem-fraction-static 过高 + prefill-max-requests 过多
修复: 降低 mem-fraction-static 到 0.75, prefill-max-requests 到 4
```

---

## 13. Git Commit 历史

### 13.1 sglang 仓库 (sglang-2tp8-0723 分支)

```
e58f15feb8 fix(chart): make aiters-tuned-gemm ConfigMap volume optional
6c694bc feat(deployments): complete 2tp8 deployment guide with Dockerfile + patches
262ce77095 fix(chart): add missing FLYDSL and AITER_INDEXER env vars
c5be37de74 feat(deployments): align 2tp8 config with 1tp8 benchmark-optimized params
d0a1b9e593 fix(responses): fix 3 BUGs in /v1/responses API
719a4fcac9 feat(router): eliminate Python proxy, native /v1/responses + /v1/messages
b220d6985d fix(proxy): preserve tool call id and name in ResponsesStreamConverter
7512b9be8d feat(deployments): add GLM-5.2 2tp8 deployment configs and scripts
```

**远程**: `github.com/tanguofu/sglang.git`
**分支**: `sglang-2tp8-0723`

### 13.2 ti-cloud-teamai 仓库 (glm52-tp8-0718 分支)

```
c1e78c6 feat(deployments): add router configs, W2 values, benchmark scripts and results
62c4fb4 docs(wiki): add GLM-5.2 2tp8 deployment and Responses API fix complete record
ba23712 feat(glm52-2tp8): align benchmark-optimized params with 1tp8 reference
```

**远程**: `git.woa.com:ti-cloud/teamai/ti-cloud-teamai.git`
**分支**: `glm52-tp8-0718`

### 13.3 关键文件清单

| 文件 | 仓库 | 说明 |
|---|---|---|
| `docker/rocm-mi308x-glm52/Dockerfile` | sglang | Docker 镜像构建 |
| `docker/rocm-mi308x-glm52/chart/values-glm52-2tp8-merged.yaml` | sglang | Helm values |
| `docker/rocm-mi308x-glm52/chart/templates/sglang-statefulset.yaml` | sglang | STS 模板 |
| `python/sglang/srt/entrypoints/openai/serving_responses.py` | sglang | Responses API 修复 |
| `python/sglang/srt/entrypoints/context.py` | sglang | reasoning_tokens 修复 |
| `docker/rocm-mi308x-glm52/patches/flydsl/fp8_mqa_logits.py` | sglang | FlyDSL 内核 |
| `deployments/glm52-tp8-0718/configs/router/` | ti-cloud-teamai | Router 配置 |
| `deployments/glm52-tp8-0718/scripts/bench_*.py` | ti-cloud-teamai | Benchmark 脚本 |

---

## 附录 A: 完整部署步骤

```bash
# 1. 构建 Docker 镜像
cd /path/to/sglang-worktree-2tp8-0723
docker build -f docker/rocm-mi308x-glm52/Dockerfile \
  -t mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3 .
docker push mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3

# 2. 创建 ConfigMaps
kubectl create configmap sglang-glm52-2tp8-responses-fix \
  -n kube-system \
  --from-file=serving_responses.py=python/sglang/srt/entrypoints/openai/serving_responses.py \
  --from-file=context.py=python/sglang/srt/entrypoints/context.py

kubectl create configmap sglang-glm52-2tp8-native-entrypoint \
  -n kube-system \
  --from-file=entrypoint-native.sh=deployments/glm52-tp8-0718/configs/router/entrypoint-native.sh

# 3. Helm 部署
helm install sglang-glm52-2tp8 \
  docker/rocm-mi308x-glm52/chart/ \
  -f docker/rocm-mi308x-glm52/chart/values-glm52-2tp8-merged.yaml \
  -n kube-system

# 4. 等待 pod 就绪 (约 10-15 分钟)
kubectl wait -n kube-system --for=condition=Ready \
  pod/sglang-glm52-2tp8-sglang-0 \
  pod/sglang-glm52-2tp8-sglang-1 \
  --timeout=900s

# 5. 预热 (JIT 编译 + CUDA Graph 捕获)
for i in $(seq 1 10); do
  curl -s -X POST "http://<router-ip>:30080/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-****" \
    -d "{\"model\": \"glm-5.2\", \"messages\": [{\"role\": \"user\", \"content\": \"hello ${i}\"}], \"max_tokens\": 30}" \
    -o /dev/null
done

# 6. Benchmark
# 20 并发 × 100 max_tokens
for i in $(seq 1 20); do
  curl -s -X POST "http://<router-ip>:30080/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-****" \
    -d "{\"model\": \"glm-5.2\", \"messages\": [{\"role\": \"user\", \"content\": \"Write poem ${i}\"}], \"max_tokens\": 100}" \
    -o /dev/null -w "req-${i}: %{time_total}s\n" &
done
wait
```

---

## 附录 B: 参数调优历史

| 日期 | 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|---|
| 2026-07-18 | mem-fraction-static | 0.82 | 0.75 | HIP OOM 修复 |
| 2026-07-18 | prefill-max-requests | 32 | 4 | HIP OOM 修复 |
| 2026-07-18 | chunked-prefill-size | 131072 | 16384 | HIP OOM 修复 |
| 2026-07-18 | ROCM_QUICK_REDUCE_QUANTIZATION | INT8 | NONE | 零精度损失 |
| 2026-07-19 | hicache-write-policy | write_through | write_back | Host cache 回读修复 |
| 2026-07-19 | hicache-ratio | 1 | 8 → 4 | Host 内存压力 |
| 2026-07-21 | prefill-max-requests | 4 | 8 | 并发提升 |
| 2026-07-21 | max-running-requests | 32 | 48 | 并发提升 |
| 2026-07-21 | cuda-graph-max-bs-decode | 16 | 32 | CUDA Graph 覆盖 |
| 2026-07-23 | mem-fraction-static | 0.75 | 0.88 | 对齐 1tp8 |
| 2026-07-23 | chunked-prefill-size | 16384 | 32768 | 对齐 1tp8 |
| 2026-07-23 | prefill-max-requests | 8 | 32 | 对齐 1tp8 |
| 2026-07-23 | schedule-conservativeness | 1.0 | 0.5 | 对齐 1tp8 |
| 2026-07-23 | watchdog-timeout | 1200 | 3600 | 对齐 1tp8 |
| 2026-07-23 | hicache-ratio | 4 | 2 | 对齐 1tp8 |
| 2026-07-23 | hicache-write-policy | write_back | write_through_selective | 对齐 1tp8 |
| 2026-07-23 | max-running-requests | 48 | 32 | 对齐 1tp8 |
| 2026-07-23 | cuda-graph-max-bs-decode | 32 | 16 | 对齐 1tp8 |
| 2026-07-24 | SGLANG_OPT_USE_AITER_INDEXER | (缺失) | 1 | 对齐 1tp8 |
| 2026-07-24 | FLYDSL_FP8_MQA_LOGITS_VARIANT | (缺失) | mfma_r4_w4 | 对齐 1tp8 |
| 2026-07-24 | enableCacheReport | false | true | 让 router 看到 cached_tokens, cache_aware 路由依据 |
| 2026-07-24 | aiters-tuned-gemm volume | (required) | optional: true | 防止 ConfigMap 删除后 pod 卡 ContainerCreating |

---

*文档结束*
