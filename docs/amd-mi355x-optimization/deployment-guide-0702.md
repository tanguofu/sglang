
# GLM-5.2 AMD MI355X 0702 优化部署方案

## 1. 概述

本方案描述 GLM-5.2-FP8 模型在 AMD MI355X (8×309GB) 上的优化部署配置。基于 SGLang v0.5.14-rocm720，通过 MTP 3/4 + FP8 KV cache + CUDA graph 精简 + dual-stream MoE 实现 40-67% 吞吐提升，且模型质量无退化。

**附件**：
- `sglang_0702_deployment_kit.tar.gz` — 完整部署包（15 个 patches + launch 脚本）
- `launch_nomultistream.sh` — 启动脚本（单独）

---

## 2. 环境要求

- **硬件**: 8× AMD MI355X (309GB VRAM each)
- **Docker 镜像**: `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260702`
- **模型**: GLM-5.2-FP8 (704GB, 141 safetensors)
- **SGLang**: v0.5.14 (rocm720)

---

## 3. 部署步骤

### 3.1 解压部署包

```bash
# 上传 sglang_0702_deployment_kit.tar.gz 到 /data/
cd /data
tar xzf sglang_0702_deployment_kit.tar.gz
# 解压后得到：
#   /data/patches/ (15 个 patch 脚本)
#   /data/launch_nomultistream.sh
```

### 3.2 确认模型路径

```bash
ls /data/models/GLM-5.2-FP8/
# 应包含 141 个 safetensors 文件 + config.json
```

### 3.3 启动服务

```bash
bash /data/launch_nomultistream.sh
```

### 3.4 验证

```bash
# 检查 API
curl -s http://localhost:30000/v1/models \
  -H 'Authorization: Bearer sk-46faecc9d0bc4dcd9db6a15c73ae91c8' | python3 -m json.tool

# 快速测试
curl -s http://localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-46faecc9d0bc4dcd9db6a15c73ae91c8' \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"What is 15*17?"}],"max_tokens":256,"temperature":0}'
```

---

## 4. 优化配置详解

### 4.1 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--speculative-num-steps` | 3 | MTP 步数（上游默认 2，我们优化为 3） |
| `--speculative-num-draft-tokens` | 4 | 每 step draft token 数（上游默认 3） |
| `--speculative-eagle-topk` | 1 | 每 step 只取 top-1 |
| `--kv-cache-dtype` | fp8_e4m3 | FP8 KV cache（gfx950 原生支持，节省 50% 显存） |
| `--max-running-requests` | 32 | 基于负载分析（99% 时间 ≤2 并发） |
| `--cuda-graph-bs-decode` | 1 2 3 4 5 6 7 8 9 10 12 16 | 精简 graph 捕获范围 |
| `--cuda-graph-max-bs-decode` | 16 | 最大 graph bs |
| `--mem-fraction-static` | 0.88 | 静态显存占比 |
| `--context-length` | 1048576 | 1M 上下文 |

### 4.2 环境变量

| 变量 | 值 | 说明 |
|------|-----|------|
| `SGLANG_USE_AITER` | 1 | 启用 AITER kernel |
| `SGLANG_ROCM_FUSED_DECODE_MLA` | 1 | 融合 decode MLA kernel |
| `SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM` | 1 | dual-stream MoE（routed + shared 并行） |
| `SGLANG_MOE_PADDING` | 1 | MoE padding 优化 |
| `ROCM_QUICK_REDUCE_QUANTIZATION` | INT8 | allreduce 量化 |
| `HIP_FORCE_DEV_KERNARG` | 1 | HIP kernel arg 优化 |
| `NCCL_MIN_NCHANNELS` | 112 | NCCL 通道数 |

### 4.3 Patches（15 个）

| Patch | 用途 |
|-------|------|
| `patch_glm_config.py` | qk_rope_head_dim override |
| `patch_dsa_backend_v2.py` | DSA backend view→reshape |
| `patch_dsa_draft_extend.py` | DSA draft extend 修复 |
| `patch_dsa_indexer_graph.py` | DSA graph HIP 支持（7 sub-patches） |
| `patch_disable_mha_swap.py` | 禁用 MHA companion swap |
| `patch_deterministic_argmax.py` | ROCm 确定性 argmax |
| `patch_draft_forward_argmax.py` | draft_forward argmax 修复 |
| `patch_hip_fusion_dual_stream_0702_v6.py` | HIP fusion + dual stream（13 sub-patches） |
| `patch_dual_stream_kw_fix.py` | dual-stream kw 修复 |
| `patch_draft_alt_stream.py` | draft model alt_stream 修复 |
| `patch_fp8_view_fix.py` | FP8→uint8 view 修复 |
| `patch_tp_style_0702.py` | mla_kv_a_proj TP style |
| `patch_cuda_fp8_include.py` | cuda_fp8.h→hip/hip_fp8.h |
| `gen_aiter_dense_0702_v2.py` | GEMM config 生成 |
| `gen_a8w8_dense.py` | a8w8 blockscale config 生成 |

---

## 5. 性能数据

### 5.1 吞吐量对比（vs Master 基线）

| 测试 | 并发 | Master tok/s | 优化 tok/s | 提升 |
|------|------|-------------|-----------|------|
| decode_1024 | C=1 | 211.7 | 305.5 | +44% |
| decode_1024 | C=8 | 1,269.2 | 1,767.0 | +39% |
| decode_2048 | C=8 | 1,091.1 | 1,553.7 | +42% |
| medium_ctx | C=4 | 656.6 | 1,251.5 | +91% |

### 5.2 质量验证

| Benchmark | 得分 | 目标 | 状态 |
|-----------|------|------|------|
| AIME 2025 (maj@4) | 96.7% | 87.7% | ✅ 超目标 |
| HLE | 70.0% | 40.5% | ✅ 超目标 |
| Terminal-Bench | 80.0% | 81.0% | ✅ 接近 |
| 输出对比 vs Master | 答案 100% 一致 | — | ✅ 无退化 |

### 5.3 MTP 指标

- Accept rate: 76-82%
- Accept length: 3.275/4.0
- Decode throughput: 178-196 tok/s (C=1)

---

## 6. 从基线更新到优化配置

### 6.1 差异清单

| 差异项 | 基线 | 优化 | 动作 |
|--------|------|------|------|
| 镜像 | 20260629 | 20260702 | 换镜像 |
| MTP steps/draft | 2/3 | 3/4 | 改参数 |
| max_running_requests | 128 | 32 | 改参数 |
| cuda-graph-bs decode | 默认 | 显式 1-16 | 加参数 |
| dual-stream | 未启用 | 启用 | 加 env |
| Patches | 7 个 | 15 个 | 上传 8 个新 patch |

### 6.2 更新步骤

```bash
# 1. 上传部署包
scp sglang_0702_deployment_kit.tar.gz root@<master-ip>:/data/
ssh root@<master-ip> "cd /data && tar xzf sglang_0702_deployment_kit.tar.gz"

# 2. 停止旧容器
ssh root@<master-ip> "docker rm -f <old-container-name>"

# 3. 启动新配置
ssh root@<master-ip> "bash /data/launch_nomultistream.sh"

# 4. 等待模型加载（~3-5 分钟）
# 5. 验证 API + 跑 benchmark
```

---

## 7. 注意事项

- **MTP4 不可用**：accept rate 过低，已否决，不要尝试 steps=4
- **`SGLANG_ROCM_USE_MULTI_STREAM` 不要启用**：低并发下损失 22% 吞吐
- **`--enable-single-batch-overlap` 不要启用**：禁用 shared expert fusion，净负收益
- **MTP 对 kernel 数值变化极度敏感**：任何 kernel 精度变更都可能导致 accept rate 下降
- **1M context 需调优**：当前 `mem_fraction_static=0.88` 下 1M tokens KV cache 不够，需降低或用更高效压缩
- **部分 patches 可能已在新镜像中修复**：建议逐个验证是否仍需要
