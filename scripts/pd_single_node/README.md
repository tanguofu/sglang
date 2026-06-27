# GLM-5.2-FP8 单机 PD 分离部署方案（MI355X / mori / XGMI）

> **参考 iwiki**: [单机 P4+D4 PD 分离部署方案](https://iwiki.woa.com/p/4022572310)  
> **父页面**: [SGLang 部署 GLM-5.2-FP8 DSA](https://iwiki.woa.com/p/4022389950)  
> **更新时间**: 2026-06-24  
> **代码路径**: `scripts/pd_single_node/`（本仓库）  
> **mori PP 修复分支**: `fix/mori-pp-kv-slices-pd-validation`

---

## 1. 结论先行

### 1.1 单机 4P+4D 是否值得做？

根据 [iwiki 4022572310](https://iwiki.woa.com/p/4022572310) 的 benchmark 结论：

| 指标 | TP8+MTP（8 卡一体） | 单机 4P+4D PD | 评估 |
|------|---------------------|---------------|------|
| decode 吞吐 | 2236–4477 tok/s | 1177–2588 tok/s | TP8 更优（约 2×） |
| mid_c32（512 tok decode） | 2153 tok/s | 414 tok/s | PD 仅 0.19×（KV 传输 + 算力减半） |
| 1M 并发 | ~2 个 | ~1 个 | TP8 更优 |
| TTFT | ~24s | ~19s | PD 略优（P 专属 prefill） |
| P/D 资源隔离 | 无 | 有 | PD 更优 |

**结论**：单机 4+4 PD **不适合**作为 decode 吞吐最大化的生产配置；其价值主要在 **prefill/decode 隔离** 和 **更低 TTFT**。若目标是 decode 性能，应优先 **TP8+MTP**；若目标是 PD 分离，应规划 **跨机 8P+8D**（见第 7 节）。

### 1.2 本仓库验证状态（2026-06-24）

验证机器：`216.128.154.57`、`149.28.114.238`（各 8× MI355X）  
镜像：`lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260620`  
部署方式：**单容器** `run_single_container.sh` + `start_pd_stack.sh`（**不用**双容器 `run_scheme.sh`）

| 方案 | 拓扑 | 154.57 | 238 | 说明 |
|------|------|--------|-----|------|
| **PD-1a** | P:TP4 + D:TP4 | Deploy ✅ / Smoke ✅ | Deploy ✅ / Smoke ✅ | 最简 4+4，D 侧 KV pool 不足以支撑 1M |
| **PD-1b** | P/D 各 PP2+TP2 | Deploy ✅ / Smoke ❌ | Deploy ✅ / Smoke ❌ | mori PP KV slice 已修，e2e 仍超时 |
| **PD-1d** | P:TP4 + D:TP4+MTP | Deploy ✅ / Smoke ✅ | Deploy ✅ / Smoke ✅ | D 侧 MTP steps=2，chat 路径通过 |

---

## 2. 部署架构

```
BM 单机 8× MI355X
┌─ docker: sglang_pd_stack (host network, ipc=host, pid=host) ─────────────┐
│  GPU0-3  Prefill  :30010   mori KV → XGMI（同节点 P2P）                    │
│  GPU4-7  Decode   :30020   NCCL_P2P_DISABLE=1（避免 hipIpc 冲突）         │
│  Router           :8000    sglang_router --pd-disaggregation --mini-lb   │
└──────────────────────────────────────────────────────────────────────────┘
```

与 iwiki 原始方案的区别：

| 项 | iwiki 原始（三进程裸跑） | 本方案（单容器） |
|----|--------------------------|------------------|
| 进程编排 | 宿主机直接起 3 个 python | 一个容器内顺序起 P→D→Router |
| KV 传输 | 文档写「单机 XGMI/local」 | 实测需显式 `MORI_DISABLE_AUTO_XGMI=0` + 排除 ionic |
| 端口 | P:30000 / D:30001 | P:30010 / D:30020（避免与已有服务冲突） |
| IB 设备 | 未强调 | **禁止** `--disaggregation-ib-device ionic_*`（会误走 RoCE） |

---

## 3. 方案矩阵（PD-1a / 1b / 1d）

| 方案 | P 侧 | D 侧 | 1M context | 适用场景 |
|------|------|------|------------|----------|
| **PD-1a** | TP4, GPU0-3 | TP4, GPU4-7 | ❌ D KV pool ~49GB | 最简单验证、低延迟隔离 |
| **PD-1b** | PP2+TP2, 39+39 层 | PP2+TP2, 39+39 层 | ✅ 理论 ~5-6 路 | 混合负载、长上下文（**e2e 未通过**） |
| **PD-1d** | TP4 | TP4 + MTP (steps=2) | ❌ | decode 吞吐相对最优的 PD 形态 |

PD-1c（P:PP2+TP2 + D:TP4+MTP）为折中方案，当前验证脚本未纳入自动回归。

---

## 4. 环境与前置条件

### 4.1 硬件 / 软件

| 项 | 值 |
|----|-----|
| GPU | 8× MI355X（309GB/GPU, gfx950） |
| GPU 互联 | 8 卡 full mesh XGMI |
| NIC | 每 GPU 一块 ionic RoCE（**单机 KV 不走此路径**） |
| 模型 | `/data/models/GLM-5.2-FP8` |
| Docker 镜像 | `lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260620` |

### 4.2 同步脚本到目标机器

```bash
# 在本仓库根目录执行
chmod +x scripts/pd_single_node/sync_to_host.sh
./scripts/pd_single_node/sync_to_host.sh root@216.128.154.57
./scripts/pd_single_node/sync_to_host.sh root@149.28.114.238
```

同步内容：

- `/data/pd_single_node/` — 部署与验证脚本
- `/data/patch_glm_config.py` — GlmMoeDsaConfig head_dim 映射修复
- `/data/patch_pp_missing_layer.py` — PP>1 时 PPMissingLayer 修复
- `/data/patch_mori_pp_kv_slices.py` — mori PP KV mem-desc 切片修复（PD-1b 必需）

---

## 5. 关键配置

### 5.1 mori 单机 XGMI（必须）

```bash
export MORI_DISABLE_AUTO_XGMI=0
export MORI_IO_NODE_ID=mi355x-single-node   # P/D 两侧相同
export MORI_RDMA_DEVICES="^ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
unset SGLANG_HOST_IP
# 不要设置 --disaggregation-ib-device
```

若缺少上述配置，mori 会尝试 ionic RoCE RDMA 建链，典型错误：

```
ibverbs.cpp:189 syscall failed with Connection timed out
```

### 5.2 NCCL / IPC（必须）

P/D 同机共享 `/dev/dri` 时，decode 侧需：

```bash
export NCCL_P2P_DISABLE=1
```

容器需 `--ipc=host --pid=host`，否则可能出现 `hipIpcGetMemHandle failed`。

### 5.3 公共 server 参数

```bash
--model-path /data/models/GLM-5.2-FP8
--trust-remote-code
--context-length 1048576
--kv-cache-dtype fp8_e4m3
--mem-fraction-static 0.85
--chunked-prefill-size 32768
--enable-fused-qk-norm-rope
--disaggregation-transfer-backend mori
--disaggregation-bootstrap-port 9000
```

### 5.4 PD-1d MTP（D 侧）

```bash
--speculative-algorithm NEXTN
--speculative-num-steps 2
--speculative-num-draft-tokens 3
--speculative-eagle-topk 1
--disable-radix-cache    # D 侧与 MTP 冲突，必须关闭 prefix cache
--max-running-requests 64
```

---

## 6. 部署与验证

### 6.1 启动单个方案

```bash
bash /data/pd_single_node/run_single_container.sh PD-1a mori
bash /data/pd_single_node/run_single_container.sh PD-1d mori
```

首次加载约 8–10 分钟，router `:8000/health` 就绪即表示 stack 起来。

### 6.2 Smoke test

```bash
docker exec sglang_pd_stack python3 /data/pd_single_node/smoke_test.py http://127.0.0.1:8000 PD-1a
```

### 6.3 全量回归（PD-1a / 1b / 1d）

```bash
nohup bash /data/pd_single_node/run_validation.sh \
  > /data/pd_single_node/logs/xgmi_full_validation.log 2>&1 &
tail -f /data/pd_single_node/logs/validation_results.txt
```

### 6.4 停止

```bash
bash /data/pd_single_node/stop_all.sh
```

### 6.5 日志

| 路径 | 内容 |
|------|------|
| `/data/pd_single_node/logs/PD-{scheme}_mori_prefill.log` | P 侧 |
| `/data/pd_single_node/logs/PD-{scheme}_mori_decode.log` | D 侧 |
| `/data/pd_single_node/logs/PD-{scheme}_mori_router.log` | Router |
| `/data/pd_single_node/logs/validation_results.txt` | 自动化结果 |

---

## 7. 已知问题与修复

| Bug | 根因 | 修复 | 状态 |
|-----|------|------|------|
| mori `Connection timed out` | 单机误走 ionic RoCE RDMA | XGMI env（§5.1） | ✅ 已修复 |
| `hipIpcGetMemHandle failed` | P/D 共享 DRI，P2P IPC 冲突 | `NCCL_P2P_DISABLE=1` + `pid=host` | ✅ 已修复 |
| `patch_glm_config.py` | GlmMoeDsa head_dim→qk_rope_head_dim | patch 脚本 | ✅ 已修复 |
| PP1 `PPMissingLayer.embedding_dim` | PP>1 缺属性 | `patch_pp_missing_layer.py` | ✅ 已修复 |
| mori PP MLA slice 不匹配 | `_get_mla_mem_desc_slices` 未走 local fast path | `patch_mori_pp_kv_slices.py` + 上游 `conn.py` 修复 | ⚠️ 部分修复 |
| PD-1b smoke 180s 超时 | PP2 首请求慢 / XGMI wait 挂起 | 待排查 `SGLANG_MORI_TRANSFER_TIMEOUT_MS` | ❌ 未解决 |

上游 mori PP 修复已提交到分支 `fix/mori-pp-kv-slices-pd-validation`：

- `python/sglang/srt/disaggregation/mori/conn.py`
- `scripts/patch_mori_pp_kv_slices.py`

---

## 8. 性能参考（iwiki BM-57 实测）

Router `:8000` 压测（warm）：

| Suite | 4P+4D PD | TP8+MTP 基线 | 比值 |
|-------|----------|--------------|------|
| short_c32 | 1177 tok/s | 2237 tok/s | 0.53× |
| short_c128 | 2588 tok/s | 4477 tok/s | 0.58× |
| mid_c32 | 414 tok/s | 2154 tok/s | 0.19× |

PD-1d D 侧 MTP：accept len 2.20–2.23，accept rate 0.60–0.62。

---

## 9. 推荐生产选型

| 负载 | 推荐配置 | 理由 |
|------|----------|------|
| 纯 decode / 对话 | **TP8+MTP steps=2** | 2–3× decode 加速，prefix cache 可用 |
| 单机要 PD 隔离 | **PD-1d**（已 e2e 通过） | PD 形态中 decode 相对最优 |
| 1M 长上下文 + 混合负载 | **PD-1b**（目标）或 **TP4/PP2 基线** | PD-1b e2e 未通过；基线已稳定 |
| 跨机 PD（目标架构） | **8P + 8D**，mori 跨机 RDMA | decode 算力不减半；需先解决 NCCL IPv6 跨机 init |

### 当前最优一体配置（非 PD）

```bash
--tp-size 8 --pp-size 1
--speculative-algorithm NEXTN --speculative-num-steps 2
--speculative-num-draft-tokens 3 --speculative-eagle-topk 1
--max-running-requests 128
```

---

## 10. 相关链接

- iwiki 部署页: https://iwiki.woa.com/p/4022572310
- iwiki 父页: https://iwiki.woa.com/p/4022389950
- iwiki 全进展: https://iwiki.woa.com/p/4022603934
- SGLang PD 文档: `docs/advanced_features/pd_disaggregation.md`
- EP8+MTP issue: https://github.com/sgl-project/sglang/issues/29039
