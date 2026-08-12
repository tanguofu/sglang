# GLM-5.2 MI308X 1P1D PD 部署（生产 live 快照）

本目录是 `kube-system` 命名空间 `sglang-1p1d` 生产部署的完整快照（2026-08-12 导出），
基于 sglang（v0.5.15.post1 ROCm 定制镜像）+ Mooncake RDMA PD 分离，在 TKE MI308X
（gfx942, 8×400G bnxt_re RoCE）集群验证通过。

上游 `chart/` 中的 Pod 模板是历史版本，已与本部署漂移；**以本目录为准**。

## 组件

| 资源 | 节点 | 说明 |
|---|---|---|
| `statefulset/sglang-1p1d-prefill` | node-21.151.225.144 | TP8 prefill，8K chunk 公平调度 |
| `statefulset/sglang-1p1d-decode` | node-21.151.225.132 | TP8 decode，NEXTN spec（steps=2） |
| `deployment/sglang-1p1d-router` | — | sgl-model-gateway，PD 路由 :30001 |

## 关键配置（相对镜像默认的偏差）

- prefill `--chunked-prefill-size 8192 --max-prefill-tokens 8192`（原 32768，防超长 prefill 饿死短请求）
- decode `--watchdog-timeout 300`（原 7200，scheduler 卡死 5 分钟内自动重建）
- `--disaggregation-transfer-backend mooncake`，`MOONCAKE_PROTOCOL=rdma`，8 卡 `bnxt_re_bond0-7` 全映射，`MC_GID_INDEX=3`
- DSA attention（`--attention-backend dsa`，tilelang 子后端），KV cache fp8_e4m3

## patches/ 运行时补丁（随启动脚本注入）

| 文件 | 作用 |
|---|---|
| `patch_overlap_hip_wait.py` | HIP `publish_ready.synchronize()` → `wait()`，修 scheduler host 阻塞死锁 |
| `patch_decode_pd_health_flush.py` | decode loop 每轮 flush synthetic health reply，修 transfer 期间 /health 饥饿 |
| `patch_scheduler_health.py` | decode prealloc/transfer 队列计入 busy 判定 |
| `conn_v0516_fixed.py` | Mooncake conn.py 修复（GQA 复制、hipSetDevice、mamba state） |
| `protocol_v0516_patched.py` / `serving_responses_v0516_patched.py` | /v1/responses bootstrap 字段透传（codex 需要） |
| `dsa_indexer_i32fix.py` | DSA MQA logits kernel i32 溢出修复 |
| `patch_eagle_argmax.py` | EAGLE deterministic argmax 修复 |
| `bf16_tuned_gemm_mi308x.csv` | AITER tuned GEMM（含 M=8192/16384/4096 及 decode 小 M，hipblaslt） |

### 二进制制品（不入库，需从源节点拷贝）

- `/data/mooncake-patched/engine.cpython-310-x86_64-linux-gnu.so` — 支持 HIP dmabuf 的 Mooncake engine（ts4 验证版）
- `/data/mooncake-patched/libbnxt_re-235.2.86.0.so` — 内核 ABI v8 兼容的 bnxt_re verbs 驱动

部署到两节点的 `/data/mooncake-patched/` 与 `/data/aiter_configs/bf16_tuned_gemm.csv`。
CSV 修改后必须按 merge key 去重（gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle），
否则 AITER merge 抛 RuntimeError 导致 CrashLoop；两节点 /data 是各自 hostPath，需分别写入。

## 验证基线（2026-08-12）

- 隔离 C8：short 102 tok/s、medium 187 tok/s、decode_c8 421 tok/s，全部 32/32
- 冷 200K prefill：~1461 tok/s（8K chunk + tuned GEMM；调优前 830）
- 公平性：400K 长请求并发下短请求 16/16 成功无超时
- 内核 5.4 无 dmabuf，GDR 走 `ibv_reg_mr` 回退路径（传输与 chunked prefill 完全重叠）

## 已知限制

- >64K 上下文 reasoning 重复：fp8 KV cache 精度退化（已知问题）
- 极端长 prefill（400K+）排空窗口内短请求可能触及 300s 客户端超时
- 严格 GPU-direct RDMA 需内核 ≥5.12（dmabuf）或 netxtreme-peer-mem
