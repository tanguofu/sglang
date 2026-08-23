# GLM-5.2 MI308X 1P1D PD 部署（生产 live 快照）

**当前可重复部署路径是 Helm chart**，不要再用本目录 YAML 手工 `kubectl apply` 覆盖：

```bash
helm upgrade --install sglang-1p1d docker/rocm-mi308x-glm52-pd/chart \
  -n kube-system
```

Worker 镜像 `mirrors.tencent.com/ti-platform/sglang-glm52-308x:v0517-gdr-kernel-v1` 已把 GDR peermem、L2 flush、DSA/HIP 补丁烤进镜像，启动脚本不再依赖 `/data/mooncake-patched` overlay。`/data` hostPath 只挂模型权重。

本目录仍是 2026-08 生产快照，便于对照；上游以 `chart/` 为准。

## 组件

| 资源 | 节点 | 说明 |
|---|---|---|
| `statefulset/sglang-1p1d-prefill` | node-NODE_PREFILL_0_IP | TP8 prefill，8K chunk 公平调度 |
| `statefulset/sglang-1p1d-decode` | node-NODE_DECODE_0_IP | TP8 decode，NEXTN spec（steps=3, draft=4, num_continuous=4） |
| `deployment/sglang-1p1d-router` | — | sgl-model-gateway，PD 路由 :30001 |

## 关键配置（相对镜像默认的偏差）

- prefill `--chunked-prefill-size 16384 --max-prefill-tokens 32768`（`--enable-hierarchical-cache` + ratio 4 会在 TP8 上分约 2TB host KV，现网 OOM 已回退）
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
| `patch_bootstrap_room_scalar.py` | 把 router 注入的 `bootstrap_room` list 收成标量，避免 Codex `/v1/responses` 把 Mooncake `update_status` 打崩（prefill exit 137） |
| `dsa_indexer_i32fix.py` | DSA MQA logits kernel i32 溢出修复 |
| `patch_eagle_argmax.py` | EAGLE deterministic argmax 修复 |
| `patch_host_staging.py` | decode 侧注入 host staging（RDMA 写 host RAM；**只在 decode 替换 kv_data_ptrs**） |
| `patch_host_staging_v2.py` | 只 hipMemcpy 实际传输的 KV pages，避免全量 66.7GB DMA 打死 bnxt_re QP |
| `patch_prefill_d2h.py` | prefill `_transfer_data`：D2H 后再 RDMA（不改 `send()` 签名） |
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

## MoE tuned + num_continuous_decode_steps=4 优化（2026-08-23）

### MoE GEMM tuned 配置（cu_num=80 修复）

**问题**：MI308X `rocminfo` 报 `Compute Unit: 80`（gfx942 CU 数），但 aiter 自带的
`a8w8_blockscale_tuned_fmoe_glm5_1.csv` 里 GLM-5.2 tuned 配置写的是 `cu_num=304`。
aiter lookup key 含 `cu_num`，用 80 查不到 304 的行，导致所有 MoE GEMM 回退到 `default`
heuristics（损失 20-40% MoE 性能）。

**修复**：把 glm5_1 配置的 cu_num 改成 80，放到母机 `/data/aiter_configs/`（hostPath），
设 `AITER_CONFIG_FMOE` 环境变量指向它。每个节点都要生成：

```bash
# 在每个 prefill/decode pod 里执行（/data 是 hostPath，会写到母机）
src=/sgl-workspace/aiter/aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_glm5_1.csv
dst=/data/aiter_configs/a8w8_blockscale_tuned_fmoe_glm5_1_cu80.csv
mkdir -p /data/aiter_configs
head -1 $src > $dst
awk -F, 'NR>1{$2=80; print}' OFS=, $src >> $dst
```

验证：日志出现 `using 1stage (kernelName1='_ZN5aiter...')` 而非 `using 2stage default`。

### num_continuous_decode_steps 3→4

**效果**：`hipEventSynchronize` 从 46 次(1181ms) 降到 0 次（profile 验证）。

**200K 场景**（bench_unique_cold.py，5/5 PASS 无损）：
- 200K cold: 313s → 273s（-13%）
- 200K warm: 10.5s → 8.3s（-21%）
- 32K 并发: 92s → 34s（-63%）

**短请求 trade-off**：高并发短请求 throughput 略降 14-43%（连续 4 步 decode 占 GPU 更久，
排队延迟增）。长上下文场景净收益，短请求高并发场景需权衡。

### 验证基线（2026-08-23）

- 200K 正确性：5/5 PASS（5 needle 全部提取）
- MoE：`expert=257,topk=9`（主路径）全部走 tuned kernel
- EAGLE：accept_length 2.8-3.1，accept_rate 0.6-0.7
- hipEventSynchronize：0 次（num_steps=4 消除）

## 200K cold-cache 修复（2026-08-15 / GDR 2026-08-16）

当前生产路径是 **GDR**（`SGLANG_PD_HOST_STAGING=0`），不再走 23GB D2H bounce：

- Mooncake：`ibv_reg_mr(GPU)` + amdgpu peermem + HipDeviceGuard（`apply_gdr_peermem.py`）
- 廉价 L2 flush（`patch_gdr_flush.py` + `gdr_l2_flush.hsaco`）：prefill writeback + 对端 8B RDMA READ；decode `buffer_inv`
- unique-needle 64K：**5/5**（~29–61s）；~157K/200K：**5/5**（~137–143s）；QP 存活，之后短请求仍 200
- 纯 GDR、不 flush：64K 会回到 `1.1.2...</think>` 乱码（HTTP 200、QP 仍活）

Host staging（`HOST=1`）仍可作为回退：prefill GPU → D2H host → RDMA → decode host → 按 `kv_indices` 选择性 H2D。`/data/mooncake-patched/engine...so.bak-hoststaging` 是旧 bounce so。

- **不要**把整份 `conn_v0516_fixed.py` 盖到 v0.5.17 上（`send()` 缺 `num_kv_tokens`）

详见 `docs/pd_200k_cold_cache_fix_progress.md`。

## 已知限制

- >64K 上下文 reasoning 重复：fp8 KV cache 精度退化（已知问题）
- 极端长 prefill（400K+）排空窗口内短请求可能触及 300s 客户端超时
- 严格 GPU-direct RDMA 需内核 ≥5.12（dmabuf）或 netxtreme-peer-mem
