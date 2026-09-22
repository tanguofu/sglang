# GLM-5.2/5.3 MI308X 1P1D PD 从零部署指南

**环境**: TKE MI308X 集群（gfx942, 内核 5.4, ROCm 7.2.0）
**镜像**: `mirrors.tencent.com/ti-platform/sglang-glm52-308x:v0919c-src-cufix`
**源码分支**: `mi308x-1p1d-src`（镜像由该分支 Dockerfile 构建，55 个构建期 assert 门禁）
**更新日期**: 2026-09-22

## 当前镜像包含的优化（v0919c-src-cufix）

| 优化 | 开关 | 效果 |
|---|---|---|
| CK dense GEMM (gfx942) | decode: `SGLANG_ROCM_USE_CK_GEMM_GFX942=1` | decode tps +2.9% |
| packed-row v2 top-k | prefill: `SGLANG_DSA_USE_TOPK_V2_PACKED=1` | prefill topk 10.4%→1.0%，TTFT −6.3% |
| gfx942 CU-count fix | 默认（运行时查 rocminfo） | extend/bs=32 路径 tilelang 正确分块 |
| fused topk v2 (coop) | decode: `SGLANG_OPT_USE_TOPK_V2=1` | 精确 topk |
| QuickReduce gfx942 fix | 默认 | fp16 饱和修复（`__floats2half2_rn`） |
| MoE cu80 tuned CSV | `AITER_CONFIG_FMOE`（hostPath） | MoE 不走 default |
| GEMM cu80 tuned CSV | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE`（hostPath） | CK GEMM 查表命中 |

## 前置条件

### 1. 节点要求

- 4 个 MI308X 节点（8 GPU/节点），内核 5.4+（GDR 走 ibv_reg_mr 回退）
- 节点间 RDMA（bnxt_re bond0-7）
- 每节点 `/data` 目录（hostPath，放模型权重 + aiter configs）

### 2. 模型权重

每个节点的 `/data/model/glm52-fp8/` 放 GLM-5.2 FP8 模型。

### 3. GPU device plugin

```bash
kubectl get ds -n kube-system amdgpu-device-plugin-daemonset
# 确保所有 MI308X 节点有 amdgpu device plugin
```

## 部署步骤

### Step 1: 准备 hostPath 数据（每个节点）

节点 `/data` 需要以下内容（全部在 repo `docker/rocm-mi308x-glm52-pd/live-1p1d/` 有快照）：

```bash
# 1a. 模型权重
/data/model/glm53-fp8/          # GLM-5.3 FP8（当前生产模型）

# 1b. aiter tuned CSV（cu80 版本，md5 与 4 节点一致）
#     repo: live-1p1d/aiter-configs/
/data/aiter_configs/a8w8_blockscale_tuned_gemm_glm5_1_cu80.csv   # GEMM（CK 查表）
/data/aiter_configs/a8w8_blockscale_tuned_fmoe_glm5_1_cu80.csv   # MoE
#     bf16 gate/indexer CSV 由 chart init 容器自动生成（aiterBf16Gemm.enabled）

# 1c. 运行时 patch（prefill 4 个 + decode 1 个，容器启动时执行）
#     repo: live-1p1d/patches/
/data/mooncake-patched/patch_big_node_backup.py    # prefill: 大节点首插即备份
/data/mooncake-patched/patch_zombie_free_fix.py    # prefill: chunked-insert 释放修复
/data/mooncake-patched/patch_evict_backup.py        # prefill: 驱逐前 D->H 备份
/data/mooncake-patched/patch_prefetch_log.py        # prefill: L3 prefetch 日志
/data/mooncake-patched/patch_tool_schema_regex.py   # 全部: 跳过 format:regex
```

**关键**：MI308X 报 `cu_num=80`，aiter 的 glm5_1 tuned 配置写 `cu_num=304`。
CSV 必须是 cu80 版本，否则 MoE 全走 default（损失 20-40%）。若从 aiter 原始
CSV 重新生成：

```bash
src=/sgl-workspace/aiter/aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_glm5_1.csv
dst=/data/aiter_configs/a8w8_blockscale_tuned_fmoe_glm5_1_cu80.csv
head -1 $src > $dst
awk -F, 'NR>1{$2=80; print}' OFS=, $src >> $dst
# 补 decode 形状 expert=256,topk=8（token 1..128），保留 257/9 给 prefill
python3 docker/rocm-mi308x-glm52-pd/scripts/merge_decode_fmoe_256_8.py $dst
```

### Step 2: 配置 chart values

编辑 `docker/rocm-mi308x-glm52-pd/chart/values.yaml`：

```yaml
# 镜像（已 push 到 mirror）
# prefills[]/decodes[] 每项的 workerImage（顶层 workerImage 是旧值，不生效）
prefills:
  - nodeName: node-21.151.225.144
    ip: "21.151.225.144"
    workerImage: mirrors.tencent.com/ti-platform/sglang-glm52-308x:v0919c-src-cufix
    extraEnv:
      AITER_CONFIG_GEMM_A8W8_BLOCKSCALE: /data/aiter_configs/a8w8_blockscale_tuned_gemm_glm5_1_cu80.csv
      AITER_LOG_TUNED_CONFIG: "1"
      SGLANG_DSA_USE_TOPK_V2_PACKED: "1"     # packed topk（prefill 侧）
  - nodeName: node-21.151.225.152
    ...同上...
decodes:
  - nodeName: node-21.151.225.132
    ip: "21.151.225.132"
    workerImage: mirrors.tencent.com/ti-platform/sglang-glm52-308x:v0919c-src-cufix
    extraEnv:
      AITER_CONFIG_GEMM_A8W8_BLOCKSCALE: /data/aiter_configs/a8w8_blockscale_tuned_gemm_glm5_1_cu80.csv
      AITER_LOG_TUNED_CONFIG: "1"
      SGLANG_OPT_USE_TOPK_V2: "1"            # fused topk（decode 侧）
      SGLANG_ROCM_USE_CK_GEMM_GFX942: "1"    # CK dense GEMM（decode 侧）
  - nodeName: node-21.151.225.172
    ...同上...

# API key
apiKey: sk-YOUR_API_KEY
```

### Step 3: Helm 部署

```bash
helm upgrade --install sglang-1p1d docker/rocm-mi308x-glm52-pd/chart \
  -n kube-system \
  -f docker/rocm-mi308x-glm52-pd/chart/values.yaml
```

### Step 4: 等待 pod ready

```bash
kubectl get pods -n kube-system | grep sglang-1p1d
# 等待所有 pod Running（首次启动有 JIT 编译，约 5-10 分钟/pod）
```

**注意**：readiness probe `initialDelaySeconds=900`（15 分钟），但 router 用自己的
健康检查（5s 间隔），不依赖 k8s readiness。pod health 200 即可服务。

### Step 5: 验证

```bash
# 1. 所有 pod health
for pod in sglang-1p1d-prefill-0 sglang-1p1d-prefill-1-0 sglang-1p1d-decode-0 sglang-1p1d-decode-1-0; do
  kubectl exec $pod -n kube-system -c sglang -- curl -s -o /dev/null -w "$pod: %{http_code}\n" localhost:30000/health
done

# 2. MoE 走 tuned（不是 default）
kubectl logs sglang-1p1d-decode-0 -n kube-system | grep "fused_moe.*using" | head -3
# 应该看到: using 1stage (kernelName1='_ZN5aiter...')  而非 using 2stage default

# 3. 200K 正确性测试
kubectl exec sglang-1p1d-prefill-0 -n kube-system -c sglang -- python3 /data/bench_unique_cold.py
# 应该 5/5 PASS
```

## 关键配置说明

### MoE tuned（AITER_CONFIG_FMOE）

- **必须**：不设这个 env，MoE 全走 default，损失 20-40%
- 配置文件在 hostPath `/data/aiter_configs/`，重建节点会丢，需重新生成
- 长期方案：baked-in 到镜像

### num_continuous_decode_steps=4

- 消除 `hipEventSynchronize`（46 次→0 次）
- 200K 场景提升 13-63%
- 短请求高并发略降 14-43%（trade-off）

### EAGLE spec decode

- `speculative-num-steps 2, num-draft-tokens 4, eagle-topk 1`（v0919c 验证组合）
- steps=3 实测净收益仅 +2%（accept 2.756 但每步多 1 draft token），维持 steps=2
- accept_length 2.2-2.5，accept_rate ~0.6

### GDR L2 flush

- `SGLANG_PD_HOST_STAGING=0`（真 GDR）
- `MC_DISABLE_HIP_TRANSPORT=1`
- 内核 5.4 走 `ibv_reg_mr` 回退 + `gdr_l2_flush.hsaco` 做 L2 一致性

## 故障排查

### pod 起不来

```bash
kubectl logs <pod> -n kube-system --previous | tail -30
# 常见: JIT 编译超时（等 5-10 分钟）、RDMA 连接失败、模型路径错误
```

### MoE 走 default

```bash
# 检查 AITER_CONFIG_FMOE env
kubectl exec <pod> -n kube-system -c sglang -- bash -c 'echo $AITER_CONFIG_FMOE'
# 检查配置文件存在
kubectl exec <pod> -n kube-system -c sglang -- ls -la /data/aiter_configs/a8w8_blockscale_tuned_fmoe_glm5_1_cu80.csv
```

### PD transfer 失败

```bash
kubectl logs <prefill-pod> -n kube-system | grep -i "TRANSFER_FAIL" | tail -5
# 常见: hicache storage error, 重启 prefill pod
```

## 回滚

```bash
# 回退 num_continuous_decode_steps 到 3
kubectl patch statefulset sglang-1p1d-decode -n kube-system --type=json \
  -p='[{"op":"replace","path":"/spec/template/spec/containers/0/args",...}]'
# 或 helm rollback
helm rollback sglang-1p1d -n kube-system
```

## 镜像构建（从源码复现）

镜像由 `mi308x-1p1d-src` 分支构建（基础镜像 upstream 11d03eaeef + Dockerfile
构建期 patch，55 个 assert 门禁保证 patch 全部命中）：

```bash
# 1) 推分支（若未推）
git push origin mi308x-1p1d-src

# 2) ti-builder（9.135.3.173）构建
ssh ti-builder
mkdir -p /data/sglang-build && cd /data/sglang-build
if [ ! -d sglang-src ]; then
  git clone --depth 1 --branch mi308x-1p1d-src \
    https://github.com/tanguofu/sglang.git sglang-src
else
  cd sglang-src && git fetch origin mi308x-1p1d-src && \
  git checkout -B mi308x-1p1d-src FETCH_HEAD
fi
cd /data/sglang-build/sglang-src
TAG=v0919c-src-cufix
docker build -f docker/rocm-mi308x-glm52-pd/Dockerfile \
  -t mirrors.tencent.com/ti-platform/sglang-glm52-308x:$TAG .
docker push mirrors.tencent.com/ti-platform/sglang-glm52-308x:$TAG
# 构建时长 ~30-60 分钟（mooncake make -j64 为主），单镜像 ~64GB
```

源码一致性已验证：本地分支 8 个关键文件 md5 与生产镜像内完全一致
（fp8_utils / topk_v2.cuh / conn.py / deepseek_v2.py / dsa_indexer.py /
eagle_worker_v2.py / tilelang_kernel.py / dsa_topk_backend.py）。

## 参考

- [P0 MoE tuned 修复详情](../../../docs/pd_200k_cold_cache_fix_progress.md)
- [chart values.yaml](../chart/values.yaml) - 完整配置
- [live-1p1d README](live-1p1d/README.md) - 生产快照说明
- [smooth-src-build-rollout-plan.md](smooth-src-build-rollout-plan.md) - 源码化构建全流程
