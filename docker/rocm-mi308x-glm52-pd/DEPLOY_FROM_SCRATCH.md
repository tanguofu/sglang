# GLM-5.2 MI308X 1P1D PD 从零部署指南

**环境**: TKE MI308X 集群（gfx942, 内核 5.4, ROCm 7.2.0）
**镜像**: `mirrors.tencent.com/ti-platform/sglang-glm52-308x:v0825-gate-indexer`
**更新日期**: 2026-08-25

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

### Step 1: 生成 MoE tuned 配置（每个节点）

**关键**：MI308X 报 `cu_num=80`，但 aiter 的 glm5_1 tuned 配置写的是 `cu_num=304`。
必须把 cu_num 改成 80，否则所有 MoE GEMM 走 default（损失 20-40%）。

在每个节点上执行（通过任意 pod 的 hostPath `/data`）：

```bash
# 在每个 prefill/decode pod 里执行
src=/sgl-workspace/aiter/aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_glm5_1.csv
dst=/data/aiter_configs/a8w8_blockscale_tuned_fmoe_glm5_1_cu80.csv
mkdir -p /data/aiter_configs
head -1 $src > $dst
awk -F, 'NR>1{$2=80; print}' OFS=, $src >> $dst

# 验证
wc -l $dst
# 补 decode 形状 expert=256,topk=8（token 1..128），保留 257/9 给 prefill
python3 docker/rocm-mi308x-glm52-pd/scripts/merge_decode_fmoe_256_8.py $dst
```

### Step 2: 配置 chart values

编辑 `docker/rocm-mi308x-glm52-pd/chart/values.yaml`：

```yaml
# 镜像（已 push 到 mirror）
workerImage: mirrors.tencent.com/ti-platform/sglang-glm52-308x:v0825-gate-indexer
routerImage: mirrors.tencent.com/ti-platform/sglang-glm52-308x-pd-router:v0516-batch1-tok

# 节点 IP（替换成实际 IP）
prefills:
  - nodeName: node-21.151.225.144
    ip: "21.151.225.144"
  - nodeName: node-21.151.225.152
    ip: "21.151.225.152"
decodeNode: node-21.151.225.132
decodeIP: "21.151.225.132"

# API key
apiKey: sk-YOUR_API_KEY

# MoE tuned config（Step 1 生成的）
workerEnv:
  AITER_CONFIG_FMOE: /data/aiter_configs/a8w8_blockscale_tuned_fmoe_glm5_1_cu80.csv
  # ... 其他 env 见 values.yaml
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

- `speculative-num-steps 3, num-draft-tokens 4, eagle-topk 1`
- accept_length 2.8-3.1，accept_rate 0.6-0.7

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

## 参考

- [P0 MoE tuned 修复详情](../../../docs/pd_200k_cold_cache_fix_progress.md)
- [chart values.yaml](../chart/values.yaml) - 完整配置
- [live-1p1d README](live-1p1d/README.md) - 生产快照说明
