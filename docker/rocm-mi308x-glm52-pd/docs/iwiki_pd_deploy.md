# SGLang PD 分离部署 GLM-5.2-FP8 DSA 完整方案 (ts4 MI308X GDR)

> **验证日期**: 2026-07-22
> **节点**: ts4-pd-test (prefill: 21.234.170.159, decode: 21.234.171.87)
> **GPU**: AMD MI308X (gfx942) × 8 per node
> **模型**: GLM-5.2-FP8 (GlmMoeDsaForCausalLM + GlmMoeDsaForCausalLMNextN)
> **传输**: Mooncake RDMA + GPUDirect RDMA (HIP dmabuf), 8× bnxt_re_bond400G
> **状态**: ✅ 全部验证通过,HTTPS httproute 可用

---

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    HTTPS Request                                │
│  https://glm52-pd.jmpti.woa.com/v1/chat/completions             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              HTTPRoute (kube-system)                             │
│  glm52-pd.jmpti.woa.com → sglang-ts4-1p1d-router:30001          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│         sglang-ts4-1p1d-router (1 replica, hostNetwork)          │
│  sgl-model-gateway v0.3.2 (api-fix-0720)                         │
│  --pd-disaggregation mode                                        │
│  inject bootstrap_host/port/room → dual dispatch                 │
└──────────┬──────────────────────────────┬───────────────────────┘
           │ tokio::join! (concurrent)     │
           ▼                               ▼
┌──────────────────────┐         ┌──────────────────────┐
│  Prefill (node 159)   │         │  Decode (node 87)     │
│  SGLang 0.5.15.post1  │  KV     │  SGLang 0.5.15.post1  │
│  --disagg-mode prefill│ ──────> │  --disagg-mode decode │
│  port 30000           │  RDMA   │  port 30000           │
│  bootstrap 8998       │  GDR    │  EAGLE spec decode    │
│  8× bnxt_re_bond      │         │  8× bnxt_re_bond      │
└──────────────────────┘         └──────────────────────┘
```

### PD 请求流程

1. **Router** 接收 `/v1/chat/completions` 请求
2. **Router** 调用 `inject_bootstrap_into_value` 注入 `bootstrap_host`(prefill IP)、`bootstrap_port`(8998)、`bootstrap_room`(u64 随机数)到 JSON body
3. **Router** 通过 `tokio::join!` 并发发送同一 JSON body 给 prefill 和 decode
4. **Prefill** 执行 prefill,注册 bootstrap_room,通过 Mooncake RDMA 发送 KV cache
5. **Decode** 连接 prefill 的 bootstrap server (port 8998) 获取拓扑,通过 RDMA 接收 KV cache
6. **Decode** 执行 EAGLE speculative decoding 生成最终响应
7. **Router** 返回 decode 的响应给客户端

---

## 2. 关键组件版本

| 组件 | 镜像/版本 | 说明 |
|------|----------|------|
| SGLang Server | `mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3` | SGLang 0.5.15.post1.dev20260718,修复 EAGLE+DSA 兼容性 |
| PD Router | `mirrors.tencent.com/ti-platform/sglang-glm52-308x-pd-router:api-fix-0720` | sgl-model-gateway v0.3.2,支持 PD 分离 + /v1/responses |
| engine.so | md5 `c3873f547d5f0b46f20979009808fa91` | A-group GDR engine.so,支持 MC_DISABLE_HIP_TRANSPORT + HIP dmabuf |
| libbnxt_re | 238.1.138.5 | ts4 bnxt_re fw 238.1.138.0 需要,stock 235.2.86.0 不兼容 |
| 模型 | `/data/model/glm52-fp8` | GLM-5.2-FP8, DSA 模型 |

### 为什么用 fix-eagle-coredump-v3 而非 api-fix-0720-gdr

- `api-fix-0720-gdr`: SGLang 0.5.15.dev20260710 (Jul 10, pre-post1),EAGLE+DSA 产生垃圾输出(accept rate 0.00)
- `fix-eagle-coredump-v3`: SGLang 0.5.15.post1.dev20260718 (Jul 18, post1),修复 EAGLE+DSA 兼容性,accept rate ~0.53-0.62

### engine.so 替换原因

`fix-eagle-coredump-v3` 镜像自带的 engine.so (md5 `d192b213`) 忽略 `MC_DISABLE_HIP_TRANSPORT=1`,导致 GDR 失效。必须替换为 A-group `api-fix-0720` 的 engine.so (md5 `c3873f547d5f0b46f20979009808fa91`)。

### libbnxt_re 替换原因

`fix-eagle-coredump-v3` 镜像自带 libbnxt_re `235.2.86.0` (stock),不支持 ts4 节点的 bnxt_re fw `238.1.138.0`。stock 驱动被禁用为 `.so-inbox`。需要安装 `238.1.138.5` 并在 libibverbs 驱动目录创建符号链接。

---

## 3. Helm 部署配置

### 3.1 values-ts4.yaml 关键配置

```yaml
# 内部 PD 通信不能有 API key 认证(router → prefill/decode)
# 外部认证在 ingress/gateway 层强制
apiKey: ""

decode:
  hostIP: 21.234.171.87
  nodeName: node-21.234.171.87
  port: 30000

prefill:
  hostIP: 21.234.170.159
  nodeName: node-21.234.170.159
  port: 30000

router:
  hostIP: 0.0.0.0
  nodeName: node-21.234.170.159
  port: 30001
  prometheusPort: 19096
  image: mirrors.tencent.com/ti-platform/sglang-glm52-308x-pd-router
  tag: api-fix-0720

image:
  sglang: mirrors.tencent.com/ti-platform/sglang-glm52-308x
  tag: fix-eagle-coredump-v3
  pullPolicy: IfNotPresent
  pullSecret: tencent-registry

model:
  path: /data/model/glm52-fp8
  name: glm-5.2
  contextLength: 524288

sglang:
  attentionBackend: dsa          # DSA 模型必须用 dsa,不能用 aiter
  dsaPrefillBackend: tilelang    # DSA 子后端: tilelang (aiter 会 GPU 内存错误)
  dsaDecodeBackend: tilelang
  memFractionStatic: "0.85"
  kvCacheDtype: fp8_e4m3
  maxRunningRequests: "128"
  tpSize: 8
  ppSize: 1

decode:
  cudaGraphBsDecode: 1 2 3 4 5 6 7 8 9 10 12 16
  cudaGraphMaxBsDecode: "16"
  speculativeAlgorithm: NEXTN    # SGLang 内部转为 EAGLE
  speculativeEagleTopk: "1"
  speculativeNumDraftTokens: "4"
  speculativeNumSteps: "3"       # post1 代码支持 steps=3

rdma:
  bootstrapPort: 8998
  ibDevice: '{"0":"bnxt_re_bond0","1":"bnxt_re_bond1","2":"bnxt_re_bond2","3":"bnxt_re_bond3","4":"bnxt_re_bond4","5":"bnxt_re_bond5","6":"bnxt_re_bond6","7":"bnxt_re_bond7"}'
```

### 3.2 Router Deployment 关键配置

```yaml
spec:
  replicas: 1
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 0           # hostNetwork + hostPort 必须 maxSurge=0
      maxUnavailable: 1     # 旧 pod 先终止再启动新 pod
  template:
    spec:
      hostNetwork: true
      dnsPolicy: ClusterFirstWithHostNet
      nodeName: node-21.234.170.159
      containers:
      - name: router
        args:
        - |
          exec python3 -m sglang_router.launch_router \
            --pd-disaggregation \
            --prefill http://21.234.170.159:30000 8998 \
            --decode http://21.234.171.87:30000 \
            --host 0.0.0.0 --port 30001 \
            --prometheus-port 19096 \
            --model-path /data/model/glm52-fp8 \
            --log-level info
```

### 3.3 Prefill/Decode 启动脚本关键部分

```bash
# 1. 复制 GDR engine.so 从宿主机
if [ -f /data/mooncake-patched/engine.cpython-310-x86_64-linux-gnu.so ]; then
  cp /data/mooncake-patched/engine.cpython-310-x86_64-linux-gnu.so \
     /opt/venv/lib/python3.10/site-packages/mooncake/engine.cpython-310-x86_64-linux-gnu.so
fi

# 2. 安装 libbnxt_re 238.1.138.5 + 符号链接
if [ -f /data/mooncake-patched/libbnxt_re-rdmav34.so ]; then
  cp /data/mooncake-patched/libbnxt_re-rdmav34.so /usr/local/lib/libbnxt_re-rdmav34.so
  ln -sf /usr/local/lib/libbnxt_re-rdmav34.so \
         /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so
  ldconfig
fi

# 3. 启动 sglang server
exec python3 -m sglang.launch_server \
  --model-path /data/model/glm52-fp8 \
  --model-impl sglang --served-model-name glm-5.2 \
  --tp-size 8 --pp-size 1 --trust-remote-code \
  --host 0.0.0.0 --port 30000 \
  --context-length 524288 \
  --attention-backend dsa \
  --mem-fraction-static 0.85 \
  --kv-cache-dtype fp8_e4m3 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device '{"0":"bnxt_re_bond0",...}' \
  --disaggregation-bootstrap-port 8998
```

### 3.4 apiKey 条件化模板

```yaml
# prefill.yaml / decode.yaml 中 --api-key 条件化
            --model-impl sglang --served-model-name {{ .Values.model.name }} \
            {{- if .Values.apiKey }}
            --api-key {{ .Values.apiKey }} \
            {{- end }}
            --tp-size {{ .Values.sglang.tpSize }} ...
```

---

## 4. GDR (GPUDirect RDMA) 配置

### 4.1 GDR 前提条件

| 条件 | 状态 | 说明 |
|------|------|------|
| 内核 ≥ 5.12 (dmabuf) | ✅ 6.6.110-42.4.tl4 | kallsyms 有 dma_pci_p2pdma_supported + dma_buf_move_notify |
| engine.so USE_HIP_DMABUF=ON | ✅ md5 c3873f54 | 有 hsa_amd_portable_export_dmabuf + ibv_reg_dmabuf_mr |
| MC_DISABLE_HIP_TRANSPORT=1 | ✅ | 禁用 intra-node HIP IPC,不影响 RDMA dmabuf |
| 不设 SGLANG_PD_HOST_STAGING=1 | ✅ | host staging 会绕过 GDR |
| 不设 MOONCAKE_DISABLE_HIP_DMABUF | ✅ | |
| MC_GID_INDEX=3 | ✅ | RoCEv2 on bnxt_re bonds |

### 4.2 网络拓扑

每节点 8 个 bnxt_re_bond,每个 /30 子网,L3 路由通过交换机(HSRP):

| Bond | Prefill (159) | Decode (87) | Gateway |
|------|---------------|-------------|---------|
| bond0 | .110 | .122 | .109 / .121 |
| bond1 | .150 | .146 | .149 / .145 |
| bond2 | .118 | .138 | .117 / .137 |
| bond3 | .114 | .130 | .113 / .129 |
| bond4 | .74 | .10 | .73 / .9 |
| bond5 | .38 | .90 | .37 / .89 |
| bond6 | .82 | .98 | .81 / .97 |
| bond7 | .154 | .46 | .153 / .45 |

### 4.3 每条 Bond 静态路由

**问题**: Linux 路由有 catch-all `29.198.0.0/15 via bond7`。没有 per-bond 路由时,所有跨机流量走 bond7,但 RDMA QP 绑定到特定 bond → QP RTR 握手超时。

**修复**: 每条 bond 添加 /30 静态路由,确保 bondN 的 RDMA 流量留在 bondN:

```bash
# Prefill (159) 示例
ip route replace 29.199.73.120/30 via 29.199.73.109 dev bond0
ip route replace 29.199.73.144/30 via 29.199.73.149 dev bond1
# ... bond2-7 同理

# Decode (87) 示例
ip route replace 29.199.73.108/30 via 29.199.73.121 dev bond0
ip route replace 29.199.73.140/30 via 29.199.73.145 dev bond1
# ... bond2-7 同理
```

这些路由由 helm chart 的 `setup-routes` initContainer 自动安装。

### 4.4 GDR 验证日志

```
I0722 rdma_context.cpp:273] RDMA device: bnxt_re_bond0, LID: 0, GID: (GID_Index 3) 00:00:00:00:00:00:00:00:00:00:ff:ff:1d:c7:49:xx
I0722 transfer_engine_impl.cpp:389] installTransport, type=rdma
I0722 transfer_engine_impl.cpp:418] HIP transport disabled by MC_DISABLE_HIP_TRANSPORT=1
```

- 8 个 bnxt_re_bond 全部发现
- RDMA transport 活跃
- HIP transport 已禁用(使用 GDR dmabuf 路径)
- 无 host staging 回退

---

## 5. DSA + EAGLE 注意事项

### 5.1 DSA 模型必须用 dsa attention backend

GLM-5.2 是 `GlmMoeDsaForCausalLM` 模型,有 `index_topk` 字段:

- `--attention-backend dsa` ✅ — 正确处理 topk_indices sparse attention
- `--attention-backend aiter` ❌ — 静默忽略 topk_indices,跑 dense attention,输出垃圾

### 5.2 DSA 子后端用 tilelang 不用 aiter

- `--dsa-prefill-backend tilelang` ✅ — HIP auto-default,工作正常
- `--dsa-decode-backend tilelang` ✅
- `--dsa-*-backend aiter` ❌ — decode 阶段 GPU 内存错误崩溃 (Memory access fault by GPU node-N)

### 5.3 EAGLE + DSA 需要 post1 SGLang

- pre-post1 (Jul 10): EAGLE+DSA accept rate 0.00,输出垃圾
- post1 (Jul 18, fix-eagle-coredump-v3): EAGLE+DSA accept rate ~0.53-0.62,输出正确

### 5.4 NEXTN speculative decoding

SGLang 内部将 NEXTN 转为 EAGLE 执行:

```yaml
speculativeAlgorithm: NEXTN
speculativeEagleTopk: "1"
speculativeNumDraftTokens: "4"
speculativeNumSteps: "3"
```

---

## 6. 部署步骤

### 6.1 宿主机准备 (两个节点都执行)

```bash
# 1. 确认模型文件
ls /data/model/glm52-fp8/

# 2. 确认 GDR engine.so 和 libbnxt_re
ls /data/mooncake-patched/
# engine.cpython-310-x86_64-linux-gnu.so  (md5 c3873f547d5f0b46f20979009808fa91)
# libbnxt_re-rdmav34.so  (238.1.138.5)

# 3. 确认 bnxt_re 设备
ibv_devinfo | grep -E "hca_id|transport|port_state" | head -32

# 4. 确认路由 (由 initContainer 自动安装)
ip route show | grep bond
```

### 6.2 Helm 部署

```bash
# 部署 (先 prefill/decode,后 router)
helm install sglang-ts4-1p1d /tmp/sglang-chart \
  -n kube-system \
  -f /tmp/sglang-chart/values-ts4.yaml \
  --timeout 10m

# 等待 prefill/decode 就绪 (需要 15-20 分钟,包含权重加载 + aiter JIT 编译)
kubectl wait pod -n kube-system sglang-ts4-1p1d-prefill-0 --for=condition=ready --timeout=20m
kubectl wait pod -n kube-system sglang-ts4-1p1d-decode-0 --for=condition=ready --timeout=20m

# 启动 router (单独 scale,避免 hostNetwork 端口冲突)
kubectl scale deployment -n kube-system sglang-ts4-1p1d-router --replicas=1
kubectl wait pod -n kube-system -l app.kubernetes.io/component=router --for=condition=ready --timeout=5m
```

### 6.3 HTTPRoute 配置

```yaml
apiVersion: gateway.networking.k8s.io/v1beta1
kind: HTTPRoute
metadata:
  name: glm52-pd
  namespace: kube-system
spec:
  parentRefs:
  - name: prod-gateway
    namespace: kube-system
  hostnames:
  - glm52-pd.jmpti.woa.com
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /
    backendRefs:
    - name: sglang-ts4-1p1d-router
      port: 30001
```

---

## 7. 验证方法

### 7.1 基础功能验证

```bash
# 通过 HTTPS httproute
curl -sS -X POST "https://glm52-pd.jmpti.woa.com/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"What is 1+1?"}],"max_tokens":50,"temperature":0}'

# 通过 router 直连
kubectl exec -n kube-system <router-pod> -- curl -sS -X POST "http://127.0.0.1:30001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"What is 1+1?"}],"max_tokens":50,"temperature":0}'
```

### 7.2 GDR 验证

```bash
# 检查 HIP transport 已禁用
kubectl logs -n kube-system sglang-ts4-1p1d-prefill-0 | grep "HIP transport disabled"
kubectl logs -n kube-system sglang-ts4-1p1d-decode-0 | grep "HIP transport disabled"

# 检查 8 个 RDMA 设备
kubectl logs -n kube-system sglang-ts4-1p1d-prefill-0 | grep "RDMA device: bnxt_re_bond" | wc -l  # 应为 8

# 检查无 host staging
kubectl logs -n kube-system sglang-ts4-1p1d-prefill-0 | grep -i "host.staging"  # 应为空
```

### 7.3 PD 流程验证

```bash
# Prefill 应有 prefill batch 日志
kubectl logs -n kube-system sglang-ts4-1p1d-prefill-0 | grep "Prefill batch"

# Decode 应有 decode batch 日志和 EAGLE accept rate
kubectl logs -n kube-system sglang-ts4-1p1d-decode-0 | grep "Decode batch"
# 预期: accept len: ~2.5-3.0, accept rate: ~0.53-0.62

# Router 应有 dual dispatch 日志 (debug 模式)
kubectl logs -n kube-system <router-pod> | grep "Sending concurrent requests"
```

### 7.4 性能基准

```bash
# 在 router pod 内运行
python3 -c "
import requests, time, concurrent.futures
url='http://127.0.0.1:30001/v1/chat/completions'
payload={'model':'glm-5.2','messages':[{'role':'user','content':'Write a haiku'}],'max_tokens':100,'temperature':0}
for c in [1, 2, 4, 8, 16, 32]:
    start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
        results=list(ex.map(lambda i: requests.post(url,json=payload,timeout=120).status_code, range(c)))
    elapsed=time.time()-start
    ok=sum(1 for s in results if s==200)
    print(f'C={c}: {ok}/{c} success, {elapsed:.2f}s, {c/elapsed:.2f} req/s')
"
```

---

## 8. 常见问题与修复

### 8.1 Router Pod 爆炸 (NodePorts 状态)

**原因**: hostNetwork + hostPort,多个 router pod 调度到同一节点时端口冲突。

**修复**:
```bash
kubectl scale deployment -n kube-system sglang-ts4-1p1d-router --replicas=0
kubectl delete pods -n kube-system -l app.kubernetes.io/component=router --grace-period=0 --force
kubectl scale deployment -n kube-system sglang-ts4-1p1d-router --replicas=1
```

**预防**: 设置 `strategy.rollingUpdate.maxSurge: 0, maxUnavailable: 1`

### 8.2 Decode 返回 401 Unauthorized

**原因**: prefill/decode 配置了 `--api-key`,router 无法传递 API key。

**修复**: 设置 `apiKey: ""`,在模板中条件化 `--api-key` 参数。

### 8.3 EAGLE 输出垃圾 (accept rate 0.00)

**原因**: 使用 pre-post1 SGLang 镜像 (api-fix-0720-gdr, Jul 10)。

**修复**: 切换到 `fix-eagle-coredump-v3` 镜像 (post1, Jul 18)。

### 8.4 "No RDMA devices found" / "Found 0 HCAs"

**原因**: libbnxt_re 版本不匹配 ts4 bnxt_re firmware。

**修复**:
```bash
cp /data/mooncake-patched/libbnxt_re-rdmav34.so /usr/local/lib/
ln -sf /usr/local/lib/libbnxt_re-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so
ldconfig
```

### 8.5 QP RTR 超时 / "session not alive"

**原因**: Linux 路由 catch-all 导致 RDMA 流量走错 bond。

**修复**: 添加 per-bond /30 静态路由(由 initContainer 自动安装)。

### 8.6 "Disaggregated request received without bootstrap room id"

**原因**: 直接向 prefill/decode 发送请求,没有 bootstrap_room 字段。

**说明**: 这是正常的 PD 行为。必须通过 router 发送请求,router 会自动注入 bootstrap_room。

---

## 9. 性能数据 (2026-07-22)

| 并发 | 成功率 | 耗时 | 吞吐 | EAGLE accept rate |
|------|--------|------|------|-------------------|
| 1 | 100% | 2.89s | 0.35 req/s | ~0.62 |
| 2 | 100% | 4.06s | 0.49 req/s | ~0.60 |
| 4 | 100% | 21.42s | 0.19 req/s | ~0.55 |
| 8 | 100% | 21.67s | 0.37 req/s | ~0.55 |
| 16 | 100% | 23.29s | 0.69 req/s | ~0.53 |
| 32 | 100% | 51.03s | 0.63 req/s | ~0.53 |

> 注: 并发 4 耗时较高可能因首次 aiter JIT 编译开销。稳定后吞吐在 0.6-0.7 req/s。

---

## 10. 关键文件清单

| 文件 | 说明 |
|------|------|
| `/tmp/sglang-chart/values-ts4.yaml` | Helm values 配置 |
| `/tmp/sglang-chart/templates/prefill.yaml` | Prefill StatefulSet 模板 |
| `/tmp/sglang-chart/templates/decode.yaml` | Decode StatefulSet 模板 |
| `/tmp/sglang-chart/templates/router.yaml` | Router Deployment 模板 |
| `/tmp/sglang-chart/templates/_helpers.tpl` | 共享模板辅助函数 |
| `/data/mooncake-patched/engine.cpython-310-x86_64-linux-gnu.so` | GDR engine.so (md5 c3873f54) |
| `/data/mooncake-patched/libbnxt_re-rdmav34.so` | libbnxt_re 238.1.138.5 |
| `/data/model/glm52-fp8/` | GLM-5.2-FP8 模型文件 |

---

## 11. 参考链接

- SGLang PD 分离官方文档: https://docs.sglang.ai/references/disaggregated_serving.html
- Mooncake 传输引擎: https://github.com/kvcache-ai/Mooncake
- sgl-model-gateway (PD Router): `/sgl-model-gateway/src/routers/http/pd_router.rs`
- 已有文档: [SGLang 部署 GLM-5.2-FP8 DSA — Bug 修复、RDMA 部署与压测总结](https://iwiki.woa.com/p/4022389950)
- 已有文档: [GLM-5.2 MI308X EAGLE Coredump 修复完整部署文档](https://iwiki.woa.com/p/4026586166)
