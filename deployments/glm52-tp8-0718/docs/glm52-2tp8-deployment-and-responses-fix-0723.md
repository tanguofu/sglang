# GLM-5.2 2tp8 部署优化与 Responses API 兼容性修复完整记录

> **文档版本**: 2026-07-24
> **作者**: guofutan
> **关联分支**: `tanguofu/sglang:sglang-2tp8-0723`, `ti-cloud-teamai:glm52-tp8-0718`

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [部署架构](#2-部署架构)
3. [Benchmark 参数对齐](#3-benchmark-参数对齐)
4. [Native Router 部署](#4-native-router-部署)
5. [Responses API 兼容性修复](#5-responses-api-兼容性修复)
6. [Benchmark 结果](#6-benchmark-结果)
7. [故障排查与已知问题](#7-故障排查与已知问题)
8. [Git 提交记录](#8-git-提交记录)
9. [文件结构](#9-文件结构)

---

## 1. 背景与目标

GLM-5.2 模型在 MI308X GPU 集群上的 2tp8 部署经历了多个阶段的优化:

1. **EAGLE coredump 修复** (2026-07-20): 修复 EAGLE speculative decode 在 MI308X 上的 coredump 问题
2. **双 worker 合并** (2026-07-21): 将 W1+W2 合并为单 STS (replicas=2)
3. **Benchmark 参数对齐** (2026-07-23): 将 2tp8 配置与经过验证的 1tp8 benchmark-optimized 配置对齐
4. **Native Router 部署** (2026-07-23): 消除 Python proxy,使用 Rust router 原生支持 /v1/responses + /v1/messages
5. **Responses API 兼容性修复** (2026-07-23): 修复 3 个 BUG (stream:null 400, reasoning_tokens=0, usage 字段格式错误)

本文档完整记录第 3-5 阶段的过程和配置。

---

## 2. 部署架构

### 2.1 拓扑

```
                    ┌─────────────────────────────┐
                    │      HTTPS Gateway           │
                    │  glm52-2tp8.jmpti.woa.com    │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Rust Router (sgl-model-    │
                    │   gateway v0.3.2 patched)    │
                    │   - cache_aware LB           │
                    │   - /v1/chat/completions     │
                    │   - /v1/responses (native)   │
                    │   - /v1/messages (native)    │
                    └──┬───────────────────────┬──┘
                       │                       │
              ┌────────▼────────┐    ┌────────▼────────┐
              │  Worker 0       │    │  Worker 1       │
              │  21.151.225.172 │    │  21.151.225.152 │
              │  8x MI308X      │    │  8x MI308X      │
              │  TP=8           │    │  TP=8           │
              │  fix-eagle-     │    │  fix-eagle-     │
              │  coredump-v3    │    │  coredump-v3    │
              └─────────────────┘    └─────────────────┘
```

### 2.2 关键组件

| 组件 | 名称 | 说明 |
|------|------|------|
| STS | `sglang-glm52-2tp8-sglang` | StatefulSet, replicas=2 |
| Router | `sglang-glm52-2tp8-router` | Deployment, 1 replica |
| Service | `sglang-glm52-2tp8-router` | ClusterIP 9.165.99.83:30080 |
| ConfigMap | `sglang-glm52-2tp8-responses-fix` | BUG 修复文件覆盖 |
| ConfigMap | `sglang-glm52-2tp8-native-entrypoint` | Router 启动脚本 |
| ConfigMap | `aiters-tuned-gemm` | GEMM 调优 CSV |
| Image | `mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3` | 含 PR #31478 |
| Router Wheel | `sglang_router-0.3.2-cp38-abi3-manylinux_2_34_x86_64.whl` | hostPath 挂载 |

### 2.3 节点信息

| 节点 | IP | GPU | 镜像 | 用途 |
|------|-----|-----|------|------|
| node-144 | 21.151.225.144 | - | - | Router pod 所在节点 |
| node-152 | 21.151.225.152 | 8x MI308X | img-ebtth3fd (TOS 4.4) | Worker 1 (pod-1) |
| node-172 | 21.151.225.172 | 8x MI308X | img-ebtth3fd (TOS 4.4) | Worker 0 (pod-0) |

---

## 3. Benchmark 参数对齐

### 3.1 对齐策略

将 2tp8 配置与经过充分验证的 1tp8 benchmark-optimized 配置对齐,分为 3 个优先级:

- **P0 (Critical)**: 影响稳定性的必须修复项
- **P1 (Performance)**: 影响性能的参数调优
- **P2 (Concurrency)**: 并发参数收敛

### 3.2 完整参数变更表

| 参数 | 旧值 | 新值 | 优先级 | 原因 |
|------|------|------|--------|------|
| `tag` | toolchoice-fix-0721 | fix-eagle-coredump-v3 | P0 | 含 PR #31478 NCCL deadlock fix |
| `eaglePatch.enabled` | true | false | P0 | v3 镜像已内置 PR #31478 |
| `cudaGraphBackendPrefill` | breakable | tc_piecewise | P0 | breakable 触发 dsa_indexer.py assert crash |
| `memFractionStatic` | 0.75 | 0.88 | P1 | OOM 是旧镜像 bug,非内存压力;0.88 与 1tp8 对齐 |
| `chunkedPrefillSize` | 16384 | 32768 | P1 | 与 1tp8 对齐,更大 chunk = 更高 prefill 吞吐 |
| `prefillMaxRequests` | 8 | 32 | P1 | 与 1tp8 对齐 |
| `scheduleConservativeness` | 1.0 | 0.5 | P1 | 与 1tp8 对齐,更激进调度 |
| `watchdogTimeout` | 1200 | 3600 | P1 | 长 prefill 需要更长超时 |
| `hicacheRatio` | 4 | 2 | P1 | DSA indexer host 内存分配减少 |
| `hicacheWritePolicy` | write_back | write_through_selective | P1 | 稳定性优先 |
| `maxRunningRequests` | 48 | 32 | P2 | 与 1tp8 对齐 |
| `cudaGraphMaxBsDecode` | 32 | 16 | P2 | 与 1tp8 对齐 |
| `cudaGraphBsDecode` | 1-16,20,24,32 | 1-16 | P2 | 移除不必要的 graph capture |

### 3.3 最终 STS 启动参数

```bash
python3 -m sglang.launch_server \
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
  --enable-metrics --skip-server-warmup \
  --watchdog-timeout 3600 --log-level info
```

### 3.4 Router 启动参数

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://21.151.225.152:30000 http://21.151.225.172:30000 \
  --policy cache_aware \
  --host 0.0.0.0 --port 30080 \
  --cache-threshold 0.2 \
  --balance-abs-threshold 1 \
  --balance-rel-threshold 1.2
```

---

## 4. Native Router 部署

### 4.1 架构变更

**之前**: Rust router → Python aiohttp proxy (namespace filter + tool type sanitize) → SGLang worker

**之后**: Rust router (patched) → SGLang worker

消除了 Python proxy 中间层,减少延迟和维护成本。

### 4.2 Router Wheel 补丁

基于 `sgl-model-gateway` commit `719a4fcac9`,包含 3 个关键补丁:

1. **`unwrap_namespace_tools`**: 展开 Codex namespace tool 容器 (Rust router 只支持 4 种 tool type)
2. **`ensure_stream_default`**: 当 client 省略 stream 字段时,默认设置 `stream: false`
3. **`build_messages_routing_text`**: 使用完整对话前缀进行 cache_aware 路由

### 4.3 Wheel 部署方式

```yaml
# Router Deployment volumeMounts
- name: wheel-cache
  mountPath: /wheel-cache
  readOnly: true
# Node hostPath
- hostPath:
    path: /tmp/wheel-cache
    type: Directory
  name: wheel-cache
```

```bash
# Entrypoint script (/opt/entrypoint-native.sh)
WHEEL_CACHE="/wheel-cache/sglang_router-0.3.2-cp38-abi3-manylinux_2_34_x86_64.whl"
if [ -f "${WHEEL_CACHE}" ]; then
  pip install --force-reinstall --no-deps "${WHEEL_CACHE}"
fi
exec python3 -m sglang_router.launch_router "$@"
```

---

## 5. Responses API 兼容性修复

### 5.1 BUG 列表

在 benchmark 和兼容性 review 中发现 3 个 BUG:

| BUG | 严重度 | 现象 | 根因 |
|-----|--------|------|------|
| BUG 1 | P1 | `/v1/responses` 不带 stream 字段返回 400 | Router 转发 `stream: null`,worker 的 ChatCompletionRequest 要求 bool |
| BUG 2 | P2 | `reasoning_tokens` 始终为 0 | Responses API 未传 `require_reasoning` 给 GenerateReqInput |
| BUG 3 | P2 | 非流式 usage 用 `prompt_tokens`/`completion_tokens` | 非流式路径直接返回 UsageInfo,未转换为 Responses API 格式 |

### 5.2 修复方案: ConfigMap 文件覆盖

由于构建完整 ROCm Docker 镜像耗时过长 (2-3 小时),采用 ConfigMap + subPath 覆盖方式:

```yaml
# STS strategic merge patch
spec:
  template:
    spec:
      volumes:
        - name: responses-fix
          configMap:
            name: sglang-glm52-2tp8-responses-fix
            defaultMode: 420
      containers:
        - name: sglang
          volumeMounts:
            - name: responses-fix
              mountPath: /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py
              subPath: serving_responses.py
              readOnly: true
            - name: responses-fix
              mountPath: /sgl-workspace/sglang/python/sglang/srt/entrypoints/context.py
              subPath: context.py
              readOnly: true
```

### 5.3 BUG 1 修复: stream:null → 400

**文件**: `serving_responses.py` line 189

```python
# Fix BUG 1: Router may forward stream:null when the client omits the
# stream field. ChatCompletionRequest requires stream as bool.
if request.stream is None:
    request.stream = False
```

**验证**: `/v1/responses` 不带 stream 字段 → HTTP 200 (之前 400)

### 5.4 BUG 2 修复: reasoning_tokens 始终为 0

**根因分析**:

Chat Completions API 在 `serving_chat.py:764` 传递 `require_reasoning=True`:
```python
require_reasoning = self._get_reasoning_from_request(request)
adapted_request = GenerateReqInput(
    ...
    require_reasoning=require_reasoning,
)
```

Scheduler 在 `batch_result_processor.py:918` 根据 `require_reasoning` 决定是否计数:
```python
def _maybe_update_reasoning_tokens(self, req, next_token_id):
    think_end_id = self.model_config.think_end_id
    if req.require_reasoning and think_end_id is not None:
        req.update_reasoning_tokens(next_token_id, think_end_id)
```

Responses API 没有传递此参数,导致 scheduler 跳过计数。

**修复**: `serving_responses.py` line 297

```python
# Fix BUG 2: Pass require_reasoning so the scheduler counts
# reasoning tokens. Without this, reasoning_tokens is always 0
# in the Responses API (the scheduler skips the counting path).
require_reasoning = self._is_thinking_enabled_for_request(request)

# ... 在 GenerateReqInput 中传递
adapted_request = GenerateReqInput(
    ...
    require_reasoning=require_reasoning,
)
```

同时在流式 tool-call 重试路径中也传递:
```python
# line 2418
adapted_request = GenerateReqInput(
    ...
    require_reasoning=adapted_request.require_reasoning,
)
```

**附加修复**: `context.py` line 100-101 (HarmonyContext 路径,GLM 不使用但补全):
```python
if "reasoning_tokens" in meta_info:
    self.num_reasoning_tokens += meta_info["reasoning_tokens"]
```

**验证**: reasoning prompt → `reasoning_tokens: 142` (之前 0)

### 5.5 BUG 3 修复: 非流式 usage 字段格式

**根因**: 非流式路径返回 `UsageInfo` (Chat Completions 格式),流式路径正确转换为 Responses API 格式。

**修复**: `serving_responses.py` lines 622-639

```python
# Convert usage from Chat Completions format (UsageInfo) to Responses
# API format (input_tokens/output_tokens/output_tokens_details) so
# non-streaming responses match the streaming path and the OpenAI
# Responses API spec.
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

**验证**:

修复前:
```json
{"usage": {"prompt_tokens": 29, "completion_tokens": 198, "reasoning_tokens": 0}}
```

修复后:
```json
{
  "usage": {
    "input_tokens": 29,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 144,
    "output_tokens_details": {"reasoning_tokens": 142},
    "total_tokens": 173
  }
}
```

---

## 6. Benchmark 结果

### 6.1 修复前 Benchmark (2026-07-23 上午)

| 场景 | 结果 |
|------|------|
| Codex CLI simple prompt | 50.5s, 13,337 tokens |
| Codex CLI tool call | 34.7s, 13,187 tokens |
| Codex CLI reasoning=low | 23.6s, 29 tokens |
| /v1/responses streaming TTFT | 0.75s |
| /v1/responses streaming TPOT | 36.1ms/event |
| 10 concurrent | 70.0 tok/s aggregate |
| 20 concurrent | 187.7 tok/s aggregate, all 200 OK |

### 6.2 修复后 Benchmark (2026-07-23 下午)

#### 单请求延迟 (非流式, 100 max_tokens)

| API | 延迟 | tokens | reasoning | 吞吐 |
|-----|------|--------|-----------|------|
| Chat Completions | 1.27-1.44s | 100 | 95-103 | 69-79 tok/s |
| Responses API | 1.24-1.38s | 98 | 95-99 | 71-79 tok/s |

两个 API 延迟基本一致 (误差在噪声范围内)。

#### 并发测试 (Responses API, 150 max_tokens)

| 并发数 | 墙上时间 | 总 tokens | 总 reasoning | 聚合吞吐 |
|--------|---------|-----------|-------------|---------|
| 5 | 12.63s | 740 | 744 | 58.6 tok/s |
| 10 | 24.76s | 1,480 | 1,489 | 59.8 tok/s |
| 20 | 27.53s | 2,960 | 2,987 | 107.5 tok/s |

#### 并发测试 (Chat Completions, 150 max_tokens)

| 并发数 | 墙上时间 | 总 tokens | 总 reasoning | 聚合吞吐 |
|--------|---------|-----------|-------------|---------|
| 5 | 3.71s | 750 | 755 | 201.9 tok/s |
| 10 | 3.74s | 1,500 | 1,509 | 401.2 tok/s |
| 20 | 3.94s | 3,000 | 3,025 | 761.7 tok/s |

### 6.3 性能差异分析

Responses API 在并发下显著慢于 Chat Completions (107.5 vs 761.7 tok/s @ 20 并发)。可能原因:

1. **冷缓存**: 刚重启后的首次请求, prefix 缓存为空
2. **代码路径开销**: `serving_responses.py` 的预处理比 `serving_chat.py` 更重
3. **请求构造差异**: Responses API 的 input 解析比 messages 解析复杂

如 Codex 用户报告延迟问题,需进一步排查。

---

## 7. 故障排查与已知问题

### 7.1 已知限制

| 功能 | 状态 | 说明 |
|------|------|------|
| `previous_response_id` | 不支持 | SGLang 无服务端状态;Codex 发送完整对话 |
| `store: true` | 半支持 | 返回 true 但不实际存储 |
| Responses API 并发性能 | 较慢 | 比 Chat Completions 慢 7x @ 20 并发 |

### 7.2 常见问题

#### Q: Pod 启动后 30 分钟仍 0/1 Running

**A**: MI308X 8-GPU 节点冷启动需要:
- Model shard 加载: ~15s
- Aiter JIT 编译: ~10-20 min (首次冷缓存)
- DSA indexer host 内存分配: ~2 min
- CUDA graph capture: ~2 min

总计约 15-30 分钟。后续重启如果 JIT cache 存在则更快。

#### Q: Router 报 401 Unauthorized

**A**: Worker 配置了 `--api-key`,Router 转发时需要携带。确认:
```bash
# Worker API key
API_KEY=sk-46faecc9d0bc4dcd9db6a15c73ae91c8
```

#### Q: /v1/responses 返回 400 "stream should be a valid boolean"

**A**: 这是 BUG 1。确认 ConfigMap `sglang-glm52-2tp8-responses-fix` 已挂载:
```bash
kubectl get sts sglang-glm52-2tp8-sglang -o jsonpath='{.spec.template.spec.volumes}' | python3 -m json.tool | grep responses-fix
```

#### Q: reasoning_tokens 为 0

**A**: 这是 BUG 2。确认 `require_reasoning` 被正确传递。检查 `serving_responses.py` line 297:
```bash
kubectl exec sglang-glm52-2tp8-sglang-0 -- grep -n "require_reasoning" /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py
```

### 7.3 验证命令

```bash
# 1. 检查 pod 状态
kubectl get pods -n kube-system | grep sglang-glm52-2tp8

# 2. 检查 ConfigMap 挂载
kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- ls -la /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py

# 3. 测试 BUG 1 (无 stream 字段)
kubectl exec -n kube-system sglang-glm52-2tp8-router-676f895454-q2gv8 -- python3 -c "
import urllib.request, json
data = json.dumps({'model':'default','input':'Hi'}).encode()
req = urllib.request.Request('http://localhost:30080/v1/responses', data=data,
    headers={'Content-Type':'application/json','Authorization':'Bearer sk-46faecc9d0bc4dcd9db6a15c73ae91c8'})
resp = urllib.request.urlopen(req, timeout=300)
print('HTTP', resp.status)
"

# 4. 测试 BUG 2+3 (reasoning_tokens 和 usage 格式)
kubectl exec -n kube-system sglang-glm52-2tp8-router-676f895454-q2gv8 -- python3 -c "
import urllib.request, json
data = json.dumps({'model':'glm-5.2','input':'What is 15+27? Think step by step.','stream':False,'max_output_tokens':200}).encode()
req = urllib.request.Request('http://localhost:30080/v1/responses', data=data,
    headers={'Content-Type':'application/json','Authorization':'Bearer sk-46faecc9d0bc4dcd9db6a15c73ae91c8'})
resp = urllib.request.urlopen(req, timeout=300)
body = json.loads(resp.read())
print(json.dumps(body.get('usage',{}), indent=2))
"
```

---

## 8. Git 提交记录

### 8.1 sglang 仓库 (`tanguofu/sglang:sglang-2tp8-0723`)

```
c5be37de74 feat(deployments): align 2tp8 config with 1tp8 benchmark-optimized params
d0a1b9e593 fix(responses): fix 3 BUGs in /v1/responses API
719a4fcac9 feat(router): eliminate Python proxy, native /v1/responses + /v1/messages
b220d6985d fix(proxy): preserve tool call id and name in ResponsesStreamConverter
3260025baa docs(wiki): add TOS 4.4 driver installation and hardware wiki
7512b9be8d feat(deployments): add GLM-5.2 2tp8 deployment configs and scripts
```

### 8.2 ti-cloud-teamai 仓库 (`ti-cloud-teamai:glm52-tp8-0718`)

```
ba23712 feat(glm52-2tp8): align benchmark-optimized params with 1tp8 reference
3218f03 docs(deployments): add precision-preserving optimization plan + P2P verification
468182a docs(deployments): add deployment & kernel operator analysis
```

### 8.3 关键 Commit 详情

#### `d0a1b9e593` - fix(responses): fix 3 BUGs in /v1/responses API

```
BUG 1: Router forwards stream:null when field omitted → 400
  Fix: Guard in serving_responses.py: if request.stream is None, set False

BUG 2: reasoning_tokens always 0 in Responses API
  Root cause: Responses API didn't pass require_reasoning to GenerateReqInput,
  so scheduler never invoked _maybe_update_reasoning_tokens()
  Fix: Compute require_reasoning via _is_thinking_enabled_for_request() and
  pass to GenerateReqInput (both initial and tool-call retry paths)
  Also fix HarmonyContext.append_output to read reasoning_tokens from meta_info

BUG 3: Non-streaming usage uses Chat Completions field names
  Non-streaming returned prompt_tokens/completion_tokens (UsageInfo)
  Streaming correctly used input_tokens/output_tokens
  Fix: Convert usage to Responses API format before returning, mirroring
  the streaming path conversion
```

#### `c5be37de74` - feat(deployments): align 2tp8 config with 1tp8 benchmark-optimized params

```
P0 critical fixes:
  - tag: fix-eagle-coredump-v3 (PR #31478 baked in)
  - eaglePatch.enabled: false (no runtime patch needed)
  - cudaGraphBackendPrefill: tc_piecewise (DSA indexer crash fix)

P1 performance params (aligned from 1tp8):
  - memFractionStatic: 0.75 → 0.88
  - chunkedPrefillSize: 16384 → 32768
  - prefillMaxRequests: 8 → 32
  - scheduleConservativeness: 1.0 → 0.5
  - watchdogTimeout: 1200 → 3600
  - hicacheRatio: 4 → 2
  - hicacheWritePolicy: write_back → write_through_selective

P2 concurrency convergence:
  - maxRunningRequests: 48 → 32
  - cudaGraphMaxBsDecode: 32 → 16
  - cudaGraphBsDecode: removes 20/24/32
```

---

## 9. 文件结构

### 9.1 源代码 (sglang worktree)

```
sglang-worktree-2tp8-0723/
├── python/sglang/srt/entrypoints/
│   ├── openai/
│   │   └── serving_responses.py    # BUG 1+2+3 修复
│   └── context.py                  # HarmonyContext reasoning_tokens 修复
├── sgl-model-gateway/
│   └── src/routers/
│       ├── openai/responses/
│       │   └── utils.rs             # ensure_stream_default (router 侧)
│       └── http/router.rs           # route_responses 路由
├── docker/rocm-mi308x-glm52/
│   └── chart/
│       ├── templates/
│       │   └── sglang-statefulset.yaml  # eaglePatch 条件化
│       └── values-glm52-2tp8-merged.yaml # 13 参数对齐
└── deployments/glm52-tp8-0718/
    └── configs/pd-manifests/
        └── sglang-glm52-2tp8-values.yaml # 部署 values
```

### 9.2 部署配置 (ti-cloud-teamai)

```
ti-cloud-teamai/deployments/glm52-tp8-0718/
├── configs/
│   ├── pd-manifests/
│   │   ├── sglang-glm52-2tp8-values.yaml
│   │   ├── sglang-glm52-2tp8-w2-values.yaml
│   │   └── ab87-sglang-0.yaml
│   └── router/
│       ├── namespace_filter_proxy.py  # (已废弃,被 native router 替代)
│       └── sglang_router-0.3.2-cp38-abi3-manylinux_2_34_x86_64.whl
├── docs/
│   ├── deployment_kernel_analysis_0718.md
│   ├── precision_preserving_optimization_plan_0718.md
│   ├── stream_compat_matrix.md
│   ├── bug_v1_responses_nonstreaming_400.md
│   └── glm52-2tp8-deployment-and-responses-fix-0723.md  # 本文档
├── scripts/
├── results/
│   └── long-agent-context-bench-ttft.json
└── README.md
```

### 9.3 Kubernetes 资源

```
kube-system namespace:
├── sts/sglang-glm52-2tp8-sglang (replicas=2)
├── deploy/sglang-glm52-2tp8-router (replicas=1)
├── svc/sglang-glm52-2tp8-router (ClusterIP 9.165.99.83:30080)
├── svc/sglang-glm52-2tp8-sglang (ClusterIP 9.165.48.178:30000)
├── svc/sglang-glm52-2tp8-sglang-headless
├── configmap/sglang-glm52-2tp8-responses-fix     # BUG 修复文件
├── configmap/sglang-glm52-2tp8-native-entrypoint # Router 启动脚本
└── configmap/aiters-tuned-gemm                   # GEMM 调优
```

---

## 附录 A: 完整 values 文件

### `sglang-glm52-2tp8-values.yaml`

```yaml
# GLM-5.2 2tp8 merged chart values (W1 + W2, replicas=2)
# Synced 2026-07-23: Aligned with 1tp8 benchmark-optimized config.
aitersTunedGemm:
  configMapName: aiters-tuned-gemm
  enabled: true
eaglePatch:
  enabled: false
image: mirrors.tencent.com/ti-platform/sglang-glm52-308x
imagePullSecret: ""
imagePullSecrets:
  - name: ti-platform-registry
router:
  enabled: true
  policy: cache_aware
  cacheThreshold: "0.2"
  balanceAbsThreshold: "1"
  balanceRelThreshold: "1.2"
  workerUrls:
    - http://21.151.225.152:30000
    - http://21.151.225.172:30000
sglang:
  chunkedPrefillSize: 32768
  contextLength: "524288"
  cudaGraphBackendPrefill: tc_piecewise
  cudaGraphBsDecode: 1 2 3 4 5 6 7 8 9 10 12 16
  cudaGraphBsPrefill: 4 8 16 32
  cudaGraphMaxBsDecode: 16
  enableAiterAllreduceFusion: true
  enableFusedQkNormRope: true
  enableHierarchicalCache: true
  enableHicache: true
  enableMixedChunk: true
  hicacheIoBackend: direct
  hicacheMemLayout: page_first_direct
  hicacheRatio: 2.0
  hicacheWritePolicy: write_through_selective
  kvCacheDtype: fp8_e4m3
  logLevel: info
  maxPrefillTokens: 32768
  maxRunningRequests: 32
  memFractionStatic: 0.88
  numaNode: 0 0 0 0 1 1 1 1
  ppSize: 1
  prefillMaxRequests: 32
  reasoningParser: glm45
  scheduleConservativeness: 0.5
  skipServerWarmup: true
  speculativeAlgorithm: NEXTN
  speculativeEagleTopk: 1
  speculativeNumDraftTokens: 4
  speculativeNumSteps: 3
  toolCallParser: glm47
  tpSize: 8
  watchdogTimeout: 3600
shmSize: 32Gi
tag: fix-eagle-coredump-v3
tolerations:
  - key: amd-gpu
    operator: Exists
    effect: NoSchedule
```

---

## 附录 B: ConfigMap 创建命令

```bash
# 1. 创建 BUG 修复 ConfigMap
kubectl create configmap -n kube-system sglang-glm52-2tp8-responses-fix \
  --from-file=serving_responses.py=./serving_responses.py \
  --from-file=context.py=./context.py

# 2. 创建 Router entrypoint ConfigMap
kubectl create configmap -n kube-system sglang-glm52-2tp8-native-entrypoint \
  --from-file=entrypoint-native.sh=./entrypoint-native.sh

# 3. STS strategic merge patch (挂载 ConfigMap)
cat <<'EOF' | kubectl patch sts -n kube-system sglang-glm52-2tp8-sglang --type=strategic -p @-
{
  "spec": {
    "template": {
      "spec": {
        "volumes": [{
          "name": "responses-fix",
          "configMap": {
            "name": "sglang-glm52-2tp8-responses-fix",
            "defaultMode": 420
          }
        }],
        "containers": [{
          "name": "sglang",
          "volumeMounts": [
            {
              "name": "responses-fix",
              "mountPath": "/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py",
              "subPath": "serving_responses.py",
              "readOnly": true
            },
            {
              "name": "responses-fix",
              "mountPath": "/sgl-workspace/sglang/python/sglang/srt/entrypoints/context.py",
              "subPath": "context.py",
              "readOnly": true
            }
          ]
        }]
      }
    }
  }
}
EOF

# 4. 重启 STS 加载新配置
kubectl delete pod -n kube-system sglang-glm52-2tp8-sglang-0 sglang-glm52-2tp8-sglang-1
```

---

## 附录 C: 相关文档

| 文档 | 位置 |
|------|------|
| EAGLE Coredump 修复完整记录 | iWiki docid 4026586166 |
| NCCL Deadlock 修复 | iWiki docid 4027205265 |
| TOS 4.4 驱动安装 | sglang-worktree-2tp8-0723 分支 |
| HCCPA1 自定义镜像 | img-ebtth3fd (2026-07-19) |
| 精度保持优化计划 | `deployments/glm52-tp8-0718/docs/precision_preserving_optimization_plan_0718.md` |
| 部署内核分析 | `deployments/glm52-tp8-0718/docs/deployment_kernel_analysis_0718.md` |
