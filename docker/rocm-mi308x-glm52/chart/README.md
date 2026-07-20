# SGLang GLM-5.2 on AMD MI308X (gfx942) — Helm Chart

Deploys SGLang GLM-5.2-FP8 inference service on AMD MI308X nodes with
EAGLE speculative decoding (NEXTN), HiCache, and envoy gateway forwarding.

**For complete reproduction guide (build image → prepare nodes → deploy →
verify), see [`../REPRODUCE.md`](../REPRODUCE.md).**

## Current Deployment (2026-07-20)

- **Image**: `mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3`
- **Base**: `lmsysorg/sglang-rocm:v0.5.15.post1-rocm720-mi30x-20260718`
- **Branch**: `fix/eagle-decode-coredump-mi308x` (based on `origin/main` @ `50c118704a`)
- **Fixes**: 3 ROCm-specific issues + 6 upstream EAGLE fixes + PR #31478

### EAGLE Coredump Fix (2026-07-20)

The original `xgmi-opt-0716d` image (sglang `b76dd0be`, 2026-07-10) suffered
8-rank GPU coredump when EAGLE decode ran after >1024-token prefill. Three
ROCm-specific issues were identified and fixed:

1. **NameError** — `scale_head_gate_graph`/`logits_head_gate_graph` defined
   inside `if _is_cuda:` block, never defined on ROCm. Fix: hoist to module
   level.
2. **AssertionError** — BCG split-op dispatch gated by `is_cuda()` on ROCm.
   Fix: use `tc_piecewise` prefill backend (upstream default for non-CUDA).
3. **Host OOM** — DSA indexer (57 GB/rank) + HiCache ratio=4 (248 GB/rank)
   exceeded node RAM. Fix: reduce `hicacheRatio` to 2.

See [REPRODUCE.md → EAGLE Coredump Fix Summary](../REPRODUCE.md#eagle-coredump-fix-summary)
for full details.

## Quick Start

```bash
# 1. Build and push image (see REPRODUCE.md for full steps)
docker build -f docker/rocm-mi308x-glm52/Dockerfile \
  -t mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3 .
docker push mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3

# 2. Label nodes
kubectl label node node-21.151.225.152 accelerator=amd-gpu sglang-model=ready --overwrite
kubectl label node node-21.151.225.172 accelerator=amd-gpu sglang-model=ready --overwrite

# 3. Deploy both workers
helm install sglang-glm52-2tp8 \
  docker/rocm-mi308x-glm52/chart/ -n kube-system \
  -f docker/rocm-mi308x-glm52/chart/values-glm52-2tp8.yaml

helm install sglang-glm52-2tp8-w2 \
  docker/rocm-mi308x-glm52/chart/ -n kube-system \
  -f docker/rocm-mi308x-glm52/chart/values-glm52-2tp8-w2.yaml

# 4. Verify (wait 3-5 min for CUDA graph capture + HiCache allocation)
kubectl get pod -n kube-system -l app=sglang -o wide
curl -s http://21.151.225.152:30000/health  # → ok
curl -s http://21.151.225.172:30000/health  # → ok

# 5. Run verification script
bash docker/rocm-mi308x-glm52/scripts/verify-v14.sh 21.151.225.152
bash docker/rocm-mi308x-glm52/scripts/verify-v14.sh 21.151.225.172
```

## Multi-Node Deployment

```bash
# Deploy to specific node (override nodeSelector)
helm install sglang-glm52-308x-t1 docker/rocm-mi308x-glm52/chart/ -n kube-system \
  --set nodeName=node-21.151.225.132 --set gateway.enabled=false

helm install sglang-glm52-308x-t2 docker/rocm-mi308x-glm52/chart/ -n kube-system \
  --set nodeName=node-21.151.225.172 --set gateway.enabled=false
```

## Values Files

Per-deployment overrides live alongside `values.yaml` (chart defaults). Use
`-f` to layer them:

| File | Release | Topology | Notes |
|------|---------|----------|-------|
| `values.yaml` | — | chart defaults | 1M context, ratio=2, tc_piecewise |
| `values-glm52-2tp8.yaml` | `sglang-glm52-2tp8` | GZ, .152 | Primary: worker + router + gateway, OOM+HiCache fix |
| `values-glm52-2tp8-w2.yaml` | `sglang-glm52-2tp8-w2` | GZ, .172 | Secondary: worker only |
| `values-glm52-test.yaml` | `sglang-glm52-test` | GZ test, 144+132 | Legacy test env |
| `values-glm52-test-w2.yaml` | `sglang-glm52-test-w2` | GZ test, 132 | Legacy test worker |
| `values-prod.yaml` | `sglang-glm52-prod` | ZW prod (planned) | Router + gateway for ZW |
| `values-test.yaml` | `sglang-glm52-test` | Legacy | Superseded by values-glm52-test.yaml |

### `values-glm52-2tp8*.yaml` (current production for codex)

Applied fixes (2026-07-18 OOM + 2026-07-19 HiCache + 2026-07-20 EAGLE coredump):

```bash
# Deploy both workers
helm install sglang-glm52-2tp8    docker/rocm-mi308x-glm52/chart/ -n kube-system -f values-glm52-2tp8.yaml
helm install sglang-glm52-2tp8-w2 docker/rocm-mi308x-glm52/chart/ -n kube-system -f values-glm52-2tp8-w2.yaml

# Upgrade (after editing values)
helm upgrade sglang-glm52-2tp8    docker/rocm-mi308x-glm52/chart/ -n kube-system -f values-glm52-2tp8.yaml
helm upgrade sglang-glm52-2tp8-w2 docker/rocm-mi308x-glm52/chart/ -n kube-system -f values-glm52-2tp8-w2.yaml
```

Key overrides from chart defaults:
- `sglang.contextLength`: 1048576 → 524288 (512K, stability)
- `sglang.memFractionStatic`: 0.88 → 0.75 (OOM fix, leaves ~46GB/rank activation)
- `sglang.chunkedPrefillSize`: 32768 → 16384 (OOM fix, smaller peak activation)
- `sglang.prefillMaxRequests`: 32 → 4 (OOM fix, bound concurrent prefills)
- `sglang.scheduleConservativeness`: 0.5 → 1.0 (OOM fix, conservative scheduling)
- `sglang.cudaGraphBackendPrefill`: tc_piecewise (NOT breakable — ROCm)
- `sglang.hicacheRatio`: 2.0 → 2 (DSA indexer needs 57 GB/rank host RAM)
- `sglang.hicacheWritePolicy`: write_through → write_back (fixes host load-back=0)
- `sglang.watchdogTimeout`: 3600 → 1200 (faster fail on stuck detokenizer)
- `router.cacheThreshold`: 0.5 → 0.2 (more aggressive prefix routing)
- `router.balanceAbsThreshold`: 32 → 1 (tighter load balance band)
- `router.balanceRelThreshold`: 1.5 → 1.2 (tighter relative balance)

**Note:** After `helm upgrade`, the router Deployment may need a manual
`kubectl patch` to remove `kubernetes.io/hostname` from `nodeSelector`
(strategic merge doesn't delete keys). The router tolerations are captured
in `values-glm52-2tp8.yaml` so they survive upgrades.

## Access

| Method | URL |
|--------|-----|
| Direct W1 | `http://21.151.225.152:30000` |
| Direct W2 | `http://21.151.225.172:30000` |
| Router (LB) | `http://21.151.225.152:30001` |
| Envoy gateway | `http://glm52-2tp8.jmpti.woa.com` |
| In-cluster | `http://sglang-glm52-2tp8-sglang.kube-system:30000` |

## Build Image

See [REPRODUCE.md → Step 1: Build the Worker Image](../REPRODUCE.md#step-1-build-the-worker-image)
for the complete build process.

```bash
# Quick reference (from repo root)
docker build -f docker/rocm-mi308x-glm52/Dockerfile \
  -t mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3 .
docker push mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3
```

The Dockerfile:
1. FROM `lmsysorg/sglang-rocm:v0.5.15.post1-rocm720-mi30x-20260718` (official base)
2. COPY `python/sglang/` (upstream main + PR #31478 + hoist fix)
3. Verify all patches present (6 upstream fix markers + PR #31478)
4. COPY FlyDSL + Triton fp8_mqa_logits patches (BLOCK_KV=64 for gfx942)
5. Set 23 ENV vars (ROCm optimized, EAGLE stable)
6. COPY `start_server.sh` (entrypoint with tc_piecewise prefill)

## Configuration

Key values in `values.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | `mirrors.tencent.com/ti-platform/sglang-glm52-308x` | SGLang image |
| `tag` | `latest` | Image tag (override to `fix-eagle-coredump-v3`) |
| `nodeName` | `""` | Pin to specific node (overrides nodeSelector) |
| `hostNetwork` | `true` | Required for TKE ENI network mode |
| `port` | `30000` | SGLang server port |
| `nodeSelector` | `accelerator=amd-gpu, sglang-model=ready` | Target AMD nodes with model |
| `model.path` | `/data/model/glm52-fp8` | Model weights path |
| `sglang.tpSize` | `8` | Tensor parallel size (all 8 GPUs) |
| `sglang.contextLength` | `1048576` | 1M context (override to 524288 for stability) |
| `sglang.memFractionStatic` | `0.88` | Memory fraction for KV cache (override to 0.75) |
| `sglang.kvCacheDtype` | `fp8_e4m3` | FP8 KV cache |
| `sglang.cudaGraphBackendPrefill` | `tc_piecewise` | **NOT breakable** (ROCm) |
| `sglang.speculativeAlgorithm` | `NEXTN` | EAGLE MTP speculative decoding |
| `sglang.speculativeNumSteps` | `3` | MTP speculative steps |
| `sglang.enableHierarchicalCache` | `true` | HiCache (GPU + CPU RAM) |
| `sglang.hicacheRatio` | `2.0` | CPU L2 cache ratio |
| `gateway.enabled` | `true` | Enable envoy gateway HTTPRoute |
| `gateway.hostname` | `glm52-308x-test.jmpti.woa.com` | External hostname |

## Readiness Probe

The readiness/liveness probe timeout is 10s (not 1s — HiCache health check
exceeds 1s). After `helm upgrade`, pods must be deleted to apply the new
probe timeout:

```bash
kubectl delete pod -n kube-system sglang-glm52-2tp8-sglang-0 --grace-period=0 --force
kubectl delete pod -n kube-system sglang-glm52-2tp8-w2-sglang-0 --grace-period=0 --force
```

## Performance (fix-eagle-coredump-v3, verified 2026-07-20)

| Metric | W1 (.152) | W2 (.172) |
|--------|-----------|-----------|
| short_c32 throughput | 153 tok/s | 146 tok/s |
| short_c128 throughput | 242 tok/s | 252 tok/s |
| mid_c2048 throughput (original coredump trigger) | 166 tok/s | 184 tok/s |
| long_c8192 throughput | 108 tok/s | 76 tok/s |
| EAGLE accept rate | 0.60 | 0.858 |
| EAGLE accept length | 2.80 | 3.575 |
| Peak decode throughput (concurrency 9) | 444 tok/s | 437 tok/s |
| HiCache host tokens | 1.85M | 1.85M |

All 4 benchmark scenarios pass with 100% success rate. No coredumps.

## Uninstall

```bash
helm uninstall sglang-glm52-2tp8 -n kube-system
helm uninstall sglang-glm52-2tp8-w2 -n kube-system
```
