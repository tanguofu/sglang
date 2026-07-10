# SGLang GLM-5.2 on AMD MI308X (gfx942) — Helm Chart

Deploys SGLang GLM-5.2-FP8 inference service on AMD MI308X nodes with envoy gateway forwarding.

## v16 Configuration (matching MI355X production)

- **DSA fused-store**: ENABLED (matching 355X, no is_hip guard)
- **Fused metadata copy/generation**: ENABLED (matching 355X)
- **BLOCK_KV=64**: gfx942 hardware limit (64KB shared memory, vs 355X's 80KB+)
- **AITER**: 9127c94a1 (base image, matching 355X exactly)
- **MTP**: steps=3, draft_tokens=4, eagle_topk=1
- **HiCache**: ratio=2.0, direct IO, page_first_direct (GPU VRAM + CPU RAM)
- **FP8 KV cache**: fp8_e4m3
- **1M context**: context_length=1048576

## Quick Start

```bash
# 1. Label target node (must have model weights at /data/model/glm52-fp8/)
kubectl label node node-21.151.225.144 sglang-model=ready --overwrite

# 2. Deploy
helm install sglang-glm52-308x docker/rocm-mi308x-glm52/chart/ -n kube-system

# 3. Verify
kubectl get pod -n kube-system -l app=sglang
kubectl logs -n kube-system sglang-glm52-308x-sglang-0 -f

# 4. Test
curl -s http://21.151.225.144:30000/health
curl -s http://glm52-308x.jmpti.woa.com/v1/models
```

## Multi-Node Deployment

```bash
# Deploy to specific node (override nodeSelector)
helm install sglang-glm52-308x-t1 docker/rocm-mi308x-glm52/chart/ -n kube-system \
  --set nodeName=node-21.151.225.132 --set gateway.enabled=false

helm install sglang-glm52-308x-t2 docker/rocm-mi308x-glm52/chart/ -n kube-system \
  --set nodeName=node-21.151.225.172 --set gateway.enabled=false
```

## Access

| Method | URL |
|--------|-----|
| Direct node | `http://21.151.225.144:30000` |
| Envoy gateway | `http://glm52-308x.jmpti.woa.com` |
| In-cluster | `http://sglang-glm52-308x-sglang.kube-system:30000` |

## Build Image (Full Reproduction)

```bash
# Clone the branch
git clone --branch 308x-glm52-opt https://github.com/tanguofu/sglang.git
cd sglang

# Build (requires Docker, ~15GB base image pull)
docker build -f docker/rocm-mi308x-glm52/Dockerfile \
  -t mirrors.tencent.com/ti-platform/sglang-glm52-308x:latest .

# Push to registry (requires /etc/hosts: 30.163.240.137 mirrors.tencent.com)
docker push mirrors.tencent.com/ti-platform/sglang-glm52-308x:latest
```

The Dockerfile:
1. FROM `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi30x-20260708` (official base)
2. COPY `python/sglang/` (0708-opt patches: 16 HIP compatibility patches)
3. Verify all patches present + 308X workarounds reverted (fused-store enabled)
4. COPY `patches/fp8_mqa_logits.py` (BLOCK_KV=64 for gfx942 64KB shared memory limit)
5. Set 21 ENV vars (matching 355X, PYTORCH_ROCM_ARCH=gfx942)
6. COPY `start_server.sh` (entrypoint with all launch params)

## Configuration

Key values in `values.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | `mirrors.tencent.com/ti-platform/sglang-glm52-308x` | SGLang image |
| `tag` | `latest` | Image tag |
| `nodeName` | `""` | Pin to specific node (overrides nodeSelector) |
| `hostNetwork` | `true` | Required for TKE ENI network mode |
| `port` | `30000` | SGLang server port |
| `nodeSelector` | `accelerator=amd-gpu, sglang-model=ready` | Target AMD nodes with model |
| `model.path` | `/data/model/glm52-fp8` | Model weights path |
| `sglang.tpSize` | `8` | Tensor parallel size (all 8 GPUs) |
| `sglang.contextLength` | `1048576` | 1M context |
| `sglang.memFractionStatic` | `0.88` | Memory fraction for KV cache |
| `sglang.kvCacheDtype` | `fp8_e4m3` | FP8 KV cache |
| `sglang.speculativeAlgorithm` | `NEXTN` | MTP speculative decoding |
| `sglang.speculativeNumSteps` | `3` | MTP speculative steps |
| `sglang.enableHierarchicalCache` | `true` | HiCache (GPU + CPU RAM) |
| `sglang.hicacheRatio` | `2.0` | CPU L2 cache ratio |
| `gateway.enabled` | `true` | Enable envoy gateway HTTPRoute |
| `gateway.hostname` | `glm52-308x.jmpti.woa.com` | External hostname |

## Readiness Probe

The readiness/liveness probe timeout is 10s (not 1s — HiCache health check
exceeds 1s). After `helm upgrade`, pods must be deleted to apply the new
probe timeout:

```bash
kubectl delete pod -n kube-system sglang-glm52-308x-sglang-0 --grace-period=0 --force
```

## Performance (v16, verified on test-0)

| Metric | v16 (MI308X) | 355X (MI355X) |
|--------|-------------|---------------|
| MTP accept_length | 3.31 | 3.71 |
| MTP accept_rate | 77.0% | 90.5% |
| Concurrent 4x | 253 tok/s | 252 tok/s |
| HiCache total | 2.85M tokens | 2.55M tokens |
| 1M context | Supported | Supported |

## Uninstall

```bash
helm uninstall sglang-glm52-308x -n kube-system
```
