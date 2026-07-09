# SGLang GLM-5.2 on AMD MI308X (gfx942) — Helm Chart

Deploys SGLang GLM-5.2-FP8 inference service on AMD MI308X nodes with envoy gateway forwarding.

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

## Access

| Method | URL |
|--------|-----|
| Direct node | `http://21.151.225.144:30000` |
| Envoy gateway | `http://glm52-308x.jmpti.woa.com` |
| In-cluster | `http://sglang-glm52-308x-sglang.kube-system:30000` |

## Build Image

```bash
# On ti-builder
git clone --branch 308x-glm52-opt https://github.com/tanguofu/sglang.git
cd sglang
docker build -f docker/rocm-mi308x-glm52/Dockerfile -t sglang-glm52-308x:latest .
docker tag sglang-glm52-308x:latest ccr.ccs.tencentyun.com/qcloud-ti-platform/sglang-glm52-308x:latest
docker push ccr.ccs.tencentyun.com/qcloud-ti-platform/sglang-glm52-308x:latest
```

## Configuration

Key values in `values.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | `ccr.ccs.tencentyun.com/qcloud-ti-platform/sglang-glm52-308x` | SGLang image |
| `tag` | `latest` | Image tag |
| `hostNetwork` | `true` | Required for TKE ENI network mode |
| `port` | `30000` | SGLang server port |
| `nodeSelector` | `accelerator=amd-gpu, sglang-model=ready` | Target AMD nodes with model |
| `model.path` | `/data/model/glm52-fp8` | Model weights path |
| `sglang.tpSize` | `8` | Tensor parallel size (all 8 GPUs) |
| `sglang.contextLength` | `1048576` | 1M context |
| `sglang.memFractionStatic` | `0.88` | Memory fraction for KV cache |
| `sglang.speculativeNumSteps` | `3` | MTP speculative steps |
| `gateway.enabled` | `true` | Enable envoy gateway HTTPRoute |
| `gateway.hostname` | `glm52-308x.jmpti.woa.com` | External hostname |

## Performance Tuning

Adjust `values.yaml`:

```yaml
sglang:
  speculativeNumSteps: 2  # iWiki 4022520413: steps=2 outperforms 3
  memFractionStatic: 0.85  # Lower if OOM on 192GB cards
```

Then:
```bash
helm upgrade sglang-glm52-308x docker/rocm-mi308x-glm52/chart/ -n kube-system
```

## Uninstall

```bash
helm uninstall sglang-glm52-308x -n kube-system
```
