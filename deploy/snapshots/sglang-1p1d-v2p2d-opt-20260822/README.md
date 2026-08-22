# SGLang 2P2D GLM-5.2 Deployment Snapshot

**Version**: v2p2d-opt-20260822
**Date**: 2026-08-22
**Architecture**: 2 Prefill + 2 Decode (PD Disaggregation)

## Images

| Component | Image |
|-----------|-------|
| Engine (prefill/decode/mooncake) | `YOUR_REGISTRY/your-project/sglang-glm52-308x:v2p2d-opt-20260822` |
| Router | `YOUR_REGISTRY/your-project/sglang-glm52-308x-pd-router:v2p2d-opt-20260822` |

## Node Assignment

| Node | Role | Pod |
|------|------|-----|
| NODE_DECODE_0_IP | Decode-0 + Mooncake Master | sglang-1p1d-decode-0 |
| NODE_PREFILL_0_IP | Prefill-0 | sglang-1p1d-prefill-0 |
| NODE_PREFILL_1_IP | Prefill-1 | sglang-1p1d-prefill-1-0 |
| NODE_DECODE_1_IP | Decode-1 | sglang-1p1d-decode-1-0 |

## Key Configuration

### Decode (both workers)
- `--speculative-algorithm NEXTN`
- `--speculative-num-steps 3` (draft_tokens = steps+1 = 4, auto-adjusted)
- `--speculative-eagle-topk 1`
- `--num-continuous-decode-steps 3`
- `--schedule-conservativeness 0.8`
- `--stream-interval 2`
- `--mem-fraction-static 0.92`
- `--kv-cache-dtype fp8_e4m3`
- `--max-running-requests 32`
- `--cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 12 16 20 24 32`

### Prefill (both workers)
- `--schedule-conservativeness 0.8`
- `--mem-fraction-static 0.88`
- `--max-prefill-tokens 32768`
- `--chunked-prefill-size 16384`
- `--enable-hierarchical-cache`
- `--hicache-ratio 1`
- `--hicache-storage-backend mooncake`
- `--hicache-mem-layout page_first_direct`
- `--hicache-write-policy write_through`

### Router
- `--pd-disaggregation`
- `--prefill http://NODE_PREFILL_0_IP:30000 8998`
- `--prefill http://NODE_PREFILL_1_IP:30000 8998`
- `--decode http://NODE_DECODE_0_IP:30000`
- `--decode http://NODE_DECODE_1_IP:30000`
- `--balance-abs-threshold 8`
- `--health-check-interval-secs 5`
- `--health-failure-threshold 3`

## Deploy from Scratch

```bash
# 1. Create namespace (if not exists)
kubectl create namespace kube-system  # usually exists

# 2. Create image pull secret
kubectl create secret docker-registry tencent-registry \
  --namespace=kube-system \
  --docker-server=YOUR_REGISTRY \
  --docker-username=<username> \
  --docker-password=<password>

# 3. Apply all resources in order
kubectl apply -f svc-sglang-1p1d-mooncake-master.yaml
kubectl apply -f svc-sglang-1p1d-router.yaml
kubectl apply -f svc-sglang-1p1d-prefill.yaml
kubectl apply -f svc-sglang-1p1d-decode.yaml
kubectl apply -f svc-sglang-1p1d-decode-1.yaml
kubectl apply -f deploy-sglang-1p1d-mooncake-master.yaml
kubectl apply -f deploy-sglang-1p1d-router.yaml
kubectl apply -f sts-sglang-1p1d-prefill.yaml
kubectl apply -f sts-sglang-1p1d-prefill-1.yaml
kubectl apply -f sts-sglang-1p1d-decode.yaml
kubectl apply -f sts-sglang-1p1d-decode-1.yaml
kubectl apply -f httproute.yaml

# 4. Wait for pods to be ready (~15-20 min per pod)
kubectl get pods -n kube-system -w | grep sglang-1p1d

# 5. Verify
kubectl exec -n kube-system deploy/sglang-1p1d-router -- \
  python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:19096/health').read().decode())"
```

## Prerequisites

- 4× AMD MI308X (gfx942) GPU nodes, each with 8× GPUs (192GB VRAM each)
- Broadcom BNXT RDMA NICs (bnxt_re_bond0-7) on each node
- `/data/model/glm52-fp8` model checkpoint on each node
- Mooncake Transfer Engine with GDR support
- Kubernetes 1.30+ with AMD GPU device plugin

## Performance (Benchmark Results)

| Metric | Value |
|--------|-------|
| Short conversation TPS | 71.4 tok/s |
| Short conversation E2E | 1.41s |
| TTFT (reasoning) | 0.29s |
| ITL p50 | 31ms |
| Concurrent (10) TPS | 287 tok/s |
| Spec accept length | 3.48/4 (87%) |
| Cache hit E2E | 1.02s |
