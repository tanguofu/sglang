# Node 19 (`node-21.234.170.19`) — GLM-5.2 TP8 Worker Verification Runbook

> Prepared 2026-07-20. Context: `cls-bmmk3vtl-context`, namespace `kube-system`.
> Goal: bring node 19 online as a sglang GLM-5.2 TP8 worker, mirroring the
> `sglang-glm52-2tp8-sglang-0` spec template (currently running on
> `node-21.151.225.152`). All new resources use the `prep19-` prefix. Do **not**
> touch existing production pods.

## 0. Pre-flight findings (already verified)

The original assumption that node 19's model dir was empty is **stale**. As of
2026-07-20 the node is download-ready and ROCm-ready:

| Check | Result |
|---|---|
| Model dir `/data/model/glm52-fp8/` | **Present, complete** — 703.8 GiB |
| `.safetensors` shard count | **141 / 141** (0 zero-size) |
| `model.safetensors.index.json` | Present, valid JSON, `total_size: 755617140416`, 141 distinct shard refs |
| `.download_complete` marker | Present — `"Completed via rsync from GZ at Fri Jul 10 21:09:36 CST 2026"` |
| Config / tokenizer files | `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json` all present |
| Host ROCm | **7.2.4** already installed at `/opt/rocm-7.2.4` (upgraded past 7.2.0) |
| ROCm 7.2.4 RPMs | `/data/rocm-upgrade-7.2.4/rpms/` present (27 RPMs, `amdgpu-dkms-firmware 30.30.4.0`) |
| GPUs | 8× AMD Instinct MI308X (`/dev/kfd`, `/dev/dri/card*`), 192 GiB VRAM each |
| Disk | 11.6 TiB total, 1.1 TiB used, **10.6 TiB available** |
| modelscope venv / download script | **Not present** on node 19 (model was rsync'd, not downloaded via modelscope — irrelevant now) |

**Conclusion: no model download is required.** Skip Section 1's download steps;
only run the completion confirmation, then deploy.

## 1. Confirm the download is complete

The model is already complete. Re-verify before deploying:

```bash
# Via the debug-ds pod on node 19 (debug-ds-fthw2):
kubectl --context cls-bmmk3vtl-context exec -n kube-system debug-ds-fthw2 -- sh -c '
  ls /host/data/model/glm52-fp8/.download_complete && \
  echo "shards: $(ls /host/data/model/glm52-fp8/*.safetensors | wc -l)" && \
  echo "zero-size: $(find /host/data/model/glm52-fp8 -name "*.safetensors" -size 0 | wc -l)" && \
  du -sh /host/data/model/glm52-fp8
'
```

Pass criteria:
- `.download_complete` marker exists.
- `shards: 141`.
- `zero-size: 0`.
- Total size ≈ **703.8 GiB** (≈ 755 GiB raw per index `total_size`).

If any shard is missing or zero-size, re-fetch from ModelScope on the sibling
node 32 (which has `/data/model_venv` + `/data/download_model.py`) and rsync
across — do not re-download the whole set.

## 2. Deploy the worker on node 19

Mirror of `sglang-glm52-2tp8-sglang-0`. Save as
`prep19-sglang-worker.yaml` and `kubectl apply`. The pod is pinned to node 19
via `nodeName` and tolerates both of its NoSchedule taints
(`dedicated=glm52-prod`, `node.cvm.com/ptl-unsupported`).

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: prep19-sglang-worker
  namespace: kube-system
  labels:
    accelerator: amd-gpu
    app: sglang
    app.kubernetes.io/name: sglang-glm52-308x
    prep19: "true"
spec:
  nodeName: node-21.234.170.19
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
  restartPolicy: Always
  terminationGracePeriodSeconds: 300
  imagePullSecrets:
    - name: tencent-registry
  tolerations:
    - key: dedicated
      value: glm52-prod
      effect: NoSchedule
    - key: node.cvm.com/ptl-unsupported
      effect: NoSchedule
    - key: node.kubernetes.io/not-ready
      operator: Exists
      effect: NoExecute
      tolerationSeconds: 300
    - key: node.kubernetes.io/unreachable
      operator: Exists
      effect: NoExecute
      tolerationSeconds: 300
  containers:
    - name: sglang
      image: mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3
      imagePullPolicy: IfNotPresent
      command: ["/bin/bash", "-c"]
      args:
        - |
          set -euo pipefail
          MODEL_PATH=${MODEL_PATH:-/data/model/glm52-fp8}
          API_KEY=${API_KEY:-sk-46faecc9d0bc4dcd9db6a15c73ae91c8}
          PORT=${PORT:-30000}
          if ! command -v numactl &>/dev/null; then
            echo "Installing numactl for NUMA memory binding..."
            apt-get update -qq && apt-get install -y -qq numactl >/dev/null 2>&1
          fi
          echo "=== GLM-5.2-FP8 SGLang (MI308X gfx942) — node19 ==="
          exec python3 -m sglang.launch_server \
            --model-path "$MODEL_PATH" \
            --model-impl sglang \
            --served-model-name glm-5.2 \
            --api-key "$API_KEY" \
            --tp-size 8 --pp-size 1 --trust-remote-code \
            --host 0.0.0.0 --port "$PORT" \
            --numa-node 0 0 0 0 1 1 1 1 \
            --context-length "524288" \
            --tool-call-parser glm47 --reasoning-parser glm45 \
            --mem-fraction-static 0.75 \
            --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 12 16 \
            --cuda-graph-max-bs-decode 16 \
            --enable-aiter-allreduce-fusion --enable-mixed-chunk \
            --chunked-prefill-size 16384 \
            --enable-fused-qk-norm-rope \
            --schedule-conservativeness 1 \
            --prefill-max-requests 4 --max-prefill-tokens 32768 \
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
            --hicache-write-policy write_back \
            --enable-metrics --skip-server-warmup \
            --watchdog-timeout 1200 --log-level info
      env:
        - {name: MODEL_PATH, value: /data/model/glm52-fp8}
        - {name: PORT, value: "30000"}
        - {name: API_KEY, value: sk-46faecc9d0bc4dcd9db6a15c73ae91c8}
        - {name: HIP_VISIBLE_DEVICES, value: "0,1,2,3,4,5,6,7"}
        - {name: NCCL_DEBUG, value: WARN}
        - {name: HSA_ENABLE_SDMA, value: "0"}
        - {name: HIP_FORCE_DEV_KERNARG, value: "1"}
        - {name: HSA_NO_SCRATCH_RECLAIM, value: "1"}
        - {name: NCCL_CUMEM_ENABLE, value: "0"}
        - {name: NCCL_MIN_NCHANNELS, value: "80"}
        - {name: NCCL_NVLS_ENABLE, value: "0"}
        - {name: PYTORCH_CUDA_ALLOC_CONF, value: expandable_segments:True}
        - {name: PYTORCH_ROCM_ARCH, value: gfx942}
        - {name: ROCM_QUICK_REDUCE_QUANTIZATION, value: NONE}
        - {name: SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN, value: "1"}
        - {name: SGLANG_DISABLE_CUDNN_CHECK, value: "1"}
        - {name: SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION, value: "false"}
        - {name: SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM, value: "1"}
        - {name: SGLANG_FORCE_COARSE_WAR_BARRIER, value: "true"}
        - {name: SGLANG_INT4_WEIGHT, value: "0"}
        - {name: SGLANG_MOE_PADDING, value: "1"}
        - {name: SGLANG_ROCM_DISABLE_LINEARQUANT, value: "0"}
        - {name: SGLANG_ROCM_FUSED_DECODE_MLA, value: "1"}
        - {name: SGLANG_SET_CPU_AFFINITY, value: "1"}
        - {name: SGLANG_USE_AITER, value: "1"}
        - {name: SGLANG_USE_ROCM700A, value: "1"}
        - {name: CUDA_ENABLE_USER_TRIGGERED_COREDUMP, value: "1"}
      ports:
        - {name: http, containerPort: 30000, hostPort: 30000, protocol: TCP}
      resources:
        limits:
          amd.com/gpu: "8"
        requests:
          amd.com/gpu: "8"
          cpu: "360"
          memory: 2100Gi
      securityContext:
        privileged: true
        capabilities:
          add: [SYS_PTRACE]
          drop: [ALL]
        seccompProfile:
          type: Unconfined
      livenessProbe:
        httpGet: {path: /health, port: 30000, scheme: HTTP}
        initialDelaySeconds: 600
        periodSeconds: 60
        failureThreshold: 5
        timeoutSeconds: 10
      readinessProbe:
        httpGet: {path: /health, port: 30000, scheme: HTTP}
        initialDelaySeconds: 120
        periodSeconds: 30
        failureThreshold: 10
        timeoutSeconds: 10
      volumeMounts:
        - {name: data, mountPath: /data}
        - {name: dev-kfd, mountPath: /dev/kfd}
        - {name: dev-dri, mountPath: /dev/dri}
        - {name: shm, mountPath: /dev/shm}
  volumes:
    - name: data
      hostPath: {path: /data, type: Directory}
    - name: dev-kfd
      hostPath: {path: /dev/kfd, type: CharDevice}
    - name: dev-dri
      hostPath: {path: /dev/dri, type: Directory}
    - name: shm
      emptyDir: {medium: Memory, sizeLimit: 32Gi}
```

Deploy:

```bash
kubectl --context cls-bmmk3vtl-context apply -f prep19-sglang-worker.yaml
kubectl --context cls-bmmk3vtl-context -n kube-system logs -f prep19-sglang-worker
```

Expected boot: ~8–12 min to load 141 FP8 shards across 8 GPUs, then `/health`
returns 200. The first readiness probe fires at +120 s; liveness at +600 s.

## 3. Verification

### 3.1 Baseline health & metrics

```bash
# Health
curl -s http://21.234.170.19:30000/health
# Model + server info
curl -s http://21.234.170.19:30000/get_model_info
# Live KV-cache / running-req metrics
curl -s http://21.234.170.19:30000/metrics | grep -E "sglang:(running|waiting|cache_hit|gen_throughput|num_used)"
```

Pass criteria:
- `/health` → 200.
- `get_model_info` reports `glm-5.2`, context length 524288.
- 8 GPUs visible in startup log (`HIP_VISIBLE_DEVICES=0..7`, all rank init OK).

### 3.2 Functional smoke test

```bash
curl -s http://21.234.170.19:30000/v1/chat/completions \
  -H "Authorization: Bearer sk-46faecc9d0bc4dcd9db6a15c73ae91c8" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2",
    "messages": [{"role":"user","content":"Count from 1 to 5."}],
    "max_tokens": 64,
    "temperature": 0
  }' | jq .
```

### 3.3 EAGLE stability stress test

The `fix-eagle-coredump-v3` image targets an EAGLE draft-model coredump. Stress
speculative decoding (`NEXTN`, 3 steps / 4 draft tokens) to confirm stability:

```bash
# Sustained concurrency, mixed prefill/decode, EAGLE on.
for i in $(seq 1 200); do
  curl -s http://21.234.170.19:30000/v1/chat/completions \
    -H "Authorization: Bearer sk-46faecc9d0bc4dcd9db6a15c73ae91c8" \
    -H "Content-Type: application/json" \
    -d '{
      "model":"glm-5.2",
      "messages":[{"role":"user","content":"Summarize the plot of Hamlet in three sentences."}],
      "max_tokens":512,"temperature":0.7
    }' > /dev/null &
done; wait
```

Pass criteria:
- Pod stays `Running`, **0 restarts**, no `SIGSEGV` / coredump in logs.
- No `watchdog-timeout` (1200 s) kills.
- `sglang:num_used_tokens` returns to ~0 after the burst (no leak).
- Compare EAGLE acceptance rate vs node-32 worker (`test32-sglang-0`) — should
  be within ~5%.

```bash
# EAGLE acceptance / draft metrics
curl -s http://21.234.170.19:30000/metrics | grep -iE "eagle|draft|accept"
# Restart count + last termination
kubectl --context cls-bmmk3vtl-context -n kube-system get pod prep19-sglang-worker \
  -o jsonpath='{.status.containerStatuses[0].restartCount} {..lastState}'
```

### 3.4 Baseline performance capture

Run the existing bench harness against node 19 and diff vs node 32:

```bash
python3 scripts/bench_long_agent_context.py \
  --endpoint http://21.234.170.19:30000 \
  --api-key sk-46faecc9d0bc4dcd9db6a15c73ae91c8 \
  --model glm-5.2 \
  --out results/node19-baseline.json
```

Capture TTFT, inter-token latency, throughput, and long-context (524k) behavior.
Compare against `results/long-agent-context-bench.json` (node-32 baseline).

## 4. Rollback / cleanup

```bash
kubectl --context cls-bmmk3vtl-context -n kube-system delete pod prep19-sglang-worker
# Remove any prep19- artifacts created during prep:
kubectl --context cls-bmmk3vtl-context -n kube-system delete job,configmap,secret -l prep19=true
```

The model dir `/data/model/glm52-fp8/` and ROCm 7.2.4 install are **not** removed
by cleanup — they are reusable node state.

## 5. Open items / blockers

- **ROCm 7.14 base image**: availability in `mirrors.tencent.com/ti-platform/`
  is **unknown**. The registry (Harbor) uses Bearer token auth; the
  `tencent-registry` imagePullSecret's basic credential does not grant tag-list
  scope, anonymous access is denied, and the Harbor v2 search API returns 404.
  No `rocm*` repo under `ti-platform/` is publicly enumerable. The worker image
  `sglang-glm52-308x:fix-eagle-coredump-v3` bundles its own ROCm userspace and
  runs fine on host ROCm 7.2.4, so a 7.14 base image is not required to bring
  node 19 online. If a 7.14 upgrade is later desired, request pull access to the
  `ti-platform` project or ask the image owner to publish/confirm a 7.14 tag.
- **No download needed**: model already present (rsync'd Jul 10). No `prep19-`
  download Job was created — creating one would redundantly re-fetch 703 GiB.
