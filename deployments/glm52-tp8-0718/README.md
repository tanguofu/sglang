# GLM-5.2 2TP8 Deployment Snapshot — 2026-07-18

This directory is a **complete redeployment reference** for the GLM-5.2 sglang
deployment on TI-Cloud (2× 8-GPU nodes, TP=8), including all configs, scripts,
eval results, and the 503 root-cause investigation that led to the service
selector fix.

## Topology

| Component | Helm release | Pods | Service | Endpoint |
|---|---|---|---|---|
| Router | `sglang-glm52-2tp8` | `sglang-glm52-2tp8-router-*` | `sglang-glm52-2tp8-router:30001` | https://glm52-2tp8.jmpti.woa.com |
| Worker 1 | `sglang-glm52-2tp8` | `sglang-glm52-2tp8-sglang-0` | `sglang-glm52-2tp8-sglang:30000` | 21.234.170.103:30000 |
| Worker 2 | `sglang-glm52-2tp8-w2` | `sglang-glm52-2tp8-w2-sglang-0` | `sglang-glm52-2tp8-w2-sglang:30000` | 21.234.170.38:30000 |
| Test (separate) | `sglang-glm52-test` / `sglang-glm52-test-w2` | `sglang-glm52-test-*-0` | (isolated) | 21.151.225.144 |

- **Model**: `glm-5.2` (reasoning model — always emits `reasoning_text` before `output_text`)
- **Context window**: 524288 tokens (512K)
- **Router policy**: `cache_aware`
- **Router metrics**: port 29000 (Prometheus)
- **External gateway**: `glm52-2tp8.jmpti.woa.com` → K8s HTTPRoute → router:30001

## Directory Layout

```
deployments/glm52-tp8-0718/
├── README.md                       ← this file
├── configs/
│   ├── cli/                        ← claude / codex / grok configs (secrets redacted)
│   ├── httproute/                  ← HTTPRoute patches + backups
│   ├── router/                     ← router args, source, metrics, custom image build
│   ├── router-optimization-0718/   ← cache_aware threshold tuning (applied 2026-07-18)
│   ├── service-patches/            ← service selector fix (503 root cause)
│   ├── sglang-custom-patches/      ← sglang custom code (DSA attention, frozen-KV MTP, chunk processor)
│   ├── pd-manifests/               ← PD (prefill-decode) pod manifests + Helm values
│   └── pd-patches/                 ← PD kubectl patches (UCX rails, sidechannel, proxy mount)
├── scripts/
│   ├── eval/                       ← multi-dimension eval suite (26 cases)
│   ├── router-investigation/       ← 503 root-cause investigation scripts
│   ├── 503-fix/                    ← service selector patch + verification
│   ├── streaming/                  ← SSE streaming compatibility tests
│   ├── benchmarks/                 ← codex latency benchmarks
│   ├── e2e/                        ← end-to-end CLI verification
│   └── server-audit/               ← server-side tunables & launch args audit
├── results/
│   ├── eval/                       ← eval results (gateway 26/26, worker 26/26)
│   ├── router-metrics/             ← router Prometheus metrics snapshot
│   ├── post-optimization-0718/     ← post-router-tuning eval results (26/26 PASS)
│   ├── bottleneck-bench-0718/      ← end-to-end pipeline bottleneck analysis + raw data
│   └── logs/                       ← sanity check logs
├── backups/services/               ← pre-fix service YAMLs (5 services)
└── docs/
    ├── bug_v1_responses_nonstreaming_400.md   ← /v1/responses non-streaming 400 bug
    ├── stream_compat_matrix.md                ← SSE event sequence reference
    ├── deployment_kernel_analysis_0718.md     ← deployment topology + per-operator review (gfx942)
    └── precision_preserving_optimization_plan_0718.md ← zero-precision-loss optimization plan + P2P topology verification
```

## Redeployment Steps

### 1. Worker Deployment (Helm)

Two Helm releases for the two 8-GPU worker nodes:

```bash
# Worker 1 (21.234.170.103)
helm install sglang-glm52-2tp8 ./charts/sglang-glm52-308x \
  --namespace kube-system \
  -f values-worker1.yaml

# Worker 2 (21.234.170.38) — separate release so it can be scaled/redeployed independently
helm install sglang-glm52-2tp8-w2 ./charts/sglang-glm52-308x \
  --namespace kube-system \
  -f values-worker2.yaml
```

Key worker launch args (see `configs/router/router_args.py` for full list):

- `--tp 8` (tensor parallelism across 8 GPUs)
- `--context-length 524288`
- `--quantization fp8` (GLM-5.2 FP8)
- `--enable-cache-aware-router`

### 2. Router Deployment

Router runs as a separate pod (`sglang-glm52-2tp8-router-*`) and routes between
worker 1 and worker 2 using the `cache_aware` policy. Router args are captured
in `configs/router/router_args.py` (1156 lines, pulled from the running pod).

Key router tunables (defaults):

| Parameter | Default | Description |
|---|---|---|
| `policy` | `cache_aware` | Worker selection policy |
| `cb_failure_threshold` | 10 | Failures before circuit opens |
| `cb_timeout_duration_secs` | 60 | Circuit open duration |
| `cb_window_duration_secs` | 120 | Failure counting window |
| `health_check_timeout_secs` | 5 | Health check timeout |
| `health_check_interval_secs` | 60 | Health check interval |
| `retry_max_retries` | 5 | Max retries per request |
| `disable_circuit_breaker` | False | CB enabled by default |

Metrics endpoint: `http://<router-pod>:29000/metrics` (Prometheus format).

### 3. HTTPRoute Setup (External Gateway)

The external gateway `glm52-2tp8.jmpti.woa.com` routes to the in-cluster
router via a K8s HTTPRoute. Worker-direct routes are also defined for
testing/bypass.

```bash
# Apply the HTTPRoute patch (worker-direct routes + catch-all)
kubectl apply -f configs/httproute/httproute_patch_v2.json

# Or build the patch from scratch
python3 configs/httproute/build_httproute_patch_v2.py | kubectl apply -f -
```

See `configs/httproute/glm52-router.yaml` for the full HTTPRoute manifest,
and `configs/httproute/httproute_backup.yaml` for the pre-patch state.

### 4. Service Selector Fix (CRITICAL — 503 Root Cause)

**Why this matters**: The Helm-managed services shipped with selectors that
only matched on `app: sglang-router` / `app: sglang`, without the
`app.kubernetes.io/instance` label. This caused production services to match
**both** production and test pods (test pods share the same `app` label but
have a different `instance` label). When the test pods were unhealthy
(restarting), 50% of gateway traffic hit them and returned 503
`no_available_workers`.

**Fix**: Add `app.kubernetes.io/instance` to the selector of all 5 production
services. Backups of the pre-fix services are in `backups/services/`.

```bash
# Back up current services first
bash scripts/503-fix/backup_svc.sh

# Apply the selector patch (idempotent — safe to re-run)
bash scripts/503-fix/patch_svc_selectors.sh
```

The patch adds these selectors:

| Service | Selector added |
|---|---|
| `sglang-glm52-2tp8-router` | `app.kubernetes.io/instance: sglang-glm52-2tp8` |
| `sglang-glm52-2tp8-sglang` | `app.kubernetes.io/instance: sglang-glm52-2tp8` |
| `sglang-glm52-2tp8-sglang-headless` | `app.kubernetes.io/instance: sglang-glm52-2tp8` |
| `sglang-glm52-2tp8-w2-sglang` | `app.kubernetes.io/instance: sglang-glm52-2tp8-w2` |
| `sglang-glm52-2tp8-w2-sglang-headless` | `app.kubernetes.io/instance: sglang-glm52-2tp8-w2` |

**Important Helm note**: This fix is applied directly to the live services via
`kubectl patch`. A subsequent `helm upgrade` will **revert** the selector unless
the Helm values files are also updated to include the `instance` label in
`service.selector`. The values files are not in this repo (they live with the
Helm chart) — update them in the chart repo before the next `helm upgrade`.

### 5. Verification

After deployment, run the verification suite in this order:

```bash
# 5a. Sanity check — gateway reachable, returns 200
bash scripts/e2e/sanity_check.sh

# 5b. Endpoint audit — confirm no test pods leaked into production services
bash scripts/router-investigation/check_endpoints.sh

# 5c. Pod label audit — confirm all pods have correct instance labels
bash scripts/router-investigation/check_pod_labels.sh

# 5d. 503 stress test — 130 requests (20 sequential + 60 concurrent + 50 stress)
bash scripts/503-fix/verify_503_fixed.sh

# 5e. Full eval — 26 cases, 11 categories, gateway-routed
python3 scripts/eval/eval_glm52_v2.py --endpoint https://glm52-2tp8.jmpti.woa.com

# 5f. Streaming compatibility — 24 cases
bash scripts/streaming/stream_compat_test.sh

# 5g. End-to-end CLI verification — 10 cases via claude/codex CLIs
bash scripts/e2e/final_e2e_verify.sh
```

Expected results (all verified 2026-07-18):

- 5a: 200 OK
- 5b: 1 endpoint per service (no test pods)
- 5c: 6 pods, distinct instance labels
- 5d: 130/130 OK, 0 503
- 5e: 26/26 PASS (see `results/eval/eval_results_v2.json`)
- 5f: 22 PASS, 0 FAIL, 2 WARN
- 5g: 10/10 PASS

### 6. CLI Configuration

Copy the configs to the local CLI config paths. **Replace the API key
placeholder** with the real token before use.

```bash
# Claude Code
cp configs/cli/claude-settings.json ~/.claude/settings.json
# Edit ~/.claude/settings.json — replace ${ANTHROPIC_AUTH_TOKEN}

# Codex CLI
cp configs/cli/codex-config.toml ~/.codex/config.toml
# Set OPENAI_API_KEY env var: export OPENAI_API_KEY="sk-..."

# Grok CLI (different endpoint — vLLM PD, 256K context)
cp configs/cli/grok-config.toml ~/.grok/config.toml
# Edit ~/.grok/config.toml — replace ${GROK_API_KEY}
```

CLI smoke test (one-liner each):

```bash
claude --print "What is 2+2?"            # → 4
codex exec "What is 2+2?"                # → 4
grok --single "What is 2+2?"             # → 4 (TUI, needs -p/--single for headless)
```

## Known Issues & Caveats

### 1. GLM-5.2 reasoning overhead in eval

GLM-5.2 is a **reasoning model** — every response emits `reasoning_text` (200-1500
tokens) before `output_text`. Eval scripts must set `max_tokens` to 1500-2500 to
avoid truncation. The v1 eval used 50-600 tokens and reported false failures.
See `scripts/eval/eval_glm52_v2.py` for the fixed version.

### 2. `/v1/responses` non-streaming returns 400

The sglang router returns 400 for non-streaming `/v1/responses` requests. All
clients must use `stream: true`. Documented in
`docs/bug_v1_responses_nonstreaming_400.md`.

### 3. Worker `/health` takes 1.00s

sglang's default `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=EnvBool(True)`
triggers a generation test on every health check, with an `asyncio.sleep(1)`
loop waiting for the detokenizer. This is **not** the cause of 503s (health
checks always return 200), but it inflates health-check latency. Set
`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=False` if sub-second health checks
are required.

### 4. `cache_aware` policy load imbalance (FIXED 2026-07-18)

**Before**: The `cache_aware` policy with `balance-abs-threshold=32` never triggered load balancing at low concurrency. All traffic went to worker 1 (.103); worker 2 (.38) was idle (27 requests total over 8h).

**After**: Tuned thresholds in `configs/router-optimization-0718/router-args-patch.json`:

```
--cache-threshold 0.2        (was 0.5)
--balance-abs-threshold 1    (was 32)
--balance-rel-threshold 1.2  (was 1.5)
```

**Verification**: Concurrent 40-request batch split 18/22 across workers. Full 26-case eval: 26/26 PASS. See `results/post-optimization-0718/`.

**End-to-end bottleneck benchmark** (2026-07-18): 27 sequential requests (4 hops × 3 prompts × 3 runs) + 20 concurrent long-prompt requests. Findings: router adds ~1ms (negligible), gateway adds 0.42–0.75s TTFT (1.4% of total), **decode phase dominates at 98% of end-to-end latency** (73% of output is reasoning tokens). 20 concurrent requests: 11/9 worker split, 0 queue, 1% KV usage, 1.75× slowdown for 20× load. See `results/bottleneck-bench-0718/ANALYSIS.md`.

**Radix cache (prefix affinity) status**: Both workers have `disable_radix_cache=False` (radix cache enabled), but `cache_hit_rate=0.0` on both. This is expected for short, diverse prompts (eval/chat workloads with low prefix overlap). The radix cache will show hits when the same system prompt + tool definitions are reused across requests (e.g., codex sessions with long shared prefixes).

**mixed_chunk NOT enabled**: Mutually exclusive with `speculative_algorithm=EAGLE`. sglang's `speculative_hook.py:_handle_eagle_family` forces `enable_mixed_chunk=False`. Keeping EAGLE (3 steps × 4 draft tokens = up to 4x decode speedup) over mixed_chunk's batching gain.

### 5. Helm upgrade reverts service selector fix

The service selector fix is applied via `kubectl patch`, not Helm. A
`helm upgrade` will overwrite the selector back to the Helm-managed value
(without `instance`), re-introducing the 503 bug. **Update the Helm values
files** to include `app.kubernetes.io/instance` in service selectors before
the next `helm upgrade`.

### 6. Grok CLI endpoint is different

Grok is configured to point at `glm52-vllm-pd.jmpti.woa.com` (vLLM PD
disaggregated, 256K context), **not** `glm52-2tp8` (sglang, 512K). This is
intentional — the vLLM PD deployment is a separate stack. Update
`configs/cli/grok-config.toml` if you want grok to use the sglang stack.

### 7. FP8 KV scaling factor unsupported

GLM-5.2's `GlmMoeDsaForCausalLM` does not implement `load_kv_cache_scales`.
Passing `--quantization-param-path` will crash. This is expected for the MLA
architecture — the FP8 KV scaling factor is computed at runtime, not loaded
from a JSON.

## SGLang Custom Code (DSA + Frozen-KV MTP)

The GLM-5.2 deployment uses a customized sglang with DSA (DeepSeek Sparse
Attention) and frozen-KV MTP (Multi-Token Prediction) support. The custom
source files are in `configs/sglang-custom-patches/`:

| File | Lines | Purpose |
|---|---|---|
| `base_srt_layers_attention_dsa_backend.py` | 3110 | DSA attention backend (DeepSeek Sparse Attention) |
| `base_srt_layers_attention_dsa_dsa_indexer.py` | 2523 | DSA indexer for sparse attention pattern |
| `base_srt_models_deepseek_nextn.py` | 424 | DeepSeek NextN layer (MTP draft model) |
| `base_srt_models_transformers.py` | 1638 | Custom transformers model loader |
| `base_srt_speculative_eagle_worker_v2.py` | 1750 | EAGLE v2 speculative worker |
| `base_srt_speculative_frozen_kv_mtp_worker_v2.py` | 775 | Frozen-KV MTP worker (two-layer, like eagle_worker_v2) |
| `chunk_processor_patch.py` | 143 | Router SSE chunk processor patch (CRLF normalization) |
| `patch_staging.py` | 185 | Staging helper for applying patches to sglang source |
| `diag_compare.sh` | 34 | Diagnostic: compare patched vs upstream source |

These files are the **base/patched sources**, not unified diffs. To apply:

```bash
# Inside the sglang source tree (e.g. /sgl-workspace/sglang/python/sglang/srt/)
cp configs/sglang-custom-patches/base_srt_layers_attention_dsa_backend.py \
   python/sglang/srt/layers/attention/dsa_backend.py
cp configs/sglang-custom-patches/base_srt_layers_attention_dsa_dsa_indexer.py \
   python/sglang/srt/layers/attention/dsa/dsa_indexer.py
# ... and so on per the file path encoded in the base_ name
```

The frozen-KV MTP worker reads the target KV cache read-only and owns no KV
pool — its "draft extend" selects the last accepted token + target hidden
state as the next-iter seed, and the seed forward runs at the start of the
next draft iteration.

## PD (Prefill-Decode) Disaggregated Manifests

The `configs/pd-manifests/` directory contains the **separate** PD deployment
manifests (vLLM-based, not sglang). These are NOT the 2tp8 sglang deployment
— they are the `glm52-vllm-pd` deployment that grok CLI points to.

| File | Purpose |
|---|---|
| `sglang-glm52-2tp8-manifest.yaml` | Full Helm manifest for the 2tp8 sglang deployment |
| `sglang-glm52-2tp8-values.yaml` | Helm values for the 2tp8 sglang deployment |
| `pd-prefill-144.yaml` | PD prefill pod on node 21.151.225.144 |
| `pd-decode-132.yaml` | PD decode pod on node 21.151.225.132 |
| `vllm-prefill.yaml` | vLLM prefill pod (basic) |
| `vllm-decode.yaml` | vLLM decode pod (basic) |
| `vllm-prefill-opt.yaml` | vLLM prefill pod (optimized — chunked prefill, cuda graph) |
| `vllm-decode-opt.yaml` | vLLM decode pod (optimized) |
| `smp-p1-decode-rails-only.yaml` | SMP P1 decode (rails-only mode) |
| `smp-p1-rails-only.yaml` | SMP P1 (rails-only mode) |

PD key parameters (from `pd-prefill-144.yaml`):

- `--context-length 1048576` (1M)
- `--chunked-prefill-size 32768`
- `--mem-fraction-static 0.90`
- `--kv-cache-dtype fp8_e4m3`
- `--disaggregation-mode prefill`
- `--disaggregation-transfer-backend mooncake`
- `--cuda-graph-bs-prefill 4 8 16 32`
- `--max-running-requests 128`

The `configs/pd-patches/` directory contains kubectl patches applied to the
PD pods (UCX rails configuration, sidechannel mounts, proxy mounts).

## Router & SGLang Metrics

### Optimization Applied 2026-07-18

**Router cache_aware thresholds tuned** (see `configs/router-optimization-0718/`):

| Parameter | Before | After | Rationale |
|---|---|---|---|
| `cache-threshold` | 0.5 | 0.2 | Lower prefix-match requirement; cache_aware triggers at 20% match instead of 50% |
| `balance-abs-threshold` | 32 | 1 | Trigger load balancing when load diff > 1 (was 32 — never triggered at low concurrency) |
| `balance-rel-threshold` | 1.5 | 1.2 | Trigger when max_load > min_load * 1.2 (was 1.5) |

**Result**: Load is now distributed across both workers. Before: worker 2 had 27 requests total (idle). After: worker 1=207, worker 2=135 `/v1/responses` (concurrent batch split 18/22).

**Worker mixed_chunk**: NOT enabled — mutually exclusive with `speculative_algorithm=EAGLE`. sglang's `speculative_hook.py` forces `enable_mixed_chunk=False` when EAGLE is active. The launch script passes `--enable-mixed-chunk` but it's silently overridden. Keeping EAGLE (3 steps × 4 draft tokens) gives better latency reduction than mixed_chunk's batching improvement.

**Apply the optimization**:

```bash
# Back up current router deployment
kubectl get deploy -n kube-system sglang-glm52-2tp8-router -o yaml > /tmp/router-backup.yaml

# Apply the patch (strategy must be Recreate due to hostPort)
kubectl patch deploy -n kube-system sglang-glm52-2tp8-router --type=json \
  -p="$(cat configs/router-optimization-0718/router-args-patch.json)"

# If rollout hangs (Pending), switch strategy to Recreate:
kubectl patch deploy -n kube-system sglang-glm52-2tp8-router --type=json \
  -p='[{"op":"replace","path":"/spec/strategy","value":{"type":"Recreate"}}]'

# Verify
bash configs/router-optimization-0718/verify_optimization_0718.sh
```

**Rollback**:

```bash
kubectl apply -f configs/router-optimization-0718/router-deploy-backup-pre-patch.yaml
# or
kubectl patch deploy -n kube-system sglang-glm52-2tp8-router --type=json \
  -p='[{"op":"replace","path":"/spec/template/spec/containers/0/args","value":["--worker-urls","http://21.234.170.103:30000","http://21.234.170.38:30000","--policy","cache_aware","--host","0.0.0.0","--port","30001","--cache-threshold","0.5","--balance-abs-threshold","32","--balance-rel-threshold","1.5"]}]'
```

**Important note on circuit breakers**: If a batch of requests with wrong API key (401) is sent, the CB will trip (`closed -> open`) because sglang router counts 401 as a worker failure. The CB auto-recovers after `cb_timeout_duration_secs=60s`. To force recovery, send 10+ successful requests. See "CB 401 incident" in `results/post-optimization-0718/`.

### Metrics Endpoints

**Router metrics** (port 29000, Prometheus format):

```bash
kubectl exec -n kube-system deploy/sglang-glm52-2tp8-router -- \
  curl -s localhost:29000/metrics
```

Key router metrics:

| Metric | Description |
|---|---|
| `smg_worker_selection_total{policy}` | Worker selection count by policy |
| `smg_worker_cb_state{worker}` | Circuit breaker state (0=CLOSED, 1=OPEN, 2=HALF_OPEN) |
| `smg_worker_cb_outcomes_total{worker,outcome}` | CB success/failure counts |
| `smg_worker_health{worker}` | Worker health (1=healthy) |
| `smg_worker_requests_active{worker}` | Active requests per worker |
| `smg_router_upstream_responses_total{status_code}` | Upstream response codes |
| `smg_http_responses_total{status_code}` | HTTP response codes (NOTE: does not count 503s) |

**SGLang worker metrics** (port 30000, requires `Authorization: Bearer <token>`):

```bash
kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- \
  curl -s -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
  localhost:30000/metrics
```

Key worker metrics:

| Metric | Description |
|---|---|
| `sglang:cache_hit_rate` | Radix cache hit rate (prefix cache affinity) |
| `sglang:token_usage` / `full_token_usage` | KV cache utilization (0.0-1.0) |
| `sglang:kv_evictable_tokens` | Tokens that can be evicted from KV cache |
| `sglang:num_running_reqs` / `num_queue_reqs` | Running / queued request counts |
| `sglang:gen_throughput` | Generation throughput (tokens/sec) |
| `sglang:per_stage_req_latency_seconds{stage}` | Latency by stage (prefill_forward, chunked_prefill, decode) |
| `sglang:routing_key_running_req_count` | Routing key distribution (for prefix affinity) |

**Current state (verified 2026-07-18 07:07 UTC)**:

- Both workers: `enable_metrics=True`, `disable_radix_cache=False` (radix ON)
- Both workers: `chunked_prefill_size=131072`, `mem_fraction_static=0.82`, `kv_cache_dtype=fp8_e4m3`
- Worker 1 (.103): `cache_hit_rate=0.0`, `token_usage=0.07`, `kv_evictable=569408`, 1641 prefill_forward calls
- Worker 2 (.38): `cache_hit_rate=0.0`, `token_usage=0.0`, `kv_evictable=256`, 0 prefill_forward calls (idle)
- Router: `smg_worker_cb_outcomes_total` only has data for .103 (worker 1), worker 2 never selected by `cache_aware`

## Investigation Trail (503 Root Cause)

The 503 investigation is documented across these files in chronological order:

1. `scripts/eval/eval_glm52_v2.py` — gateway eval revealed 12/26 failures (all 503)
2. `scripts/eval/eval_glm52_worker.py` — worker-direct eval confirmed 26/26 (model OK)
3. `scripts/router-investigation/capture_503_with_logs.sh` — router logs showed no 503
4. `scripts/router-investigation/inspect_router.py` — discovered metrics on port 29000
5. `results/router-metrics/router_metrics.txt` — `.38` worker had zero `cb_outcomes`
6. `scripts/router-investigation/test_health_behavior.sh` — health check not the cause
7. `scripts/router-investigation/test_w38_direct.sh` — `.38` worker serves 200 directly
8. `scripts/503-fix/stress_test_concurrent.sh` — 60/60 OK at one moment (intermittent)
9. `scripts/router-investigation/check_endpoints.sh` — **found test pods in prod services**
10. `scripts/router-investigation/check_pod_labels.sh` — confirmed distinct instance labels
11. `scripts/503-fix/patch_svc_selectors.sh` — applied the instance-label fix
12. `scripts/503-fix/verify_503_fixed.sh` — 130/130 OK, 0 503 confirmed

## Files NOT in This Snapshot

- Helm chart source (`charts/sglang-glm52-308x/`) — lives in a separate repo
- Helm values files (`values-worker1.yaml`, `values-worker2.yaml`) — same repo
- Worker pod launch YAML — Helm-managed, not committed here
- GPU node configuration — managed by cluster admin
- Real API keys / tokens — never committed (configs use `${...}` placeholders)
- `sglang_router-0.3.2-cp38-abi3-manylinux_2_34_x86_64.whl` (17 MB) — too large for git.
  Available in the sglang router release artifacts, or pull from the running router
  pod: `kubectl cp kube-system/<router-pod>:/opt/sglang/router/sglang_router-0.3.2-cp38-abi3-manylinux_2_34_x86_64.whl ./`

## Quick Reference

```bash
# Gateway endpoint
curl -s https://glm52-2tp8.jmpti.woa.com/v1/models | jq '.data[0].id'
# → "glm-5.2"

# Router metrics
kubectl exec -n kube-system deploy/sglang-glm52-2tp8-router -- \
  curl -s localhost:29000/metrics | grep smg_worker_cb_state

# Worker health
kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- \
  curl -s localhost:30000/health
# → {"status":"ok",...}

# Run eval
python3 scripts/eval/eval_glm52_v2.py --endpoint https://glm52-2tp8.jmpti.woa.com
```
