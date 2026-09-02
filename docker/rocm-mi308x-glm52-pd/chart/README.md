# GLM-5.3 MI308X 2P2D Helm chart

Live snapshot of `kube-system/sglang-1p1d` as of 2026-09-02 (chart 0.3.10). Two TP8 prefills and two TP8 decodes, with GDR PD, DSA tilelang + FlyDSL MQA, NEXTN, and Mooncake L3 HiCache. Serving uses `page_size=64`; Mooncake store sidecars and router `power_of_two` remain enabled. If the release still carries the DP8+EP8 decode-1 canary, apply `values-2tp8.yaml` to restore the two-TP8 topology.

Use this chart to rebuild the **current** cluster from zero. Worker discovery is the live scheme: **hostNetwork + node host IPs** (not STS DNS). Prefill and decode all bind `:30000` / `:8998` because they sit on different nodes.

## Topology

| Role | STS / deploy | Node | Host IP | Ports |
|---|---|---|---|---|
| Prefill-0 | `sglang-1p1d-prefill` | `node-21.151.225.144` | 21.151.225.144 | 30000 / 8998 |
| Prefill-1 | `sglang-1p1d-prefill-1` | `node-21.151.225.152` | 21.151.225.152 | 30000 / 8998 |
| Decode-0 | `sglang-1p1d-decode` | `node-21.151.225.132` | 21.151.225.132 | 30000 / 8998 |
| Decode-1 | `sglang-1p1d-decode-1` | `node-21.151.225.172` | 21.151.225.172 | 30000 / 8998 |
| Mooncake master | `sglang-1p1d-mooncake-master` | `node-21.151.225.152` | 21.151.225.152 | 50051 / 9003 |
| Router | `sglang-1p1d-router` | `node-21.151.225.152` | hostNetwork | 30001 / 19096 |

Router args (matches live):

```
--prefill http://21.151.225.144:30000 8998
--prefill http://21.151.225.152:30000 8998
--decode  http://21.151.225.132:30000
--decode  http://21.151.225.172:30000
```

`MOONCAKE_MASTER=sglang-1p1d-mooncake-master.kube-system.svc.cluster.local:50051`. The Service DNS follows the master pod if it is rescheduled; the RDMA data path remains P2P. `SGLANG_HOST_IP` / `MOONCAKE_LOCAL_HOSTNAME` are the worker host IPs.

Router policies: `--prefill-policy cache_aware --decode-policy power_of_two` on `v0827-pot-loads`. Prefill uses cache-aware routing for prefix reuse; decode uses power-of-two load balancing. Cross-prefill prefix reuse is Mooncake L3.

Public HTTPRoute: `https://glm52-pd-1p1d.jmpti.woa.com` → Service `sglang-1p1d-router:30001`.

## From-scratch restore

Live STS were patched with `kubectl`, so Helm field-manager conflicts are expected:

```bash
helm upgrade --install sglang-1p1d docker/rocm-mi308x-glm52-pd/chart \
  -n kube-system --force-conflicts \
  --set apiKey=sk-...
```

Do not commit the live API key. Do not `helm install` a second PD release. Do not set `--hicache-ratio 4` (OOM ~2TB host KV on TP8). Scale the router to 0 before a bounce, then back to 1 with `nodeName` pinned and `tolerations: [{operator: Exists}]` — GPU nodes are tainted; a hostNetwork pod without that toleration fails kubelet `NodePorts` and leaves zombie replicas.

Each GPU node needs `/data/model/glm52-fp8` and `/data/aiter_configs/a8w8_blockscale_tuned_fmoe_glm5_1_cu80.csv` (cu_num=80 **and** decode tiles `expert=256,topk=8` for token 1..128; see `scripts/merge_decode_fmoe_256_8.py`). Prefill will try `/data/mooncake-patched/patch_evict_backup.py` (v2) and `patch_prefetch_log.py` and continue if they are missing.

## BF16 gate / indexer overlay (chart 0.3.6, GLM-5.3, 2xTP8 decode overlay)

Do **not** merge the fat `/data/aiter_configs/bf16_tuned_gemm.csv`. Init generates a thin table (threaded, `--workers 8`) and the main container **overwrites** the image file:

`/sgl-workspace/aiter/aiter/configs/model_configs/mi308x_gfx942_bf16_tuned_gemm.csv`

Per-node hostPath (locatable):

```
/data/aiter_configs/gen_bf16_gate_indexer.py
/data/aiter_configs/mi308x_bf16_gate_indexer.csv
/data/aiter_configs/mi308x_bf16_gate_indexer.meta
```

`N=32` is DSA indexer `weights_proj` (BF16 by design). `N=256` is MoE `mlp.gate` (BF16 by design). Optional GPU retune after generate, **not** in init:

```bash
kubectl exec -n kube-system sglang-1p1d-prefill-0 -c sglang -- \
  python3 /data/aiter_configs/gen_bf16_gate_indexer.py tune --workers 8
```

Then bounce that prefill so `install` copies the tuned CSV over the image overlay. Bump `aiterBf16Gemm.version` to force regenerate.

## Paged FlyDSL status

The native paged FlyDSL decode kernel now supports the live `page_size=64 + AITER preshuffle` layout, so it no longer requires a cluster-wide `page_size=1` migration. It is **disabled everywhere** as of 2026-09-02: `values.yaml` sets `decode.pagedFlydsl.enabled: false`, and `values-2tp8.yaml` no longer enables it on decode-1 (that canary overlay was reverted — see `project_1p1d_helm_null_values_trap.md`; its `args: null` also silently stripped every decode arg from both decodes).

The runtime overlay is decode-only and guarded: it does not modify decode-0 or either prefill. The build patch and runtime patch share the same enable script, which removes the earlier drift between the Dockerfile and ConfigMap.

Microbenchmark on decode-0 (`batch=4`, `next_n=4`, `max_seq_len=32768`,
`iters=100`): FlyDSL `mfma_r4_w4` is **1.2–1.4× faster** than AITER preshuffle
on the live `page_size=64` layout, with `max_abs=0.007812` and CUDA-graph
replay passing.

## Mooncake L3 layout

Do **not** run a Mooncake master per GPU node. L3 is one shared DRAM pool:

- **1 master** (metadata only) colocated with prefill-1 at `21.151.225.152:50051`
- **Store sidecar** (`mooncake_client :50052`) on every GPU node: 64 GB on each prefill, 256 GB on each decode. Cluster L3 ≈ **640 GB**
- Prefill SGLang uses the **in-process** store (`MOONCAKE_GLOBAL_SEGMENT_SIZE=64gb`). Dummy/standalone `setup_dummy` failed on this ROCm image; sidecars still donate extra DRAM. L3 on a prefill node is wiped if that worker restarts; decode sidecars keep their segments.
- Cross-prefill hits RDMA-read the other node's segment via the master replica list
- Decode GPU is still in-flight KV + NEXTN only (no decode HiCache). Decode host DRAM is donated to L3
- Prefetch policy is `wait_complete` with `prefetch_threshold: 1` so PD waits for L3 GET

A per-node master would partition L3 and make cross-P prefix sharing impossible. Stale 0-capacity segments after pod churn are cleared by stopping workers, Recreate-restarting the master, then bringing workers back.

## Images

- Prefills / mooncake: `mirrors.tencent.com/ti-platform/sglang-glm52-308x:v0826-hicache-jit`
- Decode-0: `mirrors.tencent.com/ti-platform/sglang-glm52-308x:v0826-fused-topk`
- Decode-1: `mirrors.tencent.com/ti-platform/sglang-glm52-308x:v0826-fused-topk` (same as decode-0 since 2026-09-02; the `v0831-dp8ep8-mtp-paged-flydsl-v4` DP8+EP8 canary image was reverted)
- Router: `mirrors.tencent.com/ti-platform/sglang-glm52-308x-pd-router:v0827-pot-loads`
- Rollback workers: `mirrors.tencent.com/ti-platform/sglang-glm52-308x:v0826-fused-topk`

The current lineage keeps the FlyDSL gfx942 ragged/prefill MQA kernel, HIP event optimization, BF16 gate/indexer overlay, and decode MoE CSV. `SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=0` keeps AITER preshuffle enabled, which is why serving remains on `page_size=64`. The TP8 canary enables the native paged FlyDSL kernel on that layout.
