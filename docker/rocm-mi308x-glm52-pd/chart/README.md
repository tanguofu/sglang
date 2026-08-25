# GLM-5.2 MI308X 2P2D Helm chart

Live snapshot of `kube-system/sglang-1p1d` as of 2026-08-25 (chart 0.3.5, worker `v0525-wave1`). Two TP8 prefills + two TP8 decodes, GDR PD, DSA tilelang + FlyDSL MQA, NEXTN, Mooncake L3 HiCache. `page_size=64`, prefill L2 ratio 2, Mooncake store sidecars, router `round_robin`.

Use this chart to rebuild the **current** cluster from zero. Worker discovery is the live scheme: **hostNetwork + node host IPs** (not STS DNS). Prefill and decode all bind `:30000` / `:8998` because they sit on different nodes.

## Topology

| Role | STS / deploy | Node | Host IP | Ports |
|---|---|---|---|---|
| Prefill-0 | `sglang-1p1d-prefill` | `node-21.151.225.144` | 21.151.225.144 | 30000 / 8998 |
| Prefill-1 | `sglang-1p1d-prefill-1` | `node-21.151.225.152` | 21.151.225.152 | 30000 / 8998 |
| Decode-0 | `sglang-1p1d-decode` | `node-21.151.225.132` | 21.151.225.132 | 30000 / 8998 |
| Decode-1 | `sglang-1p1d-decode-1` | `node-21.151.225.172` | 21.151.225.172 | 30000 / 8998 |
| Mooncake master | `sglang-1p1d-mooncake-master` | `node-21.151.225.132` | 21.151.225.132 | 50051 / 9003 |
| Router | `sglang-1p1d-router` | `node-21.151.225.132` | hostNetwork | 30001 / 19096 |

Router args (matches live):

```
--prefill http://21.151.225.144:30000 8998
--prefill http://21.151.225.152:30000 8998
--decode  http://21.151.225.132:30000
--decode  http://21.151.225.172:30000
```

`MOONCAKE_MASTER=21.151.225.132:50051` (host IP, not ClusterIP). `SGLANG_HOST_IP` / `MOONCAKE_LOCAL_HOSTNAME` are the worker host IPs.

Router policies: `--prefill-policy round_robin --decode-policy round_robin`. Do not use `cache_aware` on this 2P2D — it stuck all load on P1/D0 (`load()` never increments, shared system prompt ties). Decode has no HiCache. Cross-prefill prefix reuse is Mooncake L3.

Public HTTPRoute: `https://glm52-pd-1p1d.jmpti.woa.com` → Service `sglang-1p1d-router:30001`.

## From-scratch restore

Live STS were patched with `kubectl`, so Helm field-manager conflicts are expected:

```bash
helm upgrade --install sglang-1p1d docker/rocm-mi308x-glm52-pd/chart \
  -n kube-system --force-conflicts \
  --set apiKey=sk-...
```

Do not commit the live API key. Do not `helm install` a second PD release. Do not set `--hicache-ratio 4` (OOM ~2TB host KV on TP8). Scale the router to 0 before a bounce, then back to 1 with `nodeName` pinned — a surge replica can land on eklet.

Each GPU node needs `/data/model/glm52-fp8` and `/data/aiter_configs/a8w8_blockscale_tuned_fmoe_glm5_1_cu80.csv` (cu_num=80 **and** decode tiles `expert=256,topk=8` for token 1..128; see `scripts/merge_decode_fmoe_256_8.py`). Prefill will try `/data/mooncake-patched/patch_evict_backup.py` (v2) and `patch_prefetch_log.py` and continue if they are missing.

## Mooncake L3 layout

Do **not** run a Mooncake master per GPU node. L3 is one shared DRAM pool:

- **1 master** (metadata only) colocated with decode-0 at `21.151.225.132:50051`
- **Store sidecar** (`mooncake_client :50052`) on every GPU node: 64 GB on each prefill, 256 GB on each decode. Cluster L3 ≈ **640 GB**
- Prefill SGLang uses the **in-process** store (`MOONCAKE_GLOBAL_SEGMENT_SIZE=64gb`). Dummy/standalone `setup_dummy` failed on this ROCm image; sidecars still donate extra DRAM. L3 on a prefill node is wiped if that worker restarts; decode sidecars keep their segments.
- Cross-prefill hits RDMA-read the other node's segment via the master replica list
- Decode GPU is still in-flight KV + NEXTN only (no decode HiCache). Decode host DRAM is donated to L3
- Prefetch policy is `wait_complete` with `prefetch_threshold: 1` so PD waits for L3 GET

A per-node master would partition L3 and make cross-P prefix sharing impossible. Stale 0-capacity segments after pod churn are cleared by stopping workers, Recreate-restarting the master, then bringing workers back.

## Images

- Workers / mooncake: `mirrors.tencent.com/ti-platform/sglang-glm52-308x:v0525-wave1`
- Router: `mirrors.tencent.com/ti-platform/sglang-glm52-308x-pd-router:v0516-batch1-tok`
- Rollback workers: `mirrors.tencent.com/ti-platform/sglang-glm52-308x:v0517-gdr-kernel-v1`

Wave 1 (`v0525-wave1`): FlyDSL gfx942 MQA logits, HIP `Event.wait()`, decode MoE CSV 256/8. Chart 0.3.5 unsets `SGLANG_DSA_HIP_DISABLE_PRESHUFFLE` so both P and D use `page_size=64` (Triton ≥ 3.5 in the image). Router is Recreate + `round_robin`.
