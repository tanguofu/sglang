# aiters-config

Tuned aiter BF16 GEMM config image for gfx942 (MI308X).

## Why

The `bf16_tuned_gemm.csv` is ~430 KiB. Storing it as a ConfigMap has two problems:

1. **`kubectl apply` fails** — the `last-applied-configuration` annotation
   exceeds etcd's 256 KiB limit. Only `kubectl replace` works, which is
   error-prone and invisible to `kubectl diff`.
2. **Helm ownership** — the ConfigMap `aiters-tuned-gemm` is owned by the
   `sglang-glm52-1tp8` release. Any `helm upgrade` on 1tp8 silently overwrites
   it, dropping the N=160 tuning added here.

An image is version-controlled (git-tracked CSV + Dockerfile), rebuildable, and
survives helm operations on any release.

## What's tuned

- 200 gfx942 rows inherited from upstream aiter config (N = 32..32320).
- 17 gfx942 rows **new** for `N=160 K=6144` (shared-expert / projection GEMM),
  tuned on MI308X via
  `/sgl-workspace/aiter/csrc/gemm_a16w16/gemm_a16w16_tune.py` (148 s).

  Padded-M buckets tuned: `1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 576,
  736, 1024, 6016, 8192`.

  Without these, every `N=160` GEMM logged
  `not found tuned config ... will use default config` and fell back to the
  untuned torch default (×8 TP ranks, 56 unique shapes observed in prod).

## Build & push

```bash
docker build -f docker/rocm-mi308x-glm52/aiters-config/Dockerfile \
  -t mirrors.tencent.com/ti-platform/sglang-glm52-308x-aiter-config:n160-v1 \
  docker/rocm-mi308x-glm52/aiters-config/

docker push mirrors.tencent.com/ti-platform/sglang-glm52-308x-aiter-config:n160-v1
```

## Regenerating the CSV

```bash
# 1. copy the driver into a running sglang pod
kubectl cp docker/rocm-mi308x-glm52/scripts/tune_gfx942_n160_driver.py \
  kube-system/sglang-glm52-2tp8-sglang-0:/tmp/tune_n160_driver.py

# 2. probe (print padded-M buckets, no GPU work)
kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- \
  python3 /tmp/tune_n160_driver.py probe

# 3. tune (~150 s on GPU 0)
kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- \
  python3 /tmp/tune_n160_driver.py tune

# 4. pull the result, merge with the existing CSV, rebuild the image
kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- cat /tmp/tune_n160_tuned.csv \
  > /tmp/tune_n160_tuned.csv
# merge: see scripts/patch-2tp8-rolling.sh --tune-gemm for the merge logic
```

## How the sglang pod consumes it

The StatefulSet runs an initContainer from this image that copies the CSV onto
a shared `emptyDir` volume; the main container mounts that volume at
`/etc/aiter-configs` (replacing the ConfigMap mount). See
`chart/templates/sglang-statefulset.yaml` (`aiterConfigImage`).
