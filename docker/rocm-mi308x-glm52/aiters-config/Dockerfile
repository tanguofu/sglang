# Aiter GEMM tuned-config image for gfx942 (MI308X)
#
# Bakes the tuned bf16_tuned_gemm.csv into a tiny image. An initContainer copies
# the CSV onto a shared emptyDir volume; the main sglang container mounts that
# volume at /etc/aiter-configs (replacing the ConfigMap mount).
#
# Why an image instead of a ConfigMap:
#   - ConfigMap apply fails: last-applied-configuration annotation exceeds the
#     256 KiB etcd limit (the CSV is ~430 KiB). Only `kubectl replace` works,
#     which is error-prone and invisible to `kubectl diff`.
#   - The ConfigMap is owned by the 1tp8 helm release; `helm upgrade 1tp8`
#     silently overwrites it, dropping the N=160 tuning.
#   - An image is version-controlled (git-tracked CSV + Dockerfile), rebuildable,
#     and survives helm operations on any release.
#
# Build (from repo root):
#   docker build -f docker/rocm-mi308x-glm52/aiters-config/Dockerfile \
#     -t mirrors.tencent.com/ti-platform/sglang-glm52-308x-aiter-config:n160-v1 \
#     docker/rocm-mi308x-glm52/aiters-config/
#
# Push:
#   docker push mirrors.tencent.com/ti-platform/sglang-glm52-308x-aiter-config:n160-v1
#
# CSV contents (gfx942 rows):
#   - 200 rows inherited from the upstream aiter config (N=32..32320)
#   - 17 rows NEW for N=160 K=6144 (shared-expert / projection GEMM), tuned
#     on MI308X gfx942 via /sgl-workspace/aiter/csrc/gemm_a16w16/gemm_a16w16_tune.py
#     (148s, 17 padded-M buckets: 1,2,4,8,16,32,48,64,80,96,112,128,576,736,1024,6016,8192)
#     Without these, every N=160 GEMM fell back to the untuned torch default.
FROM busybox:1.36
COPY bf16_tuned_gemm.csv /aiter-configs/bf16_tuned_gemm.csv
# health: file exists and is non-empty
HEALTHCHECK --interval=30s --timeout=3s CMD test -s /aiter-configs/bf16_tuned_gemm.csv
