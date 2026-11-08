#!/bin/sh
# Build the mHC #1 launch-fusion image (m20 + 2 .py files).
#
# PURE PREPARATION: refuses to run unless FUSION_BUILD_GO=1 AND M20_DIGEST is
# set.  Nothing in this round builds, pushes, deploys or restarts anything.
# Docker commands run on the ti-builder dind pod, the same as m20/build.sh.
#
# Why patch-from-image: m20 was VERIFIED (2026-11-08) to ship the fork-HEAD
# bytes of deepseek_v4_mhc.py / mhc.py (three independent sources: fork HEAD
# d7dab9171c blobs, live m20 pod, this pinned digest in ti-builder).  We still
# pull m20's OWN bytes from the pinned digest, but patch_mhc_fusion.py now
# asserts they are an insertion-only ancestor of the reviewed worktree and
# EMITS THE REVIEWED BYTES verbatim (the ones GPU torch.equal was measured
# against).  check_mhc_fusion.py PINNS the m20 base md5; a re-pushed tag /
# wrong digest fails the build loudly.  build.sh still requires GO + M20_DIGEST.
set -e

if [ "${FUSION_BUILD_GO:-0}" != "1" ]; then
  echo "REFUSING: pure preparation only. Set FUSION_BUILD_GO=1 after GO + frozen window." >&2
  exit 1
fi
: "${M20_DIGEST:?set M20_DIGEST from the m20 image RepoDigests}"

TAG=${FUSION_IMAGE_TAG:-ds41-prod-1033-mhc-fusion}
IMG=mirrors.tencent.com/ti-platform/sglang-glm52-308x:${TAG}
BASE=mirrors.tencent.com/ti-platform/sglang-glm52-308x:ds41-prod-1033-m20@${M20_DIGEST}
HERE=$(cd "$(dirname "$0")" && pwd)
PROTO=$(cd "$HERE/../../.." && pwd)
CTX=${FUSION_CTX:-/tmp/ds41img-mhc-fusion/build}
REMOTE=/data/ds41img-mhc-fusion/build
KUBE="kubectl --context ti-builder -n default exec dind-builder4 -c builder --"
MODEL_REL=srt/models/deepseek_v4_mhc.py
MHC_REL=kernels/ops/layernorm/mhc.py

echo "=== mhc-fusion build (GO=1) base=$BASE ==="
echo "=== base md5 is PINNED in check_mhc_fusion.py (verified 2026-11-08) ==="
rm -rf "$CTX"
mkdir -p "$CTX/sglang/$(dirname $MODEL_REL)" "$CTX/sglang/$(dirname $MHC_REL)" \
  "$CTX/m20src" "$CTX/baseline" "$CTX/staged"

# 1) m20's OWN bytes for the two files, pulled from the pinned digest.
$KUBE sh -c "docker run --rm --entrypoint cat $BASE \
  /sgl-workspace/sglang/python/sglang/$MODEL_REL" > "$CTX/m20src/deepseek_v4_mhc.py"
$KUBE sh -c "docker run --rm --entrypoint cat $BASE \
  /sgl-workspace/sglang/python/sglang/$MHC_REL" > "$CTX/m20src/mhc.py"
cp "$CTX/m20src/deepseek_v4_mhc.py" "$CTX/baseline/deepseek_v4_mhc.py"
cp "$CTX/m20src/mhc.py" "$CTX/baseline/mhc.py"
echo "=== base md5 pin is enforced by check_mhc_fusion.py below; ==="
echo "    patch_mhc_fusion.py asserts insertion-only delta before emitting bytes."

# 2) minimal, anchor-gated patch + local (no GPU) check + verify, from worktree.
python3 "$HERE/patch_mhc_fusion.py" "$CTX/m20src" "$PROTO" "$CTX/staged"
cp "$CTX/staged/deepseek_v4_mhc.py" "$CTX/sglang/$MODEL_REL"
cp "$CTX/staged/mhc.py" "$CTX/sglang/$MHC_REL"
python3 "$HERE/check_mhc_fusion.py" \
  "$CTX/sglang/$MODEL_REL" "$CTX/baseline/deepseek_v4_mhc.py" \
  "$CTX/sglang/$MHC_REL" "$CTX/baseline/mhc.py"
python3 "$HERE/verify_mhc_fusion.py" \
  "$CTX/sglang/$MHC_REL" "$CTX/sglang/$MODEL_REL"
echo "=== inputs ok: patches + check + verify ALL PASS (GPU-free) ==="

cp "$HERE/Dockerfile" "$CTX/Dockerfile"
cp "$HERE/check_mhc_fusion.py" "$CTX/check_mhc_fusion.py"
cp "$HERE/verify_mhc_fusion.py" "$CTX/verify_mhc_fusion.py"

# 3) Build on the dind pod. FROM m20@digest; the RUN re-asserts the delta.
$KUBE sh -c "rm -rf $REMOTE; mkdir -p $(dirname $REMOTE)"
kubectl --context ti-builder -n default cp "$CTX/." dind-builder4:$REMOTE -c builder
$KUBE sh -c "cd $REMOTE && docker build --build-arg M20_BASE=$BASE -t $IMG . > /tmp/build-mhc-fusion.log 2>&1" || {
  $KUBE sh -c "grep -E '(mhc-fusion:|check_mhc_fusion:|verify_mhc_fusion:|MISSING|MISMATCH|Error)' /tmp/build-mhc-fusion.log | tail -40"
  echo "build_mhc_fusion: docker build failed" >&2; exit 1; }
$KUBE sh -c "grep -E '(mhc-fusion:|check_mhc_fusion:|verify_mhc_fusion:)' /tmp/build-mhc-fusion.log | sed -E 's/^#[0-9]+ [0-9.]+ //'"

echo "=== built $IMG (NOT pushed; push is a separate GO decision) ==="
echo BUILD_MHC_FUSION_DONE
