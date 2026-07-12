#!/usr/bin/env bash
# Build and push Docker images for PD RDMA disaggregation.
# Usage: bash build-and-push.sh [TAG]
#
# Prerequisites:
#   - Colima running with --platform linux/amd64 (for cross-arch build)
#   - Docker logged in to mirrors.tencent.com

set -euo pipefail

TAG=${1:-0712-rdma10}
REGISTRY=mirrors.tencent.com/ti-platform
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=== Building prefill image (${TAG}) ==="
docker build --platform linux/amd64 \
    -f "${REPO_ROOT}/docker/rocm-mi308x-glm52/Dockerfile.prefill" \
    -t "${REGISTRY}/sglang-glm52-308x-prefill:${TAG}" \
    "${REPO_ROOT}"

echo ""
echo "=== Building decode image (${TAG}) ==="
docker build --platform linux/amd64 \
    -f "${REPO_ROOT}/docker/rocm-mi308x-glm52/Dockerfile.decode" \
    -t "${REGISTRY}/sglang-glm52-308x-decode:${TAG}" \
    "${REPO_ROOT}"

echo ""
echo "=== Pushing prefill image ==="
docker push "${REGISTRY}/sglang-glm52-308x-prefill:${TAG}"

echo ""
echo "=== Pushing decode image ==="
docker push "${REGISTRY}/sglang-glm52-308x-decode:${TAG}"

echo ""
echo "=== Done! Images pushed as ${TAG} ==="
echo "Prefill: ${REGISTRY}/sglang-glm52-308x-prefill:${TAG}"
echo "Decode:  ${REGISTRY}/sglang-glm52-308x-decode:${TAG}"
