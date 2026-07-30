#!/usr/bin/env bash
#
# upgrade-2tp8-optimized.sh — Uninstall + reinstall 2tp8 with optimized params
#
# Usage:
#   cd /Users/guofutan/ai-frameworks/sglang/docker/rocm-mi308x-glm52
#   bash scripts/upgrade-2tp8-optimized.sh
#
# This script:
#   1. Uninstalls the existing sglang-glm52-2tp8 helm release
#   2. Waits for pods to be cleaned up
#   3. Reinstalls with values-glm52-2tp8-optimized.yaml
#   4. Monitors pod startup until both workers are ready
#   5. Verifies key parameters and health
#
# Prerequisites:
#   - kubectl configured with access to kube-system namespace
#   - helm 3.x installed
#   - Chart templates updated (sglang-router.yaml, sglang-statefulset.yaml)
#   - values-glm52-2tp8-optimized.yaml in chart/ directory
#
# Note: The responses-fix ConfigMap (sglang-glm52-2tp8-responses-fix) was created
# outside of Helm and will survive the uninstall. It will be referenced by the
# new StatefulSet via the responsesFix.enabled flag.
#
set -euo pipefail

CHART_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RELEASE_NAME="sglang-glm52-2tp8"
NAMESPACE="kube-system"
VALUES_FILE="${CHART_DIR}/chart/values-glm52-2tp8-optimized.yaml"
API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
log "=== Pre-flight checks ==="

if [ ! -f "$VALUES_FILE" ]; then
  err "Values file not found: $VALUES_FILE"
  exit 1
fi

if ! kubectl get ns "$NAMESPACE" &>/dev/null; then
  err "Namespace $NAMESPACE not found"
  exit 1
fi

# Check if responses-fix ConfigMap exists (it should survive uninstall)
if kubectl get configmap "${RELEASE_NAME}-responses-fix" -n "$NAMESPACE" &>/dev/null; then
  log "responses-fix ConfigMap exists ✅ (will survive uninstall)"
else
  warn "responses-fix ConfigMap NOT found! /v1/responses patch will be missing after reinstall."
  warn "If this is intentional, ignore. Otherwise, create it before proceeding."
  read -p "Continue anyway? (y/N) " -n 1 -r
  echo
  [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# Check if aiters-tuned-gemm ConfigMap exists (shared, survives uninstall)
if kubectl get configmap aiters-tuned-gemm -n "$NAMESPACE" &>/dev/null; then
  log "aiters-tuned-gemm ConfigMap exists ✅"
else
  warn "aiters-tuned-gemm ConfigMap NOT found! GEMM tuning will be missing."
fi

echo ""

# ---------------------------------------------------------------------------
# Step 1: Uninstall existing release
# ---------------------------------------------------------------------------
log "=== Step 1: Uninstalling ${RELEASE_NAME} ==="

# Get current revision for rollback reference
CURRENT_REV=$(helm history "$RELEASE_NAME" -n "$NAMESPACE" 2>/dev/null | tail -1 | awk '{print $1}')
log "Current revision: ${CURRENT_REV:-unknown}"

helm uninstall "$RELEASE_NAME" -n "$NAMESPACE"
log "Helm release uninstalled."

# ---------------------------------------------------------------------------
# Step 2: Wait for pods to be cleaned up
# ---------------------------------------------------------------------------
log "=== Step 2: Waiting for pods to be cleaned up ==="

for i in $(seq 1 30); do
  PODS=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME" --no-headers 2>/dev/null | wc -l)
  if [ "$PODS" -eq 0 ]; then
    log "All pods cleaned up ✅"
    break
  fi
  echo -n "."
  sleep 5
done
echo ""

# Force delete any stuck pods
STUCK_PODS=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME" --no-headers 2>/dev/null | awk '{print $1}')
if [ -n "$STUCK_PODS" ]; then
  warn "Force deleting stuck pods: $STUCK_PODS"
  for pod in $STUCK_PODS; do
    kubectl delete pod "$pod" -n "$NAMESPACE" --force --grace-period=0 2>/dev/null || true
  done
  sleep 5
fi

# Wait for GPU resources to be released
log "Waiting for GPU resources to be released..."
sleep 10

echo ""

# ---------------------------------------------------------------------------
# Step 3: Install with optimized values
# ---------------------------------------------------------------------------
log "=== Step 3: Installing ${RELEASE_NAME} with optimized params ==="

helm install "$RELEASE_NAME" "${CHART_DIR}/chart/" -n "$NAMESPACE" -f "$VALUES_FILE"
log "Helm install submitted ✅"

echo ""

# ---------------------------------------------------------------------------
# Step 4: Monitor pod startup
# ---------------------------------------------------------------------------
log "=== Step 4: Monitoring pod startup ==="
log "This will take ~10-15 minutes per pod (model load + CUDA graph + HiCache)"
log "Press Ctrl+C to stop monitoring (pods will continue starting)"

POD_0="${RELEASE_NAME}-sglang-0"
POD_1="${RELEASE_NAME}-sglang-1"
READY_COUNT=0
TOTAL_PODS=2

for i in $(seq 1 180); do  # 30 min max
  STATUS_0=$(kubectl get pod "$POD_0" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
  READY_0=$(kubectl get pod "$POD_0" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null || echo "false")
  STATUS_1=$(kubectl get pod "$POD_1" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
  READY_1=$(kubectl get pod "$POD_1" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null || echo "false")

  echo -e "[$(date +%H:%M:%S)] pod-0: ${STATUS_0}/${READY_0}  pod-1: ${STATUS_1}/${READY_1}"

  if [ "$READY_0" = "true" ] && [ "$READY_1" = "true" ]; then
    log "Both pods are ready! ✅✅"
    READY_COUNT=2
    break
  fi

  # Check for crash loop
  RESTARTS_0=$(kubectl get pod "$POD_0" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].restartCount}' 2>/dev/null || echo "0")
  RESTARTS_1=$(kubectl get pod "$POD_1" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].restartCount}' 2>/dev/null || echo "0")
  if [ "${RESTARTS_0:-0}" -ge 3 ] 2>/dev/null || [ "${RESTARTS_1:-0}" -ge 3 ] 2>/dev/null; then
    err "Pod is crash-looping (restarts: pod-0=${RESTARTS_0}, pod-1=${RESTARTS_1})"
    err "Check logs: kubectl logs ${POD_0} -n ${NAMESPACE} --previous"
    exit 1
  fi

  sleep 10
done

if [ "$READY_COUNT" -lt 2 ]; then
  err "Timeout waiting for pods to be ready"
  err "Current status:"
  kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME"
  exit 1
fi

echo ""

# ---------------------------------------------------------------------------
# Step 5: Verify deployment
# ---------------------------------------------------------------------------
log "=== Step 5: Verifying deployment ==="

# Check pod status
log "Pod status:"
kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME" -o wide

echo ""

# Check key startup parameters from pod-0 logs
log "Key startup parameters (pod-0):"
kubectl logs "$POD_0" -n "$NAMESPACE" 2>/dev/null | grep -E "max_total_num_tokens|available_gpu_mem|fired up" | head -3

echo ""

# Verify eagle patch
log "Eagle patch:"
kubectl logs "$POD_0" -n "$NAMESPACE" 2>/dev/null | grep "eagle_utils.py" | head -1

# Verify write_back policy
log "HiCache write policy:"
kubectl logs "$POD_0" -n "$NAMESPACE" 2>/dev/null | grep -o "write_back\|write_through_selective" | head -1

# Verify hicacheRatio
log "HiCache host memory per rank:"
kubectl logs "$POD_0" -n "$NAMESPACE" 2>/dev/null | grep "Allocating.*host memory for hierarchical KV cache" | head -1

echo ""

# Health check via gateway
log "Health check (via gateway):"
HEALTH=$(curl -s --connect-timeout 10 --max-time 15 http://glm52-2tp8.jmpti.woa.com/health 2>&1)
if [ "$HEALTH" = "OK" ]; then
  log "Gateway health: OK ✅"
else
  warn "Gateway health: '$HEALTH' (may need a few more minutes for router to detect workers)"
fi

# Models check
log "Models check:"
MODELS=$(curl -s --connect-timeout 10 --max-time 15 http://glm52-2tp8.jmpti.woa.com/v1/models 2>&1 | head -1)
if echo "$MODELS" | grep -q "glm-5.2"; then
  log "Models endpoint: OK ✅"
else
  warn "Models endpoint: '$MODELS'"
fi

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "=== Deployment Summary ==="
log "Release: ${RELEASE_NAME} (revision $(helm history "$RELEASE_NAME" -n "$NAMESPACE" 2>/dev/null | tail -1 | awk '{print $1}'))"
log "Pods: $(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME" --no-headers 2>/dev/null | wc -l) pods"
log ""
log "Key parameter changes applied:"
log "  eaglePatch:          false → true"
log "  memFractionStatic:   0.85 → 0.75  (FIX: 恢复 OOM fix 安全垫)"
log "  chunkedPrefillSize:  32768 → 16384"
log "  prefillMaxRequests:   32 → 8"
log "  scheduleConservativeness: 0.7 → 1.0  (FIX: 恢复 OOM fix 保守调度)"
log "  speculativeNumSteps: 3 → 2  (FIX: rev10 验证 10/10, benchmark 优于 3)"
log "  maxRunningRequests:  48 → 64  (CODEX: 高并发 decode)"
log "  cudaGraphMaxBsDecode: 32 → 48  (CODEX: 覆盖 bs 33-48)"
log "  cudaGraphBsDecode:    1-32 → 1-48 (added 48)"
log "  cudaGraphBsPrefill:  1-32 → 4-32 (removed 1,2,3)"
log "  hicacheRatio:         2 → 4"
log "  hicacheWritePolicy:   write_through_selective → write_back"
log "  watchdogTimeout:      3600 → 1200"
log ""
log "  Router (CODEX 优化):"
log "    maxConcurrentRequests: 128 → 256"
log "    rateLimitTokensPerSecond: 128 → 256"
log "    queueSize: 64 → 128"
log "    queueTimeoutSecs: 120 → 60  (快速失败)"
log "    cacheThreshold: 0.2 → 0.1  (codex 相似 prefix 多，更激进匹配)"
log ""
log "Preserved (no change):"
log "  image: breakable | router: pd-resp-msg-v1"
log "  enableCacheReport, SGLANG_OPT_USE_AITER_INDEXER, FLYDSL_FP8_MQA_LOGITS_VARIANT"
log "  responses-fix ConfigMap, gateway, tolerations, replicas=2"
log ""
log "✅ Upgrade complete!"
log ""
log "Rollback if needed:"
log "  helm uninstall ${RELEASE_NAME} -n ${NAMESPACE}"
log "  helm install ${RELEASE_NAME} ${CHART_DIR}/chart/ -n ${NAMESPACE} -f <old-values.yaml>"
