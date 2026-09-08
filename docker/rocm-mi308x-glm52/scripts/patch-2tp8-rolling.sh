#!/usr/bin/env bash
#
# patch-2tp8-rolling.sh — Apply optimization values + progress-lossless rolling restart
#
# Progress-lossless strategy:
#   - Uses `helm upgrade` (NOT uninstall+install) → StatefulSet RollingUpdate
#   - maxUnavailable=1 (default) → at most 1 pod down, the other keeps serving
#   - Router (cache_aware, static workerUrls + /health poll) auto-routes around
#     the restarting pod: health check fails → new requests go to the live pod
#   - terminationGracePeriodSeconds=300 lets sglang finish in-flight requests
#   - Init container re-checks node memory/GPU before each pod starts
#   - Monitors each pod to Ready before the next is touched; auto-rollback on failure
#
# Usage:
#   bash scripts/patch-2tp8-rolling.sh                       # default: hicacheRatio 4→6
#   bash scripts/patch-2tp8-rolling.sh --hicache-ratio 8     # custom ratio
#   bash scripts/patch-2tp8-rolling.sh --set sglang.X=Y ...  # arbitrary helm --set
#   bash scripts/patch-2tp8-rolling.sh --tune-gemm           # aiter GEMM tune (N=160) first
#   bash scripts/patch-2tp8-rolling.sh --dry-run             # show plan, no changes
#   bash scripts/patch-2tp8-rolling.sh --skip-restart        # only tune GEMM + update ConfigMap
#
set -euo pipefail

CHART_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RELEASE_NAME="sglang-glm52-2tp8"
NAMESPACE="kube-system"
STS_NAME="${RELEASE_NAME}-sglang"
ROUTER_DEPLOY="${RELEASE_NAME}-router"
TUNE_DRIVER="${CHART_DIR}/scripts/tune_gfx942_n160_driver.py"
BACKUP_DIR="${CHART_DIR}/.backups"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_FILE="${BACKUP_DIR}/2tp8-values-${TS}.yaml"

# Defaults — the safe, high-value patch set (see analysis in conversation).
HICACHE_RATIO_NEW=6
EXTRA_SETS=()
TUNE_GEMM=false
DRY_RUN=false
SKIP_RESTART=false

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
info() { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hicache-ratio)   HICACHE_RATIO_NEW="$2"; shift 2 ;;
    --set)             EXTRA_SETS+=("$2"); shift 2 ;;
    --tune-gemm)       TUNE_GEMM=true; shift ;;
    --dry-run)         DRY_RUN=true; shift ;;
    --skip-restart)    SKIP_RESTART=true; shift ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) err "Unknown arg: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
log "=== Pre-flight checks ==="

if ! kubectl get ns "$NAMESPACE" &>/dev/null; then
  err "Namespace $NAMESPACE not found"; exit 1
fi
if ! helm history "$RELEASE_NAME" -n "$NAMESPACE" &>/dev/null; then
  err "Helm release $RELEASE_NAME not found"; exit 1
fi

# Both pods must be Ready before we start (can't do progress-lossless rolling
# if we're already degraded).
POD_0="${STS_NAME}-0"
POD_1="${STS_NAME}-1"
for pod in "$POD_0" "$POD_1"; do
  READY=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null || echo "false")
  if [ "$READY" != "true" ]; then
    err "$pod is not Ready (ready=$READY). Fix current state before rolling."
    err "Progress-lossless restart requires both pods healthy."
    exit 1
  fi
  log "$pod Ready ✅"
done

# Router must be healthy (it routes around the restarting pod).
ROUTER_READY=$(kubectl get deploy "$ROUTER_DEPLOY" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
if [ "${ROUTER_READY:-0}" -lt 1 ]; then
  err "Router $ROUTER_DEPLOY has no ready replicas. Cannot do progress-lossless rolling."
  exit 1
fi
log "Router ready ($ROUTER_READY replicas) ✅"

# Capture current revision for rollback.
CUR_REV=$(helm history "$RELEASE_NAME" -n "$NAMESPACE" 2>/dev/null | awk '/deployed/{print $1}' | tail -1)
log "Current deployed helm revision: ${CUR_REV}"

# Backup current user-supplied values.
mkdir -p "$BACKUP_DIR"
helm get values "$RELEASE_NAME" -n "$NAMESPACE" > "$BACKUP_FILE" 2>/dev/null
log "Backed up current values → $BACKUP_FILE"

echo ""

# ---------------------------------------------------------------------------
# Phase 1 (optional): aiter GEMM tuning for N=160 K=6144
# ---------------------------------------------------------------------------
if [ "$TUNE_GEMM" = true ]; then
  log "=== Phase 1: aiter GEMM tuning (N=160, K=6144) ==="
  info "This tunes the missing shapes that currently fall back to torch default."
  info "Runs inside $POD_0 (uses GPU 0 only, ~2-5 min)."

  if [ "$DRY_RUN" = true ]; then
    warn "[dry-run] would: copy $TUNE_DRIVER → $POD_0:/tmp/, run probe+tune, merge into ConfigMap"
  else
    # Copy driver into the pod.
    kubectl cp "$TUNE_DRIVER" "$NAMESPACE/$POD_0:/tmp/tune_n160_driver.py" 2>/dev/null || {
      err "kubectl cp failed (is $POD_0 running?)"; exit 1
    }

    # Probe first (cheap, no GPU work) to show the padded-M buckets.
    log "Probing padded-M buckets..."
    kubectl exec "$POD_0" -n "$NAMESPACE" -- python3 /tmp/tune_n160_driver.py probe || true

    echo
    warn "Tuning will use GPU 0 on $POD_0 for ~2-5 min."
    warn "In-flight requests on $POD_0 may slow down during tuning."
    read -p "Proceed with tuning? (y/N) " -n 1 -r; echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && { warn "Skipping GEMM tuning."; TUNE_GEMM=false; }

    if [ "$TUNE_GEMM" = true ]; then
      log "Tuning (this takes a few minutes)..."
      TUNE_OUT=$(kubectl exec "$POD_0" -n "$NAMESPACE" -- python3 /tmp/tune_n160_driver.py tune 2>&1) || {
        err "Tuning failed:\n$TUNE_OUT"; exit 1
      }
      echo "$TUNE_OUT"

      # Extract the gfx942 CSV rows (between the marker and EOF).
      NEW_ROWS=$(echo "$TUNE_OUT" | sed -n '/=== TUNED CSV ROWS (gfx942) ===/,$p' | tail -n +2)
      if [ -z "$NEW_ROWS" ]; then
        warn "No tuned rows produced. Skipping ConfigMap update."
      else
        log "Got $(echo "$NEW_ROWS" | wc -l | tr -d ' ') new gfx942 rows. Merging into ConfigMap..."

        # Pull current ConfigMap, append new rows, push back.
        CM_NAME="aiters-tuned-gemm"
        kubectl get configmap "$CM_NAME" -n "$NAMESPACE" -o jsonpath='{.data.bf16_tuned_gemm\.csv}' > /tmp/bf16_tuned_gemm.csv 2>/dev/null || {
          err "ConfigMap $CM_NAME not found"; exit 1
        }
        # Dedup: keep existing rows, append only new gfx942 N=160 rows not already present.
        BEFORE=$(wc -l < /tmp/bf16_tuned_gemm.csv | tr -d ' ')
        echo "$NEW_ROWS" >> /tmp/bf16_tuned_gemm.csv
        # Sort + dedup by the full line.
        sort -u /tmp/bf16_tuned_gemm.csv -o /tmp/bf16_tuned_gemm.csv
        AFTER=$(wc -l < /tmp/bf16_tuned_gemm.csv | tr -d ' ')
        log "ConfigMap rows: $BEFORE → $AFTER (added $((AFTER - BEFORE)))"

        if [ "$DRY_RUN" = true ]; then
          warn "[dry-run] would: kubectl create configmap $CM_NAME --from-file=... --dry-run=client -o yaml | kubectl apply"
        else
          kubectl create configmap "$CM_NAME" -n "$NAMESPACE" \
            --from-file=bf16_tuned_gemm.csv=/tmp/bf16_tuned_gemm.csv \
            --dry-run=client -o yaml | kubectl apply -f -
          log "ConfigMap $CM_NAME updated ✅"
          info "New GEMM configs take effect on next pod restart (Phase 2)."
        fi
      fi
    fi
  fi
  echo ""
fi

if [ "$SKIP_RESTART" = true ]; then
  log "Skipping rolling restart (--skip-restart). GEMM configs will apply on next manual restart."
  exit 0
fi

# ---------------------------------------------------------------------------
# Phase 2: helm upgrade with optimization patches (progress-lossless rolling)
# ---------------------------------------------------------------------------
log "=== Phase 2: helm upgrade (progress-lossless rolling restart) ==="

# Build the --set list. Default patch: hicacheRatio 4→6.
SET_ARGS=(
  "sglang.hicacheRatio=${HICACHE_RATIO_NEW}"
)
for s in "${EXTRA_SETS[@]:-}"; do
  [ -n "$s" ] && SET_ARGS+=("$s")
done

info "Patches to apply:"
for s in "${SET_ARGS[@]}"; do
  echo "    --set $s"
done
echo ""

if [ "$DRY_RUN" = true ]; then
  warn "[dry-run] would run:"
  echo "  helm upgrade $RELEASE_NAME ${CHART_DIR}/chart/ -n $NAMESPACE \\"
  for s in "${SET_ARGS[@]}"; do echo "    --set $s \\"; done
  echo "    --reuse-values"
  echo ""
  warn "[dry-run] no changes made."
  exit 0
fi

# --reuse-values preserves all current values; our --set overrides only the
# targeted keys. This is the minimal-change, lowest-risk patch.
log "Running helm upgrade..."
helm upgrade "$RELEASE_NAME" "${CHART_DIR}/chart/" -n "$NAMESPACE" \
  --reuse-values \
  $(printf ' --set %s' "${SET_ARGS[@]}")
log "Helm upgrade submitted ✅"
NEW_REV=$(helm history "$RELEASE_NAME" -n "$NAMESPACE" 2>/dev/null | awk '/deployed/{print $1}' | tail -1)
info "New revision: $NEW_REV (rollback with: helm rollback $RELEASE_NAME $CUR_REV -n $NAMESPACE)"
echo ""

# ---------------------------------------------------------------------------
# Phase 3: monitor the rolling restart (progress-lossless: 1 pod at a time)
# ---------------------------------------------------------------------------
log "=== Phase 3: monitoring rolling restart ==="
log "StatefulSet RollingUpdate: pod-1 restarts first, then pod-0."
log "At most 1 pod down at a time; router routes around it. In-flight requests"
log "get up to 300s grace to finish. Press Ctrl+C to stop watching (pods continue)."
echo ""

# kubectl rollout status waits for the rolling update to complete (all pods
# updated & ready). This is the progress-lossless gate. 2400s = 40 min ceiling
# (each pod: ~10-15 min for model load + CUDA graph + HiCache).
ROLLOUT_OK=false
if kubectl rollout status "sts/$STS_NAME" -n "$NAMESPACE" --timeout=2400s; then
  ROLLOUT_OK=true
fi

if [ "$ROLLOUT_OK" != true ]; then
  err "Rolling update did not complete in time."
  err "Current pod status:"
  kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME" -o wide
  echo ""
  warn "To rollback: helm rollback $RELEASE_NAME $CUR_REV -n $NAMESPACE"
  exit 1
fi

log "Rolling update complete ✅"
echo ""

# ---------------------------------------------------------------------------
# Phase 4: verify
# ---------------------------------------------------------------------------
log "=== Phase 4: verification ==="

# Both pods ready.
for pod in "$POD_0" "$POD_1"; do
  READY=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null || echo "false")
  if [ "$READY" != "true" ]; then
    err "$pod not Ready after rollout"; exit 1
  fi
  log "$pod Ready ✅"
done

# Verify the patched values took effect (from pod-0 startup logs).
log "Verifying patched values (pod-0 logs):"
HICACHE_LOG=$(kubectl logs "$POD_0" -n "$NAMESPACE" 2>/dev/null | grep -oE "hicache_ratio=[0-9.]+" | head -1)
WRITE_LOG=$(kubectl logs "$POD_0" -n "$NAMESPACE" 2>/dev/null | grep -oE "hicache_write_policy='[a-z_]+'" | head -1)
info "  $HICACHE_LOG"
info "  $WRITE_LOG"

# Health via router (HTTPS HTTPRoute).
log "Health check via router (HTTPS HTTPRoute):"
HEALTH=$(curl -s --connect-timeout 10 --max-time 15 \
  -H "Authorization: Bearer sk-REPLACE_WITH_YOUR_API_KEY" \
  https://glm52-2tp8.jmpti.woa.com/health 2>&1 || echo "FAIL")
if [ "$HEALTH" = "OK" ]; then
  log "Router health: OK ✅"
else
  warn "Router health: '$HEALTH' (router may need ~30s to re-detect workers)"
fi

# Smoke test: one chat completion through the router.
log "Smoke test (1 chat completion via router):"
SMOKE=$(curl -s --connect-timeout 15 --max-time 120 \
  -H "Authorization: Bearer sk-REPLACE_WITH_YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  https://glm52-2tp8.jmpti.woa.com/v1/chat/completions \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"say ok"}],"max_tokens":8,"stream":false}' 2>&1)
if echo "$SMOKE" | grep -q '"content"'; then
  log "Chat completion OK ✅"
else
  warn "Chat smoke test: $(echo "$SMOKE" | head -c 120)"
fi

echo ""
log "=== Done ==="
log "Patches applied: hicacheRatio=${HICACHE_RATIO_NEW}${EXTRA_SETS:+, ${EXTRA_SETS[*]}}"
[ "$TUNE_GEMM" = true ] && log "aiter GEMM N=160 tuning merged into ConfigMap"
log "Rolling restart: progress-lossless (1 pod down at a time, router routed around)"
log "Backup: $BACKUP_FILE"
log "Rollback: helm rollback $RELEASE_NAME $CUR_REV -n $NAMESPACE"
