#!/bin/bash
set -uo pipefail
RESULT=/data/pd_single_node/logs/validation_results.txt
: > "$RESULT"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$RESULT"; }

for SCHEME in PD-1a PD-1b PD-1d; do
  log "========== $SCHEME (single-container, mori XGMI) =========="
  START=$(date +%s)
  if bash /data/pd_single_node/run_single_container.sh "$SCHEME" mori > "/data/pd_single_node/logs/${SCHEME}-deploy.log" 2>&1; then
    END=$(date +%s)
    log "DEPLOY OK in $((END-START))s"
    SMOKE_START=$(date +%s)
    if docker exec sglang_pd_stack python3 /data/pd_single_node/smoke_test.py "http://127.0.0.1:8000" "$SCHEME" > "/data/pd_single_node/logs/${SCHEME}-smoke.log" 2>&1; then
      SMOKE_END=$(date +%s)
      log "SMOKE OK in $((SMOKE_END-SMOKE_START))s"
      cat "/data/pd_single_node/logs/${SCHEME}-smoke.log" >> "$RESULT"
    else
      log "SMOKE FAIL in $(( $(date +%s)-SMOKE_START ))s"
      cat "/data/pd_single_node/logs/${SCHEME}-smoke.log" >> "$RESULT" 2>/dev/null || true
      docker exec sglang_pd_stack tail -30 "/data/pd_single_node/logs/${SCHEME}_mori_prefill.log" >> "$RESULT" 2>/dev/null || true
      docker exec sglang_pd_stack tail -30 "/data/pd_single_node/logs/${SCHEME}_mori_decode.log" >> "$RESULT" 2>/dev/null || true
      docker exec sglang_pd_stack grep -E "Connection timed out|ibverbs|XGMI|Auto-created XGMI" "/data/pd_single_node/logs/${SCHEME}_mori"*.log 2>/dev/null | tail -10 >> "$RESULT" || true
    fi
    log "--- decode KV ---"
    docker exec sglang_pd_stack grep -iE "KV Cache is allocated|Memory pool|#tokens" "/data/pd_single_node/logs/${SCHEME}_mori_decode.log" 2>/dev/null | tail -3 >> "$RESULT" || true
  else
    END=$(date +%s)
    log "DEPLOY FAIL in $((END-START))s"
    tail -30 "/data/pd_single_node/logs/${SCHEME}-deploy.log" >> "$RESULT"
    docker logs sglang_pd_stack 2>&1 | tail -20 >> "$RESULT" 2>/dev/null || true
  fi
  bash /data/pd_single_node/stop_all.sh >> "$RESULT" 2>&1
  sleep 15
done
log "========== DONE =========="
docker ps --format "table {{.Names}}\t{{.Status}}" >> "$RESULT"
