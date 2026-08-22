#!/usr/bin/env bash
# Entrypoint for SGLang Router in PD disaggregation mode.
# Routes requests: prefill → KV transfer → decode → response.

set -euo pipefail

PREFILL_URL=${PREFILL_URL:-http://NODE_PREFILL_0_IP:30000}
DECODE_URL=${DECODE_URL:-http://NODE_DECODE_0_IP:30000}
PORT=${ROUTER_PORT:-30001}

echo "============================================"
echo " SGLang Router (PD Disaggregation)"
echo "============================================"
echo " Prefill: $PREFILL_URL"
echo " Decode:  $DECODE_URL"
echo " Port:    $PORT"

exec python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "$PREFILL_URL" \
    --decode "$DECODE_URL" \
    --host 0.0.0.0 --port "$PORT"
