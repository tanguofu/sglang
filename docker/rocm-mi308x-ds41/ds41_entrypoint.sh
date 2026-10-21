#!/bin/bash
# Start sglang, compile the per-batch DSV4.1 kernels, then signal readiness.
#
# The readiness file is written only after warm-up, so a readinessProbe on
# /tmp/ds41_warmup.done keeps traffic off a pod that would otherwise block for
# ~70 s on its first request of a new batch size.
set -u

PORT="${DS41_PORT:-30000}"
MODEL="${DS41_MODEL:-DeepSeek-V4.1-Flash}"
READY_FILE="${DS41_READY_FILE:-/tmp/ds41_warmup.done}"
SKIP_WARMUP="${DS41_SKIP_WARMUP:-0}"

rm -f "$READY_FILE"

python3 -m sglang.launch_server "$@" &
SERVER_PID=$!

# A failed server must fail the pod instead of warming a dead process.
for _ in $(seq 1 360); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "server exited during startup" >&2
        wait "$SERVER_PID"
        exit $?
    fi
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        break
    fi
    sleep 5
done

if [ "$SKIP_WARMUP" = "1" ]; then
    echo "warm-up skipped (DS41_SKIP_WARMUP=1)"
else
    python3 /usr/local/bin/warmup_ds41.py --port "$PORT" --model "$MODEL" || true
fi

touch "$READY_FILE"
echo "readiness file written: $READY_FILE"

wait "$SERVER_PID"
