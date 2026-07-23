#!/bin/bash
set -euo pipefail

# Entrypoint for namespace-filter router.
# Starts the Rust router on port 30081 in the background, then starts the
# Python namespace filter proxy on port 30080 which forwards to the Rust router.

export ROUTER_PORT="${ROUTER_PORT:-30081}"
export PROXY_PORT="${PROXY_PORT:-30080}"

echo "=== SGLang Router with namespace filter proxy ==="
echo "  Rust router: 0.0.0.0:${ROUTER_PORT}"
echo "  Filter proxy: 0.0.0.0:${PROXY_PORT} → 127.0.0.1:${ROUTER_PORT}"
echo

# Start Rust router in background with all original args
# (shift past the script name; remaining args go to launch_router)
python3 -m sglang_router.launch_router \
  --worker-urls \
    http://21.151.225.152:30000 \
    http://21.151.225.172:30000 \
  --policy cache_aware \
  --host 0.0.0.0 \
  --port "${ROUTER_PORT}" \
  --cache-threshold 0.2 \
  --balance-abs-threshold 1 \
  --balance-rel-threshold 1.2 \
  &
ROUTER_PID=$!

echo "Rust router started (PID ${ROUTER_PID})"

# Start the namespace filter proxy (waits for router to be ready)
exec python3 /opt/namespace_filter_proxy.py &
PROXY_PID=$!

echo "Filter proxy started (PID ${PROXY_PID})"

# Wait for either process to exit
wait -n "${ROUTER_PID}" "${PROXY_PID}"
EXIT_CODE=$?

echo "Process exited with code ${EXIT_CODE}, shutting down..."
kill "${ROUTER_PID}" "${PROXY_PID}" 2>/dev/null || true
exit "${EXIT_CODE}"
