#!/usr/bin/env bash
# Entrypoint for GLM-5.2-FP8 SGLang Prefill worker (PD disaggregation).
# Runs on the prefill node — no MTP, no decode cuda graph.

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/data/model/glm52-fp8}
API_KEY=${API_KEY:-sk-46faecc9d0bc4dcd9db6a15c73ae91c8}
PORT=${PORT:-30000}
BOOTSTRAP_PORT=${DISAGG_BOOTSTRAP_PORT:-8998}

echo "============================================"
echo " GLM-5.2-FP8 SGLang PREFILL (MI308X gfx942)"
echo "============================================"
echo " Model: $MODEL_PATH  Port: $PORT  Bootstrap: $BOOTSTRAP_PORT"

exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --model-impl sglang \
    --served-model-name glm-5.2 \
    --api-key "$API_KEY" \
    --tp-size 8 --pp-size 1 --trust-remote-code \
    --host 0.0.0.0 --port "$PORT" \
    --context-length 1048576 \
    --tool-call-parser glm47 --reasoning-parser glm45 \
    --mem-fraction-static 0.88 \
    --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 12 16 \
    --cuda-graph-max-bs-decode 16 \
    --enable-aiter-allreduce-fusion \
    --chunked-prefill-size 32768 \
    --enable-fused-qk-norm-rope \
    --schedule-conservativeness 0.5 \
    --prefill-max-requests 32 --max-prefill-tokens 32768 \
    --kv-cache-dtype fp8_e4m3 \
    --cuda-graph-backend-prefill breakable \
    --max-running-requests 128 \
    --cuda-graph-bs-prefill 4 8 16 32 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    --enable-metrics --skip-server-warmup \
    --watchdog-timeout 3600 --log-level info
