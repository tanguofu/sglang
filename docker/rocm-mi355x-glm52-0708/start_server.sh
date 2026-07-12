#!/usr/bin/env bash
# Entrypoint for GLM-5.2-FP8 SGLang worker on AMD MI355X.
#
# All environment variables are pre-set in the Dockerfile (ENV).
# This script only needs runtime-overridable params: MODEL_PATH, API_KEY, PORT.
#
# No runtime patching — all code patches are baked into the image.
#
# Usage:
#   docker run -d ... sglang-glm52-0708:latest
#   docker run -d ... -e PORT=30001 -e MODEL_PATH=/data/models/GLM-5.2-FP8 sglang-glm52-0708:latest

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/data/models/GLM-5.2-FP8}
API_KEY=${API_KEY:-sk-46faecc9d0bc4dcd9db6a15c73ae91c8}
PORT=${PORT:-30000}

echo "============================================"
echo " GLM-5.2-FP8 SGLang Worker (0708-opt)"
echo "============================================"
echo " Model:  $MODEL_PATH"
echo " Port:   $PORT"
echo " Image:  pre-patched (no runtime patching)"
echo "============================================"

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
    --enable-aiter-allreduce-fusion --enable-mixed-chunk \
    --chunked-prefill-size 32768 \
    --enable-fused-qk-norm-rope \
    --schedule-conservativeness 0.5 \
    --prefill-max-requests 32 --max-prefill-tokens 32768 \
    --kv-cache-dtype fp8_e4m3 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
    --speculative-eagle-topk 1 \
    --cuda-graph-backend-prefill breakable \
    --max-running-requests 32 \
    --cuda-graph-bs-prefill 4 8 16 32 \
    --enable-metrics --skip-server-warmup \
    --watchdog-timeout 3600 --log-level info
