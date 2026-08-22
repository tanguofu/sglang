#!/usr/bin/env bash
# Entrypoint for GLM-5.2-FP8 SGLang worker on AMD MI355X.
# Exact reproduction of amd-355-master (216.128.153.58) launch command.
#
# All patches are baked into the image at build time — no runtime patching.
# All environment variables are pre-set in the Dockerfile.
# This script only needs runtime-overridable params: MODEL_PATH, API_KEY, PORT.
#
# Usage:
#   docker run -d ... sglang-glm52-master-snapshot:latest
#   docker run -d ... -e PORT=30001 -e MODEL_PATH=/data/models/GLM-5.2-FP8 sglang-glm52-master-snapshot:latest

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/data/models/GLM-5.2-FP8}
API_KEY=${API_KEY:-sk-REPLACE_WITH_YOUR_API_KEY}
PORT=${PORT:-30000}

echo "============================================"
echo " GLM-5.2-FP8 SGLang Worker (master-snapshot)"
echo "============================================"
echo " Model:  $MODEL_PATH"
echo " Port:   $PORT"
echo " Image:  pre-patched (no runtime patching)"
echo " AITER:  9127c94a1 (base image, not upgraded)"
echo " Patches: 01-05 + 06a + 06b + 06c + 06d + Gen1/Gen2"
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
