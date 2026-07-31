#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/data/model/glm52-fp8}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-/data/model/glm52-dspark-redhat}
API_KEY=${API_KEY:-sk-46faecc9d0bc4dcd9db6a15c73ae91c8}
PORT=${PORT:-30000}

echo "============================================"
echo " GLM-5.2 DSpark Worker (MI308X gfx942)"
echo "============================================"
echo " Target: $MODEL_PATH"
echo " Draft:  $DRAFT_MODEL_PATH"
echo " Port:   $PORT"
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
    --mem-fraction-static 0.85 \
    --attention-backend dsa \
    --dsa-prefill-backend tilelang \
    --dsa-decode-backend tilelang \
    --enable-aiter-allreduce-fusion \
    --enable-fused-qk-norm-rope \
    --schedule-conservativeness 1.0 \
    --kv-cache-dtype fp8_e4m3 \
    --disable-cuda-graph \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
    --speculative-dspark-block-size 8 \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1 \
    --max-running-requests 128 \
    --enable-metrics \
    --skip-server-warmup \
    --watchdog-timeout 3600 \
    --log-level info
