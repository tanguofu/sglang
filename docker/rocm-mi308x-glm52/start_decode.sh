#!/usr/bin/env bash
# Entrypoint for GLM-5.2-FP8 SGLang Decode worker (PD disaggregation).
# Runs on the decode node — MTP enabled, no prefill cuda graph.

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/data/model/glm52-fp8}
API_KEY=${API_KEY:-sk-46faecc9d0bc4dcd9db6a15c73ae91c8}
PORT=${PORT:-30000}
BOOTSTRAP_PORT=${DISAGG_BOOTSTRAP_PORT:-8998}

echo "============================================"
echo " GLM-5.2-FP8 SGLang DECODE (MI308X gfx942)"
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
    --cuda-graph-bs-decode 1 2 4 8 16 \
    --cuda-graph-max-bs-decode 16 \
    --enable-aiter-allreduce-fusion \
    --enable-fused-qk-norm-rope \
    --schedule-conservativeness 0.5 \
    --kv-cache-dtype fp8_e4m3 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
    --speculative-eagle-topk 1 \
    --max-running-requests 128 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device '{"0":"bnxt_re_bond0","1":"bnxt_re_bond0","2":"bnxt_re_bond0","3":"bnxt_re_bond0","4":"bnxt_re_bond0","5":"bnxt_re_bond0","6":"bnxt_re_bond0","7":"bnxt_re_bond0"}' \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    --num-reserved-decode-tokens 1024 \
    --disable-overlap-schedule \
    --enable-metrics --skip-server-warmup \
    --watchdog-timeout 3600 --log-level info
