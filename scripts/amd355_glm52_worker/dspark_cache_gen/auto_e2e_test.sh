#!/bin/bash
set -e
CKPT_DIR="/data/checkpoints/deepspec/dspark_glm5_2_v9_256exp_clean"
echo "[$(date)] Waiting for step_100 checkpoint..."
until ls "$CKPT_DIR/step_100/config.json" 2>/dev/null; do
    sleep 30
done
echo "[$(date)] Checkpoint step_100 found!"
python3 /data/fix_checkpoint_config.py "$CKPT_DIR/step_100"
sed -i "s|step_180|step_100|g" /data/diagnostics/test_e2e_real.py
echo "[$(date)] Running e2e test..."
IMAGE="lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260704"
docker run --rm --device /dev/kfd --device /dev/dri \
  --network host --shm-size 64G --ipc host --privileged \
  -v /data:/data \
  -e HIP_VISIBLE_DEVICES=0 \
  -e SGLANG_USE_AITER=1 \
  -e PYTHONPATH=/data/sglang_src/python \
  "$IMAGE" \
  bash -c "cd /data/DeepSpec && python3 /data/diagnostics/test_e2e_real.py" 2>&1
echo "[$(date)] E2E test complete"
