#!/bin/bash
set -e

CKPT_DIR="/data/checkpoints/deepspec/dspark_glm5_2_v9_256exp_clean"
PROD_CONTAINER="sglang_master_copy"
DSPARK_CONTAINER="glm52_dspark_test"
PORT=30000

echo "============================================"
echo "[$(date)] DSpark Full Pipeline - Step 1: Wait for checkpoint"
echo "============================================"
until ls "$CKPT_DIR/step_100/config.json" 2>/dev/null; do
    sleep 30
done
echo "[$(date)] Checkpoint step_100 found!"

# Fix checkpoint config
python3 /data/fix_checkpoint_config.py "$CKPT_DIR/step_100"

# Step 2: E2E test
echo "============================================"
echo "[$(date)] Step 2: Running E2E test"
echo "============================================"
sed -i "s|step_180|step_100|g" /data/diagnostics/test_e2e_real.py
docker run --rm --device /dev/kfd --device /dev/dri \
  --network host --shm-size 64G --ipc host --privileged \
  -v /data:/data \
  -e HIP_VISIBLE_DEVICES=0 \
  -e SGLANG_USE_AITER=1 \
  -e PYTHONPATH=/data/sglang_src/python \
  lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260704 \
  bash -c "cd /data/DeepSpec && python3 /data/diagnostics/test_e2e_real.py" 2>&1 | tee /data/e2e_result.log

# Check if e2e passed
if grep -q "PASS" /data/e2e_result.log; then
    echo "[$(date)] E2E test PASSED - proceeding to DSpark accept_rate test"
else
    echo "[$(date)] E2E test did not pass - check /data/e2e_result.log"
    echo "[$(date)] Continuing to DSpark test anyway for diagnostic purposes"
fi

# Step 3: DSpark accept_rate test
echo "============================================"
echo "[$(date)] Step 3: Starting DSpark server + accept_rate test"
echo "============================================"

# Stop production server
docker stop $PROD_CONTAINER 2>/dev/null || true
sleep 2

# Start DSpark server
bash /data/start_dspark_test.sh "$CKPT_DIR/step_100" $PORT

# Wait for server to be ready
echo "[$(date)] Waiting for DSpark server..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "[$(date)] DSpark server ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "[$(date)] ERROR: DSpark server failed to start"
        docker logs $DSPARK_CONTAINER 2>&1 | tail -30
        docker start $PROD_CONTAINER 2>/dev/null
        exit 1
    fi
    sleep 5
done

# Run accept_rate test
echo "[$(date)] Running accept_rate test..."
python3 /data/test_accept_rate.py http://localhost:$PORT 20 2>&1 | tee /data/accept_rate_result.log

# Step 4: Restore production server
echo "============================================"
echo "[$(date)] Step 4: Restoring production EAGLE MTP server"
echo "============================================"
docker stop $DSPARK_CONTAINER 2>/dev/null
docker start $PROD_CONTAINER 2>/dev/null
echo "[$(date)] Production server restored"
echo "============================================"
echo "[$(date)] Pipeline complete! Results in /data/e2e_result.log and /data/accept_rate_result.log"
