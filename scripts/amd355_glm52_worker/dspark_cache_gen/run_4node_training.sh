#!/bin/bash
# =============================================================================
# DSpark GLM-5.2 v9 CLEAN — 4-node training orchestrator
# =============================================================================
# Launches training on all 4 nodes from your laptop.
#
# Prerequisites:
#   1. Run sync_4node_code.sh  (syncs DeepSpec code + fixes to all nodes)
#   2. Run sync_4node_cache.sh (syncs 114G clean cache to node-4, node-9)
#   3. All 4 nodes have 8 free GPUs
#   4. node-9 SSH access works (if not, see "node-9 SSH fix" section below)
#
# Usage:
#   bash run_4node_training.sh          # launch all 4 nodes
#   bash run_4node_training.sh monitor   # check training progress
#   bash run_4node_training.sh stop      # stop training on all nodes
# =============================================================================

set -e

# Node definitions: alias  IP  rank
NODES=(
    "amd-xid18k-node-2  66.42.112.222   0"
    "amd-xid18k-node-4  149.28.124.220  1"
    "amd-xid18k-node-9  104.207.141.239 2"
    "amd-355-worker     144.202.61.0    3"
)

SCRIPT_NAME="start_v9_4node_clean.sh"
CONTAINER_NAME="glm52_dspark_v9_4node_clean"
LOG_FILE="/data/v9_4node_clean_train.log"

MODE=${1:-launch}

case "$MODE" in
    launch)
        echo "[$(date)] === Launching 4-node DSpark training ==="

        # Step 1: Launch master (node-2, rank 0) first
        echo "[$(date)] Launching master: node-2 (rank 0)..."
        ssh amd-xid18k-node-2 "bash /data/${SCRIPT_NAME} 0"

        # Step 2: Wait for master to start listening on MASTER_PORT
        echo "[$(date)] Waiting for master to listen on port 29500..."
        until ssh amd-xid18k-node-2 'ss -tlnp 2>/dev/null | grep -q :29500' 2>/dev/null; do
            sleep 2
            echo -n "."
        done
        echo ""
        echo "[$(date)] Master is listening. Launching workers..."

        # Step 3: Launch remaining 3 nodes in parallel
        ssh amd-xid18k-node-4 "bash /data/${SCRIPT_NAME} 1" &
        ssh amd-xid18k-node-9 "bash /data/${SCRIPT_NAME} 2" &
        ssh amd-355-worker "bash /data/${SCRIPT_NAME} 3" &
        wait

        echo "[$(date)] All 4 nodes launched."
        echo ""
        echo "Monitor:  bash run_4node_training.sh monitor"
        echo "Stop:     bash run_4node_training.sh stop"
        ;;

    monitor)
        echo "[$(date)] === Training progress on all 4 nodes ==="
        for entry in "${NODES[@]}"; do
            read -r alias ip rank <<< "$entry"
            echo ""
            echo "--- $alias (rank $rank) ---"
            ssh -o ConnectTimeout=5 "$alias" \
                "docker ps --filter name=${CONTAINER_NAME} --format '{{.Status}}' 2>/dev/null | head -1; \
                 grep -E 'step=|loss=' ${LOG_FILE} 2>/dev/null | tail -3; \
                 rocm-smi --showuse 2>/dev/null | grep 'GPU use' | head -8" 2>&1 || echo "  (unreachable)"
        done
        ;;

    stop)
        echo "[$(date)] === Stopping training on all 4 nodes ==="
        for entry in "${NODES[@]}"; do
            read -r alias ip rank <<< "$entry"
            echo -n "  $alias (rank $rank): "
            ssh -o ConnectTimeout=5 "$alias" \
                "docker rm -f ${CONTAINER_NAME} 2>/dev/null && echo 'stopped' || echo 'not running'" 2>&1 || echo "unreachable"
        done
        ;;

    logs)
        # Tail logs from master node
        echo "[$(date)] === Tailing master (node-2) training log ==="
        ssh amd-xid18k-node-2 "tail -f ${LOG_FILE}" 2>&1
        ;;

    *)
        echo "Usage: bash $0 {launch|monitor|stop|logs}"
        echo "  launch  - start 4-node training (master first, then workers)"
        echo "  monitor - check status + progress on all nodes"
        echo "  stop    - stop training on all nodes"
        echo "  logs    - tail master training log"
        exit 1
        ;;
esac
