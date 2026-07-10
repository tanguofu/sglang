#!/bin/bash
# =============================================================================
# Sync 114G clean cache to node-4 and node-9
# =============================================================================
# The clean cache (/data/dspark_target_cache_v9_coding_clean_merged, 114G, 58
# shards) exists on 355-worker and node-2. It needs to be on ALL 4 nodes.
#
# Strategy: set up inter-node SSH keys, then rsync directly between nodes
# (fast, ~30 GB/s over RDMA). My laptop is only used for key setup.
#
# Prerequisites:
#   - 355-worker and node-2 already have the clean cache
#   - node-4 is SSH-accessible from laptop
#   - node-9 SSH must be fixed first (see "node-9 SSH fix" below)
#
# Usage:
#   bash sync_4node_cache.sh          # sync to all reachable nodes
#   bash sync_4node_cache.sh node-4   # sync to node-4 only
#   bash sync_4node_cache.sh node-9   # sync to node-9 only
# =============================================================================

set -e

SOURCE="amd-355-worker"
CACHE_PATH="/data/dspark_target_cache_v9_coding_clean_merged"
CACHE_SIZE="114G"

# =============================================================================
# Step 1: Set up inter-node SSH keys (355-worker → target)
# =============================================================================
setup_inter_node_ssh() {
    local target_alias=$1
    local target_ip=$2

    echo "[$(date)] Setting up SSH key: $SOURCE → $target_alias"

    # Generate key on 355-worker if not exists
    ssh "$SOURCE" 'test -f ~/.ssh/id_ed25519_inter || ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_inter -N "" -q; cat ~/.ssh/id_ed25519_inter.pub' > /tmp/worker_pubkey.txt 2>/dev/null

    if [ ! -s /tmp/worker_pubkey.txt ]; then
        echo "  ERROR: Could not get public key from $SOURCE"
        return 1
    fi

    # Add the key to target's authorized_keys (via laptop as relay)
    local pubkey
    pubkey=$(cat /tmp/worker_pubkey.txt)
    ssh "$target_alias" "mkdir -p ~/.ssh && grep -q '$pubkey' ~/.ssh/authorized_keys 2>/dev/null || echo '$pubkey' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys" 2>/dev/null \
        && echo "  SSH key added to $target_alias" || echo "  WARNING: Could not add key to $target_alias"

    # Also add target's host key to worker's known_hosts
    ssh "$SOURCE" "ssh-keyscan -H $target_ip 2>/dev/null >> ~/.ssh/known_hosts" 2>/dev/null || true

    # Test the connection
    ssh "$SOURCE" "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i ~/.ssh/id_ed25519_inter root@$target_ip 'echo OK'" 2>&1 | tail -1
}

# =============================================================================
# Step 2: Rsync cache from 355-worker to target
# =============================================================================
sync_cache_to() {
    local target_alias=$1
    local target_ip=$2

    echo ""
    echo "[$(date)] === Syncing $CACHE_SIZE cache to $target_alias ==="

    # Set up inter-node SSH
    setup_inter_node_ssh "$target_alias" "$target_ip"

    # Check if inter-node SSH works
    if ! ssh "$SOURCE" "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i ~/.ssh/id_ed25519_inter root@$target_ip 'echo OK'" 2>/dev/null | grep -q OK; then
        echo "  ERROR: Inter-node SSH not working ($SOURCE → $target_alias)"
        echo "  Falling back to laptop relay (will be slow)..."
        # Fallback: relay through laptop (very slow for 114G)
        # ssh "$SOURCE" "tar cf - -C /data $(basename $CACHE_PATH)" | ssh "$target_alias" "tar xf - -C /data"
        echo "  Aborting. Fix inter-node SSH first."
        return 1
    fi

    # Check if cache already exists on target
    local exists
    exists=$(ssh "$SOURCE" "ssh -i ~/.ssh/id_ed25519_inter root@$target_ip 'ls $CACHE_PATH/manifest.json 2>/dev/null && echo EXISTS || echo MISSING'" 2>/dev/null)

    if [ "$exists" = "EXISTS" ]; then
        echo "  Cache already exists on $target_alias — running rsync to verify/update..."
    else
        echo "  Cache missing on $target_alias — full rsync needed..."
    fi

    # Rsync from 355-worker to target (direct, over RDMA)
    echo "  Starting rsync (this will take ~10-30 min over RDMA)..."
    ssh "$SOURCE" "rsync -avz --progress -e 'ssh -i ~/.ssh/id_ed25519_inter -o StrictHostKeyChecking=no' \
        $CACHE_PATH/ root@$target_ip:$CACHE_PATH/" 2>&1 | tail -20

    # Verify
    local count
    count=$(ssh "$SOURCE" "ssh -i ~/.ssh/id_ed25519_inter root@$target_ip 'ls $CACHE_PATH/*.bin 2>/dev/null | wc -l'" 2>/dev/null)
    echo "  Verification: $count shard files on $target_alias (expected: 58)"
}

# =============================================================================
# Main
# =============================================================================
TARGET=${1:-all}

echo "[$(date)] === DSpark clean cache sync ==="
echo "  Source: $SOURCE ($CACHE_PATH, $CACHE_SIZE)"
echo "  Target: $TARGET"
echo ""

# Define nodes
declare -A NODE_IPS
NODE_IPS["node-4"]="149.28.124.220"
NODE_IPS["node-9"]="104.207.141.239"

case "$TARGET" in
    all)
        for node in node-4 node-9; do
            alias_name="amd-xid18k-${node}"
            ip="${NODE_IPS[$node]}"

            # Check if node is reachable from laptop
            if ssh -o ConnectTimeout=5 "$alias_name" 'echo OK' >/dev/null 2>&1; then
                sync_cache_to "$alias_name" "$ip"
            else
                echo "[$(date)] $node ($ip) unreachable from laptop — skipping"
                echo "  Fix SSH access first, then run: bash sync_4node_cache.sh $node"
            fi
        done
        ;;
    node-4|node-9)
        alias_name="amd-xid18k-${TARGET}"
        ip="${NODE_IPS[$TARGET]}"
        if ssh -o ConnectTimeout=5 "$alias_name" 'echo OK' >/dev/null 2>&1; then
            sync_cache_to "$alias_name" "$ip"
        else
            echo "ERROR: $TARGET unreachable. Fix SSH access first."
            exit 1
        fi
        ;;
    *)
        echo "Usage: bash $0 {all|node-4|node-9}"
        exit 1
        ;;
esac

echo ""
echo "[$(date)] === Cache sync complete ==="
