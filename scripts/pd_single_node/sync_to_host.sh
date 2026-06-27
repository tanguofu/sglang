#!/bin/bash
# Sync repo scripts to a bare-metal host's /data layout.
# Usage: ./sync_to_host.sh root@216.128.154.57
set -euo pipefail
HOST=${1:?Usage: sync_to_host.sh root@HOST}
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

ssh "$HOST" 'mkdir -p /data/pd_single_node/logs'
scp "$REPO_ROOT/scripts/pd_single_node/"*.sh "$REPO_ROOT/scripts/pd_single_node/"*.py "$HOST:/data/pd_single_node/"
scp "$REPO_ROOT/scripts/patch_mori_pp_kv_slices.py "$HOST:/data/patch_mori_pp_kv_slices.py"
scp "$REPO_ROOT/scripts/pd_single_node/patch_glm_config.py" "$HOST:/data/patch_glm_config.py"
scp "$REPO_ROOT/scripts/pd_single_node/patch_pp_missing_layer.py" "$HOST:/data/patch_pp_missing_layer.py"
ssh "$HOST" 'chmod +x /data/pd_single_node/*.sh /data/pd_single_node/*.py'
echo "Synced to $HOST:/data/pd_single_node/ and /data/patch_*.py"
