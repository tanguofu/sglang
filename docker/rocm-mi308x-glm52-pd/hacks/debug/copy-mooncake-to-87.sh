#!/bin/bash
set -uo pipefail
LOG=/tmp/copy-mooncake-87.log
exec > >(tee -a "$LOG") 2>&1
echo "=== copy mooncake-patched to 87 start: $(date) ==="

export DISPLAY=:0
export SSH_ASKPASS=/tmp/askpass.sh
export SSH_ASKPASS_REQUIRE=force
SSH_OPTS="-o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no -p 36000"

# ensure target dir exists
ssh $SSH_OPTS root@21.234.171.87 "mkdir -p /data/mooncake-patched"

# rsync the mooncake-patched directory
rsync -av -e "ssh $SSH_OPTS" /data/mooncake-patched/ root@21.234.171.87:/data/mooncake-patched/
RC=$?
echo "mooncake-patched rsync rc=$RC: $(date)"
echo "=== verify ==="
ssh $SSH_OPTS root@21.234.171.87 "ls -la /data/mooncake-patched/ | head -20"

echo "=== DONE: copy mooncake-patched to 87 ==="
echo "MOONCAKE_87_DONE_OK"
