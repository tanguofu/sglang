#!/bin/bash
set -uo pipefail
LOG=/tmp/parallel_rsync_87.log
exec > >(tee -a "$LOG") 2>&1
echo "=== parallel rsync to 87 start: $(date) ==="

export DISPLAY=:0
export SSH_ASKPASS=/tmp/askpass.sh
export SSH_ASKPASS_REQUIRE=force
SSH_OPTS="-o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no -p 36000"

SRC=/data/model/glm52-fp8/
DST=root@21.234.171.87:/data/model/glm52-fp8/

ssh $SSH_OPTS root@21.234.171.87 "mkdir -p /data/model/glm52-fp8"

echo "--- sync small files first (non-safetensors) ---"
rsync -av -e "ssh $SSH_OPTS" --exclude="*.safetensors" $SRC $DST
echo "small files done: $(date)"

echo "--- parallel sync safetensors (8 streams) ---"
cd /data/model/glm52-fp8
ls *.safetensors | xargs -P 8 -I {} rsync -av --partial --ignore-existing -e "ssh $SSH_OPTS" {} root@21.234.171.87:/data/model/glm52-fp8/{}
echo "safetensors parallel done: $(date)"

echo "--- verify count ---"
ssh $SSH_OPTS root@21.234.171.87 "ls /data/model/glm52-fp8/*.safetensors | wc -l"

echo "=== parallel rsync to 87 end: $(date) ==="
echo "PARALLEL_RSYNC_87_DONE_OK"
