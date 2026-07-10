#!/bin/bash
# =============================================================================
# Sync DeepSpec code + fixes to all 4 training nodes
# =============================================================================
# Copies from 355-worker (source of truth) to node-2, node-4, node-9.
# Also applies the base_trainer CPUOffload env var patch on all nodes.
#
# Files synced:
#   /data/DeepSpec/deepspec/data/parser.py              (GLM5 ChatTemplate fix)
#   /data/DeepSpec/deepspec/modeling/dspark/glm5/modeling.py  (KV projection fix)
#   /data/DeepSpec/deepspec/trainer/dspark_trainer.py   (nan_to_num fix)
#   /data/DeepSpec/deepspec/trainer/base_trainer.py     (CPUOffload env var patch)
#   /data/DeepSpec/deepspec/trainer/ckpt_manager.py     (checkpoint dir fix)
#   /data/DeepSpec/config/dspark/dspark_glm5_2_v9_clean.py
#   /data/DeepSpec/config/dspark/dspark_glm5_2_v9_clean_4node.py
#   /data/DeepSpec/run_v9_manual.py
# =============================================================================

set -e

SOURCE="amd-355-worker"
TARGETS="amd-xid18k-node-2 amd-xid18k-node-4 amd-xid18k-node-9"

# Files to sync (relative to /data/DeepSpec/)
FILES=(
    "deepspec/data/parser.py"
    "deepspec/modeling/dspark/glm5/modeling.py"
    "deepspec/trainer/dspark_trainer.py"
    "deepspec/trainer/base_trainer.py"
    "deepspec/trainer/ckpt_manager.py"
    "config/dspark/dspark_glm5_2_v9_clean.py"
    "config/dspark/dspark_glm5_2_v9_clean_4node.py"
    "run_v9_manual.py"
)

echo "[$(date)] === Syncing DeepSpec code from $SOURCE to all nodes ==="

# Step 1: Copy the 4-node config to 355-worker first (it's the source)
echo "[$(date)] Copying 4-node config to $SOURCE..."
scp /Users/guofutan/ai-frameworks/sglang/scripts/amd355_glm52_worker/dspark_cache_gen/dspark_glm5_2_v9_clean_4node.py \
    ${SOURCE}:/data/DeepSpec/config/dspark/dspark_glm5_2_v9_clean_4node.py

# Step 2: Copy the launch script to all nodes
echo "[$(date)] Copying launch script to all nodes..."
for node in $SOURCE $TARGETS; do
    scp /Users/guofutan/ai-frameworks/sglang/scripts/amd355_glm52_worker/dspark_cache_gen/start_v9_4node_clean.sh \
        ${node}:/data/start_v9_4node_clean.sh 2>/dev/null && echo "  $node: launch script OK" || echo "  $node: FAILED (SSH)"
done

# Step 3: Sync code files from 355-worker to each target node
for target in $TARGETS; do
    echo ""
    echo "[$(date)] Syncing to $target..."

    # Check SSH connectivity
    if ! ssh -o ConnectTimeout=5 "$target" 'echo OK' >/dev/null 2>&1; then
        echo "  WARNING: Cannot SSH to $target — skipping (fix SSH access first)"
        continue
    fi

    for file in "${FILES[@]}"; do
        # Use 355-worker as source, pipe through SSH to target
        ssh "$SOURCE" "cat /data/DeepSpec/$file" | \
            ssh "$target" "mkdir -p /data/DeepSpec/$(dirname $file) && cat > /data/DeepSpec/$file" 2>/dev/null \
            && echo "  $file: OK" || echo "  $file: FAILED"
    done

    # Step 4: Apply base_trainer CPUOffload env var patch
    echo "  Patching base_trainer.py (CPUOffload env var)..."
    ssh "$target" 'python3 -c "
import os
path = \"/data/DeepSpec/deepspec/trainer/base_trainer.py\"
with open(path) as f:
    content = f.read()
if \"DS_CPU_OFFLOAD\" in content:
    print(\"    Already patched\")
else:
    old = \"cpu_offload=CPUOffload(offload_params=True),\"
    new = \"cpu_offload=CPUOffload(offload_params=os.environ.get(\\\"DS_CPU_OFFLOAD\\\", \\\"0\\\") == \\\"1\\\"),\"
    if old in content:
        # Make sure os is imported
        if \"import os\" not in content:
            content = \"import os\\n\" + content
        content = content.replace(old, new)
        with open(path, \"w\") as f:
            f.write(content)
        print(\"    Patched successfully\")
    else:
        print(\"    WARNING: target line not found\")
"' 2>&1

    # Step 5: Verify fixes are in place
    echo "  Verifying fixes..."
    ssh "$target" '
        echo -n "    parser.py (.find): "; grep -c "\.find(" /data/DeepSpec/deepspec/data/parser.py 2>/dev/null || echo "MISSING"
        echo -n "    modeling.py (KV proj): "; grep -c "compressed_kv_ctx" /data/DeepSpec/deepspec/modeling/dspark/glm5/modeling.py 2>/dev/null || echo "MISSING"
        echo -n "    nan_to_num: "; grep -c "nan_to_num" /data/DeepSpec/deepspec/trainer/dspark_trainer.py 2>/dev/null || echo "MISSING"
        echo -n "    CPUOffload envvar: "; grep -c "DS_CPU_OFFLOAD" /data/DeepSpec/deepspec/trainer/base_trainer.py 2>/dev/null || echo "MISSING"
        echo -n "    4node config: "; ls /data/DeepSpec/config/dspark/dspark_glm5_2_v9_clean_4node.py 2>/dev/null && echo "OK" || echo "MISSING"
    ' 2>&1
done

echo ""
echo "[$(date)] === Code sync complete ==="
echo "Next: bash sync_4node_cache.sh  (sync 114G clean cache to node-4, node-9)"
