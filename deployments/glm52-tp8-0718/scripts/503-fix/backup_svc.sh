#!/bin/bash
# Backup all 5 services before patching
set -e
echo "=== Backing up services to /tmp/svc_backup_$(date +%Y%m%d_%H%M%S) ==="
BACKUP_DIR="/tmp/svc_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

for svc in \
  sglang-glm52-2tp8-router \
  sglang-glm52-2tp8-sglang \
  sglang-glm52-2tp8-sglang-headless \
  sglang-glm52-2tp8-w2-sglang \
  sglang-glm52-2tp8-w2-sglang-headless; do
  echo "Backing up $svc..."
  kubectl get svc "$svc" -n kube-system -o yaml > "$BACKUP_DIR/${svc}.yaml" 2>&1
done

echo ""
echo "=== Backup complete at $BACKUP_DIR ==="
ls -la "$BACKUP_DIR"
echo "$BACKUP_DIR" > /tmp/svc_backup_dir.txt
