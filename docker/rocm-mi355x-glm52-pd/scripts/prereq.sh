#!/usr/bin/env bash
# One-time prerequisites on each node: stop hung training, open firewall to peer,
# tune TCP for high concurrency. Run on BOTH bm1 and bm2.
#
# Usage:
#   ./prereq.sh <peer_ip>     # e.g. on bm1:  ./prereq.sh 149.28.114.238
set -euo pipefail

# 1) Stop the hung 4-node DSpark training container (if present) to free GPUs.
#    Destructive — only removes the named dspark container, leaves others alone.
if docker ps -a --format '{{.Names}}' | grep -q '^glm52_dspark_v10_4node$'; then
  docker rm -f glm52_dspark_v10_4node
  echo "[prereq] removed glm52_dspark_v10_4node"
else
  echo "[prereq] no glm52_dspark_v10_4node container"
fi

# 2) Allow peer PD traffic (bootstrap TCP + serving ports reachability).
PEER_IP="${1:-}"
if [ -n "$PEER_IP" ]; then
  iptables -C INPUT -s "$PEER_IP" -j ACCEPT 2>/dev/null \
    || iptables -I INPUT -s "$PEER_IP" -j ACCEPT
  echo "[prereq] iptables: ACCEPT from $PEER_IP"
else
  echo "[prereq] no peer IP given (./prereq.sh <peer_ip>) — skipping iptables"
fi

# 3) High-concurrency sysctl (best-effort on host).
sysctl -w net.ipv4.tcp_retries2=6 2>/dev/null || true
sysctl -w net.ipv4.tcp_keepalive_time=30 2>/dev/null || true
sysctl -w net.ipv4.tcp_keepalive_intvl=5 2>/dev/null || true
sysctl -w net.ipv4.tcp_keepalive_probes=3 2>/dev/null || true

# 4) Quick RDMA health check.
echo "[prereq] ionic RDMA device states:"
ibv_devinfo 2>/dev/null | grep -E "hca_id|transport|link_layer|state" | paste - - - - || true

echo "[prereq] done. ulimit -n=$(ulimit -n)"
