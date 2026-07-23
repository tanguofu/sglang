#!/bin/bash
# Run py-spy on rank-0 scheduler while driving the decode workload.
# Captures Python-layer stacks for operation-level attribution.
set -e
OUT=/data/prof19
R0=$(ps -eo pid,args | grep "scheduler_TP0" | grep -v grep | awk '{print $1}' | head -1)
echo "rank0 scheduler pid=$R0"
# sanity: confirm it's python
cat /proc/$R0/comm 2>/dev/null; echo

# Start py-spy (nonblocking: does not pause the target). 250 samples/s for 65s.
py-spy record -p "$R0" -d 65 -r 250 --nonblocking -f raw -o "$OUT/pyspy_rank0.raw" &
PSP=$!
echo "py-spy pid=$PSP, warming up 3s..."
sleep 3

# Drive decode workload for 60s (overlaps py-spy window).
python3 "$OUT/prof19_decode_workload.py" --duration 60 --concurrency 8 \
    --ctx-tokens 12000 --max-tokens 256 > "$OUT/pyspy_workload.log" 2>&1 || true

echo "waiting for py-spy to finish..."
wait $PSP
echo "=== py-spy done ==="
ls -l "$OUT/pyspy_rank0.raw"
echo "--- raw head ---"
head -5 "$OUT/pyspy_rank0.raw"
echo "--- raw line count ---"
wc -l "$OUT/pyspy_rank0.raw"
echo "--- workload summary ---"
tail -12 "$OUT/pyspy_workload.log"
