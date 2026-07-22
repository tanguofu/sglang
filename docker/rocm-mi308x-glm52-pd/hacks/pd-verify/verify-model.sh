#!/bin/bash
# Compare safetensors file sizes between source (172) and target (159)
SRC_DIR=/host/data/model/glm52-fp8
DST_NODE=node-21.234.170.159
echo "=== Checking file sizes on 159 vs 172 ==="
BAD=0
for f in $(ls $SRC_DIR/*.safetensors 2>/dev/null); do
  fname=$(basename $f)
  src_size=$(stat -c %s "$f" 2>/dev/null)
  # Get size on 159 via debug pod
  dst_size=$(kubectl debug node/$DST_NODE --image=busybox --profile=sysadmin -i -- sh -c "stat -c %s /host/data/model/glm52-fp8/$fname" 2>/dev/null | tail -1)
  if [ "$src_size" != "$dst_size" ]; then
    echo "MISMATCH: $fname src=$src_size dst=$dst_size"
    BAD=$((BAD+1))
  fi
done
echo "=== Total mismatches: $BAD ==="
