#!/usr/bin/env python3
"""Port patch 5c (dsa_backend.py MLA q reshape on HIP) to post1.

Semantic change: in the absorbed multi-latent attention path, `q.view(...)` /
`q_rope.view(...)` / `q.contiguous().view(...)` can fail on HIP when the tensor is not
contiguous in the expected way. Use `.reshape(...)` which is contiguity-tolerant.

Applied at the MLA absorbed-attention call sites (q_nope / q_rope / q_all reshaping).
Idempotent. Only the MLA-absorb reshape sites are changed; other `.view()` calls in the
file are left untouched.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/port-post1/base/srt_layers_attention_dsa_backend.py")
src = path.read_text()
changed = []

# Site pattern: the MLA absorbed path reshapes q/q_rope/q_all. Convert these specific
# view() calls to reshape(). We target by their distinctive argument shapes.
replacements = [
    # q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
    ("q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)",
     "q_nope = q.reshape(-1, layer.tp_q_head_num, layer.v_head_dim)"),
    # q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
    ("q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)",
     "q_all = q.contiguous().reshape(-1, layer.tp_q_head_num, layer.head_dim)"),
]

for old, new in replacements:
    if new in src and old not in src:
        changed.append(f"{old[:40]}...: skipped")
        continue
    n = src.count(old)
    if n == 0:
        changed.append(f"{old[:40]}...: not found (already?)")
        continue
    src = src.replace(old, new)
    changed.append(f"{old[:40]}...: {n} replaced")

# q_rope = q_rope.view(\n  -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim\n)
qrope_old = (
    "            q_rope = q_rope.view(\n"
    "                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim\n"
    "            )"
)
qrope_new = qrope_old.replace("q_rope.view(", "q_rope.reshape(")
if "q_rope.reshape(" in src and "q_rope = q_rope.view(\n                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim" not in src:
    changed.append("q_rope: skipped")
else:
    n = src.count(qrope_old)
    if n > 0:
        src = src.replace(qrope_old, qrope_new)
        changed.append(f"q_rope: {n} replaced")
    else:
        changed.append("q_rope: not found")

path.write_text(src)
print(f"[ok] {path}: " + ", ".join(changed))
