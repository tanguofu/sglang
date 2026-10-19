#!/usr/bin/env python3
"""Port patch 06c (eagle_worker_v2 DSA backend enable on HIP) to post1.

Semantic change: enable the DSA attention backend + cuda_draft_extend_graph on HIP
(RDMA), not just CUDA/MUSA. Two gated sites:
  - `if _is_cuda or _is_musa:` (DSA lazy import) -> add `or _is_hip`
  - `(_is_cuda or _is_musa) and graph_supported_backend` -> add `or _is_hip`

_is_hip is already defined in the file. Idempotent.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/port-post1/base/srt_speculative_eagle_worker_v2.py")
src = path.read_text()
changed = []

# Site 1: if _is_cuda or _is_musa:  ->  if _is_cuda or _is_musa or _is_hip:
s1_old = "if _is_cuda or _is_musa:\n"
s1_new = "if _is_cuda or _is_musa or _is_hip:\n"
if s1_new in src:
    changed.append("site1: skipped")
else:
    assert src.count(s1_old) == 1, f"06c site1: expected 1, found {src.count(s1_old)}"
    src = src.replace(s1_old, s1_new, 1)
    changed.append("site1: applied")

# Site 2: (_is_cuda or _is_musa\n        ) and graph_supported_backend
s2_old = "            _is_cuda or _is_musa\n        ) and graph_supported_backend"
s2_new = "            _is_cuda or _is_musa or _is_hip\n        ) and graph_supported_backend"
if s2_new in src:
    changed.append("site2: skipped")
else:
    assert s2_old in src, "06c site2: anchor not found"
    src = src.replace(s2_old, s2_new, 1)
    changed.append("site2: applied")

path.write_text(src)
print(f"[ok] {path}: " + ", ".join(changed))
