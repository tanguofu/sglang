#!/usr/bin/env python3
"""Supplemental patch v4: fix view->reshape in dsa_backend.py + keep v3 cos_sin_cache fixes.

Key fixes:
1. DUAL_STREAM_TOKEN_THRESHOLD = 1024 (from v2 S1)
2. cos_sin_cache init + usage fixes (from v3 S2/S3/S4)
3. metadata None guard (from v2 S5)
4. NEW: view->reshape in dsa_backend.py forward_extend (3 locations)
"""
import sys

errors = []

# ============================================================
# FILE 1: dsa_indexer.py (v3 fixes)
# ============================================================
FILE1 = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py"

with open(FILE1, "r") as f:
    c1 = f.read()

# S1: DUAL_STREAM_TOKEN_THRESHOLD
OLD_S1 = "DUAL_STREAM_TOKEN_THRESHOLD = 1024 if _is_cuda else 0"
NEW_S1 = "DUAL_STREAM_TOKEN_THRESHOLD = 1024"
if NEW_S1 in c1 and "if _is_cuda else 0" not in c1.split("DUAL_STREAM_TOKEN_THRESHOLD")[1][:50]:
    print("[SKIP] S1: DUAL_STREAM_TOKEN_THRESHOLD already fixed")
elif OLD_S1 in c1:
    c1 = c1.replace(OLD_S1, NEW_S1, 1)
    print("[OK] S1: DUAL_STREAM_TOKEN_THRESHOLD = 1024")
else:
    print("[WARN] S1: DUAL_STREAM_TOKEN_THRESHOLD pattern not found")

# S2: cos_sin_cache init - always store
OLD_S2 = """        if _use_aiter:
            cos_cache = getattr(self.rotary_emb, "cos_cache", None)
            sin_cache = getattr(self.rotary_emb, "sin_cache", None)
            if cos_cache is not None and sin_cache is not None:
                self._cos_sin_cache_val = torch.cat([cos_cache, sin_cache], dim=-1)
                if self._cos_sin_cache_val.dim() == 4:
                    self._cos_sin_cache_val = self._cos_sin_cache_val.reshape(
                        self._cos_sin_cache_val.shape[0], -1
                    )
                if self._cos_sin_cache_val.dtype != torch.float32:
                    self._cos_sin_cache_val = self._cos_sin_cache_val.to(torch.float32)
            else:
                self._cos_sin_cache_val = self.rotary_emb.cos_sin_cache"""

NEW_S2 = """        if hasattr(self.rotary_emb, 'cos_sin_cache'):
            self._cos_sin_cache_val = self.rotary_emb.cos_sin_cache
        else:
            self._cos_sin_cache_val = torch.cat([
                self.rotary_emb.cos_cache, self.rotary_emb.sin_cache
            ], dim=-1).reshape(self.rotary_emb.cos_cache.shape[0], -1).to(torch.float32)"""

if "if hasattr(self.rotary_emb, 'cos_sin_cache'):" in c1 and "_cos_sin_cache_val = self.rotary_emb.cos_sin_cache" in c1:
    print("[SKIP] S2: cos_sin_cache init already fixed")
elif OLD_S2 in c1:
    c1 = c1.replace(OLD_S2, NEW_S2, 1)
    print("[OK] S2: cos_sin_cache init now unconditional")
else:
    errors.append("S2: cos_sin_cache init pattern not found")

# S3: cos_sin_cache usage site 1
OLD_S3 = "                cos_sin = self.rotary_emb.cos_sin_cache[positions]"
NEW_S3 = "                cos_sin = self._indexer_cos_sin_cache[positions]"
if NEW_S3 in c1:
    print("[SKIP] S3: cos_sin_cache usage site 1 already fixed")
elif OLD_S3 in c1:
    c1 = c1.replace(OLD_S3, NEW_S3, 1)
    print("[OK] S3: cos_sin_cache usage site 1 -> _indexer_cos_sin_cache")
else:
    errors.append("S3: cos_sin_cache usage site 1 pattern not found")

# S4: cos_sin_cache usage site 2
OLD_S4 = "                    self.rotary_emb.cos_sin_cache.index_select(0, positions)"
NEW_S4 = "                    self._indexer_cos_sin_cache.index_select(0, positions)"
if NEW_S4 in c1:
    print("[SKIP] S4: cos_sin_cache usage site 2 already fixed")
elif OLD_S4 in c1:
    c1 = c1.replace(OLD_S4, NEW_S4, 1)
    print("[OK] S4: cos_sin_cache usage site 2 -> _indexer_cos_sin_cache")
else:
    errors.append("S4: cos_sin_cache usage site 2 pattern not found")

# S5: metadata None guard
if "FIX(breakable-target-verify)" in c1 or ("if metadata is None:" in c1 and "get_indexer_metadata" in c1):
    print("[SKIP] S5: metadata None guard already present")
else:
    TARGET = "                target_verify("
    if TARGET in c1:
        GUARD = """                # FIX(breakable-target-verify): metadata None guard
                if metadata is None:
                    metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)
                    if metadata is None:
                        return None
"""
        c1 = c1.replace(TARGET, GUARD + TARGET, 1)
        print("[OK] S5: metadata None guard added")
    else:
        print("[WARN] S5: target_verify call not found")

with open(FILE1, "w") as f:
    f.write(c1)

# ============================================================
# FILE 2: dsa_backend.py - view->reshape fixes
# ============================================================
FILE2 = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"

with open(FILE2, "r") as f:
    c2 = f.read()

# S6: view->reshape in forward_extend (3 locations)
fixes = [
    ("q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)",
     "q_nope = q.reshape(-1, layer.tp_q_head_num, layer.v_head_dim)"),
    ("q_rope = q_rope.view(",
     "q_rope = q_rope.reshape("),
    ("q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)",
     "q_all = q.contiguous().reshape(-1, layer.tp_q_head_num, layer.head_dim)"),
]

for i, (old, new) in enumerate(fixes):
    if new in c2:
        print(f"[SKIP] S6.{i+1}: view->reshape already applied")
    elif old in c2:
        c2 = c2.replace(old, new, 1)
        print(f"[OK] S6.{i+1}: view->reshape in dsa_backend.py")
    else:
        print(f"[WARN] S6.{i+1}: pattern not found: {old[:50]}")

with open(FILE2, "w") as f:
    f.write(c2)

# ============================================================
# Summary
# ============================================================
if errors:
    print(f"\n[ERROR] {len(errors)} patch(es) failed:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\n[DONE] v4 supplement: all patches applied successfully")
