---
name: dspark-accept-rate-0-v2
description: "DSpark accept_len=1.00 ROOT CAUSE: block_full_attn (bidirectional block attention) not implemented in DSA/aiter MLA backends on AMD. Only in deepseek_v4_backend (CUDA)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 18aa3e35-6df6-4d4a-85c8-8b5f7d26df92
---

# DSpark accept_rate=0% v2 — Root Cause FOUND (2026-07-08)

## Status: accept_len=1.00, accept_rate=0.00 — ROOT CAUSE CONFIRMED

## PRIMARY ROOT CAUSE: block_full_attn not implemented on AMD

**Training** uses `create_dspark_attention_mask` (bidirectional within blocks):
- Draft tokens in the same block attend to EACH OTHER bidirectionally
- `mask_draft = is_draft & (q_block_id == kv_block_id)` — all-to-all within block

**Inference** sets `block_full_attn=block_size` in `DSparkVerifyInput`, intended to
replicate bidirectional block attention. BUT:
- `block_full_attn` is ONLY implemented in `deepseek_v4_backend.py:1695-1710` (CUDA)
- It manipulates SWA page indices so all tokens in a block share the same KV span
- **DSA backend** (`dsa_backend.py`): reads `spec_info` but ignores `block_full_attn`
- **aiter MLA backend** (`aiter_mla_backend.py`): doesn't read `spec_info` at all
- GLM-5.2 DSpark uses `Glm5DSparkDecoderLayer(DeepseekV2DecoderLayer)` → standard MLA
  path, NOT V4 path → `block_full_attn` silently ignored → **causal attention**

**Why DeepSeek-V4 works (accept_len=3.72)**: `DeepseekV4DSparkModel` uses
`DeepseekV4DecoderLayer` → V4 attention backend → reads `block_full_attn` →
modifies SWA page indices → bidirectional block attention.

**Impact**: Training with bidirectional attention, inference with causal attention =
completely different information flow → draft hidden states meaningless → all rejected.

## SECONDARY: Position offset (1-position shift)
- Training: draft positions = `anchor_pos + [0..block_size-1]`
- Inference: draft positions = `prefix_len + [0..block_size-1]`
- Off by 1 (anchor_pos = prefix_len - 1 in training semantics)

## MINOR: Context attention scope
- Training: draft attends to context tokens BEFORE anchor only
- Inference: draft attends to ALL context tokens in KV cache

## EQUIVALENT (NOT the bug)
- KV projection (cat vs separate): linear layer, math identical ✅
- LayerNorm, noise embedding, main_proj+main_norm: all consistent ✅
- Double RMSNorm: training and inference both do double norm, consistent ✅
- lm_head/embed_tokens tied to target (freeze=True): consistent ✅

## Update: block_full_attn is NOT the root cause (2026-07-08)

After deeper investigation:
- MLA decode kernel already gives all draft tokens the SAME KV range
  (kv_indptr=[0, seq_len+block_size] per request). This IS bidirectional.
- custom_mask approach doesn't work: MLA mode in aiter backend ignores
  custom_mask (only non-MLA path reads it).
- Position offset fix (prefix_lens-1) applied: minimal effect (1.00→1.02).
- Double RMSNorm restored to match training: no change.

## ROOT CAUSE CONFIRMED: Training/inference forward path mismatch

### The fundamental problem
- **Official DeepSpec** only has qwen3 and gemma4 model variants (no deepseek_v4, no glm5)
- **sglang's `deepseek_v4_dspark.py`** is written to match DeepSeek's INTERNAL training code
  (not public), using `kv_from_hidden` + `DeepseekV4DecoderLayer`. This works (accept_len=3.72)
  because DeepSeek's training code also uses `kv_from_hidden`.
- **Our GLM-5 training code** (DeepSpec/deepspec/modeling/dspark/glm5/) was written referencing
  **qwen3** (which uses `cat([target, draft])` for KV in attention)
- **Our GLM-5 inference code** (sglang/glm5_dspark.py) was written referencing **deepseek_v4_dspark.py**
  (which uses `kv_from_hidden`)

**Training uses cat attention, inference uses kv_from_hidden — they reference DIFFERENT implementations!**

### Evidence
1. Official DeepSpec evaluator (`eval/dspark/draft_ops.py`) calls `model._forward_backbone()`
   — the SAME function used in training. No kv_from_hidden in eval.
2. Official qwen3 training attention: `k = cat([k_proj(target), k_proj(draft)])` (separate proj + cat)
3. Our GLM-5 training attention: `kv_a_proj_with_mqa(cat([target, draft]))` (cat then proj)
4. sglang inference: `kv_from_hidden(target)` + draft KV via normal forward (separate, no cat)
5. DeepSeek-V4 inference works because V4 training (internal) also uses kv_from_hidden

### Fix options
1. **Change training to use kv_from_hidden** (like V4): modify GLM-5 training attention to
   separate target KV materialization from draft forward. Requires retraining.
2. **Change inference to use cat attention** (like qwen3/training): modify glm5_dspark.py
   forward_backbone to concatenate target and draft hidden states for KV. Hard (needs custom
   attention in sglang).
3. **Use DeepSpec evaluator instead of sglang server**: run eval with the training forward path.
   Works for testing but not production deployment.

### Recommended: Option 1 (change training, retrain)
- Modify `Glm5DSparkMLAAttention` to NOT use `cat([target, draft])`
- Instead, materialize target KV via `kv_a_proj_with_mqa(target)` + `kv_a_layernorm`
- Draft KV via normal `kv_a_proj_with_mqa(draft)` + `kv_a_layernorm`
- This matches sglang's `kv_from_hidden` + normal forward
- Requires retraining but training code change is small

Related: [[dspark-accept-rate-0-debug]], [[glm52-dspark-route-b]],
[[dspark-cache-corruption-rootcause]].

## Environment (verified on node-2 and 355-worker)
- Checkpoint: step_180, loss=0.154, all config fixes applied
- DSpark server: runs, generates correct text, accept_len=1.00
- EAGLE MTP baseline: accept_len=2.82
- Official DeepSeek-V4 DSpark: accept_len=3.72 (with block_full_attn working)

Related: [[dspark-accept-rate-0-debug]], [[glm52-dspark-route-b]],
[[dspark-cache-corruption-rootcause]].
