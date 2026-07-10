#!/usr/bin/env python3
"""E2E test with REAL cache data: verify draft forward produces non-degenerate predictions.

Uses actual target_hidden_states from the clean cache (not synthetic random data).
"""
import torch, sys, os, json, struct
import numpy as np

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("SGLANG_USE_AITER", "1")

CKPT_DIR = "/data/checkpoints/deepspec/dspark_glm5_2_v9_256exp_clean/step_100"
MODEL_PATH = "/data/models/GLM-5.2-FP8"
CACHE_DIR = "/data/dspark_target_cache_v9_coding_clean_merged"
DEEPSPEC_PATH = "/data/DeepSpec"

sys.path.insert(0, DEEPSPEC_PATH)

print("=" * 60)
print("DSpark E2E Test with REAL cache data")
print("=" * 60)

# Step 1: Load a real sample from cache
print("\n[1] Loading real sample from cache...")
manifest = json.load(open(f"{CACHE_DIR}/manifest.json"))
hidden_size = manifest["hidden_size"]
num_layers = len(manifest["target_layer_ids"])
idx_struct = struct.Struct("<QIIQQQQQ")
idx_bytes = open(f"{CACHE_DIR}/samples.idx", "rb").read()
n = len(idx_bytes) // idx_struct.size
print(f"  Cache: {n} samples, hidden_size={hidden_size}, num_layers={num_layers}")

# Read sample 0
entry = idx_struct.unpack(idx_bytes[:idx_struct.size])
sample_id, shard_id, seq_len, ii_off, am_off, lm_off, th_off, tlh_off = entry
shard_file = open(f"{CACHE_DIR}/{manifest['shards'][shard_id]['file_name']}", "rb")

# Read target_hidden_states
th_nbytes = seq_len * num_layers * hidden_size * 2
shard_file.seek(th_off)
raw = shard_file.read(th_nbytes)
arr = np.frombuffer(raw, dtype=np.uint16).copy()
target_hidden = torch.from_numpy(arr).view(torch.bfloat16).float().reshape(seq_len, num_layers, hidden_size)
# Flatten to [seq_len, num_layers * hidden_size]
target_hidden_flat = target_hidden.reshape(seq_len, -1).to(torch.bfloat16)
print(f"  Sample 0: seq_len={seq_len}, target_hidden shape={list(target_hidden_flat.shape)}")
print(f"  target_hidden stats: mean={float(target_hidden_flat.mean()):.4g} std={float(target_hidden_flat.std()):.4g}")

# Read input_ids
shard_file.seek(ii_off)
ii_raw = shard_file.read(seq_len * 4)
input_ids = torch.from_numpy(np.frombuffer(ii_raw, dtype=np.int32).copy()).long()
print(f"  input_ids (first 10): {input_ids[:10].tolist()}")

# Read loss_mask
shard_file.seek(lm_off)
lm_raw = shard_file.read(seq_len)
loss_mask = torch.from_numpy(np.frombuffer(lm_raw, dtype=np.uint8).copy()).long()
print(f"  loss_mask sum: {loss_mask.sum().item()}")

# Step 2: Build config and load model
print("\n[2] Loading draft model...")
from transformers import AutoConfig
hf_config = AutoConfig.from_pretrained(CKPT_DIR, trust_remote_code=True)
hf_config.target_layer_ids = [15, 31, 47, 63, 76]
hf_config.mask_token_id = 154821
hf_config.num_anchors = 256
hf_config.enable_confidence_head = False
hf_config.markov_rank = 256
hf_config.markov_head_type = "vanilla"
hf_config.block_size = 7
hf_config.confidence_head_with_markov = False
hf_config.loss_decay_gamma = 1.0
hf_config.ce_loss_alpha = 1.0
hf_config.l1_loss_alpha = 0.0
hf_config._attn_implementation = "flex_attention"

from deepspec.modeling.dspark.glm5.modeling import Glm5DSparkMtp
model = Glm5DSparkMtp(hf_config)

# Load weights
from safetensors import safe_open
weights = {}
with safe_open(f"{CKPT_DIR}/model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        weights[key] = f.get_tensor(key)
model_state = {key.replace("mtp.", ""): val for key, val in weights.items()}

# Load tied weights from target
import glob
for sf in sorted(glob.glob(f"{MODEL_PATH}/*.safetensors")):
    try:
        with safe_open(sf, framework="pt", device="cpu") as f:
            if "lm_head.weight" in f.keys():
                model_state["lm_head.weight"] = f.get_tensor("lm_head.weight")
            if "model.embed_tokens.weight" in f.keys():
                model_state["embed_tokens.weight"] = f.get_tensor("model.embed_tokens.weight")
                break
    except:
        continue

missing, unexpected = model.load_state_dict(model_state, strict=False)
print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
model = model.to(dtype=torch.bfloat16, device="cuda")
model.eval()

# Step 3: Run forward with real data
print("\n[3] Running forward with real cache data...")
with torch.no_grad():
    from deepspec.modeling.dspark.common import (
        create_noise_embed, create_position_ids,
        create_dspark_attention_mask, sample_anchor_positions,
    )

    batch_size = 1
    block_size = hf_config.block_size

    # Use real input_ids and loss_mask
    input_ids_batch = input_ids.unsqueeze(0).to("cuda")  # [1, seq_len]
    loss_mask_batch = loss_mask.unsqueeze(0).to("cuda")  # [1, seq_len]
    target_hidden_batch = target_hidden_flat.unsqueeze(0).to("cuda")  # [1, seq_len, hidden*layers]

    # Sample anchor positions
    anchor_positions, block_keep_mask = sample_anchor_positions(
        seq_len=seq_len, loss_mask=loss_mask_batch,
        num_anchors=model.num_anchors, device="cuda",
    )
    num_blocks = anchor_positions.size(1)
    print(f"  anchors: {num_blocks} blocks")

    # Create noise embedding
    noise_embedding = create_noise_embed(
        model.embed_tokens, input_ids_batch, anchor_positions,
        block_keep_mask, mask_token_id=hf_config.mask_token_id,
        block_size=block_size,
    )
    print(f"  noise_embedding: {list(noise_embedding.shape)}")

    # Position IDs
    ctx_pos = torch.arange(seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
    draft_pos = create_position_ids(anchor_positions, block_size)
    full_pos = torch.cat([ctx_pos, draft_pos], dim=1)

    # Attention mask
    attn_mask = create_dspark_attention_mask(
        anchor_positions=anchor_positions, block_keep_mask=block_keep_mask,
        seq_len=seq_len, block_size=block_size, device="cuda",
    )

    # Forward
    output = model._forward_backbone(
        position_ids=full_pos,
        noise_embedding=noise_embedding,
        target_hidden_states=target_hidden_batch,
        attention_mask=attn_mask,
    )
    print(f"  output: {list(output.shape)} mean={float(output.float().mean()):.4g} std={float(output.float().std()):.4g}")

    # Compute logits
    logits = model.compute_logits(output)
    print(f"  logits: {list(logits.shape)}")

    # Get predictions for the draft block positions
    # output shape: [1, seq_len + num_blocks*block_size, hidden]
    # Draft predictions are at positions seq_len onwards
    draft_start = seq_len
    draft_logits = logits[0, draft_start:]  # [num_blocks*block_size, vocab]
    draft_preds = draft_logits.argmax(dim=-1)
    print(f"  draft predictions (first 21): {draft_preds[:21].tolist()}")

    unique = len(set(draft_preds[:21].tolist()))
    print(f"  unique predictions (first 21): {unique}/21")

    # Decode
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    decoded = [tok.decode([t]) for t in draft_preds[:21].tolist()]
    print(f"  decoded (first 21): {decoded}")

    # Also check: what does the target model predict for these positions?
    # Compare draft predictions with the actual next tokens in input_ids
    print(f"\n  Actual next tokens (from input_ids): {input_ids[1:8].tolist()}")
    actual_decoded = [tok.decode([t]) for t in input_ids[1:8].tolist()]
    print(f"  Actual next decoded: {actual_decoded}")

# Summary
print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)
if unique > 3:
    print("✅ PASS: Predictions are diverse (non-degenerate)")
else:
    print("⚠️  WARN: Predictions are mostly the same (degenerate)")
print("=" * 60)
