"""
Target weight initialization for DSpark draft model.

This module provides functions to initialize the draft model's attention + MoE
layers from the target model's corresponding layers, and freeze them.

This is CRITICAL for DSpark: the inference kv_from_hidden uses the TARGET's
kv_a_proj_with_mqa to extract KV from the draft's predicted hidden states.
If the draft's attention weights are randomly initialized (not loaded from
the target), the training/inference KV projections mismatch, causing
accept_rate ≈ 0%.
"""

import gc
import json
import os

import torch
from safetensors import safe_open


def _dequantize_fp8_block(weight_fp8, scale_inv, block_size=128):
    """Dequantize FP8 block-scaled weight to BF16.

    Args:
        weight_fp8: (out_features, in_features) float8_e4m3fn
        scale_inv: (out_features // block_size, in_features // block_size) float32
        block_size: block size (128 for GLM-5.2 FP8)

    Returns:
        weight_bf16: (out_features, in_features) bfloat16
    """
    scale_expanded = scale_inv.repeat_interleave(block_size, dim=0).repeat_interleave(
        block_size, dim=1
    )
    weight_bf16 = (weight_fp8.to(torch.float32) * scale_expanded).to(torch.bfloat16)
    return weight_bf16


def initialize_layers_from_target(
    model,
    target_model_path,
    target_layer_ids,
    precision_dtype=torch.bfloat16,
):
    """Initialize the draft model's attention + MoE layers from the target model.

    After loading, all attention + MoE + layernorm weights are frozen. Only
    main_proj, main_norm, markov_head, confidence_head, and shared_head.norm
    remain trainable.
    """
    print(
        f"[initialize_layers_from_target] Loading from {target_model_path}, "
        f"target_layer_ids={target_layer_ids}",
        flush=True,
    )

    # Load the safetensors index
    index_path = os.path.join(target_model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Build the set of target keys we need to load
    target_keys = set()
    for target_id in target_layer_ids:
        prefix = f"model.layers.{target_id}."
        for key in weight_map:
            if key.startswith(prefix):
                target_keys.add(key)
    # Also load model.norm.weight
    if "model.norm.weight" in weight_map:
        target_keys.add("model.norm.weight")

    # Group by safetensors file
    files_needed = set()
    for key in target_keys:
        files_needed.add(weight_map[key])

    print(
        f"[initialize_layers_from_target] Need {len(target_keys)} keys from "
        f"{len(files_needed)} files",
        flush=True,
    )

    # Load all needed weights from the safetensors files
    loaded_weights = {}
    for fname in sorted(files_needed):
        fpath = os.path.join(target_model_path, fname)
        with safe_open(fpath, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if key in target_keys:
                    loaded_weights[key] = sf.get_tensor(key)

    print(
        f"[initialize_layers_from_target] Loaded {len(loaded_weights)} tensors",
        flush=True,
    )

    # Map target weights to draft model weights
    # Target: model.layers.{target_id}.self_attn.{name}
    # Draft:  mtp.{draft_id}.self_attn.{name}
    state_dict = {}
    for draft_id, target_id in enumerate(target_layer_ids):
        target_prefix = f"model.layers.{target_id}."
        draft_prefix = f"mtp.{draft_id}."

        for key in loaded_weights:
            if key.startswith(target_prefix):
                draft_key = key.replace(target_prefix, draft_prefix)
                state_dict[draft_key] = loaded_weights[key]

    # Also map model.norm.weight -> mtp.norm.weight
    if "model.norm.weight" in loaded_weights:
        state_dict["mtp.norm.weight"] = loaded_weights["model.norm.weight"]

    # Dequantize FP8 weights and load into the model
    model_state = model.state_dict()
    loaded_count = 0
    skipped_count = 0
    shape_mismatches = []

    for draft_key, tensor in state_dict.items():
        if draft_key not in model_state:
            skipped_count += 1
            continue

        model_tensor = model_state[draft_key]

        # Check if this is an FP8 weight (has weight_scale_inv)
        scale_key = draft_key + ".weight_scale_inv"
        if scale_key in state_dict:
            # FP8 dequantization
            scale_inv = state_dict[scale_key]
            weight_fp8 = tensor
            if weight_fp8.dtype == torch.float8_e4m3fn:
                dequantized = _dequantize_fp8_block(weight_fp8, scale_inv)
                if dequantized.shape == model_tensor.shape:
                    model_tensor.copy_(dequantized.to(model_tensor.dtype))
                    loaded_count += 1
                else:
                    shape_mismatches.append(
                        f"  {draft_key}: dequant={tuple(dequantized.shape)} vs model={tuple(model_tensor.shape)}"
                    )
            else:
                # Not FP8, load directly
                if tensor.shape == model_tensor.shape:
                    model_tensor.copy_(tensor.to(model_tensor.dtype))
                    loaded_count += 1
                else:
                    shape_mismatches.append(
                        f"  {draft_key}: src={tuple(tensor.shape)} vs model={tuple(model_tensor.shape)}"
                    )
        else:
            # Non-quantized weight (layernorm, gate, etc.)
            if tensor.shape == model_tensor.shape:
                model_tensor.copy_(tensor.to(model_tensor.dtype))
                loaded_count += 1
            else:
                shape_mismatches.append(
                    f"  {draft_key}: src={tuple(tensor.shape)} vs model={tuple(model_tensor.shape)}"
                )

    if shape_mismatches:
        print("[initialize_layers_from_target] SHAPE MISMATCHES:", flush=True)
        for msg in shape_mismatches[:20]:
            print(msg, flush=True)

    print(
        f"[initialize_layers_from_target] Loaded {loaded_count} weights, "
        f"skipped {skipped_count}",
        flush=True,
    )

    # Freeze all attention, MoE, and layernorm weights
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        should_freeze = (
            ".self_attn." in name
            or ".mlp." in name
            or ".input_layernorm." in name
            or ".post_attention_layernorm." in name
            or "mtp.norm.weight" == name
            or name.endswith(".norm.weight")
            and "shared_head" not in name
            and "main_norm" not in name
        )
        if should_freeze:
            param.requires_grad_(False)
            frozen_count += 1
        else:
            trainable_count += 1

    print(
        f"[initialize_layers_from_target] Frozen {frozen_count} params, "
        f"{trainable_count} trainable params remain",
        flush=True,
    )

    # Clean up
    del loaded_weights, state_dict
    gc.collect()

    return model
