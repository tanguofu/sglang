"""Config builder for GLM-5.2 DSpark draft model with native MLA.

Produces a draft config with architectures=["Glm5ForCausalLMDSpark"] and
model_type="glm_moe_dsa" that is directly loadable by sglang's
Glm5ForCausalLMDSpark (Route B).

The draft config inherits MLA fields (kv_lora_rank, q_lora_rank,
qk_nope_head_dim, qk_rope_head_dim, v_head_dim) and MoE fields
(n_routed_experts, num_experts_per_tok, moe_intermediate_size,
n_shared_experts, first_k_dense_replace) from the target GLM-5.2 config.

DSpark-specific fields use the dspark_* prefix to match sglang's
Glm5DSparkModel which reads them via getattr(config, "dspark_*").
"""

import copy
import json
import os

from deepspec.modeling.dspark.common import validate_target_layer_ids


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"


def build_draft_config(target_config, model_args):
    num_target_layers = int(target_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )
    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )

    draft_config = copy.deepcopy(target_config)

    # --- Fix qk_rope_head_dim ---
    # GlmMoeDsaConfig has attribute_map = {"head_dim": "qk_rope_head_dim"},
    # which causes AutoConfig to override qk_rope_head_dim (64) with head_dim
    # (192). We must restore the correct value from the raw JSON, otherwise
    # MLA weight shapes won't match sglang's expectations.
    model_path = getattr(target_config, "_name_or_path", None)
    if model_path:
        raw_config_path = os.path.join(model_path, "config.json")
        if os.path.exists(raw_config_path):
            with open(raw_config_path) as f:
                raw_config = json.load(f)
            if "qk_rope_head_dim" in raw_config:
                draft_config.qk_rope_head_dim = int(raw_config["qk_rope_head_dim"])
    # Recompute qk_head_dim with the corrected qk_rope_head_dim
    draft_config.qk_head_dim = (
        int(draft_config.qk_nope_head_dim) + int(draft_config.qk_rope_head_dim)
    )

    # --- Override n_routed_experts (optional) ---
    # The target model has 256 experts, which may cause OOM during training
    # due to large optimizer states. Allow model_args to override to a smaller
    # number (e.g., 64) for the draft model. The draft model only needs to
    # predict target hidden states, so fewer experts are sufficient.
    if hasattr(model_args, "n_routed_experts") and model_args.n_routed_experts is not None:
        draft_config.n_routed_experts = int(model_args.n_routed_experts)

    # Keep original GLM-5.2 dimensions: v_head_dim=256, qk_nope_head_dim=192.
    # The inference absorb mode (q_nope @ w_kc → q_absorbed, attn_output @ w_vc → v)
    # is mathematically equivalent to training with kv_b_proj. No override needed.

    # --- Architecture identifiers (for sglang load_weights dispatch) ---
    draft_config.architectures = ["Glm5ForCausalLMDSpark"]
    draft_config.model_type = "glm_moe_dsa"

    # --- DSpark-specific fields (dspark_* prefix, read by sglang) ---
    draft_config.dspark_block_size = int(model_args.block_size)
    draft_config.dspark_markov_rank = markov_rank
    draft_config.dspark_noise_token_id = int(model_args.mask_token_id)
    draft_config.dspark_target_layer_ids = target_layer_ids
    draft_config.dspark_num_layers = num_draft_layers

    # --- Training-specific fields (read by DeepSpec, not by sglang) ---
    # These use non-prefixed names for backward compat with the trainer.
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.layer_types = ["full_attention"] * num_draft_layers
    draft_config.mlp_layer_types = ["sparse"] * num_draft_layers
    draft_config.indexer_types = ["full"] * num_draft_layers
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)

    # --- Attention implementation ---
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION

    # --- Ensure MLP bias field exists (DeepseekV2MLP accesses config.mlp_bias) ---
    if not hasattr(draft_config, "mlp_bias") or draft_config.mlp_bias is None:
        draft_config.mlp_bias = False

    # --- Ensure intermediate_size exists for dense MLP layers ---
    # GLM-5.2 target config has intermediate_size=12288; if missing, set default.
    if not hasattr(draft_config, "intermediate_size") or draft_config.intermediate_size is None:
        draft_config.intermediate_size = 12288

    # --- Ensure attention_bias exists ---
    if not hasattr(draft_config, "attention_bias"):
        draft_config.attention_bias = False

    # --- Ensure attention_dropout exists ---
    if not hasattr(draft_config, "attention_dropout"):
        draft_config.attention_dropout = 0.0

    # --- GLM-5.2 specific: extract rope_theta from rope_parameters dict ---
    if not hasattr(draft_config, "rope_theta") or draft_config.rope_theta is None:
        rope_params = getattr(draft_config, "rope_parameters", None)
        if isinstance(rope_params, dict) and "rope_theta" in rope_params:
            draft_config.rope_theta = rope_params["rope_theta"]
        elif hasattr(target_config, "rope_theta"):
            draft_config.rope_theta = target_config.rope_theta
        else:
            draft_config.rope_theta = 8000000.0

    # GLM-5.2 does not use sliding window
    if not hasattr(draft_config, "sliding_window") or draft_config.sliding_window is None:
        draft_config.sliding_window = None

    return draft_config


__all__ = [
    "build_draft_config",
]
