"""GLM-5.2 DSpark draft model with native MLA (Multi-head Latent Attention).

This model produces checkpoints with architectures=["Glm5ForCausalLMDSpark"]
that are directly loadable by sglang's Glm5ForCausalLMDSpark (Route B).

Key design decisions:
- Uses DeepseekV2 MLA attention (q_a_proj/q_b_proj, kv_a_proj_with_mqa/kv_b_proj)
  instead of standard transformer attention. This matches GLM-5.2's native MLA.
- The DSpark attention pattern: q is projected from draft (noise) tokens, while
  k/v are projected from concatenated [context, draft] tokens.
- MoE layers use individual expert modules (nn.ModuleList) so checkpoint weights
  have the mlp.experts.{e}.gate_proj/up_proj/down_proj format that sglang's
  deepseek_weight_loader expects.
- All DSpark submodules are under self.mtp so save_pretrained produces mtp.*
  prefixed keys, matching sglang's load_weights remapping.
- Custom rotary embedding uses qk_rope_head_dim (not head_dim) for the rope
  dimension, matching GLM-5.2's MLA rope application.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    ALL_ATTENTION_FUNCTIONS,
    DeepseekV2MLP,
    DeepseekV2PreTrainedModel,
    DeepseekV2RMSNorm,
    GradientCheckpointingLayer,
    eager_attention_forward,
)

from deepspec.modeling.dspark.common import (
    DSparkForwardOutput,
    build_eval_mask,
    create_dspark_attention_mask,
    create_noise_embed,
    create_position_ids,
    log_sampler_stats,
    sample_anchor_positions,
)
from deepspec.modeling.dspark.markov_head import build_markov_head
from deepspec.utils.sampling import sample_tokens


# ---------------------------------------------------------------------------
# Confidence head (matches sglang's DSparkConfidenceHead exactly)
# ---------------------------------------------------------------------------


class Glm5DSparkConfidenceHead(nn.Module):
    """Confidence head matching sglang's DSparkConfidenceHead.

    Uses bias=False and float32 dtype, with input_dim = hidden_size +
    markov_rank (always includes markov embeddings). This ensures the
    checkpoint's confidence_head.proj.weight has the exact shape sglang
    expects at load time.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        features = torch.cat([hidden, markov_embed], dim=-1)
        # Compute in float32 for numerical stability (matches sglang's behavior).
        # The proj weight may be bf16 after model.to(bf16), so cast explicitly.
        return F.linear(
            features.float(), self.proj.weight.float()
        ).squeeze(-1)


# ---------------------------------------------------------------------------
# Rotary embedding
# ---------------------------------------------------------------------------


class Glm5DSparkRotaryEmbedding(nn.Module):
    """Rotary embedding for MLA that uses qk_rope_head_dim (not head_dim).

    The standard DeepseekV2RotaryEmbedding computes inv_freq with
    config.head_dim, but GLM-5.2's MLA applies rope only to the
    qk_rope_head_dim portion of q/k. We need inv_freq sized for
    qk_rope_head_dim so apply_rotary_emb's view_as_complex reshape matches.
    """

    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.config = config
        dim = int(config.qk_rope_head_dim)
        rope_params = getattr(config, "rope_parameters", None)
        if isinstance(rope_params, dict) and "rope_theta" in rope_params:
            base = float(rope_params["rope_theta"])
        else:
            base = float(getattr(config, "rope_theta", 8000000.0))
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64)
                .to(device=device, dtype=torch.float)
                / dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = (
            x.device.type if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.amp.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(
                1, 2
            )
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            freqs_cis = freqs_cis * self.attention_scaling
        return freqs_cis


def apply_rotary_emb_dspark(
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding to q_pe (draft) and k_pe (context+draft).

    Unlike the standard apply_rotary_emb, this handles the DSpark case where
    q_pe and k_pe have different sequence lengths (q is draft-only, k is
    context+draft). freqs_cis covers all positions (context+draft); we slice
    the last q_len positions for q_pe.

    Args:
        q_pe: [bsz, num_heads, q_len, qk_rope_head_dim]
        k_pe: [bsz, 1, kv_len, qk_rope_head_dim] (will be expanded to num_heads)
        freqs_cis: [bsz, kv_len, qk_rope_head_dim // 2] (complex)

    Returns:
        q_pe_out: [bsz, num_heads, q_len, qk_rope_head_dim]
        k_pe_out: [bsz, 1, kv_len, qk_rope_head_dim]
    """
    q_len = q_pe.shape[-2]
    kv_len = k_pe.shape[-2]

    # q_pe: use last q_len positions of freqs_cis
    q_freqs = freqs_cis[:, -q_len:, :].unsqueeze(1)  # [bsz, 1, q_len, dim//2]
    # k_pe: use all positions
    k_freqs = freqs_cis.unsqueeze(1)  # [bsz, 1, kv_len, dim//2]

    q_ = torch.view_as_complex(
        q_pe.float().reshape(*q_pe.shape[:-1], -1, 2)
    )  # [bsz, num_heads, q_len, dim//2]
    k_ = torch.view_as_complex(
        k_pe.float().reshape(*k_pe.shape[:-1], -1, 2)
    )  # [bsz, 1, kv_len, dim//2]

    q_out = torch.view_as_real(q_ * q_freqs).flatten(3).type_as(q_pe)
    k_out = torch.view_as_real(k_ * k_freqs).flatten(3).type_as(k_pe)
    return q_out, k_out


# ---------------------------------------------------------------------------
# MLA Attention with DSpark pattern
# ---------------------------------------------------------------------------


class Glm5DSparkMLAAttention(nn.Module):
    """MLA attention for DSpark draft model.

    Follows DeepseekV2 MLA (q_a_proj -> q_a_layernorm -> q_b_proj,
    kv_a_proj_with_mqa -> kv_a_layernorm -> kv_b_proj) but with the DSpark
    bidirectional pattern: q is projected from draft (noise) tokens, while
    k/v are projected from concatenated [context, draft] tokens.

    Weight names match DeepseekV2Attention exactly so sglang can load them:
    - q_a_proj, q_a_layernorm, q_b_proj
    - kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj
    - o_proj
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            self.num_heads // self.num_key_value_heads
        )
        self.scaling = self.qk_head_dim ** (-0.5)
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        # Q projection (low-rank)
        self.q_a_proj = nn.Linear(
            self.hidden_size, config.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
        )

        # KV projection (low-rank with MQA)
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV2RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            hidden_states: draft (noise) tokens [bsz, draft_len, hidden_size]
            target_hidden_states: context tokens [bsz, ctx_len, hidden_size]
            attention_mask: DSpark block mask
            position_embeddings: freqs_cis for all positions [bsz, ctx_len+draft_len, ...]

        Returns:
            attn_output: [bsz, draft_len, hidden_size]
            None
        """
        bsz, draft_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]
        kv_len = ctx_len + draft_len

        # --- Q projection (from draft only) ---
        q = self.q_b_proj(
            self.q_a_layernorm(self.q_a_proj(hidden_states))
        )  # [bsz, draft_len, num_heads * qk_head_dim]
        q = q.view(
            bsz, draft_len, self.num_heads, self.qk_head_dim
        ).transpose(1, 2)  # [bsz, num_heads, draft_len, qk_head_dim]
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # --- KV projection (separate target + draft, matching inference kv_from_hidden) ---
        # Target KV: project separately (matches sglang kv_from_hidden path)
        compressed_kv_ctx = self.kv_a_proj_with_mqa(
            target_hidden_states
        )  # [bsz, ctx_len, kv_lora_rank + qk_rope_head_dim]
        # Draft KV: project separately (matches sglang normal forward path)
        compressed_kv_draft = self.kv_a_proj_with_mqa(
            hidden_states
        )  # [bsz, draft_len, kv_lora_rank + qk_rope_head_dim]
        # Concatenate after projection (mathematically equivalent to projecting cat,
        # but aligns code path with inference kv_from_hidden)
        compressed_kv = torch.cat(
            [compressed_kv_ctx, compressed_kv_draft], dim=1
        )  # [bsz, kv_len, kv_lora_rank + qk_rope_head_dim]
        k_nope_compressed, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        k_nope_and_v = self.kv_b_proj(
            self.kv_a_layernorm(k_nope_compressed)
        )  # [bsz, kv_len, num_heads * (qk_nope_head_dim + v_head_dim)]
        k_nope_and_v = k_nope_and_v.view(
            bsz, kv_len, self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ).transpose(1, 2)  # [bsz, num_heads, kv_len, ...]
        k_nope, value_states = torch.split(
            k_nope_and_v,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )

        # k_pe: [bsz, kv_len, qk_rope_head_dim] -> [bsz, 1, kv_len, qk_rope_head_dim]
        k_pe = k_pe.view(bsz, 1, kv_len, self.qk_rope_head_dim)

        # --- Apply rotary embedding ---
        q_pe, k_pe = apply_rotary_emb_dspark(
            q_pe, k_pe, position_embeddings
        )

        # Expand k_pe to num_heads
        k_pe = k_pe.expand(bsz, self.num_heads, kv_len, self.qk_rope_head_dim)

        # --- Concatenate nope + rope ---
        query_states = torch.cat(
            (q_nope, q_pe), dim=-1
        )  # [bsz, num_heads, draft_len, qk_head_dim]
        key_states = torch.cat(
            (k_nope, k_pe), dim=-1
        )  # [bsz, num_heads, kv_len, qk_head_dim]

        # --- Attention ---
        # Use the same attention dispatch as standard DeepseekV2 (flex_attention
        # for training with DSpark block mask, sdpa/eager as fallbacks).
        attn_implementation = getattr(
            self.config, "_attn_implementation", "flex_attention"
        )
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn_implementation, eager_attention_forward
        )
        attn_is_causal = bool(kwargs.get("is_causal", False))
        self.is_causal = attn_is_causal
        kwargs["is_causal"] = attn_is_causal
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # [bsz, num_heads, draft_len, v_head_dim] -> [bsz, draft_len, num_heads * v_head_dim]
        attn_output = (
            attn_output.transpose(1, 2)
            .reshape(bsz, draft_len, self.num_heads * self.v_head_dim)
            .contiguous()
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, None


# ---------------------------------------------------------------------------
# MoE with individual experts (for sglang-compatible weight naming)
# ---------------------------------------------------------------------------


class Glm5DSparkGate(nn.Module):
    """Router gate with e_score_correction_bias for noaux_tc routing.

    Weight names match sglang's DeepseekV2 gate:
    - gate.weight: [n_routed_experts, hidden_size]
    - gate.e_score_correction_bias: [n_routed_experts]
    """

    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.n_routed_experts, dtype=torch.float32)
        )
        # Initialize gate weight with small normal distribution (matching
        # DeepseekV2's nn.Linear default init) so routing works from step 1.
        nn.init.normal_(self.weight, mean=0.0, std=config.initializer_range)

    def forward(self, hidden_states):
        return F.linear(
            hidden_states.type(torch.float32),
            self.weight.type(torch.float32),
        )


class Glm5DSparkMoe(nn.Module):
    """MoE layer with individual expert modules.

    Uses nn.ModuleList of DeepseekV2MLP so checkpoint weights have the
    mlp.experts.{e}.gate_proj/up_proj/down_proj format that sglang's
    deepseek_weight_loader expects.

    Implements noaux_tc routing (softmax + e_score_correction_bias + topk).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = int(config.n_routed_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.0)
        )
        self.gate = Glm5DSparkGate(config)
        self.experts = nn.ModuleList(
            [
                DeepseekV2MLP(
                    config,
                    intermediate_size=int(config.moe_intermediate_size),
                )
                for _ in range(self.num_experts)
            ]
        )
        n_shared = int(
            config.n_shared_experts
            if config.n_shared_experts is not None
            else 0
        )
        if n_shared > 0:
            self.shared_experts = DeepseekV2MLP(
                config,
                intermediate_size=int(config.moe_intermediate_size) * n_shared,
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        bsz, seq_len, hidden_dim = orig_shape

        # Router logits
        router_logits = self.gate(hidden_states)  # [bsz, seq_len, num_experts]
        scores = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        scores = scores + self.gate.e_score_correction_bias
        topk_weight, topk_idx = torch.topk(
            scores, k=self.top_k, dim=-1, sorted=False
        )
        topk_weight = topk_weight * self.routed_scaling_factor
        topk_weight = topk_weight / (
            topk_weight.sum(dim=-1, keepdim=True) + 1e-6
        )

        # Flatten for expert routing
        flat_hidden = hidden_states.reshape(-1, hidden_dim)  # [N, H]
        flat_topk_idx = topk_idx.reshape(-1, self.top_k)  # [N, top_k]
        flat_topk_weight = topk_weight.reshape(-1, self.top_k)  # [N, top_k]
        num_tokens = flat_hidden.shape[0]

        output = torch.zeros_like(flat_hidden)

        for expert_idx in range(self.num_experts):
            # Find tokens that route to this expert
            token_mask = (flat_topk_idx == expert_idx).any(dim=-1)  # [N]
            if not token_mask.any():
                continue
            expert_input = flat_hidden[token_mask]  # [M, H]
            expert_output = self.experts[expert_idx](expert_input)  # [M, H]
            # Get the weight for this expert for each token
            expert_weight_mask = flat_topk_idx == expert_idx  # [N, top_k]
            expert_weights = (
                flat_topk_weight * expert_weight_mask.float()
            ).sum(dim=-1)  # [N]
            output[token_mask] += (
                expert_output * expert_weights[token_mask].unsqueeze(-1)
            )

        output = output.view(*orig_shape)
        if self.shared_experts is not None:
            output = output + self.shared_experts(residuals)
        return output


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class Glm5DSparkDecoderLayer(GradientCheckpointingLayer):
    """Decoder layer with MLA attention and DeepseekV2 MLP/MoE.

    Weight names match DeepseekV2DecoderLayer:
    - self_attn.{q_a_proj, q_a_layernorm, q_b_proj, kv_a_proj_with_mqa,
      kv_a_layernorm, kv_b_proj, o_proj}
    - mlp.{gate_proj, up_proj, down_proj} (dense) or
      mlp.{gate, experts.{e}.*, shared_experts.*} (MoE)
    - input_layernorm, post_attention_layernorm
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Glm5DSparkMLAAttention(config=config, layer_idx=layer_idx)
        # Dense MLP for first_k_dense_replace layers, MoE for the rest
        if layer_idx >= int(getattr(config, "first_k_dense_replace", 3)):
            self.mlp = Glm5DSparkMoe(config)
        else:
            self.mlp = DeepseekV2MLP(config)
        self.input_layernorm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden_states: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


# ---------------------------------------------------------------------------
# DSpark MTP module (all submodules under self.mtp for mtp.* weight prefix)
# ---------------------------------------------------------------------------


class Glm5DSparkMtp(nn.Module):
    """Container for all DSpark components.

    All submodules are under this module so save_pretrained produces
    mtp.* prefixed keys, matching sglang's load_weights expectations.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        required_fields = (
            "target_layer_ids",
            "mask_token_id",
            "num_anchors",
            "enable_confidence_head",
            "markov_rank",
        )
        for field in required_fields:
            assert hasattr(config, field), f"config.{field} must be provided."
        if int(config.markov_rank) > 0:
            assert hasattr(config, "markov_head_type"), (
                "config.markov_head_type must be provided when markov_rank > 0."
            )
        if bool(config.enable_confidence_head):
            assert hasattr(config, "confidence_head_with_markov"), (
                "config.confidence_head_with_markov must be provided when "
                "enable_confidence_head is true."
            )

        self.target_layer_ids = config.target_layer_ids
        self.block_size = int(config.block_size)
        self.mask_token_id = config.mask_token_id
        self.num_anchors = int(config.num_anchors)

        # Embedding (tied to target at runtime, but needed for training)
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )

        # Main projection: project target hidden states to draft space
        self.main_proj = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.main_norm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Decoder layers (MLA-based)
        # Registered as direct children with numeric names (mtp.0, mtp.1, ...)
        # so state_dict keys are mtp.{i}.* matching sglang's load_weights.
        self._layer_indices = list(range(config.num_hidden_layers))
        for layer_idx in self._layer_indices:
            self.add_module(
                str(layer_idx), Glm5DSparkDecoderLayer(config, layer_idx)
            )

        # Final norm
        self.norm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Shared head (norm only; head is tied to target at runtime)
        self.shared_head = nn.Module()
        self.shared_head.norm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # LM head (tied to target at runtime, but needed for training)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # Markov head
        self.markov_head = build_markov_head(config)

        # Confidence head — always created to match sglang's Glm5DSparkModel
        # which always creates DSparkConfidenceHead(hidden_size + markov_rank).
        # Uses bias=False and float32 to match sglang exactly. The loss only
        # includes confidence when confidence_head_alpha > 0.
        self.enable_confidence_head = bool(config.enable_confidence_head)
        self.confidence_head_with_markov = True  # always, to match sglang
        confidence_input_dim = int(config.hidden_size) + int(config.markov_rank)
        self.confidence_head = Glm5DSparkConfidenceHead(
            input_dim=confidence_input_dim
        )

        # Rotary embedding (no learnable params, won't appear in state_dict)
        self._rotary_emb = Glm5DSparkRotaryEmbedding(config)

    def initialize_embeddings_and_head(
        self, *, embed_tokens, lm_head, freeze=True
    ):
        assert self.embed_tokens.weight.shape == embed_tokens.weight.shape
        assert self.lm_head.weight.shape == lm_head.weight.shape
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_tokens.weight.detach())
            self.lm_head.weight.copy_(lm_head.weight.detach())
        if freeze:
            self.set_embedding_head_trainable(False)

    def set_embedding_head_trainable(self, trainable: bool):
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.shared_head.norm(hidden_states))

    def predict_confidence_step(
        self, hidden_states, prev_token_ids=None
    ):
        if self.confidence_head is None:
            return None
        # Always use markov embeddings (matches sglang's DSparkConfidenceHead)
        assert self.markov_head is not None
        assert prev_token_ids is not None
        prev_embeddings = self.markov_head.get_prev_embeddings(
            prev_token_ids
        ).to(dtype=hidden_states.dtype)
        return self.confidence_head(hidden_states, prev_embeddings).float()

    def sample_draft_tokens(
        self, base_logits, *, first_prev_token_ids, temperature=0.0,
        hidden_states=None,
    ):
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size, 0, dtype=torch.long, device=base_logits.device
            )
            return empty_tokens, base_logits
        if self.markov_head is None:
            return sample_tokens(base_logits, temperature), base_logits
        return self.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=first_prev_token_ids,
            hidden_states=hidden_states,
            temperature=temperature,
        )

    def sample_draft_token_step(
        self, base_logits, *, prev_token_ids, temperature=0.0,
        hidden_states=None,
    ):
        assert base_logits.ndim == 2
        if self.markov_head is None:
            step_logits = base_logits
        else:
            step_logits = self.markov_head.apply_step_logits(
                base_logits,
                token_ids=prev_token_ids,
                hidden_states=hidden_states,
            )
        sampled_token_ids = sample_tokens(
            step_logits.unsqueeze(1), temperature=temperature
        ).squeeze(1)
        return sampled_token_ids, step_logits

    def _forward_backbone(
        self, *, position_ids, attention_mask=None, noise_embedding=None,
        target_hidden_states=None, **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        # Project target hidden states through main_proj + main_norm
        target_hidden_states = self.main_norm(
            self.main_proj(target_hidden_states)
        )
        # Compute rotary embeddings for all positions (context + draft)
        # position_ids covers both context and draft positions
        position_embeddings = self._compute_position_embeddings(
            hidden_states, position_ids
        )
        for layer_idx in self._layer_indices:
            layer = getattr(self, str(layer_idx))
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def _compute_position_embeddings(self, hidden_states, position_ids):
        """Compute freqs_cis for all positions.

        position_ids has shape [bsz, ctx_len + draft_len] covering both
        context and draft positions. We use a dummy tensor for device/dtype
        since rotary_emb only needs them for the output.
        """
        # Create a dummy tensor with the right device/dtype for rotary_emb
        dummy = hidden_states.new_zeros(
            position_ids.shape[0], position_ids.shape[1], self.config.hidden_size
        )
        return self._rotary_emb(dummy, position_ids)

    def forward(
        self, input_ids, target_hidden_states, loss_mask,
        target_last_hidden_states=None,
    ) -> DSparkForwardOutput:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len, loss_mask=loss_mask,
            num_anchors=self.num_anchors, device=device,
        )
        noise_embedding = create_noise_embed(
            self.embed_tokens, input_ids, anchor_positions, block_keep_mask,
            mask_token_id=self.mask_token_id, block_size=self.block_size,
        )
        context_position_ids = torch.arange(
            seq_len, device=device
        ).unsqueeze(0).expand(bsz, -1)
        draft_position_ids = create_position_ids(
            anchor_positions, self.block_size
        )
        full_position_ids = torch.cat(
            [context_position_ids, draft_position_ids], dim=1
        )
        dspark_attn_mask = create_dspark_attention_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len, block_size=self.block_size, device=device,
        )
        output_hidden = self._forward_backbone(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden_states=target_hidden_states,
            attention_mask=dspark_attn_mask,
        )

        num_blocks = anchor_positions.size(1)
        output_hidden_4d = output_hidden.reshape(
            bsz, num_blocks, self.block_size, -1
        )

        label_offsets = torch.arange(
            1, self.block_size + 1, device=device
        ).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2, safe_label_indices,
        )
        aligned_target_logits = None
        if target_last_hidden_states is not None:
            target_pred_indices = (safe_label_indices - 1).clamp(min=0)
            aligned_target_hidden = torch.gather(
                target_last_hidden_states.unsqueeze(1).expand(
                    -1, anchor_positions.size(1), -1, -1,
                ),
                2,
                target_pred_indices.unsqueeze(-1).expand(
                    -1, -1, -1, target_last_hidden_states.size(-1),
                ),
            )
            aligned_target_logits = self.compute_logits(aligned_target_hidden)
        eval_mask = build_eval_mask(
            seq_len=seq_len, loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = torch.gather(
            input_ids, 1, anchor_positions
        )
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1,
        )
        draft_logits = self.compute_logits(output_hidden).reshape(
            bsz, num_blocks, self.block_size, -1,
        )
        if self.markov_head is not None:
            draft_logits = self.markov_head.apply_block_logits(
                draft_logits,
                token_ids=prev_token_ids,
                hidden_states=output_hidden_4d,
            )

        log_sampler_stats(
            seq_len=seq_len, loss_mask=loss_mask, eval_mask=eval_mask,
            block_keep_mask=block_keep_mask, block_size=self.block_size,
            num_anchors=self.num_anchors,
        )

        confidence_pred = None
        if self.confidence_head is not None:
            # Always use markov embeddings (matches sglang's forward)
            prev_embeddings = self.markov_head.get_prev_embeddings(
                prev_token_ids
            ).to(dtype=output_hidden_4d.dtype)
            confidence_pred = self.confidence_head(
                output_hidden_4d, prev_embeddings
            ).float()

        return DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
        )


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class Glm5DSparkModel(DeepseekV2PreTrainedModel):
    """GLM-5.2 DSpark draft model with native MLA.

    All DSpark submodules are under self.mtp so save_pretrained produces
    mtp.* prefixed keys. The config's architectures field is set to
    ["Glm5ForCausalLMDSpark"] for sglang compatibility.

    The rotary embedding is stored outside self.mtp (it has no learnable
    parameters, so it won't appear in the checkpoint).
    """

    _no_split_modules = ["Glm5DSparkDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.mtp = Glm5DSparkMtp(config)
        self.post_init()

    def _init_weights(self, module):
        """Initialize weights for all module types, including custom ones.

        DeepseekV2PreTrainedModel._init_weights only handles nn.Linear and
        nn.Embedding. We extend it to handle Glm5DSparkGate (nn.Parameter)
        and ensure all modules get proper initialization.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Glm5DSparkGate):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )
            module.e_score_correction_bias.data.zero_()
        elif isinstance(module, Glm5DSparkConfidenceHead):
            if module.proj.weight is not None:
                module.proj.weight.data.normal_(
                    mean=0.0, std=self.config.initializer_range
                )

    # --- Attribute delegation to self.mtp ---
    # The trainer and data validation access some attributes directly on the
    # model object (e.g., draft_model.target_layer_ids). Delegate these to mtp.

    @property
    def target_layer_ids(self):
        return self.mtp.target_layer_ids

    @property
    def block_size(self):
        return self.mtp.block_size

    @property
    def mask_token_id(self):
        return self.mtp.mask_token_id

    @property
    def num_anchors(self):
        return self.mtp.num_anchors

    @property
    def enable_confidence_head(self):
        return self.mtp.enable_confidence_head

    @property
    def confidence_head(self):
        return self.mtp.confidence_head

    @property
    def markov_head(self):
        return self.mtp.markov_head

    def initialize_embeddings_and_head(
        self, *, embed_tokens, lm_head, freeze=True
    ):
        self.mtp.initialize_embeddings_and_head(
            embed_tokens=embed_tokens, lm_head=lm_head, freeze=freeze
        )

    def set_embedding_head_trainable(self, trainable: bool):
        self.mtp.set_embedding_head_trainable(trainable)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mtp.compute_logits(hidden_states)

    def predict_confidence_step(self, hidden_states, prev_token_ids=None):
        return self.mtp.predict_confidence_step(
            hidden_states, prev_token_ids=prev_token_ids
        )

    def sample_draft_tokens(
        self, base_logits, *, first_prev_token_ids, temperature=0.0,
        hidden_states=None,
    ):
        return self.mtp.sample_draft_tokens(
            base_logits,
            first_prev_token_ids=first_prev_token_ids,
            temperature=temperature,
            hidden_states=hidden_states,
        )

    def sample_draft_token_step(
        self, base_logits, *, prev_token_ids, temperature=0.0,
        hidden_states=None,
    ):
        return self.mtp.sample_draft_token_step(
            base_logits,
            prev_token_ids=prev_token_ids,
            temperature=temperature,
            hidden_states=hidden_states,
        )

    def forward(
        self, input_ids, target_hidden_states, loss_mask,
        target_last_hidden_states=None,
    ) -> DSparkForwardOutput:
        return self.mtp(
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=target_last_hidden_states,
        )


__all__ = [
    "Glm5DSparkModel",
    "Glm5DSparkMLAAttention",
    "Glm5DSparkDecoderLayer",
    "Glm5DSparkMoe",
    "Glm5DSparkGate",
    "Glm5DSparkRotaryEmbedding",
]
