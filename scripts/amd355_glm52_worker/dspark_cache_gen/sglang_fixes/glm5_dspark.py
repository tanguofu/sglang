"""GLM-5.2 DSpark draft model for speculative decoding.

Fully aligned with the official DeepSeek-V4 DSpark implementation (PR #29538).
Uses DeepseekV2DecoderLayer (GLM-5.2's native MLA decoder layer) — the same
way DeepseekV4DSparkModel uses DeepseekV4DecoderLayer.

GLM-5.2's main model is GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM), which uses
MLA (Multi-head Latent Attention) with DSA (DeepSeek Sparse Attention). The
DSpark draft model inherits from GlmMoeDsaForCausalLM and uses the same
DeepseekV2DecoderLayer with is_nextn=True.

DSA is automatically disabled for the draft model because the checkpoint's
architectures field is "Glm5ForCausalLMDSpark" (not in is_deepseek_dsa's
architecture list), so no Indexer is created and full attention is used.

The hc_head collapse (hypercomplex attention in DeepSeek V4) is replaced
with a pass-through since GLM-5.2 has no hc_head — the final RMSNorm is
applied in forward_backbone, and shared_head.norm is applied in the
worker's markov refinement.

The draft model's lm_head and embed_tokens are tied to the target model
at runtime by DSparkWorkerV2.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.configs.model_config import is_deepseek_dsa
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import (
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer
from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLM
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import BumpAllocator, add_prefix

logger = logging.getLogger(__name__)


class DSparkMarkovHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        markov_rank: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            vocab_size,
            markov_rank,
            enable_tp=False,
            prefix=add_prefix("markov_w1", prefix),
        )
        self.markov_w2 = ParallelLMHead(
            vocab_size,
            markov_rank,
            quant_config=quant_config,
            prefix=add_prefix("markov_w2", prefix),
        )

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def project_bias(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.linear(embeddings, self.markov_w2.weight)


class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False, dtype=torch.float32)

    def forward(
        self, hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(features.float()).squeeze(-1)


class Glm5DSparkDecoderLayer(DeepseekV2DecoderLayer):
    """Decoder layer for GLM-5.2 DSpark that respects first_k_dense_replace.

    Unlike DeepseekV2DecoderLayer which forces all is_nextn layers to be MoE,
    this subclass respects the config's first_k_dense_replace so that dense
    layers (0..first_k_dense_replace-1) use DeepseekV2MLP and sparse layers
    use DeepseekV2MoE — matching the trained checkpoint's structure.
    """

    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:
        return (
            self.config.n_routed_experts is not None
            and layer_id >= self.config.first_k_dense_replace
            and layer_id % self.config.moe_layer_freq == 0
        )


class Glm5DSparkModel(nn.Module):
    """GLM-5.2 DSpark draft model.

    Mirrors DeepseekV4DSparkModel (PR #29538) but uses DeepseekV2DecoderLayer
    (GLM-5.2's native MLA decoder layer with is_nextn=True) instead of
    DeepseekV4DecoderLayer. The hc_head collapse is replaced with a
    pass-through since GLM-5.2 has no hypercomplex attention — the final
    RMSNorm is applied in forward_backbone, and shared_head.norm is applied
    later in the worker's markov refinement.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.block_size = int(getattr(config, "dspark_block_size", 7))
        self.markov_rank = int(getattr(config, "dspark_markov_rank", 256))
        self.noise_token_id = int(getattr(config, "dspark_noise_token_id", 154821))
        self.target_layer_ids = list(
            getattr(config, "dspark_target_layer_ids", [15, 31, 47, 63, 76])
        )
        self.num_dspark_layers = num_dspark_layers = get_dspark_num_layers(config)

        # Required by DeepseekV2WeightLoaderMixin.post_load_weights
        self.start_layer = 0
        self.end_layer = num_dspark_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.main_proj = ReplicatedLinear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("main_proj", prefix),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=self.rms_norm_eps)

        self.layers = nn.ModuleList(
            [
                Glm5DSparkDecoderLayer(
                    config=config,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    is_nextn=True,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(num_dspark_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=self.rms_norm_eps)

        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            self.markov_rank,
            quant_config=quant_config,
            prefix=add_prefix("markov_head", prefix),
        )
        self.confidence_head = DSparkConfidenceHead(
            config.hidden_size + self.markov_rank
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(
            config.hidden_size, eps=self.rms_norm_eps
        )

    def project_main_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        projected, _ = self.main_proj(main_hidden)
        return self.main_norm(projected)

    def forward_backbone(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual: Optional[torch.Tensor] = None
        topk_indices = None
        device = hidden_states.device
        zero_allocator = BumpAllocator(
            buffer_size=self.num_dspark_layers * 2,
            dtype=torch.float32,
            device=device,
        )
        for layer in self.layers:
            hidden_states, residual, topk_indices = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                zero_allocator,
            )
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def collapse_block_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Collapse block hidden states to per-block representation.

        DeepseekV4 uses hc_head (hypercomplex attention); GLM-5.2 has no
        hc_head, so the final RMSNorm in forward_backbone already produces
        the per-block representation. shared_head.norm is applied later
        in the worker's markov refinement.
        """
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.forward_backbone(input_ids, positions, forward_batch)
        return self.collapse_block_hidden(hidden_states)


DSPARK_DEFAULT_NUM_LAYERS = 3
_warned_dspark_num_layers_fallback = False


def get_dspark_num_layers(config: PretrainedConfig) -> int:
    """Draft depth from the checkpoint config, or DSPARK_DEFAULT_NUM_LAYERS if unset."""
    num_layers = getattr(config, "dspark_num_layers", None)
    if num_layers is None or int(num_layers) <= 0:
        global _warned_dspark_num_layers_fallback
        if not _warned_dspark_num_layers_fallback:
            _warned_dspark_num_layers_fallback = True
            logger.warning(
                "DSpark draft config has no positive dspark_num_layers; "
                "falling back to %d layers.",
                DSPARK_DEFAULT_NUM_LAYERS,
            )
        return DSPARK_DEFAULT_NUM_LAYERS
    return int(num_layers)


class Glm5ForCausalLMDSpark(GlmMoeDsaForCausalLM):
    """GLM-5.2 DSpark entry point.

    Mirrors DeepseekV4ForCausalLMDSpark (PR #29538). The draft model's
    embed_tokens and lm_head are tied to the target model at runtime
    by DSparkWorkerV2.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        # Replicate DeepseekV2ForCausalLM.__init__ setup for weight loading
        self.packed_modules_mapping = {}
        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj_with_mqa"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]
        if quant_config is not None:
            quant_config.update_packed_modules_mapping(self.packed_modules_mapping)

        self.pp_group = get_parallel().pp_group
        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        self.num_fused_shared_experts = 0
        self.determine_num_fused_shared_experts()
        self.use_dsa = is_deepseek_dsa(config)

        self.model = Glm5DSparkModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

        self.dsa_enable_prefill_cp = False
        self.mla_enable_prefill_cp = False
        self.cp_rank = self.cp_size = None

        q_lora_rank = (
            config.q_lora_rank if hasattr(config, "q_lora_rank") else None
        )
        get_attn_tp_context().init_context(q_lora_rank, is_deepseek_dsa(config))

    @property
    def block_size(self) -> int:
        return self.model.block_size

    @property
    def num_dspark_layers(self) -> int:
        return self.model.num_dspark_layers

    def project_main_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        return self.model.project_main_hidden(main_hidden)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        block_hidden = self.model(input_ids, positions, forward_batch)
        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=block_hidden
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load DSpark checkpoint weights (mtp.* prefix format).

        Mirrors DeepseekV4ForCausalLMDSpark.load_weights: filters for
        mtp.* prefix, remaps to model.* internal names, then delegates
        to the parent DeepseekV2ForCausalLM.load_weights.

        embed_tokens and lm_head (shared_head.head) are skipped — they
        are tied to the target model at runtime by DSparkWorkerV2.

        Note: DeepseekV2ForCausalLM.load_weights (do_load_weights) already
        calls post_load_weights internally, which handles kv_b_proj MLA
        split (w_kc/w_vc) and FP8/INT8 requantization. This mirrors how
        DeepseekV4ForCausalLMDSpark delegates to super().load_weights.
        """
        dspark_top_level = {
            "main_proj",
            "main_norm",
            "markov_head",
            "confidence_head",
            "shared_head",
        }

        remapped: list[Tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if not name.startswith("mtp."):
                continue
            rest = name[len("mtp.") :]

            # Skip tied weights (tied to target model at runtime)
            if rest.startswith("embed_tokens") or rest.startswith("lm_head"):
                continue
            if rest == "shared_head.head.weight":
                continue

            # Top-level DSpark weights: mtp.{key}.{rest} -> model.{key}.{rest}
            top_key = rest.split(".", 1)[0]
            if top_key in dspark_top_level:
                remapped.append((f"model.{rest}", weight))
                continue

            # norm.weight -> model.norm.weight (final backbone RMSNorm)
            if rest == "norm.weight":
                remapped.append(("model.norm.weight", weight))
                continue

            # Layer weights: mtp.{stage}.{rest} -> model.layers.{stage}.{rest}
            remapped.append((f"model.layers.{rest}", weight))

        GlmMoeDsaForCausalLM.load_weights(self, remapped, is_nextn=False)


EntryClass = [Glm5ForCausalLMDSpark]
