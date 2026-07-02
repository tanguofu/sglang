"""GLM-5.2 DSpark draft model for speculative decoding.

Adapted from the official DeepSeek-V4 DSpark implementation (PR #29538).
Uses DFlash-style non-causal (bidirectional) decoder layers — matching the
DeepSpec training code where Glm5DSparkModel inherits from Qwen3DSparkModel
with is_causal=False. Adds a Markov refinement head and confidence head.
The draft model's lm_head and embed_tokens are tied to the target model at
runtime.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

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
from sglang.srt.models.dflash import DFlashDecoderLayer
from sglang.srt.models.glm4_moe import Glm4MoeForCausalLM
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

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
            enable_tp=not is_dp_attention_enabled(),
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


class Glm5DSparkModel(nn.Module):
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
                DFlashDecoderLayer(
                    config=config,
                    layer_id=layer_id,
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
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def kv_from_hidden(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        attn_backend,
    ) -> None:
        """Materialize K/V from projected hidden states into the draft KV cache.

        Uses DFlash's kv_proj_only + apply_k_norm + apply_k_rope per layer,
        matching the DFlash worker's _append_target_hidden_sequential path.
        """
        token_to_kv_pool = attn_backend.token_to_kv_pool
        for layer in self.layers:
            attn = layer.self_attn
            k, v = attn.kv_proj_only(x)
            k = attn.apply_k_norm(k)
            k = attn.apply_k_rope(positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            token_to_kv_pool.set_kv_buffer(
                attn.attn,
                cache_loc,
                k,
                v,
                attn.attn.k_scale,
                attn.attn.v_scale,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        return self.forward_backbone(input_ids, positions, forward_batch)


def get_dspark_num_layers(config: PretrainedConfig) -> int:
    return int(getattr(config, "dspark_num_layers", 0) or 3)


class Glm5ForCausalLMDSpark(Glm4MoeForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_parallel().tp_size
        from sglang.srt.distributed import get_pp_group

        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.num_fused_shared_experts = 0
        self.determine_num_fused_shared_experts()

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

    @property
    def block_size(self) -> int:
        return self.model.block_size

    @property
    def num_dspark_layers(self) -> int:
        return self.model.num_dspark_layers

    def project_main_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        return self.model.project_main_hidden(main_hidden)

    def kv_from_hidden(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        attn_backend,
    ) -> None:
        self.model.kv_from_hidden(x, positions, cache_loc, attn_backend)

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
        Glm4MoeForCausalLM.load_weights(self, weights, is_nextn=False)


EntryClass = [Glm5ForCausalLMDSpark]
