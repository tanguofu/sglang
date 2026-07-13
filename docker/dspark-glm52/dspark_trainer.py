import os

import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

from deepspec.data import CacheCollator
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.gemma4.config import (
    build_draft_config as build_gemma4_draft_config,
)
from deepspec.modeling.dspark.glm5 import Glm5DSparkModel
from deepspec.modeling.dspark.glm5.config import (
    build_draft_config as build_glm5_draft_config,
)
from deepspec.modeling.dspark.glm5.init_from_target import (
    initialize_layers_from_target,
)
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import (
    build_draft_config as build_qwen3_draft_config,
)
from deepspec.trainer.base_trainer import BaseTrainer


class Qwen3DSparkTrainer(BaseTrainer):
    data_collator_cls = CacheCollator

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3DSparkModel(draft_config)

    # Training step.
    def run_batch(self, batch):
        import torch

        ths = batch["target_hidden_states"]
        tlhs = batch["target_last_hidden_states"]
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"]

        # Some samples in the v9 cache have NaN in target_hidden_states.
        # Replace NaN/inf with 0 before the forward pass to prevent
        # NaN from corrupting the model via FSDP gradient all-reduce.
        ths_has_nan = torch.isnan(ths).any().item()
        if ths_has_nan:
            ths = torch.nan_to_num(ths, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(tlhs).any().item():
            tlhs = torch.nan_to_num(tlhs, nan=0.0, posinf=0.0, neginf=0.0)

        # Debug: print for all ranks for the first 2 micro-steps
        debug_call = getattr(self, "_debug_call_count", 0)
        if debug_call < 2:
            self._debug_call_count = debug_call + 1
            print(
                f"[DEBUG] rank={self.global_rank} call={debug_call} "
                f"input_ids={input_ids.shape} ths={ths.shape} "
                f"ths_had_nan={ths_has_nan} "
                f"ths_max={ths.float().max().item():.4f} "
                f"ths_min={ths.float().min().item():.4f} "
                f"loss_mask_sum={loss_mask.sum().item()}",
                flush=True,
            )

        outputs = self.model(
            input_ids=input_ids,
            target_hidden_states=ths,
            loss_mask=loss_mask,
            target_last_hidden_states=tlhs,
        )

        if debug_call < 2:
            draft_logits = outputs.draft_logits
            print(
                f"[DEBUG] rank={self.global_rank} call={debug_call} "
                f"draft_logits_nan={torch.isnan(draft_logits).any().item()} "
                f"draft_logits_max={draft_logits.float().max().item():.4f} "
                f"draft_logits_min={draft_logits.float().min().item():.4f} "
                f"eval_mask_sum={outputs.eval_mask.sum().item()}",
                flush=True,
            )

        loss = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=self.args.model.loss_decay_gamma,
            ce_loss_alpha=float(self.args.model.ce_loss_alpha),
            l1_loss_alpha=float(self.args.model.l1_loss_alpha),
            confidence_head_alpha=float(self.args.model.confidence_head_alpha),
        )

        if debug_call < 2:
            print(
                f"[DEBUG] rank={self.global_rank} call={debug_call} "
                f"loss={loss.item()} "
                f"loss_nan={torch.isnan(loss).any().item()}",
                flush=True,
            )

        return loss


class Gemma4DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_gemma4_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Gemma4DSparkModel(draft_config)


class Glm5DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_glm5_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Glm5DSparkModel(draft_config)

    def build_models(self):
        """Override to initialize attention + MoE layers from target model.

        The DSpark inference kv_from_hidden uses the TARGET's kv_a_proj_with_mqa
        to extract KV from the draft's predicted hidden states. If the draft's
        attention weights are randomly initialized, the training/inference KV
        projections mismatch, causing accept_rate near 0%.

        This method loads attention + MoE + layernorm weights from the target's
        corresponding layers (target_layer_ids) and freezes them. Only main_proj,
        main_norm, markov_head, confidence_head, and shared_head.norm remain
        trainable.
        """
        import json as _json
        from safetensors import safe_open

        model_args = self.args.model
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.target_model_name_or_path,
        )
        target_config = AutoConfig.from_pretrained(
            model_args.target_model_name_or_path,
        )

        draft_model = self._build_draft_model(
            target_config=target_config,
            model_args=model_args,
        )
        draft_model = draft_model.to(
            device=self.device, dtype=self.precision_dtype
        )

        # Initialize attention + MoE layers from target model
        target_layer_ids = list(model_args.target_layer_ids)
        draft_model = initialize_layers_from_target(
            model=draft_model,
            target_model_path=model_args.target_model_name_or_path,
            target_layer_ids=target_layer_ids,
            precision_dtype=self.precision_dtype,
        )

        # Load embed_tokens and lm_head from target (same as base_trainer)
        _index_path = os.path.join(
            model_args.target_model_name_or_path, "model.safetensors.index.json"
        )
        with open(_index_path) as _f:
            _index = _json.load(_f)
        _weight_map = _index["weight_map"]
        _target_keys = ["model.embed_tokens.weight", "lm_head.weight"]
        _files_needed = set()
        for _k in _target_keys:
            _f = _weight_map.get(_k)
            if _f:
                _files_needed.add(_f)
        _weights = {}
        for _fname in _files_needed:
            _path = os.path.join(model_args.target_model_name_or_path, _fname)
            with safe_open(_path, framework="pt", device="cpu") as _sf:
                for _k in _sf.keys():
                    if _k in _target_keys:
                        _weights[_k] = _sf.get_tensor(_k)

        class _FakeEmbed(nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = nn.Parameter(weight)

        class _FakeLMHead(nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = nn.Parameter(weight)

        _embed_w = _weights["model.embed_tokens.weight"].to(self.precision_dtype)
        _lm_head_w = _weights["lm_head.weight"].to(self.precision_dtype)
        target_embed_tokens = _FakeEmbed(_embed_w)
        target_lm_head = _FakeLMHead(_lm_head_w)
        draft_model.initialize_embeddings_and_head(
            embed_tokens=target_embed_tokens,
            lm_head=target_lm_head,
            freeze=True,
        )
        return draft_model, tokenizer
