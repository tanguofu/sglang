import os
from deepspec.trainer import Glm5DSparkTrainer

# =============================================================================
# DSpark GLM-5.2 v9 CLEAN — 4-node (32 GPU) training config
# =============================================================================
# Differences from dspark_glm5_2_v9_clean.py (1-node):
#   - local_batch_size: 1 → 8  (32 GPUs, grad_accum = 256/(8*32) = 1)
#   - CPUOffload: disabled via DS_CPU_OFFLOAD=0 env var (set in launch script)
#   - num_train_epochs: 200 (same — 39000 steps at 195 steps/epoch)
#   - Estimated time: ~10-11 hours at ~1 s/step (0.98 s/step measured on 4-node)
#
# Prerequisites on ALL 4 nodes:
#   - /data/dspark_target_cache_v9_coding_clean_merged  (114G, 58 shards)
#   - /data/dspark_v9_all_coding.jsonl                  (training data)
#   - /data/models/GLM-5.2-FP8                          (target model, for config)
#   - DeepSpec code with all fixes (parser, modeling, nan_to_num, ckpt_manager)
# =============================================================================

BASE_TB_DIR = "/data/tensorboard"
BASE_CKPT_DIR = "/data/checkpoints"
project_name = "deepspec"
exp_name = "dspark_glm5_2_v9_256exp_clean_4node"
seed = 42

model = dict(
    target_model_name_or_path="/data/models/GLM-5.2-FP8",
    block_size=7,                          # MTP 7: draft predicts 7 tokens per step
    num_draft_layers=5,
    target_layer_ids=[15, 31, 47, 63, 76],
    mask_token_id=154821,
    num_anchors=256,
    n_routed_experts=256,                  # GLM-5.2 has 256 routed experts
    markov_rank=256,
    markov_head_type='vanilla',
    confidence_head_alpha=0.0,
    confidence_head_with_markov=False,
    loss_decay_gamma=1.0,
    ce_loss_alpha=1.0,
    l1_loss_alpha=0.0,
)

train = dict(
    trainer_cls=Glm5DSparkTrainer,
    lr=1.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=8,                     # 4-node: 8 × 32 GPU = 256 = global_batch_size
    global_batch_size=256,
    num_train_epochs=200,                   # 39000 steps (195 steps/epoch × 200)
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="full_shard",        # params sharded across 32 GPUs
    torch_compile=False,                    # ROCm 7.2 incompatible
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=500,                # save every 500 steps (78 checkpoints)
)

data = dict(
    target_cache_path="/data/dspark_target_cache_v9_coding_clean_merged",
    train_data_path="/data/dspark_v9_all_coding.jsonl",
    chat_template="glm5",
    max_length=1000,
    num_workers=4,
)

def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg["project_name"])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg
    return cfg

# === v9 clean overrides (same as 1-node clean config) ===
model["ce_loss_alpha"] = 1.0
model["l1_loss_alpha"] = 0.0
model["confidence_head_alpha"] = 0.0
model["loss_decay_gamma"] = 1.0
model["learning_rate"] = 1e-4
train["lr"] = 1e-4
data["target_cache_path"] = "/data/dspark_target_cache_v9_coding_clean_merged"
data["train_data_path"] = "/data/dspark_v9_all_coding.jsonl"
data["max_length"] = 1000
logging["checkpoint_dir"] = "/data/checkpoints/deepspec/dspark_glm5_2_v9_256exp_clean_4node"
logging["tensorboard_dir"] = "/data/tensorboard/deepspec/dspark_glm5_2_v9_256exp_clean_4node"
