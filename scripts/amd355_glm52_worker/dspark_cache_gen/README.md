# DSpark Cache Generation & Training Scripts

Scripts for generating clean DSpark target cache and training the GLM-5.2
DSpark draft model on AMD MI355X (ROCm). All scripts were validated on
2026-07-07 and produced clean training data (loss=1.60, not NaN).

## Root causes fixed

1. **DSPARK mode returns decode hidden states** (few tokens), not prefill
   (all tokens). Fix: use non-speculative server (no `--speculative-algorithm`).
2. **Overlap scheduler race**: D2H copy of hidden_states on `copy_stream`
   races with the next forward (CUDA graph replay) overwriting the same
   static GPU buffer on `forward_stream`. Fix: `--disable-overlap-schedule`
   server flag + `.clone()` in CUDA graph runners (see source diff in
   `python/sglang/srt/model_executor/runner/{decode,prefill}_cuda_graph_runner.py`).
3. **Script bug**: `next_write_idx = 0` but `results_buffer` is keyed by
   global sample idx. When `start_idx > 0`, no samples are written. Fix:
   `next_write_idx = start`.
4. **Missing config**: `dspark_target_layer_ids` must be in the model's
   `config.json` for the nonspec server to capture 5 layers (30720 values)
   instead of 1 (6144). Add: `"dspark_target_layer_ids": [15, 31, 47, 63, 76]`.
5. **Warmup corruption**: the first ~57 samples are corrupt (server cold
   start). Skip them with `rebuild_cache_clean.py`.

## Verified clean config

- Server: nonspec, `--quantization fp8`, `--kv-cache-dtype bf16`,
  `--enable-return-hidden-states`, `--disable-overlap-schedule`,
  `--context-length 4096`
- Cache gen: `generate_target_cache_v4.py --num-workers 16 --max-length 1000`
- Training: `local_batch_size=2`, `sharding_strategy="shard_grad_op"`,
  `global_batch_size=256`

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_target_cache_v4.py` | Generate DSpark target cache via SGLang `/generate` API |
| `check_cache_quality.py` | Scan cache for NaN/Inf/extreme hidden states |
| `rebuild_cache_clean.py` | Rebuild cache index, skipping corrupt warmup samples |
| `test_real_sample.py` | Test hidden states quality with real training samples |
| `test_max_new_tokens.py` | Compare max_new_tokens=0 vs 1 hidden states |
| `test_accept_rate.py` | Test DSpark accept_rate against a DSPARK server |
| `start_v9_1node.sh` | 1-node (8 GPU) training launcher |
| `start_v9_2node.sh` | 2-node (16 GPU) training launcher |
| `start_dspark_test.sh` | Launch DSpark server for accept_rate testing |
| `dspark_glm5_2_v9_clean.py` | Training config (clean cache path) |

## Usage

```bash
# 1. Start nonspec server (on each cache-gen node)
#    Key flags: --enable-return-hidden-states --disable-overlap-schedule
#    --kv-cache-dtype bf16  (no --speculative-algorithm)

# 2. Generate cache (16 workers, safe + fast)
python3 generate_target_cache_v4.py \
    --train-data /data/dspark_v9_all_coding.jsonl \
    --output-dir /data/dspark_target_cache_v9_coding_clean/partial_0 \
    --model-path /data/models/GLM-5.2-FP8 \
    --sglang-url http://localhost:30000 \
    --max-length 1000 --num-workers 16 \
    --start-idx 0 --end-idx 2500

# 3. Verify quality
python3 check_cache_quality.py /data/dspark_target_cache_v9_coding_clean/partial_0 50

# 4. Rebuild clean (skip warmup corruption)
python3 rebuild_cache_clean.py \
    /data/dspark_target_cache_v9_coding_clean/partial_0 \
    /data/dspark_target_cache_v9_coding_clean_merged 57

# 5. Train
bash start_v9_1node.sh

# 6. Test accept_rate (after checkpoint)
bash start_dspark_test.sh /data/checkpoints/.../step_100
python3 test_accept_rate.py http://localhost:30000 20
```
