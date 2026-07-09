#!/usr/bin/env python3
"""Parallel DSpark target cache generation for GLM-5.2 using SGLang server API.

Uses ThreadPoolExecutor to send concurrent requests to the SGLang server,
dramatically speeding up target cache generation.
"""
import argparse
import json
import os
import sys
import time
import struct
import requests
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

TARGET_CACHE_VERSION = 2
INDEX_RECORD_STRUCT = struct.Struct("<QIIQQQQQ")
INDEX_RECORD_SIZE = INDEX_RECORD_STRUCT.size
HIDDEN_DTYPE = "bfloat16"
TOKEN_DTYPE = "int32"
MASK_DTYPE = "uint8"

ASSISTANT_TOKEN_ID = 154828
USER_TOKEN_ID = 154827
SYSTEM_TOKEN_ID = 154826
EOS_TOKEN_ID = 154820
THINK_START_ID = 154841
THINK_END_ID = 154842


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-path", default="/data/models/GLM-5.2-FP8")
    p.add_argument("--sglang-url", default="http://localhost:30000")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--shard-size-mb", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=32)
    # Sharding for parallel generation across multiple nodes.
    # Each node processes a contiguous slice [start_idx, end_idx) of the dataset.
    # The original sample_id is preserved so shards can be merged in order.
    p.add_argument("--start-idx", type=int, default=0,
                   help="Inclusive start index of the dataset slice to process")
    p.add_argument("--end-idx", type=int, default=None,
                   help="Exclusive end index of the dataset slice to process")
    return p.parse_args()


def load_training_data(path, max_samples=None):
    samples = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            convs = r.get("conversations", [])
            messages = []
            for c in convs:
                role = c.get("from", c.get("role", "user"))
                if role == "human":
                    role = "user"
                elif role == "gpt":
                    role = "assistant"
                content = c.get("value", c.get("content", ""))
                messages.append({"role": role, "content": content})
            samples.append(messages)
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def format_and_tokenize(tokenizer, messages, max_length):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(text, max_length=max_length, truncation=True,
                     return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids[0]
    return text, input_ids


def create_loss_mask(input_ids):
    mask = torch.zeros(len(input_ids), dtype=torch.long)
    in_assistant = False
    for i, tid in enumerate(input_ids.tolist()):
        if tid == ASSISTANT_TOKEN_ID:
            in_assistant = True
        if in_assistant:
            mask[i] = 1
        if tid in (EOS_TOKEN_ID, THINK_END_ID) and in_assistant:
            in_assistant = False
    return mask


def get_hidden_states(sglang_url, text, max_length):
    resp = requests.post(
        f"{sglang_url}/generate",
        json={
            "text": text[:max_length * 4],
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_hidden_states": True,
        },
        timeout=600,
    )
    data = resp.json()
    hs = data.get("meta_info", {}).get("hidden_states", [])
    if not hs:
        raise RuntimeError("No hidden_states in response")
    import base64 as _b64
    if isinstance(hs[0], str):
        raw = _b64.b64decode(hs[0])
        _arr = np.frombuffer(raw, dtype=np.uint16)
        hidden = torch.from_numpy(_arr.copy()).view(torch.bfloat16).reshape(-1, 30720).float()
    else:
        hidden = torch.from_numpy(np.array(hs[0], dtype=np.float32))
    return hidden


def build_sample_bytes(sample_id, input_ids, attention_mask, loss_mask,
                       target_hidden, target_last_hidden):
    input_ids_bytes = input_ids.to(torch.int32).numpy().tobytes()
    attention_mask_bytes = attention_mask.to(torch.uint8).numpy().tobytes()
    loss_mask_bytes = loss_mask.to(torch.uint8).numpy().tobytes()
    target_hidden_bytes = target_hidden.to(torch.bfloat16).cpu().view(torch.uint16).numpy().tobytes()
    target_last_hidden_bytes = target_last_hidden.to(torch.bfloat16).cpu().view(torch.uint16).numpy().tobytes()
    total = (len(input_ids_bytes) + len(attention_mask_bytes) + len(loss_mask_bytes)
             + len(target_hidden_bytes) + len(target_last_hidden_bytes))
    return {
        "sample_id": sample_id,
        "seq_len": len(input_ids),
        "input_ids": input_ids_bytes,
        "attention_mask": attention_mask_bytes,
        "loss_mask": loss_mask_bytes,
        "target_hidden_states": target_hidden_bytes,
        "target_last_hidden_states": target_last_hidden_bytes,
        "total_nbytes": total,
    }


class SimpleCacheWriter:
    def __init__(self, output_dir, shard_size_bytes):
        self.output_dir = output_dir
        self.shard_size_bytes = shard_size_bytes
        os.makedirs(output_dir, exist_ok=True)
        self.shard_id = 0
        self.shard_files = []
        self.shard_fh = None
        self.current_shard_size = 0
        self.index_fh = open(os.path.join(output_dir, "samples.idx"), "wb")
        self.num_samples = 0
        self._open_shard()

    def _open_shard(self):
        fname = f"shard-{self.shard_id:05d}.bin"
        self.shard_files.append(fname)
        self.shard_fh = open(os.path.join(self.output_dir, fname), "wb")
        self.current_shard_size = 0

    def write(self, sample):
        input_ids_offset = self.current_shard_size
        self.shard_fh.write(sample["input_ids"])
        self.current_shard_size += len(sample["input_ids"])

        attention_mask_offset = self.current_shard_size
        self.shard_fh.write(sample["attention_mask"])
        self.current_shard_size += len(sample["attention_mask"])

        loss_mask_offset = self.current_shard_size
        self.shard_fh.write(sample["loss_mask"])
        self.current_shard_size += len(sample["loss_mask"])

        target_hidden_offset = self.current_shard_size
        self.shard_fh.write(sample["target_hidden_states"])
        self.current_shard_size += len(sample["target_hidden_states"])

        target_last_hidden_offset = self.current_shard_size
        self.shard_fh.write(sample["target_last_hidden_states"])
        self.current_shard_size += len(sample["target_last_hidden_states"])

        self.index_fh.write(INDEX_RECORD_STRUCT.pack(
            sample["sample_id"], self.shard_id, sample["seq_len"],
            input_ids_offset, attention_mask_offset, loss_mask_offset,
            target_hidden_offset, target_last_hidden_offset
        ))
        self.num_samples += 1
        if self.current_shard_size >= self.shard_size_bytes:
            self.shard_fh.flush()
            os.fsync(self.shard_fh.fileno())
            self.shard_fh.close()
            self.shard_id += 1
            self._open_shard()

    def write_manifest(self, target_layer_ids, hidden_size, target_model_name_or_path=None):
        shards = [{"shard_id": i, "file_name": f} for i, f in enumerate(self.shard_files)]
        manifest = {
            "version": TARGET_CACHE_VERSION,
            "num_samples": self.num_samples,
            "num_shards": len(shards),
            "target_layer_ids": [int(x) for x in target_layer_ids],
            "hidden_dtype": HIDDEN_DTYPE,
            "token_dtype": TOKEN_DTYPE,
            "mask_dtype": MASK_DTYPE,
            "index_record_size": INDEX_RECORD_SIZE,
            "hidden_size": int(hidden_size),
            "target_model_name_or_path": target_model_name_or_path or "",
            "shards": shards,
        }
        with open(os.path.join(self.output_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    def close(self):
        if self.shard_fh:
            self.shard_fh.flush()
            os.fsync(self.shard_fh.fileno())
            self.shard_fh.close()
        self.index_fh.flush()
        os.fsync(self.index_fh.fileno())
        self.index_fh.close()


def process_sample(args_tuple):
    idx, messages, tokenizer, sglang_url, max_length, hidden_size = args_tuple
    try:
        text, input_ids = format_and_tokenize(tokenizer, messages, max_length)
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        loss_mask = create_loss_mask(input_ids)
        hidden = get_hidden_states(sglang_url, text, max_length)
        seq_len = len(input_ids)
        if len(hidden) > seq_len:
            hidden = hidden[:seq_len]
        elif len(hidden) < seq_len:
            pad = torch.zeros(seq_len - len(hidden), hidden.shape[-1], dtype=hidden.dtype)
            hidden = torch.cat([hidden, pad], dim=0)
        target_last_hidden = hidden[:, -hidden_size:]
        sample = build_sample_bytes(idx, input_ids, attention_mask, loss_mask, hidden, target_last_hidden)
        return idx, sample, None
    except Exception as e:
        return idx, None, str(e)


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Loading training data from {args.train_data}", flush=True)
    samples = load_training_data(args.train_data, args.max_samples)
    print(f"  Loaded {len(samples)} total samples", flush=True)

    # Apply sharding: each node processes a contiguous slice [start_idx, end_idx)
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(samples)
    end = min(end, len(samples))
    slice_samples = samples[start:end]
    print(f"  Shard: samples[{start}:{end}] = {len(slice_samples)} samples", flush=True)

    target_layer_ids = [15, 31, 47, 63, 76]
    hidden_size = 6144

    # Clean old cache (only when starting fresh from idx 0; partial shards append)
    import shutil
    if start == 0 and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    writer = SimpleCacheWriter(args.output_dir, args.shard_size_mb * 1024 * 1024)

    print(f"Generating target cache with {args.num_workers} workers (max_length={args.max_length})...", flush=True)
    t0 = time.time()

    # Prepare tasks: preserve original sample_id (idx) for ordered merge
    tasks = [(start + i, messages, tokenizer, args.sglang_url, args.max_length, hidden_size)
             for i, messages in enumerate(slice_samples)]

    # Process in batches to maintain order
    batch_size = args.num_workers * 2
    results_buffer = {}

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        next_write_idx = start  # global idx, must match results_buffer keys

        for task in tasks:
            future = executor.submit(process_sample, task)
            futures[future] = task[0]

            if len(futures) >= batch_size:
                # Wait for at least some to complete
                for future in as_completed(futures):
                    idx, sample, error = future.result()
                    if error:
                        print(f"  [WARN] Sample {idx}: {error} — skipping", flush=True)
                        # Advance next_write_idx past failed samples to avoid deadlock
                        if idx == next_write_idx:
                            next_write_idx += 1
                    else:
                        results_buffer[idx] = sample
                    del futures[future]
                    break

                # Write completed samples in order
                while next_write_idx in results_buffer:
                    writer.write(results_buffer.pop(next_write_idx))
                    next_write_idx += 1

                if (next_write_idx) % 100 == 0 and next_write_idx > 0:
                    elapsed = time.time() - t0
                    rate = next_write_idx / elapsed
                    eta = (len(slice_samples) - next_write_idx) / rate if rate > 0 else 0
                    print(f"  {next_write_idx}/{len(slice_samples)} | {rate:.1f} samples/s | ETA: {eta/3600:.1f}h", flush=True)

        # Wait for remaining futures
        for future in as_completed(futures):
            idx, sample, error = future.result()
            if error:
                print(f"  [WARN] Sample {idx}: {error}", flush=True)
            else:
                results_buffer[idx] = sample

        # Write remaining samples in order
        while next_write_idx in results_buffer:
            writer.write(results_buffer.pop(next_write_idx))
            next_write_idx += 1

    writer.write_manifest(target_layer_ids, hidden_size, args.model_path)
    writer.close()

    elapsed = time.time() - t0
    print(f"\nDone! {writer.num_samples} samples written to {args.output_dir}", flush=True)
    print(f"Total time: {elapsed:.1f}s ({writer.num_samples/elapsed:.1f} samples/s)", flush=True)


if __name__ == "__main__":
    main()
