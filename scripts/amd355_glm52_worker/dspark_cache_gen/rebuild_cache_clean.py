#!/usr/bin/env python3
"""Rebuild cache index, skipping corrupt samples (NaN/extreme hidden states).

Creates a new cache dir with only clean samples, rewriting the index and
copying shard data for valid samples only.
"""
import struct, json, os, sys, shutil, torch
import numpy as np

src_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/dspark_target_cache_v9_coding_clean/partial_0"
dst_dir = sys.argv[2] if len(sys.argv) > 2 else "/data/dspark_target_cache_v9_coding_clean_merged"
skip_first = int(sys.argv[3]) if len(sys.argv) > 3 else 57

manifest = json.load(open(f"{src_dir}/manifest.json"))
hidden_size = manifest["hidden_size"]
num_layers = len(manifest["target_layer_ids"])
idx_struct = struct.Struct("<QIIQQQQQ")
idx_bytes = open(f"{src_dir}/samples.idx", "rb").read()
n = len(idx_bytes) // idx_struct.size

shard_files = {s["shard_id"]: open(f"{src_dir}/{s['file_name']}", "rb") for s in manifest["shards"]}

print(f"Source: {src_dir} ({n} samples)")
print(f"Destination: {dst_dir}")
print(f"Skipping first {skip_first} samples (warmup corruption)")
print(f"Expected clean samples: {n - skip_first}")

# Read all index entries
entries = []
for i in range(n):
    entry = idx_struct.unpack(idx_bytes[i * idx_struct.size:(i + 1) * idx_struct.size])
    entries.append(entry)

# Create destination
os.makedirs(dst_dir, exist_ok=True)

# Write clean samples to new shards
new_shard_id = 0
new_shard_fh = open(f"{dst_dir}/shard-{new_shard_id:05d}.bin", "wb")
new_shard_size = 0
shard_size_limit = 2048 * 1024 * 1024  # 2GB
new_shard_files = []
new_index_records = []
new_sample_id = 0
num_clean = 0
num_bad = 0

for pos in range(n):
    if pos < skip_first:
        # Quick-check: is this sample actually bad?
        entry = entries[pos]
        sample_id, shard_id, seq_len, ii_off, am_off, lm_off, th_off, tlh_off = entry
        fh = shard_files[shard_id]
        th_nbytes = seq_len * num_layers * hidden_size * 2
        fh.seek(th_off)
        raw = fh.read(th_nbytes)
        arr = np.frombuffer(raw, dtype=np.uint16).copy()
        th = torch.from_numpy(arr).view(torch.bfloat16).float().reshape(seq_len, num_layers, hidden_size)
        if bool(torch.isnan(th).any()) or float(th.abs().max()) > 1e4:
            num_bad += 1
            continue  # skip corrupt sample
        # Fall through — sample is actually clean
    entry = entries[pos]
    sample_id, shard_id, seq_len, ii_off, am_off, lm_off, th_off, tlh_off = entry
    fh = shard_files[shard_id]

    # Read all fields
    ii_nbytes = seq_len * 4  # int32
    am_nbytes = seq_len * 1   # uint8
    lm_nbytes = seq_len * 1   # uint8
    th_nbytes = seq_len * num_layers * hidden_size * 2  # bf16
    tlh_nbytes = th_nbytes    # same size

    fh.seek(ii_off)
    ii_data = fh.read(ii_nbytes)
    am_data = fh.read(am_nbytes) if am_off != ii_off else fh.read(am_nbytes)
    fh.seek(am_off); am_data = fh.read(am_nbytes)
    fh.seek(lm_off); lm_data = fh.read(lm_nbytes)
    fh.seek(th_off); th_data = fh.read(th_nbytes)
    fh.seek(tlh_off); tlh_data = fh.read(tlh_nbytes)

    # Write to new shard
    ii_offset = new_shard_size
    new_shard_fh.write(ii_data); new_shard_size += len(ii_data)
    am_offset = new_shard_size
    new_shard_fh.write(am_data); new_shard_size += len(am_data)
    lm_offset = new_shard_size
    new_shard_fh.write(lm_data); new_shard_size += len(lm_data)
    th_offset = new_shard_size
    new_shard_fh.write(th_data); new_shard_size += len(th_data)
    tlh_offset = new_shard_size
    new_shard_fh.write(tlh_data); new_shard_size += len(tlh_data)

    new_index_records.append(idx_struct.pack(
        new_sample_id, new_shard_id, seq_len,
        ii_offset, am_offset, lm_offset, th_offset, tlh_offset
    ))
    new_sample_id += 1
    num_clean += 1

    if new_shard_size >= shard_size_limit:
        new_shard_fh.flush()
        new_shard_fh.close()
        new_shard_files.append(f"shard-{new_shard_id:05d}.bin")
        new_shard_id += 1
        new_shard_fh = open(f"{dst_dir}/shard-{new_shard_id:05d}.bin", "wb")
        new_shard_size = 0

    if num_clean % 200 == 0:
        print(f"  Written {num_clean} clean samples (pos {pos}), {num_bad} skipped", flush=True)

new_shard_fh.flush()
new_shard_fh.close()
if new_shard_size > 0:
    new_shard_files.append(f"shard-{new_shard_id:05d}.bin")

# Write new index
with open(f"{dst_dir}/samples.idx", "wb") as f:
    for record in new_index_records:
        f.write(record)

# Write new manifest
new_manifest = {
    "version": manifest["version"],
    "num_samples": num_clean,
    "num_shards": len(new_shard_files),
    "target_layer_ids": manifest["target_layer_ids"],
    "hidden_dtype": manifest["hidden_dtype"],
    "token_dtype": manifest["token_dtype"],
    "mask_dtype": manifest["mask_dtype"],
    "index_record_size": manifest["index_record_size"],
    "hidden_size": hidden_size,
    "target_model_name_or_path": manifest["target_model_name_or_path"],
    "shards": [{"shard_id": i, "file_name": f} for i, f in enumerate(new_shard_files)],
}
with open(f"{dst_dir}/manifest.json", "w") as f:
    json.dump(new_manifest, f, indent=2)

print(f"\nDone! {num_clean} clean samples written to {dst_dir}")
print(f"  Skipped: {num_bad} corrupt samples")
print(f"  Shards: {len(new_shard_files)}")
