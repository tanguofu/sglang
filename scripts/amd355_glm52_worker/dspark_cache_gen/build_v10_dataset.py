#!/usr/bin/env python3
"""Build high-quality DSpark v10 training dataset from multiple sources.

Mixes 3 high-quality coding/instruction datasets, deduplicates, filters by
quality, and outputs ShareGPT-format JSONL compatible with generate_target_cache_v4.py.

Sources:
  1. Magpie-Qwen2.5-Coder-Pro-300K (300K, quality-scored, deduplicated)
  2. OpenHermes-2.5 coding subset (glaive-code-assist + EvolInstruct, ~234K)
  3. Code-290k-ShareGPT (289K, pure code)

Output: /data/dspark_v10_train.jsonl (ShareGPT format)
"""
import json
import os
import hashlib
import random
import pandas as pd
from collections import Counter

random.seed(42)

OUTPUT_PATH = "/data/dspark_v10_train.jsonl"
HF_CACHE = "/data/hf_cache"

# Target: ~300K high-quality samples after dedup + filtering
TARGET_SIZE = 300000


def normalize_to_sharegpt(sample):
    """Normalize any format to ShareGPT {"id": ..., "conversations": [...]}."""
    conversations = []

    # Magpie format: has "conversations" already in ShareGPT format
    if "conversations" in sample and isinstance(sample["conversations"], (list, pd.Series)):
        convs = sample["conversations"]
        if hasattr(convs, "tolist"):
            convs = convs.tolist()
        for c in convs:
            if isinstance(c, dict) and "from" in c and "value" in c:
                conversations.append({"from": c["from"], "value": c["value"]})
            elif isinstance(c, dict) and "role" in c and "content" in c:
                role = "human" if c["role"] == "user" else "gpt"
                conversations.append({"from": role, "value": c["content"]})

    # Instruction/Response format (Magicoder, CodeFeedback)
    elif "instruction" in sample and "response" in sample:
        conversations.append({"from": "human", "value": sample["instruction"]})
        conversations.append({"from": "gpt", "value": sample["response"]})

    # Query/Answer format
    elif "query" in sample and "answer" in sample:
        conversations.append({"from": "human", "value": sample["query"]})
        conversations.append({"from": "gpt", "value": sample["answer"]})

    return conversations


def compute_hash(text):
    """Compute hash of first user message for dedup."""
    return hashlib.md5(text[:500].encode("utf-8")).hexdigest()


def is_quality_sample(conversations, min_assistant_len=20, max_total_chars=12000):
    """Check if a sample meets quality criteria."""
    if len(conversations) < 2:
        return False

    # Must have at least human + gpt
    has_human = any(c["from"] == "human" for c in conversations)
    has_gpt = any(c["from"] == "gpt" for c in conversations)
    if not (has_human and has_gpt):
        return False

    # Check assistant response quality
    assistant_texts = [c["value"] for c in conversations if c["from"] == "gpt"]
    if not assistant_texts:
        return False
    if max(len(t) for t in assistant_texts) < min_assistant_len:
        return False

    # Check total length (avoid truncation)
    total_chars = sum(len(c["value"]) for c in conversations)
    if total_chars > max_total_chars:
        return False
    if total_chars < 30:
        return False

    # Check for empty/garbage content
    for c in conversations:
        if len(c["value"].strip()) == 0:
            return False

    return True


def load_magpie():
    """Load Magpie-Qwen2.5-Coder-Pro-300K with quality filtering."""
    print("Loading Magpie-Qwen2.5-Coder-Pro-300K...")
    all_dfs = []
    for i in range(4):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="Magpie-Align/Magpie-Qwen2.5-Coder-Pro-300K-v0.1",
            filename=f"data/train-0000{i}-of-00004.parquet",
            repo_type="dataset",
            cache_dir=HF_CACHE,
        )
        all_dfs.append(pd.read_parquet(path))

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total: {len(df)}")

    # Filter: good/excellent quality, positive reward, no repeats
    df = df[df["input_quality"].isin(["good", "excellent", "average"])]
    df = df[df["instruct_reward"] > 0]
    df = df[df["repeat_count"] == 0]
    df = df[df["llama_guard_2"] == "safe"]
    print(f"  After quality filter: {len(df)}")

    samples = []
    for _, row in df.iterrows():
        convs = normalize_to_sharegpt(row)
        if convs and is_quality_sample(convs):
            samples.append({
                "id": f"magpie_{row['uuid'][:12]}",
                "conversations": convs,
                "source": "magpie_coder",
                "difficulty": row.get("difficulty", ""),
                "task_category": row.get("task_category", ""),
            })
    print(f"  After format+quality check: {len(samples)}")
    return samples


def load_openhermes_coding():
    """Load OpenHermes-2.5 coding subset."""
    print("Loading OpenHermes-2.5 coding subset...")
    path = os.path.join(
        HF_CACHE,
        "datasets--teknium--OpenHermes-2.5/snapshots/",
    )
    # Find the actual snapshot dir
    snapshot_dir = None
    for d in os.listdir(path):
        full = os.path.join(path, d)
        if os.path.isdir(full):
            snapshot_dir = os.path.join(full)
            break
    if not snapshot_dir:
        print("  OpenHermes not found in cache, skipping")
        return []

    with open(os.path.join(snapshot_dir, "openhermes2_5.json")) as f:
        data = json.load(f)

    # Filter coding-related sources
    coding_sources = {"glaive-code-assist", "EvolInstruct_70k"}
    coding_data = [s for s in data if s.get("source") in coding_sources]
    print(f"  Coding subset: {len(coding_data)}")

    samples = []
    for s in coding_data:
        convs = normalize_to_sharegpt(s)
        if convs and is_quality_sample(convs):
            samples.append({
                "id": f"oh_{s.get('source', 'unknown')}_{hashlib.md5(str(s).encode()).hexdigest()[:8]}",
                "conversations": convs,
                "source": s.get("source", "openhermes"),
            })
    print(f"  After format+quality check: {len(samples)}")
    return samples


def load_code290k():
    """Load Code-290k-ShareGPT."""
    print("Loading Code-290k-ShareGPT...")
    path = os.path.join(
        HF_CACHE,
        "datasets--ajibawa-2023--Code-290k-ShareGPT/snapshots/",
    )
    snapshot_dir = None
    for d in os.listdir(path):
        full = os.path.join(path, d)
        if os.path.isdir(full):
            snapshot_dir = os.path.join(full)
            break
    if not snapshot_dir:
        print("  Code-290k not found in cache, skipping")
        return []

    with open(os.path.join(snapshot_dir, "Code-290k-ShareGPT.json")) as f:
        data = json.load(f)
    print(f"  Total: {len(data)}")

    samples = []
    for s in data:
        convs = normalize_to_sharegpt(s)
        if convs and is_quality_sample(convs):
            samples.append({
                "id": f"code290k_{s.get('id', hashlib.md5(str(s).encode()).hexdigest()[:8])}",
                "conversations": convs,
                "source": "code290k",
            })
    print(f"  After format+quality check: {len(samples)}")
    return samples


def main():
    print("=" * 60)
    print("Building DSpark v10 high-quality training dataset")
    print("=" * 60)

    # Load all sources
    magpie = load_magpie()
    openhermes = load_openhermes_coding()
    code290k = load_code290k()

    print(f"\n=== Pre-dedup counts ===")
    print(f"  Magpie: {len(magpie)}")
    print(f"  OpenHermes coding: {len(openhermes)}")
    print(f"  Code-290k: {len(code290k)}")
    print(f"  Total: {len(magpie) + len(openhermes) + len(code290k)}")

    # Combine all
    all_samples = magpie + openhermes + code290k

    # Deduplicate by first user message hash
    print(f"\n=== Deduplicating ===")
    seen_hashes = set()
    deduped = []
    duplicates = 0
    for s in all_samples:
        first_user = ""
        for c in s["conversations"]:
            if c["from"] == "human":
                first_user = c["value"]
                break
        h = compute_hash(first_user)
        if h in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(h)
        deduped.append(s)
    print(f"  Duplicates removed: {duplicates}")
    print(f"  After dedup: {len(deduped)}")

    # Shuffle
    random.shuffle(deduped)

    # Cap at target size
    if len(deduped) > TARGET_SIZE:
        deduped = deduped[:TARGET_SIZE]
        print(f"  Capped to {TARGET_SIZE}")

    # Source distribution
    source_counts = Counter(s.get("source", "unknown") for s in deduped)
    print(f"\n=== Final source distribution ===")
    for src, count in source_counts.most_common():
        print(f"  {src}: {count} ({100 * count / len(deduped):.1f}%)")

    # Length distribution
    char_lens = [sum(len(c["value"]) for c in s["conversations"]) for s in deduped]
    char_lens.sort()
    n = len(char_lens)
    print(f"\n=== Length distribution (chars) ===")
    print(f"  min={char_lens[0]}  max={char_lens[-1]}")
    print(f"  median={char_lens[n // 2]}  p95={char_lens[int(0.95 * n)]}")

    # Write output
    print(f"\n=== Writing to {OUTPUT_PATH} ===")
    with open(OUTPUT_PATH, "w") as f:
        for s in deduped:
            # Clean output: only id + conversations (compatible with cache gen)
            out = {"id": s["id"], "conversations": s["conversations"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Done! {len(deduped)} samples written to {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
