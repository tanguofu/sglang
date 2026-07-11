#!/usr/bin/env python3
"""Build DSpark v10 training dataset v2 — real human code first.

Prioritizes real human-written code over model-generated synthetic data.

Sources (by priority):
  REAL HUMAN CODE:
  1. CodeSearchNet Python (418K, real code + docstrings from GitHub)
  2. commitpackft Python+JS (109K, real GitHub commit diffs)
  3. Code-290k-ShareGPT (289K, real user coding conversations)
  4. databricks-dolly-15k (15K, human-written instructions)

  HIGH-QUALITY SYNTHETIC (supplement for diversity):
  5. Magpie-Qwen2.5-Coder-Pro-300K (quality-filtered subset only)

Output: /data/dspark_v10_train.jsonl (ShareGPT format, ~300K samples)
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
TARGET_SIZE = 300000


def compute_hash(text):
    return hashlib.md5(text[:500].encode("utf-8")).hexdigest()


def is_quality_sample(conversations, min_assistant_len=20, max_total_chars=12000):
    if len(conversations) < 2:
        return False
    has_human = any(c["from"] == "human" for c in conversations)
    has_gpt = any(c["from"] == "gpt" for c in conversations)
    if not (has_human and has_gpt):
        return False
    assistant_texts = [c["value"] for c in conversations if c["from"] == "gpt"]
    if not assistant_texts:
        return False
    if max(len(t) for t in assistant_texts) < min_assistant_len:
        return False
    total_chars = sum(len(c["value"]) for c in conversations)
    if total_chars > max_total_chars or total_chars < 30:
        return False
    for c in conversations:
        if len(c["value"].strip()) == 0:
            return False
    return True


def load_codesearchnet():
    """Load CodeSearchNet — real code with docstrings from GitHub."""
    print("Loading CodeSearchNet Python (real GitHub code)...")
    path = None
    base = os.path.join(HF_CACHE, "datasets--Nan-Do--instructional_code-search-net-python/snapshots")
    for d in os.listdir(base):
        full = os.path.join(base, d)
        if os.path.isdir(full):
            for f in os.listdir(full):
                if f.startswith("data/"):
                    p = os.path.join(full, f)
                    if os.path.exists(p):
                        path = p
                        break
    if not path:
        # Try direct structure
        for d in os.listdir(base):
            full = os.path.join(base, d, "data")
            if os.path.isdir(full):
                for f in os.listdir(full):
                    if f.endswith(".parquet"):
                        path = os.path.join(full, f)
                        break

    if not path:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="Nan-Do/instructional_code-search-net-python",
            filename="data/train-00000-of-00001-a9aba9cddbadc4a1.parquet",
            repo_type="dataset",
            cache_dir=HF_CACHE,
        )

    df = pd.read_parquet(path)
    print(f"  Total: {len(df)}")

    samples = []
    for _, row in df.iterrows():
        instruction = str(row.get("INSTRUCTION", ""))
        response = str(row.get("RESPONSE", ""))
        if len(instruction) < 10 or len(response) < 10:
            continue
        convs = [
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": response},
        ]
        if is_quality_sample(convs):
            samples.append({
                "id": f"csn_{hashlib.md5(instruction.encode()).hexdigest()[:8]}",
                "conversations": convs,
                "source": "codesearchnet",
            })
    print(f"  After quality check: {len(samples)}")
    return samples


def load_commitpackft():
    """Load commitpackft — real GitHub commit diffs (Python + JavaScript)."""
    print("Loading commitpackft (real GitHub commits)...")
    from huggingface_hub import hf_hub_download

    samples = []
    for lang in ["python", "javascript"]:
        try:
            path = hf_hub_download(
                repo_id="bigcode/commitpackft",
                filename=f"data/{lang}/data.jsonl",
                repo_type="dataset",
                cache_dir=HF_CACHE,
            )
            with open(path) as f:
                for line in f:
                    d = json.loads(line)
                    old_contents = d.get("old_contents", "")
                    new_contents = d.get("new_contents", "")
                    subject = d.get("subject", "")
                    message = d.get("message", subject)

                    # Format as instruction-response: "Apply this change" → diff
                    instruction = f"Apply the following change to {d.get('new_file', 'code')}:\n\nCommit: {subject}\n{message}"
                    # Create a diff-like response
                    response = f"```{lang}\n# --- Before ---\n{old_contents[:3000]}\n\n# --- After ---\n{new_contents[:3000]}\n```"

                    convs = [
                        {"from": "human", "value": instruction},
                        {"from": "gpt", "value": response},
                    ]
                    if is_quality_sample(convs, min_assistant_len=50):
                        samples.append({
                            "id": f"commit_{lang}_{d.get('commit', '')[:8]}",
                            "conversations": convs,
                            "source": f"commitpackft_{lang}",
                        })
        except Exception as e:
            print(f"  {lang}: Error - {e}")

    print(f"  Total after quality check: {len(samples)}")
    return samples


def load_code290k():
    """Load Code-290k-ShareGPT — real user coding conversations."""
    print("Loading Code-290k-ShareGPT (real user conversations)...")
    base = os.path.join(HF_CACHE, "datasets--ajibawa-2023--Code-290k-ShareGPT/snapshots")
    path = None
    for d in os.listdir(base):
        full = os.path.join(base, d)
        if os.path.isdir(full):
            for f in os.listdir(full):
                if f.endswith(".json"):
                    path = os.path.join(full, f)
                    break

    with open(path) as f:
        data = json.load(f)
    print(f"  Total: {len(data)}")

    samples = []
    for s in data:
        convs = []
        for c in s.get("conversations", []):
            convs.append({"from": c.get("from", "?"), "value": c.get("value", "")})
        if is_quality_sample(convs):
            samples.append({
                "id": f"code290k_{s.get('id', hashlib.md5(str(s).encode()).hexdigest()[:8])}",
                "conversations": convs,
                "source": "code290k",
            })
    print(f"  After quality check: {len(samples)}")
    return samples


def load_dolly():
    """Load databricks-dolly-15k — human-written instructions."""
    print("Loading databricks-dolly-15k (human-written)...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="databricks/databricks-dolly-15k",
        filename="databricks-dolly-15k.jsonl",
        repo_type="dataset",
        cache_dir=HF_CACHE,
    )

    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            instruction = d.get("instruction", "")
            context = d.get("context", "")
            response = d.get("response", "")
            category = d.get("category", "")

            # Combine context with instruction if present
            full_instruction = instruction
            if context:
                full_instruction = f"{instruction}\n\nContext:\n{context}"

            convs = [
                {"from": "human", "value": full_instruction},
                {"from": "gpt", "value": response},
            ]
            if is_quality_sample(convs):
                samples.append({
                    "id": f"dolly_{hashlib.md5(instruction.encode()).hexdigest()[:8]}",
                    "conversations": convs,
                    "source": f"dolly_{category}",
                })
    print(f"  After quality check: {len(samples)}")
    return samples


def load_magpie_filtered():
    """Load Magpie — only high-quality subset as supplement."""
    print("Loading Magpie-Qwen2.5-Coder (high-quality supplement)...")
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
    # Only keep excellent quality + high reward + coding/debugging category
    df = df[df["input_quality"] == "excellent"]
    df = df[df["instruct_reward"] > 5]
    df = df[df["repeat_count"] == 0]
    df = df[df["llama_guard_2"] == "safe"]
    df = df[df["task_category"] == "Coding & Debugging"]
    print(f"  After strict filter (excellent + reward>5 + coding): {len(df)}")

    samples = []
    for _, row in df.iterrows():
        convs = list(row["conversations"])
        if hasattr(convs, "tolist"):
            convs = convs.tolist()
        formatted = []
        for c in convs:
            if isinstance(c, dict) and "from" in c and "value" in c:
                formatted.append({"from": c["from"], "value": c["value"]})
        if is_quality_sample(formatted):
            samples.append({
                "id": f"magpie_{row['uuid'][:12]}",
                "conversations": formatted,
                "source": "magpie_coding",
            })
    print(f"  After format+quality check: {len(samples)}")
    return samples


def main():
    print("=" * 60)
    print("Building DSpark v10 dataset v2 — REAL HUMAN CODE FIRST")
    print("=" * 60)

    # Load all sources
    csn = load_codesearchnet()
    commits = load_commitpackft()
    code290k = load_code290k()
    dolly = load_dolly()
    magpie = load_magpie_filtered()

    print(f"\n=== Pre-dedup counts ===")
    print(f"  CodeSearchNet (real code):     {len(csn)}")
    print(f"  commitpackft (real commits):   {len(commits)}")
    print(f"  Code-290k (real conversations): {len(code290k)}")
    print(f"  Dolly (human-written):          {len(dolly)}")
    print(f"  Magpie (synthetic supplement):  {len(magpie)}")
    total = len(csn) + len(commits) + len(code290k) + len(dolly) + len(magpie)
    print(f"  Total: {total}")

    # Combine — real data first, then synthetic supplement
    all_samples = csn + commits + code290k + dolly + magpie

    # Deduplicate
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

    # Shuffle and cap
    random.shuffle(deduped)
    if len(deduped) > TARGET_SIZE:
        deduped = deduped[:TARGET_SIZE]
        print(f"  Capped to {TARGET_SIZE}")

    # Source distribution
    source_counts = Counter(s.get("source", "unknown") for s in deduped)
    print(f"\n=== Final source distribution ===")
    for src, count in source_counts.most_common():
        print(f"  {src}: {count} ({100 * count / len(deduped):.1f}%)")

    # Real vs synthetic
    real_count = sum(c for s, c in source_counts.items() if not s.startswith("magpie"))
    synth_count = sum(c for s, c in source_counts.items() if s.startswith("magpie"))
    print(f"\n  REAL human data: {real_count} ({100*real_count/len(deduped):.1f}%)")
    print(f"  Synthetic supplement: {synth_count} ({100*synth_count/len(deduped):.1f}%)")

    # Length distribution
    char_lens = [sum(len(c["value"]) for c in s["conversations"]) for s in deduped]
    char_lens.sort()
    n = len(char_lens)
    print(f"\n=== Length distribution (chars) ===")
    print(f"  min={char_lens[0]}  max={char_lens[-1]}  median={char_lens[n//2]}")
    print(f"  p25={char_lens[n//4]}  p75={char_lens[3*n//4]}  p95={char_lens[int(0.95*n)]}")

    # Write output
    print(f"\n=== Writing to {OUTPUT_PATH} ===")
    with open(OUTPUT_PATH, "w") as f:
        for s in deduped:
            out = {"id": s["id"], "conversations": s["conversations"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Done! {len(deduped)} samples written to {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
