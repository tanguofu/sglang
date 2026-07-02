#!/usr/bin/env python3
"""HLE (Humanity's Last Exam) precision evaluation for GLM-5.2 on AMD MI355X.

Dataset: cais/hle (2500 questions, multiple-choice + short-answer).
Target: 40.5% (GLM-5.2 official).

Standard HLE methodology: greedy decoding (temperature=0), thinking mode,
exact-match scoring for both MC and short-answer questions.

Usage (inside container):
  python3 /data/eval_hle.py --base-url http://localhost:30000/v1 \
    --model /data/models/GLM-5.2-FP8 --num-questions 100
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI

MC_TEMPLATE = (
    "Answer the following multiple-choice question. "
    "The last line of your response should be of the form "
    "'Answer: $LETTER' where $LETTER is one of A, B, C, D, E.\n\n"
    "{question}\n\n"
    "Choices:\n{choices}\n\n"
    "Think step by step, then put your answer on its own line after 'Answer:'."
).strip()

SA_TEMPLATE = (
    "Answer the following question. "
    "The last line of your response should be of the form "
    "'Answer: $ANSWER' where $ANSWER is your concise answer.\n\n"
    "{question}\n\n"
    "Think step by step, then put your answer on its own line after 'Answer:'."
).strip()

ANSWER_PATTERN = r"Answer:\s*(.+?)(?:\n|$)"


def normalize_text(text: str) -> str:
    """Normalize text for short-answer comparison."""
    text = text.strip().lower()
    # Remove common punctuation and articles
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(the|a|an)\b", "", text)
    text = " ".join(text.split())
    return text


def extract_answer(text: str) -> str | None:
    """Extract the last 'Answer: XXX' from the model response."""
    matches = re.findall(ANSWER_PATTERN, text)
    if matches:
        return matches[-1].strip()
    return None


def load_hle(num_questions: int = 0) -> list[dict]:
    """Load HLE dataset from HuggingFace (cais/hle)."""
    ds = load_dataset("cais/hle", split="test")
    examples = []
    for row in ds:
        # Skip image questions (text-only model)
        image = row.get("image", None)
        if image is not None:
            continue
        ex = {
            "question": row["question"],
            "answer": str(row.get("answer", "")).strip(),
            "category": row.get("category", ""),
            "answer_type": row.get("answer_type", ""),
        }
        # Build choices list for MC questions
        choices = []
        for key in ("choices", "options"):
            if key in row and row[key]:
                choices = list(row[key])
                break
        if not choices:
            # Try A/B/C/D fields
            for letter in "ABCDEFGHIJ":
                val = row.get(f"choice_{letter.lower()}", row.get(letter, None))
                if val:
                    choices.append(str(val))
        ex["choices"] = choices
        examples.append(ex)

    if num_questions > 0:
        examples = examples[:num_questions]
    return examples


def build_prompt(example: dict) -> tuple[str, bool]:
    """Build the prompt for MC or short-answer. Returns (prompt, is_mc)."""
    if example["choices"]:
        choices_text = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(example["choices"])
        )
        return MC_TEMPLATE.format(
            question=example["question"], choices=choices_text
        ), True
    return SA_TEMPLATE.format(question=example["question"]), False


def score_response(
    extracted: str | None, correct: str, is_mc: bool, choices: list
) -> bool:
    """Score a single response."""
    if extracted is None:
        return False
    extracted = extracted.strip()

    if is_mc and choices:
        # Extract letter (A, B, C, D...)
        letter_match = re.match(r"^([A-Ea-e])\b", extracted)
        if letter_match:
            idx = ord(letter_match.group(1).upper()) - 65
            if 0 <= idx < len(choices):
                return normalize_text(choices[idx]) == normalize_text(correct)
        # Fallback: compare text
        return normalize_text(extracted) == normalize_text(correct)

    # Short answer: normalized exact match
    return normalize_text(extracted) == normalize_text(correct)


def eval_question(
    client: OpenAI,
    model: str,
    example: dict,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
    idx: int,
    total: int,
) -> dict:
    """Evaluate a single HLE question."""
    prompt, is_mc = build_prompt(example)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            timeout=timeout,
        )
        content = resp.choices[0].message.content or ""
        extracted = extract_answer(content)
        correct = score_response(extracted, example["answer"], is_mc, example["choices"])
    except Exception as e:
        content = ""
        extracted = None
        correct = False
        error = str(e)
    else:
        error = None

    status = "✓" if correct else "✗"
    qtype = "MC" if is_mc else "SA"
    print(
        f"[{idx + 1}/{total}] {qtype} {status} "
        f"answer={example['answer'][:30]} extracted={(extracted or 'None')[:30]} "
        f"[{example.get('category', '')[:20]}]",
        flush=True,
    )

    return {
        "question_idx": idx,
        "category": example.get("category", ""),
        "answer_type": "MC" if is_mc else "SA",
        "correct": correct,
        "extracted": extracted,
        "expected": example["answer"],
        "response_len": len(content),
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser(description="HLE precision eval for GLM-5.2")
    parser.add_argument("--base-url", default="http://localhost:30000/v1")
    parser.add_argument("--model", default="/data/models/GLM-5.2-FP8")
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--out-dir", default="/data/eval_results")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    print("Loading HLE dataset...", flush=True)
    examples = load_hle(args.num_questions)
    total = len(examples)
    print(f"Loaded {total} text questions.", flush=True)

    all_results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                eval_question, client, args.model, ex, args.max_tokens,
                args.temperature, args.top_p, args.timeout, i, total,
            ): i
            for i, ex in enumerate(examples)
        }
        for f in as_completed(futures):
            all_results.append(f.result())

    elapsed = time.time() - start

    # Aggregate
    n_correct = sum(r["correct"] for r in all_results)
    accuracy = n_correct / len(all_results) if all_results else 0

    # Per-category breakdown
    by_type = {}
    for r in all_results:
        t = r["answer_type"]
        if t not in by_type:
            by_type[t] = {"correct": 0, "total": 0}
        by_type[t]["total"] += 1
        if r["correct"]:
            by_type[t]["correct"] += 1

    print("\n" + "=" * 60)
    print(f"HLE Results ({total} questions)")
    print(f"  Overall accuracy: {accuracy:.1%}  (target: 40.5%)")
    for t, s in sorted(by_type.items()):
        print(f"  {t}: {s['correct']}/{s['total']} = {s['correct']/s['total']:.1%}")
    print(f"  Elapsed: {elapsed / 60:.1f} min")
    print("=" * 60)

    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"hle_results_{ts}.json"
    summary = {
        "benchmark": "HLE",
        "n_questions": total,
        "accuracy": accuracy,
        "by_type": {t: {"correct": s["correct"], "total": s["total"]} for t, s in by_type.items()},
        "elapsed_sec": elapsed,
        "config": {
            "model": args.model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
        "per_question": all_results,
    }
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results saved to {out_file}", flush=True)

    if accuracy >= 0.375:
        print("✅ PASS: accuracy within tolerance of 40.5% target")
        sys.exit(0)
    else:
        print("❌ FAIL: accuracy below 37.5% (3% tolerance from 40.5% target)")
        sys.exit(1)


if __name__ == "__main__":
    main()
