#!/usr/bin/env python3
"""Official GLM-5.2 dataset eval on the live 1P1D router.

Datasets (zai-org/glm-simple-evals-dataset):
  - AIME 2025 full 30
  - GPQA-Diamond 16 (seeded subset of 198)

Protocol from the GLM-5.2 model card:
  temperature=1.0, top_p=0.95, thinking ON, reasoning_effort default Max
  AIME system prompt: Explanation / Exact Answer / Confidence

Published GLM-5.2 scores (not the same year/N for AIME):
  AIME 2026 99.2 (avg over repeats, max gen 163840)
  GPQA-Diamond 91.2 (n_repeats=8)
This run is pass@1, max_tokens=163840, in-cluster router timeout 7000s.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from typing import Any

ROUTER = os.environ.get("ROUTER", "http://sglang-1p1d-router.kube-system:30001")
MODEL = "glm-5.2"
DATA = os.environ.get("EVAL_DATA", "/data/official_eval")
OUT = os.environ.get("EVAL_OUT", "/data/official_eval_163k.json")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "163840"))
TIMEOUT = float(os.environ.get("TIMEOUT", "7000"))
GPQA_N = int(os.environ.get("GPQA_N", "16"))

AIME_SYS = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your final answer}\n"
    "Exact Answer: {your succinct, final answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}."
)
GPQA_SYS = (
    "You are answering a multiple-choice graduate-level science question. "
    "Think carefully. End with Exact Answer: {A|B|C|D} only."
)

PUBLISHED = {
    "aime_2026": 99.2,
    "gpqa_diamond": 91.2,
    "note": "AIME 2026 is not in the public zai eval dump; this run uses AIME 2025.",
}


def _post(body: dict, timeout: float) -> tuple[int, Any, float]:
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(
        ROUTER + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            dt = time.time() - t0
            try:
                return resp.status, json.loads(raw), dt
            except json.JSONDecodeError:
                return resp.status, raw[:800].decode("utf-8", "replace"), dt
    except urllib.error.HTTPError as e:
        dt = time.time() - t0
        raw = e.read()[:2000]
        try:
            return e.code, json.loads(raw), dt
        except Exception:
            return e.code, raw.decode("utf-8", "replace"), dt
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}", time.time() - t0


def _chat(messages, *, max_tokens: int, timeout: float):
    return _post(
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking": {"type": "enabled"},
        },
        timeout,
    )


def _msg(payload):
    if not isinstance(payload, dict):
        return "", "", "", {}, ""
    ch = (payload.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    finish = ch.get("finish_reason") or ""
    usage = payload.get("usage") or {}
    return content, reasoning, finish, usage, str(payload.get("error") or "")


def extract_aime(content: str, reasoning: str) -> str:
    text = "\n".join(x for x in (content, reasoning) if x)
    m = re.findall(r"Exact Answer:\s*([^\n]+)", text, flags=re.I)
    if m:
        cand = m[-1]
        n = re.search(r"-?\d+", cand)
        if n:
            return str(int(n.group()))
    boxes = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxes:
        n = re.search(r"-?\d+", boxes[-1])
        if n:
            return str(int(n.group()))
    ints = re.findall(r"-?\d+", content or "")
    if ints:
        return str(int(ints[-1]))
    return ""


def extract_letter(content: str, reasoning: str) -> str:
    text = "\n".join(x for x in (content, reasoning) if x)
    m = re.findall(r"Exact Answer:\s*([A-Da-d])\b", text)
    if m:
        return m[-1].upper()
    m = re.findall(r"\\boxed\{([A-Da-d])\}", text)
    if m:
        return m[-1].upper()
    m = re.findall(r"(?:answer|choice|option)\s*(?:is|:)\s*([A-Da-d])\b", text, flags=re.I)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([A-D])\b", content or "")
    if m:
        return m[-1].upper()
    return ""


def load_aime() -> list[dict]:
    path = os.path.join(DATA, "aime_2025.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    tasks = []
    for i, r in enumerate(rows, 1):
        q = r.get("Question") or r.get("question")
        a = str(r.get("Answer") or r.get("answer")).strip()
        n = re.search(r"-?\d+", a)
        tasks.append(
            {
                "id": f"aime25_{i:02d}",
                "suite": "aime2025",
                "sys": AIME_SYS,
                "prompt": q,
                "gold": str(int(n.group())),
            }
        )
    return tasks


def load_gpqa(n: int) -> list[dict]:
    path = os.path.join(DATA, "gpqa_diamond.csv")
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(20250819)
    picked = rng.sample(rows, n)
    tasks = []
    for i, r in enumerate(picked, 1):
        opts = [
            r["Correct Answer"],
            r["Incorrect Answer 1"],
            r["Incorrect Answer 2"],
            r["Incorrect Answer 3"],
        ]
        rng.shuffle(opts)
        labels = ["A", "B", "C", "D"]
        gold = labels[opts.index(r["Correct Answer"])]
        lines = "\n".join(f"{lab}. {opt}" for lab, opt in zip(labels, opts))
        prompt = (
            f"{r['Question']}\n\n{lines}\n\n"
            "Reply with the letter of the correct choice."
        )
        tasks.append(
            {
                "id": f"gpqa_{i:02d}",
                "suite": "gpqa_diamond",
                "sys": GPQA_SYS,
                "prompt": prompt,
                "gold": gold,
                "gold_text": r["Correct Answer"],
                "domain": r.get("High-level domain") or "",
            }
        )
    return tasks


def dump(state: dict) -> None:
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT)


def summarize(recs: list[dict], suite: str) -> dict:
    xs = [r for r in recs if r.get("suite") == suite]
    ok = [r for r in xs if r.get("ok")]
    trunc = [r for r in xs if r.get("finish") == "length"]
    err = [r for r in xs if r.get("status") != 200]
    return {
        "n": len(xs),
        "correct": len(ok),
        "acc": round(100.0 * len(ok) / len(xs), 1) if xs else 0.0,
        "truncated": len(trunc),
        "http_err": len(err),
        "mean_s": round(sum(r.get("sec") or 0 for r in xs) / len(xs), 1) if xs else 0.0,
    }


def run_one(task: dict) -> dict:
    st, payload, dt = _chat(
        [
            {"role": "system", "content": task["sys"]},
            {"role": "user", "content": task["prompt"]},
        ],
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT,
    )
    content, reasoning, finish, usage, err = _msg(payload)
    if task["suite"] == "aime2025":
        pred = extract_aime(content, reasoning)
        ok = pred == task["gold"]
    else:
        pred = extract_letter(content, reasoning)
        ok = pred == task["gold"]
    rec = {
        "id": task["id"],
        "suite": task["suite"],
        "gold": task["gold"],
        "pred": pred,
        "ok": ok,
        "status": st,
        "sec": round(dt, 1),
        "finish": finish,
        "usage": usage,
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "content_tail": (content or "")[-400:],
        "reason_head": (reasoning or "")[:240],
        "payload_err": err or (None if isinstance(payload, dict) else str(payload)[:300]),
    }
    if "gold_text" in task:
        rec["gold_text"] = task["gold_text"]
        rec["domain"] = task.get("domain")
    print(
        f"{task['id']} HTTP {st} {dt:.1f}s finish={finish} "
        f"pred={pred!s:8} gold={task['gold']!s:8} ok={ok} "
        f"rtok={usage.get('reasoning_tokens')}",
        flush=True,
    )
    return rec


def main() -> None:
    print(
        f"ROUTER={ROUTER} MAX_TOKENS={MAX_TOKENS} TIMEOUT={TIMEOUT} GPQA_N={GPQA_N}",
        flush=True,
    )
    tasks = load_aime() + load_gpqa(GPQA_N)
    state = {
        "stamp": int(time.time()),
        "protocol": {
            "temperature": 1.0,
            "top_p": 0.95,
            "thinking": True,
            "max_tokens": MAX_TOKENS,
            "router": ROUTER,
        },
        "published_glm52": PUBLISHED,
        "results": [],
    }
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT))
            done = {r["id"] for r in prev.get("results") or [] if r.get("status") == 200}
            if done:
                print(f"resume skip {sorted(done)}", flush=True)
                state["results"] = [r for r in prev["results"] if r.get("id") in done]
                tasks = [t for t in tasks if t["id"] not in done]
        except Exception as e:
            print("resume ignored", e, flush=True)

    dump(state)
    for t in tasks:
        rec = run_one(t)
        state["results"].append(rec)
        state["aime2025"] = summarize(state["results"], "aime2025")
        state["gpqa_diamond"] = summarize(state["results"], "gpqa_diamond")
        dump(state)

    state["aime2025"] = summarize(state["results"], "aime2025")
    state["gpqa_diamond"] = summarize(state["results"], "gpqa_diamond")
    dump(state)
    print("AIME2025", state["aime2025"], "published AIME2026", PUBLISHED["aime_2026"], flush=True)
    print("GPQA16", state["gpqa_diamond"], "published GPQA-D", PUBLISHED["gpqa_diamond"], flush=True)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
