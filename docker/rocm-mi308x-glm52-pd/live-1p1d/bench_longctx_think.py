#!/usr/bin/env python3
"""200K/512K unique-needle with official GLM-5.2 thinking ON.

max_tokens=8192 so think can finish. Score facts in content+reasoning.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

ROUTER = "http://sglang-1p1d-router.kube-system:30001"
MODEL = "glm-5.2"
STAMP = str(int(time.time()))
OUT = "/data/bench_longctx_think.json"
TOK_PATH = "/data/model/glm52-fp8"

FILLER = (
    "人工智能是计算机科学的一个分支，它致力于研究、开发用于模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。"
    "人工智能的研究范畴广泛，包括机器学习、深度学习、自然语言处理、计算机视觉、知识表示、自动推理、机器人学等多个领域。"
    "近年来，随着大数据技术的发展和计算能力的提升，基于深度学习的方法在图像识别、语音识别、自然语言理解等任务上取得了突破性进展。"
)

GARBLED_RE = re.compile(
    r"(1\.1\.2|NEEDLE1\.NEEDLE|</think>\s*</think>|(\bNEEDLE1\b.*){8,})",
    re.I | re.S,
)


def _post(body: dict, timeout: float) -> tuple[int, dict | str, float]:
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
                return resp.status, raw[:500].decode("utf-8", "replace"), dt
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
        return "", "", "", {}
    ch = (payload.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return (
        msg.get("content") or "",
        msg.get("reasoning_content") or "",
        ch.get("finish_reason") or "",
        payload.get("usage") or {},
    )


def build_prompt(tok, target_tokens: int, salt: str):
    facts = [
        f"NEEDLE1: The project code name is AURORA-7X-{salt}.",
        f"NEEDLE2: The warehouse dock number is 418-{salt[-4:]}.",
        f"NEEDLE3: The backup cipher is ZULU-{salt[:6]}-OK.",
        f"NEEDLE4: The duty officer callsign is KESTREL-{salt[-5:]}.",
        f"NEEDLE5: The freeze date is 2099-12-31 tagged {salt}.",
    ]
    unit = f"[{salt}] " + FILLER
    unit_ids = tok.encode(unit, add_special_tokens=False)
    q = (
        "\n\nTASK: After thinking, list the five NEEDLE facts exactly, "
        "one per line as NEEDLE1/2/3/4/5. Do not invent extra facts.\n"
    )
    overhead = len(tok.encode("\n".join(facts) + q, add_special_tokens=False)) + 64
    n_rep = max(8, (target_tokens - overhead) // max(len(unit_ids), 1))
    filler = unit * n_rep
    chunks = []
    step = max(len(filler) // 6, 1)
    pos = [step, 2 * step, 3 * step, 4 * step, 5 * step]
    cursor = 0
    for i, fact in enumerate(facts):
        p = pos[i]
        chunks.append(filler[cursor:p])
        chunks.append("\n<<<BEGIN FACT>>> " + fact + " <<<END FACT>>>\n")
        cursor = p
    chunks.append(filler[cursor:])
    chunks.append(q)
    prompt = "".join(chunks)
    ntok = len(tok.encode(prompt, add_special_tokens=False))
    return prompt, facts, ntok


def score(text: str, facts: list[str]) -> dict:
    hits = []
    for fact in facts:
        key = fact.split(":", 1)[1].strip().rstrip(".")
        distinctive = key.split()[-1]
        hits.append(distinctive in text)
    garbled = bool(GARBLED_RE.search(text))
    return {
        "hits": sum(hits),
        "n": len(facts),
        "ok": sum(hits) == len(facts) and not garbled,
        "garbled": garbled,
        "detail": hits,
    }


def run_needle(tok, target: int, timeout: float) -> dict:
    salt = f"{STAMP}-t{target}"
    print(f"\n=== NEEDLE {target} THINKING-ON salt={salt} ===", flush=True)
    t_build = time.time()
    prompt, facts, ntok = build_prompt(tok, target, salt)
    print(f"built tokens={ntok} chars={len(prompt)} in {time.time()-t_build:.1f}s", flush=True)
    status, payload, dt = _chat(
        [{"role": "user", "content": prompt}],
        max_tokens=8192,
        timeout=timeout,
    )
    content, reasoning, finish, usage = _msg(payload)
    combined = (content or "") + "\n" + (reasoning or "")
    sc = score(combined, facts)
    prompt_tok = usage.get("prompt_tokens") or ntok
    rec = {
        "target": target,
        "actual_tokens": ntok,
        "status": status,
        "sec": round(dt, 2),
        "in_tps": round(prompt_tok / dt, 1) if dt else 0,
        "finish_reason": finish,
        "usage": usage,
        "has_reasoning": bool(reasoning),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "score": sc,
        "content": (content or "")[:1200],
        "reasoning_head": (reasoning or "")[:800],
        "reasoning_tail": (reasoning or "")[-400:] if reasoning else "",
        "payload_err": payload if not isinstance(payload, dict) else None,
    }
    print(
        f"HTTP {status} {dt:.1f}s in_tps={rec['in_tps']} finish={finish} "
        f"think={bool(reasoning)} rtok={usage.get('reasoning_tokens')} "
        f"facts={sc['hits']}/{sc['n']} garbled={sc['garbled']} ok={sc['ok']}",
        flush=True,
    )
    print("content:", (content or str(payload)[:300])[:500], flush=True)
    print("reason_head:", (reasoning or "")[:300], flush=True)
    return rec


def main():
    print(f"STAMP={STAMP} THINKING ON temp=1.0 top_p=0.95 max_tokens=8192", flush=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    status, payload, dt = _chat(
        [{"role": "user", "content": "What is 9*9? Answer with the number after a short check."}],
        max_tokens=1024,
        timeout=60,
    )
    content, reasoning, finish, usage = _msg(payload)
    print("ping-think", status, dt, "think", bool(reasoning), "ans", "81" in content, finish, flush=True)
    if status != 200:
        raise SystemExit("router ping failed")

    results = {
        "stamp": STAMP,
        "thinking": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
        "needles": [],
    }
    results["needles"].append(run_needle(tok, 200000, timeout=900))
    results["needles"].append(run_needle(tok, 512000, timeout=1800))
    with open(OUT, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
