#!/usr/bin/env python3
"""200K/512K unique-needle correctness + GLM-5.2 official-alignment bench.

Run inside the prefill pod against the in-cluster PD router.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

ROUTER = "http://sglang-1p1d-router.kube-system:30001"
MODEL = "glm-5.3"
STAMP = str(int(time.time()))
OUT = "/data/bench_longctx_align.json"
TOK_PATH = "/data/model/glm53-fp8"

FILLER = (
    "人工智能是计算机科学的一个分支，它致力于研究、开发用于模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。"
    "人工智能的研究范畴广泛，包括机器学习、深度学习、自然语言处理、计算机视觉、知识表示、自动推理、机器人学等多个领域。"
    "近年来，随着大数据技术的发展和计算能力的提升，基于深度学习的方法在图像识别、语音识别、自然语言理解等任务上取得了突破性进展。"
)

GARBLED_RE = re.compile(
    r"(1\.1\.2|NEEDLE1\.NEEDLE|</think>\s*</think>|\. \. \. \.)",
    re.I,
)


def _post(path: str, body: dict, timeout: float) -> tuple[int, dict | str, float]:
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(
        ROUTER + path,
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
        raw = e.read()[:1500]
        try:
            return e.code, json.loads(raw), dt
        except Exception:
            return e.code, raw.decode("utf-8", "replace"), dt
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}", time.time() - t0


def _chat(messages, *, max_tokens=256, temperature=0.0, top_p=None, thinking=None, timeout=180.0):
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        body["top_p"] = top_p
    if thinking is False:
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["thinking"] = {"type": "disabled"}
    elif thinking is True:
        body["chat_template_kwargs"] = {"enable_thinking": True}
        body["thinking"] = {"type": "enabled"}
    return _post("/v1/chat/completions", body, timeout)


def _msg_text(payload) -> tuple[str, str]:
    if not isinstance(payload, dict):
        return "", str(payload)[:400]
    ch = (payload.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    return content, reasoning


def _usage(payload) -> dict:
    if not isinstance(payload, dict):
        return {}
    return payload.get("usage") or {}


def load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    return tok


def build_needle_prompt(tok, target_tokens: int, salt: str) -> tuple[str, list[str], int]:
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
        "\n\nTASK: List the five NEEDLE facts exactly, one per line as "
        "NEEDLE1/2/3/4/5. Do not invent extra facts. Do not think out loud.\n"
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


def score_needles(text: str, facts: list[str]) -> dict:
    hits = []
    for fact in facts:
        key = fact.split(":", 1)[1].strip().rstrip(".")
        ok = key in text or fact.split(":", 1)[0] in text and key.split()[-1] in text
        # require the distinctive token (AURORA / dock / cipher / callsign / freeze tag)
        distinctive = key.split()[-1]
        ok = distinctive in text
        hits.append(ok)
    garbled = bool(GARBLED_RE.search(text))
    return {
        "hits": sum(hits),
        "n": len(facts),
        "ok": sum(hits) == len(facts) and not garbled,
        "garbled": garbled,
        "detail": hits,
    }


def run_needle(tok, target: int, timeout: float) -> dict:
    salt = f"{STAMP}-{target}"
    print(f"\n=== NEEDLE {target} cold unique salt={salt} ===", flush=True)
    t_build = time.time()
    prompt, facts, ntok = build_needle_prompt(tok, target, salt)
    print(f"built prompt tokens={ntok} chars={len(prompt)} in {time.time()-t_build:.1f}s", flush=True)
    status, payload, dt = _chat(
        [{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
        thinking=False,
        timeout=timeout,
    )
    content, reasoning = _msg_text(payload)
    text = (content or "") + "\n" + (reasoning or "")
    usage = _usage(payload)
    sc = score_needles(text, facts)
    prompt_tok = usage.get("prompt_tokens") or ntok
    in_tps = prompt_tok / dt if dt > 0 else 0
    rec = {
        "target": target,
        "actual_tokens": ntok,
        "status": status,
        "sec": round(dt, 2),
        "in_tps": round(in_tps, 1),
        "usage": usage,
        "score": sc,
        "reply": (content or "")[:800],
        "reasoning_head": (reasoning or "")[:200],
    }
    print(
        f"HTTP {status} {dt:.1f}s in_tps={in_tps:.1f} facts={sc['hits']}/{sc['n']} "
        f"garbled={sc['garbled']} ok={sc['ok']}",
        flush=True,
    )
    print("reply:", (content or str(payload)[:300])[:400], flush=True)
    return rec


def run_align() -> dict:
    print("\n=== OFFICIAL GLM-5.2 ALIGNMENT ===", flush=True)
    cases = []

    # A. default thinking-on (official default enabled / auto)
    st, payload, dt = _chat(
        [{"role": "user", "content": "What is 17*19? Reply with the number after a short check."}],
        max_tokens=2048,
        temperature=1.0,
        top_p=0.95,
        thinking=True,
        timeout=120,
    )
    content, reasoning = _msg_text(payload)
    has_think = bool(reasoning) or "<think>" in (content or "")
    ans_ok = "323" in (content or "")
    cases.append(
        {
            "name": "think_on_temp1_topp095_math",
            "status": st,
            "sec": round(dt, 2),
            "has_reasoning": has_think,
            "answer_ok": ans_ok,
            "content": (content or "")[:400],
            "reasoning_head": (reasoning or "")[:300],
            "usage": _usage(payload),
        }
    )
    print(f"A think-on math HTTP {st} {dt:.1f}s think={has_think} ans={ans_ok}", flush=True)

    # B. official default sampling, thinking disabled (needle-style)
    st, payload, dt = _chat(
        [{"role": "user", "content": "Repeat exactly this token and nothing else: GLM52-ALIGN-OK"}],
        max_tokens=32,
        temperature=0.0,
        thinking=False,
        timeout=60,
    )
    content, reasoning = _msg_text(payload)
    cases.append(
        {
            "name": "think_off_exact_repeat",
            "status": st,
            "sec": round(dt, 2),
            "ok": "GLM52-ALIGN-OK" in (content or ""),
            "has_reasoning": bool(reasoning),
            "content": (content or "")[:200],
            "usage": _usage(payload),
        }
    )
    print(f"B think-off repeat HTTP {st} {dt:.1f}s ok={cases[-1]['ok']}", flush=True)

    # C. Chinese instruction following, official sampling
    st, payload, dt = _chat(
        [{"role": "user", "content": "用三个汉字回答：中国的首都是哪里？只要三个字。"}],
        max_tokens=64,
        temperature=0.0,
        thinking=False,
        timeout=60,
    )
    content, _ = _msg_text(payload)
    cases.append(
        {
            "name": "zh_capital_think_off",
            "status": st,
            "sec": round(dt, 2),
            "ok": "北京" in (content or ""),
            "content": (content or "")[:120],
            "usage": _usage(payload),
        }
    )
    print(f"C zh capital HTTP {st} {dt:.1f}s ok={cases[-1]['ok']}", flush=True)

    # D. default request omitting temperature (model generation_config 1.0/0.95)
    st, payload, dt = _post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say the word ready."}],
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        60,
    )
    content, _ = _msg_text(payload)
    cases.append(
        {
            "name": "omit_temp_think_off",
            "status": st,
            "sec": round(dt, 2),
            "ok": "ready" in (content or "").lower(),
            "content": (content or "")[:120],
            "usage": _usage(payload),
        }
    )
    print(f"D omit-temp HTTP {st} {dt:.1f}s ok={cases[-1]['ok']}", flush=True)

    # E. tool-call parser glm47 shape
    st, payload, dt = _post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Call get_time for timezone Asia/Shanghai. Do not explain.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "get current time",
                        "parameters": {
                            "type": "object",
                            "properties": {"tz": {"type": "string"}},
                            "required": ["tz"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "max_tokens": 256,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        90,
    )
    tool_ok = False
    if isinstance(payload, dict):
        msg = (payload.get("choices") or [{}])[0].get("message") or {}
        tcs = msg.get("tool_calls") or []
        tool_ok = any(
            (tc.get("function") or {}).get("name") == "get_time" for tc in tcs
        )
    cases.append(
        {
            "name": "tool_call_glm47",
            "status": st,
            "sec": round(dt, 2),
            "ok": tool_ok,
            "payload_head": str(payload)[:500],
            "usage": _usage(payload) if isinstance(payload, dict) else {},
        }
    )
    print(f"E tool-call HTTP {st} {dt:.1f}s ok={tool_ok}", flush=True)
    return {"cases": cases}


def main():
    print(f"STAMP={STAMP} ROUTER={ROUTER}", flush=True)
    st, payload, dt = _chat(
        [{"role": "user", "content": "ping"}],
        max_tokens=8,
        temperature=0.0,
        thinking=False,
        timeout=30,
    )
    print("ping", st, dt, _msg_text(payload)[0][:80], flush=True)
    if st != 200:
        raise SystemExit("router ping failed, abort")

    tok = load_tokenizer()
    print("tokenizer loaded", type(tok).__name__, flush=True)

    results: dict[str, Any] = {
        "stamp": STAMP,
        "router": ROUTER,
        "official_defaults": {
            "temperature": 1.0,
            "top_p": 0.95,
            "thinking": "enabled/auto",
            "max_tokens_doc": 65536,
            "context_doc": "1M",
        },
        "align": run_align(),
        "needles": [],
    }

    # 8K sanity then 200K then 512K
    results["needles"].append(run_needle(tok, 8000, timeout=180))
    results["needles"].append(run_needle(tok, 200000, timeout=900))
    results["needles"].append(run_needle(tok, 512000, timeout=1800))

    with open(OUT, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("WROTE", OUT, flush=True)
    print(json.dumps({k: results[k] if k != "needles" else [
        {kk: vv for kk, vv in n.items() if kk != "reply"}
        for n in results["needles"]
    ] for k in results if k != "align"}, ensure_ascii=False)[:2000], flush=True)


if __name__ == "__main__":
    main()
