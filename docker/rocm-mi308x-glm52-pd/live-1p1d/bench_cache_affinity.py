#!/usr/bin/env python3
"""Cache affinity + L3 check for 2P2D.

Never hits /v1/responses.
  A) unique 64K: P0 cold, P0 warm (GPU radix), P1 same blob (Mooncake L3)
  B) new unique 64K via router twice (cache_aware should stick)
  C) unique 200K via router cold + warm (compare to 1P1D 215s / 2.3s)
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

ROUTER = "http://sglang-1p1d-router.kube-system:30001"
API_KEY = "sk-REPLACE_WITH_YOUR_API_KEY"
MODEL = "glm-5.3"
TOK_PATH = "/data/model/glm53-fp8"
OUT = "/data/bench_cache_affinity.json"
SALT = str(int(time.time()))
PREFILLS = {
    "p0": "http://NODE_PREFILL_0_IP:30000",
    "p1": "http://NODE_PREFILL_1_IP:30000",
}

FILLER = (
    "人工智能是计算机科学的一个分支，它致力于研究、开发用于模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。"
    "近年来，随着大数据技术的发展和计算能力的提升，基于深度学习的方法在图像识别、语音识别、自然语言理解等任务上取得了突破性进展。"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def _req(url: str, body: dict, timeout: float, use_key: bool):
    data = json.dumps(body, ensure_ascii=False).encode()
    headers = {"Content-Type": "application/json"}
    if use_key:
        headers["Authorization"] = "Bearer " + API_KEY
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=data, headers=headers, method="POST"
    )
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raw = e.read()[:1200]
        try:
            payload = json.loads(raw)
        except Exception:
            payload = raw.decode("utf-8", "replace")
        return e.code, payload, time.time() - t0, None
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}", time.time() - t0, None

    ttft = None
    usage = {}
    finish = ""
    content = ""
    reasoning = ""
    buf = b""
    while True:
        piece = resp.read(4096)
        if not piece:
            break
        buf += piece
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            s = line.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            payload = s[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ttft is None:
                ttft = time.time() - t0
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
            if ch.get("finish_reason"):
                finish = ch.get("finish_reason")
            if obj.get("usage"):
                usage = obj["usage"]
    dt = time.time() - t0
    return (
        200,
        {
            "content": content,
            "reasoning": reasoning,
            "finish": finish,
            "usage": usage,
        },
        dt,
        ttft,
    )


def chat(url, messages, *, max_tokens, timeout, use_key, tools=None):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": True},
        "thinking": {"type": "enabled"},
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    return _req(url, body, timeout, use_key)


def rec_of(name, url, status, payload, dt, ttft, prompt_est=None):
    if isinstance(payload, dict):
        usage = payload.get("usage") or {}
        content = payload.get("content") or ""
        reasoning = payload.get("reasoning") or ""
        finish = payload.get("finish")
    else:
        usage, content, reasoning, finish = {}, "", "", ""
    prompt = usage.get("prompt_tokens") or prompt_est or 0
    comp = usage.get("completion_tokens") or 0
    cached = None
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
    in_tps = prompt / dt if dt else 0
    rec = {
        "name": name,
        "url": url,
        "status": status,
        "sec": round(dt, 2),
        "ttft_s": None if ttft is None else round(ttft, 2),
        "in_tps": round(in_tps, 1),
        "prompt_tokens": prompt,
        "completion_tokens": comp,
        "cached_tokens": cached,
        "finish": finish,
        "t_unix": time.time(),
        "payload_err": None if isinstance(payload, dict) else str(payload)[:400],
        "content_head": (content or reasoning or "")[:160],
    }
    print(
        f"{name} HTTP {status} wall={dt:.1f}s ttft={rec['ttft_s']} "
        f"in={rec['in_tps']} tok/s prompt={prompt} cached={cached} finish={finish}",
        flush=True,
    )
    return rec


def build_blob(tok, target: int, tag: str) -> str:
    unit = f"[{tag}] mock workspace dump\n" + FILLER + "\n"
    ids = tok.encode(unit, add_special_tokens=False)
    n = max(1, target // max(len(ids), 1))
    blob = unit * n
    ids = tok.encode(blob, add_special_tokens=False)
    if len(ids) > target:
        blob = tok.decode(ids[:target])
    return blob


def main():
    from transformers import AutoTokenizer

    print(f"SALT={SALT} cache affinity bench", flush=True)
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    results = {"salt": SALT, "turns": []}

    st, payload, dt, ttft = chat(
        ROUTER,
        [{"role": "user", "content": "Reply with the single word pong."}],
        max_tokens=16,
        timeout=60,
        use_key=True,
    )
    results["ping"] = rec_of("ping", ROUTER, st, payload, dt, ttft)
    if st != 200:
        json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
        print("PING FAIL", flush=True)
        return

    blob64 = build_blob(tok, 64000, SALT + "-64k")
    n64 = len(tok.encode(blob64, add_special_tokens=False))
    print(f"64k blob tokens={n64}", flush=True)
    sys64 = "You are a coding agent. Workspace dump follows. Think briefly then call read_file."
    task64 = f"Dump tag {SALT}-64k. Call read_file with path src/main.py. One sentence."
    msgs64 = [
        {"role": "system", "content": sys64},
        {"role": "user", "content": blob64 + "\n\n" + task64},
    ]

    print("=== A: GPU radix vs L3 (direct prefills) ===", flush=True)
    t_a0 = time.time()
    st, payload, dt, ttft = chat(
        PREFILLS["p0"], msgs64, max_tokens=128, timeout=400, use_key=False, tools=TOOLS
    )
    results["turns"].append(
        rec_of("a1_p0_cold_64k", PREFILLS["p0"], st, payload, dt, ttft, n64)
        | {"t_start": t_a0}
    )

    t_a1 = time.time()
    st, payload, dt, ttft = chat(
        PREFILLS["p0"], msgs64, max_tokens=128, timeout=400, use_key=False, tools=TOOLS
    )
    results["turns"].append(
        rec_of("a2_p0_warm_radix_64k", PREFILLS["p0"], st, payload, dt, ttft, n64)
        | {"t_start": t_a1}
    )

    t_a2 = time.time()
    st, payload, dt, ttft = chat(
        PREFILLS["p1"], msgs64, max_tokens=128, timeout=400, use_key=False, tools=TOOLS
    )
    results["turns"].append(
        rec_of("a3_p1_cross_l3_64k", PREFILLS["p1"], st, payload, dt, ttft, n64)
        | {"t_start": t_a2}
    )

    blob64b = build_blob(tok, 64000, SALT + "-64k-rtr")
    n64b = len(tok.encode(blob64b, add_special_tokens=False))
    msgs64b = [
        {"role": "system", "content": sys64},
        {
            "role": "user",
            "content": blob64b
            + f"\n\nDump tag {SALT}-64k-rtr. Call read_file path src/main.py.",
        },
    ]
    print("=== B: router cache_aware stickiness ===", flush=True)
    t_b0 = time.time()
    st, payload, dt, ttft = chat(
        ROUTER, msgs64b, max_tokens=128, timeout=400, use_key=True, tools=TOOLS
    )
    results["turns"].append(
        rec_of("b1_router_cold_64k", ROUTER, st, payload, dt, ttft, n64b)
        | {"t_start": t_b0}
    )
    t_b1 = time.time()
    st, payload, dt, ttft = chat(
        ROUTER, msgs64b, max_tokens=128, timeout=400, use_key=True, tools=TOOLS
    )
    results["turns"].append(
        rec_of("b2_router_warm_64k", ROUTER, st, payload, dt, ttft, n64b)
        | {"t_start": t_b1}
    )

    blob200 = build_blob(tok, 196000, SALT + "-200k")
    n200 = len(tok.encode(blob200, add_special_tokens=False))
    print(f"200k blob tokens={n200}", flush=True)
    sys200 = (
        "You are a coding agent in a large repository. Think, then act with tools."
    )
    msgs200 = [
        {"role": "system", "content": sys200},
        {
            "role": "user",
            "content": blob200
            + f"\n\nDump tag {SALT}-200k. Call read_file with path src/main.py. One sentence.",
        },
    ]
    print("=== C: 200K router cold + warm ===", flush=True)
    t_c0 = time.time()
    st, payload, dt, ttft = chat(
        ROUTER, msgs200, max_tokens=256, timeout=900, use_key=True, tools=TOOLS
    )
    results["turns"].append(
        rec_of("c1_router_cold_200k", ROUTER, st, payload, dt, ttft, n200)
        | {"t_start": t_c0}
    )
    t_c1 = time.time()
    st, payload, dt, ttft = chat(
        ROUTER, msgs200, max_tokens=256, timeout=900, use_key=True, tools=TOOLS
    )
    results["turns"].append(
        rec_of("c2_router_warm_200k", ROUTER, st, payload, dt, ttft, n200)
        | {"t_start": t_c1}
    )

    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
