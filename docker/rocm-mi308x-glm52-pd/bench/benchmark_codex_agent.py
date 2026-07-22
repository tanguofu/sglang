#!/usr/bin/env python3
"""Benchmark SGLang PD for Codex-like long agent scenarios.

Scenarios:
  1. Short prompt (baseline, ~50 tokens)
  2. Medium context (~2K tokens) — typical agent turn
  3. Long context (~8K tokens) — agent with code file
  4. Very long context (~16K tokens) — agent with multiple files
  5. Multi-turn conversation (simulating agent loop, growing context)
  6. Streaming TTFT measurement

Measures: TTFT, TPOT, E2E latency, throughput, prompt_tokens, completion_tokens
"""

import requests
import time
import json
import sys
import statistics

URL = "http://127.0.0.1:30001/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}


def gen_code_context(approx_tokens: int) -> str:
    """Generate realistic code context of approximately N tokens."""
    # A realistic Python class ~200 tokens
    block = '''
class DataProcessor:
    """Process data with validation and transformation."""

    def __init__(self, config: dict):
        self.config = config
        self.validators = {}
        self.transformers = []
        self._cache = {}

    def register_validator(self, name: str, func):
        """Register a validation function for a field."""
        self.validators[name] = func

    def register_transformer(self, func):
        """Register a data transformation function."""
        self.transformers.append(func)

    def validate(self, data: dict) -> list:
        """Run all validators, return list of errors."""
        errors = []
        for field, validator in self.validators.items():
            if field in data:
                try:
                    validator(data[field])
                except ValueError as e:
                    errors.append(f"{field}: {e}")
            else:
                errors.append(f"{field}: missing")
        return errors

    def transform(self, data: dict) -> dict:
        """Apply all transformers to data."""
        result = data.copy()
        for transformer in self.transformers:
            result = transformer(result)
        return result

    def process(self, data: dict) -> dict:
        """Full pipeline: validate then transform."""
        errors = self.validate(data)
        if errors:
            return {"success": False, "errors": errors}
        transformed = self.transform(data)
        self._cache[id(data)] = transformed
        return {"success": True, "data": transformed}
'''
    # Each block is ~200 tokens. Repeat to reach approx_tokens.
    num_blocks = max(1, approx_tokens // 200)
    return block * num_blocks


def send_request(payload: dict, stream: bool = False) -> dict:
    """Send request and return timing metrics."""
    payload["stream"] = stream
    start = time.time()
    ttft = None
    first_chunk_time = None

    if stream:
        resp = requests.post(URL, json=payload, headers=HEADERS, stream=True, timeout=300)
        resp.raise_for_status()
        content_parts = []
        for line in resp.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "") or delta.get("reasoning_content", "")
                        if content and first_chunk_time is None:
                            first_chunk_time = time.time()
                            ttft = first_chunk_time - start
                        if content:
                            content_parts.append(content)
                    except json.JSONDecodeError:
                        pass
        elapsed = time.time() - start
        full_content = "".join(content_parts)
        return {
            "ttft": ttft,
            "e2e": elapsed,
            "content_len": len(full_content),
            "content_preview": full_content[:200],
        }
    else:
        resp = requests.post(URL, json=payload, headers=HEADERS, timeout=300)
        resp.raise_for_status()
        elapsed = time.time() - start
        result = resp.json()
        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "") or message.get("reasoning_content", "")
        usage = result.get("usage", {})
        return {
            "ttft": None,
            "e2e": elapsed,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "content_len": len(content),
            "content_preview": content[:200],
            "tps": usage.get("completion_tokens", 0) / elapsed if elapsed > 0 else 0,
        }


def scenario_short():
    """Scenario 1: Short prompt baseline."""
    print("\n=== Scenario 1: Short prompt (~50 tokens) ===")
    payload = {
        "model": "glm-5.2",
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a Python function to reverse a string. Return only the code."},
        ],
        "max_tokens": 200,
        "temperature": 0,
    }
    result = send_request(payload)
    print(f"  Prompt tokens: {result['prompt_tokens']}")
    print(f"  Completion tokens: {result['completion_tokens']}")
    print(f"  E2E: {result['e2e']:.2f}s")
    print(f"  TPS: {result['tps']:.1f} tok/s")
    print(f"  Preview: {result['content_preview'][:100]}")
    return result


def scenario_medium():
    """Scenario 2: Medium context (~2K tokens) — typical agent turn."""
    print("\n=== Scenario 2: Medium context (~2K tokens, agent turn) ===")
    code_ctx = gen_code_context(1800)
    payload = {
        "model": "glm-5.2",
        "messages": [
            {"role": "system", "content": "You are a senior Python engineer. Review the following code and suggest improvements. Be concise."},
            {"role": "user", "content": f"Here is my code:\n\n```python\n{code_ctx}\n```\n\nList 3 improvements I should make."},
        ],
        "max_tokens": 500,
        "temperature": 0,
    }
    result = send_request(payload)
    print(f"  Prompt tokens: {result['prompt_tokens']}")
    print(f"  Completion tokens: {result['completion_tokens']}")
    print(f"  E2E: {result['e2e']:.2f}s")
    print(f"  TPS: {result['tps']:.1f} tok/s")
    print(f"  Preview: {result['content_preview'][:150]}")
    return result


def scenario_long():
    """Scenario 3: Long context (~8K tokens) — agent with code file."""
    print("\n=== Scenario 3: Long context (~8K tokens, code file) ===")
    code_ctx = gen_code_context(7500)
    payload = {
        "model": "glm-5.2",
        "messages": [
            {"role": "system", "content": "You are a code reviewer. Analyze the code for bugs, security issues, and performance problems. Provide a structured report."},
            {"role": "user", "content": f"Review this codebase:\n\n```python\n{code_ctx}\n```\n\nProvide a summary of key findings in 5 bullet points."},
        ],
        "max_tokens": 800,
        "temperature": 0,
    }
    result = send_request(payload)
    print(f"  Prompt tokens: {result['prompt_tokens']}")
    print(f"  Completion tokens: {result['completion_tokens']}")
    print(f"  E2E: {result['e2e']:.2f}s")
    print(f"  TPS: {result['tps']:.1f} tok/s")
    print(f"  Prefill rate: {result['prompt_tokens'] / result['e2e']:.0f} tok/s (approx, includes gen)")
    print(f"  Preview: {result['content_preview'][:150]}")
    return result


def scenario_very_long():
    """Scenario 4: Very long context (~16K tokens) — agent with multiple files."""
    print("\n=== Scenario 4: Very long context (~16K tokens, multiple files) ===")
    code_ctx = gen_code_context(15000)
    payload = {
        "model": "glm-5.2",
        "messages": [
            {"role": "system", "content": "You are an expert software architect. Given a large codebase, identify architectural patterns, dependencies, and suggest refactoring strategies."},
            {"role": "user", "content": f"Analyze this codebase architecture:\n\n{code_ctx}\n\nWhat design patterns are used? Suggest 3 refactoring improvements."},
        ],
        "max_tokens": 1000,
        "temperature": 0,
    }
    result = send_request(payload)
    print(f"  Prompt tokens: {result['prompt_tokens']}")
    print(f"  Completion tokens: {result['completion_tokens']}")
    print(f"  E2E: {result['e2e']:.2f}s")
    print(f"  TPS: {result['tps']:.1f} tok/s")
    print(f"  Prefill rate: {result['prompt_tokens'] / result['e2e']:.0f} tok/s (approx, includes gen)")
    print(f"  Preview: {result['content_preview'][:150]}")
    return result


def scenario_multi_turn():
    """Scenario 5: Multi-turn conversation (agent loop)."""
    print("\n=== Scenario 5: Multi-turn conversation (agent loop, 5 turns) ===")
    messages = [
        {"role": "system", "content": "You are a coding agent. Answer concisely."}
    ]
    turns = [
        "Write a Python function to check if a number is prime.",
        "Now add type hints and a docstring to that function.",
        "Add error handling for negative numbers.",
        "Write a unit test for this function using pytest.",
        "Now optimize it for large numbers using Miller-Rabin.",
    ]

    total_tokens = 0
    total_time = 0
    for i, turn in enumerate(turns):
        messages.append({"role": "user", "content": turn})
        payload = {
            "model": "glm-5.2",
            "messages": messages.copy(),
            "max_tokens": 300,
            "temperature": 0,
        }
        result = send_request(payload)
        assistant_content = result.get("content_preview", "")
        messages.append({"role": "assistant", "content": assistant_content})
        total_tokens += result.get("completion_tokens", 0)
        total_time += result["e2e"]
        print(f"  Turn {i+1}: prompt={result.get('prompt_tokens',0)}, completion={result.get('completion_tokens',0)}, time={result['e2e']:.2f}s")

    print(f"  Total: {total_tokens} tokens, {total_time:.2f}s, {total_tokens/total_time:.1f} tok/s")
    return {"total_tokens": total_tokens, "total_time": total_time, "tps": total_tokens / total_time}


def scenario_streaming_ttft():
    """Scenario 6: Streaming TTFT measurement at different context sizes."""
    print("\n=== Scenario 6: Streaming TTFT measurement ===")
    results = []
    for ctx_size, label in [(50, "short"), (2000, "2K"), (8000, "8K"), (16000, "16K")]:
        if ctx_size > 50:
            code_ctx = gen_code_context(ctx_size - 200)
            user_content = f"Here is code:\n\n{code_ctx}\n\nExplain what it does in 2 sentences."
        else:
            user_content = "Explain what a Python decorator is in 2 sentences."

        payload = {
            "model": "glm-5.2",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 150,
            "temperature": 0,
            "stream": True,
        }
        result = send_request(payload, stream=True)
        ttft = result.get("ttft", 0) or 0
        e2e = result["e2e"]
        print(f"  Context {label} (~{ctx_size} tokens): TTFT={ttft:.2f}s, E2E={e2e:.2f}s")
        results.append({"ctx_size": ctx_size, "label": label, "ttft": ttft, "e2e": e2e})

    return results


def scenario_concurrent_agents():
    """Scenario 7: Concurrent agent requests (multiple agents)."""
    print("\n=== Scenario 7: Concurrent agents (4 parallel, ~4K context each) ===")
    import concurrent.futures

    def agent_request(agent_id):
        code_ctx = gen_code_context(3500)
        payload = {
            "model": "glm-5.2",
            "messages": [
                {"role": "system", "content": f"You are agent {agent_id}. Review code concisely."},
                {"role": "user", "content": f"Review this code:\n{code_ctx}\n\nList 2 issues."},
            ],
            "max_tokens": 300,
            "temperature": 0,
        }
        return send_request(payload)

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(agent_request, range(4)))
    elapsed = time.time() - start

    total_completion = sum(r.get("completion_tokens", 0) for r in results)
    print(f"  4 agents completed in {elapsed:.2f}s")
    print(f"  Total completion tokens: {total_completion}")
    print(f"  Aggregate throughput: {total_completion / elapsed:.1f} tok/s")
    for i, r in enumerate(results):
        print(f"    Agent {i}: {r.get('completion_tokens',0)} tokens, {r['e2e']:.2f}s")
    return {"elapsed": elapsed, "total_completion": total_completion, "tps": total_completion / elapsed}


def main():
    print("=" * 70)
    print("SGLang PD Benchmark — Codex-like Long Agent Scenarios")
    print(f"URL: {URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = {}

    # Run all scenarios
    all_results["1_short"] = scenario_short()
    all_results["2_medium"] = scenario_medium()
    all_results["3_long"] = scenario_long()
    all_results["4_very_long"] = scenario_very_long()
    all_results["5_multi_turn"] = scenario_multi_turn()
    all_results["6_streaming_ttft"] = scenario_streaming_ttft()
    all_results["7_concurrent"] = scenario_concurrent_agents()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Scenario':<30} {'Prompt tok':>10} {'Comp tok':>10} {'E2E (s)':>10} {'TPS':>8}")
    print("-" * 70)

    for key, r in all_results.items():
        if key == "5_multi_turn":
            print(f"{'5_multi_turn (5 turns)':<30} {'~growing':>10} {r['total_tokens']:>10} {r['total_time']:>10.2f} {r['tps']:>8.1f}")
        elif key == "6_streaming_ttft":
            for item in r:
                print(f"  6_stream_{item['label']:<23} {item['ctx_size']:>10} {'~150':>10} {item['e2e']:>10.2f} {'TTFT:'+str(round(item['ttft'],2)):>8}")
        elif key == "7_concurrent":
            print(f"{'7_concurrent (4 agents)':<30} {'~4K each':>10} {r['total_completion']:>10} {r['elapsed']:>10.2f} {r['tps']:>8.1f}")
        else:
            pt = r.get("prompt_tokens", 0)
            ct = r.get("completion_tokens", 0)
            e2e = r.get("e2e", 0)
            tps = r.get("tps", 0)
            label = key.replace("_", " ")
            print(f"{label:<30} {pt:>10} {ct:>10} {e2e:>10.2f} {tps:>8.1f}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
