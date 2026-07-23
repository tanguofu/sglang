#!/usr/bin/env python3
"""Run codex with debug and extract request/response details."""
import subprocess, time, re

start = time.perf_counter()
proc = subprocess.run(
    ["/opt/homebrew/bin/codex", "exec",
     "--dangerously-bypass-approvals-and-sandbox", "-"],
    input="Reply with exactly: PONG",
    capture_output=True, text=True, timeout=120,
    env={"RUST_LOG": "debug", "PATH": "/opt/homebrew/bin:/usr/bin:/bin", "HOME": "/Users/guofutan"},
)
total = time.perf_counter() - start
print(f"Total: {total:.2f}s, rc={proc.returncode}")

# Extract key info from stderr (debug logs)
stderr = proc.stderr
print(f"\nstderr lines: {len(stderr.splitlines())}")
print(f"stdout lines: {len(proc.stdout.splitlines())}")

# Look for reasoning effort, request size, tools count
patterns = [
    r"reasoning_effort[=:]\s*(\w+)",
    r"reasoning_summary[=:]\s*(\w+)",
    r"tools[=:]\s*\[(.*?)\]",
    r"tool_count[=:]\s*(\d+)",
    r"input_tokens[=:]\s*(\d+)",
    r"output_tokens[=:]\s*(\d+)",
    r"reasoning_tokens[=:]\s*(\d+)",
    r"total_tokens[=:]\s*(\d+)",
    r"prompt_tokens[=:]\s*(\d+)",
    r"max_output_tokens[=:]\s*(\d+)",
    r"model_reasoning_effort[=:]\s*(\w+)",
    r"effort[=:]\s*(\w+)",
]
print("\n=== Extracted fields ===")
for pat in patterns:
    matches = re.findall(pat, stderr)
    if matches:
        uniq = list(set(matches))
        print(f"  {pat[:40]:<42} {uniq[:3]}")

# Find token usage near the end
print("\n=== Last 20 stderr lines with token/usage ===")
for line in stderr.splitlines()[-200:]:
    if any(k in line.lower() for k in ["token", "usage", "effort", "reasoning_summary"]):
        print(f"  {line[-180:]}")
