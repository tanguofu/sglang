#!/usr/bin/env python3
"""Run codex with --json and capture event timestamps."""
import subprocess, time, json, sys

start = time.perf_counter()
proc = subprocess.Popen(
    ["/opt/homebrew/bin/codex", "exec",
     "--dangerously-bypass-approvals-and-sandbox", "--json", "-"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
)
proc.stdin.write("Reply with exactly: PONG\n")
proc.stdin.close()

events = []
for line in proc.stdout:
    now = time.perf_counter() - start
    line = line.rstrip()
    if not line:
        continue
    try:
        evt = json.loads(line)
    except json.JSONDecodeError:
        continue
    t = evt.get("type", "unknown")
    events.append((now, t, evt))
    # Stop after we get the final message
    if t in ("task_complete", "turn_complete", "agent_message"):
        # Print last few
        pass

proc.wait(timeout=90)
total = time.perf_counter() - start

print(f"Total: {total:.2f}s, events: {len(events)}")
print(f"\nEvent timeline (first 30):")
for ts, t, evt in events[:30]:
    msg = ""
    if "message" in evt:
        msg = str(evt["message"])[:50]
    elif "content" in evt:
        msg = str(evt["content"])[:50]
    print(f"  {ts:6.2f}s  {t:<30}  {msg}")

print(f"\nEvent type counts:")
from collections import Counter
counts = Counter(t for _, t, _ in events)
for t, c in counts.most_common():
    print(f"  {t:<35} {c}")

print(f"\nLast 5 events:")
for ts, t, evt in events[-5:]:
    print(f"  {ts:6.2f}s  {t}")
