#!/bin/bash
# Final end-to-end verification of claude and codex with current (512K) config.
# Tests cover: basic, arithmetic, knowledge, code, reasoning — all via streaming.

PASS=0
FAIL=0
RESULTS=""

log() {
  local name="$1"; local status="$2"; local note="$3"
  RESULTS="${RESULTS}${name} | ${status} | ${note}\n"
  [ "$status" = "PASS" ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
}

run_claude() {
  local prompt="$1"; local name="$2"; local expect="${3:-}"
  echo ""
  echo "=== CLAUDE: $name ==="
  local start=$(/bin/date +%s)
  local out
  out=$(echo "$prompt" | /opt/homebrew/bin/claude --print 2>&1)
  local rc=$?
  local end=$(/bin/date +%s)
  local elapsed=$((end-start))
  echo "  elapsed: ${elapsed}s, rc=$rc"
  echo "  output: $(echo "$out" | /usr/bin/tail -3)"
  if [ $rc -ne 0 ]; then
    log "$name" "FAIL" "rc=$rc elapsed=${elapsed}s"
    return
  fi
  if [ -n "$expect" ] && ! echo "$out" | /usr/bin/grep -qiE "$expect"; then
    log "$name" "FAIL" "expected /$expect/ not found, elapsed=${elapsed}s"
    return
  fi
  log "$name" "PASS" "elapsed=${elapsed}s"
}

run_codex() {
  local prompt="$1"; local name="$2"; local expect="${3:-}"
  echo ""
  echo "=== CODEX: $name ==="
  local start=$(/bin/date +%s)
  local out
  out=$(echo "$prompt" | /opt/homebrew/bin/codex exec --dangerously-bypass-approvals-and-sandbox - 2>&1 | /usr/bin/grep -vE "^(hook:|warning:|tokens used|model:|provider:|approval:|sandbox:|reasoning|session|workdir:|---|OpenAI Codex)" | /usr/bin/tail -15)
  local rc=${PIPESTATUS[0]}
  local end=$(/bin/date +%s)
  local elapsed=$((end-start))
  echo "  elapsed: ${elapsed}s, rc=$rc"
  echo "  output: $(echo "$out" | /usr/bin/tail -3)"
  if [ $rc -ne 0 ]; then
    log "$name" "FAIL" "rc=$rc elapsed=${elapsed}s"
    return
  fi
  if [ -n "$expect" ] && ! echo "$out" | /usr/bin/grep -qiE "$expect"; then
    log "$name" "FAIL" "expected /$expect/ not found, elapsed=${elapsed}s"
    return
  fi
  log "$name" "PASS" "elapsed=${elapsed}s"
}

echo "=========================================="
echo "Final E2E Verification — claude + codex"
echo "Config: context=524288, auto_compact=420000"
echo "=========================================="

# Claude (Messages API streaming)
run_claude "Reply with exactly: PONG" "C1 basic" "PONG"
run_claude "What is 7+5? Reply with just the number." "C2 arithmetic" "12"
run_claude "In one short sentence, what is the capital of France?" "C3 knowledge" "Paris"
run_claude "Write a Python one-liner: print hello world. Just the code." "C4 code" "print"
run_claude "If I have 3 apples and give 1 away, how many remain? Just the number." "C5 reasoning" "2"

# Codex (Responses API streaming, full plugins/MCP/skills enabled)
run_codex "Reply with exactly: PONG" "X1 basic" "PONG"
run_codex "What is 7+5? Reply with just the number." "X2 arithmetic" "12"
run_codex "In one short sentence, what is the capital of France?" "X3 knowledge" "Paris"
run_codex "Write a Python one-liner: print hello world. Just the code." "X4 code" "print"
run_codex "If I have 3 apples and give 1 away, how many remain? Just the number." "X5 reasoning" "2"

# Summary
echo ""
echo "=========================================="
echo "Summary: PASS=$PASS FAIL=$FAIL (total=$((PASS+FAIL)))"
echo "=========================================="
echo ""
echo "Test | status | note"
echo "---------------------"
printf "$RESULTS"
