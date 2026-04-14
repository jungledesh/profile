#!/usr/bin/env bash
set -euo pipefail

echo "=== v0.1 Duration-Aware Profiler Test (Realistic Load) ==="
echo "Testing default + boundary cases with varied traffic"
echo "================================================================="

VLLM_URL="http://localhost:8000"
TEST_DURATION_SEC=420
LOAD_PID=""

cleanup() {
  if [[ -n "${LOAD_PID}" ]]; then
    kill "${LOAD_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

send_mixed_load() {
  local end=$((SECONDS + TEST_DURATION_SEC))
  while ((SECONDS < end)); do
    local conc=$((6 + RANDOM % 11))
    for ((i = 1; i <= conc; i++)); do
      case $((RANDOM % 4)) in
        0 | 1) PROMPT="Explain quantum computing in one paragraph." ;;
        2) PROMPT="Explain quantum computing concepts in detail for software engineers..." ;;
        3) PROMPT="Long context test with repeated blocks. $(printf 'Block %03d. ' {1..25})" ;;
      esac

      jq -n --arg p "$PROMPT" '{
        model: "llama3",
        messages: [{role:"user", content: $p}],
        max_tokens: 256,
        temperature: 0
      }' | curl -s -o /dev/null -H "Content-Type: application/json" -d @- \
        "${VLLM_URL}/v1/chat/completions" || true &
    done
    wait
    sleep "$(awk -v r=10 'BEGIN {x = 1/r - 0.05 + (rand()-0.5)*0.08; if (x < 0.02) x=0.02; print x}')"
  done
}

send_mixed_load &
LOAD_PID=$!

echo "Starting mixed load for ${TEST_DURATION_SEC}s..."

for dur in "2s" "30s" "32s" "1m" "5m"; do
  echo ""
  echo "=== Testing --duration ${dur} ==="
  ./profile diagnose --url "${VLLM_URL}/metrics" --duration "${dur}"
  sleep 10
done

echo ""
echo "=== Testing --duration 5m with -v ==="
./profile -v diagnose --url "${VLLM_URL}/metrics" --duration 5m

wait "${LOAD_PID}" 2>/dev/null || true

echo ""
echo "=== Duration test completed ==="
echo "Check output for correct windowing, % fired, aggregation, and boundary behavior."
