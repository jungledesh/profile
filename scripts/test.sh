#!/usr/bin/env bash
set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL="${MODEL:-llama3}"
PROFILE_BIN="${PROFILE_BIN:-./profile}"
TEST_SEC="${TEST_SEC:-75}"

post_chat() {
  local prompt="$1"
  local max_tokens="$2"

  jq -n \
    --arg model "$MODEL" \
    --arg prompt "$prompt" \
    --argjson max_tokens "$max_tokens" \
    '{
      model: $model,
      messages: [{role:"user", content:$prompt}],
      max_tokens: $max_tokens,
      temperature: 0,
      stream: false
    }' \
  | curl -sS -o /dev/null \
      -H "Content-Type: application/json" \
      -d @- \
      "${VLLM_URL}/v1/chat/completions" || true
}

send_under_batching_load() {
  local end=$((SECONDS + TEST_SEC))

  while (( SECONDS < end )); do
    # Keep a few overlapping requests alive (but far below capacity)
    post_chat "Explain RAM in 3 short bullet points for a beginner." 80 &
    post_chat "Explain CPU vs GPU in 4 short bullet points." 80 &
    post_chat "What is a database index? Answer in 5 short bullets." 80 &
    post_chat "What is caching? Give a short practical explanation." 80 &

    sleep 0.35
    wait

    # Small gap → keeps queue low + GPU underutilized
    sleep 0.45
  done
}

echo "=== Rule 1: Under-batching ==="
echo "Starting controlled low-occupancy load for ${TEST_SEC}s..."

send_under_batching_load &
LOAD_PID=$!

sleep 10
"${PROFILE_BIN}" diagnose --url "${VLLM_URL}/metrics" --duration 30s

wait "${LOAD_PID}" 2>/dev/null || true

echo ""
echo "Expected:"
echo "- running > 0.75"
echo "- occupancy << max_num_seqs (very low %)"
echo "- GPU util < 62%"
echo "- waiting < 2"
echo "- Rule 1 fires"