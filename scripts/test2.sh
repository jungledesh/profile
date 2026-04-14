#!/usr/bin/env bash
set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL="${MODEL:-llama3}"
PROFILE_BIN="${PROFILE_BIN:-./profile}"
TEST_SEC="${TEST_SEC:-180}"

uuid_or_fallback() {
  if command -v uuidgen >/dev/null 2>&1; then
    uuidgen
  else
    cat /proc/sys/kernel/random/uuid 2>/dev/null || echo "req-$(date +%s)-$RANDOM"
  fi
}

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

make_long_context() {
  python3 - <<'PY'
chunk = (
    "This is a long production-style context about distributed inference, "
    "GPU scheduling, queueing, prefix reuse, KV cache growth, prefill-decode balance, "
    "memory pressure, and latency spikes under multi-tenant workloads. "
)
print("".join([f"[CTX {i:03d}] " + chunk for i in range(1, 36)]))
PY
}

LONG_CONTEXT="$(make_long_context)"

send_kv_pressure_load() {
  local end=$((SECONDS + TEST_SEC))
  while (( SECONDS < end )); do
    for ((i=0; i<8; i++)); do
      local rid
      rid="$(uuid_or_fallback)"
      prompt="REQUEST=${rid}
${LONG_CONTEXT}

Using the full context above:
1. Write a detailed summary
2. List 12 risks
3. List 12 recommendations
4. Explain tradeoffs in depth"
      post_chat "$prompt" 256 &
    done
    sleep 0.8
  done
  wait
}

echo "=== Rule 2: KV Cache Pressure Test ==="
echo "Starting long-context load to push KV usage..."
echo "IMPORTANT: Start vLLM with --gpu-memory-utilization 0.55 or lower"

send_kv_pressure_load &
LOAD_PID=$!

sleep 25

echo ""
echo "=== Running 30s diagnose ==="
"${PROFILE_BIN}" diagnose --url "${VLLM_URL}/metrics" --duration 30s

echo ""
echo "=== Running 1m diagnose ==="
"${PROFILE_BIN}" diagnose --url "${VLLM_URL}/metrics" --duration 1m

wait "${LOAD_PID}" 2>/dev/null || true

echo ""
echo "=== Rule 2 test completed ==="