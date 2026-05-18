#!/usr/bin/env bash
# Usage:
#   MODE=r1 ./load.sh          # under-batching: low concurrency, short prompts
#   MODE=r2 ./load.sh          # KV cache pressure: high concurrency, long context
#
# Runs forever. Kill with Ctrl-C.

set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL="${MODEL:-llama3}"
MODE="${MODE:-r1}"

post() {
  local prompt="$1" max_tokens="$2"
  jq -n \
    --arg model "$MODEL" \
    --arg prompt "$prompt" \
    --argjson max_tokens "$max_tokens" \
    '{model:$model, messages:[{role:"user",content:$prompt}],
      max_tokens:$max_tokens, temperature:0, stream:false}' \
  | curl -sS -o /dev/null \
      -H "Content-Type: application/json" \
      -d @- \
      "${VLLM_URL}/v1/chat/completions" || true
}

long_context() {
  python3 -c "
chunk = 'Distributed inference, GPU scheduling, KV cache growth, prefill-decode balance, memory pressure, and latency spikes under multi-tenant workloads. '
print(''.join(f'[{i:03d}] ' + chunk for i in range(1, 40)))
"
}

LONG_CTX="$(long_context)"

load_r1() {
  # 2 concurrent requests, long output, gap between batches.
  # Goal: keep GPU util < 62%, running > 0.75, waiting < 2, TPOT >= 100ms.
  while true; do
    post "Write a detailed 500-word essay on GPU architecture and
          how tensor cores accelerate matrix multiplication." 600 &
    post "Write a detailed 500-word essay on LLM inference
          optimization techniques including batching and KV cache." 600 &
    sleep 2.0
    wait
    sleep 0.5
  done
}

load_r2() {
  # 8 concurrent long-context requests → KV cache fills up
  # Run vLLM with --gpu-memory-utilization 0.55 to make r2 fire faster
  while true; do
    for ((i = 0; i < 8; i++)); do
      post "${LONG_CTX}

Summarise the above. List 10 risks and 10 recommendations." 256 &
    done
    sleep 0.8
    wait
  done
}

echo "load.sh — MODE=${MODE}  target=${VLLM_URL}"
echo "Ctrl-C to stop."
echo ""

case "$MODE" in
  r1) load_r1 ;;
  r2) load_r2 ;;
  *)  echo "Unknown MODE=${MODE}. Use r1 or r2." >&2; exit 1 ;;
esac
