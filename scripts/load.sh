#!/usr/bin/env bash
# Usage:
#   MODE=r1  ./load.sh         # under-batching: low concurrency, continuous, no gaps
#   MODE=seq ./load.sh         # under-batching (extreme): 1 sequential request at a time
#   MODE=r2  ./load.sh         # KV cache pressure: high concurrency, long context
#   MODE=r5  ./load.sh         # concurrency saturation: more requests than max_num_seqs
#
# Tuning env vars:
#   CONCURRENCY=N              # number of concurrent requests (default: 2 for r1/seq, 8 for r2, 3× max_num_seqs for r5)
#   MAX_TOKENS=N               # max output tokens per request (default: 600 for r1/seq, 256 for r2/r5)
#   CONTEXT_CHUNKS=N           # number of context chunks for r2 long prompt (default: 40)
#
# r2 prerequisite: lower --gpu-memory-utilization in vLLM start script to
# constrain KV cache space (e.g. 0.50–0.65). The right value depends on your
# GPU VRAM and model size — start low and raise until r2 fires.
#
# r5 prerequisite: start vLLM with a low --max-num-seqs (e.g. 16 or 32) so
# the slot cap is easy to hit. Keep prompts short so KV stays healthy — you
# want the scheduler cap to be the bottleneck, not memory.
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

new_uuid() {
  if command -v uuidgen >/dev/null 2>&1; then
    uuidgen
  elif [[ -r /proc/sys/kernel/random/uuid ]]; then
    cat /proc/sys/kernel/random/uuid
  else
    echo "req-$(date +%s)-$RANDOM"
  fi
}

long_context() {
  local chunks="${CONTEXT_CHUNKS:-40}"
  python3 -c "
chunk = 'Distributed inference, GPU scheduling, KV cache growth, prefill-decode balance, memory pressure, and latency spikes under multi-tenant workloads. '
print(''.join(f'[{i:03d}] ' + chunk for i in range(1, int('${chunks}') + 1)))
"
}

LONG_CTX="$(long_context)"

load_r1() {
  # Continuous low-concurrency load — no sleep gaps.
  # CONCURRENCY controls batch size. Keep small to starve the GPU each decode
  # step without burst/idle cycles that skew NVML averages.
  local concurrency="${CONCURRENCY:-2}"
  local max_tokens="${MAX_TOKENS:-600}"
  local prompt="Write a detailed essay on GPU architecture and how tensor cores accelerate matrix multiplication."
  while true; do
    for i in $(seq 1 "$concurrency"); do
      post "$prompt" "$max_tokens" &
    done
    wait
  done
}

load_seq() {
  # 1 sequential request at a time — maximum under-batching.
  # Each decode step processes exactly 1 token, exposing raw weight-load cost.
  local max_tokens="${MAX_TOKENS:-600}"
  local prompt="Write a detailed essay on GPU architecture and how tensor cores accelerate matrix multiplication."
  while true; do
    post "$prompt" "$max_tokens"
  done
}

load_r2() {
  # High concurrency, long context, unique prompts — no sleep gaps.
  # Unique IDs bust prefix cache so KV blocks are not reused across requests.
  # See r2 prerequisite note at the top of this file.
  local concurrency="${CONCURRENCY:-8}"
  local max_tokens="${MAX_TOKENS:-256}"
  while true; do
    for ((i = 0; i < concurrency; i++)); do
      local uid
      uid="$(new_uuid)"
      post "[REQ-${uid}]
${LONG_CTX}

Summarise the above. List 10 risks and 10 recommendations." "$max_tokens" &
    done
    wait
  done
}

load_r5() {
  # Keeps exactly CONCURRENCY requests in flight at all times.
  # No batch gaps — as each request completes, a new one fires immediately.
  # This saturates max_num_seqs and builds a persistent wait queue.
  # Keep prompts short so KV stays healthy (r2 should not fire).
  local concurrency="${CONCURRENCY:-512}"
  local max_tokens="${MAX_TOKENS:-4096}"
  local prompt="Explain in detail what a transformer model is, how attention works, and why it replaced RNNs for sequence modeling tasks."
  for ((i = 0; i < concurrency; i++)); do
    while true; do
      post "$prompt" "$max_tokens"
    done &
  done
  wait
}

echo "load.sh — MODE=${MODE}  target=${VLLM_URL}"
echo "Ctrl-C to stop."
echo ""

case "$MODE" in
  r1)  load_r1 ;;
  seq) load_seq ;;
  r2)  load_r2 ;;
  r5)  load_r5 ;;
  *)   echo "Unknown MODE=${MODE}. Use r1, seq, r2, or r5." >&2; exit 1 ;;
esac
