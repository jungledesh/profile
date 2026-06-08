#!/usr/bin/env bash
# Usage:
#   MODE=demo ./load.sh        # coding assistant: shared system prompt, varied code questions
#   MODE=r1   ./load.sh        # under-batching: low concurrency, continuous, no gaps
#   MODE=seq  ./load.sh        # under-batching (extreme): 1 sequential request at a time
#   MODE=r2   ./load.sh        # KV cache pressure: high concurrency, long context
#   MODE=r5   ./load.sh        # concurrency saturation: more requests than max_num_seqs
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

load_demo() {
  # Coding assistant scenario: all requests share a long system prompt (prefix),
  # each user asks a different code question. Realistic mix — 70% shared prefix
  # tokens, 30% unique. Concurrency tuned to fill max_num_seqs=32 and spill into
  # the queue so R5 fires after R3 is addressed.
  local concurrency="${CONCURRENCY:-50}"
  local max_tokens="${MAX_TOKENS:-300}"

  # Long shared system prompt — same across every request (prefix cache target)
  local system_prompt="You are an expert software engineer and coding assistant. \
You write clean, efficient, well-documented code. You follow best practices for \
the language in use, prefer clarity over cleverness, and always explain your \
reasoning. When asked to review code, you identify bugs, suggest improvements, \
and explain the trade-offs. You are familiar with Python, Rust, Go, TypeScript, \
and C++. You respond concisely unless asked for detail."

  # Varied code questions — different each request to avoid trivial cache hits
  # on the question itself, while the system prompt stays shared
  local questions=(
    "Write a Python function that flattens a nested list of arbitrary depth."
    "Review this Rust code for memory safety issues: fn get(v: &Vec<i32>, i: usize) -> i32 { v[i] }"
    "Implement a rate limiter in Go using a token bucket algorithm."
    "What is the difference between a mutex and a semaphore? Give a code example in C++."
    "Write a TypeScript utility type that makes all nested properties optional."
    "Explain why this Python code has a bug: def append(x, lst=[]): lst.append(x); return lst"
    "Write an async Rust function that retries an HTTP request up to 3 times with exponential backoff."
    "How do I implement a trie in Python? Show insert and search."
    "What does 'move semantics' mean in Rust? Give a before/after example."
    "Write a Go function that finds all duplicate elements in a slice."
    "Implement a debounce function in TypeScript."
    "Explain the difference between heap and stack allocation with C++ examples."
    "Write a Python decorator that measures and logs function execution time."
    "How do I avoid data races in Rust when sharing state across threads?"
    "Write a binary search implementation in Go with proper error handling."
  )
  local n=${#questions[@]}

  for ((i = 0; i < concurrency; i++)); do
    (
      local idx=0
      while true; do
        local q="${questions[$((idx % n))]}"
        idx=$((idx + 1))
        local combined="${system_prompt}

User: ${q}"
        post "$combined" "$max_tokens"
      done
    ) &
  done
  wait
}

echo "load.sh — MODE=${MODE}  target=${VLLM_URL}"
echo "Ctrl-C to stop."
echo ""

case "$MODE" in
  demo) load_demo ;;
  r1)   load_r1 ;;
  seq)  load_seq ;;
  r2)   load_r2 ;;
  r5)   load_r5 ;;
  *)    echo "Unknown MODE=${MODE}. Use demo, r1, seq, r2, or r5." >&2; exit 1 ;;
esac
