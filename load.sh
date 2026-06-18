#!/usr/bin/env bash
# load.sh — RAG-mode load generator for vLLM / Qwen
# Profile will analyze the traffic this produces.
#
# Usage:
#   chmod +x load.sh
#   ./load.sh
#   ./load.sh --url http://localhost:8000 --concurrency 16 --requests 200

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
URL="${VLLM_URL:-http://localhost:8000}"
MODEL="${VLLM_MODEL:-Qwen3.6-27B}"   # matches --served-model-name in scripts/demo.sh
CONCURRENCY="${CONCURRENCY:-8}"
TOTAL_REQUESTS="${TOTAL_REQUESTS:-100}"

# RAG profile: long prompt (~1500 tokens), short answer (~150 tokens)
MAX_TOKENS=200
TEMPERATURE=0.0   # deterministic — typical for RAG retrieval answers

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --url)         URL="$2";          shift 2 ;;
    --model)       MODEL="$2";        shift 2 ;;
    --concurrency) CONCURRENCY="$2";  shift 2 ;;
    --requests)    TOTAL_REQUESTS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

ENDPOINT="${URL}/v1/chat/completions"

# ── RAG prompt (simulates retrieved context + user question) ──────────────────
# ~1500 tokens of context to mimic real RAG retrieval chunks.
read -r -d '' SYSTEM_PROMPT << 'EOF'
You are a helpful assistant. Answer the user's question using only the provided context. Be concise and factual. Do not add information not present in the context.

Context:
[Document 1 — Infrastructure overview]
The inference cluster runs on 8× H100 SXM5 nodes. Each node has 80 GB HBM3 VRAM and 3.35 TB/s memory bandwidth. Nodes are interconnected via NVLink 4.0 at 900 GB/s bidirectional. The cluster is managed by Kubernetes with custom scheduling for GPU affinity. Tensor parallelism is set to 4 for models over 30B parameters. Pipeline parallelism is disabled due to latency overhead at current batch sizes. KV cache utilization target is set to 85% to leave headroom for preemption recovery. The vLLM serving layer uses continuous batching with chunked prefill enabled. Prefill chunk size is 512 tokens. Max sequence length is 8192 tokens. GPU memory utilization is set to 0.90. Swap space is 4 GB per GPU for KV cache overflow.

[Document 2 — Operational runbook]
When p99 TTFT exceeds 2000ms, the on-call engineer should first check KV cache utilization via the Prometheus dashboard. If KV cache is above 90%, reduce max_num_seqs from 256 to 128 and monitor for 5 minutes. If KV cache is below 70%, the bottleneck is likely prefill compute — check GPU utilization and consider increasing chunked prefill chunk size to 1024. When decode throughput drops below 500 tok/s on H100, check for preemptions in the vllm_num_preemptions_total metric. Preemptions above 10/min indicate KV pressure. The recommended remediation is to enable prefix caching if not already enabled, then tune max_num_seqs downward until preemptions drop to 0. Do not change tensor parallelism during a live incident — it requires a server restart.

[Document 3 — Model configuration]
Qwen2.5-72B-Instruct is deployed with dtype bfloat16. Weight size is approximately 144 GB, requiring TP=2 minimum on H100 80GB nodes. With TP=4, KV cache headroom is approximately 160 GB across 4 GPUs. KV cache dtype is auto (inherits bfloat16). Prefix caching is enabled with a prefix cache size of 512 MB. The model context window is 128K tokens but vLLM is configured to 8192 for cost efficiency. Rope scaling is disabled. The tokenizer is the standard Qwen2.5 tokenizer with 151936 vocabulary size.
EOF

read -r -d '' USER_QUESTION << 'EOF'
What should I do when KV cache utilization is above 90% and p99 TTFT is high?
EOF

# ── Single request function ───────────────────────────────────────────────────
send_request() {
  local req_id=$1
  local payload
  payload=$(jq -cn \
    --arg model "$MODEL" \
    --arg system "$SYSTEM_PROMPT" \
    --arg user "$USER_QUESTION" \
    --argjson max_tokens "$MAX_TOKENS" \
    --argjson temperature "$TEMPERATURE" \
    '{
      model: $model,
      messages: [
        {role: "system", content: $system},
        {role: "user", content: $user}
      ],
      max_tokens: $max_tokens,
      temperature: $temperature,
      stream: false
    }')

  local start_ms
  start_ms=$(date +%s%3N)

  local http_code
  http_code=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$ENDPOINT" \
    -H "Content-Type: application/json" \
    -d "$payload" \
    --max-time 120)

  local end_ms
  end_ms=$(date +%s%3N)
  local elapsed=$((end_ms - start_ms))

  echo "req=${req_id} status=${http_code} elapsed=${elapsed}ms"
}

export -f send_request
export ENDPOINT MODEL SYSTEM_PROMPT USER_QUESTION MAX_TOKENS TEMPERATURE

# ── Run ───────────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  vLLM RAG load test"
echo "  endpoint    : ${ENDPOINT}"
echo "  model       : ${MODEL}"
echo "  concurrency : ${CONCURRENCY}"
echo "  requests    : ${TOTAL_REQUESTS}"
echo "  prompt      : ~1500 tokens (RAG context)"
echo "  max_tokens  : ${MAX_TOKENS}"
echo "  temperature : ${TEMPERATURE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Verify endpoint is up before hammering it
if ! curl -sf "${URL}/health" > /dev/null 2>&1; then
  echo "[!] ${URL}/health not reachable — is vLLM running?"
  exit 1
fi

echo "[+] Server healthy. Sending ${TOTAL_REQUESTS} requests at concurrency ${CONCURRENCY}..."
echo ""

START=$(date +%s%3N)

seq 1 "$TOTAL_REQUESTS" | \
  xargs -P "$CONCURRENCY" -I{} bash -c 'send_request "$@"' _ {}

END=$(date +%s%3N)
TOTAL_MS=$((END - START))
TOTAL_S=$(echo "scale=1; $TOTAL_MS / 1000" | bc)
RPS=$(echo "scale=1; $TOTAL_REQUESTS / ($TOTAL_MS / 1000)" | bc)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done"
echo "  total time  : ${TOTAL_S}s"
echo "  throughput  : ${RPS} req/s"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
