#!/usr/bin/env bash
set -euo pipefail

# Optional: override --max-num-seqs (default 256 matches vLLM when omitted)
INFER_DELAY="${PROFILE_TEST_INFER_DELAY:-0.2}"
SMI_WINDOW_SEC="${PROFILE_TEST_SMI_WINDOW_SEC:-2}"
ITERATIONS="${PROFILE_TEST_ITERATIONS:-5}"
GPU_ID="${PROFILE_TEST_GPU_ID:-0}"

echo "=== GPU In-Flight Validation Test ==="
echo "===================================="

cleanup() {
  [[ -n "${RESP_FILE:-}" && -f "$RESP_FILE" ]] && rm -f "$RESP_FILE"
  [[ -n "${SMI_FILE:-}" && -f "$SMI_FILE" ]] && rm -f "$SMI_FILE"
  [[ -n "${PROMPT_FILE:-}" && -f "$PROMPT_FILE" ]] && rm -f "$PROMPT_FILE"
}
trap cleanup EXIT

PROMPT_FILE="$(mktemp)"

cat > "$PROMPT_FILE" << 'PROMPT'
Explain quantum computing concepts in detail for software engineers.
Cover qubits, superposition, entanglement, gates, limitations,
applications, and provide analogies.
PROMPT

for i in {1..10}; do
  echo "Additional context block for longer prefill." >> "$PROMPT_FILE"
done

for ((i=1;i<=ITERATIONS;i++)); do
  echo ""
  echo "=== Iteration $i ==="

  RESP_FILE="$(mktemp)"
  SMI_FILE="$(mktemp)"

  curl -sS -o "$RESP_FILE" http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
      --arg prompt "$(cat "$PROMPT_FILE")" \
      '{
        model: "llama3",
        messages: [{role:"user",content:$prompt}],
        max_tokens: 256,
        temperature: 0
      }')" &
  CURL_PID=$!

  sleep "$INFER_DELAY"

  MAX_SEQ_ARGS=()
  if [[ -n "${PROFILE_MAX_NUM_SEQS:-}" ]]; then
    MAX_SEQ_ARGS=(--max-num-seqs "$PROFILE_MAX_NUM_SEQS")
  fi

  ./profile diagnose "${MAX_SEQ_ARGS[@]}" &
  PROFILE_PID=$!

  timeout "${SMI_WINDOW_SEC}s" nvidia-smi --id="${GPU_ID}" \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm \
    --format=csv,noheader,nounits \
    -lms 250 > "$SMI_FILE" 2>/dev/null || true

  wait "$PROFILE_PID"
  wait "$CURL_PID"

  echo ""
  echo "nvidia-smi (2s window avg):"

  awk -F', ' '
    BEGIN { gpu_sum=mem_sum=pwr_sum=0; count=0 }
    {
      gpu_sum += $2
      mem_sum += $3
      pwr_sum += $6
      used=$4
      total=$5
      temp=$8
      sm=$9
      count++
    }
    END {
      if (count == 0) {
        print "  (no samples collected)"
        exit
      }
      printf "  %-15s : %.1f\n", "GPU util %", gpu_sum / count
      printf "  %-15s : %.1f\n", "Mem ctrl util %", mem_sum / count
      printf "  %-15s : %d / %d MiB (%.1f)\n", "VRAM % used", used, total, used * 100.0 / total
      printf "  Power draw      : %.0f W\n", pwr_sum / count
      printf "  Temp            : %d C\n", temp
      printf "  SM clock        : %d MHz\n", sm
    }
  ' "$SMI_FILE"

  rm -f "$RESP_FILE" "$SMI_FILE"
done

echo ""
echo "Done."