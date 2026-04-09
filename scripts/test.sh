#!/usr/bin/env bash
set -euo pipefail

# ================================================
# H100 SXM (and similar): continuous inference + comparison run
#
# - Continuous load: mixed prompt sizes, concurrent waves, steady pacing
# - Default 300s (5 min) for PROFILE vs nvidia-smi comparison
# - Each cycle: ./profile diagnose and nvidia-smi in parallel; pipe only (no temp files)
#
# All knobs are constants below — no environment variables required.
# ================================================

# --- tune here only ---
INFER_DURATION_SEC=300
REQUEST_RATE=5
CONCURRENCY=4
GPU_ID=0
# Keep in sync with Profile NVML: 9 samples × 250 ms ≈ 2 s (src/collectors/gpu.rs).
SMI_WINDOW_SEC=2
VLLM_OPENAI_BASE="http://localhost:8000"
VLLM_METRICS_URL="${VLLM_OPENAI_BASE}/metrics"
# --- end tune ---

if ((INFER_DURATION_SEC < 300)); then
  echo "WARNING: INFER_DURATION_SEC is ${INFER_DURATION_SEC}; use at least 300 for stable H100 comparison (edit scripts/test.sh)." >&2
fi

echo "=== Continuous Real-World Inference Test (H100 / SXM-class) ==="
echo "Duration     : ${INFER_DURATION_SEC} seconds (default 5 min for comparison)"
echo "Rate (approx): ${REQUEST_RATE} req/s"
echo "Concurrency  : ${CONCURRENCY} (per wave)"
echo "vLLM         : ${VLLM_OPENAI_BASE}"
echo "Sampling     : profile diagnose || nvidia-smi in parallel (no temp files)"
echo "==============================================================="

PROMPT_SHORT="Explain quantum computing in one paragraph."
PROMPT_MEDIUM="Explain quantum computing concepts in detail for software engineers. Cover qubits, superposition, entanglement, gates, limitations, applications, and provide analogies."
PROMPT_LONG="$PROMPT_MEDIUM
Additional context block for longer prefill. $(printf 'Extra long context line %03d.\n' {1..15})"

send_requests() {
  local end_time=$((SECONDS + INFER_DURATION_SEC))
  while ((SECONDS < end_time)); do
    for ((i = 1; i <= CONCURRENCY; i++)); do
      case $((RANDOM % 3)) in
        0) PROMPT="$PROMPT_SHORT" ;;
        1) PROMPT="$PROMPT_MEDIUM" ;;
        2) PROMPT="$PROMPT_LONG" ;;
      esac
      jq -n \
        --arg prompt "$PROMPT" \
        '{
          model: "llama3",
          messages: [{role:"user",content:$prompt}],
          max_tokens: 256,
          temperature: 0
        }' | curl -sS -o /dev/null -H "Content-Type: application/json" -d @- \
        "${VLLM_OPENAI_BASE}/v1/chat/completions" || true &
      sleep 0.05
    done
    wait
    local sleep_sec
    sleep_sec="$(awk -v r="$REQUEST_RATE" 'BEGIN {
      if (r <= 0) { print 0.2; exit }
      x = 1 / r - 0.05
      if (x < 0.01) x = 0.01
      print x
    }')"
    sleep "$sleep_sec"
  done
}

echo "Starting continuous load for ${INFER_DURATION_SEC} seconds..."
send_requests &
SENDER_PID=$!

END_TIME=$((SECONDS + INFER_DURATION_SEC))
while ((SECONDS < END_TIME)); do
  echo ""
  echo "=== Profile + nvidia-smi @ $(date '+%H:%M:%S') (parallel) ==="

  ./profile diagnose -u "$VLLM_METRICS_URL" &
  PROFILE_PID=$!

  # nvidia-smi runs in parallel (process sub); wait only in this shell — not inside a pipeline subshell.
  exec 3< <(
    timeout "${SMI_WINDOW_SEC}s" nvidia-smi --id="${GPU_ID}" \
      --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm \
      --format=csv,noheader,nounits \
      -lms 250 2>/dev/null || true
  )

  wait "$PROFILE_PID"

  echo "nvidia-smi (${SMI_WINDOW_SEC}s window, same cycle as PROFILE above):"
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
      printf "  %-15s : %.1f%%\n", "GPU util", gpu_sum / count
      printf "  %-15s : %.1f%%\n", "Mem ctrl util", mem_sum / count
      printf "  %-15s : %d / %d MiB (%.1f%%)\n", "VRAM used", used, total, used * 100.0 / total
      printf "  Power draw      : %.0f W\n", pwr_sum / count
      printf "  Temp            : %d C\n", temp
      printf "  SM clock        : %d MHz\n", sm
    }
  ' <&3
  exec 3<&-

  sleep 6
done

wait "$SENDER_PID"

echo ""
echo "Test finished. Continuous load + comparison cycles completed."
