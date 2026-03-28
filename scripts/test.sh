#!/usr/bin/env bash
set -euo pipefail

# Requires: bash, curl, jq (1.5+ for --rawfile), awk, mktemp, nvidia-smi, tail, cat
for _need in curl jq awk nvidia-smi; do
  command -v "$_need" >/dev/null || {
    echo "error: missing required command: $_need" >&2
    exit 1
  }
done

# Test configuration (override via env vars)
INFER_DELAY="${PROFILE_TEST_INFER_DELAY:-0.2}"      # seconds to wait after starting request
SMI_WINDOW_SEC="${PROFILE_TEST_SMI_WINDOW_SEC:-2}"  # sampling window
ITERATIONS="${PROFILE_TEST_ITERATIONS:-5}"
GPU_ID="${PROFILE_TEST_GPU_ID:-0}"

SMI_SAMPLES=$((SMI_WINDOW_SEC * 4)) # 250ms cadence → 4 samples/sec

echo "=== GPU In-Flight Validation Test ==="
echo "Config: INFER_DELAY=${INFER_DELAY}s | SMI_WINDOW=${SMI_WINDOW_SEC}s | Iterations=${ITERATIONS} | GPU=${GPU_ID}"
echo "Goal: Compare ./profile diagnose (2s avg) vs nvidia-smi during active inference"
echo "========================================"

RESP_FILE=""
SMI_FILE=""
PROMPT_FILE=""

cleanup() {
  [[ -n "${RESP_FILE:-}" && -f "$RESP_FILE" ]] && rm -f "$RESP_FILE"
  [[ -n "${SMI_FILE:-}" && -f "$SMI_FILE" ]] && rm -f "$SMI_FILE"
  [[ -n "${PROMPT_FILE:-}" && -f "$PROMPT_FILE" ]] && rm -f "$PROMPT_FILE"
}
trap cleanup EXIT

# Create long prompt (forces substantial prefill)
PROMPT_FILE="$(mktemp "${TMPDIR:-/tmp}/profile-test-prompt.XXXXXX")"
cat > "$PROMPT_FILE" <<'EOF'
You are writing an internal technical explainer for software engineers who are new to quantum computing.

Write a detailed, well-structured note with sections covering:
1. What a qubit is
2. Superposition
3. Entanglement
4. Measurement
5. Quantum gates vs classical logic gates
6. Why quantum computers may speed up certain classes of problems
7. Why they do not speed up all problems
8. Current limitations of noisy quantum hardware
9. Error correction and fault tolerance
10. Applications in chemistry, optimization, simulation, and cryptography
11. Practical reasons most developers should still think classically today
12. A short conclusion

For each major concept, include: a plain-English explanation, a concrete analogy, and one misconception to avoid.
Use a serious but clear engineering tone.
EOF

# Append more context to lengthen prompt
for _ in {1..12}; do
cat >> "$PROMPT_FILE" <<'EOF'

Additional context block: Quantum computing is often misunderstood as a magical faster computer, but its real promise is narrower and more subtle. The explanation should distinguish between theoretical asymptotic advantage and practical end-to-end system performance, including overheads, noise, error rates, and the difference between current NISQ systems and future fault-tolerant systems.
EOF
done

echo "Starting request-aligned GPU validation loop..."

for ((i=1; i<=ITERATIONS; i++)); do
  echo ""
  echo "=== Iteration $i ==="

  RESP_FILE="$(mktemp "${TMPDIR:-/tmp}/profile-test-resp.XXXXXX")"
  SMI_FILE="$(mktemp "${TMPDIR:-/tmp}/profile-test-smi.XXXXXX")"

  echo "Starting in-flight inference request (long prompt, max_tokens=1024)..."
  # --rawfile avoids ARG_MAX from embedding the whole prompt in the shell argv
  curl -sS -o "$RESP_FILE" http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
      --rawfile prompt "$PROMPT_FILE" \
      '{
        model: "llama3",
        messages: [{ role: "user", content: $prompt }],
        max_tokens: 1024,
        temperature: 0
      }')" &
  CURL_PID=$!

  # Wait so inference is active
  sleep "$INFER_DELAY"

  echo "Launching profile diagnose during active inference..."
  ./profile diagnose &
  PROFILE_PID=$!

  echo "Collecting nvidia-smi samples (250ms cadence over ${SMI_WINDOW_SEC}s)..."
  nvidia-smi --id="${GPU_ID}" \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm \
    --format=csv,noheader,nounits \
    -lms 250 -c "$SMI_SAMPLES" > "$SMI_FILE"

  wait "$PROFILE_PID"
  wait "$CURL_PID"

  echo ""
  echo "Model response preview (first 400 chars):"
  jq -r '(.choices[0].message.content? // "") | .[0:400]' "$RESP_FILE" || echo "(preview unavailable)"

  echo ""
  echo "nvidia-smi raw samples:"
  cat "$SMI_FILE"

  echo ""
  echo "nvidia-smi last sample:"
  tail -n 1 "$SMI_FILE"

  echo ""
  echo "nvidia-smi averages (approximate):"
  awk -F', ' '
    {
      gpu += $2; mem += $3; pwr += $6; count++
    }
    END {
      if (count>0) {
        printf "  GPU util: %.1f%%\n", gpu/count;
        printf "  Mem util: %.1f%%\n", mem/count;
        printf "  Power:    %.0f W\n", pwr/count;
      }
    }' "$SMI_FILE"

  rm -f "$RESP_FILE" "$SMI_FILE"
  RESP_FILE=""
  SMI_FILE=""

  echo ""
  echo "Sleeping 2 seconds before next iteration..."
  sleep 2
done

echo ""
echo "GPU test loop complete."
