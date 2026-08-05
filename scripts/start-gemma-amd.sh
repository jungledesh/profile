#!/usr/bin/env bash
# Gemma 4 26B-A4B (MoE) on AMD — counterpart to scripts/start-gemma.sh.
# Runs inside vllm/vllm-openai-rocm base image which already has vLLM + PyTorch
# + ROCm installed. No pip install needed. Same posture as start-amd.sh.
#
# Gemma 4 is Apache-2.0 and ungated — no HF_TOKEN required (a read token only
# helps with download rate limits). 25.2B total / 3.8B active; ~58GB bf16.

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

MODEL_REPO="${MODEL_REPO:-google/gemma-4-26B-A4B-it}"
SERVED_NAME="${SERVED_NAME:-gemma-4-26b-a4b}"

APP_DIR="${APP_DIR:-/home/appuser/app}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/gemma4-26b-a4b}"
TMUX_SESSION="${TMUX_SESSION:-vllm}"
LOG_FILE="${APP_DIR}/vllm.log"

echo "Starting AMD container (Gemma 4 26B-A4B)..."

mkdir -p "$APP_DIR" "$MODELS_DIR"

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
fi

if [[ ! -d "$MODEL_PATH" ]] || [[ -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
    echo "Downloading ${MODEL_REPO}..."
    mkdir -p "$MODEL_PATH"
    hf download "$MODEL_REPO" --local-dir "$MODEL_PATH"
else
    echo "Model already present."
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Killing existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
fi

# Minimal flags; Profile diagnoses the rest.
#   --max-model-len 32768   same as the NVIDIA run so numbers compare 1:1.
#   --trust-remote-code     permit model-shipped code (harmless if unused).
#   (no --max-num-seqs, no --gpu-memory-utilization) let vLLM pick.
# If ROCm graph capture misbehaves on this arch, add --enforce-eager via EXTRA_ARGS.
EXTRA_ARGS="${EXTRA_ARGS:-}"
tmux new-session -d -s "$TMUX_SESSION" \
"bash -lc 'python -m vllm.entrypoints.openai.api_server \
  --model \"$MODEL_PATH\" \
  --served-model-name $SERVED_NAME \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --trust-remote-code \
  $EXTRA_ARGS \
  2>&1 | tee \"$LOG_FILE\"'"

echo
echo "vLLM running (Gemma 4 26B-A4B) in tmux session '$TMUX_SESSION'"
echo "Attach with: tmux attach -t $TMUX_SESSION"
echo "Check: curl -s localhost:8000/metrics | grep cache_config_info"

# Interactive shell when stdin is a TTY; otherwise keep container alive.
if [[ -t 0 ]]; then
  exec bash -l
else
  tail -f /dev/null
fi
