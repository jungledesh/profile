#!/usr/bin/env bash
# AMD counterpart to start.sh. Runs inside vllm/vllm-openai-rocm base image
# which already has vLLM + PyTorch + ROCm installed. No pip install needed.

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

APP_DIR="${APP_DIR:-/home/appuser/app}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/llama3-8b}"
SERVED_NAME="${SERVED_NAME:-llama3}"
export SERVED_NAME
export PROFILE_MODEL=llama
TMUX_SESSION="${TMUX_SESSION:-vllm}"
LOG_FILE="${APP_DIR}/vllm.log"

echo "Starting AMD container..."

mkdir -p "$APP_DIR" "$MODELS_DIR"

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
fi

if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]] && [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN required for gated model download (meta-llama/Meta-Llama-3-8B-Instruct)."
    echo "Pass it with: docker run -e HF_TOKEN=hf_... "
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]] || [[ -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
    echo "Downloading model..."
    mkdir -p "$MODEL_PATH"
    hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir "$MODEL_PATH"
else
    echo "Model already present."
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Killing existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
fi

tmux new-session -d -s "$TMUX_SESSION" \
"bash -lc 'python -m vllm.entrypoints.openai.api_server \
  --model \"$MODEL_PATH\" \
  --served-model-name $SERVED_NAME \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.80 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  2>&1 | tee \"$LOG_FILE\"'"

echo
echo "vLLM running in tmux session '$TMUX_SESSION'"
echo "Attach with: tmux attach -t $TMUX_SESSION"
echo "Edit tests: vim $APP_DIR/test.sh"

# Interactive shell when stdin is a TTY; otherwise keep container alive.
if [[ -t 0 ]]; then
  exec bash -l
else
  tail -f /dev/null
fi
