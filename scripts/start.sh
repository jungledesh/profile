#!/usr/bin/env bash
set -euo pipefail

# --------------------------
# Paths
# --------------------------
APP_DIR="${APP_DIR:-/home/appuser/app}"
VENV_DIR="${VENV_DIR:-/home/appuser/vllm-env}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/llama3-8b}"
TMUX_SESSION="${TMUX_SESSION:-vllm}"

echo "Starting container..."

# --------------------------
# Ensure required dirs exist
# --------------------------
mkdir -p "$APP_DIR" "$MODELS_DIR"

# --------------------------
# Create / activate Python venv
# --------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --------------------------
# Install latest stable vLLM
# --------------------------
python -m pip install --upgrade pip
python -m pip install --upgrade uv
uv pip install -U vllm --torch-backend=auto
python -m pip install -U huggingface_hub

# --------------------------
# Hugging Face login if token provided
# --------------------------
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "Logging into Hugging Face..."
    huggingface-cli login --token "$HF_TOKEN"
fi

# --------------------------
# Download model if missing
# --------------------------
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Downloading model..."
    huggingface-cli download \
        meta-llama/Meta-Llama-3-8B-Instruct \
        --local-dir "$MODEL_PATH"
else
    echo "Model already present."
fi

# --------------------------
# Restart tmux session cleanly
# --------------------------
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Killing existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
fi

# --------------------------
# Start vLLM server
# --------------------------
tmux new-session -d -s "$TMUX_SESSION" \
"source \"$VENV_DIR/bin/activate\" && \
python -m vllm.entrypoints.openai.api_server \
  --model \"$MODEL_PATH\" \
  --served-model-name llama3 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 1"

echo
echo "vLLM running in tmux session '$TMUX_SESSION'"
echo "Attach with: tmux attach -t $TMUX_SESSION"

# --------------------------
# Keep container alive
# --------------------------
tail -f /dev/null