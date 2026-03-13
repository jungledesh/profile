#!/usr/bin/env bash
set -euo pipefail

# --------------------------
# Paths
# --------------------------
APP_DIR="/root/app"                  # safe location for Rust binary + script
VENV_DIR="${VENV_DIR:-$APP_DIR/vllm-env}"  # Python venv inside app dir
MODELS_DIR="${MODELS_DIR:-/workspace/models}"  # ephemeral mount for models
MODEL_PATH="$MODELS_DIR/llama3-8b"

echo "Starting container..."

# --------------------------
# Create Python venv if missing
# --------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --------------------------
# Install vLLM (CUDA) at container start
# --------------------------
pip install --upgrade pip
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124

# --------------------------
# Ensure models directory exists
# --------------------------
mkdir -p "$MODELS_DIR"

# --------------------------
# HuggingFace login if token provided
# --------------------------
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "Logging into HuggingFace..."
  huggingface-cli login --token "$HF_TOKEN"
  # Optional: add --add-to-git-credential if you need git push/pull too
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
# Start vLLM server
# --------------------------
tmux new-session -d -s vllm \
"source $VENV_DIR/bin/activate && python -m vllm.entrypoints.openai.api_server \
 --model $MODEL_PATH \
 --served-model-name llama3 \
 --host 0.0.0.0 \
 --port 8000 \
 --dtype auto \
 --gpu-memory-utilization 0.8 \
 --tensor-parallel-size 1"

echo ""
echo "vLLM running in tmux session 'vllm'"
echo "Attach with: tmux attach -t vllm"

# --------------------------
# Keep container alive
# --------------------------
tail -f /dev/null