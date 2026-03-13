#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODEL_PATH="$MODELS_DIR/llama3-8b"

echo "Starting container..."

# create Python venv if missing
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install vLLM (CUDA) at container start
pip install --upgrade pip
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124

mkdir -p "$MODELS_DIR"

# Authenticate with HuggingFace if token provided
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "Logging into HuggingFace..."
  echo "$HF_TOKEN" | huggingface-cli login --token --stdin
fi

# Download model if not present
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Downloading model..."
  huggingface-cli download \
    meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir "$MODEL_PATH"
else
  echo "Model already present."
fi

echo "Starting vLLM server..."

tmux new-session -d -s vllm \
"python -m vllm.entrypoints.openai.api_server \
 --model $MODEL_PATH \
 --served-model-name llama3 \
 --port 8000 \
 --dtype auto \
 --gpu-memory-utilization 0.8 \
 --tensor-parallel-size 1"

echo ""
echo "vLLM running in tmux session 'vllm'"
echo "Attach with: tmux attach -t vllm"

# Keep container alive
tail -f /dev/null