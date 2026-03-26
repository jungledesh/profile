#!/usr/bin/env bash

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

APP_DIR="${APP_DIR:-/home/appuser/app}"
VENV_DIR="${VENV_DIR:-/home/appuser/vllm-env}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/llama3-8b}"
TMUX_SESSION="${TMUX_SESSION:-vllm}"
LOG_FILE="${APP_DIR}/vllm.log"

echo "Starting container..."

mkdir -p "$APP_DIR" "$MODELS_DIR"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install --upgrade uv
uv pip install -U vllm --torch-backend=auto
uv pip install "huggingface-hub>=0.34,<1.0"

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
fi

HF_CLI="${VENV_DIR}/bin/huggingface-cli"

if [[ ! -d "$MODEL_PATH" ]] || [[ -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
    echo "Downloading model..."
    mkdir -p "$MODEL_PATH"
    [[ -x "$HF_CLI" ]] || {
        echo "missing $HF_CLI after hub install" >&2
        exit 1
    }
    "$HF_CLI" download \
        meta-llama/Meta-Llama-3-8B-Instruct \
        --local-dir "$MODEL_PATH"
else
    echo "Model already present."
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Killing existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
fi

tmux new-session -d -s "$TMUX_SESSION" \
"bash -lc 'source \"$VENV_DIR/bin/activate\" && \
python -m vllm.entrypoints.openai.api_server \
  --model \"$MODEL_PATH\" \
  --served-model-name llama3 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 1 \
  2>&1 | tee \"$LOG_FILE\"'"

echo
echo "vLLM running in tmux session '$TMUX_SESSION'"
echo "Attach with: tmux attach -t $TMUX_SESSION"

tail -f /dev/null
