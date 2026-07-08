#!/usr/bin/env bash
# OS packages (curl, jq, gawk, tmux, etc.) are installed in the Dockerfile; this script runs as appuser and does not apt-get.

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

# Pinned Python stack (override via env if needed).
PIP_VERSION="${PIP_VERSION:-26.0.1}"
UV_VERSION="${UV_VERSION:-0.11.1}"
VLLM_VERSION="${VLLM_VERSION:-0.24.0}"

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

python -m pip install "pip==${PIP_VERSION}"
python -m pip install "uv==${UV_VERSION}"
uv pip install "vllm==${VLLM_VERSION}"
# vLLM 0.24.0 pins huggingface_hub<1.0 but its transformers dep needs >=1.0.
pip install --upgrade huggingface_hub

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
fi

if [[ ! -d "$MODEL_PATH" ]] || [[ -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
    echo "Downloading model..."
    mkdir -p "$MODEL_PATH"
    hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir "$MODEL_PATH"
else
    echo "Model already present."
fi

# vLLM 0.24.0 ships nvidia-cuda-runtime 13.x as a pip package.
# The dynamic linker needs the path to libcudart.so.13.
CUDA13_LIB=$(find "$VENV_DIR" -path "*/nvidia/cu13/lib" -type d 2>/dev/null | head -1)
if [[ -n "$CUDA13_LIB" ]]; then
    export LD_LIBRARY_PATH="${CUDA13_LIB}:${LD_LIBRARY_PATH:-}"
    echo "CUDA 13 libs: $CUDA13_LIB"
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Killing existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
fi

tmux new-session -d -s "$TMUX_SESSION" \
"bash -lc 'source \"$VENV_DIR/bin/activate\" && \
export LD_LIBRARY_PATH=\"${CUDA13_LIB}:\${LD_LIBRARY_PATH:-}\" && \
python -m vllm.entrypoints.openai.api_server \
  --model \"$MODEL_PATH\" \
  --served-model-name llama3 \
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

# Interactive shell when stdin is a TTY (e.g. docker run -it); otherwise keep container alive.
if [[ -t 0 ]]; then
  exec bash -l
else
  tail -f /dev/null
fi
