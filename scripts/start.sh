#!/usr/bin/env bash
# OS packages (curl, jq, gawk, tmux, etc.) are installed in the Dockerfile; this script runs as appuser and does not apt-get.

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

# Pinned Python stack (override via env if needed).
PIP_VERSION="${PIP_VERSION:-26.0.1}"
UV_VERSION="${UV_VERSION:-0.11.1}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"

APP_DIR="${APP_DIR:-/home/appuser/app}"
VENV_DIR="${VENV_DIR:-/home/appuser/vllm-env}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/qwen36-27b}"
# Own the served identity for this launcher. Image must not bake SERVED_NAME=gemma
# (see Dockerfile nvidia stage); ${SERVED_NAME:-} only helps when the env is unset.
SERVED_NAME="${SERVED_NAME:-Qwen3.6-27B}"
export SERVED_NAME
export PROFILE_MODEL=qwen
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

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
fi

if [[ ! -d "$MODEL_PATH" ]] || [[ -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
    echo "Downloading Qwen3.6-27B..."
    mkdir -p "$MODEL_PATH"
    hf download Qwen/Qwen3.6-27B --local-dir "$MODEL_PATH"
else
    echo "Model already present."
fi

# vLLM 0.25.x ships nvidia-cuda-runtime 13.x as a pip package.
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

# Model/client flags only, plus the one boot requirement vLLM names on this
# hybrid model. No other scheduler/memory tuning; Profile diagnoses the rest.
#   --max-num-seqs 345           serve default 1024 exceeds Mamba cache blocks
#                                (345 on H100 @ default gpu-memory-utilization).
#                                Boot error: lower max_num_seqs to at most 345.
#   --trust-remote-code          custom arch; load fails without it
#   --enable-auto-tool-choice    agent workload must be allowed to call tools
#   --tool-call-parser           qwen3_coder: XML tool calls (hermes mangles them)
#   --reasoning-parser           qwen3: route <think> out of content or client breaks
# --max-model-len omitted: vLLM derives model config max (262144).
tmux new-session -d -s "$TMUX_SESSION" \
"bash -lc 'source \"$VENV_DIR/bin/activate\" && \
export LD_LIBRARY_PATH=\"${CUDA13_LIB}:\${LD_LIBRARY_PATH:-}\" && \
vllm serve \"$MODEL_PATH\" \
  --served-model-name $SERVED_NAME \
  --max-num-seqs 345 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  2>&1 | tee \"$LOG_FILE\"'"

echo
echo "vLLM running in tmux session '$TMUX_SESSION'"
echo "Served as: $SERVED_NAME  (load/swarm: PROFILE_MODEL=qwen)"
echo "Attach with: tmux attach -t $TMUX_SESSION"

# Interactive shell when stdin is a TTY (e.g. docker run -it); otherwise keep container alive.
if [[ -t 0 ]]; then
  exec bash -l
else
  tail -f /dev/null
fi
