#!/usr/bin/env bash
# Qwen3.8-27B BF16 on NVIDIA (H100). Same pattern as start-muse.sh / start-gemma.sh:
# pip-install vLLM, download weights, serve in tmux for SWE-bench agent traffic.
#
# Default: Qwen/Qwen3.8-27B (~54 GB). Fits H100 80 GB with KV left. No FP8, no MTP.
# Override: MODEL_REPO=Qwen/Qwen3.8-27B-FP8 MODEL_PATH=/workspace/models/qwen38-27b-fp8
#
# OS packages installed in the Dockerfile; runs as appuser, no apt-get.

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

PIP_VERSION="${PIP_VERSION:-26.0.1}"
UV_VERSION="${UV_VERSION:-0.11.1}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.8-27B}"
SERVED_NAME="${SERVED_NAME:-Qwen3.8-27B}"
export SERVED_NAME
export PROFILE_MODEL=qwen

APP_DIR="${APP_DIR:-/home/appuser/app}"
VENV_DIR="${VENV_DIR:-/home/appuser/vllm-env}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/qwen38-27b}"
TMUX_SESSION="${TMUX_SESSION:-vllm}"
LOG_FILE="${APP_DIR}/vllm.log"

echo "Starting container (Qwen3.8-27B BF16, SWE-bench agents)..."

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
    echo "Downloading ${MODEL_REPO}..."
    mkdir -p "$MODEL_PATH"
    hf download "$MODEL_REPO" --local-dir "$MODEL_PATH"
else
    echo "Model already present."
fi

# vLLM 0.25.x ships nvidia-cuda-runtime 13.x as a pip package.
CUDA13_LIB=$(find "$VENV_DIR" -path "*/nvidia/cu13/lib" -type d 2>/dev/null | head -1)
TMUX_CUDA_EXPORT=""
if [[ -n "$CUDA13_LIB" ]]; then
    export LD_LIBRARY_PATH="${CUDA13_LIB}:${LD_LIBRARY_PATH:-}"
    echo "CUDA 13 libs: $CUDA13_LIB"
    TMUX_CUDA_EXPORT="export LD_LIBRARY_PATH=\"${CUDA13_LIB}:\${LD_LIBRARY_PATH:-}\" && "
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Killing existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
fi

# Model/client flags only, plus the hybrid boot cap. Profile diagnoses the rest.
#   --max-model-len 32768        native 262144 will not leave KV on H100 with ~54 GB BF16
#   --max-num-seqs 345           Qwen3.6 H100 boot: default 1024 exceeds Mamba blocks
#                                (345 @ default gpu-memory-utilization). 3.8 is the
#                                same 3:1 DeltaNet/attention interleave.
#   --trust-remote-code          custom arch
#   --enable-auto-tool-choice    agent swarm must be allowed to call tools
#   --tool-call-parser           qwen3_coder: XML tool calls (hermes mangles them)
#   --reasoning-parser           qwen3: route <think> out of content or the client breaks
# No --speculative-config (MTP off). No --kv-cache-dtype (Profile may prescribe it).
EXTRA_ARGS="${EXTRA_ARGS:-}"
tmux new-session -d -s "$TMUX_SESSION" \
"bash -lc 'source \"$VENV_DIR/bin/activate\" && \
${TMUX_CUDA_EXPORT}vllm serve \"$MODEL_PATH\" \
  --served-model-name $SERVED_NAME \
  --max-model-len 32768 \
  --max-num-seqs 345 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  $EXTRA_ARGS \
  2>&1 | tee \"$LOG_FILE\"'"

echo
echo "vLLM running (Qwen3.8-27B BF16) in tmux session '$TMUX_SESSION'"
echo "Served as: $SERVED_NAME  (load/swarm: PROFILE_MODEL=qwen)"
echo "Attach with: tmux attach -t $TMUX_SESSION"
echo "Swarm: PROFILE_MODEL=qwen ./agent-swarm.sh setup && PROFILE_MODEL=qwen ./agent-swarm.sh run"

if [[ -t 0 ]]; then
  exec bash -l
else
  tail -f /dev/null
fi
