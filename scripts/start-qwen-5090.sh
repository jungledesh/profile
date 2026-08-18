#!/usr/bin/env bash
# Qwen3.8-27B NVFP4 on NVIDIA RTX 5090. Same pattern as start-muse.sh:
# pip-install vLLM into the image venv, download weights, serve in tmux.
#
# BF16 (~54 GB) does not fit 32 GB. Default: Inferact NVFP4 (vLLM-native 4-bit).
# H100 BF16 remains ./start.sh. Override:
#   MODEL_REPO=Qwen/Qwen3.8-27B-FP8 MODEL_PATH=/workspace/models/qwen38-27b-fp8
#
# OS packages installed in the Dockerfile; runs as appuser, no apt-get.

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

PIP_VERSION="${PIP_VERSION:-26.0.1}"
UV_VERSION="${UV_VERSION:-0.11.1}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"

MODEL_REPO="${MODEL_REPO:-Inferact/Qwen3.8-27B-NVFP4}"
SERVED_NAME="${SERVED_NAME:-Qwen3.8-27B}"
export SERVED_NAME
export PROFILE_MODEL=qwen
# Profile maps modelopt/nvfp4 → 4-bit when /info is unread or scheme is opaque.
export QUANTIZATION="${QUANTIZATION:-modelopt}"

APP_DIR="${APP_DIR:-/home/appuser/app}"
VENV_DIR="${VENV_DIR:-/home/appuser/vllm-env}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/qwen38-27b-nvfp4}"
TMUX_SESSION="${TMUX_SESSION:-vllm}"
LOG_FILE="${APP_DIR}/vllm.log"

echo "Starting container (Qwen3.8-27B NVFP4 on RTX 5090, SWE-bench agents)..."

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

DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}"
if [[ ! -d "$MODEL_PATH" ]] || [[ -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
    if [[ "$DOWNLOAD_MODEL" != "1" ]]; then
        echo "ERROR: DOWNLOAD_MODEL=0 but no weights at $MODEL_PATH."
        echo "Set DOWNLOAD_MODEL=1 or mount weights at MODEL_PATH."
        exit 1
    fi
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

# Same 5090 start posture as Muse: short window, 32 seats, Triton. Profile
# diagnoses the rest. Qwen parsers match start.sh (H100 BF16).
#   --max-model-len 32768        native 262144 will not leave KV on 32 GB
#   --max-num-seqs 32            5090 start (H100 hybrid boot used 345)
#   --attention-backend TRITON_ATTN   Blackwell: avoid missing FlashInfer XQA
#   --trust-remote-code          custom arch
#   --enable-auto-tool-choice    agent swarm
#   --tool-call-parser           qwen3_coder: XML tool calls (hermes mangles them)
#   --reasoning-parser           qwen3: route <think> out of content or the client breaks
# No --speculative-config (MTP off). No --kv-cache-dtype (Profile may prescribe it).
EXTRA_ARGS="${EXTRA_ARGS:-}"
tmux new-session -d -s "$TMUX_SESSION" \
"bash -lc 'source \"$VENV_DIR/bin/activate\" && \
${TMUX_CUDA_EXPORT}vllm serve \"$MODEL_PATH\" \
  --served-model-name $SERVED_NAME \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --attention-backend TRITON_ATTN \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  $EXTRA_ARGS \
  2>&1 | tee \"$LOG_FILE\"'"

echo
echo "vLLM running (Qwen3.8-27B NVFP4, RTX 5090) in tmux session '$TMUX_SESSION'"
echo "Served as: $SERVED_NAME  (load/swarm: PROFILE_MODEL=qwen)"
echo "Quant hint for Profile: QUANTIZATION=$QUANTIZATION (NVFP4 / modelopt)"
echo "Attach with: tmux attach -t $TMUX_SESSION"
echo "Swarm: PROFILE_MODEL=qwen ./agent-swarm.sh setup && PROFILE_MODEL=qwen ./agent-swarm.sh run"
echo "Start with AGENTS=2, then raise. Same as the Muse 5090 loop."

if [[ -t 0 ]]; then
  exec bash -l
else
  tail -f /dev/null
fi
