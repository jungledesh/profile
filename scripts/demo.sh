#!/usr/bin/env bash
# Demo setup — Qwen3.6-27B-Instruct on RunPod A100 40GB (A100 SXM, TP=1)
# RAG scenario: realistic document load via MODE=rag, KV pressure expected under 30 workers.
#
# Hardware constraints:
#   40 GB HBM2e VRAM
#   --gpu-memory-utilization 0.90 → ~36 GB for vLLM
#   27B FP8 weights ≈ 27 GB → ~9 GB KV headroom
#   Decode ceiling ~57 tok/s (1555 GB/s ÷ 27B)
#   KV saturates at ~18 concurrent RAG requests (2000-token prompts, GQA)
#   Closest RunPod proxy for DGX Spark behavior: limited KV headroom drives
#   the same queue buildup and R2 pressure seen on GB10 unified memory

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

# Pinned versions — kept in sync with start.sh
PIP_VERSION="${PIP_VERSION:-26.0.1}"
UV_VERSION="${UV_VERSION:-0.11.1}"
VLLM_VERSION="${VLLM_VERSION:-0.18.0}"
HUGGINGFACE_HUB_VERSION="${HUGGINGFACE_HUB_VERSION:-0.36.2}"
TORCH_BACKEND="${TORCH_BACKEND:-cu126}"

APP_DIR="${APP_DIR:-/home/appuser/app}"
VENV_DIR="${VENV_DIR:-/home/appuser/vllm-env}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/qwen36-27b}"
TMUX_SESSION="${TMUX_SESSION:-vllm}"
LOG_FILE="${APP_DIR}/vllm.log"

echo "Starting demo environment — Qwen3.6-27B-Instruct / A100 40GB..."

mkdir -p "$APP_DIR" "$MODELS_DIR"

# Venv
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

python -m pip install "pip==${PIP_VERSION}"
python -m pip install "uv==${UV_VERSION}"
uv pip install "vllm==${VLLM_VERSION}" --torch-backend="${TORCH_BACKEND}"
uv pip install "huggingface-hub==${HUGGINGFACE_HUB_VERSION}"

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
fi

HF_CLI="${VENV_DIR}/bin/huggingface-cli"

# Download model if not present
if [[ ! -d "$MODEL_PATH" ]] || [[ -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
    echo "Downloading Qwen3.6-27B-Instruct (~27 GB FP8)..."
    mkdir -p "$MODEL_PATH"
    [[ -x "$HF_CLI" ]] || { echo "missing $HF_CLI after hub install" >&2; exit 1; }
    "$HF_CLI" download \
        Qwen/Qwen3.6-27B-Instruct \
        --local-dir "$MODEL_PATH"
else
    echo "Model already present."
fi

# Kill existing session
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Killing existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
fi

# vLLM config:
#   --dtype float8                  FP8 weights — 27 GB, fits in 40 GB VRAM
#   --gpu-memory-utilization 0.90   ~36 GB for vLLM, ~9 GB KV headroom
#   --max-model-len 8192            constrain context to preserve KV headroom;
#                                   at 32768 the single sequence fills all ~9 GB
#   --tensor-parallel-size 1        single A100 — no TP needed
#   --trust-remote-code             required for Qwen3.6 custom architecture
#   no --max-num-seqs               let vLLM auto-size from available KV headroom
tmux new-session -d -s "$TMUX_SESSION" \
"bash -lc 'source \"$VENV_DIR/bin/activate\" && \
python -m vllm.entrypoints.openai.api_server \
  --model \"$MODEL_PATH\" \
  --served-model-name Qwen3.6-27B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float8 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --trust-remote-code \
  2>&1 | tee \"$LOG_FILE\"'"

echo
echo "vLLM running in tmux session '$TMUX_SESSION'"
echo "Model: Qwen3.6-27B-Instruct (FP8)"
echo "GPU:   A100 40GB — ~9 GB KV headroom, saturates at ~18 concurrent RAG requests"
echo "Log:   $LOG_FILE"
echo "Attach: tmux attach -t $TMUX_SESSION"
echo ""
echo "Wait for 'Application startup complete' in the log, then:"
echo ""
echo "  Terminal 2 — load:"
echo "    MODEL=Qwen3.6-27B-Instruct MODE=rag CONCURRENCY=30 ./load.sh"
echo ""
echo "  Terminal 3 — profile:"
echo "    profile diagnose --url http://localhost:8000/metrics --tensor-parallel-size 1 --duration 2m"

if [[ -t 0 ]]; then
  exec bash -l
else
  tail -f /dev/null
fi
