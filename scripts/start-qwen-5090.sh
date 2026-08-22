#!/usr/bin/env bash
# Qwen3.8-27B NVFP4 on NVIDIA RTX 5090. Same pattern as start-muse.sh:
# pip-install vLLM into the image venv, download weights, serve in tmux.
#
# BF16 (~54 GB) does not fit 32 GB. Default: Inferact NVFP4 (vLLM-native 4-bit).
# H100 BF16 remains ./start.sh. Override (MODEL_REVISION required, 40-char SHA):
#   MODEL_REPO=Qwen/Qwen3.8-27B-FP8 MODEL_REVISION=<sha> \
#   MODEL_PATH=/workspace/models/qwen38-27b-fp8
#
# OS packages installed in the Dockerfile; runs as appuser, no apt-get.

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

PIP_VERSION="${PIP_VERSION:-26.0.1}"
UV_VERSION="${UV_VERSION:-0.11.1}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"

DEFAULT_MODEL_REPO="Inferact/Qwen3.8-27B-NVFP4"
# Reviewed 2026-08-17 from huggingface.co/api/models/Inferact/Qwen3.8-27B-NVFP4
DEFAULT_MODEL_REVISION="6128240ebaf4eaa7bad2b3d1c72c37d677c5f462"
MODEL_REPO="${MODEL_REPO:-$DEFAULT_MODEL_REPO}"
if [[ "$MODEL_REPO" != "$DEFAULT_MODEL_REPO" ]]; then
    if [[ -z "${MODEL_REVISION:-}" ]]; then
        echo "ERROR: MODEL_REPO override requires MODEL_REVISION (40-char commit SHA)."
        exit 1
    fi
else
    MODEL_REVISION="${MODEL_REVISION:-$DEFAULT_MODEL_REVISION}"
fi
if [[ ! "$MODEL_REVISION" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "ERROR: MODEL_REVISION must be a 40-character commit SHA (got: ${MODEL_REVISION})."
    exit 1
fi

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
VLLM_PORT="${VLLM_PORT:-8000}"
READY_TIMEOUT_SECS="${READY_TIMEOUT_SECS:-600}"
READY_POLL_SECS="${READY_POLL_SECS:-2}"

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

download_model() {
    echo "Downloading ${MODEL_REPO} @ ${MODEL_REVISION}..."
    mkdir -p "$MODEL_PATH"
    hf download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$MODEL_PATH"
}

verify_model() {
    echo "Verifying ${MODEL_REPO} @ ${MODEL_REVISION} in ${MODEL_PATH}..."
    hf cache verify "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$MODEL_PATH" --fail-on-missing-files
}

DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}"
if [[ ! -d "$MODEL_PATH" ]] || [[ -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
    if [[ "$DOWNLOAD_MODEL" != "1" ]]; then
        echo "ERROR: DOWNLOAD_MODEL=0 but no weights at $MODEL_PATH."
        echo "Set DOWNLOAD_MODEL=1 or mount weights at MODEL_PATH."
        exit 1
    fi
    download_model
else
    echo "Model directory present; verifying."
fi

if ! verify_model; then
    if [[ "$DOWNLOAD_MODEL" != "1" ]]; then
        echo "ERROR: weight verification failed at $MODEL_PATH and DOWNLOAD_MODEL=0."
        echo "Set DOWNLOAD_MODEL=1 or mount a complete snapshot at MODEL_PATH."
        exit 1
    fi
    echo "Verification failed; retrying download."
    download_model
    if ! verify_model; then
        echo "ERROR: weight verification failed after retry."
        exit 1
    fi
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
#   --enable-auto-tool-choice    agent swarm
#   --tool-call-parser           qwen3_coder: XML tool calls (hermes mangles them)
#   --reasoning-parser           qwen3: route <think> out of content or the client breaks
# No --trust-remote-code: Inferact NVFP4 has no repo .py; vLLM 0.25 ships qwen3_5.
# No --speculative-config (MTP off). No --kv-cache-dtype (Profile may prescribe it).
EXTRA_ARGS="${EXTRA_ARGS:-}"
tmux new-session -d -s "$TMUX_SESSION" \
"bash -lc 'set -euo pipefail; source \"$VENV_DIR/bin/activate\" && \
${TMUX_CUDA_EXPORT}vllm serve \"$MODEL_PATH\" \
  --served-model-name $SERVED_NAME \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --attention-backend TRITON_ATTN \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  $EXTRA_ARGS \
  2>&1 | tee \"$LOG_FILE\"'"

# vLLM 0.25.1 registers GET /health, not /health/ready.
ready=0
elapsed=0
echo "Waiting for vLLM /health on port ${VLLM_PORT} (timeout ${READY_TIMEOUT_SECS}s)..."
while (( elapsed < READY_TIMEOUT_SECS )); do
    if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "ERROR: tmux session '$TMUX_SESSION' ended before readiness. See $LOG_FILE."
        exit 1
    fi
    if curl -fsS --max-time 2 "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep "$READY_POLL_SECS"
    elapsed=$((elapsed + READY_POLL_SECS))
done
if [[ "$ready" != "1" ]]; then
    echo "ERROR: vLLM did not become ready within ${READY_TIMEOUT_SECS}s. See $LOG_FILE."
    exit 1
fi

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
