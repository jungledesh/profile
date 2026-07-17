#!/usr/bin/env bash
# Gemma 4 validation run — sliding-window attention, a third memory architecture.
# Purpose: confirm Profile's observed-first capacity path handles sliding-window
# correctly (state_pages should deduce to ~0, no hybrid label). NOT the demo.
#
# Gemma is gated on HuggingFace: accept the license once, then pass HF_TOKEN.
# OS packages installed in the Dockerfile; runs as appuser, no apt-get.

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

PIP_VERSION="${PIP_VERSION:-26.0.1}"
UV_VERSION="${UV_VERSION:-0.11.1}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"

# Gemma 4 31B (interleaved sliding-window + global attention; multimodal
# image+text + reasoning model). ~66GB bf16 incl. vision encoder; fits H100 80GB.
# Apache-2.0, NOT gated — no license to accept, download is open.
MODEL_REPO="${MODEL_REPO:-google/gemma-4-31B-it}"

APP_DIR="${APP_DIR:-/home/appuser/app}"
VENV_DIR="${VENV_DIR:-/home/appuser/vllm-env}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/gemma4-31b}"
TMUX_SESSION="${TMUX_SESSION:-vllm}"
LOG_FILE="${APP_DIR}/vllm.log"

echo "Starting container (Gemma 4 validation)..."

mkdir -p "$APP_DIR" "$MODELS_DIR"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

python -m pip install "pip==${PIP_VERSION}"
python -m pip install "uv==${UV_VERSION}"
uv pip install "vllm==${VLLM_VERSION}"

# Gemma 4 is Apache-2.0 and ungated — no token required for access.
# A read token only helps with download rate limits; use it if you have one.
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

CUDA13_LIB=$(find "$VENV_DIR" -path "*/nvidia/cu13/lib" -type d 2>/dev/null | head -1)
if [[ -n "$CUDA13_LIB" ]]; then
    export LD_LIBRARY_PATH="${CUDA13_LIB}:${LD_LIBRARY_PATH:-}"
    echo "CUDA 13 libs: $CUDA13_LIB"
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Killing existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
fi

# Minimal serve for a correctness test — no reasoning/tool parsers.
# Gemma is not a reasoning model and this run only needs the server up so
# Profile can read cache_config_info under load.sh traffic.
#   --max-model-len 32768   comparable to the Qwen run; Gemma supports it.
#   --trust-remote-code     permit any model-shipped code (harmless if unused).
#   (no --max-num-seqs)     let vLLM pick, so we observe its natural
#                           sliding-window allocation — that IS the test.
tmux new-session -d -s "$TMUX_SESSION" \
"bash -lc 'source \"$VENV_DIR/bin/activate\" && \
export LD_LIBRARY_PATH=\"${CUDA13_LIB}:\${LD_LIBRARY_PATH:-}\" && \
vllm serve \"$MODEL_PATH\" \
  --served-model-name gemma-4-31b \
  --max-model-len 32768 \
  --trust-remote-code \
  2>&1 | tee \"$LOG_FILE\"'"

echo
echo "vLLM running (Gemma 4) in tmux session '$TMUX_SESSION'"
echo "Attach with: tmux attach -t $TMUX_SESSION"
echo "Validate: curl -s localhost:8000/metrics | grep cache_config_info"
echo "Expect: labels present; state_pages ~0; no hybrid overhead."

if [[ -t 0 ]]; then
  exec bash -l
else
  tail -f /dev/null
fi
