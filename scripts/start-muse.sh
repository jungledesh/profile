#!/usr/bin/env bash
# Muse Glimmer 30B on NVIDIA — Meta's consumer agentic model (distilled from Muse Spark).
# Default for the NVIDIA image on RTX 5090: NVFP4 (~25 GB), DFlash off.
# Profile + agent-swarm use PROFILE_MODEL=muse.
#
# Quant for 32 GB (5090): Inferact/Muse-Glimmer-30B-NVFP4-W4A4 (ModelOpt NVFP4).
# Meta's K-Quant GGUF is llama.cpp; this is the vLLM-native 4-bit path Profile can price.
# Override:
#   MODEL_REPO=RedHatAI/Muse-Glimmer-30B-FP8-block  # ~33 GB; tight on 32 GB
#   MODEL_REPO=meta-models/Muse-Glimmer-30B         # BF16; needs ~>55 GB
#
# Requires a Muse-capable vLLM (image: vllm/vllm-openai:muse-glimmer-*).
# DFlash stays off until we turn it on deliberately (EXTRA_ARGS / speculative-config).
#
# OS packages installed in the Dockerfile; runs as appuser, no apt-get.

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

MODEL_REPO="${MODEL_REPO:-Inferact/Muse-Glimmer-30B-NVFP4-W4A4}"
SERVED_NAME="${SERVED_NAME:-muse-glimmer-30b}"
export SERVED_NAME
export PROFILE_MODEL=muse
# Profile maps modelopt/nvfp4 → 4-bit when /info is unread or scheme is opaque.
export QUANTIZATION="${QUANTIZATION:-modelopt}"

APP_DIR="${APP_DIR:-/home/appuser/app}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/muse-glimmer-30b-nvfp4}"
TMUX_SESSION="${TMUX_SESSION:-vllm}"
LOG_FILE="${APP_DIR}/vllm.log"

echo "Starting container (Muse Glimmer 30B NVFP4, DFlash off)..."

mkdir -p "$APP_DIR" "$MODELS_DIR"

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

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Killing existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
fi

# Minimal flags; Profile diagnoses seats / memory / batch tokens.
#   --trust-remote-code          Muse arch requires it
#   --enable-auto-tool-choice    agent swarm
#   --tool-call-parser / --reasoning-parser muse_glimmer
#                                channel-scoped framing; without these, empty output
#   --max-model-len 32768        leave KV headroom on 32 GB after ~25 GB weights
# No --speculative-config (DFlash off).
EXTRA_ARGS="${EXTRA_ARGS:-}"
tmux new-session -d -s "$TMUX_SESSION" \
"bash -lc 'vllm serve \"$MODEL_PATH\" \
  --served-model-name $SERVED_NAME \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  $EXTRA_ARGS \
  2>&1 | tee \"$LOG_FILE\"'"

echo
echo "vLLM running (Muse Glimmer 30B) in tmux session '$TMUX_SESSION'"
echo "Served as: $SERVED_NAME  (load/swarm: PROFILE_MODEL=muse)"
echo "Quant hint for Profile: QUANTIZATION=$QUANTIZATION (NVFP4 / modelopt)"
echo "Attach with: tmux attach -t $TMUX_SESSION"
echo "Swarm: PROFILE_MODEL=muse ./agent-swarm.sh setup && PROFILE_MODEL=muse ./agent-swarm.sh run"

if [[ -t 0 ]]; then
  exec bash -l
else
  tail -f /dev/null
fi
