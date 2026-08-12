#!/usr/bin/env bash
# Qwen3.6-27B on NVIDIA. OS packages are in the Dockerfile; runs as appuser.
# Image ships Muse-capable vLLM (vllm/vllm-openai:muse-glimmer-*); no pip install.
#
# Default CMD is start-muse.sh. For Qwen:
#   ./start.sh
#   PROFILE_MODEL=qwen ./agent-swarm.sh run

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

APP_DIR="${APP_DIR:-/home/appuser/app}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/qwen36-27b}"
# Own the served identity for this launcher. Image must not bake SERVED_NAME=muse.
SERVED_NAME="${SERVED_NAME:-Qwen3.6-27B}"
export SERVED_NAME
export PROFILE_MODEL=qwen
TMUX_SESSION="${TMUX_SESSION:-vllm}"
LOG_FILE="${APP_DIR}/vllm.log"

echo "Starting container (Qwen3.6-27B)..."

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
    echo "Downloading Qwen3.6-27B..."
    mkdir -p "$MODEL_PATH"
    hf download Qwen/Qwen3.6-27B --local-dir "$MODEL_PATH"
else
    echo "Model already present."
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Killing existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
fi

# Model/client flags only. No scheduler/memory tuning; Profile diagnoses the rest.
#   --max-num-seqs 345           serve default 1024 exceeds Mamba cache blocks
#                                (345 on H100 @ default gpu-memory-utilization).
#   --trust-remote-code          custom arch; load fails without it
#   --enable-auto-tool-choice    agent workload must be allowed to call tools
#   --tool-call-parser           qwen3_coder: XML tool calls (hermes mangles them)
#   --reasoning-parser           qwen3: route <think> out of content or client breaks
EXTRA_ARGS="${EXTRA_ARGS:-}"
tmux new-session -d -s "$TMUX_SESSION" \
"bash -lc 'vllm serve \"$MODEL_PATH\" \
  --served-model-name $SERVED_NAME \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 345 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  $EXTRA_ARGS \
  2>&1 | tee \"$LOG_FILE\"'"

echo
echo "vLLM running (Qwen3.6-27B) in tmux session '$TMUX_SESSION'"
echo "Served as: $SERVED_NAME  (load/swarm: PROFILE_MODEL=qwen)"
echo "Attach with: tmux attach -t $TMUX_SESSION"

if [[ -t 0 ]]; then
  exec bash -l
else
  tail -f /dev/null
fi
