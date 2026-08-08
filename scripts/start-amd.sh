#!/usr/bin/env bash
# AMD counterpart to start.sh. Qwen3.6-27B only — no other model downloads.
# Runs inside vllm/vllm-openai-rocm base image which already has vLLM + PyTorch
# + ROCm installed. No pip install needed.
#
# Qwen is ungated — no HF_TOKEN required (a read token only helps with rate limits).

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

APP_DIR="${APP_DIR:-/home/appuser/app}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/qwen36-27b}"
# Own the served identity for this launcher. Image must not bake SERVED_NAME=gemma
# (see Dockerfile amd stage); ${SERVED_NAME:-} only helps when the env is unset.
SERVED_NAME="${SERVED_NAME:-Qwen3.6-27B}"
export SERVED_NAME
export PROFILE_MODEL=qwen
TMUX_SESSION="${TMUX_SESSION:-vllm}"
LOG_FILE="${APP_DIR}/vllm.log"

echo "Starting AMD container (Qwen3.6-27B)..."

mkdir -p "$APP_DIR" "$MODELS_DIR"

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
fi

# DOWNLOAD_MODEL=0 skips the pull (weights must already be at MODEL_PATH).
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
#   --trust-remote-code          custom arch; load fails without it
#   --enable-auto-tool-choice    agent swarm must be allowed to call tools
#   --tool-call-parser           qwen3_coder: XML tool calls (hermes mangles them)
#   --reasoning-parser           qwen3: route <think> out of content or client breaks
# --max-num-seqs omitted: NVIDIA H100 needs 345 (Mamba cache); MI300X has more
#   VRAM so vLLM picks. If boot fails on Mamba blocks, pass via EXTRA_ARGS.
# --max-model-len omitted: vLLM derives model config max.
EXTRA_ARGS="${EXTRA_ARGS:-}"
tmux new-session -d -s "$TMUX_SESSION" \
"bash -lc 'python -m vllm.entrypoints.openai.api_server \
  --model \"$MODEL_PATH\" \
  --served-model-name $SERVED_NAME \
  --host 0.0.0.0 \
  --port 8000 \
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
echo "Swarm: ./agent-swarm.sh setup && ./agent-swarm.sh run"

# Interactive shell when stdin is a TTY; otherwise keep container alive.
if [[ -t 0 ]]; then
  exec bash -l
else
  tail -f /dev/null
fi
