#!/usr/bin/env bash
# Gemma 4 26B-A4B (MoE) on AMD — counterpart to scripts/start-gemma.sh.
# Runs inside vllm/vllm-openai-rocm base image which already has vLLM + PyTorch
# + ROCm installed. No pip install needed. Same posture as start-amd.sh.
#
# Gemma 4 is Apache-2.0 and ungated — no HF_TOKEN required (a read token only
# helps with download rate limits). 25.2B total / 3.8B active; ~58GB bf16.

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

MODEL_REPO="${MODEL_REPO:-google/gemma-4-26B-A4B-it}"
# Own the served identity for this launcher. Image must not bake SERVED_NAME=qwen
# (see Dockerfile amd stage); ${SERVED_NAME:-} only helps when the env is unset.
SERVED_NAME="${SERVED_NAME:-gemma-4-26b-a4b}"
export SERVED_NAME
export PROFILE_MODEL=gemma

APP_DIR="${APP_DIR:-/home/appuser/app}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/gemma4-26b-a4b}"
TMUX_SESSION="${TMUX_SESSION:-vllm}"
LOG_FILE="${APP_DIR}/vllm.log"

echo "Starting AMD container (Gemma 4 26B-A4B)..."

mkdir -p "$APP_DIR" "$MODELS_DIR"

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
fi

# DOWNLOAD_MODEL=0 skips the ~58GB pull (weights must already be at MODEL_PATH).
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

# Minimal flags; Profile diagnoses the rest.
#   --max-model-len 32768   same as the NVIDIA run so numbers compare 1:1.
#   --trust-remote-code     permit model-shipped code (harmless if unused).
#   (no --max-num-seqs, no --gpu-memory-utilization) let vLLM pick.
# Tool-call wiring for agent swarm (same as start-gemma.sh). Disable with
# ENABLE_GEMMA_TOOLS=0. If ROCm graph capture misbehaves, add --enforce-eager
# via EXTRA_ARGS.
ENABLE_GEMMA_TOOLS="${ENABLE_GEMMA_TOOLS:-1}"
TOOL_ARGS=""
if [[ "$ENABLE_GEMMA_TOOLS" == "1" ]]; then
    TOOL_ARGS="--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4"
    # Vendored template (repo scripts/, copied next to this script by the
    # Dockerfile). Gemma's model-bundled template rejects the system role the
    # swarm sends (400 on every request); the recipe jinja handles system +
    # tools. Fall back to the vLLM install, then to model-bundled.
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    GEMMA4_TEMPLATE="${GEMMA4_TEMPLATE:-$SCRIPT_DIR/tool_chat_template_gemma4.jinja}"
    if [[ ! -f "$GEMMA4_TEMPLATE" ]]; then
        GEMMA4_TEMPLATE="$(
            python -c "
import pathlib
import vllm
root = pathlib.Path(vllm.__file__).resolve().parent
cands = list(root.rglob('tool_chat_template_gemma4.jinja'))
print(cands[0] if cands else '')
" 2>/dev/null || true
        )"
    fi
    if [[ -n "$GEMMA4_TEMPLATE" && -f "$GEMMA4_TEMPLATE" ]]; then
        TOOL_ARGS="$TOOL_ARGS --chat-template ${GEMMA4_TEMPLATE}"
        echo "Gemma tool-call template: $GEMMA4_TEMPLATE"
    else
        echo "Using model-bundled chat template (no recipe jinja found)."
        echo "System-role requests will 400; vendored template missing from image."
    fi
fi
EXTRA_ARGS="${EXTRA_ARGS:-}"
tmux new-session -d -s "$TMUX_SESSION" \
"bash -lc 'python -m vllm.entrypoints.openai.api_server \
  --model \"$MODEL_PATH\" \
  --served-model-name $SERVED_NAME \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  $TOOL_ARGS \
  $EXTRA_ARGS \
  2>&1 | tee \"$LOG_FILE\"'"

echo
echo "vLLM running (Gemma 4 26B-A4B) in tmux session '$TMUX_SESSION'"
echo "Served as: $SERVED_NAME  (load/swarm: PROFILE_MODEL=gemma)"
echo "Attach with: tmux attach -t $TMUX_SESSION"
echo "Check: curl -s localhost:8000/metrics | grep cache_config_info"

# Interactive shell when stdin is a TTY; otherwise keep container alive.
if [[ -t 0 ]]; then
  exec bash -l
else
  tail -f /dev/null
fi
