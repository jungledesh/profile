#!/usr/bin/env bash
# Gemma 4 26B-A4B (MoE) on NVIDIA — the demo comparison model against Qwen3.6-27B.
# 25.2B total / 3.8B active params, interleaved local/global attention; ~58GB bf16,
# fits one H100 80GB. Catalog has measured params; KV capacity comes from vLLM's
# observed kv_cache_max_concurrency (KV geometry fields are deliberately None).
#
# Gemma 4 is Apache-2.0 and ungated on HuggingFace — no license click-through,
# no HF_TOKEN required (a read token only helps with download rate limits).
#
# Other Gemma 4 sizes via override, e.g. the 31B validation run:
#   MODEL_REPO=google/gemma-4-31B-it MODEL_PATH=/workspace/models/gemma4-31b \
#   SERVED_NAME=gemma-4-31b ./start-gemma.sh
#
# OS packages installed in the Dockerfile; runs as appuser, no apt-get.

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

PIP_VERSION="${PIP_VERSION:-26.0.1}"
UV_VERSION="${UV_VERSION:-0.11.1}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"

MODEL_REPO="${MODEL_REPO:-google/gemma-4-26B-A4B-it}"
SERVED_NAME="${SERVED_NAME:-gemma-4-26b-a4b}"
export SERVED_NAME
export PROFILE_MODEL="${PROFILE_MODEL:-gemma}"

APP_DIR="${APP_DIR:-/home/appuser/app}"
VENV_DIR="${VENV_DIR:-/home/appuser/vllm-env}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/gemma4-26b-a4b}"
TMUX_SESSION="${TMUX_SESSION:-vllm}"
LOG_FILE="${APP_DIR}/vllm.log"

echo "Starting container (Gemma 4 26B-A4B)..."

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

# Minimal flags; Profile diagnoses the rest. Same posture as start.sh.
#   --max-model-len 32768   comparable to the Qwen demo run; native 256K would
#                           eat the KV pool before the comparison says anything.
#   --trust-remote-code     permit model-shipped code (harmless if unused).
#   (no --max-num-seqs)     let vLLM pick; Profile observes and prescribes.
#
# Tool-call wiring for agent-swarm.sh (vLLM Gemma 4 recipe). Disable with
# ENABLE_GEMMA_TOOLS=0. EXTRA_ARGS still appends after these defaults.
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
"bash -lc 'source \"$VENV_DIR/bin/activate\" && \
export LD_LIBRARY_PATH=\"${CUDA13_LIB}:\${LD_LIBRARY_PATH:-}\" && \
vllm serve \"$MODEL_PATH\" \
  --served-model-name $SERVED_NAME \
  --max-model-len 32768 \
  --trust-remote-code \
  $TOOL_ARGS \
  $EXTRA_ARGS \
  2>&1 | tee \"$LOG_FILE\"'"

echo
echo "vLLM running (Gemma 4 26B-A4B) in tmux session '$TMUX_SESSION'"
echo "Served as: $SERVED_NAME  (load/swarm: PROFILE_MODEL=gemma)"
echo "Attach with: tmux attach -t $TMUX_SESSION"
echo "Check: curl -s localhost:8000/metrics | grep cache_config_info"
echo "Expect: kv_cache_max_concurrency label present (Profile's Observed capacity)."

# Interactive shell when stdin is a TTY (e.g. docker run -it); otherwise keep container alive.
if [[ -t 0 ]]; then
  exec bash -l
else
  tail -f /dev/null
fi
