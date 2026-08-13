#!/usr/bin/env bash
# Muse Glimmer 30B on NVIDIA — same pattern as start.sh / start-gemma.sh:
# pip-install vLLM into the image venv, download weights, serve in tmux.
#
# Meta's consumer agentic model (distilled from Muse Spark). Default for the
# NVIDIA image on RTX 5090: NVFP4 (~25 GB), DFlash off.
#
# Quant for 32 GB (5090): Inferact/Muse-Glimmer-30B-NVFP4-W4A4 (ModelOpt NVFP4).
# Meta's K-Quant GGUF is llama.cpp; this is the vLLM-native 4-bit path.
# Override:
#   MODEL_REPO=RedHatAI/Muse-Glimmer-30B-FP8-block  # ~33 GB; tight on 32 GB
#   MODEL_REPO=meta-models/Muse-Glimmer-30B         # BF16; needs ~>55 GB
#
# If stock vLLM rejects the Muse arch/parsers, this script already installs from
# the Muse support branch (VLLM_PIP_SPEC). Do not swap the Docker base image.
#
# OS packages installed in the Dockerfile; runs as appuser, no apt-get.

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

PIP_VERSION="${PIP_VERSION:-26.0.1}"
UV_VERSION="${UV_VERSION:-0.11.1}"
# Muse Glimmer is not in any PyPI wheel yet (PR vllm-project/vllm#51655 still open).
# Same install path as Qwen/Gemma (uv pip into VENV_DIR); different package source.
# Pin: last commit on tiezhen/new-model-support from the Aug 12 5090 demo day.
# Floating branch tip merged main on Aug 13; pip then resolved torch 2.13.0+cu130
# which cannot init on driver 570 (CUDA API 12090). Override only to debug.
VLLM_PIP_SPEC="${VLLM_PIP_SPEC:-git+https://github.com/xianbaoqian/vllm.git@98f86b9c02329200a0390aecfe598e27928cbf40}"

MODEL_REPO="${MODEL_REPO:-Inferact/Muse-Glimmer-30B-NVFP4-W4A4}"
SERVED_NAME="${SERVED_NAME:-muse-glimmer-30b}"
export SERVED_NAME
export PROFILE_MODEL=muse
# Profile maps modelopt/nvfp4 → 4-bit when /info is unread or scheme is opaque.
export QUANTIZATION="${QUANTIZATION:-modelopt}"

APP_DIR="${APP_DIR:-/home/appuser/app}"
VENV_DIR="${VENV_DIR:-/home/appuser/vllm-env}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/muse-glimmer-30b-nvfp4}"
TMUX_SESSION="${TMUX_SESSION:-vllm}"
LOG_FILE="${APP_DIR}/vllm.log"

echo "Starting container (Muse Glimmer 30B NVFP4, DFlash off)..."

# Pin torch CUDA to the host driver. PyPI torch 2.13 defaults to cu130 (needs
# driver 580). 5090 needs CUDA >= 12.8. Last Blackwell wheel that loads on
# driver 570 is 2.11.0+cu128. Do not pin cu126 (no sm_120).
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')"
DRIVER_MAJOR="${DRIVER_VER%%.*}"
if [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] && (( DRIVER_MAJOR >= 580 )); then
    TORCH_BACKEND=cu130
    TORCH_INDEX="https://download.pytorch.org/whl/cu130"
    TORCH_PINS=(torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0)
else
    TORCH_BACKEND=cu128
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    TORCH_PINS=(torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0)
fi
echo "NVIDIA driver ${DRIVER_VER:-unknown}; pinning torch ${TORCH_BACKEND} (${TORCH_PINS[*]})"

mkdir -p "$APP_DIR" "$MODELS_DIR"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

python -m pip install "pip==${PIP_VERSION}"
python -m pip install "uv==${UV_VERSION}"
# Precompiled kernels: avoid a full CUDA compile on the Muse git ref.
export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
echo "Installing vLLM from: ${VLLM_PIP_SPEC}"
uv pip install "${VLLM_PIP_SPEC}"
# Muse git / PyPI resolve torch 2.13.0+cu130. Re-pin to the driver wheel after.
# Do not set UV_TORCH_BACKEND during the git install: cu128 has no 2.13 wheel.
uv pip install --force-reinstall --index-url "${TORCH_INDEX}" "${TORCH_PINS[@]}"
echo "Torch forced: ${TORCH_PINS[*]} from ${TORCH_INDEX}"
# Muse git pins flashinfer==0.6.16.post3; that release crashes on Python 3.10
# (array.array[int]). Force an older wheel after install; do not co-resolve.
FLASHINFER_PIN="${FLASHINFER_PIN:-flashinfer-python==0.6.15.post1}"
uv pip install --force-reinstall --no-deps "${FLASHINFER_PIN}"
echo "FlashInfer forced: ${FLASHINFER_PIN}"

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

# cu130 wheels need libcudart.so.13 on the linker path. Do not prepend cu13
# libs in front of a cu128 torch; that is the 570 failure mode.
CUDA13_LIB=""
if [[ "$TORCH_BACKEND" == "cu130" ]]; then
    CUDA13_LIB=$(find "$VENV_DIR" -path "*/nvidia/cu13/lib" -type d 2>/dev/null | head -1)
    if [[ -n "$CUDA13_LIB" ]]; then
        export LD_LIBRARY_PATH="${CUDA13_LIB}:${LD_LIBRARY_PATH:-}"
        echo "CUDA 13 libs: $CUDA13_LIB"
    fi
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Killing existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
fi

# Minimal flags; Profile diagnoses seats / memory / batch tokens.
#   --trust-remote-code          Muse arch
#   --enable-auto-tool-choice    agent swarm
#   --tool-call-parser / --reasoning-parser muse_glimmer
#   --max-model-len 32768        KV headroom on 32 GB after ~25 GB weights
# No --speculative-config (DFlash off).
EXTRA_ARGS="${EXTRA_ARGS:-}"
TMUX_CUDA_EXPORT=""
if [[ -n "$CUDA13_LIB" ]]; then
    TMUX_CUDA_EXPORT="export LD_LIBRARY_PATH=\"${CUDA13_LIB}:\${LD_LIBRARY_PATH:-}\" && "
fi
tmux new-session -d -s "$TMUX_SESSION" \
"bash -lc 'source \"$VENV_DIR/bin/activate\" && \
${TMUX_CUDA_EXPORT}vllm serve \"$MODEL_PATH\" \
  --served-model-name $SERVED_NAME \
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
