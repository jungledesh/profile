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
# Pin a merge commit ON tiezhen/new-model-support whose tree has muse_glimmer
# parsers (vllm/tool_parsers/muse_glimmer_tool_parser.py). Do not pin a mainline
# SHA that was merged in (98f86b9c has no parsers; serve dies with KeyError).
# Torch CUDA is re-pinned after install for driver 570. Override only to debug.
VLLM_PIP_SPEC="${VLLM_PIP_SPEC:-git+https://github.com/xianbaoqian/vllm.git@1f7f0715848c9acc56ea40faa21c13a02bdc8357}"

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
SETUP_TMUX="${SETUP_TMUX:-muse-setup}"
LOG_FILE="${APP_DIR}/vllm.log"

# RunPod SSH (ssh.runpod.io) drops during long uv fetches. Foreground install
# then gets SIGHUP. Re-exec in tmux when this is an interactive shell. Container
# entrypoint has no tty, so it is left alone. Distinct from TMUX_SESSION (serve).
if [[ -z "${TMUX:-}" && -t 0 && "${MUSE_SKIP_TMUX:-}" != "1" ]]; then
    if tmux has-session -t "$SETUP_TMUX" 2>/dev/null; then
        echo "Attaching to existing tmux session $SETUP_TMUX"
        exec tmux attach -t "$SETUP_TMUX"
    fi
    echo "Re-exec in tmux session $SETUP_TMUX (SSH drop will not kill the install)"
    exec tmux new-session -s "$SETUP_TMUX" "$0" "$@"
fi

echo "Starting container (Muse Glimmer 30B NVFP4, DFlash off)..."

# Pin torch CUDA to the host driver. PyPI torch 2.13 defaults to cu130 (needs
# driver 580). 5090 needs CUDA >= 12.8. Last Blackwell wheel that loads on
# driver 570 is 2.11.0+cu128. Do not pin cu126 (no sm_120).
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
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
# cu12 runtime wheels live on pypi.nvidia.com. uv's default HTTP timeout is
# 30s; that CDN times out from a pod. Parallel 200-800 MB fetches make it worse.
# SSH `export` does not reach this process (container CMD).
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-2}"

install_torch_pins() {
    local attempt
    for attempt in 1 2 3 4 5 6 7 8; do
        if uv pip install --index-url "${TORCH_INDEX}" "$@"; then
            return 0
        fi
        echo "Torch install failed (attempt ${attempt}/8), retrying in 20s..."
        sleep 20
    done
    echo "ERROR: torch install failed after 8 attempts." >&2
    return 1
}

# Precompiled kernels match CUDA 13. On driver 570 / cu128, compile against the
# 12.8 image toolkit for Blackwell (sm_120). Override with VLLM_USE_PRECOMPILED=1.
VLLM_INSTALL_ARGS=()
TORCH_CONSTRAINTS=""
if [[ "$TORCH_BACKEND" == "cu128" ]]; then
    export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-0}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
    install_torch_pins "${TORCH_PINS[@]}"
    echo "Torch pre-pinned for compile: ${TORCH_PINS[*]}"
    # --no-build-isolation uses the venv, not a pep517 isolated env. Muse
    # setup.py imports packaging; install the rest of build-system.requires
    # except torch==2.13.0 (that pin is why we are on 2.11+cu128).
    uv pip install "packaging>=24.2" "cmake>=3.26.1" ninja \
        "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" \
        "setuptools-rust>=1.9.0" wheel jinja2
    TORCH_CONSTRAINTS="$(mktemp)"
    printf '%s\n' "${TORCH_PINS[@]}" > "$TORCH_CONSTRAINTS"
    VLLM_INSTALL_ARGS+=(--no-build-isolation --constraint "$TORCH_CONSTRAINTS")
else
    export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
fi
echo "Installing vLLM from: ${VLLM_PIP_SPEC} (VLLM_USE_PRECOMPILED=${VLLM_USE_PRECOMPILED})"
uv pip install "${VLLM_INSTALL_ARGS[@]}" "${VLLM_PIP_SPEC}"
if [[ -n "$TORCH_CONSTRAINTS" ]]; then
    rm -f "$TORCH_CONSTRAINTS"
fi
# Muse git / PyPI resolve torch 2.13.0+cu130. Re-pin to the driver wheel after.
# Do not set UV_TORCH_BACKEND during the git install: cu128 has no 2.13 wheel.
install_torch_pins --force-reinstall "${TORCH_PINS[@]}"
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
    CUDA13_LIB=$(find "$VENV_DIR" -path "*/nvidia/cu13/lib" -type d 2>/dev/null | head -1 || true)
    if [[ -z "$CUDA13_LIB" ]]; then
        echo "ERROR: cu130 torch needs libcudart.so.13 under $VENV_DIR (nvidia/cu13/lib)." >&2
        exit 1
    fi
    export LD_LIBRARY_PATH="${CUDA13_LIB}:${LD_LIBRARY_PATH:-}"
    echo "CUDA 13 libs: $CUDA13_LIB"
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Killing existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
fi

# FlashInfer 0.6.15 JIT for SM120 (5090) requires CUDA >= 12.9 nvcc. This image
# is 12.8, so _normalize_cuda_arch raises, TARGET_CUDA_ARCHS stays empty, and
# warmup dies with "FlashInfer requires GPUs with sm75 or higher". The GPU is
# sm_120. Use the PyTorch sampler instead. Override to 1 on a CUDA 12.9+ image.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

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
export VLLM_USE_FLASHINFER_SAMPLER=\"${VLLM_USE_FLASHINFER_SAMPLER}\" && \
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
