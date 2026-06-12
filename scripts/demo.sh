#!/usr/bin/env bash
# Demo setup — Qwen3.6-27B on RunPod A100 SXM 80GB (TP=1)
# RAG scenario: realistic document load via MODE=rag.
#
# Hardware constraints:
#   80 GB HBM2e VRAM (~2000 GB/s)
#   --gpu-memory-utilization 0.90 → ~72 GB for vLLM
#   27B BF16 weights ≈ 54 GB → ~18 GB KV headroom
#   Decode ceiling ~37 tok/s (2000 GB/s ÷ 54 GB)
#   Note: FP8 (--quantization fp8) crashes on A100 — SM80 has no native FP8;
#   Marlin fallback fails on Qwen3.6's 4304-wide visual encoder layer.
#   BF16 is correct. KV saturates at ~23 concurrent at actual ~2500 token RAG usage.
#   CONCURRENCY=30 LAMBDA=0.5 keeps ~26 in-flight — above saturation, R2 fires.

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

echo "Starting demo environment — Qwen3.6-27B / A100 SXM 80GB..."

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
    echo "Downloading Qwen3.6-27B (~54 GB BF16)..."
    mkdir -p "$MODEL_PATH"
    [[ -x "$HF_CLI" ]] || { echo "missing $HF_CLI after hub install" >&2; exit 1; }
    "$HF_CLI" download \
        Qwen/Qwen3.6-27B \
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
#   --dtype auto                    BF16 (native for Qwen3.6) — 54 GB weights
#   --gpu-memory-utilization 0.90   ~72 GB for vLLM, ~18 GB KV headroom
#   --max-model-len 8192            18 GB KV headroom / 784-token blocks = ~94 blocks.
#                                   8192-token sequences need ~11 blocks each → ~8 max
#                                   full-length seqs. At actual ~2500 token RAG usage
#                                   (~4 blocks each) → ~23 concurrent before saturation.
#                                   32768 would need ~42 blocks/seq → only 2 fit → crash.
#   --tensor-parallel-size 1        single A100 — no TP needed
#   --trust-remote-code             required for Qwen3.6 custom architecture
#   --enforce-eager                 skip torch.compile (vLLM 0.18 / torch version mismatch on FakeTensorMode)
#   no --max-num-seqs               let vLLM auto-size from available KV headroom
tmux new-session -d -s "$TMUX_SESSION" \
"bash -lc 'source \"$VENV_DIR/bin/activate\" && \
python -m vllm.entrypoints.openai.api_server \
  --model \"$MODEL_PATH\" \
  --served-model-name Qwen3.6-27B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --enforce-eager \
  --trust-remote-code \
  2>&1 | tee \"$LOG_FILE\"'"

echo
echo "vLLM running in tmux session '$TMUX_SESSION'"
echo "Model: Qwen3.6-27B (BF16)"
echo "GPU:   A100 SXM 80GB — ~18 GB KV headroom (BF16), decode ceiling ~37 tok/s"
echo "Log:   $LOG_FILE"
echo "Attach: tmux attach -t $TMUX_SESSION"
echo ""
echo "Wait for 'Application startup complete' in the log, then:"
echo ""
echo "  Terminal 2 — load:"
echo "    MODEL=Qwen3.6-27B MODE=rag CONCURRENCY=30 LAMBDA=0.5 ./load.sh"
echo ""
echo "  Terminal 3 — profile:"
echo "    profile diagnose --url http://localhost:8000/metrics --tensor-parallel-size 1 --duration 2m"

if [[ -t 0 ]]; then
  exec bash -l
else
  tail -f /dev/null
fi
