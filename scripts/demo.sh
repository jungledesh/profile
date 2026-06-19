#!/usr/bin/env bash
# Hero's Journey demo — Qwen3.6-27B on RunPod A100 SXM 80GB (TP=1)
# Four acts, all using MODE=rag with ~4K token prompts (3K system + ~1K doc).
#
# Hardware constraints:
#   80 GB HBM2e VRAM (~2000 GB/s)
#   --gpu-memory-utilization 0.90 → ~72 GB for vLLM
#   27B BF16 weights ≈ 54 GB → ~16-18 GB KV headroom
#   Decode ceiling ~37 tok/s (2000 GB/s ÷ 54 GB)
#   KV max concurrent seqs at max_model_len=8192: ~15 (profile computes this)
#   KV max concurrent seqs at max_model_len=4096: ~30 (Act 3 pivot)
#
#   Note: FP8 (--quantization fp8) crashes on A100 — SM80 has no native FP8;
#   Marlin fallback fails on Qwen3.6's 4304-wide visual encoder layer. BF16 only.
#
# This script starts vLLM for Act 1 config (max_model_len=8192, no --max-num-seqs).
# For Acts 3 and 4, kill the tmux session and restart vLLM manually with updated flags.

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
# Pin instrumentator — vLLM 0.18.0 pulls 8.0.0 which breaks on FastAPI 0.137+ (_IncludedRouter has no .path).
# 7.0.2 satisfies vLLM's >=7.0.0 requirement and is the patched release after 7.0.1 was yanked.
uv pip install "prometheus-fastapi-instrumentator==7.0.2"
# Patch routing.py to guard against _IncludedRouter objects that lack .path (FastAPI 0.137+ regression).
python3 -c "
import re, pathlib
p = pathlib.Path('${VENV_DIR}/lib/python3.10/site-packages/prometheus_fastapi_instrumentator/routing.py')
if p.exists():
    src = p.read_text()
    old = '            route_name = route.path'
    new = '            if not hasattr(route, \"path\"):\n                continue\n            route_name = route.path'
    if old in src and new not in src:
        p.write_text(src.replace(old, new, 1))
        print('routing.py patched.')
    else:
        print('routing.py already patched or pattern not found — skipping.')
else:
    print('routing.py not found — skipping patch.')
"
uv pip install "huggingface-hub==${HUGGINGFACE_HUB_VERSION}"
uv pip install "transformers>=5.0.0"

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

# vLLM config — Act 1 / Act 2 (change manually for Acts 3 and 4):
#   --dtype auto                    BF16 (native for Qwen3.6) — 54 GB weights
#   --gpu-memory-utilization 0.90   ~72 GB for vLLM, ~16-18 GB KV headroom
#   --max-model-len 8192            Act 1+2: full context window, kv_max_seqs ≈ 15
#                                   Act 3+4: restart with --max-model-len 4096 --max-num-seqs 30
#   --tensor-parallel-size 1        single A100 — no TP needed
#   --trust-remote-code             required for Qwen3.6 custom architecture
#   --enforce-eager                 skip torch.compile (vLLM 0.18 / torch version mismatch on FakeTensorMode)
#   (no --max-num-seqs)             defaults to 256 — scheduler admits freely, KV is the constraint
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
echo "vLLM starting — Qwen3.6-27B (BF16), max_model_len=8192"
echo ""
echo "  Watch:   tail -f $LOG_FILE"
echo "  Attach:  tmux attach -t $TMUX_SESSION"
echo ""
echo "Wait for 'Application startup complete', then:"
echo "  Terminal 2: MODE=rag CONCURRENCY=4 LAMBDA=4 ./load.sh"
echo "  Terminal 3: profile diagnose --url http://localhost:8000/metrics"

if [[ -t 0 ]]; then
  exec bash -l
else
  tail -f /dev/null
fi
