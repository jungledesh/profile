#!/usr/bin/env bash
# Apply R5 fix: restart vLLM with --max-num-seqs 128
# Run this during the demo after R5 fires.

set -euo pipefail

VENV_DIR="${VENV_DIR:-/home/appuser/vllm-env}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/qwen25-coder-7b}"
APP_DIR="${APP_DIR:-/home/appuser/app}"
TMUX_SESSION="${TMUX_SESSION:-vllm}"
LOG_FILE="${APP_DIR}/vllm.log"

echo "Applying fix: --max-num-seqs 32 → 128"
echo ""

# Kill existing vLLM
tmux send-keys -t "$TMUX_SESSION" C-c ""
sleep 3

# Restart with the fix applied
tmux send-keys -t "$TMUX_SESSION" \
"bash -lc 'source \"$VENV_DIR/bin/activate\" && \
python -m vllm.entrypoints.openai.api_server \
  --model \"$MODEL_PATH\" \
  --served-model-name qwen-coder \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  2>&1 | tee \"$LOG_FILE\"'" \
Enter

echo "vLLM restarting..."
echo "Profile will detect when it's back up."
