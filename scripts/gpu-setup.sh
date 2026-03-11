#!/usr/bin/env bash
# Automate GPU machine setup for vLLM + profile development.
# Run on a fresh Debian/Ubuntu GPU host (after nvidia-smi works).
# Requires interactive HF token when huggingface-cli login runs.

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/workspace/models}"
VENV_DIR="${VENV_DIR:-./vllm-env}"

echo "==> Updating system and installing packages (git, openssh-client, curl, wget, build-essential, tmux, python3-venv, python3-pip)..."
apt-get update -qq && apt-get upgrade -y -qq
apt-get install -y git openssh-client curl wget build-essential tmux python3-venv python3-pip

echo "==> Installing Rust (rustup)..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# shellcheck source=/dev/null
source "$HOME/.cargo/env"
rustc --version
cargo --version

echo "==> SSH key for private repo (skip if you already have access)..."
if [[ ! -f "$HOME/.ssh/id_ed25519" ]]; then
  mkdir -p "$HOME/.ssh"
  ssh-keygen -t ed25519 -C "gpu-profile-setup" -f "$HOME/.ssh/id_ed25519" -N ""
  echo "    Add this public key to GitHub (https://github.com/settings/keys), then press Enter to continue."
  cat "$HOME/.ssh/id_ed25519.pub"
  read -r
fi

echo "==> Cloning profile repo (SSH)..."
REPO_DIR="${REPO_DIR:-/workspace/profile}"
if [[ -d "$REPO_DIR/.git" ]]; then
  echo "    (repo already exists at $REPO_DIR, skipping clone)"
else
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone git@github.com:jungledesh/profile.git "$REPO_DIR"
fi

echo "==> Creating vLLM virtual environment..."
if [[ -d "$VENV_DIR" ]]; then
  echo "    (venv already exists at $VENV_DIR, skipping create)"
else
  python3 -m venv "$VENV_DIR"
fi
VENV_DIR="$(cd "$VENV_DIR" && pwd)"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "==> Installing vLLM (CUDA 12.8)..."
pip install -q --upgrade pip
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128

echo "==> Verifying vLLM..."
python -c "import vllm; print('vLLM', vllm.__version__)"

echo "==> Creating models directory..."
mkdir -p "$MODELS_DIR"

echo "==> Hugging Face login (paste your token when prompted)..."
huggingface-cli login

echo "==> Downloading Meta-Llama-3-8B-Instruct..."
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir "$MODELS_DIR/llama3-8b"

echo "==> Starting vLLM server in detached tmux session 'vllm'..."
tmux new-session -d -s vllm "source '$VENV_DIR/bin/activate' && python -m vllm.entrypoints.openai.api_server --model $MODELS_DIR/llama3-8b --served-model-name llama3 --port 8000 --dtype auto --gpu-memory-utilization 0.8 --tensor-parallel-size 1"

echo ""
echo "Done. Server is running in tmux session 'vllm'. Attach with: tmux attach -t vllm"
echo "See docs/gpu-setup.md for API examples (e.g. curl http://localhost:8000/health)."
