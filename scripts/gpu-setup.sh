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

echo "==> SSH key for private repo..."
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
if [[ ! -f "$HOME/.ssh/id_ed25519" ]]; then
  ssh-keygen -t ed25519 -C "gpu-profile-setup" -f "$HOME/.ssh/id_ed25519" -N ""
  echo "    Add this public key to GitHub: https://github.com/settings/keys"
  echo "    (New SSH key -> paste below, then Save)"
  cat "$HOME/.ssh/id_ed25519.pub"
  echo ""
  read -r -p "After adding the key to GitHub, press Enter to continue..."
fi
chmod 600 "$HOME/.ssh/id_ed25519" 2>/dev/null || true

echo "==> Adding github.com to known_hosts..."
ssh-keyscan -t ed25519 github.com >> "$HOME/.ssh/known_hosts" 2>/dev/null || true
chmod 600 "$HOME/.ssh/known_hosts" 2>/dev/null || true

echo "==> Testing SSH access to GitHub (using $HOME/.ssh/id_ed25519)..."
SSH_OUT=$(ssh -i "$HOME/.ssh/id_ed25519" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 git@github.com 2>&1) || true
if echo "$SSH_OUT" | grep -qi "successfully authenticated\|Hi .*! You've successfully"; then
  echo "    SSH access OK."
else
  echo "    SSH test failed. GitHub said:"
  echo "    $SSH_OUT"
  echo ""
  echo "    If you added the key: use the same GitHub account that can open jungledesh/profile."
  echo "    Try manually: ssh -i $HOME/.ssh/id_ed25519 -o IdentitiesOnly=yes git@github.com"
  exit 1
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
