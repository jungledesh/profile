# GPU machine setup for development

Use these steps to set up a GPU machine for vLLM and profile development.

**CI image (GHCR):** 

`ghcr.io/jungledesh/inference` — tags: `latest`, `<sha>` (push to main); `pr-<N>`, `pr-<N>-<sha>` (PRs). 

E.g. `ghcr.io/jungledesh/inference:latest` or `ghcr.io/jungledesh/inference:pr-17`.

---

## 1. Verify GPU and drivers

```bash
nvidia-smi
```

Confirm the GPU is visible and drivers are installed.

---

## 2. System packages (including git)

```bash
apt-get update -qq && apt-get upgrade -y -qq
apt-get install -y git openssh-client curl wget build-essential tmux python3-venv python3-pip
```

Git and OpenSSH are needed to clone the repo (private repo uses SSH); the rest are for general use and the vLLM venv.

---

## 3. Rust and Cargo

rustup installs both the Rust compiler (`rustc`) and the Cargo build tool. The profile project needs **Rust 1.85+** (for current dependencies; older Cargo fails with `edition2024` required).

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
source "$HOME/.cargo/env"
rustup update stable   # ensure latest stable (avoids "edition2024" / Cargo too old)

rustc --version   # Should show e.g., rustc 1.85.x or newer
cargo --version   # Should show e.g., cargo 1.85.x or newer
```

If you already had Rust installed and see `feature edition2024 is required` or similar, run `rustup update stable` and try again.

---

## 4. SSH (for private repo)

Generate a key on the GPU machine and add the public key to GitHub so you can clone over SSH. The script uses this key with `-i` and `IdentitiesOnly=yes` when cloning.

```bash
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
ssh-keygen -t ed25519 -C "gpu-profile-setup" -f "$HOME/.ssh/id_ed25519" -N ""

# Show the public key — add this to GitHub
cat "$HOME/.ssh/id_ed25519.pub"
```

1. Copy the line that starts with `ssh-ed25519`.
2. On GitHub: **Profile (top right) → Settings → SSH and GPG keys → New SSH key**. Paste the key, give it a title (e.g. `GPU box`), save.
3. Add github.com to known_hosts and test:
   ```bash
   ssh-keyscan -t ed25519 github.com >> "$HOME/.ssh/known_hosts" 2>/dev/null
   ssh -i "$HOME/.ssh/id_ed25519" -o IdentitiesOnly=yes git@github.com
   ```
   You should see “Hi …! You've successfully authenticated, but GitHub does not provide shell access.”

---

## 5. Clone the repo

Use the SSH URL so the private repo is accessible. The script clones into `REPO_DIR` (default `/workspace/profile`).

```bash
# Default: clone into /workspace/profile
git clone git@github.com:jungledesh/profile.git /workspace/profile
cd /workspace/profile
```

**Clone a specific ref (PR, branch, or commit):** The setup script respects `PROFILE_REF`. Use it to run the GPU setup against a PR or a branch/commit instead of the default branch.

| Goal | Set before running the script |
|------|--------------------------------|
| Default branch (main) | (omit, or `PROFILE_REF=` ) |
| A pull request (e.g. PR #42) | `PROFILE_REF=42` |
| A branch | `PROFILE_REF=my-feature` |
| A commit SHA | `PROFILE_REF=abc123def...` |

Example — setup using the code from PR #42:

```bash
PROFILE_REF=42 ./scripts/gpu-setup.sh
```

Example — manual clone then checkout a PR:

```bash
git clone git@github.com:jungledesh/profile.git
cd profile
git fetch origin pull/42/head:pr-42
git checkout pr-42
```

---

## 6. Python virtual environment and vLLM

The script uses `VENV_DIR` (default `./vllm-env`). Create and activate, then install vLLM (CUDA 12.8):

```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
pip install --upgrade pip

pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
```

---

## 7. Verify vLLM

```bash
python -c "import vllm; print('vLLM', vllm.__version__)"
```

---

## 8. Hugging Face login and model download

The script uses `MODELS_DIR` (default `/workspace/models`). Log in with `hf auth login`, then download the model:

```bash
mkdir -p /workspace/models

# Use current HF CLI (hf auth login / hf download)
pip install -q -U huggingface_hub
hf auth login

hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir /workspace/models/llama3-8b
```

**Token-based login (e.g. containers / CI):** If you have `HF_TOKEN` set and want non-interactive login:

```bash
huggingface-cli login --token "$HF_TOKEN"
```

**Alternative download (same result):** You can use `huggingface-cli download` instead of `hf download`:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir /workspace/models/llama3-8b
```

After login, the Llama 3 8B Instruct model will be in `/workspace/models/llama3-8b`.

---

## 9. Run vLLM OpenAI-compatible server

The script starts the server in a **detached** tmux session (`tmux new-session -d -s vllm`). Manually, you can run it in a new session:

```bash
source vllm-env/bin/activate

tmux new -s vllm

python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/llama3-8b \
  --served-model-name llama3 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 1
```

Use `--host 0.0.0.0` so the server listens on all interfaces (needed in containers or remote access). The server listens on port 8000. Detach from the session with `Ctrl+b` then `d`; reattach with `tmux attach -t vllm`.

**Start in detached session (same as start.sh):**

```bash
source vllm-env/bin/activate

tmux new-session -d -s vllm "source vllm-env/bin/activate && python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/llama3-8b \
  --served-model-name llama3 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 1"
```

Then attach with: `tmux attach -t vllm`.

**After setup (new shell):** To use Rust/Cargo and the vLLM venv in a new terminal, run:

```bash
source $HOME/.cargo/env
source vllm-env/bin/activate   # or path from VENV_DIR if you overrode it
```

---

## 10. Calling the model (from inside the server)

Run these from the same machine where the server is running (or use the server’s hostname instead of `localhost`).

**List models**

```bash
curl http://localhost:8000/v1/models
```

Example response:

```json
{
  "data": [
    {
      "id": "llama3"
    }
  ]
}
```

**Chat completion (simple)**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

Example response:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help?"
      }
    }
  ]
}
```

**Chat completion with sampling parameters**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [
      {"role": "user", "content": "Explain reinforcement learning simply"}
    ],
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 200
  }'
```

Example response:

```json
{
  "id": "chatcmpl-a97ae0033481ba8a",
  "object": "chat.completion",
  "created": 1772841804,
  "model": "llama3",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Reinforcement learning! A fascinating topic in AI that's all about learning through rewards and punishments.\n\n**What is Reinforcement Learning?**\n\nReinforcement learning is a type of machine learning where an agent learns to take actions in an environment to maximize a reward. The agent doesn't just learn from examples, but rather through trial and error, by interacting with the environment and receiving feedback in the form of rewards or penalties.\n\n**Key Components:**\n\n1. **Agent**: The AI system that takes actions in the environment.\n2. **Environment**: The world or situation where the agent operates.\n3. **Actions**: The things the agent can do, such as move left or right.\n4. **States**: The current situation or condition of the environment.\n5. **Reward**: A numerical value that the agent receives for its actions, indicating how good or bad they were.\n\n**How it Works:**\n\n1. The agent starts in an initial state.\n2. The agent takes an action,",
        "refusal": null,
        "annotations": null,
        "audio": null,
        "function_call": null,
        "tool_calls": [],
        "reasoning": null
      },
      "logprobs": null,
      "finish_reason": "length",
      "stop_reason": null,
      "token_ids": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "prompt_tokens": 15,
    "total_tokens": 215,
    "completion_tokens": 200,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "prompt_token_ids": null,
  "kv_transfer_params": null
}
```

**Health check**

```bash
curl http://localhost:8000/health
```

A successful request appears in the vLLM server INFO logs as: `127.0.0.1:39294 - "GET /health HTTP/1.1" 200 OK` (port number may differ).
