# GPU machine setup for development

Use these steps to set up a GPU machine for vLLM and profile development.

**Automated setup:** From the repo root, run `./scripts/gpu-setup.sh` (after `nvidia-smi` works). It installs packages, creates the venv, installs vLLM, runs HF login and Llama download, and starts the server in a detached tmux session. The steps below match what the script does and are useful for manual runs or reference.

---

## 1. Verify GPU and drivers

```bash
nvidia-smi
```

Confirm the GPU is visible and drivers are installed.

---

## 2. System packages (including git)

```bash
apt update && apt upgrade -y
apt install -y git curl wget build-essential tmux python3-venv python3-pip
```

Git is needed to clone the repo; the rest are for general use and the vLLM venv.

---

## 3. Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

rustc --version   # Should show e.g., rustc 1.78.0
cargo --version   # Should show e.g., cargo 1.78.0
```

---

## 4. Clone the repo

```bash
git clone https://github.com/jungledesh/profile
cd profile
```

---

## 5. Python virtual environment and vLLM

```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
pip install --upgrade pip

pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
```

---

## 6. Verify vLLM

```bash
python -c "import vllm; print(vllm.__version__)"
```

---

## 7. Hugging Face login and model download (optional)

vLLM includes Hugging Face Hub support; no separate `pip install huggingface-hub` is needed.

```bash
mkdir -p /workspace/models

huggingface-cli login

huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir /workspace/models/llama3-8b
```

After login, the Llama 3 8B Instruct model will be in `/workspace/models/llama3-8b`.

---

## 8. Run vLLM OpenAI-compatible server

Make sure you are in the vLLM Python environment (`source vllm-env/bin/activate`), then:

```bash
tmux new -s vllm

python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/llama3-8b \
  --served-model-name llama3 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 1
```

The server listens on port 8000. Detach from the session with `Ctrl+b` then `d`; reattach with `tmux attach -t vllm`.

---

## 9. Calling the model (from inside the server)

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
