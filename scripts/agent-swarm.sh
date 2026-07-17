#!/usr/bin/env bash
# Agent swarm setup for the Profile demo: install Grok Build + write validated config.
# Idempotent: safe to run repeatedly. Swarm launch logic comes later.
#
# Requires: vLLM serving Qwen3.6-27B on localhost:8000 (see start.sh).

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

# ── Grok Build binary ───────────────────────────────────────────────────────
if ! command -v grok >/dev/null 2>&1; then
    echo "Installing Grok Build..."
    curl -fsSL https://x.ai/cli/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v grok >/dev/null 2>&1 || { echo "grok not on PATH after install" >&2; exit 1; }

# ── Config (validated on H100, Jul 16 2026; see demo_setup_validated.md §5) ─
mkdir -p "$HOME/.grok"
cat > "$HOME/.grok/config.toml" <<'EOF'
[models]
default = "qwen-local"

[model.qwen-local]
model = "Qwen3.6-27B"
base_url = "http://localhost:8000/v1"
api_key = "none"
name = "Qwen3.6-27B (vLLM)"
context_window = 32768

# Internal harness calls request the model named "grok-build" explicitly.
# Redirect it locally. api_backend must be chat_completions: the built-in
# defaults to the Responses API, which crashes on vLLM reasoning stream events.
[model.grok-build]
model = "Qwen3.6-27B"
base_url = "http://localhost:8000/v1"
api_key = "none"
api_backend = "chat_completions"
context_window = 32768

[features]
telemetry = false
EOF

echo "Grok Build ready. Config written to ~/.grok/config.toml"
echo "Smoke test:"
echo '  grok -p "create hello.py that prints hello, then run it" -m qwen-local \'
echo '    --yolo --disable-web-search --cwd /workspace/scratch --output-format json'
