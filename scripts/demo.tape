# Profile v2 — Demo Recording
# Scenario: Coding assistant · Qwen2.5-Coder-7B-Instruct · H100
#
# Prerequisites before hitting `vhs demo.tape`:
#   1. vLLM running in tmux session "vllm" (demo.sh)
#   2. Load running: MODEL=qwen-coder MODE=demo ./load.sh &
#   3. profile binary on PATH
#
# The tape runs profile inside tmux window 0, then opens window 1 to apply
# the fix, then switches back. vLLM restart is detected automatically.

Output    demo.gif
Set Theme "Catppuccin Mocha"
Set FontSize 13
Set Width 1200
Set Height 700
Set FontFamily "JetBrains Mono"
Set TypingSpeed 50ms
Set Padding 20

# ── Start a tmux session for the demo ────────────────────────────────────────
Type "tmux new-session -s demo"
Enter
Sleep 1s

# ── Window 0: run profile ─────────────────────────────────────────────────────
Type "profile diagnose --url http://localhost:8000/metrics --duration 1m -m 32"
Enter

# Collection window (1m) + analysis
Sleep 1m15s

# PHYSICS block + R5 fires. Profile prints:
# "Apply your change to vLLM. Profile will detect when vLLM restarts."
Sleep 4s

# ── Window 1: apply the fix ───────────────────────────────────────────────────
Ctrl+b
c
Sleep 500ms

Type "./fix_r5.sh"
Enter

# vLLM shuts down and restarts (~60s on H100 for Qwen2.5-Coder-7B)
Sleep 70s

# ── Back to profile ───────────────────────────────────────────────────────────
Ctrl+b
0
Sleep 500ms

# Profile prints "Connection restored. Resuming in 5s..." then re-measures
Sleep 1m15s

# Delta block: throughput ↑, TTFT ↓, Efficiency +Xpp
# R3 may fire (low prefix hit rate) or "No issues detected"
Sleep 5s
