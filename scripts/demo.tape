# Profile v2 — Demo Recording
# Scenario: Coding assistant · Qwen2.5-Coder-7B-Instruct · H100
#
# Prerequisites before hitting `vhs demo.tape`:
#   1. vLLM running in tmux (demo.sh)
#   2. Load running in a second pane: MODEL=qwen-coder MODE=demo ./load.sh
#   3. profile binary on PATH
#
# During the tape, after R5 fires and the "Apply your change" prompt appears,
# switch to the vLLM tmux pane and restart with --max-num-seqs 128.
# Profile detects the restart automatically — no Enter needed.

Output    demo.gif
Set Theme "Catppuccin Mocha"
Set FontSize 13
Set Width 1200
Set Height 700
Set FontFamily "JetBrains Mono"
Set TypingSpeed 50ms
Set Padding 20

# ── Run profile ───────────────────────────────────────────────────────────────
Type "profile diagnose --url http://localhost:8000/metrics --duration 1m -m 32"
Enter

# Collection window (1m) + analysis
Sleep 1m15s

# Output appears: PHYSICS block + R5 fires (concurrency saturation)
# Profile prints "Apply your change to vLLM. Profile will detect when vLLM restarts."
Sleep 4s

# ── Apply fix (off-tape: restart vLLM with --max-num-seqs 128) ────────────────
# Profile prints "Waiting for vLLM to restart..."
# vLLM comes back up (~60s for Qwen2.5-Coder-7B on H100)
# Profile prints "Connection restored. Resuming in 5s..."
Sleep 80s

# ── Second measurement window ─────────────────────────────────────────────────
Sleep 1m15s

# Delta block appears: throughput ↑, TTFT ↓, Efficiency +Xpp
# Either R3 fires (low prefix hit rate) or "No issues detected"
Sleep 5s
