# profile — vLLM Inference Diagnoser

**Less words. More insights.**  
Find and fix inference bottlenecks in minutes. 

## What it does

`profile` turns raw vLLM + GPU metrics into **clear diagnosis and actionable fixes**.

- Samples GPU + vLLM signals every **250ms**
- Supports instant snapshots and longer analysis (`--duration 30s | 1m | 5m | ...`)
- Detects real production bottlenecks:
  - **Under-batching** — GPU has headroom, but scheduler occupancy is too low
  - **KV Cache Pressure** — KV usage near capacity → eviction risk
  - **Low Prefix Cache reuse** — prompts don’t share context → wasted performance
  - **Parallelism mismatch** — model weights exceed single-GPU VRAM but TP not set
  - **Concurrency Saturation** — scheduler pinned at max_num_seqs cap with queue building

It tells you **what’s wrong, why it’s happening, and what to fix first**, so you can reduce cost per token.

### Reports 3 kinds of numbers

1. **Under-load behavior only** — profile filters out idle windows and reports only what happens under real traffic. A session-wide average diluted by quiet time hides the problem; profile shows you the character of your server when it's actually working.

2. **Aggregated, not instantaneous** — a single 250ms sample is noise. Profile collects across your full observation window and aggregates correctly: rates as duration-weighted means, latency histograms as true deltas, gauges as last stable value. What you see is representative, not lucky.

3. **Hardware-referenced** — every number is shown against what your specific GPU and model are theoretically capable of. Not a generic benchmark. Not a guess. Your actual decode ceiling, your actual weight footprint, your actual headroom.

## Why use this

vLLM `/metrics` shows numbers.  
`profile` answers:

- Why is my GPU at 50%?
- Why is throughput lower than expected?
- Where am I wasting tokens / memory?

## Installation

**Linux binary (recommended for quick start)**  
Download from the [latest release](https://github.com/jungledesh/profile/releases).

```bash
chmod +x profile
./profile diagnose --url http://localhost:8000/metrics --duration 5m
```

### Cargo

```
cargo install --git https://github.com/jungledesh/profile
```

### Pip package

coming soon 

### Quick Start

```
# Instant snapshot (2s)
./profile diagnose --url http://localhost:8000/metrics

# Recommended: 5-minute analysis
./profile diagnose --url http://localhost:8000/metrics --duration 5m

# Verbose mode
./profile -v diagnose --url http://localhost:8000/metrics --duration 5m
```

## Example Output

```bash
KV Cache Pressure
Seen in 80% of windows

Cause:
- KV usage 93.5% — near capacity
- High concurrency with long sequences

Recommendation:
  • Reduce active sequence count (lower concurrency)
  • Shorten prompts/outputs where possible
  • Use fp8 KV cache (--kv-cache-dtype=fp8)
```

## Development Notes

Built as a focused solo project to make vLLM inference diagnostics  
**predictable, truthful, and actionable**.

This is **v2.0.0** — currently optimized for single-GPU setups.

Feedback and real-world usage are highly valuable.  
A deeper technical write-up is coming soon.