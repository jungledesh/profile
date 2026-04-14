# profile — vLLM Inference Profiler

**Less words. More insight.**  
Find and fix inference bottlenecks in minutes. 

## What it does

`profile` turns raw vLLM + GPU metrics into **clear diagnosis and actionable fixes**.

- Samples GPU + vLLM signals every **250ms**
- Supports instant snapshots and longer analysis (`--duration 30s | 1m | 5m | ...`)
- Detects real production bottlenecks:
  - **Under-batching** — GPU has headroom, but scheduler occupancy is too low
  - **KV Cache Pressure** — KV usage near capacity → eviction risk
  - **Low Prefix Cache reuse** — prompts don’t share context → wasted performance

It tells you **what’s wrong, why it’s happening, and what to fix first**, so you can reduce cost per token.

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

This is **v0.1.0** — currently optimized for single-GPU setups.

Feedback and real-world usage are highly valuable.  
A deeper technical write-up is coming soon.