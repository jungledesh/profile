# Profile

Less Words. Less Noise. More Signal. More Value.

A physics-grounded, cost-aware optimization loop for vLLM inference servers.

**[Website](https://jungledesh.github.io/profile/index.html)** | **[Docs](https://jungledesh.github.io/profile/docs.html)**

---

## Profile vs other tools


|                                        | Profile | Other tools |
| -------------------------------------- | ------- | ----------- |
| Physics ceiling (roofline math)        | ✓       | ✗           |
| Filters idle, only analyzes under load | ✓       | ✗           |
| Bottleneck detection                   | ✓       | ✓           |
| Closed loop: measures delta after fix  | ✓       | ✗           |
| Cost per 1M tokens + recoverable waste | ✓       | ✗           |
| Prescriptive fixes, not just alerts    | ✓       | ✗           |


Profile is the first of its kind in the market. We are not just another monitoring tool; we provide actionable intelligence grounded in physics.

---

## The Value

You are paying for hardware. Are you using it?
Profile computes the theoretical physics ceiling for your exact model and GPU, measures your live traffic, and tells you precisely why you are leaving money on the table. Every recommendation is accountable. You apply the fix, Profile measures the delta.

### Real World Impact: Qwen3.6-27B on A100-SXM4-80GB

📺 **[Watch the 15x optimization demo](https://www.youtube.com/watch?v=XuPPKBteWH0)**

**Before Profile:**

- Throughput: `31 tok/s`
- Economics: `$13.26 / 1M tokens`

**After Profile:**

- Throughput: `470 tok/s`
- Economics: `$0.89 / 1M tokens`

**Result:** A **15x throughput increase** and a **93% cost reduction**. Profile tracked the live traffic, dynamically recommended concurrency and model length adjustments, and identified the exact moment the server became structurally saturated. Instead of blindly tweaking configs, Profile advised spinning up a replica to preserve latency.

```text
+----------------------------------------------------------------------------------------------------+
|PROFILE v2.1.4 [Qwen3.6-27B] [NVIDIA A100-SXM4-80GB] (1m from 2026-06-18 22:08:40 UTC)              |
|                                                                                                    |
|GPU =>     EFFICIENCY 8.1% | POWER 390W | 0.83 J/tok | $0.89/1M tok (est) | vRAM 77/80GB (peak 79GB)|
|                                                                                                    |
|vLLM:                                                                                               |
|REQUESTS   run 100 (95.6%) | wait 149 | max 105                                                     |
|LATENCY    ttft 52.9s (p95 129.2s) | tpot 199ms (p95 295ms)                                         |
|CACHE      kv_cache 81.5% avg | pfix_cache 61.6%                                                    |
|THROUGHPUT 470 tok/s                                                                                |
|                                                                                                    |
|ISSUES:                                                                                             |
|                                                                                                    |
|[!] Concurrency Saturation                                                                          |
|  Seen in 50% of windows                                                                            |
|                                                                                                    |
|  Fix:                                                                                              |
|    • KV at 81%: scheduler at cap, pool full. No config change helps.                               |
|    • Add a replica to scale out.                                                                   |
|                                                                                                    |
|~$1.38/hr lost to scheduler queuing                                                                 |
+----------------------------------------------------------------------------------------------------+
```

---

## Install & Run

```bash
# Download
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/jungledesh/profile/releases/latest/download/profile-installer.sh | sh

# Start profiling your vLLM server
profile diagnose --url http://localhost:8000/metrics --duration 1m
```

*Or build from source: `cargo install --git https://github.com/jungledesh/profile`*

---

## Configuration


| Flag                     | Default                         | Description                                                           |
| ------------------------ | ------------------------------- | --------------------------------------------------------------------- |
| `-u, --url`              | `http://localhost:8000/metrics` | vLLM metrics endpoint                                                 |
| `--duration`             | `30s`                           | Sampling window (`30s`, `1m`, `2m`, `3m`)                             |
| `-m, --max-num-seqs`     | Prompted if absent              | Pass directly to skip prompt. Auto-read from `/metrics` if available. |
| `--tensor-parallel-size` | Env or unset                    | TP degree (overrides `TENSOR_PARALLEL_SIZE`)                          |
| `--cost-per-hour`        | Catalog estimate                | GPU cost in USD/hr (overrides catalog estimate)                       |
| `-v`                     | Off                             | Show non-triggered rules and physics limits                           |


---

## What it detects

- **R1 Under-batching (Under-utilized compute)**: `GPU Efficiency < 60%`. Hardware under-fed; compute headroom wasted.
- **R2 KV cache pressure**: `KV Usage ≥ 88%`. VRAM near capacity, preemption risk rising. Dynamically calculates exact sequence length reductions.
- **R3 Low prefix reuse**: `Hit Rate < 35%` at `> 1000 tok/s`. Prefill compute wasted on identical prompts.
- **R4 OOM risk**: Model weight footprint structurally exceeds available VRAM.
- **R5 Concurrency saturation**: `max_num_seqs` cap hit. Requests queueing, TTFT degrading.

---

## Documentation

Comprehensive internals are available in our docs.

- **[Data](https://jungledesh.github.io/profile/docs.html#data)**: How we aggregate 2s windows from parallel NVML and vLLM polls.
- **[Catalog](https://jungledesh.github.io/profile/docs.html#catalog)**: Hardcoded GPU memory bandwidth, FLOPs, and market prices.
- **[Math](https://jungledesh.github.io/profile/docs.html#math)**: The physics formulas powering the efficiency percentage.
- **[Rules](https://jungledesh.github.io/profile/docs.html#rules)**: The precise mathematical conditions that trigger recommendations.
- **[Limitations](https://jungledesh.github.io/profile/docs.html#limitations)**: Where the math is approximate and why.
- **[Design](https://jungledesh.github.io/profile/docs.html#design)**: The philosophy behind the engine.

---

## Crafted, not just engineered.

- Every element in the UI earns its place. If it does not help the user, it is not there.
- Plain language. No jargon, where a plain word works. The goal is to help, not impress.
- Idle windows are ignored. Profile only measures behavior under active load. That is where waste lives.
- Hardware and model agnostic. Roofline math derives limits fresh each run: peak memory bandwidth for decode, peak FLOPs for prefill. No calibration files, no pre-baked assumptions.
- Honest under uncertainty. If a metric is unavailable, it shows `-` and moves on. No fabricated values.
- Prescriptive. Profile tells you what to change and how. Waits while you apply it. Re-measures and reports the exact delta.

---

## License

Apache License 2.0. Copyright 2026 Gagandeep Singh.

For production teams requiring cluster-wide aggregation, multi-engine support, or custom hardware cataloging: **[jungledesh@gmail.com](mailto:jungledesh@gmail.com)**
