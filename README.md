# Profile [ Under Construction ]

A physics-grounded, cost-aware optimization loop for vLLM inference servers.

**The Problem:** Inference servers run below hardware capacity. Operators cannot see why.

**The Solution:** Profile computes the hardware ceiling for your model and GPU, measures live throughput against it, and identifies the vLLM startup flags to change.

**[Website](https://jungledesh.github.io/profile/index.html)** | **[Docs](https://jungledesh.github.io/profile/docs.html)**

---

## How to use Profile

Profile is not a passive dashboard. It is an interactive optimization loop. It analyzes your vLLM `/metrics`, pinpoints the primary bottleneck, and prescribes specific vLLM startup flags to fix it.

### Prerequisites
- NVIDIA GPU with NVML (for hardware metrics).
- vLLM running with `/metrics` reachable (default `http://localhost:8000/metrics`).
- Active production-like load during the `--duration` window. Idle servers produce no signal.

### Launch support

Profile launch supports single-GPU deployments only. Multi-GPU support is on the roadmap.

| GPU memory | Supported models |
| --- | --- |
| 80 GB | Qwen3 32B; Qwen2.5 32B; DeepSeek-R1-Distill 32B; GLM 32B; Gemma 4 31B/27B/26B; Qwen3 30B-A3B; Qwen3.6 27B; Gemma 3 27B; Gemma 2 27B |
| 48 GB | Qwen3 14B; Qwen2.5 14B; DeepSeek-R1-Distill 14B; Phi-4 14B; Gemma 3 12B; Gemma 2 9B |
| 24 GB | Llama 3 8B; Qwen3 8B/4B/1.7B/0.6B; Nemotron 8B; DeepSeek-R1-Distill 8B/7B/1.5B; Qwen2.5 7B; DeepSeek 7B; Mistral 7B; Gemma 3 4B/1B |

Context limits depend on KV dtype and runtime configuration. With bf16 KV, the 32B dense models on 80 GB support about 18K tokens for one request, and Llama 3 8B on 24 GB supports about 19K tokens.

### Install & Run
```bash
# Download
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/jungledesh/profile/releases/latest/download/profile-installer.sh | sh

# Start profiling your vLLM server
profile diagnose --url http://localhost:8000/metrics --duration 2m
```

*Or build from source: `cargo install --git https://github.com/jungledesh/profile`*

### The Optimization Loop
When sampling ends, Profile prints a summary block. Look at `ISSUES`. The `Fix:` tells you what to change in your vLLM startup command.

*Read*
Profile prints a performance snapshot followed by an `ISSUES` block. Here is an example issue. Do exactly what the `Fix:` section recommends.
```text
+----------------------------------------------------------------------------------------------------+
|PROFILE v2.1.4 [Qwen3.6-27B] [NVIDIA A100-SXM4-80GB] (1m from 2026-06-18 21:57:54 UTC)              |
|                                                                                                    |
|GPU =>     EFFICIENCY 4.9% | POWER 391W | 1.37 J/tok | $1.46/1M tok (est) | vRAM 71/80GB            |
|                                                                                                    |
|vLLM:                                                                                               |
|REQUESTS   run 53 (20.8%) | wait 196 | max 256                                                      |
|LATENCY    ttft 50.0s (p95 96.0s) | tpot 193ms (p95 292ms)                                          |
|CACHE      kv_cache 98.0% avg (99.7% peak) | pfix_cache -                                           |
|THROUGHPUT 285 tok/s                                                                                |
|TRAFFIC    qps 0.7 | req_total 80 | gen_total 28429 | preempt/s 0.00 | preempt_total 0              |
|                                                                                                    |
|ISSUES:                                                                                             |
|                                                                                                    |
|[!] KV Cache Pressure                                                                               |
|  Seen in 83% of windows                                                                            |
|  Cause:                                                                                            |
|  - KV cache hit 99.7% peak (threshold: 88%)                                                        |
|  - Queue backpressure: 196 requests waiting on KV admission                                        |
|                                                                                                    |
|  Fix:                                                                                              |
|    • Lower --max-num-seqs to ≤13 (physics ceiling for max_model_len=8192)                          |
|    • Raise --gpu-memory-utilization (check VRAM header for avail mem) to expand KV pool            |
|    • Enable --enable-prefix-caching to share KV blocks across identical prompt prefixes            |
|    • Switch --kv-cache-dtype fp8 to halve KV memory footprint                                      |
|    • Lower --max-model-len (current: 8192) to safely raise concurrency.                            |
|                                                                                                    |
|  Expected: Wait queue drains, TTFT recovers once KV pool has capacity.                             |
|  Confidence: High                                                                                  |
+----------------------------------------------------------------------------------------------------+
```

> *Note: `[!] KV Cache Pressure` corresponds directly to rule **R2** in the Flag Recommendations Map below.*

*Restart & Measure*
Apply primary and secondary fixes together in one restart. One flag per restart wastes time. Restart vLLM. Profile resumes on vLLM re-start and measures the new baseline. Repeat this process until the bottleneck clears or you reach hardware saturation.

```text
Apply your change. Press Enter when done.
Connection restored. Resuming in 5s...

New --max-num-seqs [current: 256]: 13

Measuring delta...

  Config changed, baseline reset.

  Throughput   285 -> 133 tok/s ↓
  TTFT         50046 -> 57230ms ↑  (p95 96000 -> 78000ms ↓)
  TPOT         193.2 -> 75.6ms ↓   (p95 292.3 -> 97.5ms ↓)
  Efficiency   -2.6pp ↓

ECONOMICS:
  J/tok        1.37 -> 2.14 ↑
  Cost/1M tok  $1.46 -> $3.14 ↑ (est)
  Waste        $1.43 -> $1.47/hr
```

*Iterate*
Notice the throughput dropped? Profile reports regressions honestly. Fixing one bottleneck often exposes the next. Keep iterating, results improve over multiple restarts.

*Scale Out*
Eventually, no config change will help. You have hit hardware saturation. Time to scale out.
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

### vLLM Flag Recommendations Map
Profile detects these bottlenecks and recommends the following vLLM flag changes:

| Diagnosis                     | When it fires                                      | vLLM flags to change                                                                                                              |
| ----------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **R1 Under-batching**         | GPU efficiency <60%, no queue                      | Increase client concurrency (not a vLLM flag)                                                                                     |
| **R2 KV cache pressure**      | KV ≥88%, preemptions, or admission backlog         | Lower `--max-num-seqs` or `--max-model-len`; or raise `--gpu-memory-utilization`, switch to fp8 KV cache                          |
| **R3 Low prefix reuse**       | Prefix hit rate <35%                               | Add `--enable-prefix-caching`; restructure prompts if already enabled                                                             |
| **R4 OOM risk**               | Weights exceed VRAM                                | Set `--tensor-parallel-size` (Profile computes the value)                                                                         |
| **R5 Concurrency saturation** | Queueing, running at `--max-num-seqs` cap            | Raise `--max-num-seqs` if KV <80%; otherwise add a replica or lower `--max-model-len`                                             |

> *Note: Profile may list several fixes in one Fix: block. Apply them together when relevant. See [Rules](https://jungledesh.github.io/profile/docs.html#rules) for thresholds and edge cases.*

### Profile CLI Configuration
These are flags for the `profile` CLI itself, *not* vLLM.

| Flag                     | Default                         | Description                                                  |
| ------------------------ | ------------------------------- | ------------------------------------------------------------ |
| `-u, --url`              | `http://localhost:8000/metrics` | vLLM metrics endpoint                                        |
| `--duration`             | `30s`                           | Sampling window (`30s`, `1m`, `2m`, `3m`)                    |
| `-m, --max-num-seqs`     | Prompted if absent              | Pass to skip prompt. Auto-read from `/metrics` if available. |
| `--tensor-parallel-size` | Env or unset                    | TP degree (overrides `TENSOR_PARALLEL_SIZE`)                 |
| `--cost-per-hour`        | Catalog estimate                | GPU cost in USD/hr (overrides catalog estimate)              |
| `-v`                     | Off                             | Show non-triggered rules and physics limits                  |

---

## Proof: Qwen3.6-27B on A100-SXM4-80GB

📺 **[Watch the 15x optimization demo](https://www.youtube.com/watch?v=XuPPKBteWH0)**

**Before Profile:** `31 tok/s` | `$13.26 / 1M tokens`  
**After Profile:** `470 tok/s` | `$0.89 / 1M tokens`

**Result:** 15x throughput. 93% cost cut. Profile tracked live traffic and guided specific vLLM config changes (`--max-num-seqs`, prefix caching, FP8 KV cache, `--gpu-memory-utilization`) until hardware saturation was reached.

---

## Why Profile?

Profile provides actionable intelligence grounded in hardware physics to maximize compute utilization, replacing passive metric alerts.

| Feature                                | Profile | Others |
| -------------------------------------- | ------- | ------ |
| Physics ceiling (roofline math)        | ✓       | ✗      |
| Filters idle, only analyzes under load | ✓       | ✗      |
| Bottleneck detection                   | ✓       | ✓      |
| Closed loop: measures delta after fix  | ✓       | ✗      |
| Cost per 1M tokens + recoverable waste | ✓       | ✗      |
| Prescriptive fixes, not just alerts    | ✓       | ✗      |

---

## Documentation

Start with the Workflow, then Rules. The rest is reference material.

- **[Workflow](https://jungledesh.github.io/profile/docs.html#home)**: Usage, output walkthrough, flag mapping.
- **[Rules](https://jungledesh.github.io/profile/docs.html#rules)**: Thresholds, confidence, and edge cases.
- **[Data](https://jungledesh.github.io/profile/docs.html#data)**: Metric sources.
- **[Math](https://jungledesh.github.io/profile/docs.html#math)**: Physics behind efficiency %.
- **[Catalog](https://jungledesh.github.io/profile/docs.html#catalog)**: GPU bandwidth, FLOPs, and prices.
- **[Limitations](https://jungledesh.github.io/profile/docs.html#limitations)**: Where the math is approximate.
- **[Design](https://jungledesh.github.io/profile/docs.html#design)**: Engine design philosophy.

---

## Engineering Principles

- Actionable UI: Elements without direct utility are excluded.
- Plain Language: Documentation avoids unnecessary jargon.
- Hardware Agnostic: Roofline limits are computed dynamically per run without calibration files.
- Honest. Unavailable metrics show `-`. No fabricated values.

---

## License

Apache License 2.0. Copyright 2026 Gagandeep Singh.

Need cluster aggregation, multi-engine, custom hardware catalog, or more support? [Open an issue](https://github.com/jungledesh/profile/issues) or email **[jungledesh@gmail.com](mailto:jungledesh@gmail.com)**
