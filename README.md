# profile — vLLM Inference Diagnoser

Find and fix inference bottlenecks. Know what your hardware is capable of, and whether you're getting it.

**v2 is work in progress.**

---

## What it does

Samples GPU + vLLM signals every 250ms. Computes a physics baseline (roofline). Runs gap analysis. Tells you what's wrong, why, and what to fix first.

- Filters idle windows, reports only under-load behavior
- Every number is hardware-referenced: your GPU, your model, your actual ceiling
- Aggregated across your observation window, not a single noisy sample

**Detects real production bottlenecks:**

- **Under-batching** — GPU has headroom, scheduler occupancy too low
- **KV Cache Pressure** — KV usage near capacity, eviction risk
- **Concurrency Saturation** — scheduler pinned at `max_num_seqs` cap, queue building

---

## Example Output

```
+------------------------------------------------------------------------------------------+
|PROFILE v2.0.0 [llama3] [NVIDIA H200] (2m from 2026-06-03 22:25:45 UTC)                  |
|GPU =>     EFFICIENCY 31.7% | POWER 421W | vRAM 70/140GB                                  |
|                                                                                          |
|vLLM:                                                                                     |
|REQUESTS   run 63 (98.0%) | wait 25 | max 64                                              |
|LATENCY    ttft 656ms | tpot 11ms                                                         |
|PROMPT     kv_cache 1.1% avg                                                              |
|THROUGHPUT 5970 tok/s | pfix_cache 88.9%                                                  |
|TRAFFIC    qps 46.5 | req_total 6336 | gen_total 812499 | preempt/s 0.00 | preempt_total 0|
|                                                                                          |
|ISSUES:                                                                                   |
|                                                                                          |
|[!] Concurrency Saturation                                                                |
|  Seen in 58% of windows                                                                  |
|                                                                                          |
|  Cause:                                                                                  |
|    • --max-num-seqs=64 hit: scheduler won't admit more sequences                         |
|    • 33% of requests waiting (32 of 96 active)                                           |
|    • TTFT 0.6s                                                                           |
|                                                                                          |
|  Fix:                                                                                    |
|    • Raise --max-num-seqs above 64 (KV cache 1% used, pool has room)                     |
|                                                                                          |
|  Expected: Queue drains, TTFT recovers.                                                  |
|  Confidence: High                                                                        |
|                                                                                          |
|No issues for Under-batching and KV cache pressure                                        |
+------------------------------------------------------------------------------------------+
```

---

## Installation

**Linux binary**
Download from the [latest release](https://github.com/jungledesh/profile/releases).

```bash
chmod +x profile
./profile diagnose --url http://localhost:8000/metrics --duration 5m
```

**Cargo**

```bash
cargo install --git https://github.com/jungledesh/profile
```

---

## Usage

```bash
# Instant snapshot
./profile diagnose --url http://localhost:8000/metrics

# Recommended: 5-minute window
./profile diagnose --url http://localhost:8000/metrics --duration 5m

# Verbose mode
./profile -v diagnose --url http://localhost:8000/metrics --duration 5m
```

---

Built to make vLLM inference diagnostics predictable, truthful, and actionable.  
Feedback and real-world usage are welcome.

---

## License

**Profile** is open source software licensed under the **Apache License 2.0**.  
See the full [LICENSE](LICENSE) file for details.

```
Copyright 2026 Gagandeep Singh

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Commercial Use

The core of Profile is and will remain fully open source under the Apache 2.0 license.  

If you're using Profile in production or want to discuss commercial licensing, support contracts, or custom development, reach out: **jungledesh@gmail.com**
