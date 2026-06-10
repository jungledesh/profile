# Profile

Less Words. Less Noise. More Signal. More Value.

A physics-grounded, cost-aware optimizer for vLLM inference servers.

---

## Profile vs other tools

| | Profile | Other tools |
| ------------------------------------ | ------- | ----------- |
| Physics ceiling (roofline math) | ✓ | ✗ |
| Filters idle, only analyzes under load | ✓ | ✗ |
| Bottleneck detection | ✓ | ✓ |
| Closed loop: measures delta after fix | ✓ | ✗ |
| Cost per 1M tokens + recoverable waste | ✓ | ✗ |
| Prescriptive fixes, not just alerts | ✓ | ✗ |
| GPU metrics | ✓ | ✓ |
| Prometheus `/metrics` | ✓ | ✓ |

---

## How it works

Profile watches your vLLM server under load. Computes the roofline for your exact model and GPU, shows where you stand against that ceiling, and tells you what to fix. Apply a change, Profile re-measures, reports the exact delta. Every recommendation is accountable.

---

## Value in a minute

### Download

```bash
curl -L https://github.com/jungledesh/profile/releases/latest/download/profile -o profile
chmod +x profile && mv profile /usr/local/bin/
```

### Or build from source

```bash
cargo install --git https://github.com/jungledesh/profile
```

### Run

```bash
profile diagnose --url http://localhost:8000/metrics
```

---

## Configuration

```bash
profile diagnose [flags]
```

**Requires:** vLLM instance exporting Prometheus metrics at `/metrics` and NVIDIA Linux runtime with NVML (`libnvidia-ml.so`) access.

| Flag                       | Default                         | Description                                          |
| -------------------------- | ------------------------------- | ---------------------------------------------------- |
| `-u, --url`                | `http://localhost:8000/metrics` | vLLM metrics endpoint                                |
| `--duration`               | `30s`                           | Sampling window (`30s`, `2m`, `5m`)                  |
| `-m, --max-num-seqs`       | auto-detected                   | Read from `/metrics`; prompted if absent             |
| `--tensor-parallel-size`   | env / unset                     | TP degree (overrides `TENSOR_PARALLEL_SIZE`)         |
| `--cost-per-hour`          | catalog estimate                | GPU cost in USD/hr (overrides catalog estimate)      |
| `-v`                       | off                             | Show non-triggered rules and physics limits          |

---

## Sample output

```
$ ./profile diagnose --duration 2m -v

Profile needs max_num_seqs for accurate diagnosis.

Find it in your vLLM start command: --max-num-seqs N (default: 256 if not set).
Enter value: 32

+--------------------------------------------------------------------------------------------------+
|PROFILE v2.1.0 [meta-llama/Llama-3.1-8B-Instruct] [NVIDIA H100 80GB HBM3] (2m from 2026-06-03) |
|                                                                                                  |
|GPU =>      EFFICIENCY 36.2% | POWER 312W | 0.20 J/tok | $0.61/1M tok (est) | vRAM 62/80GB      |
|                                                                                                  |
|vLLM:                                                                                             |
|REQUESTS   run 1 (3.1%) | wait 1 | max 32                                                         |
|LATENCY    ttft 420ms | tpot 35ms                                                                 |
|CACHE      kv_cache 71.2% avg | pfix_cache 52.4%                                                  |
|THROUGHPUT 1580 tok/s                                                                             |
|TRAFFIC    qps 12.4 | req_total 1488 | gen_total 189600 | preempt/s 0.00 | preempt_total 0        |
|                                                                                                  |
|ISSUES:                                                                                           |
|                                                                                                  |
|[!] Under-batching — Insufficient Concurrency                                                     |
|  Seen in 60% of windows                                                                          |
|                                                                                                  |
|  Occupancy  3.1%  (threshold: < 25%)                                                             |
|  Requests   1 running, 1 waiting  (max: 32)                                                      |
|                                                                                                  |
|  Cause:                                                                                          |
|    Hardware capacity under-fed by client. Not enough requests arriving to keep the server busy.  |
|                                                                                                  |
|  Fix:                                                                                            |
|    • Batch more requests or increase client concurrency (31 slots idle)                            |
|                                                                                                  |
|  Expected: Higher throughput, stable TPOT.                                                       |
|  Confidence: High                                                                                |
|                                                                                                  |
|At current efficiency, ~64% of compute cost is wasted — ~$2.20/hr recoverable.                   |
|                                                                                                  |
|KV cache pressure: not triggered                                                                  |
|OOM risk: not triggered                                                                           |
|Low prefix reuse: not triggered                                                                   |
|Concurrency saturation: not triggered                                                             |
+--------------------------------------------------------------------------------------------------+

Apply your change to vLLM.
Profile will detect when vLLM restarts automatically.
Press Enter to skip and re-measure now.
```

After you apply a fix:

```
Waiting for vLLM to restart...
Connection restored. Resuming in 5s...

Measuring delta...
  Throughput  1580 → 2990 tok/s ↑
  TTFT        4823 → 1204ms ↓  (p99 9847 → 2156ms ↓)
  Efficiency  +32.4pp ↑

ECONOMICS:
  Cost/1M tok   $0.61 → $0.32 ↓ (est)
  Recoverable   $2.20 → $1.08/hr ↓

Direction: Better

No issues detected. Efficiency: 68.6% of hardware ceiling.
```

---

## Bottlenecks detected

- **Under-batching**: hardware under-fed by client; compute headroom wasted
- **KV cache pressure**: VRAM near capacity, preemption risk rising
- **Low prefix reuse**: prefix cache hit rate too low; prefill compute wasted
- **Concurrency saturation**: `max_num_seqs` cap hit; requests queueing, TTFT degrading
- **OOM risk**: model weight footprint structurally exceeds available VRAM

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

Apache License 2.0. See [LICENSE](LICENSE).

```
Copyright 2026 Gagandeep Singh
```

For production teams requiring cluster-wide aggregation, multi-engine support, or custom hardware cataloging: **[jungledesh@gmail.com](mailto:jungledesh@gmail.com)**
