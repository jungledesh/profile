# Profile

Less Words. Less Noise. More Signal. More Value.

A physics-grounded, cost-aware optimizer for vLLM.

Profile measures live telemetry against the absolute physical limits of your hardware to expose waste, surface immediate configuration fixes, and reclaim infrastructure spend.

---

## Design

- **Ground Truth.** Maps live decode and prefill phases against your hardware's exact memory bandwidth and compute ceilings. Efficiency is derived from silicon physics, never estimated. If a metric is missing or telemetry is low-confidence, Profile flags it explicitly, zero artificial certainty.
- **Zero Noise.** Traffic-gated profiling. The analyzer sleeps during idle cycles and only samples when the node is actively crunching tokens. Idle state is not waste; Profile only measures when there is actual work to optimize.
- **Closed Loop.** Tracks configuration changes across container restarts to output precise performance and cost deltas. To prevent alert fatigue, Profile surfaces exactly one high-priority signal per iteration — isolate the bottleneck, deploy the fix, re-measure.

---

## Value in a Minute

### 1. Download

```bash
curl -L https://github.com/jungledesh/profile/releases/latest/download/profile -o profile
```

### 2. Install

```bash
chmod +x profile && mv profile /usr/local/bin/
```

### 3. Run

```bash
profile diagnose --url http://localhost:8000/metrics --duration 2m
```

Or build from source & run:

```bash
cargo install --git https://github.com/jungledesh/profile
profile diagnose --url http://localhost:8000/metrics --duration 2m
```

### 4. Sample Output

Initial diagnose:

```
$ profile diagnose --url http://localhost:8000/metrics --duration 2m -m 256

PHYSICS:
  Model      Meta-Llama-3-8B-Instruct (8.0B params, bf16)
  GPU        NVIDIA H200 SXM (141GB, 4800 GB/s BW, 67 TFLOP/s)
  Decode     282 tok/s ceiling (est) — at 184 tok/s — 65.2% efficient
  Prefill    1,543 tok/s ceiling (est)
  Weights    16.0 GB — 124.6 GB KV headroom
  Floors     TPOT min 3.5ms | Prefill min 0.6ms

───────────────────────────────────────────────────────────────────────────────

[!] Concurrency Saturation

  Seen in 87% of windows

  Cause:
  - max_num_seqs cap (32) hit: 29.4 avg running / 32 max
  - Queue ratio: 67.8% of requests waiting

  Fix:
  • Raise --max-num-seqs above 32 (start: 64–128)
  • Monitor KV usage — keep below 88% after increase
  • Reduce max_tokens if prompts are long

  Expected: Queue drains, TTFT recovers.
  Confidence: High

───────────────────────────────────────────────────────────────────────────────

Apply your change to vLLM.
Profile will detect when vLLM restarts automatically.
Press Enter to skip and re-measure now.
```

After applying the fix, Profile re-measures and reports the delta:

```
Measuring delta...
  Throughput  142 → 281 tok/s ↑
  TTFT        4823 → 1204ms ↓  (p99 9847 → 2156ms ↓)
  Efficiency  +32.4pp ↑

ECONOMICS:
  Cost/1M tok   $0.42 → $0.21 ↓ (est)
  Recoverable   $1.84 → $0.92/hr ↓

Direction: Better

No issues detected. Efficiency: 65.2% of hardware ceiling.
```

---

## Production Bottlenecks Detected

- **Concurrency Saturation:** `max_num_seqs` cap hit under high load. Requests stack in the queue, destroying TTFT.
- **KV Cache Pressure:** VRAM near capacity, driving high token-eviction and preemption risks.
- **Low Prefix Reuse:** Prefix cache hit rate too low under load — prompts share too little common prefix, wasting prefill compute and inflating TTFT.
- **Under-Batching:** Upstream orchestration or client failure leaving massive GPU compute headroom unutilized.
- **OOM Risk:** Model weight configuration structurally exceeds physical VRAM topology before execution begins.

---

## Configuration Reference

```bash
./profile diagnose [flags]
```


| Flag                     | Default                         | Description                                                   |
| ------------------------ | ------------------------------- | ------------------------------------------------------------- |
| `--url`                  | `http://localhost:8000/metrics` | Target vLLM metrics endpoint                                  |
| `--duration`             | `30s`                           | Sampling window (e.g., `30s`, `2m`, `5m`)                     |
| `-m`                     | `256`                           | Fallback value for `max_num_seqs` if absent from metrics      |
| `--tensor-parallel-size` | env / unset                     | Tensor parallel degree (overrides `TENSOR_PARALLEL_SIZE`)     |
| `--cost-per-hour`        | catalog estimate                | Exact host instance rate for raw dollar tracking              |
| `-v`                     | off                             | Verbose mode. Exposes non-triggered rules and physical limits |


---

## Environment Requirements

- vLLM instance exporting standard Prometheus metrics
- NVIDIA Linux runtime with native NVML (`libnvidia-ml.so`) access

---

## License

Profile is open source under the **Apache License 2.0**.
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

For production teams requiring cluster-wide aggregation, specialized hardware cataloging, or custom architecture support: **[jungledesh@gmail.com](mailto:jungledesh@gmail.com)**