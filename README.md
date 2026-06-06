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
profile diagnose --url http://localhost:8000/metrics
```

Or build from source & run:

```bash
cargo install --git https://github.com/jungledesh/profile
profile diagnose --url http://localhost:8000/metrics
```

### 2. Live Interactive Optimization

```
Measuring delta...
  Throughput      6125 → 6084 tok/s ↓
  TTFT            36685 → 37849ms ↑
  Efficiency      -0.2pp ↓

ECONOMICS:
  Cost/1M tok     $0.16 → $0.16 (est)
  Recoverable     $2.38 → $2.39/hr ↑

No significant change.
Apply fix: raise --max-num-seqs above 64, then re-measure.

Apply your change to vLLM.
Profile will detect when vLLM restarts automatically.
Press Enter to skip and re-measure now.
```

---

## Production Bottlenecks Detected

- **Concurrency Saturation:** `max_num_seqs` cap hit under high load. Requests stack in the queue, destroying TTFT.
- **KV Cache Pressure:** VRAM near capacity, driving high token-eviction and preemption risks.
- **Under-Batching:** Upstream orchestration or client failure leaving massive GPU compute headroom unutilized.
- **OOM Risk:** Model weight configuration structurally exceeds physical VRAM topology before execution begins.

---

## Configuration Reference

```bash
./profile diagnose [flags]
```


| Flag              | Default                         | Description                                                   |
| ----------------- | ------------------------------- | ------------------------------------------------------------- |
| `--url`           | `http://localhost:8000/metrics` | Target vLLM metrics endpoint                                  |
| `--duration`      | `30s`                           | Sampling window (e.g., `30s`, `2m`, `5m`)                     |
| `-m`              | `256`                           | Fallback value for `max_num_seqs` if absent from metrics      |
| `--cost-per-hour` | catalog estimate                | Exact host instance rate for raw dollar tracking              |
| `-v`              | off                             | Verbose mode. Exposes non-triggered rules and physical limits |


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