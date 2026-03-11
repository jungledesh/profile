# Profile Roadmap

**Pain → Metric → Action**

---

## Summary

| Version | Focus | Question |
|---------|--------|----------|
| **V1** | Waste Detection | *Is my cluster inefficient?* |
| **V2** | Pipeline Cause | *Where (prefill/decode) is the bottleneck?* |
| **V3** | Memory Diagnostics | *Why is memory wasting / KV inefficient?* |
| **V4** | Scheduler Diagnostics | *Why aren't we batching effectively?* |
| **V5** | Hardware Physics | *Deeper hardware limitation?* |
| **V6** | Optimization Engine | *How do I fix it?* |
| **V7** | Integrations | *Production fit?* |
| **V8** | Autonomous (future) | *Can it self-optimize?* |

**Flow:** Waste → Cause → Deep cause (optional) → Hardware cause (optional) → Fix → Integrate → Automate

---

## V1 — Waste Detection [CLI + TUI]

**Goal:** Reveal obvious inefficiencies in the inference stack.

Should include KV cache and all other version metrics to give users the surface-level symptom of their problem—tempting them to go deeper for more versions (premium tiers).

### Metrics

- Tokens/sec (TPS)
- Time to First Token (TTFT)
- Time per Output Token (TPOT)
- GPU utilization
- GPU power draw
- Cost per token
- Joules / request
- Tokens / watt
- P99
- Cost ≈ `(total_tokens / TPS / 3600) × hourly_rate` (user-supplied)

### Example

| Metric | Value |
|--------|--------|
| Model | Llama-3-8B |
| GPU | A100 |
| Tokens/sec | 420 |
| TTFT | 1.9s |
| TPOT | 85ms |
| GPU Utilization | 29% |
| Power | 310W |
| Cost / 1K tokens | $0.0072 |

**Insight:** GPU utilization extremely low. You are paying for idle compute.

**Purpose of V1 demo:** *Your cluster is inefficient.*

---

## V2 — Pipeline Cause Detection

**Goal:** Identify which stage of inference is inefficient.

### Metrics

- Prefill latency
- Decode latency
- Prefill tokens/sec
- Decode tokens/sec
- Weight load time
- KV cache init time

### Example

| Metric | Value |
|--------|--------|
| Prefill latency | 1.6s |
| Decode latency | 0.3s |

**Insight:** TTFT dominated by prefill stage. Large prompt processing slowing requests.

**Purpose:** *Which stage of inference is inefficient?*

---

## V4 — Scheduler Diagnostics

**Goal:** Identify batching and scheduling inefficiencies.

### Metrics

- Active batch size
- Maximum batch size
- Queue delay
- Batch wait time

### Example A

| Metric | Value |
|--------|--------|
| Average batch size | 2 / max 16 |
| Queue delay | 0ms |
| Batch wait | 0ms |

**Insight:** Batch collapse detected. Requests are not batching together.

### Example B

| Metric | Value |
|--------|--------|
| Queue delay | 450ms |
| Batch wait | 300ms |

**Insight:** Scheduler bottleneck. Requests waiting too long before execution.

**Purpose:** *Why GPUs are underutilized.*

---

## V3 — Memory Diagnostics *(only if demand exists; after V2)*

**Goal:** Diagnose memory inefficiencies.

### Metrics

- KV cache usage
- KV cache fragmentation
- Prefix reuse rate

### Example A

| Metric | Value |
|--------|--------|
| KV cache utilization | 92% |
| KV fragmentation | 38% |
| Prefix reuse rate | 3% |

**Insight:** KV cache fragmentation high. Memory blocks are inefficiently used.

### Example B

| Metric | Value |
|--------|--------|
| Prompt tokens | 1200 |
| Prefix reuse | 0% |

**Insight:** Prefix caching disabled. Large prompts repeatedly recomputed.

**Purpose:** *Explain memory inefficiencies.*

---

## V5 — Hardware Physics (advanced)

**Goal:** Identify low-level GPU bottlenecks.

### Metrics

- HBM bandwidth usage
- Memory stall cycles
- **Arithmetic Intensity** — Formula: *Arithmetic Intensity = Floating Point Operations / Bytes of Memory Accessed*  
  *Why:* Memory-wall detector. If low, proves the customer needs quantization (V6), not more GPUs.
- PCIe/NVLink throughput
- **PCIe/NVLink Saturation**  
  *Why:* For multi-GPU setups, reveals if the bottleneck is the bridge between cards rather than the model.
- GPU clock throttling

### Example A

| Metric | Value |
|--------|--------|
| HBM bandwidth | 92% |
| Memory stall cycles | high |
| SM utilization | 40% |

**Insight:** Memory wall detected. GPU waiting on memory reads.

### Example B

| Metric | Value |
|--------|--------|
| NVLink throughput | 85GB/s |
| PCIe | saturation detected |

**Insight:** Multi-GPU communication bottleneck.

**Purpose:** *Explain hardware-level inefficiencies.*

---

## V6 — Optimization Engine

**Goal:** Turn diagnostics into clear recommendations.

**Structure:** Insight → Diagnosis → Action → Expected Impact

### Example 1: Batch collapse

| Step | Content |
|------|---------|
| **Insight** | GPU utilization = 29%. Tokens/sec below expected baseline. |
| **Diagnosis** | Batch collapse detected. Average batch size = 2 / 16. |
| **Action** | Enable continuous batching. Set batching window to 15ms. |
| **Expected impact** | GPU util: 29% → ~60%. Tokens/sec: +40–50%. Cost/token: -25–35%. |

### Example 2: Prefill-dominated TTFT

| Step | Content |
|------|---------|
| **Insight** | TTFT = 2.1s. Prefill latency dominates. |
| **Diagnosis** | Large prompts repeatedly recomputed. Prefix reuse rate = 0%. |
| **Action** | Enable prefix caching. |
| **Expected impact** | TTFT: -40%. Energy per request: -20%. |

### Example 3: KV / prefix caching

| Step | Content |
|------|---------|
| **Insight** | High KV fragmentation + low prefix reuse. |
| **Diagnosis** | External fragmentation from variable-length requests or disabled prefix caching. |
| **Action** | Enable `--enable-prefix-caching` + tune `--max-num-seqs` / `gpu-memory-utilization` for more KV space. |
| **Expected** | TTFT -30–60%, joules/request -20–40%, effective context 2–3× larger. |

### Example 4: Memory wall + quantization

| Step | Content |
|------|---------|
| **Insight** | Memory wall (low arithmetic intensity + high stall cycles). |
| **Diagnosis** | Decode phase dominated by memory bandwidth, not compute. |
| **Action** | Apply quantization (e.g. FP8/AWQ) + check if continuous batching is enabled. |
| **Expected** | Tokens/sec +50–150%, cost/token -30–50% (with &lt;1–2% accuracy hit on most tasks). |

**Addition: Quantization Sensitivity.** When recommending INT8/FP8, provide a reasoning layer that weighs $J/req$ savings against potential hit to model accuracy.

**Purpose:** *Tell users how to fix inefficiencies.*

---

## V7 — Integrations

**Goal:** Fit into existing observability infrastructure.

Emit **Insight / Diagnosis / Action / Expected** JSON blobs via REST or OTLP so agentic platforms can ingest and act (e.g. auto-tune batch window).

### Export interfaces

- OpenTelemetry
- Prometheus
- StatsD
- JSON / REST API

### Example

```bash
profile export prometheus
profile export otlp
profile export statsd
```

**Prometheus metric example:**

```
profile_gpu_utilization 0.32
profile_tokens_per_second 410
profile_cost_per_1k_tokens 0.007
```

### Usage

- Grafana dashboards
- Datadog alerts
- OpenTelemetry pipelines

**Purpose:** *Allow teams to integrate Profile into production monitoring.*

---

## V8 — Autonomous Optimization (future)

**Goal:** Automatically tune inference configuration.

**Structure:** Insight → Diagnosis → Action → Result

### Example

| Step | Content |
|------|---------|
| **Insight** | GPU utilization = 24%. Queue depth = 15. Average batch size = 1. |
| **Diagnosis** | Batch collapse due to strict latency target. Batch window effectively disabled. |
| **Action** | Batch window increased from 0ms → 12ms. Max batch size set to 16. |
| **Result** | Tokens/sec: +52%. Cost/token: -34%. |

**Purpose:** *Automatically optimize inference systems.*

---

## The Product Narrative

Each version answers a deeper question:

| Version | Question |
|---------|----------|
| **V1** | Is my cluster inefficient? |
| **V2 / V4** | Where is the bottleneck? |
| **V3 / V5** | What deeper system limitation is causing it? |
| **V6** | How do I fix it? |
| **V7** | How do I integrate this into production? |
| **V8** | Can this be optimized automatically? |

This roadmap keeps Profile focused on the core value:

**Reveal waste → Explain cause → Fix inefficiency**
