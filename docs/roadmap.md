# Profile

**Pain → Metric → Diagnosis → Cause → Evidence → Fix**

- Metrics (and Waste) surface the pain
- Top Issues rank what matters
- Each issue carries Diagnosis, Cause, and Evidence
- Confidence qualifies certainty
- Recommended Plan is the prioritized Fix (Impact Transparency + Basis)
- Do Nothing is the clean case when no major issue applies

**Metric integrity:** Ship a number only when the metric’s documented meaning equals the claim; otherwise omit or label “unavailable.”

---

## Value in 1 minute

1-minute install → metrics, issues, and fixes

---

## Summary

| Version | Focus | Minute 1 Value Delivered | Rules in Engine | Question Answered | MVP Priority |
|---------|--------|--------------------------|-----------------|-------------------|--------------|
| **v1** | Full Core Diagnostics | Complete trusted story (4 surface rules + do-nothing) | 5 rules | Is my cluster inefficient? | MVP (Week 1–2) |
| **v2** | Full Core + Pipeline Depth | Complete story + prefill/decode rules | 7 rules | Where is the bottleneck? | MVP (Week 1–2) |
| **v4** | Full Core + Scheduler Depth | Complete story + batching rules | 8 rules | Why aren't we batching effectively? | MVP (Week 2–3) |
| **v3** | Full Core + Memory Depth | Complete story + KV/memory rules | 10 rules | Why is memory wasting / KV inefficient? | Post-MVP |
| **v5** | Full Core + Hardware Physics | Complete story + low-level hardware rules | 12 rules | Deeper hardware limitation? | Post-MVP |
| **v6** | Full Core + Richest Optimization Engine | Complete story + all rules + quantization sensitivity | All rules | How do I fix it? | MVP core (Week 2–4) |
| **v7** | Full Core + Integrations | Complete story + exportable JSON/OTLP fix blobs | All rules | Production fit? | Post-MVP |
| **v8** | Full Core + Autonomous | Complete story + self-optimizing fixes | All rules | Can it self-optimize? | Future |
| **v9** | Full Core + Cluster Level | Complete story + cross-node aggregation & imbalance rules | +6 cluster rules | Why is my cluster underperforming at scale? | post-v8 |

---

**Note:** For vLLM, data comes from **vLLM `/metrics`** + **DCGM/NVIDIA-SMI** (per node). Cluster mode aggregates via config file.

---

## Output Format

```text
=== PROFILE DIAGNOSE ===

Metrics:
GPU util: 29%
TPS: 420
TTFT: 1.9s
...

Waste:
~70% GPU idle

Top Issues (ranked):
1. Batch collapse (High confidence: 0.91)
2. Prefill overhead (Medium: 0.63)

Diagnosis:
Requests are not batching effectively

Cause:
GPU underutilized — batch occupancy stays low relative to capacity

Evidence:
- Avg batch size: 2 / 16
- Queue delay: 0ms

Recommended Plan:
[1] Enable continuous batching
   Impact: +40–60% TPS (typical), up to +100% in ideal batching conditions
   Basis: Batch scaling curve + current utilization

[2] Enable prefix caching
   Impact: TTFT -30–45% (typical), up to -60% in high-reuse workloads

---
Status: Issues detected (or “No major inefficiencies detected”)
```

*(Cluster mode in v9 adds aggregate view + per-node drill-down + cross-node issues.)*

---

## v1 — Full Core Diagnostics [CLI + TUI]

**Goal:** One command → complete, trusted, production-ready story.

### Work

- **CLI:** `profile diagnose --url http://localhost:8000` (Rust binary via Maturin → `pip install profile`)
- Scrape vLLM Prometheus + DCGM
- Central decision engine with **exactly the 5 rules** below
- Unified TUI + live refresh + `--export json`

### vLLM `/metrics` scrape list (v1)

**Authority:** `# HELP` (and `# TYPE`) on the **live** `/metrics` response for the scraped version override web docs when wording differs.

| # | Metric | Type | Use in Profile |
|---|--------|------|------------------|
| 1 | `vllm:num_requests_running` | Gauge | HELP defines this as *Number of requests in model execution batches.* Treat each scrape as **one instant** of batch occupancy; compare to `max_num_seqs` in batch-collapse logic. **Evidence / UI copy:** use that HELP wording verbatim. (Time-averaged “batch size” only if you explicitly average across scrapes.) |
| 2 | `vllm:num_requests_waiting` | Gauge | Queue pressure (per HELP on scrape). |
| 3 | `vllm:time_to_first_token_seconds` | Histogram | TTFT (sum/count or quantiles per parsing rules). |
| 4 | `vllm:request_queue_time_seconds` | Histogram | Queue delay. |
| 5 | `vllm:request_prefill_time_seconds` | Histogram | Prefill latency. |
| 6 | `vllm:generation_tokens_total` | Counter | Output TPS = Δcounter / Δt over a **fixed** scrape/window definition. |
| 7 | `vllm:prefix_cache_hits` | Counter | Prefix reuse numerator (if exposed; else unavailable). |
| 8 | `vllm:prefix_cache_queries` | Counter | Prefix reuse denominator (if exposed; else unavailable). |
| 9 | `max_num_seqs` | Config / gauge | Configured max concurrent seqs (prefer startup config or documented API; scrape only if HELP proves a gauge). |
| 10 | `vllm:request_prompt_tokens` | Histogram | Prompt tokens **per request** (per HELP on scrape). With **5** + **3**, supports high-confidence prefill story: prefill/TTFT **and** long-prompt distribution (e.g. mass above 512). Exact series name must match target binary. |

### GPU metrics (v1)

From **NVML** on the node (not `/metrics`). Same integrity rule: only claim what the API reports.

| # | Signal | Use in Profile |
|---|--------|----------------|
| G1 | GPU util % | Batch collapse, low util, do-nothing |
| G2 | Power draw (W) | Low-GPU-util evidence; optional ratio to limit |
| G3 | Power limit (W) | Denominator for “% of cap” (not marketing TDP) |
| G4 | VRAM used / total | Context / evidence where useful |

**Not scraped:** `expected_baseline(model, gpu, avg_batch)` — calibrated table inside Profile, not a Prometheus series.

### Rule engine

#### 1. Batch collapse

| Field | Detail |
|-------|--------|
| **Condition** | `avg_batch_size < 0.5 × max_num_seqs` **and** `gpu_util < 50%` |
| **Confidence** | `0.95 - (avg_batch_size / max_num_seqs)` (clamped 0.6–0.95) |
| **Evidence** | Avg batch size, queue delay, GPU util |
| **Fix** | Enable continuous batching + 15 ms window |
| **Impact** | +40–60% TPS (typical), up to +100% in ideal batching conditions |
| **Basis** | Batch scaling curve on A100/H100-class GPUs |

#### 2. Low GPU utilization (idle compute)

| Field | Detail |
|-------|--------|
| **Condition** | `gpu_util < 40%` for >30s **and** `queue_delay < 50ms` |
| **Confidence** | 0.92 (if power < 60% TDP) |
| **Evidence** | GPU util, power draw, TPS vs baseline |
| **Fix** | Enable continuous batching |
| **Impact** | GPU util → 55–70% (typical), TPS +35–55% (typical), up to +80% in ideal conditions |
| **Basis** | vLLM production benchmarks (same hardware class) |

#### 3. Prefill bottleneck

| Field | Detail |
|-------|--------|
| **Condition** | `prefill_latency / TTFT > 0.7` **and** `prompt_tokens > 512` |
| **Confidence** | 0.88 (if prefix reuse < 10%) |
| **Evidence** | Prefill % of TTFT, prompt tokens |
| **Fix** | Enable `--enable-prefix-caching` |
| **Impact** | TTFT -30–45% (typical), up to -60% in high-reuse workloads |
| **Basis** | Prefix-cache hit-rate scaling from vLLM logs |

#### 4. Low throughput vs baseline

| Field | Detail |
|-------|--------|
| **Condition** | `tps < 0.65 × expected_baseline(model, gpu, avg_batch)` |
| **Confidence** | 0.82 |
| **Evidence** | Current TPS vs baseline gap |
| **Fix** | Prioritized batching + prefix caching |
| **Impact** | TPS +40–70% (typical), up to +120% in ideal conditions |
| **Basis** | Calibrated on 12 public vLLM benchmarks |

#### 5. Do-nothing / optimal case (always last)

| Field | Detail |
|-------|--------|
| **Condition** | No rules trigger **and** `gpu_util > 65%` **and** `tps > 0.85 × baseline` |
| **Output** | “No major inefficiencies detected. System is near optimal. Monitor under 2× load.” |

_Note — Issue(s) handling: rules ranked by **confidence × estimated_impact**. Top **2–3** shown in Recommended Plan._

---

## v2–v9 — Same engine, same output

**v2 through v9** are unchanged in product shape: each version **extends the same central decision engine** and uses the **same output format**. Higher versions **add rules**; they do not fork the UX or duplicate vLLM.

---

## v2 — Full Core + Pipeline Depth

**Goal:** Same output + deeper prefill/decode rules added to engine.

### Work

Add **2 pipeline rules** to the shared engine. Same unified TUI.

---

## v4 — Full Core + Scheduler Depth

**Goal:** Same output + explicit batching rules.

### Work

Add **scheduler rules** to engine. Same unified output.

---

## v3 — Full Core + Memory Depth (post-MVP)

**Goal:** Same output + KV fragmentation & prefix reuse rules.

### Work

Extend engine (small vLLM PR recommended for clean metrics). Same unified TUI.

---

## v5 — Full Core + Hardware Physics (post-MVP)

**Goal:** Same output + arithmetic intensity / memory-wall rules (DCGM).

### Work

Extend engine with DCGM data. Optional `--advanced` flag.

---

## v6 — Full Core + Richest Optimization Engine

**Goal:** Same output powered by the **complete** engine.

### Work

Central decision-tree + **quantization sensitivity** layer (J/req savings vs accuracy risk). All original examples now live here with updated impact phrasing.

---

## v7 — Full Core + Production Integrations (post-MVP)

**Goal:** Same output + export enriched blobs.

### Work

`profile export` otlp/prometheus/json with full `{metrics, waste, diagnosis, cause, evidence, action, impact, basis, confidence}`.

**Cause** is its own key (not folded into `diagnosis`) so Grafana, agents, and ticket systems can show, route on, and store “what we think is wrong” vs “why it’s happening” without splitting one text field.

---

## v8 — Full Core + Autonomous (future)

**Goal:** Same output + safe auto-apply loop (with `--dry-run` first).

---

## v9 — Full Core + Cluster Level Diagnostics (post-v8)

**Goal:** Same complete trusted story + cross-node / cluster-scale causes & fixes.

### Work

- **New CLI:** `profile diagnose --cluster config.yaml` (or `--nodes node1:8000,node2:8000`)
- Aggregate metrics across nodes
- Add **6 new cluster rules** to the shared engine
- Unified TUI shows cluster summary + per-node drill-down

### Cluster-specific rules (added in v9)

| Rule | Condition / notes | Fix / impact |
|------|-------------------|--------------|
| **Load imbalance** | TPS variance > 25% across nodes | **Fix:** Adjust router weights / enable consistent hashing. **Impact:** Cluster TPS +25–45% (typical), up to +70% in ideal balanced conditions |
| **Straggler nodes** | One node’s P99 > 2× cluster median | **Impact:** P99 latency -30–50% (typical), up to -70% after straggler elimination |
| **Cross-node routing inefficiency** | Some nodes idle while queue > 0 on others | **Impact:** Overall cluster util +20–40% (typical), up to +60% in ideal routing |
| **Cluster-wide batching collapse** | Aggregate avg batch size across all nodes | **Impact:** Cluster TPS +35–55% (typical), up to +90% in ideal conditions |
| **Prefix / KV inefficiency at scale** | Low global prefix reuse | **Impact:** TTFT -25–40% (typical), up to -55% with cross-node caching |
| **Network / communication bottleneck** | High NCCL/InfiniBand saturation | **Impact:** End-to-end throughput +15–35% (typical), up to +50% after network tuning |

### Example cluster output addition

```text
Cluster Summary (3 nodes):
Node1 TPS: 500   Node2 TPS: 220   Node3 TPS: 480

Top Cluster Issues:
1. Load imbalance (High confidence: 0.93)
   Evidence: TPS variance 45%, Node2 underutilized

Recommended Plan:
[1] Rebalance router (consistent hashing)
   Impact: Cluster TPS +25–45% (typical), up to +70% in ideal balanced conditions
```
