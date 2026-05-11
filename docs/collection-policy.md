# Ground-truth capture policy (v2)

Normative rules for how Profile combines vLLM `/metrics` scrapes and NVML polls.

**What Profile measures — and what it doesn't.**
Profile reports on the character of the server under load, not the aggregate experience of every request in a session. Latency, throughput, GPU utilization, and cache hit rates are computed over **evaluable windows only** — periods where `num_requests_running > 0.75` OR `tok/s > 20`. Idle periods are excluded entirely from these aggregates.

This is a deliberate design choice: time-weighting across evaluable windows acts as a low-pass filter against volumetric noise. A 1-second burst of 10,000 tiny fast requests gets 1 second of vote; 59 seconds of steady-state 100ms latency keeps its dominance. Profile measures the stability of the provider, not the distribution of the workload that hit it.

**Evaluable window gate:** `window_is_evaluable` is true when there is enough concurrent load or tok/s. Diagnostic aggregates prioritize **what happened under load** while still exposing **true cumulative counters** from the latest collection.

---

## Policy table

| Category | Examples | Single ~2 s window (9 samples) | Multi-window aggregation | Rationale |
|----------|----------|-------------------------------|---------------------------|-----------|
| **State gauges** | `kv_cache_usage_perc`, `num_requests_swapped`, `cpu_cache_usage_perc`, `max_num_seqs`, `temperature_c`, `sm_clock_mhz`, `vram_used_mb` | **Last scrape / last poll** | **Last evaluable window’s last value** | State during the last **active** period (diagnostic focus). |
| **Static / config** | `model_name`, `gpu_name`, `vram_total_mb`, `gpu_index`, `gpu_uuid`, `power_limit_watts`, `cache_config` | **Last scrape** | **Last evaluable window’s last value** | Stable for the lifetime of this run. |
| **Catalog-resolved (model)** | `family`, `param_count`, `active_param_count`, `num_layers`, `hidden_dim`, `is_moe` | **Once at startup** — looked up from `model_name` via `model_catalog::lookup_model` | **Unchanged for run** | Static derivation from model name; `None` on no match. |
| **Catalog-resolved (GPU)** | `arch`, `peak_flops_f32_tflops`, `peak_bw_gbps` | **Once at startup** — looked up from `gpu_name` via `gpu_catalog::lookup_gpu` | **Unchanged for run** | Static derivation from GPU name; `None` on no match. NVML does not expose theoretical peak FLOPS. |
| **Utilization** | `gpu_util_pct`, `mem_util_pct`, `power_watts`, `num_requests_running`, `num_requests_waiting` | **Mean of 9** (last scrape per window for running/waiting) | **Time-weighted mean** (evaluable windows only) | Average load / concurrency over the observed period. |
| **Counters (cumulative)** | `request_success_total`, `num_preemptions_total`, `generation_tokens_total` | **Last scrape (cumulative)** | **Chronologically last collected window** (evaluable or not) | True server cumulative totals at end of collection. |
| **Rates** | `request_success_per_sec`, `num_preemptions_per_sec`, `generation_tokens_per_sec` | **Δ ÷ window duration** | **Time-weighted mean** of per-window rates (**evaluable windows only**) | Effective throughput while active. |
| **Latency / histograms** | `ttft_ms`, `tpot_ms`, `prefill_latency_ms`, `queue_delay_ms`, `prompt_tokens_mean` | Per-window summary (Δ where possible) | **Time-weighted mean** (evaluable windows only) | Average behavior during active periods. |
| **Derived ratios** | `prefix_cache_hit_rate` | **Δ hits / Δ queries** in window | **Last evaluable window only** | Most recent active behavior. |

---

## Precision notes

- **Cumulative counters (multi-window):** Taken from **`windows.last()`** after collection order — not from the last *evaluable* window — so totals match Prometheus “since process start” even if the final slice was idle.
- **Rates:** Per-window rate is already Δ ÷ that window’s duration. Multi-window = duration-weighted mean across **evaluable** windows only. If counters go backwards or reset inside a window, that window’s rate is `None` (no guessing).
- **Static fields:** Treated as constant for the run; rare events (driver reload, GPU reset) can change them.
- **Catalog-resolved fields:** Derived once from the raw `model_name` / `gpu_name` strings at `StaticContext` construction. Never re-queried mid-run. `None` when the name doesn't match any catalog entry — consumers must handle gracefully. GPU `peak_flops_f32_tflops` is non-tensor-core FP32 (conservative roofline input); B200 and GB10 values are estimates.
- **State vs utilization:** State = what the system looks like at the **end of the last active window**. Utilization = how busy the GPU was **over time** (averages).
- **All windows non-evaluable:** The aggregated snapshot is the **chronologically last raw window** in full (nothing to weight).
- **`sm_clock_mhz` min tracking deferred:** A sagged clock during evaluable windows indicates thermal or power throttling. Tracking the minimum would expose this, but idle gaps between polls make a low min ambiguous without a base/boost clock reference to compare against. Deferred until the GPU catalog carries per-GPU base and boost clock data. For now, temperature peak serves as the throttle signal.

---

## What the numbers mean

Profile reports three kinds of numbers. Only the first is documented here; the others are pending final policy decisions.

**Under-load behavior** — evaluable windows only. Idle periods excluded.

| Metric | Examples |
|--------|---------|
| Latency | `ttft_ms`, `tpot_ms`, `prefill_latency_ms`, `queue_delay_ms` |
| Throughput | `generation_tokens_per_sec`, `request_success_per_sec` |
| GPU utilization | `gpu_util_pct`, `mem_util_pct`, `power_watts` |
| Cache hit rates | `prefix_cache_hit_rate` |

A number in this category answers: *"While the server was actively handling requests, how did it behave?"* It does not represent session-wide averages and will differ from APM tools that include idle time.

---

## Cadence

- **250 ms** between samples; count = `sample_count_for(window)` (e.g. ~2 s → 9 ticks).
- GPU and vLLM collection run **in parallel** (`collect_snapshot_for_window`).

---

## Code

`aggregate_windows` in `src/profiler/mod.rs` implements the multi-window column. Collectors: `src/collectors/vllm.rs`, `src/collectors/gpu.rs`. Catalogs: `src/context/model_catalog.rs`, `src/context/gpu_catalog.rs`.
