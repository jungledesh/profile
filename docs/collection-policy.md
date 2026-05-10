# Ground-truth capture policy (v2)

Normative rules for how Profile combines vLLM `/metrics` scrapes and NVML polls. **Evaluable window:** `window_is_evaluable` is true (enough concurrent load or tok/s). Diagnostic aggregates prioritize **what happened under load** while still exposing **true cumulative counters** from the latest collection.

---

## Policy table

| Category | Examples | Single ~2 s window (9 samples) | Multi-window aggregation | Rationale |
|----------|----------|-------------------------------|---------------------------|-----------|
| **State gauges** | `kv_cache_usage_perc`, `num_requests_swapped`, `cpu_cache_usage_perc`, `num_requests_running`, `num_requests_waiting`, `max_num_seqs`, `temperature_c`, `sm_clock_mhz`, `vram_used_mb` | **Last scrape / last poll** | **Last evaluable window’s last value** | State during the last **active** period (diagnostic focus). |
| **Static / config** | `model_name`, `gpu_name`, `vram_total_mb`, `gpu_index`, `gpu_uuid`, `power_limit_watts`, `cache_config` | **Last scrape** | **Last evaluable window’s last value** | Stable for the lifetime of this run. |
| **Utilization** | `gpu_util_pct`, `mem_util_pct`, `power_watts` | **Mean of 9** | **Time-weighted mean** (evaluable windows only) | Average load over the observed period. |
| **Counters (cumulative)** | `request_success_total`, `num_preemptions_total`, `generation_tokens_total` | **Last scrape (cumulative)** | **Chronologically last collected window** (evaluable or not) | True server cumulative totals at end of collection. |
| **Rates** | `request_success_per_sec`, `num_preemptions_per_sec`, `generation_tokens_per_sec` | **Δ ÷ window duration** | **Time-weighted mean** of per-window rates (**evaluable windows only**) | Effective throughput while active. |
| **Latency / histograms** | `ttft_ms`, `tpot_ms`, `prefill_latency_ms`, `queue_delay_ms`, `prompt_tokens_mean` | Per-window summary (Δ where possible) | **Time-weighted mean** (evaluable windows only) | Average behavior during active periods. |
| **Derived ratios** | `prefix_cache_hit_rate` | **Δ hits / Δ queries** in window | **Last evaluable window only** | Most recent active behavior. |

---

## Precision notes

- **Cumulative counters (multi-window):** Taken from **`windows.last()`** after collection order — not from the last *evaluable* window — so totals match Prometheus “since process start” even if the final slice was idle.
- **Rates:** Per-window rate is already Δ ÷ that window’s duration. Multi-window = duration-weighted mean across **evaluable** windows only. If counters go backwards or reset inside a window, that window’s rate is `None` (no guessing).
- **Static fields:** Treated as constant for the run; rare events (driver reload, GPU reset) can change them.
- **State vs utilization:** State = what the system looks like at the **end of the last active window**. Utilization = how busy the GPU was **over time** (averages).
- **All windows non-evaluable:** The aggregated snapshot is the **chronologically last raw window** in full (nothing to weight).

---

## Cadence

- **250 ms** between samples; count = `sample_count_for(window)` (e.g. ~2 s → 9 ticks).
- GPU and vLLM collection run **in parallel** (`collect_snapshot_for_window`).

---

## Code

`aggregate_windows` in `src/profiler/mod.rs` implements the multi-window column. Collectors: `src/collectors/vllm.rs`, `src/collectors/gpu.rs`.
