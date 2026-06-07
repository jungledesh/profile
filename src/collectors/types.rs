use std::time::SystemTime;

pub use prometheus_parse::HistogramCount;

/// Config fields extracted from the `vllm:cache_config_info` labeled gauge.
/// All `Option<T>` — absent when the metric isn't present in the scrape.
#[derive(Debug, Clone, Default)]
pub struct CacheConfigLabels {
    pub block_size: Option<u32>,
    pub num_gpu_blocks: Option<u32>,
    /// KV cache element dtype (e.g. "auto", "fp8", "fp16").
    pub cache_dtype: Option<String>,
    pub enable_prefix_caching: Option<bool>,
    pub enable_chunked_prefill: Option<bool>,
}

/// One `/metrics` scrape: cumulative prefix cache counters (internal + external).
#[derive(Debug, Clone, Default)]
pub struct PrefixCacheScrapeSample {
    pub hits: Option<f64>,
    pub queries: Option<f64>,
}

/// Δ(sum) and Δ(count) for a Prometheus histogram between **first → last** scrape in a window.
/// Sum units match the histogram (seconds for latency histograms, token-sum for `request_prompt_tokens`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistogramWindowMass {
    pub sum_delta: f64,
    pub count_delta: f64,
}

/// vLLM Prometheus scrape.
///
/// How fields are combined across scrapes and diagnose windows: **`docs/collection-policy.md`**.
///
/// **`Option<f64>`:** `Some` values are defined; `None` means that quantity could not be computed
/// (missing series, zero denominator, reset, or zero-length window). `Some(0.0)` is a real zero where
/// applicable (e.g. 0% prefix hits in-window), not “missing data.”
///
/// - **Histogram means** (TTFT, TPOT, prefill, queue, prompt mean): `None` if no observations or
///   **Δcount ≤ 0** in the window (unless last-scrape cumulative fallback applies).
/// - **Multi-window diagnose:** combined mean = **ΣΔsum / ΣΔcount** over evaluable windows using
///   [`HistogramWindowMass`]; falls back to duration-weighted blend of window means if no mass (`docs/collection-policy.md`).
/// - **`generation_tokens_per_sec`:** `None` if missing counters, negative Δ, or zero time window.
/// - **`request_success_per_sec` / `num_preemptions_per_sec`:** Δ counter / window duration (first→last scrape), same rules as generation tokens.
/// - **`prefix_cache_hit_rate`:** Per window: `(Δhits)/(Δqueries)` over first→last scrape. Multi-window diagnose: **`(Σ Δhits)/(Σ Δqueries)`** over evaluable windows (`docs/collection-policy.md`). `None` if no valid query mass.
#[derive(Debug, Clone, Default)]
pub struct VllmRawMetrics {
    pub model_name: Option<String>,

    /// Queue-depth style gauges: **last** `/metrics` scrape in the collection window.
    /// Multi-window diagnose: **time-weighted mean** across evaluable windows (same as `gpu_util_pct`).
    pub num_requests_running: Option<f64>,
    pub num_requests_waiting: Option<f64>,
    /// KV cache usage %: last scrape in a single window; duration-weighted mean across evaluable windows in diagnose aggregate.
    pub kv_cache_usage_perc: Option<f64>,
    /// Same as `kv_cache_usage_perc` after multi-window aggregate; carried for display clarity.
    pub kv_cache_avg_perc: Option<f64>,
    /// Max KV cache usage % seen across scrapes in this window (0–100). Multi-window: max over evaluable windows.
    pub kv_cache_peak_perc: Option<f64>,

    // Histograms: prefer Δsum/Δcount from **first** → **last** scrape (9th sample, ~2s apart);
    // else cumulative mean from the last scrape.
    pub ttft_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    /// p99 TTFT from histogram bucket delta (first→last scrape in window). None if no traffic or counter reset.
    pub ttft_p99_ms: Option<f64>,
    /// p99 TPOT from histogram bucket delta (first→last scrape in window). None if no traffic or counter reset.
    pub tpot_p99_ms: Option<f64>,
    /// Raw histogram delta buckets for TTFT (first→last scrape in window). Empty if no traffic, reset, or histogram unavailable.
    /// Used for mathematically correct multi-window p99 aggregation — merge vectors, then recompute quantile.
    pub ttft_p99_buckets: Vec<HistogramCount>,
    /// Raw histogram delta buckets for TPOT (first→last scrape in window). Empty if no traffic, reset, or histogram unavailable.
    pub tpot_p99_buckets: Vec<HistogramCount>,
    pub prefill_latency_ms: Option<f64>,
    pub queue_delay_ms: Option<f64>,
    /// `request_prompt_tokens` histogram: mean tokens (Δ window or last-scrape fallback).
    pub prompt_tokens_mean: Option<f64>,

    /// Wall-clock seconds from first→last `/metrics` scrape in this collection window.
    pub window_duration_secs: Option<f64>,

    /// Per-window histogram observation mass (first→last scrape). Used for multi-window **ΣΔsum / ΣΔcount** aggregation.
    pub ttft_window_mass: Option<HistogramWindowMass>,
    pub tpot_window_mass: Option<HistogramWindowMass>,
    pub prefill_window_mass: Option<HistogramWindowMass>,
    pub queue_window_mass: Option<HistogramWindowMass>,
    pub prompt_tokens_window_mass: Option<HistogramWindowMass>,

    /// Cumulative generation tokens (last scrape per window), summed over label sets.
    /// Multi-window diagnose: from the **chronologically last** collected window (`docs/collection-policy.md`).
    pub generation_tokens_total: Option<f64>,
    /// Δ generation tokens / s over the first→last scrape window (output throughput).
    pub generation_tokens_per_sec: Option<f64>,
    /// Prefix cache hit rate. Single window: `(Δhits)/(Δqueries)` first→last scrape. Multi-window aggregate: sum of valid window deltas — see `docs/collection-policy.md`.
    pub prefix_cache_hit_rate: Option<f64>,
    /// Cumulative prefix counters per scrape (same order as collector: 9 × ~250ms).
    pub prefix_cache_scrape_samples: Vec<PrefixCacheScrapeSample>,

    // Not always available
    pub max_num_seqs: Option<u32>,

    // Memory pressure / offload state
    /// Sequences actively offloaded to CPU KV cache (PCIe is now on the decode path).
    /// Last scrape in the `/metrics` window (not averaged across the 9 samples).
    pub num_requests_swapped: Option<f64>,
    /// Cumulative preemptions (last scrape). Multi-window: chronological last window.
    pub num_preemptions_total: Option<f64>,
    /// Preemptions per second over the first→last scrape window.
    pub num_preemptions_per_sec: Option<f64>,
    /// CPU KV cache block usage 0–100. Last scrape in the window.
    pub cpu_cache_usage_perc: Option<f64>,

    // Traffic
    /// Cumulative successful requests (last scrape). Multi-window: chronological last window.
    pub request_success_total: Option<f64>,
    /// Completed requests per second (Δ `request_success` / window); use for real QPS.
    pub request_success_per_sec: Option<f64>,

    /// Fields extracted from `vllm:cache_config_info` labels. Default when absent.
    pub cache_config: CacheConfigLabels,
}

/// NVML / DCGM / nvidia-smi scrape
#[derive(Debug, Clone, Default)]
pub struct GpuRawMetrics {
    pub gpu_name: Option<String>,
    /// Device index on this host (`CUDA_VISIBLE_DEVICES` / NVML ordering).
    pub gpu_index: Option<u32>,
    /// Stable per-device identifier from the driver (e.g. `GPU-xxxxxxxx-xxxx-...`).
    pub gpu_uuid: Option<String>,
    pub gpu_util_pct: Option<f64>,
    pub mem_util_pct: Option<f64>,
    pub power_watts: Option<f64>,
    pub power_limit_watts: Option<f64>,
    pub vram_used_mb: Option<u64>,
    /// Max VRAM used (MiB) across NVML polls in this window. Multi-window: max over evaluable windows.
    pub vram_peak_mb: Option<u64>,
    pub vram_total_mb: Option<u64>,
    pub temperature_c: Option<f64>,
    /// Max GPU temperature (°C) across NVML polls in this window. Multi-window: max over evaluable windows (with landing fold-in); stdout parenthetical uses threshold in `output/stdout.rs`.
    pub temperature_peak_c: Option<f64>,
    pub sm_clock_mhz: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct RawSnapshot {
    /// When the GPU collector finished its sampling window (last NVML poll).
    pub gpu_observed_at: SystemTime,
    /// When the vLLM collector finished its last `/metrics` scrape in the window.
    pub vllm_observed_at: SystemTime,
    /// When the snapshot was assembled after both collectors joined.
    pub timestamp: SystemTime,
    pub vllm: VllmRawMetrics,
    pub gpu: GpuRawMetrics,
}

/// A window is structurally valid if core telemetry was collected successfully.
/// Zero traffic is a valid observation — it proves the server is idle, not that
/// collection failed. Only skip windows where timing or metrics are absent entirely.
pub fn window_is_evaluable(s: &RawSnapshot) -> bool {
    // Duration must be present and positive — needed for all rate calculations.
    let duration_ok = s
        .vllm
        .window_duration_secs
        .filter(|w| w.is_finite() && *w > f64::EPSILON)
        .is_some();

    // vLLM metrics must have been collected — num_requests_running being Some
    // (even if 0) means the endpoint responded successfully.
    let vllm_ok = s.vllm.num_requests_running.is_some();

    duration_ok && vllm_ok
}

pub const ACTIVE_KV_CACHE_PCT_THRESHOLD: f64 = 30.0;
pub const ACTIVE_GPU_UTIL_PCT_THRESHOLD: f64 = 20.0;

/// A window contributed real work — used to gate aggregated means.
/// Separate from `window_is_evaluable`, which only checks structural validity.
///
/// Requires `running_reqs > 0` plus at least one activity signal:
///   - kv_cache_pct > 30%  (decode + sustained load)
///   - gpu_util_pct > 20%  (prefill bursts; secondary — NVML is coarse)
///
/// If both signals are absent, `running_reqs > 0` alone is the fallback.
pub fn window_is_active(s: &RawSnapshot) -> bool {
    let running = s
        .vllm
        .num_requests_running
        .filter(|r| r.is_finite() && *r > 0.0);
    let Some(_) = running else {
        return false;
    };

    let kv = s.vllm.kv_cache_usage_perc;
    let gpu = s.gpu.gpu_util_pct;

    match (kv, gpu) {
        (None, None) => true,
        (Some(k), None) => k > ACTIVE_KV_CACHE_PCT_THRESHOLD,
        (None, Some(g)) => g > ACTIVE_GPU_UTIL_PCT_THRESHOLD,
        (Some(k), Some(g)) => {
            k > ACTIVE_KV_CACHE_PCT_THRESHOLD || g > ACTIVE_GPU_UTIL_PCT_THRESHOLD
        }
    }
}

#[cfg(test)]
mod window_evaluable_tests {
    use super::*;
    use std::time::SystemTime;

    fn snap(run: Option<f64>, duration: Option<f64>) -> RawSnapshot {
        RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: VllmRawMetrics {
                num_requests_running: run,
                window_duration_secs: duration,
                ..Default::default()
            },
            gpu: Default::default(),
        }
    }

    #[test]
    fn false_when_running_missing() {
        assert!(!window_is_evaluable(&snap(None, Some(2.0))));
    }

    #[test]
    fn false_when_duration_missing() {
        assert!(!window_is_evaluable(&snap(Some(0.0), None)));
    }

    #[test]
    fn true_when_idle_with_valid_telemetry() {
        assert!(window_is_evaluable(&snap(Some(0.0), Some(2.0))));
    }

    #[test]
    fn false_when_duration_zero() {
        assert!(!window_is_evaluable(&snap(Some(1.0), Some(0.0))));
    }
}

#[cfg(test)]
mod window_active_tests {
    use super::*;
    use std::time::SystemTime;

    fn snap(run: f64, kv: Option<f64>, gpu: Option<f64>) -> RawSnapshot {
        RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: VllmRawMetrics {
                num_requests_running: Some(run),
                kv_cache_usage_perc: kv,
                window_duration_secs: Some(2.0),
                ..Default::default()
            },
            gpu: GpuRawMetrics {
                gpu_util_pct: gpu,
                ..Default::default()
            },
        }
    }

    #[test]
    fn false_when_running_zero() {
        assert!(!window_is_active(&snap(0.0, Some(50.0), Some(50.0))));
    }

    #[test]
    fn false_when_running_zero_and_both_signals_absent() {
        assert!(!window_is_active(&snap(0.0, None, None)));
    }

    #[test]
    fn true_when_both_signals_absent_and_running_positive() {
        assert!(window_is_active(&snap(5.0, None, None)));
    }

    #[test]
    fn true_when_kv_above_threshold() {
        assert!(window_is_active(&snap(5.0, Some(31.0), None)));
        assert!(!window_is_active(&snap(5.0, Some(30.0), None)));
    }

    #[test]
    fn true_when_gpu_above_threshold() {
        assert!(window_is_active(&snap(5.0, None, Some(21.0))));
        assert!(!window_is_active(&snap(5.0, None, Some(20.0))));
    }

    #[test]
    fn false_when_both_present_neither_above_threshold() {
        assert!(!window_is_active(&snap(5.0, Some(10.0), Some(10.0))));
    }

    #[test]
    fn true_when_both_present_either_above_threshold() {
        assert!(window_is_active(&snap(5.0, Some(10.0), Some(25.0))));
        assert!(window_is_active(&snap(5.0, Some(35.0), Some(10.0))));
    }
}
