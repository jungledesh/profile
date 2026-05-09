use std::time::SystemTime;

/// Config fields extracted from the `vllm:cache_config_info` labeled gauge.
/// All `Option<T>` — absent when the metric isn't present in the scrape.
#[derive(Debug, Clone, Default)]
pub struct CacheConfigLabels {
    pub block_size: Option<u32>,
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

/// vLLM Prometheus scrape.
///
/// How fields are combined across scrapes and diagnose windows: **`docs/collection-policy.md`**.
///
/// **`Option<f64>`:** `Some` values are defined; `None` means that quantity could not be computed
/// (missing series, zero denominator, reset, or zero-length window). `Some(0.0)` is a real zero where
/// applicable (e.g. 0% prefix hits in-window), not “missing data.”
///
/// - **Histogram means** (TTFT, TPOT, prefill, queue, prompt mean): `None` if no observations or
///   **Δcount ≤ 0** in the window.
/// - **`generation_tokens_per_sec`:** `None` if missing counters, negative Δ, or zero time window.
/// - **`request_success_per_sec` / `num_preemptions_per_sec`:** Δ counter / window duration (first→last scrape), same rules as generation tokens.
/// - **`prefix_cache_hit_rate`:** `None` if **`Δqueries ≤ 0`** or invalid deltas; `Some(0.0)` means 0% hits in-window.
#[derive(Debug, Clone, Default)]
pub struct VllmRawMetrics {
    pub model_name: Option<String>,

    /// Queue-depth style gauges: **last** `/metrics` scrape in the collection window (not averaged).
    pub num_requests_running: Option<f64>,
    pub num_requests_waiting: Option<f64>,
    /// Last scrape in the window.
    pub kv_cache_usage_perc: Option<f64>,

    // Histograms: prefer Δsum/Δcount from **first** → **last** scrape (9th sample, ~2s apart);
    // else cumulative mean from the last scrape.
    pub ttft_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub prefill_latency_ms: Option<f64>,
    pub queue_delay_ms: Option<f64>,
    /// `request_prompt_tokens` histogram: mean tokens (Δ window or last-scrape fallback).
    pub prompt_tokens_mean: Option<f64>,

    /// Cumulative generation tokens (last scrape per window), summed over label sets.
    /// Multi-window diagnose: from the **chronologically last** collected window (`docs/collection-policy.md`).
    pub generation_tokens_total: Option<f64>,
    /// Δ generation tokens / s over the first→last scrape window (output throughput).
    pub generation_tokens_per_sec: Option<f64>,
    /// Prefix cache hit rate: `(Δhits)/(Δqueries)` over first→last scrape (internal + external).
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
    pub vram_total_mb: Option<u64>,
    pub temperature_c: Option<f64>,
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

/// Mean `num_requests_running` above this (exclusive) counts as evaluable traffic.
pub const EVALUABLE_RUNNING_GT: f64 = 0.75;
/// Generation throughput above this (exclusive, tok/s) counts as evaluable when running is low or missing.
pub const EVALUABLE_TOK_PER_SEC_GT: f64 = 20.0;

/// A window is evaluable if there is meaningful activity: enough concurrent requests or enough throughput.
pub fn window_is_evaluable(s: &RawSnapshot) -> bool {
    let run_ok = s
        .vllm
        .num_requests_running
        .filter(|x| x.is_finite())
        .map(|r| r > EVALUABLE_RUNNING_GT)
        .unwrap_or(false);
    let tok_ok = s
        .vllm
        .generation_tokens_per_sec
        .filter(|x| x.is_finite())
        .map(|t| t > EVALUABLE_TOK_PER_SEC_GT)
        .unwrap_or(false);
    run_ok || tok_ok
}
