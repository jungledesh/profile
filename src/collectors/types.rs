use std::time::SystemTime;

pub use prometheus_parse::HistogramCount;

/// Host KV offload buffer from `cache_config_info.kv_offloading_size`.
///
/// vLLM's docstring: size is the total summed across TP ranks. Irrelevant at
/// launch (tp pinned to 1); matters when multi-GPU lands.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum KvOffloadState {
    /// Label key absent: build predates the feature.
    #[default]
    Unsupported,
    /// Literal `None` or `0`: offload not enabled.
    Off,
    /// Finite size > 0 GiB: already on.
    Enabled(f64),
    /// Label present but value was not `None` and not a finite number.
    Unreadable,
}

/// Host RAM and container cgroup cap, read together or not at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemoryFacts {
    pub available_bytes: u64,
    /// `None` = unlimited container cap (`max` or no cgroup file).
    pub container_limit_bytes: Option<u64>,
}

/// Config fields extracted from the `vllm:cache_config_info` labeled gauge.
/// All `Option<T>` - absent when the metric isn't present in the scrape.
///
/// `kv_cache_size_tokens`, `kv_cache_max_concurrency`, `mamba_block_size`, and
/// `mamba_page_size_padded` are allocator-computed in vLLM v0.25.1
/// (`vllm/config/cache.py`). Their docstring notes that `num_gpu_blocks ×
/// block_size` can be wrong for hybrid models. Never guess when absent.
///
/// These are consequences of config, not config themselves. Do not add them to
/// `drift.rs::config_changed`.
#[derive(Debug, Clone, Default)]
pub struct CacheConfigLabels {
    pub block_size: Option<u32>,
    pub num_gpu_blocks: Option<u32>,
    /// KV cache element dtype (e.g. "auto", "fp8", "fp16").
    pub cache_dtype: Option<String>,
    pub enable_prefix_caching: Option<bool>,
    /// Chunked prefill flag. On modern vLLM this lives on `SchedulerConfig`, not
    /// `CacheConfig`, so `cache_config_info` usually omits it (`None`). Older
    /// builds that still label it here parse as before. Prefer
    /// `VllmConfig.enable_chunked_prefill` after `/info` / `/server_info` enrichment.
    pub enable_chunked_prefill: Option<bool>,
    /// Fraction of GPU memory reserved for the engine (cache_config_info label).
    pub gpu_memory_utilization: Option<f64>,
    /// Total KV token capacity reported by the allocator.
    pub kv_cache_size_tokens: Option<u64>,
    /// Max concurrent full-context sequences at `max_model_len` (may be fractional).
    pub kv_cache_max_concurrency: Option<f64>,
    pub mamba_block_size: Option<u32>,
    pub mamba_page_size_padded: Option<u64>,
    pub kv_offloading: KvOffloadState,
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

/// vLLM Prometheus scrape. `Option<f64>`: `Some` = defined, `None` = missing/reset/zero-denom.
/// Aggregation rules across scrapes and windows: `docs/collection-policy.md`.
#[derive(Debug, Clone, Default)]
pub struct VllmRawMetrics {
    pub model_name: Option<String>,

    /// Queue-depth style gauges: **last** `/metrics` scrape in the collection window.
    /// Multi-window diagnose: **time-weighted mean** across evaluable windows (same as `gpu_util_pct`).
    pub num_requests_running: Option<f64>,
    /// Max `num_requests_running` across scrapes in this window. Multi-window: max over
    /// evaluable windows (not mean). Used to disprove full-context `kv_cache_max_concurrency`.
    pub num_requests_running_peak: Option<f64>,
    pub num_requests_waiting: Option<f64>,
    /// Max `num_requests_waiting` across scrapes in this window. Multi-window: max over
    /// evaluable windows (not mean). Caps kv-offload parked-request demand.
    pub num_requests_waiting_peak: Option<f64>,
    /// Same-sample seat wall, evaluated end-of-window: any scrape had running in
    /// the churn band at `max_num_seqs` (`[max - slack, max + 0.5]`, slack =
    /// clamp(completions/s × 250ms, 1, max(1% of max, 1))) and waiting >= 2.
    /// Samples above max + 0.5 are evaluated-false (chunked), not skipped.
    /// `None` = zero evaluable samples (max unread, or run/wait never both finite
    /// while max was known). Per-window only; R5 significance counts windows that
    /// fire. The diagnose summary leaves this `None` (no unused OR fold).
    pub seat_wall_cooccurred: Option<bool>,
    /// Max over active windows of `(kv_cache_usage_perc / 100) / num_requests_running`
    /// for windows with `running >= 1`. Multi-window aggregate only; single-window
    /// resolve may fall back to the landing gauges. Used for kv-offload sizing.
    pub kv_frac_per_running_peak: Option<f64>,
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
    /// Used for mathematically correct multi-window p99 aggregation - merge vectors, then recompute quantile.
    pub ttft_p99_buckets: Vec<HistogramCount>,
    /// Raw histogram delta buckets for TPOT (first→last scrape in window). Empty if no traffic, reset, or histogram unavailable.
    pub tpot_p99_buckets: Vec<HistogramCount>,
    /// p95 TTFT from histogram bucket delta. None if no traffic or counter reset.
    pub ttft_p95_ms: Option<f64>,
    /// p95 TPOT from histogram bucket delta. None if no traffic or counter reset.
    pub tpot_p95_ms: Option<f64>,
    /// True when the corresponding quantile landed in the +Inf bucket (`*_ms` is a floor).
    pub ttft_p99_clamped: bool,
    pub tpot_p99_clamped: bool,
    pub ttft_p95_clamped: bool,
    pub tpot_p95_clamped: bool,
    pub prefill_latency_ms: Option<f64>,
    pub queue_delay_ms: Option<f64>,
    /// `request_prompt_tokens` histogram: mean tokens (Δ window or last-scrape fallback).
    pub prompt_tokens_mean: Option<f64>,
    /// `request_prompt_tokens` histogram: p99 from first→last scrape bucket delta.
    pub prompt_tokens_p99: Option<f64>,
    /// Raw histogram delta buckets for prompt tokens (first→last scrape in window).
    pub prompt_tokens_p99_buckets: Vec<HistogramCount>,
    /// `request_generation_tokens` histogram: mean tokens (Δ window or last-scrape fallback).
    pub generation_tokens_mean: Option<f64>,
    /// `request_generation_tokens` histogram: p99 from first→last scrape bucket delta.
    pub generation_tokens_p99: Option<f64>,
    /// Raw histogram delta buckets for generation tokens (first→last scrape in window).
    pub generation_tokens_p99_buckets: Vec<HistogramCount>,
    /// Completed-request count from the `request_generation_tokens` +Inf bucket delta.
    /// Request count, not token count. Gates --max-model-len suggestions (>= 100) and
    /// the $/1M output tok turnover check (completed >= mean running).
    pub generation_tokens_completed: Option<f64>,

    /// Wall-clock seconds from first→last `/metrics` scrape in this collection window.
    pub window_duration_secs: Option<f64>,

    /// Per-window histogram observation mass (first→last scrape). Used for multi-window **ΣΔsum / ΣΔcount** aggregation.
    pub ttft_window_mass: Option<HistogramWindowMass>,
    pub tpot_window_mass: Option<HistogramWindowMass>,
    pub prefill_window_mass: Option<HistogramWindowMass>,
    pub queue_window_mass: Option<HistogramWindowMass>,
    pub prompt_tokens_window_mass: Option<HistogramWindowMass>,
    pub generation_tokens_window_mass: Option<HistogramWindowMass>,

    /// Cumulative generation tokens (last scrape per window), summed over label sets.
    /// Multi-window diagnose: from the **chronologically last** collected window (`docs/collection-policy.md`).
    pub generation_tokens_total: Option<f64>,
    /// Δ generation tokens / s over the first→last scrape window (output throughput).
    pub generation_tokens_per_sec: Option<f64>,
    /// Duration-weighted mean generation tok/s over the energy-pair window set only.
    /// Same windows as [`GpuRawMetrics::aligned_power_watts`]. Multi-window aggregate.
    pub aligned_generation_tokens_per_sec: Option<f64>,
    /// Δ prompt tokens / s over the first→last scrape window (input throughput).
    pub prompt_tokens_per_sec: Option<f64>,
    /// Prefix cache hit rate. Single window: `(Δhits)/(Δqueries)` first→last scrape. Multi-window aggregate: sum of valid window deltas - see `docs/collection-policy.md`.
    pub prefix_cache_hit_rate: Option<f64>,
    /// Cumulative prefix counters per scrape (same order as collector: 9 × ~250ms).
    pub prefix_cache_scrape_samples: Vec<PrefixCacheScrapeSample>,

    // Not always available
    pub max_num_seqs: Option<u32>,
    /// From `vllm_max_num_batched_tokens` gauge when present. Often absent on
    /// modern vLLM (value lives on `SchedulerConfig`); filled from `/info` /
    /// `/server_info` in `build_config` when the gauge is missing.
    pub max_num_batched_tokens: Option<u32>,

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

    /// Model weight dtype from vLLM Prometheus label metrics (`vllm:model_config_info`, `vllm:info`), if present.
    pub model_weight_dtype: Option<String>,
}

/// GPU telemetry scrape
#[derive(Debug, Clone, Default)]
pub struct GpuRawMetrics {
    pub gpu_name: Option<String>,
    /// Device index on this host (device driver ordering).
    pub gpu_index: Option<u32>,
    /// Stable per-device identifier from the driver (e.g. `GPU-xxxxxxxx-xxxx-...`).
    pub gpu_uuid: Option<String>,
    pub pcie_bus_id: Option<String>,
    pub gpu_util_pct: Option<f64>,
    pub mem_util_pct: Option<f64>,
    /// Per-window raw power draw (W). Cleared on multi-window aggregate so unaligned
    /// last-window power cannot leak into display or energy.
    pub power_watts: Option<f64>,
    /// Duration-weighted mean power over the energy-pair window set (clocks aligned,
    /// gen tok/s > 0, and sum power > 0). Display and J/tok use this only.
    /// `None` when no energy-pair windows contribute.
    pub aligned_power_watts: Option<f64>,
    pub power_limit_watts: Option<f64>,
    pub vram_used_mb: Option<u64>,
    /// Max VRAM used (MiB) across GPU polls in this window. Multi-window: max over evaluable windows.
    pub vram_peak_mb: Option<u64>,
    pub vram_total_mb: Option<u64>,
    pub temperature_c: Option<f64>,
    /// Max GPU temperature (°C) across GPU polls in this window. Multi-window: max over evaluable windows (with landing fold-in); stdout parenthetical uses threshold in `output/stdout.rs`.
    pub temperature_peak_c: Option<f64>,
    pub sm_clock_mhz: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuIdentity {
    Uuid(String),
    Pcie(String),
    Index(u32),
    Simple(String),
    Unknown,
}

impl std::fmt::Display for GpuIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuIdentity::Uuid(u) => write!(f, "{u}"),
            GpuIdentity::Pcie(p) => write!(f, "{p}"),
            GpuIdentity::Index(i) => write!(f, "GPU-{i}"),
            GpuIdentity::Simple(s) => write!(f, "{s}"),
            GpuIdentity::Unknown => write!(f, "Unknown"),
        }
    }
}

pub type GpuFingerprint = Vec<GpuIdentity>;

impl GpuRawMetrics {
    pub fn identity(&self) -> GpuIdentity {
        if let Some(uuid) = &self.gpu_uuid {
            GpuIdentity::Uuid(uuid.clone())
        } else if let Some(pcie) = &self.pcie_bus_id {
            GpuIdentity::Pcie(pcie.clone())
        } else if let Some(idx) = self.gpu_index {
            GpuIdentity::Index(idx)
        } else {
            GpuIdentity::Unknown
        }
    }

    /// True when fingerprint falls back to GPU device index (no UUID or PCIe bus ID, but has index).
    pub fn uses_index_only(&self) -> bool {
        self.gpu_uuid.is_none() && self.pcie_bus_id.is_none() && self.gpu_index.is_some()
    }

    pub fn display_id(&self) -> String {
        if let Some(uuid) = &self.gpu_uuid {
            let id = if uuid.starts_with("GPU-") || uuid.starts_with("MIG-") {
                &uuid[4..uuid.len().min(12)]
            } else {
                &uuid[..uuid.len().min(8)]
            };
            return id.to_string();
        }
        if let Some(pcie) = &self.pcie_bus_id {
            return pcie.clone();
        }
        self.gpu_index
            .map(|i| i.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    }
}

#[derive(Debug, Clone, Default)]
pub struct AggregateGpuMetrics {
    pub gpu_util_pct: Option<f64>,
    pub mem_util_pct: Option<f64>,
    /// Sum of per-GPU [`GpuRawMetrics::aligned_power_watts`]. Display/energy only.
    ///
    /// Each per-GPU value is already a duration-weighted mean over that GPU's
    /// energy-pair windows. Sum-of-means equals mean-of-sums only when every GPU
    /// contributes on the same window set. At tp=1 that holds. Multi-GPU must not
    /// inherit this field as Σenergy/Σtokens without a shared window mask.
    pub aligned_power_watts: Option<f64>,
    pub vram_used_mb: Option<u64>,
    pub vram_peak_mb: Option<u64>,
    pub sum_vram_total_mb: Option<u64>,
    pub temperature_peak_c: Option<f64>,
    pub gpu_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawSnapshot {
    /// When the GPU collector finished its sampling window (last GPU poll).
    pub gpu_observed_at: SystemTime,
    /// When the vLLM collector finished its last `/metrics` scrape in the window.
    pub vllm_observed_at: SystemTime,
    /// When the snapshot was assembled after both collectors joined.
    pub timestamp: SystemTime,
    pub vllm: VllmRawMetrics,
    pub gpus: Vec<GpuRawMetrics>,
    /// Host RAM + container cap when both reads succeed at collect time.
    pub host_memory: Option<HostMemoryFacts>,
}

impl Default for RawSnapshot {
    fn default() -> Self {
        Self {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: VllmRawMetrics::default(),
            gpus: Vec::new(),
            host_memory: None,
        }
    }
}

impl RawSnapshot {
    /// GPUs profile actually observed this window.
    pub fn collected_gpu_count(&self) -> usize {
        self.gpus.len()
    }

    pub fn aggregate_gpu(&self) -> AggregateGpuMetrics {
        // util/mem: mean across GPUs - max would overstate cluster saturation
        let mut sum_util = 0.0_f64;
        let mut n_util = 0_u32;
        let mut sum_mem = 0.0_f64;
        let mut n_mem = 0_u32;
        let mut sum_power = 0.0;
        let mut n_power = 0;
        let mut sum_vram = 0_u64;
        let mut n_vram = 0;
        let mut max_vram_peak = None;
        let mut sum_vram_total = None;
        let mut max_temp: Option<f64> = None;
        let mut gpu_name = None;

        for g in &self.gpus {
            if gpu_name.is_none() {
                gpu_name = g.gpu_name.clone();
            }
            if let Some(u) = g.gpu_util_pct {
                sum_util += u;
                n_util += 1;
            }
            if let Some(m) = g.mem_util_pct {
                sum_mem += m;
                n_mem += 1;
            }
            if let Some(p) = g.aligned_power_watts {
                sum_power += p;
                n_power += 1;
            }
            if let Some(vu) = g.vram_used_mb {
                sum_vram += vu;
                n_vram += 1;
            }
            if let Some(vp) = g.vram_peak_mb {
                max_vram_peak = Some(max_vram_peak.unwrap_or(0).max(vp));
            }
            if let Some(vt) = g.vram_total_mb {
                sum_vram_total = Some(sum_vram_total.unwrap_or(0) + vt);
            }
            if let Some(t) = g.temperature_peak_c {
                max_temp = Some(max_temp.map_or(t, |cur| cur.max(t)));
            }
        }

        AggregateGpuMetrics {
            gpu_util_pct: (n_util > 0).then_some(sum_util / f64::from(n_util)),
            mem_util_pct: (n_mem > 0).then_some(sum_mem / f64::from(n_mem)),
            aligned_power_watts: (n_power > 0).then_some(sum_power),
            vram_used_mb: (n_vram > 0).then_some(sum_vram),
            vram_peak_mb: max_vram_peak,
            sum_vram_total_mb: sum_vram_total,
            temperature_peak_c: max_temp,
            gpu_name,
        }
    }

    pub fn fingerprint(&self) -> GpuFingerprint {
        let raw: Vec<_> = self.gpus.iter().map(|g| g.identity()).collect();
        if snapshot_uses_display_names(&self.gpus) {
            raw.iter()
                .enumerate()
                .map(|(i, _)| GpuIdentity::Simple(format!("Device-{i}")))
                .collect()
        } else {
            raw
        }
    }
}

pub(crate) fn validate_gpu_identities(gpus: &[GpuRawMetrics]) -> anyhow::Result<()> {
    let unknown = gpus
        .iter()
        .filter(|g| matches!(g.identity(), GpuIdentity::Unknown))
        .count();
    if unknown > 0 && unknown < gpus.len() {
        anyhow::bail!(
            "Mixed identifiers: {unknown} of {} GPUs are Unknown.",
            gpus.len()
        );
    }

    let mut has_uuid = false;
    let mut has_pcie = false;
    let mut has_index = false;

    for g in gpus {
        match g.identity() {
            GpuIdentity::Uuid(_) => has_uuid = true,
            GpuIdentity::Pcie(_) => has_pcie = true,
            GpuIdentity::Index(_) => has_index = true,
            _ => {}
        }
    }

    let types_count = has_uuid as u8 + has_pcie as u8 + has_index as u8;
    if types_count > 1 {
        anyhow::bail!("Mixed identifiers (UUID/PCIe/Index). Cannot track safely.");
    }

    Ok(())
}

pub fn snapshot_uses_display_names(gpus: &[GpuRawMetrics]) -> bool {
    !gpus.is_empty()
        && gpus
            .iter()
            .all(|g| matches!(g.identity(), GpuIdentity::Unknown))
}

pub fn snapshot_uses_index_only(gpus: &[GpuRawMetrics]) -> bool {
    !snapshot_uses_display_names(gpus)
        && !gpus.is_empty()
        && gpus.iter().any(GpuRawMetrics::uses_index_only)
}

/// MiB → decimal GB (SI, 1e9 bytes). Matches `weight_gb` / `kv_headroom_gb`.
///
/// NVML and ROCm report bytes; collectors store MiB (`bytes / 1024²`). Physics
/// subtracts decimal-GB weights, so this conversion is mandatory at the
/// MiB→GB boundary. Do not use `mib / 1024` (that yields GiB and understates
/// headroom by ~7%).
///
/// Operator-facing VRAM lines may still show binary GB (`mib / 1024`) so an
/// 80 GiB card reads as 80GB; that display path must not feed headroom math.
pub fn mib_to_decimal_gb(mib: u64) -> f64 {
    (mib as f64) * ((1024.0 * 1024.0) / 1_000_000_000.0)
}

/// Tensor-parallel degree for roofline, cost, and R4.
/// Config/CLI wins when set; otherwise infer from collected GPU count.
/// When config exceeds collected GPUs, clamp to collected - physics must not assume
/// hardware we did not observe.
pub fn effective_tensor_parallel(config_tp: Option<u32>, collected_gpus: usize) -> Option<u32> {
    if collected_gpus == 0 {
        return None;
    }
    let collected = collected_gpus as u32;
    match config_tp {
        Some(cfg) => Some(cfg.min(collected).max(1)),
        None => Some(collected),
    }
}

/// A window is structurally valid if core telemetry was collected successfully.
/// Zero traffic is a valid observation - it proves the server is idle, not that
/// collection failed. Only skip windows where timing or metrics are absent entirely.
pub fn window_is_evaluable(s: &RawSnapshot) -> bool {
    // Duration must be present and positive - needed for all rate calculations.
    let duration_ok = s
        .vllm
        .window_duration_secs
        .filter(|w| w.is_finite() && *w > f64::EPSILON)
        .is_some();

    // vLLM metrics must have been collected - num_requests_running being Some
    // (even if 0) means the endpoint responded successfully.
    let vllm_ok = s.vllm.num_requests_running.is_some();

    duration_ok && vllm_ok
}

/// True when the window is evaluable but shows no meaningful traffic.
/// Used to detect idle servers that have working telemetry.
pub fn window_is_idle(s: &RawSnapshot) -> bool {
    if !window_is_evaluable(s) {
        return false;
    }
    let no_running = s
        .vllm
        .num_requests_running
        .filter(|r| r.is_finite())
        .is_none_or(|r| r < 1.0);
    let no_throughput = s
        .vllm
        .generation_tokens_per_sec
        .filter(|t| t.is_finite())
        .is_none_or(|t| t < 1.0);
    // None waiting preserves idle (same convention as running/gen). Some(waiting>=1)
    // means admission backlog: not idle even if nothing is decoding yet.
    let no_waiting = s
        .vllm
        .num_requests_waiting
        .filter(|w| w.is_finite())
        .is_none_or(|w| w < 1.0);
    no_running && no_throughput && no_waiting
}

/// True when the window is evaluable and not idle.
/// Inverse of idle for evaluable windows. Gates aggregated gauge means
/// (throughput, TTFT, TPOT, running/waiting). Low values are honest; blank is not.
pub fn window_is_active(s: &RawSnapshot) -> bool {
    window_is_evaluable(s) && !window_is_idle(s)
}

/// Max |gpu_observed_at − vllm_observed_at| (seconds) for a cross-source join.
/// Cross-source joins only. Rules read single-source data and must not use this.
pub(crate) const MAX_OBSERVATION_SKEW_SECS: f64 = 1.0;

/// Absolute observation-time delta in seconds.
pub(crate) fn skew_secs(a: SystemTime, b: SystemTime) -> f64 {
    match a.duration_since(b) {
        Ok(d) => d.as_secs_f64(),
        Err(e) => -e.duration().as_secs_f64(),
    }
    .abs()
}

/// True when GPU and vLLM observation clocks are close enough to join.
pub(crate) fn observations_aligned(s: &RawSnapshot) -> bool {
    skew_secs(s.gpu_observed_at, s.vllm_observed_at) <= MAX_OBSERVATION_SKEW_SECS
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
            gpus: vec![],
            host_memory: None,
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

    #[test]
    fn window_is_idle_zero_traffic() {
        let s = RawSnapshot {
            vllm: VllmRawMetrics {
                num_requests_running: Some(0.0),
                generation_tokens_per_sec: Some(0.0),
                window_duration_secs: Some(2.0),
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(window_is_idle(&s));
    }

    #[test]
    fn window_is_idle_when_waiting_none() {
        let s = RawSnapshot {
            vllm: VllmRawMetrics {
                num_requests_running: Some(0.0),
                generation_tokens_per_sec: Some(0.0),
                num_requests_waiting: None,
                window_duration_secs: Some(2.0),
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(window_is_idle(&s));
    }

    #[test]
    fn window_is_not_idle_with_admission_backlog() {
        let s = RawSnapshot {
            vllm: VllmRawMetrics {
                num_requests_running: Some(0.0),
                generation_tokens_per_sec: Some(0.0),
                num_requests_waiting: Some(5.0),
                window_duration_secs: Some(2.0),
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(!window_is_idle(&s));
        assert!(window_is_active(&s));
    }

    #[test]
    fn window_is_idle_false_when_not_evaluable() {
        let s = RawSnapshot {
            vllm: VllmRawMetrics {
                num_requests_running: None,
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(!window_is_idle(&s));
    }
}

#[cfg(test)]
mod window_active_tests {
    use super::*;
    use std::time::SystemTime;

    fn snap(run: f64, tps: Option<f64>, kv: Option<f64>, gpu: Option<f64>) -> RawSnapshot {
        RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: VllmRawMetrics {
                num_requests_running: Some(run),
                generation_tokens_per_sec: tps,
                kv_cache_usage_perc: kv,
                window_duration_secs: Some(2.0),
                ..Default::default()
            },
            gpus: vec![GpuRawMetrics {
                gpu_util_pct: gpu,
                ..Default::default()
            }],
            host_memory: None,
        }
    }

    #[test]
    fn false_when_idle_even_with_high_kv_gpu() {
        assert!(!window_is_active(&snap(
            0.0,
            Some(0.0),
            Some(50.0),
            Some(50.0)
        )));
        assert!(!window_is_active(&snap(
            0.5,
            Some(0.0),
            Some(50.0),
            Some(50.0)
        )));
    }

    #[test]
    fn false_when_not_evaluable() {
        let s = RawSnapshot {
            vllm: VllmRawMetrics {
                num_requests_running: None,
                window_duration_secs: Some(2.0),
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(!window_is_active(&s));
    }

    #[test]
    fn true_when_running_material_regardless_of_kv_gpu() {
        // Former dead zone: trickle traffic below old 30%/20% heuristics.
        assert!(window_is_active(&snap(
            2.0,
            Some(5.0),
            Some(10.0),
            Some(5.0)
        )));
        assert!(window_is_active(&snap(
            5.0,
            Some(0.0),
            Some(10.0),
            Some(10.0)
        )));
        assert!(window_is_active(&snap(5.0, None, None, None)));
        // Finish ramp: running drained but tok/s still material.
        assert!(window_is_active(&snap(
            0.0,
            Some(5.0),
            Some(10.0),
            Some(5.0)
        )));
    }

    #[test]
    fn mirrors_not_idle_for_evaluable_windows() {
        let active = snap(3.0, Some(10.0), Some(1.0), Some(1.0));
        assert!(window_is_evaluable(&active));
        assert!(!window_is_idle(&active));
        assert!(window_is_active(&active));

        let idle = snap(0.0, Some(0.0), Some(90.0), Some(90.0));
        assert!(window_is_evaluable(&idle));
        assert!(window_is_idle(&idle));
        assert!(!window_is_active(&idle));
    }
}

#[cfg(test)]
mod effective_tensor_parallel_tests {
    use super::effective_tensor_parallel;

    #[test]
    fn config_tp_clamped_when_collected_fewer_than_config() {
        assert_eq!(effective_tensor_parallel(Some(4), 2), Some(2));
    }

    #[test]
    fn infers_from_collected_when_unset() {
        assert_eq!(effective_tensor_parallel(None, 2), Some(2));
    }

    #[test]
    fn none_when_no_gpus_collected() {
        assert_eq!(effective_tensor_parallel(Some(4), 0), None);
    }
}

#[cfg(test)]
mod mib_to_decimal_gb_tests {
    use super::mib_to_decimal_gb;

    #[test]
    fn h100_80gib_is_decimal_not_binary() {
        let mib = 80 * 1024;
        let gb = mib_to_decimal_gb(mib);
        // 81920 MiB × 1.048576e-3 ≈ 85.90 decimal GB, not 80 GiB.
        assert!((gb - 85.89934592).abs() < 1e-6);
        assert!((gb - (mib as f64 / 1024.0)).abs() > 5.0);
    }
}

#[cfg(test)]
mod gpu_identity_tests {
    use super::*;

    #[test]
    fn index_identity_when_uuid_and_pcie_missing() {
        let g = GpuRawMetrics {
            gpu_index: Some(0),
            ..Default::default()
        };
        assert!(g.uses_index_only());
        assert!(snapshot_uses_index_only(&[g]));
    }

    #[test]
    fn fingerprint_substitutes_soul_names_when_all_unknown() {
        let snap = RawSnapshot {
            gpus: vec![GpuRawMetrics::default(), GpuRawMetrics::default()],
            ..Default::default()
        };

        let fp = snap.fingerprint();
        assert_eq!(fp.len(), 2);
        assert!(matches!(fp[0], GpuIdentity::Simple(_)));
        assert!(matches!(fp[1], GpuIdentity::Simple(_)));
        assert!(snapshot_uses_display_names(&snap.gpus));
    }

    #[test]
    fn validate_rejects_mixed_unknown_and_known() {
        let gpus = vec![
            GpuRawMetrics {
                gpu_uuid: Some("GPU-abc".into()),
                ..Default::default()
            },
            GpuRawMetrics::default(),
        ];
        let err = validate_gpu_identities(&gpus).unwrap_err();
        assert!(err.to_string().contains("1 of 2 GPUs are Unknown"));
    }
}

pub(crate) fn validate_tensor_parallel_scope(
    collected_gpus: u32,
    tp: Option<u32>,
) -> anyhow::Result<()> {
    if collected_gpus == 0 {
        anyhow::bail!("0 GPUs collected - cannot profile without GPU telemetry. Check GPU driver.");
    }
    if let Some(tp_val) = tp {
        if collected_gpus < tp_val {
            anyhow::bail!(
                "TP {tp_val} requires {tp_val} GPUs but only {collected_gpus} visible. Check your --tensor-parallel-size or GPU visibility."
            );
        }
        if collected_gpus > tp_val {
            anyhow::bail!(
                "Profile sees {collected_gpus} GPUs but TP is {tp_val}. Restrict GPU visibility to match vLLM's GPU scope."
            );
        }
    }
    Ok(())
}

pub(crate) fn check_topology_drift(
    base: &GpuFingerprint,
    fp: &GpuFingerprint,
) -> anyhow::Result<()> {
    if fp != base {
        use std::collections::HashSet;
        let base_set: HashSet<_> = base.iter().collect();
        let fp_set: HashSet<_> = fp.iter().collect();
        let missing: Vec<_> = base
            .iter()
            .filter(|g| !fp_set.contains(g))
            .map(|g| g.to_string())
            .collect();
        let new: Vec<_> = fp
            .iter()
            .filter(|g| !base_set.contains(g))
            .map(|g| g.to_string())
            .collect();
        let mut changes = Vec::new();
        if !missing.is_empty() {
            changes.push(format!("missing [{}]", missing.join(", ")));
        }
        if !new.is_empty() {
            changes.push(format!("new [{}]", new.join(", ")));
        }

        if base
            .iter()
            .all(|g| matches!(g, GpuIdentity::Unknown | GpuIdentity::Simple(_)))
        {
            eprintln!(
                "\n[i] Warning: Topology drift detected ({}), but GPU identities are unknown/simple. Continuing.",
                changes.join(", ")
            );
            return Ok(());
        }

        anyhow::bail!(
            "Topology drift detected: {}. Hardware unstable.",
            changes.join(", ")
        );
    }
    Ok(())
}

#[cfg(test)]
mod scope_and_drift_tests {
    use super::*;

    #[test]
    fn validate_tensor_parallel_scope_bails_when_no_gpus_collected() {
        assert!(validate_tensor_parallel_scope(0, Some(4)).is_err());
    }

    #[test]
    fn test_validate_tensor_parallel_scope_blind_spot() {
        let err = validate_tensor_parallel_scope(2, Some(4)).unwrap_err();
        assert!(
            err.to_string()
                .contains("TP 4 requires 4 GPUs but only 2 visible")
        );
    }

    #[test]
    fn test_validate_tensor_parallel_scope_dilution() {
        let err = validate_tensor_parallel_scope(8, Some(4)).unwrap_err();
        assert!(err.to_string().contains("Profile sees 8 GPUs but TP is 4"));
        assert!(err.to_string().contains("Restrict GPU visibility"));
    }

    #[test]
    fn test_validate_tensor_parallel_scope_ok() {
        assert!(validate_tensor_parallel_scope(4, Some(4)).is_ok());
        assert!(validate_tensor_parallel_scope(2, None).is_ok());
    }

    #[test]
    fn test_check_topology_drift_ok() {
        let base = vec![GpuIdentity::Index(0), GpuIdentity::Index(1)];
        let fp = vec![GpuIdentity::Index(0), GpuIdentity::Index(1)];
        assert!(check_topology_drift(&base, &fp).is_ok());
    }

    #[test]
    fn test_check_topology_drift_missing_and_new() {
        let base = vec![GpuIdentity::Index(0), GpuIdentity::Index(1)];
        let fp = vec![GpuIdentity::Index(0), GpuIdentity::Index(2)];
        let err = check_topology_drift(&base, &fp).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Topology drift detected:"));
        assert!(msg.contains("missing [GPU-1]"));
        assert!(msg.contains("new [GPU-2]"));
    }

    #[test]
    fn effective_tensor_parallel_infers_from_collected_when_config_unset() {
        assert_eq!(effective_tensor_parallel(None, 8), Some(8));
        assert_eq!(effective_tensor_parallel(None, 2), Some(2));
        assert_eq!(effective_tensor_parallel(None, 0), None);
    }

    #[test]
    fn effective_tensor_parallel_clamps_config_to_collected() {
        assert_eq!(effective_tensor_parallel(Some(4), 2), Some(2));
        assert_eq!(effective_tensor_parallel(Some(2), 4), Some(2));
    }

    #[test]
    fn aggregate_gpu_temperature_peak_is_max_without_sentinel() {
        let snap = RawSnapshot {
            gpus: vec![
                GpuRawMetrics {
                    temperature_peak_c: Some(70.0),
                    ..Default::default()
                },
                GpuRawMetrics {
                    temperature_peak_c: Some(88.0),
                    ..Default::default()
                },
                GpuRawMetrics {
                    temperature_peak_c: None,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        assert_eq!(snap.aggregate_gpu().temperature_peak_c, Some(88.0));
    }
}
