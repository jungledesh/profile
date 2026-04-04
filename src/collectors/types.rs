use std::time::SystemTime;

/// vLLM Prometheus scrape
#[derive(Debug, Clone, Default)]
pub struct VllmRawMetrics {
    pub model_name: Option<String>,

    // Raw gauges
    pub num_requests_running: Option<f64>,
    pub num_requests_waiting: Option<f64>,
    pub kv_cache_usage_perc: Option<f64>,

    // Histograms: prefer Δsum/Δcount from **first** → **last** scrape (8th sample, ~2s apart);
    // else cumulative mean from the last scrape.
    pub ttft_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub prefill_latency_ms: Option<f64>,
    pub queue_delay_ms: Option<f64>,
    /// `request_prompt_tokens` histogram: mean tokens (Δ window or last-scrape fallback).
    pub prompt_tokens_mean: Option<f64>,

    /// Cumulative generation tokens (last scrape), summed over label sets.
    pub generation_tokens_total: Option<f64>,
    /// Δ generation tokens / s over the first→last scrape window (output throughput).
    pub generation_tokens_per_sec: Option<f64>,
    /// Cumulative hits / queries from the last scrape (internal + external; 0.0–1.0).
    pub prefix_cache_hit_rate: Option<f64>,

    // Not always available
    pub max_num_seqs: Option<u32>,
}

impl VllmRawMetrics {
    pub fn has_scrape_data(&self) -> bool {
        self.model_name.is_some()
            || self.num_requests_running.is_some()
            || self.num_requests_waiting.is_some()
            || self.kv_cache_usage_perc.is_some()
            || self.ttft_ms.is_some()
            || self.tpot_ms.is_some()
            || self.prefill_latency_ms.is_some()
            || self.queue_delay_ms.is_some()
            || self.prompt_tokens_mean.is_some()
            || self.generation_tokens_total.is_some()
            || self.generation_tokens_per_sec.is_some()
            || self.prefix_cache_hit_rate.is_some()
    }
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
    pub timestamp: SystemTime,
    pub vllm: VllmRawMetrics,
    pub gpu: GpuRawMetrics,
}

impl RawSnapshot {
    pub fn is_empty(&self) -> bool {
        !self.vllm.has_scrape_data() && self.gpu.gpu_util_pct.is_none()
    }
}
