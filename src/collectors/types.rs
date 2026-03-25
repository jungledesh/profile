use std::time::SystemTime;

/// vLLM Prometheus scrape
#[derive(Debug, Clone, Default)]
pub struct VllmRawMetrics {
    pub model_name: Option<String>,
    pub tps: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub prefill_latency_ms: Option<f64>,
    pub avg_batch_size: Option<f64>,
    pub max_num_seqs: Option<u32>,
    pub queue_delay_ms: Option<f64>,
    pub num_requests_waiting: Option<f64>,
    pub kv_cache_usage_pct: Option<f64>,
}

/// NVML / DCGM / nvidia-smi scrape
#[derive(Debug, Clone, Default)]
pub struct GpuRawMetrics {
    pub gpu_name: Option<String>,
    pub gpu_util_pct: Option<f64>,
    pub mem_util_pct: Option<f64>,
    pub power_watts: Option<f64>,
    pub power_limit_watts: Option<f64>,
    pub memory_used_mb: Option<u64>,
    pub memory_total_mb: Option<u64>,
    pub temperature_c: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct RawSnapshot {
    pub timestamp: SystemTime,
    pub vllm: VllmRawMetrics,
    pub gpu: GpuRawMetrics,
}

impl RawSnapshot {
    pub fn is_empty(&self) -> bool {
        self.vllm.tps.is_none() && self.gpu.gpu_util_pct.is_none()
    }
}
