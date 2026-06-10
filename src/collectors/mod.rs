//! GPU + vLLM `/metrics` scrape.
//!
//! **Parallel:** NVML and `/metrics` run **concurrently** (`std::thread`). Cadence: `sampling`.

pub mod config;
pub mod gpu;
pub mod sampling;
pub mod traffic;
pub mod types;
pub mod vllm;

pub use config::{build_config, VllmConfig};
pub use traffic::{traffic_from_snapshot, TrafficSource, TrafficState};
pub use types::{
    window_is_active, window_is_evaluable, CacheConfigLabels, GpuRawMetrics, HistogramCount,
    HistogramWindowMass, PrefixCacheScrapeSample, RawSnapshot, VllmRawMetrics,
};
pub(crate) use vllm::merge_p99_bucket_vecs;

use std::thread;
use std::time::Duration;

pub fn collect_snapshot_for_window(
    vllm_metrics_input: &str,
    window: Duration,
) -> anyhow::Result<RawSnapshot> {
    let url = vllm_metrics_input.to_string();

    let gpu_handle = thread::spawn(move || gpu::collect_gpu_metrics_for(window));
    let vllm_handle = thread::spawn(move || vllm::collect_vllm_metrics_for(&url, window));

    let (gpu, gpu_observed_at) = gpu_handle
        .join()
        .map_err(|_| anyhow::anyhow!("GPU collector panicked"))??;
    let (vllm, vllm_observed_at) = vllm_handle
        .join()
        .map_err(|_| anyhow::anyhow!("vLLM collector panicked"))??;

    Ok(RawSnapshot {
        gpu_observed_at,
        vllm_observed_at,
        timestamp: std::time::SystemTime::now(),
        vllm,
        gpu,
    })
}
