//! GPU + vLLM `/metrics` scrape.
//!
//! **Parallel:** NVML and `/metrics` run **concurrently** (`std::thread`). Cadence: `sampling`.

pub mod gpu;
pub mod sampling;
pub mod types;
pub mod vllm;

pub use types::{
    window_is_evaluable, GpuRawMetrics, PrefixCacheScrapeSample, RawSnapshot, VllmRawMetrics,
};

use std::thread;
use std::time::Duration;

pub fn collect_snapshot(
    vllm_metrics_input: &str,
    max_num_seqs_from_cli: u32,
) -> anyhow::Result<RawSnapshot> {
    collect_snapshot_for_window(
        vllm_metrics_input,
        max_num_seqs_from_cli,
        Duration::from_secs(2),
    )
}

pub fn collect_snapshot_for_window(
    vllm_metrics_input: &str,
    max_num_seqs_from_cli: u32,
    window: Duration,
) -> anyhow::Result<RawSnapshot> {
    let url = vllm_metrics_input.to_string();

    let gpu_handle = thread::spawn(move || gpu::collect_gpu_metrics_for(window));
    let vllm_handle = thread::spawn(move || vllm::collect_vllm_metrics_for(&url, window));

    let (gpu, gpu_observed_at) = gpu_handle
        .join()
        .map_err(|_| anyhow::anyhow!("GPU collector panicked"))??;
    let (mut vllm, vllm_observed_at) = vllm_handle
        .join()
        .map_err(|_| anyhow::anyhow!("vLLM collector panicked"))??;

    if vllm.max_num_seqs.is_none() {
        vllm.max_num_seqs = Some(max_num_seqs_from_cli);
    }

    Ok(RawSnapshot {
        gpu_observed_at,
        vllm_observed_at,
        timestamp: std::time::SystemTime::now(),
        vllm,
        gpu,
    })
}
