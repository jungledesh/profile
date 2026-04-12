//! GPU + vLLM `/metrics` scrape.
//!
//! **Parallel:** NVML and `/metrics` run **concurrently** (`std::thread`). Cadence: `sampling`.

pub mod gpu;
pub mod sampling;
pub mod types;
pub mod vllm;

pub use types::{GpuRawMetrics, PrefixCacheScrapeSample, RawSnapshot, VllmRawMetrics};

use std::thread;

pub fn collect_snapshot(
    vllm_metrics_input: &str,
    max_num_seqs_from_cli: u32,
) -> anyhow::Result<RawSnapshot> {
    let url = vllm_metrics_input.to_string();

    let gpu_handle = thread::spawn(gpu::collect_gpu_metrics);
    let vllm_handle = thread::spawn(move || vllm::collect_vllm_metrics(&url));

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
