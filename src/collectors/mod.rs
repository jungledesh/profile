//! GPU + optional vLLM `/metrics` scrape.

pub mod gpu;
pub mod types;
pub mod vllm;

pub use types::{GpuRawMetrics, RawSnapshot, VllmRawMetrics};

pub fn collect_snapshot(vllm_base_url: &str) -> anyhow::Result<RawSnapshot> {
    let vllm = vllm::collect_vllm_metrics(vllm_base_url)?;
    let gpu = gpu::collect_gpu_metrics()?;

    Ok(RawSnapshot {
        timestamp: std::time::SystemTime::now(),
        vllm,
        gpu,
    })
}
