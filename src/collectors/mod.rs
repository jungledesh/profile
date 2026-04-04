//! GPU + optional vLLM `/metrics` scrape.

pub mod gpu;
pub mod types;
pub mod vllm;

pub use types::{GpuRawMetrics, RawSnapshot, VllmRawMetrics};

pub fn collect_snapshot(
    vllm_base_url: &str,
    max_num_seqs_from_cli: u32,
) -> anyhow::Result<RawSnapshot> {
    let mut vllm = vllm::collect_vllm_metrics(vllm_base_url)?;
    if vllm.max_num_seqs.is_none() {
        vllm.max_num_seqs = Some(max_num_seqs_from_cli);
    }
    let gpu = gpu::collect_gpu_metrics()?;

    Ok(RawSnapshot {
        timestamp: std::time::SystemTime::now(),
        vllm,
        gpu,
    })
}
