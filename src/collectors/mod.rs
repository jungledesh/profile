//! GPU + vLLM `/metrics` scrape.
//!
//! **Parallel:** GPU and `/metrics` run **concurrently** (`std::thread`). Cadence: `sampling`.

pub mod config;
pub mod gpu;
pub mod sampling;
pub mod types;
pub mod vllm;

#[cfg(test)]
mod test_support;
#[cfg(test)]
pub(crate) use test_support::{RawSnapshotFixture, snap_vllm};

pub use config::{VllmConfig, build_config};
pub(crate) use types::observations_aligned;
pub use types::{
    AggregateGpuMetrics, CacheConfigLabels, GpuFingerprint, GpuRawMetrics, HistogramCount,
    HistogramWindowMass, PrefixCacheScrapeSample, RawSnapshot, VllmRawMetrics,
    effective_tensor_parallel, mib_to_decimal_gb, snapshot_uses_display_names,
    snapshot_uses_index_only, window_is_active, window_is_evaluable, window_is_idle,
};
pub(crate) use vllm::merge_p99_bucket_vecs;

use std::thread;
use std::time::Duration;

pub fn collect_snapshot_for_window(
    vllm_metrics_input: &str,
    window: Duration,
    tensor_parallel_size: u32,
    gpu_indices: &[u32],
) -> anyhow::Result<RawSnapshot> {
    let url = vllm_metrics_input.to_string();
    let indices = gpu_indices.to_vec();

    let gpu_handle = thread::spawn(move || gpu::collect_gpu_metrics_for(window, Some(&indices)));
    let vllm_handle = thread::spawn(move || vllm::collect_vllm_metrics_for(&url, window));

    let (mut gpus, gpu_observed_at, _) = gpu_handle
        .join()
        .map_err(|_| anyhow::anyhow!("GPU collector panicked"))??;

    types::validate_tensor_parallel_scope(
        u32::try_from(gpus.len()).unwrap_or(u32::MAX),
        Some(tensor_parallel_size),
    )?;

    types::validate_gpu_identities(&gpus)?;

    gpus.sort_by_key(|a| a.identity());

    let (vllm, vllm_observed_at) = vllm_handle
        .join()
        .map_err(|_| anyhow::anyhow!("vLLM collector panicked"))??;

    Ok(RawSnapshot {
        gpu_observed_at,
        vllm_observed_at,
        timestamp: std::time::SystemTime::now(),
        vllm,
        gpus,
    })
}

#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::types::{GpuRawMetrics, RawSnapshot};
    use std::time::SystemTime;

    pub fn snap_with_gpu_indices(indices: &[u32]) -> RawSnapshot {
        RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: Default::default(),
            gpus: indices
                .iter()
                .map(|&gpu_index| GpuRawMetrics {
                    gpu_index: Some(gpu_index),
                    ..Default::default()
                })
                .collect(),
        }
    }
}
