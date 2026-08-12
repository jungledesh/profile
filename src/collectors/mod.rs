//! GPU + vLLM `/metrics` scrape.
//!
//! **Parallel:** GPU and `/metrics` run **concurrently** (`std::thread`). Cadence: `sampling`.

pub mod config;
pub mod gpu;
pub mod host_memory;
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
    HistogramWindowMass, HostMemoryFacts, KvOffloadState, PrefixCacheScrapeSample, RawSnapshot,
    VllmRawMetrics, effective_tensor_parallel, mib_to_decimal_gb, snapshot_uses_display_names,
    snapshot_uses_index_only, window_is_active, window_is_evaluable, window_is_idle,
};
pub(crate) use vllm::merge_p99_bucket_vecs;

use std::sync::OnceLock;
use std::thread;
use std::time::Duration;

/// Shared blocking client. Success is memoized; a failed build is retried next call
/// (never cache `None` — that would kill the session after a transient failure).
static SHARED_HTTP_CLIENT: OnceLock<reqwest::blocking::Client> = OnceLock::new();

pub(crate) fn shared_http_client() -> Option<&'static reqwest::blocking::Client> {
    if let Some(client) = SHARED_HTTP_CLIENT.get() {
        return Some(client);
    }
    let client = reqwest::blocking::Client::builder()
        .use_rustls_tls()
        .build()
        .ok()?;
    let _ = SHARED_HTTP_CLIENT.set(client);
    SHARED_HTTP_CLIENT.get()
}

pub fn collect_snapshot_for_window(
    vllm_metrics_input: &str,
    window: Duration,
    tensor_parallel_size: u32,
    gpu_indices: &[u32],
    known_max_num_seqs: Option<u32>,
) -> anyhow::Result<RawSnapshot> {
    let url = vllm_metrics_input.to_string();
    let indices = gpu_indices.to_vec();

    let gpu_handle = thread::spawn(move || gpu::collect_gpu_metrics_for(window, Some(&indices)));
    let vllm_handle =
        thread::spawn(move || vllm::collect_vllm_metrics_for(&url, window, known_max_num_seqs));

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

    let host_memory = host_memory::read_host_memory_facts();

    Ok(RawSnapshot {
        gpu_observed_at,
        vllm_observed_at,
        timestamp: std::time::SystemTime::now(),
        vllm,
        gpus,
        host_memory,
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
            host_memory: None,
        }
    }
}
