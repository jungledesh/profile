#[cfg(feature = "amd")]
mod amd;
#[cfg(feature = "nvidia")]
mod nvidia;
mod polling;

#[cfg(not(any(feature = "nvidia", feature = "amd")))]
compile_error!("profile requires at least one GPU backend feature: `nvidia` or `amd`");

use std::time::{Duration, SystemTime};

use anyhow::Result;

use super::GpuRawMetrics;

/// Pre-start GPU snapshot for assignment heuristics.
/// Lightweight: no polling loop, single-shot read.
pub struct GpuScanEntry {
    pub idx: u32,
    pub name: String,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub pids: Vec<u32>,
}

/// Single-shot scan of all GPUs on host. Used by gpu_assignment before profiling starts.
/// Returns None if no GPU driver is available.
pub fn scan_host_gpus() -> Option<Vec<GpuScanEntry>> {
    #[cfg(all(feature = "nvidia", feature = "amd"))]
    {
        nvidia::scan_host_gpus()
            .filter(|v| !v.is_empty())
            .or_else(amd::scan_host_gpus)
    }
    #[cfg(all(feature = "nvidia", not(feature = "amd")))]
    {
        nvidia::scan_host_gpus().filter(|v| !v.is_empty())
    }
    #[cfg(all(feature = "amd", not(feature = "nvidia")))]
    {
        amd::scan_host_gpus()
    }
}

pub fn collect_gpu_metrics_for(
    window: Duration,
    explicit_indices: Option<&[u32]>,
) -> Result<(Vec<GpuRawMetrics>, SystemTime, Option<u32>)> {
    #[cfg(all(feature = "nvidia", feature = "amd"))]
    {
        let (metrics, ts, host_count) = nvidia::collect(window, explicit_indices)?;
        if host_count.is_none_or(|c| c == 0) {
            return amd::collect(window, explicit_indices);
        }
        Ok((metrics, ts, host_count))
    }
    #[cfg(all(feature = "nvidia", not(feature = "amd")))]
    {
        nvidia::collect(window, explicit_indices)
    }
    #[cfg(all(feature = "amd", not(feature = "nvidia")))]
    {
        amd::collect(window, explicit_indices)
    }
}

/// Whether the host has the vendor toolchain needed for FP8 KV cache.
pub fn fp8_compiler_available() -> bool {
    #[cfg(all(feature = "nvidia", feature = "amd"))]
    {
        if nvidia::host_gpu_count().is_some_and(|c| c > 0) {
            return nvidia::fp8_compiler_available();
        }
        amd::fp8_compiler_available()
    }
    #[cfg(all(feature = "nvidia", not(feature = "amd")))]
    {
        nvidia::fp8_compiler_available()
    }
    #[cfg(all(feature = "amd", not(feature = "nvidia")))]
    {
        amd::fp8_compiler_available()
    }
}
