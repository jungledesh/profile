//! GPU backends. NVIDIA (NVML) is cross-platform; AMD (libdrm/amdgpu) is not
//! compiled for Windows (no `std::os::fd` / DRM).

/// AMD feature is meaningful off Windows. On Windows the feature may be enabled
/// in Cargo.toml defaults but the backend is not compiled.
#[cfg(all(feature = "amd", not(target_os = "windows")))]
mod amd;
#[cfg(feature = "nvidia")]
mod nvidia;
mod polling;

#[cfg(not(any(feature = "nvidia", all(feature = "amd", not(target_os = "windows")))))]
compile_error!(
    "profile requires a GPU backend: feature `nvidia`, or `amd` on non-Windows (libdrm/amdgpu)"
);

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
    #[cfg(all(feature = "nvidia", feature = "amd", not(target_os = "windows")))]
    {
        nvidia::scan_host_gpus()
            .filter(|v| !v.is_empty())
            .or_else(amd::scan_host_gpus)
    }
    #[cfg(all(
        feature = "nvidia",
        not(all(feature = "amd", not(target_os = "windows")))
    ))]
    {
        nvidia::scan_host_gpus().filter(|v| !v.is_empty())
    }
    #[cfg(all(feature = "amd", not(target_os = "windows"), not(feature = "nvidia")))]
    {
        amd::scan_host_gpus()
    }
}

pub fn collect_gpu_metrics_for(
    window: Duration,
    explicit_indices: Option<&[u32]>,
) -> Result<(Vec<GpuRawMetrics>, SystemTime, Option<u32>)> {
    // Probe order unchanged when both exist: NVIDIA first, AMD if NVML absent/empty.
    #[cfg(all(feature = "nvidia", feature = "amd", not(target_os = "windows")))]
    {
        let (metrics, ts, host_count) = nvidia::collect(window, explicit_indices)?;
        if host_count.is_none_or(|c| c == 0) {
            amd::collect(window, explicit_indices)
        } else {
            Ok((metrics, ts, host_count))
        }
    }
    #[cfg(all(
        feature = "nvidia",
        not(all(feature = "amd", not(target_os = "windows")))
    ))]
    {
        nvidia::collect(window, explicit_indices)
    }
    #[cfg(all(feature = "amd", not(target_os = "windows"), not(feature = "nvidia")))]
    {
        amd::collect(window, explicit_indices)
    }
}

/// Whether the host has the vendor toolchain needed for FP8 KV cache.
pub fn fp8_compiler_available() -> bool {
    #[cfg(all(feature = "nvidia", feature = "amd", not(target_os = "windows")))]
    {
        if nvidia::host_gpu_count().is_some_and(|c| c > 0) {
            nvidia::fp8_compiler_available()
        } else {
            amd::fp8_compiler_available()
        }
    }
    #[cfg(all(
        feature = "nvidia",
        not(all(feature = "amd", not(target_os = "windows")))
    ))]
    {
        nvidia::fp8_compiler_available()
    }
    #[cfg(all(feature = "amd", not(target_os = "windows"), not(feature = "nvidia")))]
    {
        amd::fp8_compiler_available()
    }
}
