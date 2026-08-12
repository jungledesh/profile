use std::sync::OnceLock;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::{Clock, ClockId, TemperatureSensor};

use super::super::GpuRawMetrics;
use super::super::sampling::{run_sampling_loop, sample_count_for};
use super::polling::{GpuPoll, PollAggregateState, resolve_device_indices};

const MIB: u64 = 1024 * 1024;

static NVML_SESSION: OnceLock<Nvml> = OnceLock::new();

/// Success-only session cache. Init failure is not memoized, so a driver that
/// comes up after window 1 is retried on the next window (old per-window behavior).
fn session_nvml() -> Option<&'static Nvml> {
    if let Some(nvml) = NVML_SESSION.get() {
        return Some(nvml);
    }
    let nvml = Nvml::init().ok()?;
    let _ = NVML_SESSION.set(nvml);
    NVML_SESSION.get()
}

#[cfg(all(feature = "amd", not(target_os = "windows")))]
/// Raw host GPU count from NVML. No CVD filtering, no polling.
/// Returns None if NVML unavailable.
pub(super) fn host_gpu_count() -> Option<u32> {
    session_nvml()?.device_count().ok()
}

/// Single-shot NVML scan for gpu_assignment. No polling, no window.
pub(super) fn scan_host_gpus() -> Option<Vec<super::GpuScanEntry>> {
    let nvml = session_nvml()?;
    let host_count = nvml.device_count().ok()?;
    let mut out = Vec::with_capacity(host_count as usize);
    for idx in 0..host_count {
        let device = nvml.device_by_index(idx).ok();
        let name = device
            .as_ref()
            .and_then(|d| d.name().ok())
            .unwrap_or_else(|| "GPU".to_string());
        let mem = device.as_ref().and_then(|d| d.memory_info().ok());
        let vram_used_mb = mem.as_ref().map(|m| m.used / MIB).unwrap_or(0);
        let vram_total_mb = mem.map(|m| m.total / MIB).unwrap_or(0);
        let pids = device
            .as_ref()
            .and_then(|d| d.running_compute_processes().ok())
            .map(|procs| procs.iter().map(|p| p.pid).collect())
            .unwrap_or_default();
        out.push(super::GpuScanEntry {
            idx,
            name,
            vram_used_mb,
            vram_total_mb,
            pids,
        });
    }
    Some(out)
}

/// True when `/usr/local/cuda/bin/nvcc` exists on this host.
pub(super) fn fp8_compiler_available() -> bool {
    std::path::Path::new("/usr/local/cuda/bin/nvcc").exists()
}

/// Returns `(metrics, observed_at, host_count)` after the last NVML poll for the requested window.
pub(super) fn collect(
    window: Duration,
    explicit_indices: Option<&[u32]>,
) -> Result<(Vec<GpuRawMetrics>, SystemTime, Option<u32>)> {
    let Some(nvml) = session_nvml() else {
        return Ok((vec![], SystemTime::now(), None));
    };

    let host_count = nvml.device_count().unwrap_or(0);
    if host_count == 0 {
        return Ok((vec![], SystemTime::now(), Some(0)));
    }

    let device_indices: Vec<u32> = if let Some(ei) = explicit_indices {
        ei.to_vec()
    } else {
        // Not shared with AMD: CUDA_VISIBLE_DEVICES allows UUID/MIG tokens
        // (resolved via NVML); AMD ROCR/HIP lists are numeric indices only.
        let mut cvd_indices = Vec::new();
        if let Ok(cvd) = std::env::var("CUDA_VISIBLE_DEVICES") {
            for part in cvd.split(',') {
                let part = part.trim();
                if let Ok(idx) = part.parse::<u32>() {
                    cvd_indices.push(idx);
                } else if (part.starts_with("GPU-") || part.starts_with("MIG-"))
                    && let Ok(device) = nvml.device_by_uuid(part)
                    && let Ok(idx) = device.index()
                {
                    cvd_indices.push(idx);
                }
            }
        }
        resolve_device_indices(cvd_indices, host_count)
    };

    if device_indices.is_empty() {
        return Ok((vec![], SystemTime::now(), Some(host_count)));
    }

    let sample_count = sample_count_for(window);
    let mut all_device_polls: std::collections::HashMap<u32, PollAggregateState> =
        std::collections::HashMap::new();
    for &d in &device_indices {
        all_device_polls.insert(d, PollAggregateState::default());
    }

    run_sampling_loop(sample_count, |_i| {
        for &d in &device_indices {
            let device = nvml.device_by_index(d).map_err(|e| {
                anyhow::anyhow!("NVML device {d} lost during telemetry polling: {e}")
            })?;
            let mut tick = GpuPoll::default();

            if let Ok(u) = device.utilization_rates() {
                tick.util_gpu = Some(u.gpu);
                tick.util_mem = Some(u.memory);
            }

            if let Ok(mw) = device.power_usage() {
                tick.power_watts = Some(mw as f64 / 1000.0);
            }

            if let Ok(mem) = device.memory_info() {
                tick.vram_used_mb = Some(mem.used / MIB);
                tick.vram_total_mb = Some(mem.total / MIB);
            }

            if let Ok(t) = device.temperature(TemperatureSensor::Gpu) {
                tick.temperature_c = Some(f64::from(t));
            }

            if let Ok(mhz) = device.clock(Clock::SM, ClockId::Current) {
                tick.sm_clock_mhz = Some(mhz);
            }

            if let Some(slot) = all_device_polls.get_mut(&d) {
                slot.update(&tick);
            }
        }
        Ok(())
    })?;

    let mut results = Vec::with_capacity(device_indices.len());
    let mut failed_indices = Vec::new();

    for &d in &device_indices {
        match nvml.device_by_index(d) {
            Ok(device) => {
                let gpu_name = device.name().ok();
                let gpu_index = device.index().ok();
                let gpu_uuid = device.uuid().ok();
                let pcie_bus_id = device.pci_info().ok().map(|p| p.bus_id.to_string());
                let power_limit_watts = device
                    .power_management_limit()
                    .ok()
                    .map(|mw| mw as f64 / 1000.0);

                let agg = all_device_polls
                    .get(&d)
                    .map(PollAggregateState::finish)
                    .unwrap_or_else(|| PollAggregateState::default().finish());
                results.push(agg.into_gpu_raw_metrics(
                    gpu_name,
                    gpu_index,
                    gpu_uuid,
                    pcie_bus_id,
                    power_limit_watts,
                ));
            }
            Err(_) => {
                failed_indices.push(d);
            }
        }
    }

    if !failed_indices.is_empty() {
        anyhow::bail!(
            "NVML failed to read telemetry for requested GPU indices: {:?}",
            failed_indices
        );
    }

    Ok((results, SystemTime::now(), Some(host_count)))
}
