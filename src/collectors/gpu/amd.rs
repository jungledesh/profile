use std::panic::{self, AssertUnwindSafe};
use std::thread;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use libamdgpu_top::AMDGPU::DeviceHandle;
use libamdgpu_top::DevicePath;
use libamdgpu_top::VramUsage;
use libamdgpu_top::stat::{GpuActivity, Sensors};

use super::super::GpuRawMetrics;
use super::super::sampling::{SAMPLE_INTERVAL, sample_count_for};
use super::polling::{GpuPoll, aggregate_polls, resolve_device_indices};

const MIB: u64 = 1024 * 1024;

fn amdgpu_device_paths() -> Option<Vec<DevicePath>> {
    let paths = panic::catch_unwind(AssertUnwindSafe(DevicePath::get_device_path_list)).ok()?;
    let paths: Vec<_> = paths
        .into_iter()
        .filter(|p| p.is_amdgpu() && p.render.exists())
        .collect();
    if paths.is_empty() { None } else { Some(paths) }
}

/// Container-visible AMD GPU count. Filtered to render nodes present in /dev/dri/.
/// No env filtering, no polling. Returns None if AMD driver unavailable.
pub(super) fn host_gpu_count() -> Option<u32> {
    let paths = amdgpu_device_paths()?;
    Some(paths.len() as u32)
}

/// Single-shot AMD GPU scan for gpu_assignment. No polling, no window.
pub(super) fn scan_host_gpus() -> Option<Vec<super::GpuScanEntry>> {
    let paths = amdgpu_device_paths()?;
    let mut out = Vec::with_capacity(paths.len());
    for (i, device_path) in paths.iter().enumerate() {
        let init_result = panic::catch_unwind(AssertUnwindSafe(|| device_path.init()));
        let Ok(Ok(device_handle)) = init_result else {
            continue;
        };
        let mem_info = device_handle.memory_info().ok();
        let vram_used_mb = mem_info
            .as_ref()
            .map(|m| m.vram.heap_usage / MIB)
            .unwrap_or(0);
        let vram_total_mb = mem_info.map(|m| m.vram.total_heap_size / MIB).unwrap_or(0);
        // Assumption: profile runs as the same user as vLLM.
        // AMD has no kernel-driver PID query like NVML.
        // PIDs come from /proc/*/fdinfo/, restricted to same-user or root.
        // Multi-GPU / TP auto-detection degrades to the interactive prompt
        // if running as a different user. Revisit when multi-GPU/TP launches on AMD.
        let pids = device_path
            .arc_proc_index
            .lock()
            .ok()
            .map(|procs| procs.iter().map(|p| p.pid as u32).collect())
            .unwrap_or_default();
        out.push(super::GpuScanEntry {
            idx: i as u32,
            name: device_path.device_name.clone(),
            vram_used_mb,
            vram_total_mb,
            pids,
        });
    }
    if out.is_empty() { None } else { Some(out) }
}

/// True when `/opt/rocm/bin/hipcc` or `/opt/rocm/bin/amdclang++` exists on this host.
pub(super) fn fp8_compiler_available() -> bool {
    std::path::Path::new("/opt/rocm/bin/hipcc").exists()
        || std::path::Path::new("/opt/rocm/bin/amdclang++").exists()
}

struct AmdDevice {
    device_path: DevicePath,
    device_handle: DeviceHandle,
    sensors: Sensors,
    vram_usage: VramUsage,
}

fn init_amd_devices(device_paths: &[DevicePath], indices: &[u32]) -> Result<Vec<AmdDevice>> {
    let mut devices = Vec::with_capacity(indices.len());
    for &idx in indices {
        let Some(device_path) = device_paths.get(idx as usize) else {
            anyhow::bail!("AMD GPU {idx} lost during telemetry polling: index out of range");
        };
        let device_handle = panic::catch_unwind(AssertUnwindSafe(|| device_path.init()))
            .map_err(|_| {
                anyhow::anyhow!("AMD GPU {idx}: init panicked (device likely inaccessible)")
            })?
            .map_err(|e| anyhow::anyhow!("AMD GPU {idx} lost during telemetry polling: {e}"))?;
        let ext_info = device_handle
            .device_info()
            .map_err(|e| anyhow::anyhow!("AMD GPU {idx} lost during telemetry polling: {e}"))?;
        let Some(sensors) = Sensors::new(&device_handle, &device_path.pci, &ext_info) else {
            anyhow::bail!("AMD GPU {idx} lost during telemetry polling: sensors unavailable");
        };
        let mem_info = device_handle
            .memory_info()
            .map_err(|e| anyhow::anyhow!("AMD GPU {idx} lost during telemetry polling: {e}"))?;
        let mut vram_usage = VramUsage::new(&mem_info);
        vram_usage.update_usable_heap_size(&device_handle);
        devices.push(AmdDevice {
            device_path: device_path.clone(),
            device_handle,
            sensors,
            vram_usage,
        });
    }
    Ok(devices)
}

fn poll_amd_device(device: &mut AmdDevice) -> GpuPoll {
    let mut tick = GpuPoll::default();

    let activity = GpuActivity::get_from_sysfs(&device.device_path.sysfs_path);
    tick.util_gpu = activity.gfx.map(|v| v as u32);
    tick.util_mem = activity.umc.map(|v| v as u32);

    device.sensors.update_without_device_handle();
    device.sensors.update(&device.device_handle);
    tick.power_watts = device
        .sensors
        .average_power
        .as_ref()
        .or(device.sensors.input_power.as_ref())
        .map(|p| f64::from(p.value) / 1_000_000.0);

    device.vram_usage.update_usage(&device.device_handle);
    let mem_info = &device.vram_usage.0;
    tick.vram_used_mb = Some(mem_info.vram.heap_usage / MIB);
    tick.vram_total_mb = Some(mem_info.vram.total_heap_size / MIB);

    let temp = device
        .sensors
        .junction_temp
        .as_ref()
        .or(device.sensors.edge_temp.as_ref());
    tick.temperature_c = temp.map(|t| t.current as f64 / 1000.0);

    tick.sm_clock_mhz = device.sensors.sclk;

    tick
}

fn power_limit_watts(sensors: &Sensors) -> Option<f64> {
    sensors
        .power_cap
        .as_ref()
        .map(|pc| f64::from(pc.current) / 1_000_000.0)
}

/// Returns `(metrics, observed_at, host_count)` after the last poll for the requested window.
pub(super) fn collect(
    window: Duration,
    explicit_indices: Option<&[u32]>,
) -> Result<(Vec<GpuRawMetrics>, SystemTime, Option<u32>)> {
    let Some(device_paths) = amdgpu_device_paths() else {
        return Ok((vec![], SystemTime::now(), None));
    };

    let host_count = device_paths.len() as u32;

    let device_indices: Vec<u32> = if let Some(ei) = explicit_indices {
        ei.to_vec()
    } else {
        resolve_device_indices(parse_amd_visible_devices(), host_count)
    };

    if device_indices.is_empty() {
        return Ok((vec![], SystemTime::now(), Some(host_count)));
    }

    let mut devices = init_amd_devices(&device_paths, &device_indices)?;

    let sample_count = sample_count_for(window);
    let mut all_device_polls: std::collections::HashMap<u32, Vec<GpuPoll>> =
        std::collections::HashMap::new();
    for &d in &device_indices {
        all_device_polls.insert(d, Vec::with_capacity(sample_count));
    }

    for i in 0..sample_count {
        for (slot_idx, &d) in device_indices.iter().enumerate() {
            let tick = poll_amd_device(&mut devices[slot_idx]);
            if let Some(slot) = all_device_polls.get_mut(&d) {
                slot.push(tick);
            }
        }

        if i + 1 < sample_count {
            thread::sleep(SAMPLE_INTERVAL);
        }
    }

    let mut results = Vec::with_capacity(device_indices.len());
    for (slot_idx, &d) in device_indices.iter().enumerate() {
        let device = &devices[slot_idx];
        let agg = aggregate_polls(all_device_polls.get(&d).map_or(&[], |v| v));
        let power_limit_watts = power_limit_watts(&device.sensors);

        results.push(GpuRawMetrics {
            gpu_name: Some(device.device_path.device_name.clone()),
            gpu_index: Some(d),
            gpu_uuid: None,
            pcie_bus_id: Some(device.device_path.pci.to_string()),
            gpu_util_pct: agg.gpu_util_pct,
            mem_util_pct: agg.mem_util_pct,
            power_watts: agg.power_watts,
            power_limit_watts,
            vram_used_mb: agg.vram_used_mb,
            vram_peak_mb: agg.vram_peak_mb,
            vram_total_mb: agg.vram_total_mb,
            temperature_c: agg.temperature_c,
            temperature_peak_c: agg.temperature_peak_c,
            sm_clock_mhz: agg.sm_clock_mhz,
        });
    }

    Ok((results, SystemTime::now(), Some(host_count)))
}

fn parse_device_indices(var: &str) -> Vec<u32> {
    var.split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect()
}

fn parse_amd_visible_devices() -> Vec<u32> {
    let var = std::env::var("ROCR_VISIBLE_DEVICES")
        .or_else(|_| std::env::var("HIP_VISIBLE_DEVICES"))
        .unwrap_or_default();
    parse_device_indices(&var)
}

#[cfg(test)]
mod tests {
    use super::parse_device_indices;

    #[test]
    fn empty_string_returns_empty() {
        assert_eq!(parse_device_indices(""), Vec::<u32>::new());
    }

    #[test]
    fn single_index() {
        assert_eq!(parse_device_indices("0"), vec![0]);
    }

    #[test]
    fn multiple_indices() {
        assert_eq!(parse_device_indices("0,1,2"), vec![0, 1, 2]);
    }

    #[test]
    fn whitespace_trimmed() {
        assert_eq!(parse_device_indices(" 0 , 1 , 2 "), vec![0, 1, 2]);
    }

    #[test]
    fn non_integers_skipped() {
        assert_eq!(parse_device_indices("0,abc,2"), vec![0, 2]);
    }

    #[test]
    fn negative_values_skipped() {
        // u32 parse rejects negative numbers.
        assert_eq!(parse_device_indices("0,-1,2"), vec![0, 2]);
    }

    #[test]
    fn uuid_style_skipped() {
        // AMD does not support UUID-based device selection.
        assert_eq!(parse_device_indices("GPU-abc123,0"), vec![0]);
    }
}
