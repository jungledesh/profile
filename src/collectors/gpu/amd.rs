use std::panic::{self, AssertUnwindSafe};
use std::time::{Duration, SystemTime};

use anyhow::Result;
use libamdgpu_top::AMDGPU::DeviceHandle;
use libamdgpu_top::AMDGPU::MetricsInfo;
use libamdgpu_top::AMDGPU::SENSOR_INFO::SENSOR_TYPE;
use libamdgpu_top::DevicePath;
use libamdgpu_top::VramUsage;
use libamdgpu_top::stat::{GpuActivity, Sensors};

use super::super::GpuRawMetrics;
use super::super::sampling::{run_sampling_loop, sample_count_for};
use super::polling::{GpuPoll, aggregate_polls};

const MIB: u64 = 1024 * 1024;

fn amdgpu_device_paths() -> Option<Vec<DevicePath>> {
    let paths = match panic::catch_unwind(AssertUnwindSafe(DevicePath::get_device_path_list)) {
        Ok(paths) => paths,
        Err(_) => {
            eprintln!("Warning: AMD GPU driver probe panicked. Skipping AMD detection.");
            return None;
        }
    };
    let paths: Vec<_> = paths
        .into_iter()
        .filter(|p| p.is_amdgpu() && p.render.exists())
        .collect();
    if paths.is_empty() { None } else { Some(paths) }
}

/// Successfully initialized AMDGPU path, keyed by original probe ordinal.
/// Carries the open `DeviceHandle` from the inventory probe so scan/collect
/// do not call `init()` a second time on the same device.
struct AmdReadyPath {
    original_index: u32,
    device_path: DevicePath,
    device_handle: DeviceHandle,
}

/// Probe ordinal that failed `init()`, kept for visible-device warnings.
struct AmdFailedPath {
    original_index: u32,
    name: String,
}

struct AmdDeviceInventory {
    ready: Vec<AmdReadyPath>,
    failed: Vec<AmdFailedPath>,
}

/// Single source of truth for scan and collect: paths that successfully init.
/// Original path indices are preserved so assignment and telemetry agree.
fn amd_device_inventory() -> Option<AmdDeviceInventory> {
    let paths = amdgpu_device_paths()?;
    let mut ready = Vec::new();
    let mut failed = Vec::new();
    for (i, device_path) in paths.into_iter().enumerate() {
        let original_index = i as u32;
        let name = device_path.device_name.clone();
        match panic::catch_unwind(AssertUnwindSafe(|| device_path.init())) {
            Ok(Ok(device_handle)) => {
                ready.push(AmdReadyPath {
                    original_index,
                    device_path,
                    device_handle,
                });
            }
            Err(_) => {
                eprintln!("Warning: AMD GPU {original_index} init panicked. Skipping device.");
                failed.push(AmdFailedPath {
                    original_index,
                    name,
                });
            }
            Ok(Err(_)) => {
                failed.push(AmdFailedPath {
                    original_index,
                    name,
                });
            }
        }
    }
    if ready.is_empty() {
        None
    } else {
        Some(AmdDeviceInventory { ready, failed })
    }
}

/// Pure seam: split init outcomes into ready vs failed by original index.
fn partition_amd_init_outcomes(
    outcomes: &[(u32, String, bool)],
) -> (Vec<(u32, String)>, Vec<AmdFailedPath>) {
    let mut ready = Vec::new();
    let mut failed = Vec::new();
    for (original_index, name, ok) in outcomes {
        if *ok {
            ready.push((*original_index, name.clone()));
        } else {
            failed.push(AmdFailedPath {
                original_index: *original_index,
                name: name.clone(),
            });
        }
    }
    (ready, failed)
}

/// Select original path indices for collection.
/// Empty env → all ready. Env/explicit indices map by original ordinal (never by
/// compacted slot), so a visible-devices index for a failed init cannot alias another GPU.
fn select_amd_original_indices(
    explicit: Option<&[u32]>,
    env_indices: Vec<u32>,
    ready: &[(u32, String)],
    failed: &[AmdFailedPath],
) -> Vec<u32> {
    let requested: Vec<u32> = if let Some(ei) = explicit {
        ei.to_vec()
    } else if env_indices.is_empty() {
        ready.iter().map(|(i, _)| *i).collect()
    } else {
        env_indices
    };

    let mut selected = Vec::with_capacity(requested.len());
    for idx in requested {
        if ready.iter().any(|(i, _)| *i == idx) {
            selected.push(idx);
            continue;
        }
        if let Some(f) = failed.iter().find(|f| f.original_index == idx) {
            eprintln!(
                "Warning: AMD GPU {} ({}) is not available (init failed). Skipping.",
                f.original_index, f.name
            );
            continue;
        }
        eprintln!("Warning: AMD GPU {idx} is out of range for initialized devices. Skipping.");
    }
    selected
}

/// Single-shot AMD GPU scan for gpu_assignment. No polling, no window.
pub(super) fn scan_host_gpus() -> Option<Vec<super::GpuScanEntry>> {
    let inventory = amd_device_inventory()?;
    let mut out = Vec::with_capacity(inventory.ready.len());
    for entry in inventory.ready {
        let i = entry.original_index;
        let device_path = &entry.device_path;
        let mem_info = entry.device_handle.memory_info().ok();
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
            idx: i,
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
    sensors: Option<Sensors>,
    vram_usage: VramUsage,
}

fn init_amd_devices(
    ready: &mut std::collections::HashMap<u32, AmdReadyPath>,
    original_indices: &[u32],
) -> Result<Vec<AmdDevice>> {
    let mut devices = Vec::with_capacity(original_indices.len());
    for &idx in original_indices {
        let Some(entry) = ready.remove(&idx) else {
            anyhow::bail!("AMD GPU {idx} lost during telemetry polling: index out of range");
        };
        let device_path = entry.device_path;
        let device_handle = entry.device_handle;
        let ext_info = device_handle
            .device_info()
            .map_err(|e| anyhow::anyhow!("AMD GPU {idx} lost during telemetry polling: {e}"))?;
        // No bail. sensors == None means Stage 1 (hwmon) unavailable; Stage 2/3 still run.
        let sensors = Sensors::new(&device_handle, &device_path.pci, &ext_info);
        let mem_info = device_handle
            .memory_info()
            .map_err(|e| anyhow::anyhow!("AMD GPU {idx} lost during telemetry polling: {e}"))?;
        let mut vram_usage = VramUsage::new(&mem_info);
        vram_usage.update_usable_heap_size(&device_handle);
        devices.push(AmdDevice {
            device_path,
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

    if let Some(ref mut sensors) = device.sensors {
        sensors.update_without_device_handle();
        sensors.update(&device.device_handle);
    }

    // Stage 1: hwmon (only if sensors available).
    // HwmonPower.value is already watts; HwmonTemp.current is already °C.
    if let Some(ref sensors) = device.sensors {
        tick.power_watts = sanitize_watts(
            sensors
                .average_power
                .as_ref()
                .or(sensors.input_power.as_ref())
                .map(|p| f64::from(p.value)),
        );

        let temp = sensors
            .junction_temp
            .as_ref()
            .or(sensors.edge_temp.as_ref());
        tick.temperature_c = sanitize_temp(temp.map(|t| t.current as f64));
    }

    // Stage 2: gpu_metrics binary blob (sysfs). Native watts / °C.
    if (tick.power_watts.is_none() || tick.temperature_c.is_none())
        && let Ok(metrics) = device.device_handle.get_gpu_metrics()
    {
        if tick.power_watts.is_none() {
            tick.power_watts = sanitize_watts(metrics.get_average_socket_power().map(f64::from));
        }
        if tick.temperature_c.is_none() {
            tick.temperature_c = sanitize_temp(metrics.get_temperature_hotspot().map(f64::from));
        }
    }

    // Stage 3: DRM ioctl via render node. Power only; GPU_TEMP is -EOPNOTSUPP on MI300X.
    if tick.power_watts.is_none() {
        tick.power_watts = sanitize_watts(
            device
                .device_handle
                .sensor_info(SENSOR_TYPE::GPU_AVG_POWER)
                .ok()
                .map(f64::from),
        );
    }

    device.vram_usage.update_usage(&device.device_handle);
    let mem_info = &device.vram_usage.0;
    tick.vram_used_mb = Some(mem_info.vram.heap_usage / MIB);
    tick.vram_total_mb = Some(mem_info.vram.total_heap_size / MIB);

    tick.sm_clock_mhz = device.sensors.as_ref().and_then(|s| s.sclk);

    tick
}

/// Thermodynamic floor for active silicon. Sub-1W readings are sensor noise, not load.
fn sanitize_watts(watts: Option<f64>) -> Option<f64> {
    watts.filter(|&w| w > 1.0)
}

/// Thermodynamic floor for active silicon. Sub-5°C readings are sensor noise, not load.
fn sanitize_temp(temp: Option<f64>) -> Option<f64> {
    temp.filter(|&t| t > 5.0)
}

fn power_limit_watts(sensors: &Sensors) -> Option<f64> {
    // PowerCap.current is already watts.
    sanitize_watts(sensors.power_cap.as_ref().map(|pc| f64::from(pc.current)))
}

/// Returns `(metrics, observed_at, host_count)` after the last poll for the requested window.
pub(super) fn collect(
    window: Duration,
    explicit_indices: Option<&[u32]>,
) -> Result<(Vec<GpuRawMetrics>, SystemTime, Option<u32>)> {
    let Some(inventory) = amd_device_inventory() else {
        return Ok((vec![], SystemTime::now(), None));
    };

    let host_count = inventory.ready.len() as u32;
    let ready_meta: Vec<(u32, String)> = inventory
        .ready
        .iter()
        .map(|e| (e.original_index, e.device_path.device_name.clone()))
        .collect();
    let device_indices = select_amd_original_indices(
        explicit_indices,
        parse_amd_visible_devices(),
        &ready_meta,
        &inventory.failed,
    );

    if device_indices.is_empty() {
        return Ok((vec![], SystemTime::now(), Some(host_count)));
    }

    let mut ready_by_idx: std::collections::HashMap<u32, AmdReadyPath> = inventory
        .ready
        .into_iter()
        .map(|e| (e.original_index, e))
        .collect();
    let mut devices = init_amd_devices(&mut ready_by_idx, &device_indices)?;

    let sample_count = sample_count_for(window);
    let mut all_device_polls: std::collections::HashMap<u32, Vec<GpuPoll>> =
        std::collections::HashMap::new();
    for &d in &device_indices {
        all_device_polls.insert(d, Vec::with_capacity(sample_count));
    }

    run_sampling_loop(sample_count, |_i| {
        for (slot_idx, &d) in device_indices.iter().enumerate() {
            let tick = poll_amd_device(&mut devices[slot_idx]);
            if let Some(slot) = all_device_polls.get_mut(&d) {
                slot.push(tick);
            }
        }
        Ok(())
    })?;

    let mut results = Vec::with_capacity(device_indices.len());
    for (slot_idx, &d) in device_indices.iter().enumerate() {
        let device = &devices[slot_idx];
        let agg = aggregate_polls(all_device_polls.get(&d).map_or(&[], |v| v));
        let power_limit_watts = device.sensors.as_ref().and_then(power_limit_watts);

        results.push(GpuRawMetrics {
            gpu_name: Some(device.device_path.device_name.clone()),
            gpu_index: Some(d),
            gpu_uuid: None,
            pcie_bus_id: Some(device.device_path.pci.to_string()),
            gpu_util_pct: agg.gpu_util_pct,
            mem_util_pct: agg.mem_util_pct,
            power_watts: agg.power_watts,
            aligned_power_watts: None,
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
    // Numeric indices only. Unlike CUDA_VISIBLE_DEVICES, ROCR/HIP do not take
    // UUID tokens here - keep parse local rather than force a shared helper.
    let var = std::env::var("ROCR_VISIBLE_DEVICES")
        .or_else(|_| std::env::var("HIP_VISIBLE_DEVICES"))
        .unwrap_or_default();
    parse_device_indices(&var)
}

#[cfg(test)]
mod tests {
    use super::{
        AmdFailedPath, parse_device_indices, partition_amd_init_outcomes, sanitize_temp,
        sanitize_watts, select_amd_original_indices,
    };

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

    #[test]
    fn sanitize_watts_rejects_noise_and_zero() {
        assert_eq!(sanitize_watts(None), None);
        assert_eq!(sanitize_watts(Some(0.0)), None);
        assert_eq!(sanitize_watts(Some(0.000494)), None);
        assert_eq!(sanitize_watts(Some(1.0)), None);
        assert_eq!(sanitize_watts(Some(1.01)), Some(1.01));
        assert_eq!(sanitize_watts(Some(750.0)), Some(750.0));
    }

    #[test]
    fn sanitize_temp_rejects_noise_and_zero() {
        assert_eq!(sanitize_temp(None), None);
        assert_eq!(sanitize_temp(Some(0.0)), None);
        assert_eq!(sanitize_temp(Some(0.065)), None);
        assert_eq!(sanitize_temp(Some(5.0)), None);
        assert_eq!(sanitize_temp(Some(5.01)), Some(5.01));
        assert_eq!(sanitize_temp(Some(65.0)), Some(65.0));
    }

    #[test]
    fn partial_init_keeps_original_index_and_host_count() {
        // 2 paths, second init fails → 1 ready GPU at original index 0.
        let outcomes = [
            (0_u32, "GPU-0".to_string(), true),
            (1_u32, "GPU-1".to_string(), false),
        ];
        let (ready, failed) = partition_amd_init_outcomes(&outcomes);
        assert_eq!(ready, vec![(0, "GPU-0".to_string())]);
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].original_index, 1);
        let host_count = ready.len() as u32;
        assert_eq!(host_count, 1);
        let selected = select_amd_original_indices(None, vec![], &ready, &failed);
        assert_eq!(selected, vec![0]);
        // Assignment-style explicit index must stay the original ordinal.
        let selected_explicit = select_amd_original_indices(Some(&[0]), vec![], &ready, &failed);
        assert_eq!(selected_explicit, vec![0]);
    }

    #[test]
    fn visible_index_for_failed_init_does_not_alias_ready_gpu() {
        // Paths 0 failed, 1 ok. Env asks for index 0 → warn/skip, never poll GPU 1 as "0".
        let outcomes = [
            (0_u32, "phantom".to_string(), false),
            (1_u32, "real".to_string(), true),
        ];
        let (ready, failed) = partition_amd_init_outcomes(&outcomes);
        assert_eq!(ready, vec![(1, "real".to_string())]);
        let selected = select_amd_original_indices(None, vec![0], &ready, &failed);
        assert!(selected.is_empty());
        let selected_ok = select_amd_original_indices(None, vec![1], &ready, &failed);
        assert_eq!(selected_ok, vec![1]);
    }

    #[test]
    fn all_init_selects_every_original_index() {
        let outcomes = [
            (0_u32, "a".to_string(), true),
            (1_u32, "b".to_string(), true),
        ];
        let (ready, failed) = partition_amd_init_outcomes(&outcomes);
        assert!(failed.is_empty());
        assert_eq!(
            select_amd_original_indices(None, vec![], &ready, &failed),
            vec![0, 1]
        );
        assert_eq!(ready.len() as u32, 2);
    }

    #[test]
    fn zero_init_partition_is_empty() {
        let outcomes: [(u32, String, bool); 0] = [];
        let (ready, failed) = partition_amd_init_outcomes(&outcomes);
        assert!(ready.is_empty());
        assert!(failed.is_empty());
        assert!(select_amd_original_indices(None, vec![], &ready, &failed).is_empty());
    }

    #[test]
    fn failed_path_struct_carries_name_for_warnings() {
        let failed = [AmdFailedPath {
            original_index: 2,
            name: "MI300X".into(),
        }];
        let ready = [(0_u32, "other".to_string())];
        assert!(select_amd_original_indices(None, vec![2], &ready, &failed).is_empty());
    }
}
