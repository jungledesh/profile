use std::thread;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::{Clock, ClockId, TemperatureSensor};

use super::GpuRawMetrics;
#[cfg(test)]
use super::sampling::SAMPLE_COUNT;
use super::sampling::{SAMPLE_INTERVAL, sample_count_for};

const MIB: u64 = 1024 * 1024;

/// Raw host GPU count from NVML. No CVD filtering, no polling.
/// Returns None if NVML unavailable.
pub fn host_gpu_count() -> Option<u32> {
    Nvml::init().ok()?.device_count().ok()
}

#[derive(Default)]
struct GpuPoll {
    util_gpu: Option<u32>,
    util_mem: Option<u32>,
    power_watts: Option<f64>,
    vram_used_mb: Option<u64>,
    vram_total_mb: Option<u64>,
    temperature_c: Option<f64>,
    sm_clock_mhz: Option<u32>,
}

#[derive(Debug, PartialEq)]
struct AggregatedPolls {
    gpu_util_pct: Option<f64>,
    mem_util_pct: Option<f64>,
    power_watts: Option<f64>,
    vram_used_mb: Option<u64>,
    vram_peak_mb: Option<u64>,
    vram_total_mb: Option<u64>,
    temperature_c: Option<f64>,
    temperature_peak_c: Option<f64>,
    sm_clock_mhz: Option<u32>,
}

fn aggregate_polls(polls: &[GpuPoll]) -> AggregatedPolls {
    let mut sum_gpu = 0.0f64;
    let mut sum_mem = 0.0f64;
    let mut n_util = 0u32;
    let mut sum_power = 0.0f64;
    let mut n_power = 0u32;

    let mut vram_used_mb = None;
    let mut vram_peak_mb: Option<u64> = None;
    let mut vram_total_mb = None;
    let mut temperature_c = None;
    let mut temperature_peak_c: Option<f64> = None;
    let mut sm_clock_mhz = None;

    for p in polls {
        if let (Some(g), Some(m)) = (p.util_gpu, p.util_mem) {
            sum_gpu += f64::from(g);
            sum_mem += f64::from(m);
            n_util += 1;
        }

        if let Some(w) = p.power_watts {
            sum_power += w;
            n_power += 1;
        }

        if let Some(u) = p.vram_used_mb {
            vram_used_mb = Some(u);
            vram_peak_mb = Some(match vram_peak_mb {
                Some(pk) => pk.max(u),
                None => u,
            });
        }
        if let Some(t) = p.vram_total_mb {
            vram_total_mb = Some(t);
        }
        if let Some(t) = p.temperature_c.filter(|x| x.is_finite()) {
            temperature_c = Some(t);
            temperature_peak_c = Some(match temperature_peak_c {
                Some(pk) => pk.max(t),
                None => t,
            });
        }
        if let Some(c) = p.sm_clock_mhz {
            sm_clock_mhz = Some(c);
        }
    }

    let gpu_util_pct = (n_util > 0).then_some(sum_gpu / f64::from(n_util));
    let mem_util_pct = (n_util > 0).then_some(sum_mem / f64::from(n_util));
    let power_watts = (n_power > 0).then_some(sum_power / f64::from(n_power));

    AggregatedPolls {
        gpu_util_pct,
        mem_util_pct,
        power_watts,
        vram_used_mb,
        vram_peak_mb,
        vram_total_mb,
        temperature_c,
        temperature_peak_c,
        sm_clock_mhz,
    }
}

/// Returns `(metrics, observed_at, host_count)` after the last NVML poll for the requested window.
pub fn collect_gpu_metrics_for(
    window: Duration,
    explicit_indices: Option<&[u32]>,
) -> Result<(Vec<GpuRawMetrics>, SystemTime, Option<u32>)> {
    let Ok(nvml) = Nvml::init() else {
        return Ok((vec![], SystemTime::now(), None));
    };

    let host_count = nvml.device_count().unwrap_or(0);
    if host_count == 0 {
        return Ok((vec![], SystemTime::now(), Some(0)));
    }

    let device_indices: Vec<u32> = if let Some(ei) = explicit_indices {
        ei.to_vec()
    } else {
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
    let mut all_device_polls: std::collections::HashMap<u32, Vec<GpuPoll>> =
        std::collections::HashMap::new();
    for &d in &device_indices {
        all_device_polls.insert(d, Vec::with_capacity(sample_count));
    }

    for i in 0..sample_count {
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
                slot.push(tick);
            }
        }

        if i + 1 < sample_count {
            thread::sleep(SAMPLE_INTERVAL);
        }
    }

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

                let agg = aggregate_polls(all_device_polls.get(&d).unwrap());

                results.push(GpuRawMetrics {
                    gpu_name,
                    gpu_index,
                    gpu_uuid,
                    pcie_bus_id,
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

/// NVML device indices to poll. Scope vs TP is validated after collection in
/// `validate_tensor_parallel_scope` - this only resolves CVD vs full host.
pub(crate) fn resolve_device_indices(cvd_indices: Vec<u32>, host_device_count: u32) -> Vec<u32> {
    if cvd_indices.is_empty() {
        (0..host_device_count).collect()
    } else {
        cvd_indices
    }
}

#[cfg(test)]
fn sample_poll(ug: u32, um: u32, p: f64, vu: u64, vt: u64, temp: f64, sm: u32) -> GpuPoll {
    GpuPoll {
        util_gpu: Some(ug),
        util_mem: Some(um),
        power_watts: Some(p),
        vram_used_mb: Some(vu),
        vram_total_mb: Some(vt),
        temperature_c: Some(temp),
        sm_clock_mhz: Some(sm),
    }
}

#[cfg(test)]
mod resolve_device_indices_tests {
    use super::resolve_device_indices;

    #[test]
    fn no_cvd_returns_all_host_gpus() {
        assert_eq!(
            resolve_device_indices(vec![], 8),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn cvd_overrides_host_gpus_exactly() {
        assert_eq!(resolve_device_indices(vec![2, 3], 8), vec![2, 3]);
    }

    #[test]
    fn cvd_subset_of_host() {
        assert_eq!(resolve_device_indices(vec![0, 1], 8), vec![0, 1]);
    }

    #[test]
    fn cvd_larger_than_tp_is_collected_whole_for_dilution_abort() {
        assert_eq!(
            resolve_device_indices(vec![0, 1, 2, 3], 8),
            vec![0, 1, 2, 3]
        );
    }
}

#[cfg(test)]
mod aggregate_polls_tests {
    use super::*;

    #[test]
    fn aggregate_identical_polls_equals_sample_values() {
        let polls: Vec<_> = (0..SAMPLE_COUNT)
            .map(|_| sample_poll(80, 20, 300.0, 1000, 8000, 55.0, 2100))
            .collect();
        let a = aggregate_polls(&polls);
        assert_eq!(a.gpu_util_pct, Some(80.0));
        assert_eq!(a.mem_util_pct, Some(20.0));
        assert_eq!(a.power_watts, Some(300.0));
        assert_eq!(a.vram_used_mb, Some(1000));
        assert_eq!(a.vram_peak_mb, Some(1000));
        assert_eq!(a.vram_total_mb, Some(8000));
        assert_eq!(a.temperature_c, Some(55.0));
        assert_eq!(a.temperature_peak_c, Some(55.0));
        assert_eq!(a.sm_clock_mhz, Some(2100));
    }

    #[test]
    fn aggregate_means_util_and_power_averages_last_for_rest() {
        let polls = vec![
            sample_poll(0, 0, 100.0, 100, 8000, 40.0, 1000),
            sample_poll(100, 50, 200.0, 200, 8000, 50.0, 2000),
        ];
        let a = aggregate_polls(&polls);
        assert_eq!(a.gpu_util_pct, Some(50.0));
        assert_eq!(a.mem_util_pct, Some(25.0));
        assert_eq!(a.power_watts, Some(150.0));
        assert_eq!(a.vram_used_mb, Some(200));
        assert_eq!(a.vram_peak_mb, Some(200));
        assert_eq!(a.vram_total_mb, Some(8000));
        assert_eq!(a.temperature_c, Some(50.0));
        assert_eq!(a.temperature_peak_c, Some(50.0));
        assert_eq!(a.sm_clock_mhz, Some(2000));
    }

    #[test]
    fn aggregate_vram_peak_is_max_across_polls_not_last_only() {
        let polls = vec![
            sample_poll(80, 20, 300.0, 50 * 1024, 8000, 55.0, 2100),
            sample_poll(80, 20, 300.0, 10 * 1024, 8000, 55.0, 2100),
        ];
        let a = aggregate_polls(&polls);
        assert_eq!(a.vram_used_mb, Some(10 * 1024));
        assert_eq!(a.vram_peak_mb, Some(50 * 1024));
    }

    #[test]
    fn aggregate_temperature_peak_is_max_across_polls_not_last_only() {
        let polls = vec![
            sample_poll(80, 20, 300.0, 1000, 8000, 72.0, 2100),
            sample_poll(80, 20, 300.0, 1000, 8000, 65.0, 2100),
        ];
        let a = aggregate_polls(&polls);
        assert_eq!(a.temperature_c, Some(65.0));
        assert_eq!(a.temperature_peak_c, Some(72.0));
    }

    #[test]
    fn aggregate_skips_ticks_without_util_pair() {
        let polls = vec![
            GpuPoll {
                power_watts: Some(100.0),
                ..Default::default()
            },
            sample_poll(50, 25, 200.0, 1, 2, 3.0, 4),
        ];
        let a = aggregate_polls(&polls);
        assert_eq!(a.gpu_util_pct, Some(50.0));
        assert_eq!(a.mem_util_pct, Some(25.0));
        assert_eq!(a.power_watts, Some(150.0));
    }

    #[test]
    fn aggregate_empty_is_all_none() {
        assert_eq!(
            aggregate_polls(&[]),
            AggregatedPolls {
                gpu_util_pct: None,
                mem_util_pct: None,
                power_watts: None,
                vram_used_mb: None,
                vram_peak_mb: None,
                vram_total_mb: None,
                temperature_c: None,
                temperature_peak_c: None,
                sm_clock_mhz: None,
            }
        );
    }
}
