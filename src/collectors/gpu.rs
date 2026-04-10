use std::thread;
use std::time::SystemTime;

use anyhow::Result;
use nvml_wrapper::enum_wrappers::device::{Clock, ClockId, TemperatureSensor};
use nvml_wrapper::Nvml;

use super::sampling::{SAMPLE_COUNT, SAMPLE_INTERVAL};
use super::GpuRawMetrics;

const MIB: u64 = 1024 * 1024;

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
    vram_total_mb: Option<u64>,
    temperature_c: Option<f64>,
    sm_clock_mhz: Option<u32>,
}

fn aggregate_polls(polls: &[GpuPoll]) -> AggregatedPolls {
    let mut sum_gpu = 0.0f64;
    let mut sum_mem = 0.0f64;
    let mut n_util = 0u32;
    let mut sum_power = 0.0f64;
    let mut n_power = 0u32;

    let mut vram_used_mb = None;
    let mut vram_total_mb = None;
    let mut temperature_c = None;
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
        }
        if let Some(t) = p.vram_total_mb {
            vram_total_mb = Some(t);
        }
        if let Some(t) = p.temperature_c {
            temperature_c = Some(t);
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
        vram_total_mb,
        temperature_c,
        sm_clock_mhz,
    }
}

/// Returns `(metrics, observed_at)` after the last NVML poll in the sampling window.
pub fn collect_gpu_metrics() -> Result<(GpuRawMetrics, SystemTime)> {
    let Ok(nvml) = Nvml::init() else {
        return Ok((GpuRawMetrics::default(), SystemTime::now()));
    };
    let Ok(device) = nvml.device_by_index(0) else {
        return Ok((GpuRawMetrics::default(), SystemTime::now()));
    };

    let gpu_name = device.name().ok();
    let gpu_index = device.index().ok();
    let gpu_uuid = device.uuid().ok();
    let power_limit_watts = device
        .power_management_limit()
        .ok()
        .map(|mw| mw as f64 / 1000.0);

    let mut polls = Vec::with_capacity(SAMPLE_COUNT);

    for i in 0..SAMPLE_COUNT {
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

        polls.push(tick);

        if i + 1 < SAMPLE_COUNT {
            thread::sleep(SAMPLE_INTERVAL);
        }
    }

    let agg = aggregate_polls(&polls);

    Ok((
        GpuRawMetrics {
            gpu_name,
            gpu_index,
            gpu_uuid,
            gpu_util_pct: agg.gpu_util_pct,
            mem_util_pct: agg.mem_util_pct,
            power_watts: agg.power_watts,
            power_limit_watts,
            vram_used_mb: agg.vram_used_mb,
            vram_total_mb: agg.vram_total_mb,
            temperature_c: agg.temperature_c,
            sm_clock_mhz: agg.sm_clock_mhz,
        },
        SystemTime::now(),
    ))
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
    assert_eq!(a.vram_total_mb, Some(8000));
    assert_eq!(a.temperature_c, Some(55.0));
    assert_eq!(a.sm_clock_mhz, Some(2100));
}

#[cfg(test)]
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
    assert_eq!(a.vram_total_mb, Some(8000));
    assert_eq!(a.temperature_c, Some(50.0));
    assert_eq!(a.sm_clock_mhz, Some(2000));
}

#[cfg(test)]
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

#[cfg(test)]
#[test]
fn aggregate_empty_is_all_none() {
    assert_eq!(
        aggregate_polls(&[]),
        AggregatedPolls {
            gpu_util_pct: None,
            mem_util_pct: None,
            power_watts: None,
            vram_used_mb: None,
            vram_total_mb: None,
            temperature_c: None,
            sm_clock_mhz: None,
        }
    );
}
