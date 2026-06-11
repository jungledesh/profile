use std::thread;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use nvml_wrapper::enum_wrappers::device::{Clock, ClockId, TemperatureSensor};
use nvml_wrapper::Nvml;

#[cfg(test)]
use super::sampling::SAMPLE_COUNT;
use super::sampling::{sample_count_for, SAMPLE_INTERVAL};
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
    vram_peak_mb: Option<u64>,
    vram_total_mb: Option<u64>,
    temperature_c: Option<f64>,
    temperature_peak_c: Option<f64>,
    sm_clock_mhz: Option<u32>,
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct DeviceSample {
    util_gpu: u32,
    util_mem: u32,
    power_watts: f64,
    vram_used_mb: u64,
    vram_total_mb: u64,
    temperature_c: f64,
}

fn poll_tp_devices(nvml: &Nvml, tp: u32) -> GpuPoll {
    let mut tick = GpuPoll::default();
    let mut sum_util_gpu = 0u32;
    let mut sum_util_mem = 0u32;
    let mut n_util = 0u32;
    let mut sum_power = 0.0f64;
    let mut n_power = 0u32;
    let mut vram_used = 0u64;
    let mut vram_total = 0u64;
    let mut max_temp: Option<f64> = None;

    for dev_idx in 0..tp {
        let Ok(dev) = nvml.device_by_index(dev_idx) else {
            continue;
        };

        if let Ok(u) = dev.utilization_rates() {
            sum_util_gpu += u.gpu;
            sum_util_mem += u.memory;
            n_util += 1;
        }
        if let Ok(mw) = dev.power_usage() {
            sum_power += mw as f64 / 1000.0;
            n_power += 1;
        }
        if let Ok(mem) = dev.memory_info() {
            vram_used += mem.used / MIB;
            vram_total += mem.total / MIB;
        }
        if let Ok(t) = dev.temperature(TemperatureSensor::Gpu) {
            let tc = f64::from(t);
            max_temp = Some(max_temp.map_or(tc, |prev: f64| prev.max(tc)));
        }
    }

    let sm_clock = nvml
        .device_by_index(0)
        .ok()
        .and_then(|primary| primary.clock(Clock::SM, ClockId::Current).ok());

    tick.util_gpu = (n_util > 0).then_some(sum_util_gpu / n_util);
    tick.util_mem = (n_util > 0).then_some(sum_util_mem / n_util);
    tick.power_watts = (n_power > 0).then_some(sum_power);
    tick.vram_used_mb = Some(vram_used);
    tick.vram_total_mb = Some(vram_total);
    tick.temperature_c = max_temp;
    tick.sm_clock_mhz = sm_clock;
    tick
}

#[cfg(test)]
fn poll_tp_device_samples(samples: &[DeviceSample], sm_clock_mhz: Option<u32>) -> GpuPoll {
    let mut tick = GpuPoll::default();
    let mut sum_util_gpu = 0u32;
    let mut sum_util_mem = 0u32;
    let mut n_util = 0u32;
    let mut sum_power = 0.0f64;
    let mut n_power = 0u32;
    let mut vram_used = 0u64;
    let mut vram_total = 0u64;
    let mut max_temp: Option<f64> = None;

    for s in samples {
        sum_util_gpu += s.util_gpu;
        sum_util_mem += s.util_mem;
        n_util += 1;
        sum_power += s.power_watts;
        n_power += 1;
        vram_used += s.vram_used_mb;
        vram_total += s.vram_total_mb;
        max_temp = Some(max_temp.map_or(s.temperature_c, |prev| prev.max(s.temperature_c)));
    }

    tick.util_gpu = (n_util > 0).then_some(sum_util_gpu / n_util);
    tick.util_mem = (n_util > 0).then_some(sum_util_mem / n_util);
    tick.power_watts = (n_power > 0).then_some(sum_power);
    tick.vram_used_mb = Some(vram_used);
    tick.vram_total_mb = Some(vram_total);
    tick.temperature_c = max_temp;
    tick.sm_clock_mhz = sm_clock_mhz;
    tick
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
                Some(p) => p.max(u),
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

/// Returns `(metrics, observed_at)` after the last NVML poll for the requested window.
pub fn collect_gpu_metrics_for(
    window: Duration,
    tp_size: u32,
) -> Result<(GpuRawMetrics, SystemTime)> {
    let Ok(nvml) = Nvml::init() else {
        return Ok((GpuRawMetrics::default(), SystemTime::now()));
    };
    let Ok(device) = nvml.device_by_index(0) else {
        return Ok((GpuRawMetrics::default(), SystemTime::now()));
    };

    let tp = tp_size.max(1);

    // Static identifiers from GPU 0 (homogeneous setup — all GPUs same model)
    let gpu_name = device.name().ok();
    let gpu_index = device.index().ok();
    let gpu_uuid = device.uuid().ok();
    let power_limit_watts = device
        .power_management_limit()
        .ok()
        .map(|mw| mw as f64 / 1000.0);

    let sample_count = sample_count_for(window);
    let mut polls = Vec::with_capacity(sample_count);

    for i in 0..sample_count {
        polls.push(poll_tp_devices(&nvml, tp));

        if i + 1 < sample_count {
            thread::sleep(SAMPLE_INTERVAL);
        }
    }

    let agg = aggregate_polls(&polls);

    Ok((
        GpuRawMetrics {
            gpu_name,
            gpu_index,
            gpu_uuid,
            gpu_count: tp,
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
fn tp2_doubles_power_and_vram() {
    let single = DeviceSample {
        util_gpu: 80,
        util_mem: 20,
        power_watts: 300.0,
        vram_used_mb: 1000,
        vram_total_mb: 8000,
        temperature_c: 55.0,
    };
    let tp1 = poll_tp_device_samples(std::slice::from_ref(&single), Some(2100));
    let tp2 = poll_tp_device_samples(&[single, single], Some(2100));

    assert_eq!(tp1.util_gpu, tp2.util_gpu);
    assert_eq!(tp1.power_watts, Some(300.0));
    assert_eq!(tp2.power_watts, Some(600.0));
    assert_eq!(tp1.vram_used_mb, Some(1000));
    assert_eq!(tp2.vram_used_mb, Some(2000));
    assert_eq!(tp1.vram_total_mb, Some(8000));
    assert_eq!(tp2.vram_total_mb, Some(16000));

    let a1 = aggregate_polls(&[tp1]);
    let a2 = aggregate_polls(&[tp2]);
    assert_eq!(a2.power_watts, Some(2.0 * a1.power_watts.unwrap()));
    assert_eq!(a2.vram_used_mb, Some(2 * a1.vram_used_mb.unwrap()));

    let metrics = GpuRawMetrics {
        gpu_count: 2,
        ..Default::default()
    };
    assert_eq!(metrics.gpu_count, 2);
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
    assert_eq!(a.vram_peak_mb, Some(1000));
    assert_eq!(a.vram_total_mb, Some(8000));
    assert_eq!(a.temperature_c, Some(55.0));
    assert_eq!(a.temperature_peak_c, Some(55.0));
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
    assert_eq!(a.vram_peak_mb, Some(200));
    assert_eq!(a.vram_total_mb, Some(8000));
    assert_eq!(a.temperature_c, Some(50.0));
    assert_eq!(a.temperature_peak_c, Some(50.0));
    assert_eq!(a.sm_clock_mhz, Some(2000));
}

#[cfg(test)]
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

#[cfg(test)]
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
            vram_peak_mb: None,
            vram_total_mb: None,
            temperature_c: None,
            temperature_peak_c: None,
            sm_clock_mhz: None,
        }
    );
}
