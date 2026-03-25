use anyhow::Result;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use nvml_wrapper::Nvml;

use super::GpuRawMetrics;

const MIB: u64 = 1024 * 1024;

pub fn collect_gpu_metrics() -> Result<GpuRawMetrics> {
    let Ok(nvml) = Nvml::init() else {
        return Ok(GpuRawMetrics::default());
    };
    let Ok(device) = nvml.device_by_index(0) else {
        return Ok(GpuRawMetrics::default());
    };

    let gpu_name = device.name().ok();
    let util = device.utilization_rates().ok();
    let gpu_util_pct = util.as_ref().map(|u| u.gpu as f64);
    let mem_util_pct = util.as_ref().map(|u| u.memory as f64);

    let mem = device.memory_info().ok();
    let memory_used_mb = mem.as_ref().map(|i| i.used / MIB);
    let memory_total_mb = mem.as_ref().map(|i| i.total / MIB);

    let power_watts = device.power_usage().ok().map(|mw| mw as f64 / 1000.0);
    let power_limit_watts = device
        .power_management_limit()
        .ok()
        .map(|lim| lim as f64 / 1000.0);
    let temperature_c = device
        .temperature(TemperatureSensor::Gpu)
        .ok()
        .map(|t| t as f64);

    Ok(GpuRawMetrics {
        gpu_name,
        gpu_util_pct,
        mem_util_pct,
        memory_used_mb,
        memory_total_mb,
        power_watts,
        power_limit_watts,
        temperature_c,
    })
}
