use anyhow::Result;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use nvml_wrapper::Nvml;

use super::GpuRawMetrics;

pub fn collect_gpu_metrics() -> Result<GpuRawMetrics> {
    let Ok(nvml) = Nvml::init() else {
        return Ok(GpuRawMetrics::default());
    };
    let Ok(device) = nvml.device_by_index(0) else {
        return Ok(GpuRawMetrics::default());
    };

    let mut m = GpuRawMetrics::default();
    m.gpu_name = device.name().ok();
    if let Ok(u) = device.utilization_rates() {
        m.gpu_util_pct = Some(u.gpu as f64);
        m.mem_util_pct = Some(u.memory as f64);
    }
    if let Ok(info) = device.memory_info() {
        let mb = 1024 * 1024;
        m.memory_used_mb = Some(info.used / mb);
        m.memory_total_mb = Some(info.total / mb);
    }
    if let Ok(mw) = device.power_usage() {
        m.power_watts = Some(mw as f64 / 1000.0);
    }
    if let Ok(lim) = device.power_management_limit() {
        m.power_limit_watts = Some(lim as f64 / 1000.0);
    }
    if let Ok(t) = device.temperature(TemperatureSensor::Gpu) {
        m.temperature_c = Some(t as f64);
    }

    Ok(m)
}
