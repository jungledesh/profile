//! GPU collector — NVML only (V1: SM_Util + friendly name). Hardware-first.

use nvml_wrapper::{Device, Nvml};

pub(super) fn gpu_utilization() -> Option<f32> {
    let nvml = Nvml::init().ok()?;
    let device: Device = nvml.device_by_index(0).ok()?;
    let util = device.utilization_rates().ok()?;
    Some(util.gpu as f32)
}

pub(super) fn gpu_name() -> Option<String> {
    let nvml = Nvml::init().ok()?;
    let device: Device = nvml.device_by_index(0).ok()?;
    device.name().ok()
}
