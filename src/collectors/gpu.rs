//! GPU collector — NVML only (V1: friendly name; util moved to sampler).

use nvml_wrapper::{Device, Nvml};

pub(super) fn gpu_name() -> Option<String> {
    let nvml = Nvml::init().ok()?;
    let device: Device = nvml.device_by_index(0).ok()?;
    device.name().ok()
}

// Deprecated: use sampler buffer instead for high-freq util
#[deprecated(note = "Use sampler for high-frequency GPU utilization")]
pub(super) fn gpu_utilization() -> Option<f32> {
    // Keep for backward compat if needed, but prefer sampler
    let nvml = Nvml::init().ok()?;
    let device: Device = nvml.device_by_index(0).ok()?;
    let util = device.utilization_rates().ok()?;
    Some(util.gpu as f32)
}
