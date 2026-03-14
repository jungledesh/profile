//! GPU collector — NVML only (V1: friendly name; util moved to sampler).

use nvml_wrapper::{Device, Nvml};

pub(super) fn gpu_name() -> Option<String> {
    let nvml = Nvml::init().ok()?;
    let device: Device = nvml.device_by_index(0).ok()?;
    device.name().ok()
}
