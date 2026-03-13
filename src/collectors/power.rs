//! Power collector — NVML only (V1: Power_W in watts). Hardware-first.

use nvml_wrapper::{Device, Nvml};

pub(super) fn power_draw() -> Option<f32> {
    let nvml = Nvml::init().ok()?;
    let device: Device = nvml.device_by_index(0).ok()?;
    let mw = device.power_usage().ok()?;
    Some(mw as f32 / 1000.0) // convert milliwatts → watts
}
