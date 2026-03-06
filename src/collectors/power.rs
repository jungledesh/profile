//! Power draw collector.
//!
//! Will be backed by NVML (or remote profile-agent) later.

/// Read current power draw in watts. Returns None if unavailable.
pub fn power_draw() -> Option<f32> {
    // Stub: no NVML yet.
    None
}
