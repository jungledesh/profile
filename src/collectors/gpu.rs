//! GPU utilization collector.
//!
//! Will be backed by NVML (or remote profile-agent) later.

/// Read current GPU utilization (0.0–100.0). Returns None if unavailable.
pub fn gpu_utilization() -> Option<f32> {
    // Stub: no NVML yet.
    None
}
