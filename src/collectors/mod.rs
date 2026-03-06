//! Collectors: GPU util, power draw, token stats.
//!
//! Gather system and GPU metrics. Currently stubs; will be backed by
//! vLLM and NVML (or a remote profile-agent) later.

mod gpu;
mod power;
mod tokens;

pub use gpu::gpu_utilization;
pub use power::power_draw;
pub use tokens::token_stats;

/// Snapshot of all collector metrics. Used by the profiler.
#[derive(Debug, Default, Clone)]
pub struct Snapshot {
    pub gpu_util: Option<f32>,
    pub power_w: Option<f32>,
    pub tokens_per_sec: Option<f32>,
}

/// Take a snapshot of current metrics from all collectors.
pub fn snapshot() -> Snapshot {
    Snapshot {
        gpu_util: gpu_utilization(),
        power_w: power_draw(),
        tokens_per_sec: token_stats().map(|s| s.tokens_per_sec),
    }
}
