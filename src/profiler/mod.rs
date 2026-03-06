//! Profiler: run requests, measure latency, compute metrics.
//!
//! Core profiling logic. In the current setup this layer is invoked directly
//! by the CLI. In a future setup (profile-agent) this runs on the GPU machine
//! and is driven over HTTP.

use crate::collectors;

/// Result of a profile run. Holds metrics and config used.
#[derive(Debug, Clone)]
pub struct ProfileResult {
    /// Config path if one was specified.
    pub config_path: Option<String>,
    // Future: latency stats, token stats, GPU/power samples, etc.
}

/// Run a profile: execute requests, measure latency, compute metrics.
/// Today this is a dry-run; later it will use collectors (vLLM, NVML).
pub fn run_profile(config_path: Option<&str>) -> anyhow::Result<ProfileResult> {
    // Stub: no real requests yet. When implemented, this will:
    // - run requests (via vLLM or agent)
    // - measure latency
    // - sample GPU util, power, token stats via collectors
    let _ = collectors::snapshot();
    Ok(ProfileResult {
        config_path: config_path.map(String::from),
    })
}
