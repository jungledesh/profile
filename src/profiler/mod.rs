//! Profiler: run requests, measure latency, compute metrics. Used by CLI now; by agent over HTTP later.

use crate::collectors;

#[derive(Debug, Clone)]
pub struct ProfileResult {
    pub config_path: Option<String>,
    pub snapshot: collectors::Snapshot, // V1 waste detection data
}

/// When reading from config_path, canonicalize and restrict to expected dir to avoid path traversal.
pub fn run_profile(config_path: Option<&str>) -> anyhow::Result<ProfileResult> {
    let snapshot = collectors::snapshot();

    Ok(ProfileResult {
        config_path: config_path.map(String::from),
        snapshot,
    })
}
