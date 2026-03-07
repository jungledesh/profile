//! Profiler: run requests, measure latency, compute metrics. Used by CLI now; by agent over HTTP later.

use crate::collectors;

#[derive(Debug, Clone)]
pub struct ProfileResult {
    pub config_path: Option<String>,
}

pub fn run_profile(config_path: Option<&str>) -> anyhow::Result<ProfileResult> {
    let _ = collectors::snapshot();
    Ok(ProfileResult {
        config_path: config_path.map(String::from),
    })
}
