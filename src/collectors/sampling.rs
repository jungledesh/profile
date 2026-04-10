//! Shared scrape cadence for GPU NVML and vLLM `/metrics` (parallel in `collect_snapshot`).

use std::time::Duration;

pub const SAMPLE_COUNT: usize = 9;
pub const SAMPLE_INTERVAL: Duration = Duration::from_millis(250);
