//! Shared scrape cadence for GPU NVML and vLLM `/metrics` (parallel in `collect_snapshot`).

use std::time::Duration;

pub const SAMPLE_COUNT: usize = 9;
pub const SAMPLE_INTERVAL: Duration = Duration::from_millis(250);

pub fn sample_count_for(window: Duration) -> usize {
    let interval_ms = SAMPLE_INTERVAL.as_millis();
    let window_ms = window.as_millis();
    let ticks = (window_ms / interval_ms) + 1;
    ticks.max(2) as usize
}
