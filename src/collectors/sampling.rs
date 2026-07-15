//! Shared scrape cadence for GPU and vLLM `/metrics` (parallel in `collect_snapshot`).

use std::thread;
use std::time::Duration;

use anyhow::Result;

pub const SAMPLE_INTERVAL: Duration = Duration::from_millis(250);

pub fn sample_count_for(window: Duration) -> usize {
    let interval_ms = SAMPLE_INTERVAL.as_millis();
    let window_ms = window.as_millis();
    let ticks = (window_ms / interval_ms) + 1;
    ticks.max(2) as usize
}

/// Run `tick` once per sample; sleep between samples, never after the last.
pub(crate) fn run_sampling_loop(
    sample_count: usize,
    mut tick: impl FnMut(usize) -> Result<()>,
) -> Result<()> {
    for i in 0..sample_count {
        tick(i)?;
        if i + 1 < sample_count {
            thread::sleep(SAMPLE_INTERVAL);
        }
    }
    Ok(())
}
