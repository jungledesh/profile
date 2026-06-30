//! No-issue exit path: identifies the primary physical boundary capping efficiency.
//! Cascade is mutually exclusive - each stage is only reached if harder limits above it are clear.

/// The physical or systemic boundary capping efficiency when no rules fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimaryLimiter {
    /// KV cache is filling VRAM before batch size can saturate memory bandwidth.
    Capacity,
    /// Insufficient concurrent requests to amortize kernel launch and scheduler overhead.
    Traffic,
    /// TPOT is within 20% of the theoretical floor - hardware is saturated.
    Physics,
    /// Chunked prefill is sharing decode memory bandwidth with prefill GEMMs.
    PrefillInterference,
    /// Batch sizes healthy, VRAM available - GPU is waiting on framework/system.
    FrameworkOverhead,
}

// All thresholds provisional - calibrate against real workloads before hardening.
const KV_CAPACITY_LIMITER_PERC: f64 = 80.0; // below r2's crisis threshold of 88.0
const TRAFFIC_LIMITER_RIDGE_FRACTION: f64 = 0.25;
const TRAFFIC_LIMITER_MIN_RUNNING: f64 = 8.0;
const PHYSICS_LIMITER_FLOOR_MARGIN: f64 = 1.2; // within 20% of tpot_floor_ms

/// Identify the primary constraint when all rules pass.
///
/// Returns `None` if insufficient data to determine any limiter
/// (missing baseline or metrics). Caller shows generic message.
///
/// Cascade order: hardest physical limits first. Each stage is only
/// reached if the stage above it did not fire.
pub fn identify(
    kv_cache_usage_perc: Option<f64>,
    num_running: Option<f64>,
    ridge_batch_size: Option<f64>,
    tpot_ms: Option<f64>,
    tpot_floor_ms: Option<f64>,
    chunked_prefill_enabled: Option<bool>,
) -> Option<PrimaryLimiter> {
    // 1. Capacity - KV cache full enough to cap concurrency growth.
    if kv_cache_usage_perc.is_some_and(|kv| kv >= KV_CAPACITY_LIMITER_PERC) {
        return Some(PrimaryLimiter::Capacity);
    }

    // 2. Traffic - VRAM available but not enough concurrent requests to
    //    amortize overhead and saturate memory bandwidth.
    if let (Some(running), Some(ridge)) = (num_running, ridge_batch_size) {
        let threshold = (ridge * TRAFFIC_LIMITER_RIDGE_FRACTION).max(TRAFFIC_LIMITER_MIN_RUNNING);
        if running < threshold {
            return Some(PrimaryLimiter::Traffic);
        }
    }

    // 3. Physics - TPOT near theoretical floor; hardware is saturated.
    if let (Some(tpot), Some(floor)) = (tpot_ms, tpot_floor_ms)
        && tpot <= floor * PHYSICS_LIMITER_FLOOR_MARGIN
    {
        return Some(PrimaryLimiter::Physics);
    }

    // 4. Prefill interference - chunked prefill sharing decode bandwidth.
    if chunked_prefill_enabled == Some(true) {
        return Some(PrimaryLimiter::PrefillInterference);
    }

    // 5. Framework overhead - batch healthy, VRAM free, not at physics ceiling.
    //    Only fire if we have enough signal to rule out data absence.
    if num_running.is_some() && tpot_ms.is_some() {
        return Some(PrimaryLimiter::FrameworkOverhead);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_fires_when_kv_above_threshold() {
        assert_eq!(
            identify(
                Some(85.0),
                Some(50.0),
                Some(40.0),
                Some(20.0),
                Some(5.0),
                Some(false)
            ),
            Some(PrimaryLimiter::Capacity)
        );
    }

    #[test]
    fn traffic_fires_when_running_below_ridge_fraction() {
        assert_eq!(
            identify(
                Some(50.0),
                Some(5.0),
                Some(100.0),
                Some(20.0),
                Some(5.0),
                Some(false)
            ),
            Some(PrimaryLimiter::Traffic)
        );
    }

    #[test]
    fn physics_fires_when_tpot_near_floor() {
        assert_eq!(
            identify(
                Some(50.0),
                Some(50.0),
                Some(40.0),
                Some(12.0),
                Some(10.0),
                Some(false)
            ),
            Some(PrimaryLimiter::Physics)
        );
    }

    #[test]
    fn physics_does_not_fire_when_tpot_well_above_floor() {
        // tpot=15ms, floor=10ms: 15 > 10*1.2 (12ms) → should NOT fire
        // If params transposed: 10 <= 15*1.2 (18ms) → would fire → catches the bug
        let result = identify(
            Some(50.0),
            Some(50.0),
            Some(40.0),
            Some(15.0),
            Some(10.0),
            Some(false),
        );
        assert_ne!(result, Some(PrimaryLimiter::Physics));
    }

    #[test]
    fn prefill_interference_fires_when_chunked_prefill_on() {
        assert_eq!(
            identify(
                Some(50.0),
                Some(50.0),
                Some(40.0),
                Some(50.0),
                Some(10.0),
                Some(true)
            ),
            Some(PrimaryLimiter::PrefillInterference)
        );
    }

    #[test]
    fn framework_overhead_is_fallthrough() {
        assert_eq!(
            identify(
                Some(50.0),
                Some(50.0),
                Some(40.0),
                Some(50.0),
                Some(10.0),
                Some(false)
            ),
            Some(PrimaryLimiter::FrameworkOverhead)
        );
    }

    #[test]
    fn none_returned_when_data_missing() {
        assert_eq!(identify(None, None, None, None, None, None), None);
        assert_eq!(
            identify(Some(50.0), None, Some(40.0), None, Some(10.0), Some(false)),
            None
        );
    }

    #[test]
    fn capacity_takes_priority_over_traffic() {
        assert_eq!(
            identify(
                Some(85.0),
                Some(2.0),
                Some(100.0),
                Some(20.0),
                Some(5.0),
                Some(false)
            ),
            Some(PrimaryLimiter::Capacity)
        );
    }

    #[test]
    fn traffic_takes_priority_over_physics() {
        assert_eq!(
            identify(
                Some(50.0),
                Some(5.0),
                Some(100.0),
                Some(11.0),
                Some(10.0),
                Some(false)
            ),
            Some(PrimaryLimiter::Traffic)
        );
    }
}
