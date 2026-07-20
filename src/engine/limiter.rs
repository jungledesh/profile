//! No-issue path: identifies the primary physical boundary capping efficiency.
//! Cascade is mutually exclusive, each stage is only reached if harder limits above it are clear.

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
/// Effective prompt/decode ratio where prefill begins to interfere even below R6.
/// Provisional, calibrate on RunPod.
const LIMITER_PREFILL_RATIO_MIN: f64 = 0.5;

/// Aggregated run evidence, same courtroom as rule evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LimiterEvidence {
    pub kv_cache_peak_perc: Option<f64>,
    pub mean_running: Option<f64>,
    pub ridge_batch_size: Option<f64>,
    pub mean_tpot_ms: Option<f64>,
    pub tpot_floor_ms: Option<f64>,
    pub effective_prompt_decode_ratio: Option<f64>,
    pub chunked_prefill_enabled: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CeilingUnknown;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimiterVerdict {
    Known(PrimaryLimiter),
    CeilingUnknown(CeilingUnknown),
}

/// Identify the primary constraint when all rules pass.
///
/// Returns `None` if insufficient data to determine any limiter
/// (missing baseline or metrics). Caller shows generic message.
///
/// Cascade order: hardest physical limits first. Each stage is only
/// reached if the stage above it did not fire.
pub fn identify(e: &LimiterEvidence) -> Option<LimiterVerdict> {
    // 1. Capacity - KV cache full enough to cap concurrency growth.
    if e.kv_cache_peak_perc
        .is_some_and(|kv| kv.is_finite() && kv >= KV_CAPACITY_LIMITER_PERC)
    {
        return Some(LimiterVerdict::Known(PrimaryLimiter::Capacity));
    }

    // 2. Traffic - VRAM available but not enough concurrent requests to
    //    amortize overhead and saturate memory bandwidth.
    if let (Some(running), Some(ridge)) = (e.mean_running, e.ridge_batch_size) {
        let threshold = (ridge * TRAFFIC_LIMITER_RIDGE_FRACTION).max(TRAFFIC_LIMITER_MIN_RUNNING);
        if running < threshold {
            return Some(LimiterVerdict::Known(PrimaryLimiter::Traffic));
        }
    }

    // 3. Physics - TPOT near theoretical floor; hardware is saturated.
    // Unknown GPU: no floor, no confident name.
    if e.mean_tpot_ms
        .is_some_and(|tpot| tpot.is_finite() && tpot > 0.0)
        && e.tpot_floor_ms
            .is_none_or(|floor| !floor.is_finite() || floor <= 0.0)
    {
        return Some(LimiterVerdict::CeilingUnknown(CeilingUnknown));
    }
    if let (Some(tpot), Some(floor)) = (e.mean_tpot_ms, e.tpot_floor_ms)
        && tpot <= floor * PHYSICS_LIMITER_FLOOR_MARGIN
    {
        return Some(LimiterVerdict::Known(PrimaryLimiter::Physics));
    }

    // 4. Prefill interference - measured prefill signal with chunked prefill as precondition.
    if e.chunked_prefill_enabled == Some(true)
        && e.effective_prompt_decode_ratio
            .is_some_and(|ratio| ratio.is_finite() && ratio >= LIMITER_PREFILL_RATIO_MIN)
    {
        return Some(LimiterVerdict::Known(PrimaryLimiter::PrefillInterference));
    }

    // 5. Framework overhead - batch healthy, VRAM free, not at physics ceiling.
    //    Only fire if we have enough signal to rule out data absence.
    if e.mean_running.is_some() && e.mean_tpot_ms.is_some() {
        return Some(LimiterVerdict::Known(PrimaryLimiter::FrameworkOverhead));
    }

    None
}

pub fn limiter_line(e: &LimiterEvidence) -> Option<String> {
    match identify(e)? {
        LimiterVerdict::CeilingUnknown(_) => {
            Some("Hardware ceiling unknown (GPU not in catalog).".to_string())
        }
        LimiterVerdict::Known(PrimaryLimiter::Capacity) => {
            let kv = e.kv_cache_peak_perc?;
            Some(format!(
                "Capped by memory: KV cache at {kv:.0}% peak (R2 fires at 88%). Concurrency cannot grow further on this pool."
            ))
        }
        LimiterVerdict::Known(PrimaryLimiter::Traffic) => {
            let running = e.mean_running?;
            let ridge = e.ridge_batch_size?;
            Some(format!(
                "Capped by traffic: {:.0} requests running, hardware has room for ~{ridge:.0}. More concurrent requests raises throughput.",
                running.trunc()
            ))
        }
        LimiterVerdict::Known(PrimaryLimiter::Physics) => {
            let tpot = e.mean_tpot_ms?;
            let floor = e.tpot_floor_ms?;
            Some(format!(
                "Capped by hardware: TPOT {tpot:.1}ms vs ~{floor:.1}ms floor. This GPU is saturated, scale out to go faster."
            ))
        }
        LimiterVerdict::Known(PrimaryLimiter::PrefillInterference) => {
            let ratio = e.effective_prompt_decode_ratio?;
            Some(format!(
                "Capped by prefill: prompt work at {ratio:.1}x of decode (effective). Prefill shares bandwidth with decode on every step."
            ))
        }
        LimiterVerdict::Known(PrimaryLimiter::FrameworkOverhead) => Some(
            "Capped by framework: batch healthy, memory free, hardware unsaturated. GPU is waiting on vLLM/CPU overhead.".to_string(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(
        kv: Option<f64>,
        running: Option<f64>,
        ridge: Option<f64>,
        tpot: Option<f64>,
        floor: Option<f64>,
        ratio: Option<f64>,
        chunked: Option<bool>,
    ) -> LimiterEvidence {
        LimiterEvidence {
            kv_cache_peak_perc: kv,
            mean_running: running,
            ridge_batch_size: ridge,
            mean_tpot_ms: tpot,
            tpot_floor_ms: floor,
            effective_prompt_decode_ratio: ratio,
            chunked_prefill_enabled: chunked,
        }
    }

    #[test]
    fn capacity_fires_when_kv_above_threshold() {
        assert_eq!(
            identify(&ev(
                Some(85.0),
                Some(50.0),
                Some(40.0),
                Some(20.0),
                Some(5.0),
                Some(0.2),
                Some(false)
            )),
            Some(LimiterVerdict::Known(PrimaryLimiter::Capacity))
        );
    }

    #[test]
    fn traffic_fires_when_running_below_ridge_fraction() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(5.0),
                Some(100.0),
                Some(20.0),
                Some(5.0),
                Some(0.2),
                Some(false)
            )),
            Some(LimiterVerdict::Known(PrimaryLimiter::Traffic))
        );
    }

    #[test]
    fn physics_fires_when_tpot_near_floor() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(50.0),
                Some(40.0),
                Some(12.0),
                Some(10.0),
                Some(0.2),
                Some(false)
            )),
            Some(LimiterVerdict::Known(PrimaryLimiter::Physics))
        );
    }

    #[test]
    fn physics_does_not_fire_when_tpot_well_above_floor() {
        // tpot=15ms, floor=10ms: 15 > 10*1.2 (12ms) → should NOT fire
        // If params transposed: 10 <= 15*1.2 (18ms) → would fire → catches the bug
        let result = identify(&ev(
            Some(50.0),
            Some(50.0),
            Some(40.0),
            Some(15.0),
            Some(10.0),
            Some(0.2),
            Some(false),
        ));
        assert_ne!(result, Some(LimiterVerdict::Known(PrimaryLimiter::Physics)));
    }

    #[test]
    fn prefill_interference_fires_when_chunked_prefill_on_and_ratio_high() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(50.0),
                Some(40.0),
                Some(50.0),
                Some(10.0),
                Some(0.6),
                Some(true)
            )),
            Some(LimiterVerdict::Known(PrimaryLimiter::PrefillInterference))
        );
    }

    #[test]
    fn framework_overhead_is_fallthrough() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(50.0),
                Some(40.0),
                Some(50.0),
                Some(10.0),
                Some(0.1),
                Some(false)
            )),
            Some(LimiterVerdict::Known(PrimaryLimiter::FrameworkOverhead))
        );
    }

    #[test]
    fn none_returned_when_data_missing() {
        assert_eq!(
            identify(&ev(None, None, None, None, None, None, None)),
            None
        );
        assert_eq!(
            identify(&ev(
                Some(50.0),
                None,
                Some(40.0),
                None,
                Some(10.0),
                None,
                Some(false)
            )),
            None
        );
    }

    #[test]
    fn capacity_takes_priority_over_traffic() {
        assert_eq!(
            identify(&ev(
                Some(85.0),
                Some(2.0),
                Some(100.0),
                Some(20.0),
                Some(5.0),
                Some(0.6),
                Some(false)
            )),
            Some(LimiterVerdict::Known(PrimaryLimiter::Capacity))
        );
    }

    #[test]
    fn traffic_takes_priority_over_physics() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(5.0),
                Some(100.0),
                Some(11.0),
                Some(10.0),
                Some(0.6),
                Some(false)
            )),
            Some(LimiterVerdict::Known(PrimaryLimiter::Traffic))
        );
    }

    #[test]
    fn unknown_gpu_degrades_before_prefill_or_framework() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(50.0),
                Some(100.0),
                Some(11.0),
                None,
                Some(0.7),
                Some(true)
            )),
            Some(LimiterVerdict::CeilingUnknown(CeilingUnknown))
        );
    }

    #[test]
    fn prefill_requires_chunked_precondition() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(50.0),
                Some(100.0),
                Some(50.0),
                Some(10.0),
                Some(0.6),
                Some(false)
            )),
            Some(LimiterVerdict::Known(PrimaryLimiter::FrameworkOverhead))
        );
    }

    #[test]
    fn prefill_requires_effective_ratio_threshold() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(50.0),
                Some(100.0),
                Some(50.0),
                Some(10.0),
                Some(0.3),
                Some(true)
            )),
            Some(LimiterVerdict::Known(PrimaryLimiter::FrameworkOverhead))
        );
    }

    #[test]
    fn capacity_line_names_r2_relationship() {
        let line = limiter_line(&ev(
            Some(84.0),
            Some(50.0),
            Some(100.0),
            Some(20.0),
            Some(10.0),
            Some(0.2),
            Some(false),
        ))
        .expect("line");
        assert_eq!(
            line,
            "Capped by memory: KV cache at 84% peak (R2 fires at 88%). Concurrency cannot grow further on this pool."
        );
    }

    #[test]
    fn traffic_line_contains_no_flags() {
        let line = limiter_line(&ev(
            Some(20.0),
            Some(6.0),
            Some(153.0),
            Some(20.0),
            Some(10.0),
            Some(0.2),
            Some(false),
        ))
        .expect("line");
        assert!(line.contains("Capped by traffic"));
        assert!(!line.contains("--max-num-seqs"));
    }

    #[test]
    fn prefix_corrected_ratio_avoids_prefill_verdict() {
        assert_eq!(
            identify(&ev(
                Some(20.0),
                Some(50.0),
                Some(100.0),
                Some(50.0),
                Some(10.0),
                Some(0.3),
                Some(true)
            )),
            Some(LimiterVerdict::Known(PrimaryLimiter::FrameworkOverhead))
        );
    }
}
