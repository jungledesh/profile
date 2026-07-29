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
/// Efficiency headroom below this: hardware ceiling (matches former loop_runner gate).
pub(crate) const HEADROOM_LIMITER_THRESHOLD_PCT: f64 = 10.0;
/// Effective prompt/decode ratio where prefill begins to interfere even below R6.
/// Provisional, calibrate on RunPod.
const LIMITER_PREFILL_RATIO_MIN: f64 = 0.5;

/// Aggregated run evidence, same courtroom as rule evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LimiterEvidence {
    /// Duration-weighted mean KV usage across the run. Capacity fires on this;
    /// peak is display context only. Mean is always <= peak, so requiring both
    /// reduces to requiring mean.
    pub kv_cache_mean_perc: Option<f64>,
    pub kv_cache_peak_perc: Option<f64>,
    pub mean_running: Option<f64>,
    pub ridge_batch_size: Option<f64>,
    pub mean_tpot_ms: Option<f64>,
    pub tpot_floor_ms: Option<f64>,
    pub effective_prompt_decode_ratio: Option<f64>,
    pub chunked_prefill_enabled: Option<bool>,
    /// `100 - efficiency_pct` from baseline; below [`HEADROOM_LIMITER_THRESHOLD_PCT`]
    /// triggers Physics even when TPOT is not near its floor.
    pub headroom_pct: Option<f64>,
    /// Evaluable windows in the run. Below `ENGINE_MIN_PERSISTENT_WINDOWS`,
    /// identify declines (same trust bar as rules).
    pub n_eval: usize,
    /// When baseline is absent and the limiter names CeilingUnknown.
    pub ceiling_unknown_reason: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CeilingUnknown;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimiterVerdict {
    Known(PrimaryLimiter),
    CeilingUnknown(CeilingUnknown),
}

/// Outcome of the limiter cascade, including which earlier stages lacked evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdentifyResult {
    pub verdict: Option<LimiterVerdict>,
    /// KV mean was None (capacity stage could not run).
    pub capacity_skipped: bool,
    /// Running or ridge was None (traffic stage could not run).
    pub traffic_skipped: bool,
}

/// Identify the primary constraint when all rules pass.
///
/// Returns `verdict: None` if insufficient data to determine any limiter
/// (missing baseline or metrics). Caller shows generic message.
///
/// Cascade order: hardest physical limits first. Each stage is only
/// reached if the stage above it did not fire.
pub fn identify(e: &LimiterEvidence) -> IdentifyResult {
    let capacity_skipped = e.kv_cache_mean_perc.is_none();
    let traffic_skipped = e.mean_running.is_none() || e.ridge_batch_size.is_none();

    // Sparse runs: decline before naming a boundary (healthy-exit and quiet report
    // share this gate; callers must not invent their own window thresholds).
    if e.n_eval < super::ENGINE_MIN_PERSISTENT_WINDOWS {
        return IdentifyResult {
            verdict: None,
            capacity_skipped,
            traffic_skipped,
        };
    }

    // 1. Capacity - KV cache full enough to cap concurrency growth.
    if e.kv_cache_mean_perc
        .is_some_and(|kv| kv.is_finite() && kv >= KV_CAPACITY_LIMITER_PERC)
    {
        return IdentifyResult {
            verdict: Some(LimiterVerdict::Known(PrimaryLimiter::Capacity)),
            capacity_skipped,
            traffic_skipped,
        };
    }

    // 2. Traffic - VRAM available but not enough concurrent requests to
    //    amortize overhead and saturate memory bandwidth.
    if let (Some(running), Some(ridge)) = (e.mean_running, e.ridge_batch_size) {
        let threshold = (ridge * TRAFFIC_LIMITER_RIDGE_FRACTION).max(TRAFFIC_LIMITER_MIN_RUNNING);
        if running < threshold {
            return IdentifyResult {
                verdict: Some(LimiterVerdict::Known(PrimaryLimiter::Traffic)),
                capacity_skipped,
                traffic_skipped,
            };
        }
    }

    // 3. Physics - efficiency headroom exhausted, or TPOT near theoretical floor.
    if e.headroom_pct
        .is_some_and(|h| h.is_finite() && h < HEADROOM_LIMITER_THRESHOLD_PCT)
    {
        return IdentifyResult {
            verdict: Some(LimiterVerdict::Known(PrimaryLimiter::Physics)),
            capacity_skipped,
            traffic_skipped,
        };
    }
    // Unknown GPU: no floor, no confident TPOT-based name.
    if e.mean_tpot_ms
        .is_some_and(|tpot| tpot.is_finite() && tpot > 0.0)
        && e.tpot_floor_ms
            .is_none_or(|floor| !floor.is_finite() || floor <= 0.0)
    {
        return IdentifyResult {
            verdict: Some(LimiterVerdict::CeilingUnknown(CeilingUnknown)),
            capacity_skipped,
            traffic_skipped,
        };
    }
    if let (Some(tpot), Some(floor)) = (e.mean_tpot_ms, e.tpot_floor_ms)
        && tpot <= floor * PHYSICS_LIMITER_FLOOR_MARGIN
    {
        return IdentifyResult {
            verdict: Some(LimiterVerdict::Known(PrimaryLimiter::Physics)),
            capacity_skipped,
            traffic_skipped,
        };
    }

    // 4. Prefill interference - measured prefill signal with chunked prefill as precondition.
    if e.chunked_prefill_enabled == Some(true)
        && e.effective_prompt_decode_ratio
            .is_some_and(|ratio| ratio.is_finite() && ratio >= LIMITER_PREFILL_RATIO_MIN)
    {
        return IdentifyResult {
            verdict: Some(LimiterVerdict::Known(PrimaryLimiter::PrefillInterference)),
            capacity_skipped,
            traffic_skipped,
        };
    }

    // 5. Framework overhead - only fire if we have enough signal to rule out data absence.
    if e.mean_running.is_some() && e.mean_tpot_ms.is_some() {
        return IdentifyResult {
            verdict: Some(LimiterVerdict::Known(PrimaryLimiter::FrameworkOverhead)),
            capacity_skipped,
            traffic_skipped,
        };
    }

    IdentifyResult {
        verdict: None,
        capacity_skipped,
        traffic_skipped,
    }
}

pub fn limiter_line(e: &LimiterEvidence) -> Option<String> {
    let result = identify(e);
    match result.verdict? {
        LimiterVerdict::CeilingUnknown(_) => {
            let reason = e
                .ceiling_unknown_reason
                .unwrap_or("hardware ceiling inputs incomplete");
            Some(format!("Hardware ceiling unknown ({reason})."))
        }
        LimiterVerdict::Known(PrimaryLimiter::Capacity) => {
            let mean = e.kv_cache_mean_perc?;
            let peak = e.kv_cache_peak_perc;
            let peak_s = peak.map(|p| format!(", {p:.0}% peak")).unwrap_or_default();
            Some(format!(
                "Capped by memory: KV cache at {mean:.0}% avg{peak_s} (R2 fires at {:.0}%). Concurrency cannot grow further on this pool.",
                crate::engine::rules::KV_CACHE_PRESSURE_MIN_PERC
            ))
        }
        LimiterVerdict::Known(PrimaryLimiter::Traffic) => {
            let running = e.mean_running?;
            let ridge = e.ridge_batch_size?;
            let mut line = format!(
                "Capped by traffic: {:.0} requests running, hardware has room for ~{ridge:.0}. More concurrent requests raises throughput.",
                running.trunc()
            );
            if result.capacity_skipped {
                line.push_str(" Memory unmeasured.");
            }
            Some(line)
        }
        LimiterVerdict::Known(PrimaryLimiter::Physics) => {
            if let (Some(tpot), Some(floor)) = (e.mean_tpot_ms, e.tpot_floor_ms)
                && tpot.is_finite()
                && floor.is_finite()
                && floor > 0.0
                && tpot <= floor * PHYSICS_LIMITER_FLOOR_MARGIN
            {
                Some(format!(
                    "Capped by hardware: TPOT {tpot:.1}ms vs ~{floor:.1}ms floor. This GPU is saturated, scale out to go faster."
                ))
            } else if e
                .headroom_pct
                .is_some_and(|h| h.is_finite() && h < HEADROOM_LIMITER_THRESHOLD_PCT)
            {
                Some(format!(
                    "Capped by hardware: efficiency headroom below {HEADROOM_LIMITER_THRESHOLD_PCT:.0}%. This GPU is saturated, scale out to go faster."
                ))
            } else {
                None
            }
        }
        LimiterVerdict::Known(PrimaryLimiter::PrefillInterference) => {
            let ratio = e.effective_prompt_decode_ratio?;
            Some(format!(
                "Capped by prefill: prompt work at {ratio:.1}x of decode (effective). Prefill shares bandwidth with decode on every step."
            ))
        }
        LimiterVerdict::Known(PrimaryLimiter::FrameworkOverhead) => {
            let mut line = String::from("Capped by framework: ");
            let mut cleared = Vec::new();
            if !result.traffic_skipped {
                cleared.push("batch healthy");
            }
            if !result.capacity_skipped {
                cleared.push("memory free");
            }
            if !cleared.is_empty() {
                line.push_str(&cleared.join(", "));
                line.push_str(". ");
            }
            line.push_str("GPU is waiting on vLLM/CPU overhead.");
            if result.capacity_skipped {
                line.push_str(" KV unmeasured.");
            }
            if result.traffic_skipped {
                // traffic_skipped = missing running or ridge; FrameworkOverhead
                // already requires running, so this is ridge (batch saturation target).
                line.push_str(" Ridge unmeasured.");
            }
            Some(line)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn ev(
        kv_mean: Option<f64>,
        kv_peak: Option<f64>,
        running: Option<f64>,
        ridge: Option<f64>,
        tpot: Option<f64>,
        floor: Option<f64>,
        ratio: Option<f64>,
        chunked: Option<bool>,
        headroom: Option<f64>,
    ) -> LimiterEvidence {
        LimiterEvidence {
            kv_cache_mean_perc: kv_mean,
            kv_cache_peak_perc: kv_peak,
            mean_running: running,
            ridge_batch_size: ridge,
            mean_tpot_ms: tpot,
            tpot_floor_ms: floor,
            effective_prompt_decode_ratio: ratio,
            chunked_prefill_enabled: chunked,
            headroom_pct: headroom,
            n_eval: crate::engine::ENGINE_MIN_PERSISTENT_WINDOWS,
            ceiling_unknown_reason: None,
        }
    }

    #[test]
    fn identify_declines_below_min_persistent_windows() {
        let mut e = ev(
            Some(85.0),
            Some(85.0),
            Some(50.0),
            Some(40.0),
            Some(20.0),
            Some(5.0),
            None,
            Some(false),
            None,
        );
        e.n_eval = 1;
        assert_eq!(identify(&e).verdict, None);
        assert!(limiter_line(&e).is_none());
        e.n_eval = crate::engine::ENGINE_MIN_PERSISTENT_WINDOWS;
        assert_eq!(
            identify(&e).verdict,
            Some(LimiterVerdict::Known(PrimaryLimiter::Capacity))
        );
    }

    #[test]
    fn capacity_fires_when_kv_above_threshold() {
        assert_eq!(
            identify(&ev(
                Some(85.0),
                Some(85.0),
                Some(50.0),
                Some(40.0),
                Some(20.0),
                Some(5.0),
                Some(0.2),
                Some(false),
                None
            ))
            .verdict,
            Some(LimiterVerdict::Known(PrimaryLimiter::Capacity))
        );
    }

    #[test]
    fn traffic_fires_when_running_below_ridge_fraction() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(50.0),
                Some(5.0),
                Some(100.0),
                Some(20.0),
                Some(5.0),
                Some(0.2),
                Some(false),
                None
            ))
            .verdict,
            Some(LimiterVerdict::Known(PrimaryLimiter::Traffic))
        );
    }

    #[test]
    fn physics_fires_when_tpot_near_floor() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(50.0),
                Some(50.0),
                Some(40.0),
                Some(12.0),
                Some(10.0),
                Some(0.2),
                Some(false),
                None
            ))
            .verdict,
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
            Some(50.0),
            Some(40.0),
            Some(15.0),
            Some(10.0),
            Some(0.2),
            Some(false),
            None,
        ))
        .verdict;
        assert_ne!(result, Some(LimiterVerdict::Known(PrimaryLimiter::Physics)));
    }

    #[test]
    fn physics_fires_when_headroom_below_threshold_even_if_tpot_above_floor() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(50.0),
                Some(50.0),
                Some(40.0),
                Some(50.0),
                Some(10.0),
                Some(0.2),
                Some(false),
                Some(2.0),
            ))
            .verdict,
            Some(LimiterVerdict::Known(PrimaryLimiter::Physics))
        );
        let line = limiter_line(&ev(
            Some(50.0),
            Some(50.0),
            Some(50.0),
            Some(40.0),
            Some(50.0),
            Some(10.0),
            Some(0.2),
            Some(false),
            Some(2.0),
        ))
        .expect("line");
        assert!(line.contains("headroom below 10%"));
    }

    #[test]
    fn prefill_interference_fires_when_chunked_prefill_on_and_ratio_high() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(50.0),
                Some(50.0),
                Some(40.0),
                Some(50.0),
                Some(10.0),
                Some(0.6),
                Some(true),
                None
            ))
            .verdict,
            Some(LimiterVerdict::Known(PrimaryLimiter::PrefillInterference))
        );
    }

    #[test]
    fn framework_overhead_is_fallthrough() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(50.0),
                Some(50.0),
                Some(40.0),
                Some(50.0),
                Some(10.0),
                Some(0.1),
                Some(false),
                None
            ))
            .verdict,
            Some(LimiterVerdict::Known(PrimaryLimiter::FrameworkOverhead))
        );
    }

    #[test]
    fn none_returned_when_data_missing() {
        assert_eq!(
            identify(&ev(None, None, None, None, None, None, None, None, None)).verdict,
            None
        );
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(50.0),
                None,
                Some(40.0),
                None,
                Some(10.0),
                None,
                Some(false),
                None
            ))
            .verdict,
            None
        );
    }

    #[test]
    fn capacity_takes_priority_over_traffic() {
        assert_eq!(
            identify(&ev(
                Some(85.0),
                Some(92.0),
                Some(2.0),
                Some(100.0),
                Some(20.0),
                Some(5.0),
                Some(0.6),
                Some(false),
                None
            ))
            .verdict,
            Some(LimiterVerdict::Known(PrimaryLimiter::Capacity))
        );
    }

    #[test]
    fn traffic_fires_when_peak_high_but_mean_below_capacity_bar() {
        assert_eq!(
            identify(&ev(
                Some(73.4),
                Some(92.5),
                Some(64.0),
                Some(295.0),
                Some(20.0),
                Some(5.0),
                Some(0.2),
                Some(false),
                None
            ))
            .verdict,
            Some(LimiterVerdict::Known(PrimaryLimiter::Traffic))
        );
    }

    #[test]
    fn traffic_takes_priority_over_physics() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(50.0),
                Some(5.0),
                Some(100.0),
                Some(11.0),
                Some(10.0),
                Some(0.6),
                Some(false),
                None
            ))
            .verdict,
            Some(LimiterVerdict::Known(PrimaryLimiter::Traffic))
        );
    }

    #[test]
    fn unknown_gpu_degrades_before_prefill_or_framework() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(50.0),
                Some(50.0),
                Some(100.0),
                Some(11.0),
                None,
                Some(0.7),
                Some(true),
                None
            ))
            .verdict,
            Some(LimiterVerdict::CeilingUnknown(CeilingUnknown))
        );
    }

    #[test]
    fn ceiling_unknown_line_uses_evidence_reason() {
        let mut e = ev(
            Some(50.0),
            Some(50.0),
            Some(50.0),
            Some(100.0),
            Some(11.0),
            None,
            Some(0.7),
            Some(true),
            None,
        );
        e.ceiling_unknown_reason = Some("model not in catalog");
        assert_eq!(
            limiter_line(&e).as_deref(),
            Some("Hardware ceiling unknown (model not in catalog).")
        );
    }

    #[test]
    fn ceiling_unknown_line_fallback_is_neutral_when_reason_unset() {
        // CeilingUnknown with floor None/non-finite/<=0 and reason unset
        // (baseline present with a bad floor, or hand-built evidence). Must not
        // invent "GPU not in catalog".
        let e = ev(
            Some(50.0),
            Some(50.0),
            Some(50.0),
            Some(100.0),
            Some(11.0),
            None,
            Some(0.7),
            Some(true),
            None,
        );
        assert_eq!(
            identify(&e).verdict,
            Some(LimiterVerdict::CeilingUnknown(CeilingUnknown))
        );
        assert_eq!(
            limiter_line(&e).as_deref(),
            Some("Hardware ceiling unknown (hardware ceiling inputs incomplete).")
        );
    }

    #[test]
    fn prefill_requires_chunked_precondition() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(50.0),
                Some(50.0),
                Some(100.0),
                Some(50.0),
                Some(10.0),
                Some(0.6),
                Some(false),
                None
            ))
            .verdict,
            Some(LimiterVerdict::Known(PrimaryLimiter::FrameworkOverhead))
        );
    }

    #[test]
    fn prefill_requires_effective_ratio_threshold() {
        assert_eq!(
            identify(&ev(
                Some(50.0),
                Some(50.0),
                Some(50.0),
                Some(100.0),
                Some(50.0),
                Some(10.0),
                Some(0.3),
                Some(true),
                None
            ))
            .verdict,
            Some(LimiterVerdict::Known(PrimaryLimiter::FrameworkOverhead))
        );
    }

    #[test]
    fn capacity_line_names_r2_relationship() {
        let line = limiter_line(&ev(
            Some(84.0),
            None,
            Some(50.0),
            Some(100.0),
            Some(20.0),
            Some(10.0),
            Some(0.2),
            Some(false),
            None,
        ))
        .expect("line");
        assert_eq!(
            line,
            "Capped by memory: KV cache at 84% avg (R2 fires at 88%). Concurrency cannot grow further on this pool."
        );
    }

    #[test]
    fn capacity_line_shows_mean_and_peak() {
        let line = limiter_line(&ev(
            Some(85.0),
            Some(92.0),
            Some(50.0),
            Some(100.0),
            Some(20.0),
            Some(10.0),
            Some(0.2),
            Some(false),
            None,
        ))
        .expect("line");
        assert_eq!(
            line,
            "Capped by memory: KV cache at 85% avg, 92% peak (R2 fires at 88%). Concurrency cannot grow further on this pool."
        );
    }

    #[test]
    fn traffic_line_contains_no_flags() {
        let line = limiter_line(&ev(
            Some(20.0),
            Some(20.0),
            Some(6.0),
            Some(153.0),
            Some(20.0),
            Some(10.0),
            Some(0.2),
            Some(false),
            None,
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
                Some(20.0),
                Some(50.0),
                Some(100.0),
                Some(50.0),
                Some(10.0),
                Some(0.3),
                Some(true),
                None
            ))
            .verdict,
            Some(LimiterVerdict::Known(PrimaryLimiter::FrameworkOverhead))
        );
    }

    #[test]
    fn framework_line_claims_only_cleared_stages() {
        let both_cleared = limiter_line(&ev(
            Some(20.0),
            Some(20.0),
            Some(50.0),
            Some(100.0),
            Some(50.0),
            Some(10.0),
            Some(0.2),
            Some(false),
            None,
        ))
        .expect("line");
        assert!(both_cleared.contains("batch healthy"));
        assert!(both_cleared.contains("memory free"));
        assert!(!both_cleared.contains("KV unmeasured"));
        assert!(!both_cleared.contains("Ridge unmeasured"));

        let kv_skipped = limiter_line(&ev(
            None,
            None,
            Some(50.0),
            Some(100.0),
            Some(50.0),
            Some(10.0),
            Some(0.2),
            Some(false),
            None,
        ))
        .expect("line");
        assert!(kv_skipped.contains("KV unmeasured"));
        assert!(kv_skipped.contains("batch healthy"));
        assert!(!kv_skipped.contains("memory free"));

        let traffic_skipped = limiter_line(&ev(
            Some(20.0),
            Some(20.0),
            Some(50.0),
            None,
            Some(50.0),
            Some(10.0),
            Some(0.2),
            Some(false),
            None,
        ))
        .expect("line");
        assert!(traffic_skipped.contains("memory free"));
        assert!(traffic_skipped.contains("Ridge unmeasured"));
        assert!(!traffic_skipped.contains("batch healthy"));
    }

    #[test]
    fn traffic_line_notes_memory_unmeasured_when_capacity_skipped() {
        let line = limiter_line(&ev(
            None,
            None,
            Some(5.0),
            Some(100.0),
            Some(20.0),
            Some(5.0),
            Some(0.2),
            Some(false),
            None,
        ))
        .expect("line");
        assert!(line.contains("Capped by traffic"));
        assert!(line.contains("Memory unmeasured"));
    }
}
