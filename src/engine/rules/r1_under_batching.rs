use crate::collectors::RawSnapshot;

use super::Recommendation;

/// Occupancy fraction below which the server is considered under-loaded.
const UNDER_BATCHING_OCCUPANCY_PCT: f64 = 0.25;

/// Waiting requests below this means no backlog pressure.
const UNDER_BATCHING_WAITING_LT: f64 = 2.0;

/// Prefill saturation ratio above which the server is considered prefill-bound.
/// 40% of wall-clock time in prefill compute = server is not starved for work.
const UNDER_BATCHING_PREFILL_SATURATION_MAX: f64 = 0.40;

/// Absolute prefill time floor (seconds). Catches long windows where ratio dilutes.
/// If sum_delta > 4.0s, server did substantial prefill work regardless of window length.
const UNDER_BATCHING_PREFILL_ABS_SECS: f64 = 4.0;

#[derive(Debug, Clone, PartialEq)]
pub struct UnderBatchingDetail {
    pub running: f64,
    pub waiting: f64,
    pub max_num_seqs: Option<u32>,
    pub occupancy_pct: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Rule1Outcome {
    Fired(UnderBatchingDetail),
    NotFired,
}

pub fn rule1_under_batching(snapshot: &RawSnapshot) -> Rule1Outcome {
    let running = snapshot.vllm.num_requests_running;
    let max_num_seqs = snapshot.vllm.max_num_seqs;

    // 1. Hard abort — window duration required
    let window_secs = match snapshot.vllm.window_duration_secs {
        Some(w) if w.is_finite() && w > f64::EPSILON => w,
        _ => return Rule1Outcome::NotFired,
    };

    // 2. Hard abort — max_num_seqs required
    let Some(max_n) = max_num_seqs.filter(|&n| n > 0) else {
        return Rule1Outcome::NotFired;
    };

    // 3. Hard abort — running required and > 0
    let Some(run) = running.filter(|v| v.is_finite() && *v > 0.0) else {
        return Rule1Outcome::NotFired;
    };

    // 4. Occupancy + backlog check
    let occupancy = run / f64::from(max_n);
    let Some(wait) = snapshot.vllm.num_requests_waiting.filter(|v| v.is_finite()) else {
        return Rule1Outcome::NotFired;
    };
    if occupancy >= UNDER_BATCHING_OCCUPANCY_PCT || wait >= UNDER_BATCHING_WAITING_LT {
        return Rule1Outcome::NotFired;
    }

    // 5. Gate 1 — prefill saturation
    // Known limitation: Prometheus histograms only record on request completion.
    // A chunked prefill spanning the full window duration will read sum_delta=0
    // and bypass this gate until the request completes. Accepted — documented limitation.
    if let Some(mass) = snapshot.vllm.prefill_window_mass {
        let by_ratio = (mass.sum_delta / window_secs) > UNDER_BATCHING_PREFILL_SATURATION_MAX;
        let by_abs = mass.sum_delta > UNDER_BATCHING_PREFILL_ABS_SECS;
        if by_ratio || by_abs {
            return Rule1Outcome::NotFired;
        }
    }

    Rule1Outcome::Fired(UnderBatchingDetail {
        running: run,
        waiting: wait,
        max_num_seqs: Some(max_n),
        occupancy_pct: occupancy * 100.0,
    })
}

pub fn r1_recommendation(snapshot: &RawSnapshot) -> Option<Recommendation> {
    let Rule1Outcome::Fired(d) = rule1_under_batching(snapshot) else {
        return None;
    };
    Some(Recommendation {
        rule_name: "under_batching",
        impact: 4,
        confidence: 0.8,
        action: "Increase client concurrency".to_string(),
        expected_impact: "Higher throughput, stable TPOT".to_string(),
        display_lines: format_under_batching_fired(&d, 0.8),
    })
}

pub(super) fn format_under_batching_fired(d: &UnderBatchingDetail, confidence: f64) -> Vec<String> {
    let threshold = UNDER_BATCHING_OCCUPANCY_PCT * 100.0;
    let idle_slots = d
        .max_num_seqs
        .map(|n| format!("{:.0}", f64::from(n) - d.running))
        .unwrap_or_else(|| "?".to_string());
    let max_str = d
        .max_num_seqs
        .map(|n| n.to_string())
        .unwrap_or_else(|| "?".to_string());
    let confidence_str = if confidence >= 0.8 { "High" } else { "Medium" };

    vec![
        "[!] Under-batching — Insufficient Concurrency".to_string(),
        String::new(),
        format!(
            "  Occupancy  {:.1}%  (threshold: < {threshold:.0}%)",
            d.occupancy_pct
        ),
        format!(
            "  Requests   {:.0} running, {:.0} waiting  (max: {max_str})",
            d.running, d.waiting
        ),
        String::new(),
        "  Cause:".to_string(),
        "    Hardware capacity under-fed by client. Not enough requests arriving to keep the server busy."
            .to_string(),
        String::new(),
        "  Fix:".to_string(),
        format!(
            "    • Batch more requests or increase client concurrency ({idle_slots} slots idle)"
        ),
        String::new(),
        "  Expected: Higher throughput, lower TPOT at scale.".to_string(),
        format!("  Confidence: {confidence_str}"),
    ]
}

pub(super) fn format_under_batching_window_issue(
    d: &UnderBatchingDetail,
    seen_pct: u32,
    confidence: f64,
) -> Vec<String> {
    let mut lines = format_under_batching_fired(d, confidence);
    lines.insert(1, format!("  Seen in {seen_pct}% of windows"));
    lines
}

pub(super) fn aggregate_r1_detail(details: &[UnderBatchingDetail]) -> UnderBatchingDetail {
    debug_assert!(
        !details.is_empty(),
        "aggregate_r1_detail called with empty details"
    );
    let n = details.len() as f64;
    let running = details.iter().map(|d| d.running).sum::<f64>() / n;
    let waiting = details.iter().map(|d| d.waiting).sum::<f64>() / n;
    let occupancy_pct = details.iter().map(|d| d.occupancy_pct).sum::<f64>() / n;
    UnderBatchingDetail {
        running,
        waiting,
        max_num_seqs: details.first().and_then(|d| d.max_num_seqs),
        occupancy_pct,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{HistogramWindowMass, VllmRawMetrics};
    use std::time::SystemTime;

    const TEST_WINDOW_SECS: f64 = 2.0;

    fn snap(running: Option<f64>, max_num_seqs: Option<u32>, waiting: Option<f64>) -> RawSnapshot {
        snap_with_gates(running, max_num_seqs, waiting, None, Some(TEST_WINDOW_SECS))
    }

    fn snap_with_gates(
        running: Option<f64>,
        max_num_seqs: Option<u32>,
        waiting: Option<f64>,
        prefill_mass: Option<HistogramWindowMass>,
        window_duration_secs: Option<f64>,
    ) -> RawSnapshot {
        let t = SystemTime::UNIX_EPOCH;
        RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: VllmRawMetrics {
                num_requests_running: running,
                num_requests_waiting: waiting,
                max_num_seqs,
                prefill_window_mass: prefill_mass,
                window_duration_secs,
                ..Default::default()
            },
            gpu: Default::default(),
        }
    }

    fn entry_fired_snap() -> RawSnapshot {
        snap(Some(5.0), Some(256), Some(0.0))
    }

    #[test]
    fn fires_when_occupancy_low() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching(&s) {
            Rule1Outcome::Fired(d) => {
                assert!((d.occupancy_pct - (5.0 / 256.0 * 100.0)).abs() < 0.1);
            }
            Rule1Outcome::NotFired => panic!("expected fired"),
        }
    }

    #[test]
    fn fires_at_occupancy_below_threshold() {
        let s = snap(Some(63.0), Some(256), Some(0.0));
        match rule1_under_batching(&s) {
            Rule1Outcome::Fired(d) => {
                assert!(d.occupancy_pct < 25.0);
            }
            Rule1Outcome::NotFired => panic!("expected fired below 25% occupancy"),
        }
    }

    #[test]
    fn mutes_at_occupancy_threshold() {
        let s = snap(Some(64.0), Some(256), Some(0.0));
        assert!(matches!(rule1_under_batching(&s), Rule1Outcome::NotFired));
    }

    #[test]
    fn mutes_when_no_traffic() {
        let s = snap(Some(0.0), Some(256), Some(0.0));
        assert!(matches!(rule1_under_batching(&s), Rule1Outcome::NotFired));
    }

    #[test]
    fn mutes_when_backpressure_at_two() {
        let s = snap(Some(5.0), Some(256), Some(2.0));
        assert!(matches!(rule1_under_batching(&s), Rule1Outcome::NotFired));
    }

    #[test]
    fn fires_when_waiting_one_below_backpressure_gate() {
        let s = snap(Some(5.0), Some(256), Some(1.0));
        assert!(matches!(rule1_under_batching(&s), Rule1Outcome::Fired(_)));
    }

    #[test]
    fn mutes_when_max_num_seqs_missing() {
        let s = snap(Some(5.0), None, Some(0.0));
        assert!(matches!(rule1_under_batching(&s), Rule1Outcome::NotFired));
    }

    #[test]
    fn mutes_when_max_num_seqs_is_zero() {
        let s = snap(Some(5.0), Some(0), Some(0.0));
        assert!(matches!(rule1_under_batching(&s), Rule1Outcome::NotFired));
    }

    #[test]
    fn mutes_when_running_missing() {
        let s = snap(None, Some(256), Some(0.0));
        assert!(matches!(rule1_under_batching(&s), Rule1Outcome::NotFired));
    }

    #[test]
    fn mutes_when_window_duration_missing() {
        let s = snap_with_gates(Some(5.0), Some(256), Some(0.0), None, None);
        assert!(matches!(rule1_under_batching(&s), Rule1Outcome::NotFired));
    }

    #[test]
    fn prefill_gate_suppresses_when_ratio_above_threshold() {
        let s = snap_with_gates(
            Some(5.0),
            Some(256),
            Some(0.0),
            Some(HistogramWindowMass {
                sum_delta: 5.0,
                count_delta: 10.0,
            }),
            Some(10.0),
        );
        assert!(matches!(rule1_under_batching(&s), Rule1Outcome::NotFired));
    }

    #[test]
    fn prefill_gate_suppresses_when_sum_delta_above_abs_floor() {
        let s = snap_with_gates(
            Some(5.0),
            Some(256),
            Some(0.0),
            Some(HistogramWindowMass {
                sum_delta: 5.0,
                count_delta: 10.0,
            }),
            Some(100.0),
        );
        assert!(matches!(rule1_under_batching(&s), Rule1Outcome::NotFired));
    }

    #[test]
    fn fires_when_prefill_below_thresholds() {
        let s = snap_with_gates(
            Some(5.0),
            Some(256),
            Some(0.0),
            Some(HistogramWindowMass {
                sum_delta: 1.0,
                count_delta: 2.0,
            }),
            Some(10.0),
        );
        assert!(matches!(rule1_under_batching(&s), Rule1Outcome::Fired(_)));
    }

    #[test]
    fn r1_recommendation_fires_without_baseline() {
        let s = entry_fired_snap();
        let r = r1_recommendation(&s).expect("fired");
        assert_eq!(r.rule_name, "under_batching");
        assert!((r.confidence - 0.8).abs() < 1e-9);
    }
}
