use crate::collectors::RawSnapshot;
use crate::engine::PhysicsBaseline;

use super::Recommendation;

/// Occupancy fraction below which the server is considered under-loaded.
const UNDER_BATCHING_OCCUPANCY_PCT: f64 = 0.10;

/// Waiting requests below this means no backlog pressure.
const UNDER_BATCHING_WAITING_LT: f64 = 2.0;

/// Efficiency below this (actual tps / decode ceiling) confirms genuine starvation.
/// Small models on frontier GPUs (e.g. Llama-8B on H200) may saturate network or
/// `max_num_seqs` before reaching 20% — lower to 10–15% if false fires observed.
const UNDER_BATCHING_EFFICIENCY_MAX_PCT: f64 = 20.0;

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
pub struct MissReport {
    pub running: Option<f64>,
    pub max_num_seqs: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Rule1Outcome {
    Fired(UnderBatchingDetail),
    NotFired(MissReport),
}

pub fn rule1_under_batching(snapshot: &RawSnapshot, baseline: &PhysicsBaseline) -> Rule1Outcome {
    let running = snapshot.vllm.num_requests_running;
    let max_num_seqs = snapshot.vllm.max_num_seqs;
    let waiting = snapshot.vllm.num_requests_waiting;

    let miss = || MissReport {
        running,
        max_num_seqs,
    };

    let Some(rv) = running.filter(|v| v.is_finite() && *v > 0.0) else {
        return Rule1Outcome::NotFired(miss());
    };
    let Some(max_n) = max_num_seqs.filter(|&n| n > 0) else {
        return Rule1Outcome::NotFired(miss());
    };
    let Some(wv) = waiting.filter(|v| v.is_finite()) else {
        return Rule1Outcome::NotFired(miss());
    };

    let occupancy = rv / f64::from(max_n);

    if !(occupancy < UNDER_BATCHING_OCCUPANCY_PCT && wv < UNDER_BATCHING_WAITING_LT) {
        return Rule1Outcome::NotFired(miss());
    }

    // Gate 1: prefill saturation — suppress if server is prefill-bound.
    // Hard abort if window duration missing — cannot calculate rates without dt.
    // Known limitation: Prometheus histograms only record on request completion.
    // A chunked prefill spanning the full window duration will read sum_delta=0
    // and bypass this gate until the request completes. Accepted — documented limitation.
    let window_secs = match snapshot.vllm.window_duration_secs {
        Some(w) if w.is_finite() && w > f64::EPSILON => w,
        _ => return Rule1Outcome::NotFired(miss()),
    };

    if let Some(mass) = snapshot.vllm.prefill_window_mass {
        let by_ratio = (mass.sum_delta / window_secs) > UNDER_BATCHING_PREFILL_SATURATION_MAX;
        let by_abs = mass.sum_delta > UNDER_BATCHING_PREFILL_ABS_SECS;
        if by_ratio || by_abs {
            return Rule1Outcome::NotFired(miss());
        }
    }

    // Gate 2: efficiency — suppress if server is performing well relative to hardware ceiling.
    // If efficiency is high, low occupancy means healthy lull between bursts, not starvation.
    if let Some(eff) = baseline.efficiency_pct {
        if eff >= UNDER_BATCHING_EFFICIENCY_MAX_PCT {
            return Rule1Outcome::NotFired(miss());
        }
    }

    Rule1Outcome::Fired(UnderBatchingDetail {
        running: rv,
        waiting: wv,
        max_num_seqs: Some(max_n),
        occupancy_pct: occupancy * 100.0,
    })
}

pub fn r1_recommendation(
    snapshot: &RawSnapshot,
    baseline: Option<&PhysicsBaseline>,
) -> Option<Recommendation> {
    let baseline = baseline?;
    let Rule1Outcome::Fired(d) = rule1_under_batching(snapshot, baseline) else {
        return None;
    };
    Some(Recommendation {
        rule_name: "under_batching",
        impact: 4,
        confidence: if baseline.efficiency_pct.is_some() { 0.9 } else { 0.7 },
        action: "Increase client concurrency".to_string(),
        expected_impact: "Higher throughput, stable TPOT".to_string(),
        display_lines: format_under_batching_fired(&d, baseline.efficiency_pct),
    })
}

pub(super) fn format_under_batching_fired(
    d: &UnderBatchingDetail,
    efficiency_pct: Option<f64>,
) -> Vec<String> {
    let threshold = UNDER_BATCHING_OCCUPANCY_PCT * 100.0;
    let idle_slots = d
        .max_num_seqs
        .map(|n| format!("{:.0}", f64::from(n) - d.running))
        .unwrap_or_else(|| "?".to_string());
    let max_str = d
        .max_num_seqs
        .map(|n| n.to_string())
        .unwrap_or_else(|| "?".to_string());
    let confidence = match efficiency_pct {
        Some(_) => "High",
        None => "Medium",
    };

    vec![
        "[!] Under-batching — Low Occupancy".to_string(),
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
        "    Server has unused capacity and no backlog. Clients are not sending enough requests."
            .to_string(),
        String::new(),
        "  Fix:".to_string(),
        format!(
            "    • Batch more requests or increase client concurrency ({idle_slots} slots idle)"
        ),
        String::new(),
        "  Expected: Higher throughput, lower TPOT at scale.".to_string(),
        format!("  Confidence: {confidence}"),
    ]
}

pub(super) fn format_under_batching_window_issue(
    d: &UnderBatchingDetail,
    seen_pct: u32,
    efficiency_pct: Option<f64>,
) -> Vec<String> {
    let mut lines = format_under_batching_fired(d, efficiency_pct);
    lines.insert(1, format!("  Seen in {seen_pct}% of windows"));
    lines
}

pub(super) fn aggregate_r1_detail(
    details: &[UnderBatchingDetail],
    _summary: &RawSnapshot,
    _baseline: Option<&PhysicsBaseline>,
) -> UnderBatchingDetail {
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
    use crate::engine::baseline::{CeilingEstimate, PhysicsBaseline, WeightDtypeSource};

    const TEST_WINDOW_SECS: f64 = 2.0;

    fn test_baseline() -> PhysicsBaseline {
        baseline_with_efficiency(None)
    }

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
        use std::time::SystemTime;
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

    fn baseline_with_efficiency(efficiency_pct: Option<f64>) -> PhysicsBaseline {
        PhysicsBaseline {
            decode: CeilingEstimate {
                lower: 100.0,
                expected: 100.0,
                upper: 100.0,
            },
            prefill: None,
            efficiency_pct,
            headroom_pct: None,
            weight_dtype_source: WeightDtypeSource::Fallback,
            weight_gb: 1.0,
            kv_headroom_gb: None,
            tpot_floor_ms: 10.0,
            prefill_latency_floor_ms: None,
            ridge_batch_size: 1.0,
        }
    }

    fn entry_fired_snap() -> RawSnapshot {
        snap(Some(5.0), Some(256), Some(0.0))
    }

    #[test]
    fn fires_when_occupancy_low() {
        let base = test_baseline();
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching(&s, &base) {
            Rule1Outcome::Fired(d) => {
                assert!((d.occupancy_pct - (5.0 / 256.0 * 100.0)).abs() < 0.1);
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired"),
        }
    }

    #[test]
    fn fires_at_occupancy_boundary() {
        let base = test_baseline();
        let s = snap(Some(25.0), Some(256), Some(0.0));
        match rule1_under_batching(&s, &base) {
            Rule1Outcome::Fired(d) => {
                assert!(d.occupancy_pct < 10.0);
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired at 9.8% occupancy"),
        }
    }

    #[test]
    fn mutes_at_occupancy_threshold() {
        let base = test_baseline();
        let s = snap(Some(26.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching(&s, &base),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_no_traffic() {
        let base = test_baseline();
        let s = snap(Some(0.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching(&s, &base),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_backpressure_at_two() {
        let base = test_baseline();
        let s = snap(Some(5.0), Some(256), Some(2.0));
        assert!(matches!(
            rule1_under_batching(&s, &base),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn fires_when_waiting_one_below_backpressure_gate() {
        let base = test_baseline();
        let s = snap(Some(5.0), Some(256), Some(1.0));
        assert!(matches!(
            rule1_under_batching(&s, &base),
            Rule1Outcome::Fired(_)
        ));
    }

    #[test]
    fn mutes_when_max_num_seqs_missing() {
        let base = test_baseline();
        let s = snap(Some(5.0), None, Some(0.0));
        assert!(matches!(
            rule1_under_batching(&s, &base),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_running_missing() {
        let base = test_baseline();
        let s = snap(None, Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching(&s, &base),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_window_duration_missing() {
        let base = test_baseline();
        let s = snap_with_gates(Some(5.0), Some(256), Some(0.0), None, None);
        assert!(matches!(
            rule1_under_batching(&s, &base),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn r1_recommendation_none_when_baseline_missing() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        assert!(r1_recommendation(&s, None).is_none());
    }

    #[test]
    fn prefill_gate_suppresses_when_ratio_above_threshold() {
        let base = test_baseline();
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
        assert!(matches!(
            rule1_under_batching(&s, &base),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn prefill_gate_suppresses_when_sum_delta_above_abs_floor() {
        let base = test_baseline();
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
        assert!(matches!(
            rule1_under_batching(&s, &base),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn efficiency_gate_suppresses_when_at_or_above_threshold() {
        let base = baseline_with_efficiency(Some(20.0));
        assert!(matches!(
            rule1_under_batching(&entry_fired_snap(), &base),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn fires_when_gates_absent() {
        let base = test_baseline();
        assert!(matches!(
            rule1_under_batching(&entry_fired_snap(), &base),
            Rule1Outcome::Fired(_)
        ));
    }

    #[test]
    fn fires_when_prefill_below_thresholds_and_efficiency_low() {
        let base = baseline_with_efficiency(Some(19.0));
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
        assert!(matches!(
            rule1_under_batching(&s, &base),
            Rule1Outcome::Fired(_)
        ));
    }
}
