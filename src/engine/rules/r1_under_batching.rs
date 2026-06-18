use crate::collectors::RawSnapshot;

use super::Recommendation;

/// Occupancy fraction below which the server is considered under-loaded.
const UNDER_BATCHING_OCCUPANCY_PCT: f64 = 0.25;

/// Waiting requests below this means no backlog pressure.
const UNDER_BATCHING_WAITING_LT: f64 = 2.0;

/// Prefill saturation ratio above which the server is considered prefill-bound.
/// 40% of wall-clock time in prefill compute = server is not starved for work.
const UNDER_BATCHING_PREFILL_SATURATION_MAX: f64 = 0.40;
const EFFICIENCY_STARVATION_PCT: f64 = 60.0;

#[derive(Debug, Clone, PartialEq)]
pub struct UnderBatchingDetail {
    pub running: f64,
    pub waiting: f64,
    pub max_num_seqs: Option<u32>,
    pub occupancy_pct: f64,
    /// Some if the physics efficiency gate fired; None if the occupancy fallback fired.
    pub efficiency_pct: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R1MissReport {
    /// Ratio of prefill time to window duration when the prefill gate suppressed.
    /// None when suppressed for any other reason (missing data, occupancy, backlog).
    pub prefill_saturation_ratio: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Rule1Outcome {
    Fired(UnderBatchingDetail),
    NotFired(R1MissReport),
}

pub fn rule1_under_batching(
    snapshot: &RawSnapshot,
    config_max_num_seqs: Option<u32>,
) -> Rule1Outcome {
    rule1_under_batching_with_efficiency(snapshot, config_max_num_seqs, None)
}

pub(super) fn rule1_under_batching_with_efficiency(
    snapshot: &RawSnapshot,
    config_max_num_seqs: Option<u32>,
    efficiency_pct: Option<f64>,
) -> Rule1Outcome {
    let running = snapshot.vllm.num_requests_running;

    // 1. Hard abort — window duration required
    let window_secs = match snapshot.vllm.window_duration_secs {
        Some(w) if w.is_finite() && w > f64::EPSILON => w,
        _ => {
            return Rule1Outcome::NotFired(R1MissReport {
                prefill_saturation_ratio: None,
            });
        }
    };

    // 2. Hard abort — max_num_seqs required (scrape or config)
    let Some(max_n) = snapshot
        .vllm
        .max_num_seqs
        .or(config_max_num_seqs)
        .filter(|&n| n > 0)
    else {
        return Rule1Outcome::NotFired(R1MissReport {
            prefill_saturation_ratio: None,
        });
    };

    // 3. Hard abort — running required and > 0
    let Some(run) = running.filter(|v| v.is_finite() && *v > 0.0) else {
        return Rule1Outcome::NotFired(R1MissReport {
            prefill_saturation_ratio: None,
        });
    };

    // 4. Occupancy + backlog check
    let occupancy = run / f64::from(max_n);
    let Some(wait) = snapshot.vllm.num_requests_waiting.filter(|v| v.is_finite()) else {
        return Rule1Outcome::NotFired(R1MissReport {
            prefill_saturation_ratio: None,
        });
    };
    if wait >= UNDER_BATCHING_WAITING_LT {
        return Rule1Outcome::NotFired(R1MissReport {
            prefill_saturation_ratio: None,
        });
    }
    let efficiency_starvation_gate = efficiency_pct.filter(|e| e.is_finite());
    let use_efficiency_path = efficiency_starvation_gate.is_some();
    if let Some(eff) = efficiency_starvation_gate {
        if eff >= EFFICIENCY_STARVATION_PCT {
            return Rule1Outcome::NotFired(R1MissReport {
                prefill_saturation_ratio: None,
            });
        }
    } else if occupancy >= UNDER_BATCHING_OCCUPANCY_PCT {
        return Rule1Outcome::NotFired(R1MissReport {
            prefill_saturation_ratio: None,
        });
    }

    // 5. Gate 1 — prefill saturation
    // Known limitation: Prometheus histograms only record on request completion.
    // A chunked prefill spanning the full window duration will read sum_delta=0
    // and bypass this gate until the request completes. Accepted — documented limitation.
    if let Some(mass) = snapshot.vllm.prefill_window_mass {
        if mass.count_delta > 0.0 {
            let mean_prefill_secs = mass.sum_delta / mass.count_delta;
            let ratio = mean_prefill_secs / window_secs;
            if ratio > UNDER_BATCHING_PREFILL_SATURATION_MAX {
                return Rule1Outcome::NotFired(R1MissReport {
                    prefill_saturation_ratio: Some(ratio),
                });
            }
        }
    }

    Rule1Outcome::Fired(UnderBatchingDetail {
        running: run,
        waiting: wait,
        max_num_seqs: Some(max_n),
        occupancy_pct: occupancy * 100.0,
        efficiency_pct: if use_efficiency_path {
            efficiency_starvation_gate
        } else {
            None
        },
    })
}

pub fn r1_recommendation(
    snapshot: &RawSnapshot,
    config_max_num_seqs: Option<u32>,
    efficiency_pct: Option<f64>,
) -> Option<Recommendation> {
    let Rule1Outcome::Fired(d) =
        rule1_under_batching_with_efficiency(snapshot, config_max_num_seqs, efficiency_pct)
    else {
        return None;
    };
    Some(Recommendation {
        rule_name: "under_batching",
        impact: 4,
        confidence: 0.8,
        action: "Batch more requests or increase client concurrency".to_string(),
        short_action: r1_short_action(d.running, d.max_num_seqs),
        expected_impact: "Higher throughput, stable TPOT".to_string(),
        display_lines: format_under_batching_fired(&d, snapshot, 0.8),
    })
}

pub fn r1_verbose_miss_line(
    snapshot: &RawSnapshot,
    config_max_num_seqs: Option<u32>,
    efficiency_pct: Option<f64>,
) -> String {
    match rule1_under_batching_with_efficiency(snapshot, config_max_num_seqs, efficiency_pct) {
        Rule1Outcome::NotFired(m) => {
            if let Some(ratio) = m.prefill_saturation_ratio {
                format!(
                    "Under-batching: not triggered (prefill saturated at {:.0}%)",
                    ratio * 100.0
                )
            } else {
                "Under-batching: not triggered".to_string()
            }
        }
        Rule1Outcome::Fired(_) => "Under-batching: not triggered".to_string(),
    }
}

pub(super) fn r1_short_action(running: f64, max_num_seqs: Option<u32>) -> String {
    match max_num_seqs {
        Some(max_n) => {
            let idle = (f64::from(max_n) - running).max(0.0);
            format!("batch more requests or increase client concurrency ({idle:.0} slots idle)")
        }
        None => "batch more requests or increase client concurrency".to_string(),
    }
}

pub(super) fn format_under_batching_fired(
    d: &UnderBatchingDetail,
    snapshot: &RawSnapshot,
    confidence: f64,
) -> Vec<String> {
    let Some(max_n) = d.max_num_seqs else {
        // Structurally unreachable: r1 hard-aborts without max_num_seqs.
        return Vec::new();
    };
    let display_run = snapshot
        .vllm
        .num_requests_running
        .filter(|v| v.is_finite())
        .unwrap_or(d.running);
    let display_wait = snapshot
        .vllm
        .num_requests_waiting
        .filter(|v| v.is_finite())
        .unwrap_or(d.waiting);
    let threshold = UNDER_BATCHING_OCCUPANCY_PCT * 100.0;
    let max_str = max_n.to_string();
    let metric_line = if let Some(eff) = d.efficiency_pct {
        format!("  Efficiency  {eff:.1}%  (threshold: < {EFFICIENCY_STARVATION_PCT:.0}%)")
    } else {
        format!(
            "  Occupancy  {:.1}%  (threshold: < {threshold:.0}%)",
            d.occupancy_pct
        )
    };
    let fix_line = match d.max_num_seqs {
        Some(max_n) => {
            let idle = (f64::from(max_n) - display_run).max(0.0);
            format!(
                "    • Batch more requests or increase client concurrency ({idle:.0} slots idle)"
            )
        }
        None => "    • Batch more requests or increase client concurrency".to_string(),
    };
    let confidence_str = if confidence >= 0.8 { "High" } else { "Medium" };

    vec![
        "[!] Under-batching: Insufficient Concurrency".to_string(),
        String::new(),
        metric_line,
        format!(
            "  Requests   {:.0} running, {:.0} waiting  (max: {max_str})",
            display_run, display_wait
        ),
        String::new(),
        "  Cause:".to_string(),
        "    Hardware capacity under-fed by client. Not enough requests arriving to keep the server busy."
            .to_string(),
        String::new(),
        "  Fix:".to_string(),
        fix_line,
        String::new(),
        "  Expected: Higher throughput, stable TPOT.".to_string(),
        format!("  Confidence: {confidence_str}"),
    ]
}

pub(super) fn format_under_batching_window_issue(
    d: &UnderBatchingDetail,
    seen_pct: u32,
    snapshot: &RawSnapshot,
    confidence: f64,
) -> Vec<String> {
    let mut lines = format_under_batching_fired(d, snapshot, confidence);
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
        efficiency_pct: {
            let values: Vec<f64> = details.iter().filter_map(|d| d.efficiency_pct).collect();
            if values.is_empty() {
                None
            } else {
                Some(values.iter().sum::<f64>() / values.len() as f64)
            }
        },
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
        match rule1_under_batching(&s, None) {
            Rule1Outcome::Fired(d) => {
                assert!((d.occupancy_pct - (5.0 / 256.0 * 100.0)).abs() < 0.1);
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired"),
        }
    }

    #[test]
    fn fires_at_occupancy_below_threshold() {
        let s = snap(Some(63.0), Some(256), Some(0.0));
        match rule1_under_batching(&s, None) {
            Rule1Outcome::Fired(d) => {
                assert!(d.occupancy_pct < 25.0);
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired below 25% occupancy"),
        }
    }

    #[test]
    fn mutes_at_occupancy_threshold() {
        let s = snap(Some(64.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_no_traffic() {
        let s = snap(Some(0.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_backpressure_at_two() {
        let s = snap(Some(5.0), Some(256), Some(2.0));
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn fires_when_waiting_one_below_backpressure_gate() {
        let s = snap(Some(5.0), Some(256), Some(1.0));
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::Fired(_)
        ));
    }

    #[test]
    fn mutes_when_max_num_seqs_missing() {
        let s = snap(Some(5.0), None, Some(0.0));
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_max_num_seqs_is_zero() {
        let s = snap(Some(5.0), Some(0), Some(0.0));
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn fires_when_config_max_provides_capacity_and_occupancy_low() {
        let s = snap(Some(4.0), None, Some(0.0));
        match rule1_under_batching(&s, Some(64)) {
            Rule1Outcome::Fired(d) => {
                assert_eq!(d.max_num_seqs, Some(64));
                assert!((d.occupancy_pct - (4.0 / 64.0 * 100.0)).abs() < 0.1);
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired with config max 64"),
        }
    }

    #[test]
    fn mutes_at_occupancy_threshold_with_config_max_only() {
        let s = snap(Some(64.0), None, Some(0.0));
        assert!(matches!(
            rule1_under_batching(&s, Some(64)),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_running_missing() {
        let s = snap(None, Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_window_duration_missing() {
        let s = snap_with_gates(Some(5.0), Some(256), Some(0.0), None, None);
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn prefill_gate_suppresses_when_ratio_above_threshold() {
        let s = snap_with_gates(
            Some(5.0),
            Some(256),
            Some(0.0),
            Some(HistogramWindowMass {
                sum_delta: 4.0,
                count_delta: 2.0,
            }),
            Some(4.0),
        );
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn prefill_gate_does_not_suppress_when_concurrent_short_prefills_inflate_sum() {
        // 10 requests × 0.2s prefill = sum_delta 2.0s, but mean = 0.2s in 1.0s window = 20%
        let s = snap_with_gates(
            Some(5.0),
            Some(256),
            Some(0.0),
            Some(HistogramWindowMass {
                sum_delta: 2.0,
                count_delta: 10.0,
            }),
            Some(1.0),
        );
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::Fired(_)
        ));
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
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::Fired(_)
        ));
    }

    #[test]
    fn prefill_gate_miss_report_carries_ratio() {
        let s = snap_with_gates(
            Some(5.0),
            Some(256),
            Some(0.0),
            Some(HistogramWindowMass {
                sum_delta: 1.6,
                count_delta: 2.0,
            }),
            Some(1.0),
        );
        match rule1_under_batching(&s, None) {
            Rule1Outcome::NotFired(m) => {
                let ratio = m.prefill_saturation_ratio.expect("ratio present");
                assert!((ratio - 0.80).abs() < 1e-9);
            }
            Rule1Outcome::Fired(_) => panic!("expected not fired"),
        }
    }

    #[test]
    fn r1_verbose_miss_line_shows_prefill_saturation_ratio() {
        let s = snap_with_gates(
            Some(5.0),
            Some(256),
            Some(0.0),
            Some(HistogramWindowMass {
                sum_delta: 1.6,
                count_delta: 2.0,
            }),
            Some(1.0),
        );
        let line = r1_verbose_miss_line(&s, None, None);
        assert!(line.contains("prefill saturated at 80%"));
    }

    #[test]
    fn r1_recommendation_fires_without_baseline() {
        let s = entry_fired_snap();
        let r = r1_recommendation(&s, None, None).expect("fired");
        assert_eq!(r.rule_name, "under_batching");
        assert!((r.confidence - 0.8).abs() < 1e-9);
    }

    #[test]
    fn short_action_includes_batch_or_increase_concurrency() {
        let s = entry_fired_snap();
        let r = r1_recommendation(&s, None, None).expect("fired");
        assert_eq!(
            r.short_action,
            "batch more requests or increase client concurrency (251 slots idle)"
        );
    }

    #[test]
    fn fix_line_omits_kv_ceiling_even_when_known() {
        let s = entry_fired_snap();
        let r = r1_recommendation(&s, None, None).expect("fired");
        let text = r.display_lines.join("\n");
        assert!(text
            .contains("    • Batch more requests or increase client concurrency (251 slots idle)"));
        assert!(!text.contains("hardware limit"));
        assert!(!text.contains("KV ceiling"));
    }

    #[test]
    fn format_under_batching_fired_shows_efficiency_on_physics_path() {
        let s = snap(Some(64.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(&s, None, Some(25.0)) {
            Rule1Outcome::Fired(d) => {
                assert_eq!(d.efficiency_pct, Some(25.0));
                let text = format_under_batching_fired(&d, &s, 0.8).join("\n");
                assert!(text.contains("Efficiency  25.0%"));
                assert!(text.contains("threshold: < 60%"));
                assert!(!text.contains("Occupancy"));
            }
            Rule1Outcome::NotFired(_) => panic!("expected fire via efficiency path"),
        }
    }

    #[test]
    fn format_under_batching_fired_shows_occupancy_on_fallback_path() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(&s, None, None) {
            Rule1Outcome::Fired(d) => {
                assert!(d.efficiency_pct.is_none());
                let text = format_under_batching_fired(&d, &s, 0.8).join("\n");
                assert!(text.contains("Occupancy"));
                assert!(text.contains("threshold: < 25%"));
                assert!(!text.contains("Efficiency"));
            }
            Rule1Outcome::NotFired(_) => panic!("expected fire via occupancy fallback"),
        }
    }
}
