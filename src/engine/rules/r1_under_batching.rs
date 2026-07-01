use crate::collectors::RawSnapshot;

use super::Recommendation;
use super::rule_names;

/// Occupancy ceiling: R1 does not fire above this. Server is not starved.
const OCCUPANCY_CEILING_PCT: f64 = 0.75;

/// Occupancy fallback threshold for unknown GPUs (no physics available).
const OCCUPANCY_FALLBACK_PCT: f64 = 0.25;

/// Config-relative efficiency below this means server is underperforming its config.
const CONFIG_EFFICIENCY_STARVATION_PCT: f64 = 60.0;

/// Prefill time fraction above this means prefill is the bottleneck, not traffic.
/// Ships as fixed 30%; replaced by f(ops_per_byte) after R6 calibration.
const PREFILL_FRACTION_GATE: f64 = 0.30;

/// Waiting requests below this means no backlog pressure.
const UNDER_BATCHING_WAITING_LT: f64 = 2.0;

#[derive(Debug, Clone, PartialEq)]
pub struct UnderBatchingDetail {
    pub running: f64,
    pub waiting: f64,
    pub max_num_seqs: Option<u32>,
    pub occupancy_pct: f64,
    pub efficiency_pct: Option<f64>,
    pub config_relative_efficiency_pct: Option<f64>,
    pub known_gpu: bool,
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

pub(super) fn rule1_under_batching_with_efficiency(
    snapshot: &RawSnapshot,
    config_max_num_seqs: Option<u32>,
    efficiency_pct: Option<f64>,
    config_relative_efficiency_pct: Option<f64>,
    prefill_time_fraction: Option<f64>,
) -> Rule1Outcome {
    let running = snapshot.vllm.num_requests_running;

    // 1. Hard abort - window duration required
    let _window_secs = match snapshot.vllm.window_duration_secs {
        Some(w) if w.is_finite() && w > f64::EPSILON => w,
        _ => {
            return Rule1Outcome::NotFired(R1MissReport {
                prefill_saturation_ratio: None,
            });
        }
    };

    // 2. Hard abort - max_num_seqs required (scrape or config)
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

    // 3. Hard abort - running required and > 0
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
    let efficiency_pct = efficiency_pct.filter(|e| e.is_finite());

    // Physics-level prefill gate: if prefill dominates, R6 handles it, not R1.
    if let Some(pf) = prefill_time_fraction.filter(|f| f.is_finite())
        && pf >= PREFILL_FRACTION_GATE
    {
        return Rule1Outcome::NotFired(R1MissReport {
            prefill_saturation_ratio: Some(pf),
        });
    }

    let known_gpu = config_relative_efficiency_pct.is_some();

    if known_gpu {
        // Known GPU: config-relative efficiency AND occupancy ceiling. Both must pass.
        // config_relative_efficiency_pct is Some (known_gpu = true).
        // unwrap_or(100.0) only triggers if the value is NaN/Inf (stripped by filter).
        // 100.0 = assume server is performing well = don't fire R1.
        let config_eff = config_relative_efficiency_pct
            .filter(|e| e.is_finite())
            .unwrap_or(100.0);
        if config_eff >= CONFIG_EFFICIENCY_STARVATION_PCT {
            return Rule1Outcome::NotFired(R1MissReport {
                prefill_saturation_ratio: None,
            });
        }
        if occupancy >= OCCUPANCY_CEILING_PCT {
            return Rule1Outcome::NotFired(R1MissReport {
                prefill_saturation_ratio: None,
            });
        }
    } else if occupancy >= OCCUPANCY_FALLBACK_PCT {
        // Unknown GPU: stricter occupancy-only fallback.
        return Rule1Outcome::NotFired(R1MissReport {
            prefill_saturation_ratio: None,
        });
    }

    Rule1Outcome::Fired(UnderBatchingDetail {
        running: run,
        waiting: wait,
        max_num_seqs: Some(max_n),
        occupancy_pct: occupancy * 100.0,
        efficiency_pct,
        config_relative_efficiency_pct,
        known_gpu,
    })
}

pub fn r1_recommendation(
    snapshot: &RawSnapshot,
    config_max_num_seqs: Option<u32>,
    efficiency_pct: Option<f64>,
    config_relative_efficiency_pct: Option<f64>,
    prefill_time_fraction: Option<f64>,
) -> Option<Recommendation> {
    let Rule1Outcome::Fired(d) = rule1_under_batching_with_efficiency(
        snapshot,
        config_max_num_seqs,
        efficiency_pct,
        config_relative_efficiency_pct,
        prefill_time_fraction,
    ) else {
        return None;
    };
    let confidence = if d.known_gpu { 0.8 } else { 0.5 };
    Some(Recommendation {
        rule_name: rule_names::UNDER_BATCHING,
        layer: 4,
        impact: 4,
        confidence,
        action: "Batch more requests or increase client concurrency".to_string(),
        short_action: r1_short_action(d.running, d.max_num_seqs),
        expected_impact: "Higher throughput, stable TPOT".to_string(),
        display_lines: format_under_batching_fired(&d, snapshot, confidence),
    })
}

pub fn r1_verbose_miss_line(
    snapshot: &RawSnapshot,
    config_max_num_seqs: Option<u32>,
    efficiency_pct: Option<f64>,
    config_relative_efficiency_pct: Option<f64>,
    prefill_time_fraction: Option<f64>,
) -> String {
    match rule1_under_batching_with_efficiency(
        snapshot,
        config_max_num_seqs,
        efficiency_pct,
        config_relative_efficiency_pct,
        prefill_time_fraction,
    ) {
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
    let metric_line = if let Some(eff) = d.config_relative_efficiency_pct {
        format!(
            "  Config efficiency  {eff:.1}%  (threshold: < {CONFIG_EFFICIENCY_STARVATION_PCT:.0}%)"
        )
    } else {
        format!(
            "  Occupancy  {:.1}%  (threshold: < {:.0}%)",
            d.occupancy_pct,
            OCCUPANCY_FALLBACK_PCT * 100.0
        )
    };
    let max_str = max_n.to_string();
    let idle = (f64::from(max_n) - display_run).max(0.0);
    let fix_line =
        format!("    • Batch more requests or increase client concurrency ({idle:.0} slots idle)");
    let confidence_str = if confidence >= 0.8 {
        "High"
    } else if confidence >= 0.6 {
        "Medium"
    } else {
        "Low"
    };

    let mut lines = vec![
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
    ];
    if !d.known_gpu {
        lines.push(
            "  Note: GPU not in catalog. Diagnosis based on occupancy only (low confidence)."
                .to_string(),
        );
    }
    lines
}

pub(super) fn format_under_batching_window_issue(
    d: &UnderBatchingDetail,
    seen_pct: u32,
    snapshot: &RawSnapshot,
    confidence: f64,
) -> Vec<String> {
    super::with_seen_pct(
        format_under_batching_fired(d, snapshot, confidence),
        seen_pct,
    )
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
        config_relative_efficiency_pct: {
            let values: Vec<f64> = details
                .iter()
                .filter_map(|d| d.config_relative_efficiency_pct)
                .collect();
            if values.is_empty() {
                None
            } else {
                Some(values.iter().sum::<f64>() / values.len() as f64)
            }
        },
        known_gpu: details.first().is_some_and(|d| d.known_gpu),
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
            gpus: vec![],

            nvml_host_gpu_count: None,
        }
    }

    fn entry_fired_snap() -> RawSnapshot {
        snap(Some(5.0), Some(256), Some(0.0))
    }

    #[test]
    fn fires_when_occupancy_low() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(&s, None, None, None, None) {
            Rule1Outcome::Fired(d) => {
                assert!((d.occupancy_pct - (5.0 / 256.0 * 100.0)).abs() < 0.1);
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired"),
        }
    }

    #[test]
    fn fires_at_occupancy_below_threshold() {
        let s = snap(Some(63.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(&s, None, None, None, None) {
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
            rule1_under_batching_with_efficiency(&s, None, None, None, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_no_traffic() {
        let s = snap(Some(0.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(&s, None, None, None, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_backpressure_at_two() {
        let s = snap(Some(5.0), Some(256), Some(2.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(&s, None, None, None, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn fires_when_waiting_one_below_backpressure_gate() {
        let s = snap(Some(5.0), Some(256), Some(1.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(&s, None, None, None, None),
            Rule1Outcome::Fired(_)
        ));
    }

    #[test]
    fn mutes_when_max_num_seqs_missing() {
        let s = snap(Some(5.0), None, Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(&s, None, None, None, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_max_num_seqs_is_zero() {
        let s = snap(Some(5.0), Some(0), Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(&s, None, None, None, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn fires_when_config_max_provides_capacity_and_occupancy_low() {
        let s = snap(Some(4.0), None, Some(0.0));
        match rule1_under_batching_with_efficiency(&s, Some(64), None, None, None) {
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
            rule1_under_batching_with_efficiency(&s, Some(64), None, None, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_running_missing() {
        let s = snap(None, Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(&s, None, None, None, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_window_duration_missing() {
        let s = snap_with_gates(Some(5.0), Some(256), Some(0.0), None, None);
        assert!(matches!(
            rule1_under_batching_with_efficiency(&s, None, None, None, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn r1_recommendation_fires_without_baseline() {
        let s = entry_fired_snap();
        let r = r1_recommendation(&s, None, None, None, None).expect("fired");
        assert_eq!(r.rule_name, rule_names::UNDER_BATCHING);
        assert!((r.confidence - 0.5).abs() < 1e-9);
    }

    #[test]
    fn short_action_includes_batch_or_increase_concurrency() {
        let s = entry_fired_snap();
        let r = r1_recommendation(&s, None, None, None, None).expect("fired");
        assert_eq!(
            r.short_action,
            "batch more requests or increase client concurrency (251 slots idle)"
        );
    }

    #[test]
    fn fix_line_omits_kv_ceiling_even_when_known() {
        let s = entry_fired_snap();
        let r = r1_recommendation(&s, None, None, None, None).expect("fired");
        let text = r.display_lines.join("\n");
        assert!(
            text.contains(
                "    • Batch more requests or increase client concurrency (251 slots idle)"
            )
        );
        assert!(!text.contains("hardware limit"));
        assert!(!text.contains("KV ceiling"));
    }

    #[test]
    fn format_under_batching_fired_shows_config_efficiency_on_known_gpu_path() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(&s, None, Some(25.0), Some(15.0), None) {
            Rule1Outcome::Fired(d) => {
                assert!(d.known_gpu);
                assert_eq!(d.config_relative_efficiency_pct, Some(15.0));
                let text = format_under_batching_fired(&d, &s, 0.8).join("\n");
                assert!(text.contains("Config efficiency  15.0%"));
                assert!(text.contains("threshold: < 60%"));
                assert!(!text.contains("Occupancy"));
            }
            Rule1Outcome::NotFired(_) => panic!("expected fire via config efficiency path"),
        }
    }

    #[test]
    fn format_under_batching_fired_shows_occupancy_on_unknown_gpu_path() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(&s, None, None, None, None) {
            Rule1Outcome::Fired(d) => {
                assert!(!d.known_gpu);
                let text = format_under_batching_fired(&d, &s, 0.5).join("\n");
                assert!(text.contains("Occupancy"));
                assert!(text.contains("threshold: < 25%"));
                assert!(text.contains("low confidence"));
                assert!(!text.contains("Config efficiency"));
            }
            Rule1Outcome::NotFired(_) => panic!("expected fire via occupancy fallback"),
        }
    }

    #[test]
    fn known_gpu_fires_when_config_eff_low_and_occupancy_low() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(&s, None, None, Some(15.0), None) {
            Rule1Outcome::Fired(d) => {
                assert!(d.known_gpu);
                assert_eq!(d.config_relative_efficiency_pct, Some(15.0));
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired"),
        }
    }

    #[test]
    fn known_gpu_mutes_when_config_eff_high() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(&s, None, None, Some(75.0), None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn known_gpu_mutes_when_occupancy_above_ceiling() {
        let s = snap(Some(200.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(&s, None, None, Some(15.0), None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn unknown_gpu_fires_when_occupancy_below_fallback() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(&s, None, None, None, None) {
            Rule1Outcome::Fired(d) => {
                assert!(!d.known_gpu);
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired on unknown GPU fallback"),
        }
    }

    #[test]
    fn unknown_gpu_mutes_when_occupancy_above_fallback() {
        let s = snap(Some(70.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(&s, None, None, None, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn prefill_fraction_gate_suppresses_before_occupancy_check() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(&s, None, None, Some(15.0), Some(0.35)) {
            Rule1Outcome::NotFired(m) => {
                assert!(m.prefill_saturation_ratio.is_some());
            }
            Rule1Outcome::Fired(_) => panic!("expected suppressed by prefill gate"),
        }
    }

    #[test]
    fn unknown_gpu_confidence_is_low() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        let r = r1_recommendation(&s, None, None, None, None).expect("fired");
        assert!((r.confidence - 0.5).abs() < 1e-9);
    }

    #[test]
    fn known_gpu_confidence_is_high() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        let r = r1_recommendation(&s, None, None, Some(15.0), None).expect("fired");
        assert!((r.confidence - 0.8).abs() < 1e-9);
    }

    #[test]
    fn full_and_gate_fires_when_all_conditions_met() {
        // All four gates pass: config_eff=15% < 60%, occupancy=1.95% < 75%,
        // prefill=0.20 < 0.30, waiting=0 < 2.
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(&s, None, None, Some(15.0), Some(0.20)) {
            Rule1Outcome::Fired(d) => {
                assert!(d.known_gpu);
                assert_eq!(d.config_relative_efficiency_pct, Some(15.0));
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired with all gates passing"),
        }
    }
}
