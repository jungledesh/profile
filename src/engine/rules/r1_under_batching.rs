use crate::collectors::RawSnapshot;

#[cfg(test)]
use super::Recommendation;
use super::r6_prefill_bound::{PROMPT_GEN_RATIO_MILD, effective_prompt_tps};
#[cfg(test)]
use super::rule_names;

/// Occupancy ceiling: R1 does not fire above this. Server is not starved.
const OCCUPANCY_CEILING_PCT: f64 = 0.75;

/// Occupancy fallback threshold for unknown GPUs (no physics available).
const OCCUPANCY_FALLBACK_PCT: f64 = 0.25;

/// Config-relative efficiency below this means server is underperforming its config.
const CONFIG_EFFICIENCY_STARVATION_PCT: f64 = 60.0;

/// Waiting requests below this means no backlog pressure.
const UNDER_BATCHING_WAITING_LT: f64 = 2.0;

/// KV usage above this triggers a "monitor KV" note in R1's fix output.
/// 75% sits below R5's 80% safe-to-scale gate and well below R2's 88% threshold.
pub(super) const KV_MONITOR_WARNING_PCT: f64 = 75.0;

#[derive(Debug, Clone, PartialEq)]
pub struct UnderBatchingDetail {
    pub running: f64,
    pub waiting: f64,
    pub max_num_seqs: Option<u32>,
    pub effective_max: f64,
    pub occupancy_pct: f64,
    pub efficiency_pct: Option<f64>,
    pub config_relative_efficiency_pct: Option<f64>,
    pub known_gpu: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Rule1Outcome {
    Fired(UnderBatchingDetail),
    NotFired,
}

/// Inputs for R1 evaluation (per-window and single-window paths).
#[derive(Debug, Clone, Copy)]
pub struct R1EvalInput<'a> {
    pub snapshot: &'a RawSnapshot,
    pub config_max_num_seqs: Option<u32>,
    pub efficiency_pct: Option<f64>,
    pub config_relative_efficiency_pct: Option<f64>,
    pub prompt_tokens_per_sec: Option<f64>,
    pub generation_tokens_per_sec: Option<f64>,
    pub prefix_cache_hit_rate: Option<f64>,
    pub ridge_batch_size: Option<f64>,
}

pub(super) fn rule1_under_batching_with_efficiency(input: R1EvalInput<'_>) -> Rule1Outcome {
    let R1EvalInput {
        snapshot,
        config_max_num_seqs,
        efficiency_pct,
        config_relative_efficiency_pct,
        prompt_tokens_per_sec,
        generation_tokens_per_sec,
        prefix_cache_hit_rate,
        ridge_batch_size,
    } = input;
    let running = snapshot.vllm.num_requests_running;

    // 1. Hard abort - window duration required
    if !snapshot
        .vllm
        .window_duration_secs
        .is_some_and(|w| w.is_finite() && w > f64::EPSILON)
    {
        return Rule1Outcome::NotFired;
    }

    // 2. Hard abort - max_num_seqs required (scrape or config)
    let Some(max_n) = snapshot
        .vllm
        .max_num_seqs
        .or(config_max_num_seqs)
        .filter(|&n| n > 0)
    else {
        return Rule1Outcome::NotFired;
    };

    let effective_max = match ridge_batch_size.filter(|r| r.is_finite() && *r > 0.0) {
        Some(ridge) => f64::from(max_n).min(ridge),
        None => f64::from(max_n),
    };

    // 3. Hard abort - running required and > 0
    let Some(run) = running.filter(|v| v.is_finite() && *v > 0.0) else {
        return Rule1Outcome::NotFired;
    };

    // 4. Occupancy + backlog check
    let occupancy = run / effective_max;
    let Some(wait) = snapshot.vllm.num_requests_waiting.filter(|v| v.is_finite()) else {
        return Rule1Outcome::NotFired;
    };
    if wait >= UNDER_BATCHING_WAITING_LT {
        return Rule1Outcome::NotFired;
    }
    let efficiency_pct = efficiency_pct.filter(|e| e.is_finite());

    // Physics-level prefill gate: if prefill dominates, R6 handles it, not R1.
    if let (Some(p_tps), Some(g_tps)) = (
        prompt_tokens_per_sec.filter(|v| v.is_finite() && *v > 0.0),
        generation_tokens_per_sec.filter(|v| v.is_finite() && *v >= 0.0),
    ) {
        let p_tps = effective_prompt_tps(p_tps, prefix_cache_hit_rate);
        let ratio = if g_tps > 0.0 {
            p_tps / g_tps
        } else {
            f64::INFINITY
        };
        if ratio >= PROMPT_GEN_RATIO_MILD {
            return Rule1Outcome::NotFired;
        }
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
            return Rule1Outcome::NotFired;
        }
        if occupancy >= OCCUPANCY_CEILING_PCT {
            return Rule1Outcome::NotFired;
        }
    } else if occupancy >= OCCUPANCY_FALLBACK_PCT {
        // Unknown GPU: stricter occupancy-only fallback.
        return Rule1Outcome::NotFired;
    }

    Rule1Outcome::Fired(UnderBatchingDetail {
        running: run,
        waiting: wait,
        max_num_seqs: Some(max_n),
        effective_max,
        occupancy_pct: occupancy * 100.0,
        efficiency_pct,
        config_relative_efficiency_pct,
        known_gpu,
    })
}

#[cfg(test)]
pub fn r1_recommendation(input: R1EvalInput<'_>) -> Option<Recommendation> {
    let Rule1Outcome::Fired(d) = rule1_under_batching_with_efficiency(input) else {
        return None;
    };
    let confidence = if d.known_gpu { 0.8 } else { 0.5 };
    let kv_warning = input
        .snapshot
        .vllm
        .kv_cache_usage_perc
        .is_some_and(|kv| kv.is_finite() && kv >= KV_MONITOR_WARNING_PCT);
    Some(Recommendation {
        rule_name: rule_names::UNDER_BATCHING,
        layer: 4,
        impact: 4,
        confidence,
        action: "Batch more requests or increase client concurrency".to_string(),
        short_action: r1_short_action(d.running, d.effective_max),
        expected_impact: "Higher throughput, stable TPOT".to_string(),
        display_lines: format_under_batching_fired(&d, confidence, kv_warning),
    })
}

pub(super) fn r1_short_action(running: f64, effective_max: f64) -> String {
    let idle = (effective_max - running).max(0.0);
    format!("batch more requests or increase client concurrency ({idle:.0} slots idle)")
}

pub(super) fn format_under_batching_fired(
    d: &UnderBatchingDetail,
    confidence: f64,
    kv_warning: bool,
) -> Vec<String> {
    let Some(max_n) = d.max_num_seqs else {
        // Structurally unreachable: r1 hard-aborts without max_num_seqs.
        return Vec::new();
    };
    let max_str = max_n.to_string();
    let idle = (d.effective_max - d.running).max(0.0);
    let fix_line = if d.effective_max < max_n as f64 {
        format!(
            "      • Batch more requests or increase client concurrency ({idle:.0} slots idle before hardware degrades TPOT)"
        )
    } else {
        format!("      • Batch more requests or increase client concurrency ({idle:.0} slots idle)")
    };
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
        format!(
            "    Requests (avg when starved)   {:.0} running, {:.0} waiting  (max: {max_str})",
            d.running, d.waiting
        ),
        String::new(),
        "    Cause:".to_string(),
        "      Hardware capacity under-fed by client. Not enough requests arriving to keep the server busy."
            .to_string(),
        String::new(),
        "    Fix:".to_string(),
        fix_line,
    ];
    if kv_warning {
        lines.push("      • Monitor KV cache when scaling up.".to_string());
    }
    lines.push(String::new());
    lines.push("    Expected: Higher throughput, stable TPOT.".to_string());
    lines.push(format!("    Confidence: {confidence_str}"));
    if !d.known_gpu {
        lines.push(
            "    Note: GPU not in catalog. Diagnosis based on occupancy only (low confidence)."
                .to_string(),
        );
    }
    lines
}

pub(super) fn format_under_batching_window_issue(
    d: &UnderBatchingDetail,
    seen_pct: u32,
    confidence: f64,
    kv_warning: bool,
) -> Vec<String> {
    super::with_seen_pct(
        format_under_batching_fired(d, confidence, kv_warning),
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
        effective_max: details.iter().map(|d| d.effective_max).sum::<f64>() / n,
        occupancy_pct,
        efficiency_pct: super::mean_of_present(details.iter().filter_map(|d| d.efficiency_pct)),
        config_relative_efficiency_pct: super::mean_of_present(
            details
                .iter()
                .filter_map(|d| d.config_relative_efficiency_pct),
        ),
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
        }
    }

    #[derive(Default, Copy, Clone)]
    struct R1InputOpts {
        config_max_num_seqs: Option<u32>,
        efficiency_pct: Option<f64>,
        config_relative_efficiency_pct: Option<f64>,
        prompt_tokens_per_sec: Option<f64>,
        generation_tokens_per_sec: Option<f64>,
        prefix_cache_hit_rate: Option<f64>,
        ridge_batch_size: Option<f64>,
    }

    fn r1_input(snapshot: &RawSnapshot, opts: R1InputOpts) -> R1EvalInput<'_> {
        R1EvalInput {
            snapshot,
            config_max_num_seqs: opts.config_max_num_seqs,
            efficiency_pct: opts.efficiency_pct,
            config_relative_efficiency_pct: opts.config_relative_efficiency_pct,
            prompt_tokens_per_sec: opts.prompt_tokens_per_sec,
            generation_tokens_per_sec: opts.generation_tokens_per_sec,
            prefix_cache_hit_rate: opts.prefix_cache_hit_rate,
            ridge_batch_size: opts.ridge_batch_size,
        }
    }

    fn entry_fired_snap() -> RawSnapshot {
        snap(Some(5.0), Some(256), Some(0.0))
    }

    #[test]
    fn fires_when_occupancy_low() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(r1_input(&s, R1InputOpts::default())) {
            Rule1Outcome::Fired(d) => {
                assert!((d.occupancy_pct - (5.0 / 256.0 * 100.0)).abs() < 0.1);
            }
            Rule1Outcome::NotFired => panic!("expected fired"),
        }
    }

    #[test]
    fn fires_at_occupancy_below_threshold() {
        let s = snap(Some(63.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(r1_input(&s, R1InputOpts::default())) {
            Rule1Outcome::Fired(d) => {
                assert!(d.occupancy_pct < 25.0);
            }
            Rule1Outcome::NotFired => panic!("expected fired below 25% occupancy"),
        }
    }

    #[test]
    fn mutes_at_occupancy_threshold() {
        let s = snap(Some(64.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(r1_input(&s, R1InputOpts::default())),
            Rule1Outcome::NotFired
        ));
    }

    #[test]
    fn mutes_when_no_traffic() {
        let s = snap(Some(0.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(r1_input(&s, R1InputOpts::default())),
            Rule1Outcome::NotFired
        ));
    }

    #[test]
    fn mutes_when_backpressure_at_two() {
        let s = snap(Some(5.0), Some(256), Some(2.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(r1_input(&s, R1InputOpts::default())),
            Rule1Outcome::NotFired
        ));
    }

    #[test]
    fn fires_when_waiting_one_below_backpressure_gate() {
        let s = snap(Some(5.0), Some(256), Some(1.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(r1_input(&s, R1InputOpts::default())),
            Rule1Outcome::Fired(_)
        ));
    }

    #[test]
    fn mutes_when_max_num_seqs_missing() {
        let s = snap(Some(5.0), None, Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(r1_input(&s, R1InputOpts::default())),
            Rule1Outcome::NotFired
        ));
    }

    #[test]
    fn mutes_when_max_num_seqs_is_zero() {
        let s = snap(Some(5.0), Some(0), Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(r1_input(&s, R1InputOpts::default())),
            Rule1Outcome::NotFired
        ));
    }

    #[test]
    fn fires_when_config_max_provides_capacity_and_occupancy_low() {
        let s = snap(Some(4.0), None, Some(0.0));
        match rule1_under_batching_with_efficiency(r1_input(
            &s,
            R1InputOpts {
                config_max_num_seqs: Some(64),
                ..Default::default()
            },
        )) {
            Rule1Outcome::Fired(d) => {
                assert_eq!(d.max_num_seqs, Some(64));
                assert!((d.occupancy_pct - (4.0 / 64.0 * 100.0)).abs() < 0.1);
            }
            Rule1Outcome::NotFired => panic!("expected fired with config max 64"),
        }
    }

    #[test]
    fn mutes_at_occupancy_threshold_with_config_max_only() {
        let s = snap(Some(64.0), None, Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(r1_input(
                &s,
                R1InputOpts {
                    config_max_num_seqs: Some(64),
                    ..Default::default()
                },
            )),
            Rule1Outcome::NotFired
        ));
    }

    #[test]
    fn mutes_when_running_missing() {
        let s = snap(None, Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(r1_input(&s, R1InputOpts::default())),
            Rule1Outcome::NotFired
        ));
    }

    #[test]
    fn mutes_when_window_duration_missing() {
        let s = snap_with_gates(Some(5.0), Some(256), Some(0.0), None, None);
        assert!(matches!(
            rule1_under_batching_with_efficiency(r1_input(&s, R1InputOpts::default())),
            Rule1Outcome::NotFired
        ));
    }

    #[test]
    fn r1_recommendation_fires_without_baseline() {
        let s = entry_fired_snap();
        let r = r1_recommendation(r1_input(&s, R1InputOpts::default())).expect("fired");
        assert_eq!(r.rule_name, rule_names::UNDER_BATCHING);
        assert!((r.confidence - 0.5).abs() < 1e-9);
    }

    #[test]
    fn short_action_includes_batch_or_increase_concurrency() {
        let s = entry_fired_snap();
        let r = r1_recommendation(r1_input(&s, R1InputOpts::default())).expect("fired");
        assert_eq!(
            r.short_action,
            "batch more requests or increase client concurrency (251 slots idle)"
        );
    }

    #[test]
    fn r1_recommendation_adds_kv_warning_from_snapshot() {
        let mut s = entry_fired_snap();
        s.vllm.kv_cache_usage_perc = Some(75.0);
        let r = r1_recommendation(r1_input(&s, R1InputOpts::default())).expect("fired");
        let text = r.display_lines.join("\n");
        assert!(text.contains("Monitor KV cache when scaling up."));
    }

    #[test]
    fn fix_line_omits_kv_ceiling_even_when_known() {
        let s = entry_fired_snap();
        let r = r1_recommendation(r1_input(&s, R1InputOpts::default())).expect("fired");
        let text = r.display_lines.join("\n");
        assert!(text.contains(
            "      • Batch more requests or increase client concurrency (251 slots idle)"
        ));
        assert!(!text.contains("hardware limit"));
        assert!(!text.contains("KV ceiling"));
    }

    #[test]
    fn format_under_batching_fired_shows_requests_on_known_gpu_path() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(r1_input(
            &s,
            R1InputOpts {
                efficiency_pct: Some(25.0),
                config_relative_efficiency_pct: Some(15.0),
                ..Default::default()
            },
        )) {
            Rule1Outcome::Fired(d) => {
                assert!(d.known_gpu);
                assert_eq!(d.config_relative_efficiency_pct, Some(15.0));
                let text = format_under_batching_fired(&d, 0.8, false).join("\n");
                assert!(text.contains("Requests (avg when starved)"));
                assert!(text.contains("running"));
                assert!(text.contains("max: 256"));
                assert!(!text.contains("Config efficiency"));
            }
            Rule1Outcome::NotFired => panic!("expected fire via config efficiency path"),
        }
    }

    #[test]
    fn format_under_batching_fired_shows_note_on_unknown_gpu_path() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(r1_input(&s, R1InputOpts::default())) {
            Rule1Outcome::Fired(d) => {
                assert!(!d.known_gpu);
                let text = format_under_batching_fired(&d, 0.5, false).join("\n");
                assert!(text.contains("Requests (avg when starved)"));
                assert!(text.contains("low confidence"));
                assert!(!text.contains("Config efficiency"));
            }
            Rule1Outcome::NotFired => panic!("expected fire via occupancy fallback"),
        }
    }

    #[test]
    fn kv_warning_shown_at_75() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(r1_input(&s, R1InputOpts::default())) {
            Rule1Outcome::Fired(d) => {
                let text = format_under_batching_fired(&d, 0.5, true).join("\n");
                assert!(text.contains("Monitor KV cache when scaling up."));
            }
            Rule1Outcome::NotFired => panic!("expected fired"),
        }
    }

    #[test]
    fn kv_warning_absent_below_75() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(r1_input(&s, R1InputOpts::default())) {
            Rule1Outcome::Fired(d) => {
                let text = format_under_batching_fired(&d, 0.5, false).join("\n");
                assert!(!text.contains("Monitor KV cache when scaling up."));
            }
            Rule1Outcome::NotFired => panic!("expected fired"),
        }
    }

    #[test]
    fn known_gpu_fires_when_config_eff_low_and_occupancy_low() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(r1_input(
            &s,
            R1InputOpts {
                config_relative_efficiency_pct: Some(15.0),
                ..Default::default()
            },
        )) {
            Rule1Outcome::Fired(d) => {
                assert!(d.known_gpu);
                assert_eq!(d.config_relative_efficiency_pct, Some(15.0));
            }
            Rule1Outcome::NotFired => panic!("expected fired"),
        }
    }

    #[test]
    fn known_gpu_mutes_when_config_eff_high() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(r1_input(
                &s,
                R1InputOpts {
                    config_relative_efficiency_pct: Some(75.0),
                    ..Default::default()
                },
            )),
            Rule1Outcome::NotFired
        ));
    }

    #[test]
    fn known_gpu_mutes_when_occupancy_above_ceiling() {
        let s = snap(Some(200.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(r1_input(
                &s,
                R1InputOpts {
                    config_relative_efficiency_pct: Some(15.0),
                    ..Default::default()
                },
            )),
            Rule1Outcome::NotFired
        ));
    }

    #[test]
    fn unknown_gpu_fires_when_occupancy_below_fallback() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(r1_input(&s, R1InputOpts::default())) {
            Rule1Outcome::Fired(d) => {
                assert!(!d.known_gpu);
            }
            Rule1Outcome::NotFired => panic!("expected fired on unknown GPU fallback"),
        }
    }

    #[test]
    fn unknown_gpu_mutes_when_occupancy_above_fallback() {
        let s = snap(Some(70.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(r1_input(&s, R1InputOpts::default())),
            Rule1Outcome::NotFired
        ));
    }

    #[test]
    fn prefill_ratio_gate_suppresses_before_occupancy_check() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(r1_input(
            &s,
            R1InputOpts {
                config_relative_efficiency_pct: Some(15.0),
                prompt_tokens_per_sec: Some(600.0),
                generation_tokens_per_sec: Some(100.0),
                ..Default::default()
            },
        )) {
            Rule1Outcome::NotFired => {}
            Rule1Outcome::Fired(_) => panic!("expected suppressed by prefill gate"),
        }
    }

    #[test]
    fn prefill_gate_not_suppressed_when_prefix_cache_deflates_ratio() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        // Raw ratio 600/100 = 6.0x would suppress. Effective: 600*(1-0.90) = 60/100 = 0.6x.
        match rule1_under_batching_with_efficiency(r1_input(
            &s,
            R1InputOpts {
                config_relative_efficiency_pct: Some(15.0),
                prompt_tokens_per_sec: Some(600.0),
                generation_tokens_per_sec: Some(100.0),
                prefix_cache_hit_rate: Some(0.90),
                ..Default::default()
            },
        )) {
            Rule1Outcome::Fired(_) => {}
            Rule1Outcome::NotFired => panic!("prefix cache should prevent suppression"),
        }
    }

    #[test]
    fn prefill_gate_suppresses_on_pure_prefill_zero_gen() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(r1_input(
            &s,
            R1InputOpts {
                config_relative_efficiency_pct: Some(15.0),
                prompt_tokens_per_sec: Some(600.0),
                generation_tokens_per_sec: Some(0.0),
                ..Default::default()
            },
        )) {
            Rule1Outcome::NotFired => {}
            Rule1Outcome::Fired(_) => panic!("R1 should suppress on pure prefill"),
        }
    }

    #[test]
    fn ridge_cap_silences_at_compute_knee() {
        // 155 running, max_num_seqs=256, ridge=153. effective_max=153. occupancy=101%.
        let s = snap(Some(155.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(r1_input(
                &s,
                R1InputOpts {
                    config_relative_efficiency_pct: Some(15.0),
                    ridge_batch_size: Some(153.0),
                    ..Default::default()
                },
            )),
            Rule1Outcome::NotFired
        ));
    }

    #[test]
    fn without_ridge_155_of_256_fires() {
        let s = snap(Some(155.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching_with_efficiency(r1_input(
                &s,
                R1InputOpts {
                    config_relative_efficiency_pct: Some(15.0),
                    ..Default::default()
                },
            )),
            Rule1Outcome::Fired(_)
        ));
    }

    #[test]
    fn unknown_gpu_confidence_is_low() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        let r = r1_recommendation(r1_input(&s, R1InputOpts::default())).expect("fired");
        assert!((r.confidence - 0.5).abs() < 1e-9);
    }

    #[test]
    fn known_gpu_confidence_is_high() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        let r = r1_recommendation(r1_input(
            &s,
            R1InputOpts {
                config_relative_efficiency_pct: Some(15.0),
                ..Default::default()
            },
        ))
        .expect("fired");
        assert!((r.confidence - 0.8).abs() < 1e-9);
    }

    #[test]
    fn full_and_gate_fires_when_all_conditions_met() {
        // All four gates pass: config_eff=15% < 60%, occupancy=1.95% < 75%,
        // prefill=0.20 < 0.30, waiting=0 < 2.
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(r1_input(
            &s,
            R1InputOpts {
                config_relative_efficiency_pct: Some(15.0),
                prompt_tokens_per_sec: Some(400.0),
                generation_tokens_per_sec: Some(100.0),
                ..Default::default()
            },
        )) {
            Rule1Outcome::Fired(d) => {
                assert!(d.known_gpu);
                assert_eq!(d.config_relative_efficiency_pct, Some(15.0));
            }
            Rule1Outcome::NotFired => panic!("expected fired with all gates passing"),
        }
    }
}
