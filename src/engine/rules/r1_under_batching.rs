use crate::collectors::RawSnapshot;

#[cfg(test)]
use super::Recommendation;
use super::effective_max_and_binder;
#[cfg(test)]
use super::rule_names;
use super::usable_kv_concurrency;

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

/// Alias for the shared binding-wall enum. R1 measures occupancy across all three
/// walls (config, ridge, memory) raw, with no safety margin: margining a measurement
/// fabricates. R5/R7 reuse the same enum for their margined recommendations.
pub(super) type R1BindingWall = super::BindingWall;

#[derive(Debug, Clone, PartialEq)]
pub struct UnderBatchingDetail {
    pub running: f64,
    pub waiting: f64,
    pub max_num_seqs: Option<u32>,
    pub effective_max: f64,
    pub binding_wall: R1BindingWall,
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
    pub ridge_batch_size: Option<f64>,
}

pub(super) fn rule1_under_batching_with_efficiency(input: R1EvalInput<'_>) -> Rule1Outcome {
    let R1EvalInput {
        snapshot,
        config_max_num_seqs,
        efficiency_pct,
        config_relative_efficiency_pct,
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

    let (effective_max, binding_wall) =
        effective_max_and_binder(max_n, ridge_batch_size, usable_kv_concurrency(snapshot));

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
        binding_wall,
        occupancy_pct: occupancy * 100.0,
        efficiency_pct,
        config_relative_efficiency_pct,
        known_gpu,
    })
}

fn r1_fix_lines(
    idle: f64,
    binding_wall: R1BindingWall,
    max_model_len: Option<u32>,
    prompt_mean: Option<f64>,
    generation_mean: Option<f64>,
) -> Vec<String> {
    match binding_wall {
        R1BindingWall::Ridge => vec![format!(
            "      • Batch more requests or increase client concurrency ({idle:.0} slots idle before hardware degrades TPOT)"
        )],
        R1BindingWall::Config => vec![format!(
            "      • Batch more requests or increase client concurrency ({idle:.0} slots idle)"
        )],
        // Memory wall: full-context count is a floor, not an action cap. Label the
        // assumption; never put the idle count in the action line.
        R1BindingWall::Memory { cap } => {
            let mut lines =
                vec!["      • Batch more requests or increase client concurrency.".to_string()];
            let Some(m) = max_model_len.filter(|&m| m > 0) else {
                return lines;
            };
            let window = super::format_observed_context_tokens(f64::from(m));
            let mut sub = format!("        Fits {cap} at the full {window} window");
            if let Some(obs) =
                super::capacity_at_observed_request_sizes(cap, m, prompt_mean, generation_mean)
            {
                sub.push_str(&format!("; ~{obs} at observed request sizes (est)"));
            }
            sub.push('.');
            lines.push(sub);
            lines
        }
    }
}

/// Context for R1 Fix lines that need config / traffic means (memory-wall subline).
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct R1FormatCtx {
    pub max_model_len: Option<u32>,
    pub prompt_mean: Option<f64>,
    pub generation_mean: Option<f64>,
}

impl R1FormatCtx {
    pub(super) fn from_snapshot(snapshot: &RawSnapshot, max_model_len: Option<u32>) -> Self {
        let ok = |v: Option<f64>| v.filter(|x| x.is_finite() && *x >= 0.0);
        Self {
            max_model_len,
            prompt_mean: ok(snapshot.vllm.prompt_tokens_mean),
            generation_mean: ok(snapshot.vllm.generation_tokens_mean),
        }
    }
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
        display_lines: format_under_batching_fired(
            &d,
            confidence,
            kv_warning,
            &R1FormatCtx::default(),
        ),
        terminal: false,
    })
}

pub(super) fn format_under_batching_fired(
    d: &UnderBatchingDetail,
    confidence: f64,
    kv_warning: bool,
    fmt: &R1FormatCtx,
) -> Vec<String> {
    let Some(max_n) = d.max_num_seqs else {
        // Structurally unreachable: r1 hard-aborts without max_num_seqs.
        return Vec::new();
    };
    let max_str = max_n.to_string();
    let idle = (d.effective_max - d.running).max(0.0);
    let fix_lines = r1_fix_lines(
        idle,
        d.binding_wall,
        fmt.max_model_len,
        fmt.prompt_mean,
        fmt.generation_mean,
    );
    let confidence_str = super::confidence_label(confidence);

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
    ];
    lines.extend(fix_lines);
    if kv_warning {
        lines.push(super::KV_SCALE_CAUTION.to_string());
    }
    lines.push(String::new());
    lines.push(
        "    Expected: Higher throughput. TPOT stable until the GPU is fully fed, then it starts to rise."
            .to_string(),
    );
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
    fmt: &R1FormatCtx,
) -> Vec<String> {
    super::with_seen_pct(
        format_under_batching_fired(d, confidence, kv_warning, fmt),
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
    // Walls are knowledge, not samples: keep the tightest known (effective_max, binder)
    // pair. If effective_max is tied, take the harsher wall (memory > ridge > config).
    let tightest = details.iter().fold(&details[0], |best, d| {
        if d.effective_max < best.effective_max
            || (d.effective_max == best.effective_max && d.binding_wall > best.binding_wall)
        {
            d
        } else {
            best
        }
    });
    let effective_max = tightest.effective_max;
    let binding_wall = tightest.binding_wall;
    let occupancy_pct = if effective_max > 0.0 && effective_max.is_finite() {
        (running / effective_max) * 100.0
    } else {
        0.0
    };
    UnderBatchingDetail {
        running,
        waiting,
        max_num_seqs: details.first().and_then(|d| d.max_num_seqs),
        effective_max,
        binding_wall,
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
            host_memory: None,
        }
    }

    #[derive(Default, Copy, Clone)]
    struct R1InputOpts {
        config_max_num_seqs: Option<u32>,
        efficiency_pct: Option<f64>,
        config_relative_efficiency_pct: Option<f64>,
        ridge_batch_size: Option<f64>,
    }

    fn r1_input(snapshot: &RawSnapshot, opts: R1InputOpts) -> R1EvalInput<'_> {
        R1EvalInput {
            snapshot,
            config_max_num_seqs: opts.config_max_num_seqs,
            efficiency_pct: opts.efficiency_pct,
            config_relative_efficiency_pct: opts.config_relative_efficiency_pct,
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
    fn fix_line_includes_idle_slots() {
        let s = entry_fired_snap();
        let r = r1_recommendation(r1_input(&s, R1InputOpts::default())).expect("fired");
        let text = r.display_lines.join("\n");
        assert!(
            text.contains("Batch more requests or increase client concurrency (251 slots idle)")
        );
    }

    #[test]
    fn r1_recommendation_adds_kv_warning_from_snapshot() {
        let mut s = entry_fired_snap();
        s.vllm.kv_cache_usage_perc = Some(75.0);
        let r = r1_recommendation(r1_input(&s, R1InputOpts::default())).expect("fired");
        let text = r.display_lines.join("\n");
        assert!(text.contains("        Monitor KV cache when scaling up."));
        assert!(!text.contains("• Monitor"));
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
                let text =
                    format_under_batching_fired(&d, 0.8, false, &R1FormatCtx::default()).join("\n");
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
                let text =
                    format_under_batching_fired(&d, 0.5, false, &R1FormatCtx::default()).join("\n");
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
                let text =
                    format_under_batching_fired(&d, 0.5, true, &R1FormatCtx::default()).join("\n");
                assert!(text.contains("        Monitor KV cache when scaling up."));
                assert!(!text.contains("• Monitor"));
            }
            Rule1Outcome::NotFired => panic!("expected fired"),
        }
    }

    #[test]
    fn kv_warning_absent_below_75() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(r1_input(&s, R1InputOpts::default())) {
            Rule1Outcome::Fired(d) => {
                let text =
                    format_under_batching_fired(&d, 0.5, false, &R1FormatCtx::default()).join("\n");
                assert!(!text.contains("Monitor KV cache when scaling up."));
                assert!(!text.contains("• Monitor"));
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
    fn fires_on_prefill_shaped_under_batching_window() {
        // High prompt/gen ratio used to defer R1 when R6 fired. Gate is gone:
        // under-batching evidence alone must still fire so ME can suppress.
        let mut s = snap(Some(5.0), Some(256), Some(0.0));
        s.vllm.prompt_tokens_per_sec = Some(600.0);
        s.vllm.generation_tokens_per_sec = Some(100.0);
        match rule1_under_batching_with_efficiency(r1_input(
            &s,
            R1InputOpts {
                config_relative_efficiency_pct: Some(15.0),
                ..Default::default()
            },
        )) {
            Rule1Outcome::Fired(d) => assert!(d.known_gpu),
            Rule1Outcome::NotFired => panic!("expected R1 fire on prefill-shaped under-batching"),
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
    fn memory_binds_when_kv_capacity_below_ridge_and_max() {
        // kv_capacity 35 < ridge 153 < max 256; running 6 → idle 29; memory label.
        let mut s = snap(Some(6.0), Some(256), Some(0.0));
        s.vllm.cache_config.kv_cache_max_concurrency = Some(35.0);
        match rule1_under_batching_with_efficiency(r1_input(
            &s,
            R1InputOpts {
                config_relative_efficiency_pct: Some(15.0),
                ridge_batch_size: Some(153.0),
                ..Default::default()
            },
        )) {
            Rule1Outcome::Fired(d) => {
                assert!((d.effective_max - 35.0).abs() < 1e-9);
                assert_eq!(d.binding_wall, R1BindingWall::Memory { cap: 35 });
                let idle = (d.effective_max - d.running).max(0.0);
                assert!((idle - 29.0).abs() < 1e-9);
                // No max_model_len in fmt → direction-only (no floor subline).
                let text =
                    format_under_batching_fired(&d, 0.8, false, &R1FormatCtx::default()).join("\n");
                assert!(!text.contains("slots idle"));
                assert!(text.contains("Batch more requests or increase client concurrency."));
                assert!(!text.contains("worst-case"));
                assert!(!text.contains("degrades TPOT"));
                // With window + means: floor labeled + observed-sizes clause.
                let fmt = R1FormatCtx {
                    max_model_len: Some(8192),
                    prompt_mean: Some(2000.0),
                    generation_mean: Some(2096.0),
                };
                // pool = 35 * 8192; mean = 4096 → floor(70) = 70
                let text = format_under_batching_fired(&d, 0.8, false, &fmt).join("\n");
                assert!(text.contains("Fits 35 at the full 8.2k window"));
                assert!(text.contains("~70 at observed request sizes (est)"));
                assert!(!text.contains("slots idle"));
            }
            Rule1Outcome::NotFired => panic!("expected memory-bound under-batching"),
        }
    }

    #[test]
    fn contradicted_kv_cap_falls_to_ridge() {
        // Same as memory_binds, but peak running 40 beats cap 35 → usable None → ridge.
        let mut s = snap(Some(6.0), Some(256), Some(0.0));
        s.vllm.cache_config.kv_cache_max_concurrency = Some(35.0);
        s.vllm.num_requests_running_peak = Some(40.0);
        match rule1_under_batching_with_efficiency(r1_input(
            &s,
            R1InputOpts {
                config_relative_efficiency_pct: Some(15.0),
                ridge_batch_size: Some(153.0),
                ..Default::default()
            },
        )) {
            Rule1Outcome::Fired(d) => {
                assert!((d.effective_max - 153.0).abs() < 1e-9);
                assert_eq!(d.binding_wall, R1BindingWall::Ridge);
            }
            Rule1Outcome::NotFired => panic!("expected ridge-bound under-batching"),
        }
    }

    #[test]
    fn observed_absent_no_memory_claim() {
        let s = snap(Some(6.0), Some(256), Some(0.0));
        match rule1_under_batching_with_efficiency(r1_input(
            &s,
            R1InputOpts {
                config_relative_efficiency_pct: Some(15.0),
                ridge_batch_size: Some(153.0),
                ..Default::default()
            },
        )) {
            Rule1Outcome::Fired(d) => {
                assert!((d.effective_max - 153.0).abs() < 1e-9);
                assert_eq!(d.binding_wall, R1BindingWall::Ridge);
                let text =
                    format_under_batching_fired(&d, 0.8, false, &R1FormatCtx::default()).join("\n");
                assert!(text.contains("slots idle before hardware degrades TPOT"));
                assert!(!text.contains("worst-case"));
                assert!(!text.contains("memory fits"));
            }
            Rule1Outcome::NotFired => panic!("expected ridge-bound fire"),
        }
    }

    #[test]
    fn kv_capacity_float_floors_to_integer_slots() {
        let (eff, wall) = effective_max_and_binder(256, Some(153.0), Some(24.64));
        assert!((eff - 24.0).abs() < 1e-9);
        assert_eq!(wall, R1BindingWall::Memory { cap: 24 });
    }

    #[test]
    fn tie_prefers_memory_over_ridge_and_config() {
        let (eff, wall) = effective_max_and_binder(35, Some(35.0), Some(35.9));
        assert!((eff - 35.0).abs() < 1e-9);
        assert_eq!(wall, R1BindingWall::Memory { cap: 35 });
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
    fn aggregate_picks_tightest_wall_not_first_binder() {
        let ridge = UnderBatchingDetail {
            running: 10.0,
            waiting: 0.0,
            max_num_seqs: Some(256),
            effective_max: 153.0,
            binding_wall: R1BindingWall::Ridge,
            occupancy_pct: (10.0 / 153.0) * 100.0,
            efficiency_pct: Some(12.0),
            config_relative_efficiency_pct: Some(15.0),
            known_gpu: true,
        };
        let mem = UnderBatchingDetail {
            running: 8.0,
            waiting: 0.0,
            max_num_seqs: Some(256),
            effective_max: 35.0,
            binding_wall: R1BindingWall::Memory { cap: 35 },
            occupancy_pct: (8.0 / 35.0) * 100.0,
            efficiency_pct: Some(12.0),
            config_relative_efficiency_pct: Some(15.0),
            known_gpu: true,
        };
        let agg = aggregate_r1_detail(&[ridge.clone(), ridge, mem.clone(), mem]);
        assert!((agg.effective_max - 35.0).abs() < 1e-9);
        assert_eq!(agg.binding_wall, R1BindingWall::Memory { cap: 35 });
        let text =
            format_under_batching_fired(&agg, 0.8, false, &R1FormatCtx::default()).join("\n");
        assert!(!text.contains("slots idle"));
        assert!(text.contains("Batch more requests or increase client concurrency."));
        assert!(!text.contains("worst-case"));
        assert!(!text.contains("degrades TPOT"));
    }

    #[test]
    fn aggregate_all_ridge_keeps_ridge() {
        let d = UnderBatchingDetail {
            running: 10.0,
            waiting: 0.0,
            max_num_seqs: Some(256),
            effective_max: 153.0,
            binding_wall: R1BindingWall::Ridge,
            occupancy_pct: (10.0 / 153.0) * 100.0,
            efficiency_pct: Some(12.0),
            config_relative_efficiency_pct: Some(15.0),
            known_gpu: true,
        };
        let agg = aggregate_r1_detail(&[d.clone(), d]);
        assert!((agg.effective_max - 153.0).abs() < 1e-9);
        assert_eq!(agg.binding_wall, R1BindingWall::Ridge);
        let text =
            format_under_batching_fired(&agg, 0.8, false, &R1FormatCtx::default()).join("\n");
        assert!(text.contains("slots idle before hardware degrades TPOT"));
        assert!(!text.contains("worst-case"));
        assert!(!text.contains("memory fits"));
    }

    #[test]
    fn r1_fix_line_config_wall_omits_tpot_clause() {
        let d = UnderBatchingDetail {
            running: 5.0,
            waiting: 0.0,
            max_num_seqs: Some(256),
            effective_max: 256.0,
            binding_wall: R1BindingWall::Config,
            occupancy_pct: (5.0 / 256.0) * 100.0,
            efficiency_pct: Some(12.0),
            config_relative_efficiency_pct: Some(15.0),
            known_gpu: true,
        };
        let text = format_under_batching_fired(&d, 0.8, false, &R1FormatCtx::default()).join("\n");
        assert!(text.contains("(251 slots idle)"));
        assert!(!text.contains("degrades TPOT"));
        assert!(!text.contains("worst-case"));
    }

    #[test]
    fn r1_fix_line_ridge_wall_includes_tpot_clause() {
        let idle = 147.0;
        let ridge = r1_fix_lines(idle, R1BindingWall::Ridge, None, None, None).join("\n");
        assert!(ridge.contains("degrades TPOT"));
        assert!(!ridge.contains("worst-case"));
        let mem =
            r1_fix_lines(idle, R1BindingWall::Memory { cap: 35 }, None, None, None).join("\n");
        assert!(!mem.contains("degrades TPOT"));
        assert!(!mem.contains("worst-case"));
        assert!(!mem.contains("slots idle"));
        assert!(mem.contains("Batch more requests or increase client concurrency."));
        let cfg = r1_fix_lines(idle, R1BindingWall::Config, None, None, None).join("\n");
        assert!(!cfg.contains("degrades TPOT"));
        assert!(!cfg.contains("worst-case"));
        assert!(cfg.contains("slots idle"));
    }

    #[test]
    fn aggregate_mem_absent_no_memory_claim() {
        let d = UnderBatchingDetail {
            running: 6.0,
            waiting: 0.0,
            max_num_seqs: Some(256),
            effective_max: 153.0,
            binding_wall: R1BindingWall::Ridge,
            occupancy_pct: (6.0 / 153.0) * 100.0,
            efficiency_pct: None,
            config_relative_efficiency_pct: Some(15.0),
            known_gpu: true,
        };
        let agg = aggregate_r1_detail(&[d.clone(), d]);
        assert_eq!(agg.binding_wall, R1BindingWall::Ridge);
        let text =
            format_under_batching_fired(&agg, 0.8, false, &R1FormatCtx::default()).join("\n");
        assert!(!text.contains("memory fits"));
        assert!(text.contains("slots idle before hardware degrades TPOT"));
        assert!(!text.contains("worst-case"));
    }

    #[test]
    fn aggregate_occupancy_uses_run_effective_max() {
        let a = UnderBatchingDetail {
            running: 10.0,
            waiting: 0.0,
            max_num_seqs: Some(256),
            effective_max: 153.0,
            binding_wall: R1BindingWall::Ridge,
            occupancy_pct: (10.0 / 153.0) * 100.0,
            efficiency_pct: None,
            config_relative_efficiency_pct: Some(15.0),
            known_gpu: true,
        };
        let b = UnderBatchingDetail {
            running: 8.0,
            waiting: 0.0,
            max_num_seqs: Some(256),
            effective_max: 35.0,
            binding_wall: R1BindingWall::Memory { cap: 35 },
            occupancy_pct: (8.0 / 35.0) * 100.0,
            efficiency_pct: None,
            config_relative_efficiency_pct: Some(15.0),
            known_gpu: true,
        };
        let agg = aggregate_r1_detail(&[a, b]);
        let mean_running = (10.0 + 8.0) / 2.0;
        let expected = (mean_running / 35.0) * 100.0;
        assert!((agg.occupancy_pct - expected).abs() < 1e-9);
        // Must not be the mean of per-window occupancies.
        let mean_occ = ((10.0 / 153.0) * 100.0 + (8.0 / 35.0) * 100.0) / 2.0;
        assert!((agg.occupancy_pct - mean_occ).abs() > 1.0);
    }

    #[test]
    fn aggregate_equal_min_keeps_first_window() {
        let first = UnderBatchingDetail {
            running: 5.0,
            waiting: 0.0,
            max_num_seqs: Some(256),
            effective_max: 35.0,
            binding_wall: R1BindingWall::Memory { cap: 35 },
            occupancy_pct: (5.0 / 35.0) * 100.0,
            efficiency_pct: None,
            config_relative_efficiency_pct: Some(15.0),
            known_gpu: true,
        };
        let second = UnderBatchingDetail {
            running: 7.0,
            waiting: 0.0,
            max_num_seqs: Some(256),
            effective_max: 35.0,
            binding_wall: R1BindingWall::Memory { cap: 35 },
            occupancy_pct: (7.0 / 35.0) * 100.0,
            efficiency_pct: None,
            config_relative_efficiency_pct: Some(15.0),
            known_gpu: true,
        };
        let agg = aggregate_r1_detail(&[first, second]);
        assert!((agg.effective_max - 35.0).abs() < 1e-9);
        assert_eq!(agg.binding_wall, R1BindingWall::Memory { cap: 35 });
    }

    #[test]
    fn aggregate_equal_value_prefers_harsher_wall_either_order() {
        let ridge = UnderBatchingDetail {
            running: 5.0,
            waiting: 0.0,
            max_num_seqs: Some(256),
            effective_max: 35.0,
            binding_wall: R1BindingWall::Ridge,
            occupancy_pct: (5.0 / 35.0) * 100.0,
            efficiency_pct: None,
            config_relative_efficiency_pct: Some(15.0),
            known_gpu: true,
        };
        let mem = UnderBatchingDetail {
            running: 7.0,
            waiting: 0.0,
            max_num_seqs: Some(256),
            effective_max: 35.0,
            binding_wall: R1BindingWall::Memory { cap: 35 },
            occupancy_pct: (7.0 / 35.0) * 100.0,
            efficiency_pct: None,
            config_relative_efficiency_pct: Some(15.0),
            known_gpu: true,
        };
        let forward = aggregate_r1_detail(&[ridge.clone(), mem.clone()]);
        assert!((forward.effective_max - 35.0).abs() < 1e-9);
        assert_eq!(forward.binding_wall, R1BindingWall::Memory { cap: 35 });
        let reversed = aggregate_r1_detail(&[mem, ridge]);
        assert!((reversed.effective_max - 35.0).abs() < 1e-9);
        assert_eq!(reversed.binding_wall, R1BindingWall::Memory { cap: 35 });
    }

    #[test]
    fn full_and_gate_fires_when_all_conditions_met() {
        // Known-GPU path: config_eff=15% < 60%, occupancy=1.95% < 75%, waiting=0 < 2.
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
            Rule1Outcome::NotFired => panic!("expected fired with all gates passing"),
        }
    }

    #[test]
    fn memory_wall_journey_iter2_two_line_form() {
        // Journey iter-2: full-context 15 @ 18200; mean prompt+gen 9100 → ~30 (est).
        let d = UnderBatchingDetail {
            running: 10.0,
            waiting: 0.0,
            max_num_seqs: Some(175),
            effective_max: 15.0,
            binding_wall: R1BindingWall::Memory { cap: 15 },
            occupancy_pct: (10.0 / 15.0) * 100.0,
            efficiency_pct: Some(12.0),
            config_relative_efficiency_pct: Some(15.0),
            known_gpu: true,
        };
        let fmt = R1FormatCtx {
            max_model_len: Some(18200),
            prompt_mean: Some(5000.0),
            generation_mean: Some(4100.0),
        };
        let text = format_under_batching_fired(&d, 0.8, false, &fmt).join("\n");
        assert!(text.contains("Batch more requests or increase client concurrency."));
        assert!(
            text.contains("Fits 15 at the full 18.2k window; ~30 at observed request sizes (est).")
        );
        assert!(!text.contains("slots idle"));
    }

    #[test]
    fn memory_wall_degrades_without_observed_context() {
        let d = UnderBatchingDetail {
            running: 10.0,
            waiting: 0.0,
            max_num_seqs: Some(175),
            effective_max: 15.0,
            binding_wall: R1BindingWall::Memory { cap: 15 },
            occupancy_pct: (10.0 / 15.0) * 100.0,
            efficiency_pct: Some(12.0),
            config_relative_efficiency_pct: Some(15.0),
            known_gpu: true,
        };
        let fmt = R1FormatCtx {
            max_model_len: Some(18200),
            prompt_mean: None,
            generation_mean: None,
        };
        let text = format_under_batching_fired(&d, 0.8, false, &fmt).join("\n");
        assert!(text.contains("Fits 15 at the full 18.2k window."));
        assert!(!text.contains("observed request sizes"));
        assert!(!text.contains("slots idle"));
    }

    #[test]
    fn memory_wall_degrades_without_max_model_len() {
        let lines = r1_fix_lines(
            5.0,
            R1BindingWall::Memory { cap: 15 },
            None,
            Some(1.0),
            Some(1.0),
        );
        assert_eq!(
            lines,
            vec!["      • Batch more requests or increase client concurrency.".to_string()]
        );
    }
}
