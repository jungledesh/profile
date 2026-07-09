use crate::collectors::{effective_tensor_parallel, window_is_evaluable};
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine::Report;
use crate::engine::baseline::{self, WeightDtypeSource};

use super::r1_under_batching::{
    KV_MONITOR_WARNING_PCT, R1EvalInput, Rule1Outcome, UnderBatchingDetail, aggregate_r1_detail,
    format_under_batching_window_issue, r1_short_action, rule1_under_batching_with_efficiency,
};
use super::r2_kv_cache_pressure::{
    KvAdmissionBacklogDetail, KvCachePressureDetail, KvFormatCtx, Rule2Outcome,
    aggregate_backlog_detail, aggregate_r2_detail, format_kv_admission_backlog_issue,
    format_kv_cache_window_issue, kv_pressure_confidence, r2_action, r2_backlog_short_action,
    r2_kv_pressure_short_action, rule2_kv_admission_backlog, rule2_kv_cache_pressure,
};
use super::r3_low_prefix_reuse::{
    LowPrefixReuseDetail, Rule3Outcome, aggregate_r3_detail, format_low_prefix_window_issue,
    rule3_low_prefix_reuse,
};
use super::r4_oom_risk::r4_recommendation;
use super::r5_concurrency_saturation::{
    ConcurrencySaturationDetail, aggregate_concurrency_saturation_detail,
    format_concurrency_saturation_window_issue, r5_action, r5_short_action,
    rule5_concurrency_saturation,
};
use super::r6_prefill_bound::{
    PrefillBoundDetail, PrefillBoundEvalInput, Rule6Outcome, aggregate_r6_detail,
    confidence as r6_confidence, evaluate as r6_evaluate, format_prefill_bound_window_issue,
    impact as r6_impact, prefill_fix_lines as r6_prefill_fix_lines, severity as r6_severity,
};
use super::r7_config_headroom::{
    ConfigHeadroomDetail, aggregate_r7_detail, format_config_headroom_window_issue,
    rule7_config_headroom,
};
use super::{IssueGroup, Recommendation, compute_kv_max_seqs, rule_is_significant, rule_names};

const SUPPRESSION_TABLE: &[(&str, &str)] = &[
    (rule_names::OOM_RISK, rule_names::KV_CACHE_PRESSURE),
    (rule_names::OOM_RISK, rule_names::KV_ADMISSION_BACKLOG),
];

struct WindowRuleEval {
    skipped: usize,
    n_eval: usize,
    r1_fired: usize,
    r1_kv_warning_count: usize,
    r1_details: Vec<UnderBatchingDetail>,
    r2_fired: usize,
    r2_details: Vec<KvCachePressureDetail>,
    r2_backlog_fired: usize,
    r2_backlog_details: Vec<KvAdmissionBacklogDetail>,
    r3_fired: usize,
    r3_details: Vec<LowPrefixReuseDetail>,
    r5_fired: usize,
    r5_details: Vec<ConcurrencySaturationDetail>,
    r6_fired: usize,
    r6_details: Vec<PrefillBoundDetail>,
    r7_fired: usize,
    r7_details: Vec<ConfigHeadroomDetail>,
    session_kv_peak: Option<f64>,
}

impl WindowRuleEval {
    fn r1_significant(&self) -> bool {
        rule_is_significant(self.r1_fired, self.n_eval)
    }

    fn r2_significant(&self) -> bool {
        rule_is_significant(self.r2_fired, self.n_eval)
    }

    fn r2_backlog_significant(&self) -> bool {
        rule_is_significant(self.r2_backlog_fired, self.n_eval)
    }

    fn r3_significant(&self) -> bool {
        rule_is_significant(self.r3_fired, self.n_eval)
    }

    fn r5_significant(&self) -> bool {
        rule_is_significant(self.r5_fired, self.n_eval)
    }

    fn r6_significant(&self) -> bool {
        rule_is_significant(self.r6_fired, self.n_eval)
    }

    fn r7_significant(&self) -> bool {
        rule_is_significant(self.r7_fired, self.n_eval)
    }
}

pub(crate) fn aggregate_prefix_hit_rate_for_windows(windows: &[RuntimeWindow]) -> Option<f64> {
    // Average hit rate across ALL evaluable windows, not just windows where r3
    // fired. Filtering by rule outcome biases the result low: high-performing
    // windows (hit_rate above threshold, r3 silent) would be excluded.
    let (sum, count) = windows
        .iter()
        .filter(|w| window_is_evaluable(&w.snapshot))
        .filter_map(|w| {
            w.snapshot
                .vllm
                .prefix_cache_hit_rate
                .filter(|r| r.is_finite())
        })
        .fold((0.0_f64, 0usize), |(s, c), v| (s + v, c + 1));
    (count > 0).then_some(sum / count as f64)
}

fn eval_window_rules(
    windows: &[RuntimeWindow],
    summary: &AnalysisInput<'_>,
    summary_efficiency_pct: Option<f64>,
) -> Option<WindowRuleEval> {
    if windows.is_empty() {
        return None;
    }

    let mut skipped = 0usize;
    let mut eval = WindowRuleEval {
        skipped: 0,
        n_eval: 0,
        r1_fired: 0,
        r1_kv_warning_count: 0,
        r1_details: Vec::new(),
        r2_fired: 0,
        r2_details: Vec::new(),
        r2_backlog_fired: 0,
        r2_backlog_details: Vec::new(),
        r3_fired: 0,
        r3_details: Vec::new(),
        r5_fired: 0,
        r5_details: Vec::new(),
        r6_fired: 0,
        r6_details: Vec::new(),
        r7_fired: 0,
        r7_details: Vec::new(),
        session_kv_peak: None,
    };

    for w in windows {
        if !window_is_evaluable(&w.snapshot) {
            skipped += 1;
            continue;
        }
        eval.n_eval += 1;

        let snap = &w.snapshot;
        if let Some(kv) = snap
            .vllm
            .kv_cache_peak_perc
            .or(snap.vllm.kv_cache_usage_perc)
            .filter(|v| v.is_finite())
        {
            eval.session_kv_peak = Some(eval.session_kv_peak.map_or(kv, |peak| peak.max(kv)));
        }

        // Per-window baseline: shared by R1 and R6.
        let win_input = AnalysisInput::new(summary.ctx, w);
        let win_baseline = baseline::compute(&win_input);

        match rule1_under_batching_with_efficiency(R1EvalInput {
            snapshot: snap,
            config_max_num_seqs: summary.ctx.config.max_num_seqs,
            efficiency_pct: summary_efficiency_pct,
            config_relative_efficiency_pct: win_baseline
                .as_ref()
                .and_then(|b| b.config_relative_efficiency_pct),
            prompt_tokens_per_sec: snap.vllm.prompt_tokens_per_sec,
            generation_tokens_per_sec: snap.vllm.generation_tokens_per_sec,
            prefix_cache_hit_rate: snap.vllm.prefix_cache_hit_rate,
            ridge_batch_size: win_baseline.as_ref().map(|b| b.ridge_batch_size),
        }) {
            Rule1Outcome::Fired(d) => {
                eval.r1_fired += 1;
                if snap
                    .vllm
                    .kv_cache_usage_perc
                    .is_some_and(|kv| kv.is_finite() && kv >= KV_MONITOR_WARNING_PCT)
                {
                    eval.r1_kv_warning_count += 1;
                }
                eval.r1_details.push(d);
            }
            Rule1Outcome::NotFired(_) => {}
        }
        match rule2_kv_cache_pressure(snap) {
            Rule2Outcome::Fired(d) => {
                eval.r2_fired += 1;
                eval.r2_details.push(d);
            }
            Rule2Outcome::NotFired => {}
        }
        if let Some(d) = rule2_kv_admission_backlog(snap) {
            eval.r2_backlog_fired += 1;
            eval.r2_backlog_details.push(d);
        }
        match rule3_low_prefix_reuse(snap) {
            Rule3Outcome::Fired(d) => {
                eval.r3_fired += 1;
                eval.r3_details.push(d);
            }
            Rule3Outcome::NotFired => {}
        }
        if let Some(d) = rule5_concurrency_saturation(
            snap,
            snap.vllm
                .kv_cache_peak_perc
                .or(snap.vllm.kv_cache_usage_perc),
            summary.ctx.config.max_num_seqs,
        ) {
            eval.r5_fired += 1;
            eval.r5_details.push(d);
        }

        match r6_evaluate(PrefillBoundEvalInput {
            prompt_tokens_per_sec: snap.vllm.prompt_tokens_per_sec,
            generation_tokens_per_sec: snap.vllm.generation_tokens_per_sec,
            decode_efficiency_pct: win_baseline.as_ref().and_then(|b| b.efficiency_pct),
            tpot_ms: snap.vllm.tpot_ms,
            tpot_floor_ms: win_baseline.as_ref().map(|b| b.tpot_floor_ms),
            prefix_cache_hit_rate: snap.vllm.prefix_cache_hit_rate,
            snapshot: snap,
            chunked_prefill_enabled: summary.ctx.config.enable_chunked_prefill,
        }) {
            Rule6Outcome::Fired(d) => {
                eval.r6_fired += 1;
                eval.r6_details.push(d);
            }
            Rule6Outcome::NotFired => {}
        }

        let ridge = win_baseline.as_ref().map(|b| b.ridge_batch_size);
        if let Some(d) = rule7_config_headroom(snap, summary.ctx.config.max_num_seqs, ridge) {
            eval.r7_fired += 1;
            eval.r7_details.push(d);
        }
    }

    eval.skipped = skipped;
    Some(eval)
}

// session_hit_rate: all-evaluable-windows average hit rate for display in r3 recommendation body.
// Caller must compute this from the full window slice, not from r3-fired windows only.
// Pass None on the single-window path (no session to average).
fn build_report_from_eval(
    eval: &WindowRuleEval,
    summary: AnalysisInput<'_>,
    session_hit_rate: Option<f64>,
    baseline: Option<baseline::PhysicsBaseline>,
) -> Report {
    if eval.n_eval == 0 {
        let kv_max_seqs = compute_kv_max_seqs(
            baseline.as_ref().and_then(|b| b.kv_headroom_gb),
            summary.ctx.config.max_model_len,
            &summary.ctx.model,
            summary.ctx.config.kv_cache_dtype.as_deref(),
            effective_tensor_parallel(
                summary.ctx.config.tensor_parallel_size,
                summary.window.snapshot.collected_gpu_count(),
            ),
        );
        return Report {
            baseline,
            groups: Vec::new(),
            suppressed_rules: Vec::new(),
            kv_max_seqs,
            n_eval: 0,
            skipped: 0,
        };
    }

    let summary_snap = &summary.window.snapshot;
    let max_model_len = summary.ctx.config.max_model_len;
    let prompt_tokens_mean = summary_snap.vllm.prompt_tokens_mean;
    let kv_headroom_gb = baseline.as_ref().and_then(|b| b.kv_headroom_gb);
    let nvcc_available = summary.ctx.nvcc_available;
    let kv_max_seqs: Option<u32> = compute_kv_max_seqs(
        kv_headroom_gb,
        max_model_len,
        &summary.ctx.model,
        summary.ctx.config.kv_cache_dtype.as_deref(),
        effective_tensor_parallel(
            summary.ctx.config.tensor_parallel_size,
            summary.window.snapshot.collected_gpu_count(),
        ),
    );
    let r2_significant = eval.r2_significant();
    let r2_backlog_significant = eval.r2_backlog_significant();

    let mut recs: Vec<Recommendation> = Vec::new();

    if eval.r1_significant() {
        let d = aggregate_r1_detail(&eval.r1_details);
        let confidence = if d.known_gpu { 0.8 } else { 0.5 };
        let kv_warning = rule_is_significant(eval.r1_kv_warning_count, eval.r1_fired);
        let display_lines = format_under_batching_window_issue(
            &d,
            pct(eval.r1_fired, eval.n_eval),
            confidence,
            kv_warning,
        );
        recs.push(Recommendation {
            rule_name: rule_names::UNDER_BATCHING,
            layer: 4,
            impact: 4,
            confidence,
            action: "Batch more requests or increase client concurrency".to_string(),
            short_action: r1_short_action(d.running, d.effective_max),
            expected_impact: "Higher throughput, stable TPOT".to_string(),
            display_lines,
        });
    }

    if r2_significant {
        let r2_agg = aggregate_r2_detail(&eval.r2_details);
        let conf = kv_pressure_confidence(eval.r2_fired, eval.n_eval);
        let display_lines = format_kv_cache_window_issue(
            &r2_agg,
            pct(eval.r2_fired, eval.n_eval),
            &KvFormatCtx {
                snapshot: summary_snap,
                max_model_len,
                kv_headroom_gb,
                kv_max_seqs,
                nvcc_available,
            },
            eval.r2_fired,
            eval.n_eval,
        );
        recs.push(Recommendation {
            rule_name: rule_names::KV_CACHE_PRESSURE,
            layer: 2,
            impact: 5,
            confidence: conf,
            action: r2_action(r2_agg.preemptions_active, kv_max_seqs, max_model_len),
            short_action: if r2_agg.preemptions_active {
                r2_kv_pressure_short_action().to_string()
            } else {
                r2_backlog_short_action().to_string()
            },
            expected_impact: "Reduced KV evictions and lower latency variance".to_string(),
            display_lines,
        });
    } else if r2_backlog_significant {
        let agg = aggregate_backlog_detail(&eval.r2_backlog_details);
        let display_lines = format_kv_admission_backlog_issue(
            &agg,
            pct(eval.r2_backlog_fired, eval.n_eval),
            &KvFormatCtx {
                snapshot: summary_snap,
                max_model_len,
                kv_headroom_gb,
                kv_max_seqs,
                nvcc_available,
            },
            eval.r2_backlog_fired,
            eval.n_eval,
        );
        recs.push(Recommendation {
            rule_name: rule_names::KV_ADMISSION_BACKLOG,
            layer: 2,
            impact: 5,
            confidence: kv_pressure_confidence(eval.r2_backlog_fired, eval.n_eval),
            action: r2_action(false, kv_max_seqs, max_model_len),
            short_action: r2_backlog_short_action().to_string(),
            expected_impact: "Wait queue drains, TTFT recovers.".to_string(),
            display_lines,
        });
    }

    if eval.r5_significant()
        && let Some(agg) =
            aggregate_concurrency_saturation_detail(&eval.r5_details, eval.session_kv_peak)
    {
        let display_lines = format_concurrency_saturation_window_issue(
            &agg,
            pct(eval.r5_fired, eval.n_eval),
            max_model_len,
            kv_max_seqs,
            summary_snap,
        );
        recs.push(Recommendation {
            rule_name: rule_names::CONCURRENCY_SATURATION,
            layer: 3,
            impact: 4,
            confidence: match (agg.ttft_ms.or(agg.ttft_p99_ms), agg.kv_cache_usage_perc) {
                (Some(_), Some(_)) => 0.9,
                _ => 0.6,
            },
            action: r5_action(&agg, kv_max_seqs, max_model_len, prompt_tokens_mean),
            short_action: r5_short_action(&agg, kv_max_seqs, max_model_len),
            expected_impact: "Queue drains, TTFT recovers.".to_string(),
            display_lines,
        });
    }

    if eval.r7_significant() {
        let d = aggregate_r7_detail(&eval.r7_details);
        // Both ridge and empirical KV present = higher confidence in the recommendation.
        let conf = if d.ridge_batch_size.is_some() && d.empirical_kv_seqs.is_some() {
            0.8
        } else {
            0.6
        };
        let display_lines =
            format_config_headroom_window_issue(&d, pct(eval.r7_fired, eval.n_eval), conf);
        recs.push(Recommendation {
            rule_name: rule_names::CONFIG_HEADROOM,
            layer: 6,
            impact: 3,
            confidence: conf,
            action: format!(
                "Raise --max-num-seqs from {} to {}",
                d.max_num_seqs, d.recommended_seqs
            ),
            short_action: format!("raise max_num_seqs to {}", d.recommended_seqs),
            expected_impact: "Higher concurrency ceiling, better hardware utilization.".to_string(),
            display_lines,
        });
    }

    if eval.r3_significant() {
        let d = aggregate_r3_detail(&eval.r3_details, summary_snap);
        let enable_prefix = summary_snap.vllm.cache_config.enable_prefix_caching;
        let (action, short_action, impact, confidence) = if d.hit_rate.is_none() {
            (
                "Enable --enable-prefix-caching".to_string(),
                "enable prefix caching".to_string(),
                3,
                0.95_f64,
            )
        } else {
            (
                "Move shared context to prompt prefix; standardize prompt templates".to_string(),
                "standardize prompts to share prefix context".to_string(),
                2,
                0.9_f64,
            )
        };
        recs.push(Recommendation {
            rule_name: rule_names::LOW_PREFIX_REUSE,
            layer: 5,
            impact,
            confidence,
            action,
            short_action,
            expected_impact: "Higher prefix cache hit rate and lower TTFT".to_string(),
            display_lines: format_low_prefix_window_issue(
                &d,
                pct(eval.r3_fired, eval.n_eval),
                enable_prefix,
                session_hit_rate,
            ),
        });
    }

    if eval.r6_significant() {
        let d = aggregate_r6_detail(&eval.r6_details);
        let sev = r6_severity(d.prompt_gen_ratio);
        let conf = r6_confidence(sev);
        let imp = r6_impact(sev);
        let display_lines = format_prefill_bound_window_issue(&d, pct(eval.r6_fired, eval.n_eval));
        let (_, action, short_action, expected_impact) = r6_prefill_fix_lines(&d, sev);
        recs.push(Recommendation {
            rule_name: rule_names::PREFILL_BOUND,
            layer: 5,
            impact: imp,
            confidence: conf,
            action,
            short_action,
            expected_impact,
            display_lines,
        });
    }

    if let Some(r4) = r4_recommendation(
        baseline.as_ref().and_then(|b| b.kv_headroom_gb),
        effective_tensor_parallel(
            summary.ctx.config.tensor_parallel_size,
            summary.window.snapshot.collected_gpu_count(),
        ),
        baseline.as_ref().map(|b| b.weight_gb),
        summary.ctx.gpu.vram_gb,
        summary.ctx.config.gpu_memory_utilization,
        baseline
            .as_ref()
            .map(|b| b.weight_dtype_source)
            .unwrap_or(WeightDtypeSource::Fallback),
    ) {
        recs.push(r4);
    }

    finalize_report_groups(recs, baseline, kv_max_seqs, eval.n_eval, eval.skipped)
}

pub(crate) fn finalize_report_groups(
    recs: Vec<Recommendation>,
    baseline: Option<baseline::PhysicsBaseline>,
    kv_max_seqs: Option<u32>,
    n_eval: usize,
    skipped: usize,
) -> Report {
    let mut suppressed_rules = Vec::new();
    let Some(min_layer) = recs.iter().map(|r| r.layer).min() else {
        return Report {
            baseline,
            groups: Vec::new(),
            suppressed_rules,
            kv_max_seqs,
            n_eval,
            skipped,
        };
    };

    let primary_name = recs
        .iter()
        .filter(|r| r.layer == min_layer)
        .max_by(|a, b| {
            let sa = a.impact as f64 * a.confidence;
            let sb = b.impact as f64 * b.confidence;
            sa.total_cmp(&sb)
        })
        .map(|r| r.rule_name)
        .unwrap_or("higher-priority rule");

    let mut recs: Vec<Recommendation> = recs
        .into_iter()
        .filter(|r| {
            if r.layer == min_layer {
                true
            } else {
                suppressed_rules.push((r.rule_name, primary_name));
                false
            }
        })
        .collect();

    let fired_names: Vec<&str> = recs.iter().map(|r| r.rule_name).collect();
    for (suppressor, suppressed) in SUPPRESSION_TABLE {
        if fired_names.contains(suppressor) {
            let before = recs.len();
            recs.retain(|r| r.rule_name != *suppressed);
            if recs.len() < before {
                suppressed_rules.push((suppressed, suppressor));
            }
        }
    }

    recs.sort_by(|a, b| {
        let sa = a.impact as f64 * a.confidence;
        let sb = b.impact as f64 * b.confidence;
        sb.total_cmp(&sa)
    });

    let groups: Vec<IssueGroup> = recs
        .into_iter()
        .map(|r| IssueGroup {
            primary: r,
            secondary: Vec::new(),
        })
        .collect();

    Report {
        baseline,
        groups,
        suppressed_rules,
        kv_max_seqs,
        n_eval,
        skipped,
    }
}

/// Multi-window rule evaluation, same significance gates as `format_diagnose_rules_for_windows`.
pub fn build_report_for_windows(windows: &[RuntimeWindow], summary: AnalysisInput<'_>) -> Report {
    let baseline = baseline::compute(&summary);
    let summary_efficiency_pct = baseline.as_ref().and_then(|b| b.efficiency_pct);
    let Some(eval) = eval_window_rules(windows, &summary, summary_efficiency_pct) else {
        return Report {
            baseline,
            groups: Vec::new(),
            suppressed_rules: Vec::new(),
            kv_max_seqs: None,
            n_eval: 0,
            skipped: windows.len(),
        };
    };
    let session_hit_rate = aggregate_prefix_hit_rate_for_windows(windows);
    build_report_from_eval(&eval, summary, session_hit_rate, baseline)
}

fn pct(fired: usize, total: usize) -> u32 {
    if total == 0 {
        return 0;
    }
    ((fired as f64 / total as f64) * 100.0).round() as u32
}
