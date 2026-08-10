use crate::collectors::{effective_tensor_parallel, window_is_evaluable, window_is_idle};
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine::Report;
use crate::engine::baseline::{self, WeightDtypeSource, effective_kv_cache_dtype};

use super::r1_under_batching::{
    KV_MONITOR_WARNING_PCT, R1EvalInput, R1FormatCtx, Rule1Outcome, UnderBatchingDetail,
    aggregate_r1_detail, format_under_batching_window_issue, rule1_under_batching_with_efficiency,
};
use super::r2_kv_cache_pressure::{
    KvAdmissionBacklogDetail, KvCachePressureDetail, KvFormatCtx, Rule2Outcome,
    aggregate_backlog_detail, aggregate_r2_detail, format_kv_admission_backlog_issue_with_terminal,
    format_kv_cache_window_issue, kv_pressure_confidence, model_is_hybrid, resolve_r2_kv_capacity,
    rule2_kv_admission_backlog, rule2_kv_cache_pressure,
};
use super::r3_low_prefix_reuse::{
    LowPrefixReuseDetail, Rule3Outcome, aggregate_r3_detail, format_low_prefix_window_issue,
    rule3_low_prefix_reuse,
};
use super::r4_oom_risk::{R4FloorEvidence, r4_recommendation_with_request_floor};
use super::r5_concurrency_saturation::{
    ConcurrencySaturationDetail, aggregate_concurrency_saturation_detail,
    format_concurrency_saturation_window_issue, r5_confidence, rule5_concurrency_saturation,
};
use super::r6_prefill_bound::{
    PrefillBoundDetail, PrefillBoundEvalInput, Rule6Outcome, TPOT_UNVERIFIED_CONFIDENCE_CAP,
    aggregate_r6_detail, confidence as r6_confidence, effective_prompt_tps,
    evaluate as r6_evaluate, format_prefill_bound_window_issue_with_terminal, impact as r6_impact,
    severity as r6_severity,
};
use super::r7_config_headroom::{
    ConfigHeadroomDetail, aggregate_r7_detail, format_config_headroom_window_issue, r7_confidence,
    rule7_config_headroom,
};
use super::{
    KvBoundSource, Recommendation, catalog_state_pages_mismatch, compute_kv_max_seqs_for_cache,
    kv_full_context_cap_for_r1, recommended_seqs, resolve_kv_bound, rule_is_significant,
    rule_names, usable_kv_concurrency,
};

const SUPPRESSION_TABLE: &[(&str, &str)] = &[
    (rule_names::OOM_RISK, rule_names::KV_CACHE_PRESSURE),
    (rule_names::OOM_RISK, rule_names::KV_ADMISSION_BACKLOG),
    // Must run before the min-layer filter: R6 (L5) would otherwise be dropped
    // when R1 (L4) is present, making this entry a no-op.
    (rule_names::PREFILL_BOUND, rule_names::UNDER_BATCHING),
];

struct WindowRuleEval {
    skipped_broken: usize,
    skipped_idle: usize,
    energy_skew_skipped: usize,
    gauge_missing: crate::engine::GaugeMissingCounts,
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
    // Run-level mean(running) across evaluable windows; feeds the empirical KV bound.
    sum_running: f64,
    count_running: usize,
    sum_waiting: f64,
    count_waiting: usize,
    // Run-level means for limiter evidence and prefill effective ratio.
    sum_tpot_ms: f64,
    count_tpot_ms: usize,
    sum_effective_ratio: f64,
    count_effective_ratio: usize,
}

impl WindowRuleEval {
    fn mean_running(&self) -> Option<f64> {
        (self.count_running > 0).then(|| self.sum_running / self.count_running as f64)
    }

    fn mean_waiting(&self) -> Option<f64> {
        (self.count_waiting > 0).then(|| self.sum_waiting / self.count_waiting as f64)
    }

    fn mean_tpot_ms(&self) -> Option<f64> {
        (self.count_tpot_ms > 0).then(|| self.sum_tpot_ms / self.count_tpot_ms as f64)
    }

    fn mean_effective_ratio(&self) -> Option<f64> {
        (self.count_effective_ratio > 0)
            .then(|| self.sum_effective_ratio / self.count_effective_ratio as f64)
    }
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

    let mut skipped_broken = 0usize;
    let mut skipped_idle = 0usize;
    let mut eval = WindowRuleEval {
        skipped_broken: 0,
        skipped_idle: 0,
        energy_skew_skipped: 0,
        gauge_missing: crate::engine::GaugeMissingCounts::default(),
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
        sum_running: 0.0,
        count_running: 0,
        sum_waiting: 0.0,
        count_waiting: 0,
        sum_tpot_ms: 0.0,
        count_tpot_ms: 0,
        sum_effective_ratio: 0.0,
        count_effective_ratio: 0,
    };

    for w in windows {
        if !window_is_evaluable(&w.snapshot) {
            skipped_broken += 1;
            continue;
        }
        if window_is_idle(&w.snapshot) {
            skipped_idle += 1;
            continue;
        }
        eval.n_eval += 1;

        let snap = &w.snapshot;
        if !crate::collectors::observations_aligned(snap) {
            eval.energy_skew_skipped += 1;
        }
        if snap
            .vllm
            .num_requests_waiting
            .filter(|v| v.is_finite())
            .is_none()
        {
            eval.gauge_missing.under_batching += 1;
            eval.gauge_missing.concurrency_saturation += 1;
        }
        let kv_present = snap
            .vllm
            .kv_cache_usage_perc
            .filter(|v| v.is_finite())
            .is_some()
            || snap
                .vllm
                .kv_cache_peak_perc
                .filter(|v| v.is_finite())
                .is_some();
        if !kv_present {
            eval.gauge_missing.kv_cache_pressure += 1;
        }
        // r3: hit-rate gauge required unless prefix caching is explicitly off (that path fires).
        let prefix_off = snap.vllm.cache_config.enable_prefix_caching == Some(false);
        if !prefix_off
            && snap
                .vllm
                .prefix_cache_hit_rate
                .filter(|x| x.is_finite())
                .is_none()
        {
            eval.gauge_missing.low_prefix_reuse += 1;
        }

        if let Some(kv) = snap
            .vllm
            .kv_cache_peak_perc
            .or(snap.vllm.kv_cache_usage_perc)
            .filter(|v| v.is_finite())
        {
            eval.session_kv_peak = Some(eval.session_kv_peak.map_or(kv, |peak| peak.max(kv)));
        }

        if let Some(run) = snap
            .vllm
            .num_requests_running
            .filter(|v| v.is_finite() && *v >= 0.0)
        {
            eval.sum_running += run;
            eval.count_running += 1;
        }
        if let Some(wait) = snap
            .vllm
            .num_requests_waiting
            .filter(|v| v.is_finite() && *v >= 0.0)
        {
            eval.sum_waiting += wait;
            eval.count_waiting += 1;
        }
        if let Some(tpot) = snap.vllm.tpot_ms.filter(|v| v.is_finite() && *v > 0.0) {
            eval.sum_tpot_ms += tpot;
            eval.count_tpot_ms += 1;
        }
        if let (Some(prompt), Some(gen_tps)) = (
            snap.vllm
                .prompt_tokens_per_sec
                .filter(|v| v.is_finite() && *v >= 0.0),
            snap.vllm
                .generation_tokens_per_sec
                .filter(|v| v.is_finite() && *v > 0.0),
        ) {
            let eff_prompt = effective_prompt_tps(prompt, snap.vllm.prefix_cache_hit_rate);
            let ratio = eff_prompt / gen_tps;
            if ratio.is_finite() {
                eval.sum_effective_ratio += ratio;
                eval.count_effective_ratio += 1;
            }
        }

        // Per-window baseline: shared by R1 and R6.
        let win_input = AnalysisInput::new(summary.ctx, w);
        let win_baseline = baseline::compute(&win_input);

        let r6_outcome = r6_evaluate(PrefillBoundEvalInput {
            prompt_tokens_per_sec: snap.vllm.prompt_tokens_per_sec,
            generation_tokens_per_sec: snap.vllm.generation_tokens_per_sec,
            decode_efficiency_pct: win_baseline.as_ref().and_then(|b| b.efficiency_pct),
            tpot_ms: snap.vllm.tpot_ms,
            tpot_floor_ms: win_baseline.as_ref().map(|b| b.tpot_floor_ms),
            prefix_cache_hit_rate: snap.vllm.prefix_cache_hit_rate,
            snapshot: snap,
            chunked_prefill_enabled: summary.ctx.config.enable_chunked_prefill,
            ridge_batch_size: win_baseline.as_ref().map(|b| b.ridge_batch_size),
            max_num_batched_tokens: snap
                .vllm
                .max_num_batched_tokens
                .or(summary.ctx.config.max_num_batched_tokens),
        });
        match r6_outcome {
            Rule6Outcome::Fired(d) => {
                eval.r6_fired += 1;
                eval.r6_details.push(d);
            }
            Rule6Outcome::NotFired => {}
        }

        match rule1_under_batching_with_efficiency(R1EvalInput {
            snapshot: snap,
            config_max_num_seqs: summary.ctx.config.max_num_seqs,
            efficiency_pct: summary_efficiency_pct,
            config_relative_efficiency_pct: win_baseline
                .as_ref()
                .and_then(|b| b.config_relative_efficiency_pct),
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
            Rule1Outcome::NotFired => {}
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

        let ridge = win_baseline.as_ref().map(|b| b.ridge_batch_size);
        // Per-window derived KV bound from this window's baseline headroom. Firing
        // uses Observed else derived else this window's empirical; the displayed
        // recommendation is overridden at report time with the run-level resolution.
        let win_derived = compute_kv_max_seqs_for_cache(
            win_baseline.as_ref().and_then(|b| b.kv_headroom_gb),
            summary.ctx.config.max_model_len,
            &summary.ctx.model,
            effective_kv_cache_dtype(
                snap.vllm.cache_config.cache_dtype.as_deref(),
                summary.ctx.config.kv_cache_dtype.as_deref(),
            ),
            effective_tensor_parallel(
                summary.ctx.config.tensor_parallel_size,
                snap.collected_gpu_count(),
            ),
            &snap.vllm.cache_config,
        )
        .max_seqs;
        if let Some(d) = rule7_config_headroom(
            snap,
            summary.ctx.config.max_num_seqs,
            ridge,
            win_derived,
            model_is_hybrid(&summary.ctx.model),
        ) {
            eval.r7_fired += 1;
            eval.r7_details.push(d);
        }
    }

    eval.skipped_broken = skipped_broken;
    eval.skipped_idle = skipped_idle;
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
    let summary_snap = &summary.window.snapshot;
    let tp = effective_tensor_parallel(
        summary.ctx.config.tensor_parallel_size,
        summary_snap.collected_gpu_count(),
    );
    let kv_cache_dtype = effective_kv_cache_dtype(
        summary_snap.vllm.cache_config.cache_dtype.as_deref(),
        summary.ctx.config.kv_cache_dtype.as_deref(),
    );

    if eval.n_eval == 0 {
        let derived = compute_kv_max_seqs_for_cache(
            baseline.as_ref().and_then(|b| b.kv_headroom_gb),
            summary.ctx.config.max_model_len,
            &summary.ctx.model,
            kv_cache_dtype,
            tp,
            &summary_snap.vllm.cache_config,
        );
        return Report {
            baseline,
            recommendations: Vec::new(),
            suppressed_rules: Vec::new(),
            suppressed_recs: Vec::new(),
            kv_max_seqs: derived.max_seqs,
            catalog_state_mismatch: catalog_state_pages_mismatch(
                &summary_snap.vllm.cache_config,
                summary.ctx.config.max_model_len,
                &summary.ctx.model,
            ),
            n_eval: 0,
            skipped_broken: eval.skipped_broken,
            skipped_idle: eval.skipped_idle,
            energy_skew_skipped: eval.energy_skew_skipped,
            gauge_missing: eval.gauge_missing.clone(),
            limiter_evidence: None,
        };
    }

    let max_model_len = summary.ctx.config.max_model_len;
    let kv_headroom_gb = baseline.as_ref().and_then(|b| b.kv_headroom_gb);
    let fp8_compiler_available = summary.ctx.fp8_compiler_available;
    // Derived ceiling for R5 / verbose / Report.kv_max_seqs. R2 prefers observed.
    let derived_capacity = compute_kv_max_seqs_for_cache(
        kv_headroom_gb,
        max_model_len,
        &summary.ctx.model,
        kv_cache_dtype,
        tp,
        &summary_snap.vllm.cache_config,
    );
    let kv_max_seqs = derived_capacity.max_seqs;
    let (r2_kv_max_seqs, r2_capacity_label) = resolve_r2_kv_capacity(
        usable_kv_concurrency(summary_snap),
        kv_max_seqs,
        model_is_hybrid(&summary.ctx.model),
    );
    // Resolve the KV bound and margined recommendation once for the run: Observed,
    // else derived, else empirical (run-level mean(running) / peak(kv%)). R5 and R7
    // both read this so they never print two different recommended values.
    let ridge_run = baseline.as_ref().map(|b| b.ridge_batch_size);
    let (run_kv_bound, run_kv_source, run_kv_floor) = resolve_kv_bound(
        usable_kv_concurrency(summary_snap),
        kv_max_seqs,
        model_is_hybrid(&summary.ctx.model),
        eval.mean_running(),
        eval.session_kv_peak,
        summary_snap.vllm.num_requests_running_peak,
    );
    let run_rec = recommended_seqs(
        ridge_run,
        run_kv_bound,
        run_kv_source,
        run_kv_floor,
        summary.ctx.config.max_num_seqs,
        baseline.as_ref().map(|b| b.kv_cache_dtype_source),
    );
    let r2_significant = eval.r2_significant();
    let r2_backlog_significant = eval.r2_backlog_significant();
    let limiter_evidence = Some(crate::engine::limiter::LimiterEvidence {
        kv_cache_mean_perc: summary_snap
            .vllm
            .kv_cache_usage_perc
            .filter(|v| v.is_finite()),
        kv_cache_peak_perc: eval.session_kv_peak,
        mean_running: eval.mean_running(),
        mean_waiting: eval.mean_waiting(),
        ridge_batch_size: ridge_run.filter(|r| r.is_finite() && *r > 0.0),
        mean_tpot_ms: eval.mean_tpot_ms(),
        tpot_floor_ms: baseline.as_ref().map(|b| b.tpot_floor_ms),
        effective_prompt_decode_ratio: eval.mean_effective_ratio(),
        chunked_prefill_enabled: summary.ctx.config.enable_chunked_prefill,
        headroom_pct: baseline.as_ref().and_then(|b| b.headroom_pct),
        n_eval: eval.n_eval,
        ceiling_unknown_reason: baseline
            .is_none()
            .then(|| baseline::baseline_missing_reason(summary.ctx)),
    });

    let mut recs: Vec<Recommendation> = Vec::new();

    if eval.r1_significant() {
        let d = aggregate_r1_detail(&eval.r1_details);
        let confidence = if d.known_gpu { 0.8 } else { 0.5 };
        let kv_warning = rule_is_significant(eval.r1_kv_warning_count, eval.r1_fired);
        let r1_fmt = R1FormatCtx::from_snapshot(
            summary_snap,
            max_model_len,
            kv_full_context_cap_for_r1(summary_snap, kv_max_seqs),
        );
        let display_lines = format_under_batching_window_issue(
            &d,
            pct(eval.r1_fired, eval.n_eval),
            confidence,
            kv_warning,
            &r1_fmt,
        );
        recs.push(Recommendation {
            rule_name: rule_names::UNDER_BATCHING,
            layer: 4,
            impact: 4,
            confidence,
            display_lines,
            terminal: false,
        });
    }

    if r2_significant {
        let r2_agg = aggregate_r2_detail(&eval.r2_details);
        let conf = kv_pressure_confidence(eval.r2_fired, eval.n_eval);
        let (display_lines, terminal) = format_kv_cache_window_issue(
            &r2_agg,
            pct(eval.r2_fired, eval.n_eval),
            &KvFormatCtx {
                snapshot: summary_snap,
                max_model_len,
                kv_headroom_gb,
                kv_max_seqs: r2_kv_max_seqs,
                config_max_num_seqs: summary.ctx.config.max_num_seqs,
                capacity_label: r2_capacity_label,
                fp8_compiler_available,
                model: Some(&summary.ctx.model),
                tp,
                kv_cache_dtype,
            },
            eval.r2_fired,
            eval.n_eval,
        );
        recs.push(Recommendation {
            rule_name: rule_names::KV_CACHE_PRESSURE,
            layer: 2,
            impact: 5,
            confidence: conf,
            display_lines,
            terminal,
        });
    } else if r2_backlog_significant {
        let agg = aggregate_backlog_detail(&eval.r2_backlog_details);
        let (display_lines, terminal) = format_kv_admission_backlog_issue_with_terminal(
            &agg,
            pct(eval.r2_backlog_fired, eval.n_eval),
            &KvFormatCtx {
                snapshot: summary_snap,
                max_model_len,
                kv_headroom_gb,
                kv_max_seqs: r2_kv_max_seqs,
                config_max_num_seqs: summary.ctx.config.max_num_seqs,
                capacity_label: r2_capacity_label,
                fp8_compiler_available,
                model: Some(&summary.ctx.model),
                tp,
                kv_cache_dtype,
            },
            eval.r2_backlog_fired,
            eval.n_eval,
        );
        recs.push(Recommendation {
            rule_name: rule_names::KV_ADMISSION_BACKLOG,
            layer: 2,
            impact: 5,
            confidence: kv_pressure_confidence(eval.r2_backlog_fired, eval.n_eval),
            display_lines,
            terminal,
        });
    }

    if eval.r5_significant()
        && let Some(agg) =
            aggregate_concurrency_saturation_detail(&eval.r5_details, eval.session_kv_peak)
    {
        let (display_lines, terminal) = format_concurrency_saturation_window_issue(
            &agg,
            pct(eval.r5_fired, eval.n_eval),
            max_model_len,
            run_rec.as_ref(),
            summary_snap,
        );
        let empirical = run_rec.is_some_and(|r| r.empirical);
        recs.push(Recommendation {
            rule_name: rule_names::CONCURRENCY_SATURATION,
            layer: 3,
            impact: 4,
            confidence: r5_confidence(&agg, empirical),
            display_lines,
            terminal,
        });
    }

    if eval.r7_significant() {
        let d = aggregate_r7_detail(&eval.r7_details);
        // Run-level target can tighten below the per-window median that fired R7.
        // No headroom left → premise failed; drop the recommendation.
        let target = run_rec.map_or(d.recommended_seqs, |r| r.target);
        let still_headroom = target > d.max_num_seqs;
        if still_headroom {
            let display = ConfigHeadroomDetail {
                recommended_seqs: target,
                ridge_batch_size: ridge_run.filter(|r| r.is_finite() && *r > 0.0),
                ..d
            };
            let conf = r7_confidence(run_rec.as_ref());
            // A derived bound inherits its KV pricing's provenance. Unknown dtype =
            // priced on assumption = not a measured claim. Auto is vLLM-defined
            // semantics, not a guess. Observed is allocator-reported, independent
            // of our pricing.
            let memory_bound_resolved = match run_kv_source {
                Some(KvBoundSource::Observed) => true,
                Some(KvBoundSource::Derived | KvBoundSource::DerivedHybrid) => {
                    baseline.as_ref().is_some_and(|b| {
                        b.kv_cache_dtype_source != baseline::KvCacheDtypeSource::Unknown
                    })
                }
                None => false,
            };
            let display_lines = format_config_headroom_window_issue(
                &display,
                pct(eval.r7_fired, eval.n_eval),
                conf,
                run_rec.as_ref(),
                memory_bound_resolved,
            );
            recs.push(Recommendation {
                rule_name: rule_names::CONFIG_HEADROOM,
                layer: 6,
                impact: 3,
                confidence: conf,
                display_lines,
                terminal: false,
            });
        }
    }

    if eval.r3_significant() {
        let d = aggregate_r3_detail(&eval.r3_details, summary_snap);
        let enable_prefix = summary_snap.vllm.cache_config.enable_prefix_caching;
        let (impact, confidence) = if d.hit_rate.is_none() {
            (3, 0.95_f64)
        } else {
            (2, 0.9_f64)
        };
        recs.push(Recommendation {
            rule_name: rule_names::LOW_PREFIX_REUSE,
            layer: 5,
            impact,
            confidence,
            display_lines: format_low_prefix_window_issue(
                &d,
                pct(eval.r3_fired, eval.n_eval),
                enable_prefix,
                session_hit_rate,
            ),
            terminal: false,
        });
    }

    if eval.r6_significant() {
        let d = aggregate_r6_detail(&eval.r6_details);
        let sev = r6_severity(d.prompt_gen_ratio);
        let conf = if d.tpot_unverified {
            r6_confidence(sev).min(TPOT_UNVERIFIED_CONFIDENCE_CAP)
        } else {
            r6_confidence(sev)
        };
        let imp = r6_impact(sev);
        let (display_lines, terminal) =
            format_prefill_bound_window_issue_with_terminal(&d, pct(eval.r6_fired, eval.n_eval));
        recs.push(Recommendation {
            rule_name: rule_names::PREFILL_BOUND,
            layer: 5,
            impact: imp,
            confidence: conf,
            display_lines,
            terminal,
        });
    }

    // Always resolve KV pricing provenance; Unknown still prices the floor but
    // R4 names the guess and caps confidence (see r4_recommendation_with_request_floor).
    let (kv_bpp, kv_src) = match baseline.as_ref() {
        Some(b) => (b.kv_bytes_per_element, b.kv_cache_dtype_source),
        None => baseline::resolve_kv_cache_element(kv_cache_dtype),
    };
    let request_bytes = if tp.is_none_or(|value| value <= 1) {
        max_model_len.and_then(|len| baseline::bytes_per_seq(&summary.ctx.model, len, kv_bpp))
    } else {
        None
    };
    if let Some(r4) = r4_recommendation_with_request_floor(
        baseline.as_ref().and_then(|b| b.kv_headroom_gb),
        tp,
        baseline.as_ref().map(|b| b.weight_gb),
        summary.ctx.gpu.vram_gb,
        summary.ctx.config.gpu_memory_utilization,
        baseline
            .as_ref()
            .map(|b| b.weight_dtype_source)
            .unwrap_or(WeightDtypeSource::Fallback),
        R4FloorEvidence {
            request_bytes,
            kv_cache_dtype_source: kv_src,
        },
    ) {
        recs.push(r4);
    }

    finalize_report_groups(
        recs,
        baseline,
        ReportCapacityMetadata {
            kv_max_seqs,
            catalog_state_mismatch: catalog_state_pages_mismatch(
                &summary_snap.vllm.cache_config,
                max_model_len,
                &summary.ctx.model,
            ),
        },
        eval.n_eval,
        crate::engine::EvalSkipStats {
            skipped_broken: eval.skipped_broken,
            skipped_idle: eval.skipped_idle,
            energy_skew_skipped: eval.energy_skew_skipped,
            gauge_missing: eval.gauge_missing.clone(),
            limiter_evidence,
        },
    )
}

struct ReportCapacityMetadata {
    kv_max_seqs: Option<u32>,
    catalog_state_mismatch: Option<(u64, u64)>,
}

/// Suppressing live R2 evidence requires the strongest claim: weights alone overflow VRAM.
/// h in (-3, 0) can be buffer squeeze with weights fitting, and a running server disproves
/// "cannot fit". R4's own firing is unchanged, only the right to hide R2 is gated harder.
fn oom_weights_alone_overflow(kv_headroom_gb: Option<f64>) -> bool {
    kv_headroom_gb.is_some_and(|h| h.is_finite() && h < -baseline::ACTIVATION_KV_BUFFER_GB)
}

fn finalize_report_groups(
    recs: Vec<Recommendation>,
    baseline: Option<baseline::PhysicsBaseline>,
    capacity: ReportCapacityMetadata,
    n_eval: usize,
    skips: crate::engine::EvalSkipStats,
) -> Report {
    let ReportCapacityMetadata {
        kv_max_seqs,
        catalog_state_mismatch,
    } = capacity;
    let mut suppressed_rules = Vec::new();
    let mut suppressed_recs = Vec::new();

    // ME table BEFORE min-layer filter so cross-layer suppressions (R6→R1) land.
    // Same-layer rows (OOM→KV) are unchanged: both survive until ME, then KV drops.
    // Soft field: skip R6→R1 so min-layer makes R1 primary; R6/R3 → suppressed_recs
    // for remeasure reveal. Bound path keeps the ME row (and terminals on R6 primary).
    let mut recs = recs;
    let fired_names: Vec<&str> = recs.iter().map(|r| r.rule_name).collect();
    let oom_weights_alone_overflow =
        oom_weights_alone_overflow(baseline.as_ref().and_then(|b| b.kv_headroom_gb));
    let soft_field = skips
        .limiter_evidence
        .as_ref()
        .is_some_and(crate::engine::limiter::soft_field);
    for &(suppressor, suppressed) in SUPPRESSION_TABLE {
        if fired_names.contains(&suppressor) {
            if suppressor == rule_names::OOM_RISK && !oom_weights_alone_overflow {
                continue;
            }
            if soft_field
                && suppressor == rule_names::PREFILL_BOUND
                && suppressed == rule_names::UNDER_BATCHING
            {
                continue;
            }
            let (removed, kept): (Vec<_>, Vec<_>) =
                recs.into_iter().partition(|r| r.rule_name == suppressed);
            if !removed.is_empty() {
                suppressed_rules.push((suppressed, suppressor));
                suppressed_recs.extend(removed);
            }
            recs = kept;
        }
    }

    let Some(min_layer) = recs.iter().map(|r| r.layer).min() else {
        return Report {
            baseline,
            recommendations: Vec::new(),
            suppressed_rules,
            suppressed_recs,
            kv_max_seqs,
            catalog_state_mismatch,
            n_eval,
            skipped_broken: skips.skipped_broken,
            skipped_idle: skips.skipped_idle,
            energy_skew_skipped: skips.energy_skew_skipped,
            gauge_missing: skips.gauge_missing,
            limiter_evidence: skips.limiter_evidence,
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

    let mut primary_recs = Vec::new();
    for r in recs {
        if r.layer == min_layer {
            primary_recs.push(r);
        } else {
            suppressed_rules.push((r.rule_name, primary_name));
            suppressed_recs.push(r);
        }
    }

    primary_recs.sort_by(|a, b| {
        let sa = a.impact as f64 * a.confidence;
        let sb = b.impact as f64 * b.confidence;
        sb.total_cmp(&sa)
    });
    suppressed_recs.sort_by(|a, b| {
        let sa = a.impact as f64 * a.confidence;
        let sb = b.impact as f64 * b.confidence;
        sb.total_cmp(&sa)
    });

    Report {
        baseline,
        recommendations: primary_recs,
        suppressed_rules,
        suppressed_recs,
        kv_max_seqs,
        catalog_state_mismatch,
        n_eval,
        skipped_broken: skips.skipped_broken,
        skipped_idle: skips.skipped_idle,
        energy_skew_skipped: skips.energy_skew_skipped,
        gauge_missing: skips.gauge_missing,
        limiter_evidence: skips.limiter_evidence,
    }
}

/// Multi-window rule evaluation, same significance gates as `format_diagnose_rules_for_windows`.
pub fn build_report_for_windows(windows: &[RuntimeWindow], summary: AnalysisInput<'_>) -> Report {
    let baseline = baseline::compute(&summary);
    let summary_efficiency_pct = baseline.as_ref().and_then(|b| b.efficiency_pct);
    let Some(eval) = eval_window_rules(windows, &summary, summary_efficiency_pct) else {
        return Report {
            baseline,
            recommendations: Vec::new(),
            suppressed_rules: Vec::new(),
            suppressed_recs: Vec::new(),
            kv_max_seqs: None,
            catalog_state_mismatch: None,
            n_eval: 0,
            skipped_broken: windows.len(),
            skipped_idle: 0,
            energy_skew_skipped: 0,
            gauge_missing: Default::default(),
            limiter_evidence: None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::KvCacheDtypeSource;
    use crate::engine::baseline::{CeilingEstimate, PhysicsBaseline, WeightDtypeSource};

    fn stub_rec(rule_name: &'static str, layer: u8) -> Recommendation {
        Recommendation {
            rule_name,
            layer,
            impact: 5,
            confidence: 0.9,
            display_lines: vec![rule_name.to_string()],
            terminal: false,
        }
    }

    fn baseline_with_headroom(h: f64) -> PhysicsBaseline {
        PhysicsBaseline {
            decode: CeilingEstimate {
                lower: 1.0,
                expected: 1.0,
                upper: 1.0,
            },
            prefill: None,
            efficiency_pct: None,
            headroom_pct: None,
            weight_dtype_source: WeightDtypeSource::EnvVar,
            weight_gb: 70.0,
            weight_bytes_per_param: 2,
            kv_bytes_per_element: 2,
            kv_cache_dtype_source: KvCacheDtypeSource::Auto,
            kv_headroom_gb: Some(h),
            tpot_floor_ms: 1.0,
            prefill_latency_floor_ms: None,
            ridge_batch_size: 1.0,
            config_relative_efficiency_pct: None,
            cost: None,
        }
    }

    fn finalize_oom_kv(h: f64) -> Report {
        finalize_report_groups(
            vec![
                stub_rec(rule_names::OOM_RISK, 2),
                stub_rec(rule_names::KV_CACHE_PRESSURE, 2),
            ],
            Some(baseline_with_headroom(h)),
            ReportCapacityMetadata {
                kv_max_seqs: None,
                catalog_state_mismatch: None,
            },
            15,
            crate::engine::EvalSkipStats {
                skipped_broken: 0,
                skipped_idle: 0,
                energy_skew_skipped: 0,
                gauge_missing: Default::default(),
                limiter_evidence: None,
            },
        )
    }

    #[test]
    fn buffer_squeeze_oom_keeps_kv_pressure() {
        // h = -1.0 is inside the 3GB activation buffer: R4 may fire, R2 survives.
        let report = finalize_oom_kv(-1.0);
        let names: Vec<_> = report.recommendations.iter().map(|r| r.rule_name).collect();
        assert!(names.contains(&rule_names::OOM_RISK));
        assert!(names.contains(&rule_names::KV_CACHE_PRESSURE));
        assert!(
            report.suppressed_rules.is_empty(),
            "buffer squeeze must not suppress R2: {:?}",
            report.suppressed_rules
        );
    }

    #[test]
    fn weights_alone_overflow_suppresses_kv_pressure() {
        // h = -4.0 is past the 3GB buffer: weights alone overflow → R2 suppressed.
        let report = finalize_oom_kv(-4.0);
        assert!(
            report
                .recommendations
                .iter()
                .any(|r| r.rule_name == rule_names::OOM_RISK)
        );
        assert!(
            !report
                .recommendations
                .iter()
                .any(|r| r.rule_name == rule_names::KV_CACHE_PRESSURE)
        );
        assert!(
            report
                .suppressed_rules
                .iter()
                .any(|(s, by)| *s == rule_names::KV_CACHE_PRESSURE && *by == rule_names::OOM_RISK)
        );
    }

    #[test]
    fn oom_weights_alone_overflow_gate() {
        assert!(!oom_weights_alone_overflow(Some(-1.0)));
        assert!(!oom_weights_alone_overflow(Some(
            -baseline::ACTIVATION_KV_BUFFER_GB
        )));
        assert!(oom_weights_alone_overflow(Some(-4.0)));
        assert!(!oom_weights_alone_overflow(None));
    }
}
