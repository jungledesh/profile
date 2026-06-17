use std::time::SystemTime;

use crate::collectors::{window_is_evaluable, RawSnapshot, VllmRawMetrics};
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine::baseline::{self, CostSource, PhysicsBaseline, WeightDtypeSource};

mod r1_under_batching;
mod r2_kv_cache_pressure;
mod r3_low_prefix_reuse;
mod r4_oom_risk;
mod r5_concurrency_saturation;

pub use r1_under_batching::{
    r1_recommendation, r1_verbose_miss_line, rule1_under_batching, R1MissReport, Rule1Outcome,
    UnderBatchingDetail,
};
pub use r2_kv_cache_pressure::{
    r2_recommendation, rule2_kv_admission_backlog, rule2_kv_cache_pressure,
    KvAdmissionBacklogDetail, KvCachePressureDetail, Rule2Outcome,
};
pub use r3_low_prefix_reuse::{
    r3_recommendation, rule3_low_prefix_reuse, LowPrefixReuseDetail, Rule3Outcome,
};
pub use r4_oom_risk::{r4_advisory, r4_recommendation};
pub use r5_concurrency_saturation::{
    r5_recommendation, rule5_concurrency_saturation, ConcurrencySaturationDetail,
};

use r1_under_batching::{
    aggregate_r1_detail, format_under_batching_window_issue, r1_short_action,
    rule1_under_batching_with_efficiency,
};
use r2_kv_cache_pressure::{
    aggregate_backlog_detail, aggregate_r2_detail, format_kv_admission_backlog_issue,
    format_kv_cache_window_issue, kv_pressure_confidence, r2_action, r2_backlog_short_action,
    r2_kv_pressure_short_action, KvPressureFormatOpts,
};
#[cfg(test)]
use r3_low_prefix_reuse::format_low_prefix_hit_rate_fired;
use r3_low_prefix_reuse::{aggregate_r3_detail, format_low_prefix_window_issue};
use r5_concurrency_saturation::{
    aggregate_concurrency_saturation_detail, format_concurrency_saturation_window_issue, r5_action,
    r5_short_action,
};

pub(super) const MAX_OBSERVATION_SKEW_SECS: f64 = 1.0;
/// Enforces >= 6s temporal substance (3 windows × 2s).
pub(super) const ENGINE_MIN_PERSISTENT_WINDOWS: usize = 3;
/// Enforces >= 25% density floor across evaluable windows.
pub(super) const ENGINE_MIN_WINDOW_PCT: f64 = 0.25;

// TODO(r5): sampling cliff — sampling temperature is not in collected metrics; wire rule when available.

/// Returns ` (currently N)` when known, empty string when not.
/// Used by rules that surface --max-model-len in Fix bullets.
pub(super) fn model_len_suffix(max_model_len: Option<u32>) -> String {
    match max_model_len {
        Some(v) => format!(" (currently {v})"),
        None => String::new(),
    }
}

pub(super) fn compute_kv_max_seqs(
    kv_headroom_gb: Option<f64>,
    max_model_len: Option<u32>,
    model: &crate::context::ModelArch,
    kv_cache_dtype: Option<&str>,
) -> Option<u32> {
    use crate::engine::baseline::{kv_bytes_per_element, kv_max_concurrent_seqs};
    let headroom = kv_headroom_gb?;
    let max_len = max_model_len?;
    let num_layers = model.num_kv_layers.or(model.num_layers)?;
    let num_kv_heads = model.num_kv_heads?;
    let head_dim = model.head_dim?;
    let kv_bpp = kv_bytes_per_element(kv_cache_dtype, 2);
    kv_max_concurrent_seqs(
        headroom,
        max_len,
        num_layers,
        num_kv_heads,
        head_dim,
        kv_bpp,
    )
}

fn kv_ceiling_unknown_verbose_line(
    kv_max_seqs: Option<u32>,
    verbose_rules: bool,
) -> Option<String> {
    if verbose_rules && kv_max_seqs.is_none() {
        Some(
            "[i] KV max-num-seqs ceiling unavailable (missing baseline/model/config fields)."
                .to_string(),
        )
    } else {
        None
    }
}

/// True when a rule fired in enough evaluable windows to be statistically stable.
pub fn rule_is_significant(fired: usize, total_evaluable: usize) -> bool {
    if total_evaluable == 0 {
        return false;
    }
    let pct = fired as f64 / total_evaluable as f64;
    fired >= ENGINE_MIN_PERSISTENT_WINDOWS && pct >= ENGINE_MIN_WINDOW_PCT
}

#[derive(Debug, Clone, PartialEq)]
pub struct Recommendation {
    pub rule_name: &'static str,
    /// 1–5; 5 = highest impact
    pub impact: u8,
    /// 0.0–1.0
    pub confidence: f64,
    /// Prescriptive: what to change
    pub action: String,
    /// One-liner for closed-loop direction block
    pub short_action: String,
    pub expected_impact: String,
    /// Pre-formatted cause + recommendation lines for stdout
    pub display_lines: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct IssueGroup {
    pub primary: Recommendation,
    pub secondary: Vec<Recommendation>,
}

impl IssueGroup {
    pub fn score(&self) -> f64 {
        self.primary.impact as f64 * self.primary.confidence
    }
}

const NO_ISSUES_LINE: &str = "No issues detected in this snapshot.";
const R2_SUPPRESSED_BY_R4_VERBOSE_LINE: &str = "  ↳ KV pressure suppressed (symptom of the above)";

fn metrics_scrape_url(metrics_input: &str) -> String {
    let base = metrics_input.trim_end_matches('/');
    if base.ends_with("/metrics") {
        base.to_string()
    } else {
        format!("{base}/metrics")
    }
}

/// R2 core metric: KV cache usage gauge from `/metrics`.
fn r2_kv_cache_advisory(snapshot: &RawSnapshot, metrics_url: &str) -> Option<Vec<String>> {
    if snapshot
        .vllm
        .kv_cache_usage_perc
        .filter(|v| v.is_finite())
        .is_some()
    {
        return None;
    }
    if !snapshot
        .vllm
        .num_requests_running
        .is_some_and(|r| r.is_finite() && r > 0.0)
    {
        return None;
    }
    let url = metrics_scrape_url(metrics_url);
    Some(vec![format!(
        "[i] KV Cache Pressure: core metric unavailable. Run: curl {url} | grep gpu_cache_usage_perc"
    )])
}

fn rule_display_block(
    g: &IssueGroup,
    verbose_rules: bool,
    r2_suppressed_by_r4: bool,
) -> Vec<String> {
    let mut block = g.primary.display_lines.clone();
    if verbose_rules && r2_suppressed_by_r4 && g.primary.rule_name == "oom_risk" {
        block.push(R2_SUPPRESSED_BY_R4_VERBOSE_LINE.to_string());
    }
    block
}

fn compute_waste_per_hr(baseline: Option<&PhysicsBaseline>, tps: Option<f64>) -> Option<f64> {
    let b = baseline?;
    let eff = b.efficiency_pct.filter(|e| e.is_finite())?;
    let cost = b.cost.as_ref()?;
    if !matches!(
        cost.cost_source,
        CostSource::UserProvided | CostSource::Catalog
    ) {
        return None;
    }
    let cpm = cost.cost_per_million_tokens?;
    let tps = tps.filter(|v| v.is_finite() && *v > 0.0)?;
    let cost_per_hr = cpm * tps * 3600.0 / 1_000_000.0;
    if cost_per_hr <= 0.0 {
        return None;
    }
    let waste_fraction = (1.0 - eff / 100.0).max(0.0);
    let waste = cost_per_hr * waste_fraction;
    if !waste.is_finite() || waste <= 0.0 {
        return None;
    }
    Some(waste)
}

// COUPLING: strings must match Recommendation.rule_name values in each rule file.
pub(super) fn waste_label_suffix(rule_names: &[&str]) -> Option<&'static str> {
    match rule_names.len() {
        0 => None,
        1 => match rule_names[0] {
            "under_batching" => Some("wasted on idle compute"),
            "kv_cache_pressure" => Some("lost to memory thrashing"),
            "low_prefix_reuse" => Some("wasted on redundant prefill"),
            "concurrency_saturation" => Some("lost to scheduler queuing"),
            _ => Some("unclassified overhead"),
        },
        _ => Some("lost to compounding bottlenecks"),
    }
}

/// Appends per-issue waste line when efficiency and cost data are available.
pub(super) fn append_waste_line(
    lines: &mut Vec<String>,
    groups: &[IssueGroup],
    baseline: Option<&PhysicsBaseline>,
    tps: Option<f64>,
) {
    let rule_names: Vec<&str> = groups.iter().map(|g| g.primary.rule_name).collect();
    let Some(suffix) = waste_label_suffix(&rule_names) else {
        return;
    };
    let Some(waste_per_hr) = compute_waste_per_hr(baseline, tps) else {
        return;
    };
    if !lines.is_empty() && !lines.last().is_some_and(|l| l.is_empty()) {
        lines.push(String::new());
    }
    lines.push(format!("~${waste_per_hr:.2}/hr {suffix}"));
}

/// User-facing lines when no window met `window_is_evaluable` (shared by stdout and rule formatters).
pub fn no_evaluable_diagnose_lines(verbose: bool, windows: &[RuntimeWindow]) -> Vec<String> {
    let mut out = vec![
        "No qualifying load was detected during this run. Profile only diagnoses behavior under active traffic.".to_string(),
        "Run diagnose again while the server is handling requests (raise concurrency or wait for steady load).".to_string(),
    ];
    if verbose {
        let total = windows.len();
        if total == 0 {
            out.push("Note: No collection windows were recorded.".to_string());
        } else {
            let skipped = windows
                .iter()
                .filter(|w| !window_is_evaluable(&w.snapshot))
                .count();
            if skipped > 0 {
                out.push(format!(
                    "Note: {skipped} of {total} collected windows dropped. Telemetry failure. Diagnosis may be incomplete."
                ));
            }
        }
    }
    out
}

fn append_not_triggered_lines(
    out: &mut Vec<String>,
    names: &[&str],
    verbose_rules: bool,
    r1_context: Option<(&RawSnapshot, Option<u32>, Option<f64>)>,
) {
    if names.is_empty() {
        return;
    }
    if !out.is_empty() && !out.last().is_some_and(|l| l.is_empty()) {
        out.push(String::new());
    }
    for name in names {
        let line = if *name == "Under-batching" && verbose_rules {
            match r1_context {
                Some((snap, max_seqs, efficiency_pct)) => {
                    r1_verbose_miss_line(snap, max_seqs, efficiency_pct)
                }
                None => format!("{name}: not triggered"),
            }
        } else {
            format!("{name}: not triggered")
        };
        out.push(line);
    }
}

fn not_triggered_from_fired_names(
    fired_names: &std::collections::HashSet<&'static str>,
    r2_suppressed_by_r4: bool,
    r2_adv_present: bool,
    r4_adv_present: bool,
) -> Vec<&'static str> {
    let mut names = Vec::new();
    if !fired_names.contains("under_batching") {
        names.push("Under-batching");
    }
    if !fired_names.contains("kv_cache_pressure")
        && !fired_names.contains("kv_admission_backlog")
        && !r2_suppressed_by_r4
        && !r2_adv_present
    {
        names.push("KV cache pressure");
    }
    if !fired_names.contains("oom_risk") && !r4_adv_present {
        names.push("OOM risk");
    }
    if !fired_names.contains("concurrency_saturation") {
        names.push("Concurrency saturation");
    }
    if !fired_names.contains("low_prefix_reuse") {
        names.push("Low prefix reuse");
    }
    names
}

pub fn format_diagnose_rules(
    input: AnalysisInput<'_>,
    verbose_rules: bool,
    metrics_url: &str,
) -> Vec<String> {
    let snapshot = &input.window.snapshot;
    if !window_is_evaluable(snapshot) {
        return no_evaluable_diagnose_lines(verbose_rules, std::slice::from_ref(input.window));
    }

    let report = super::build_report(input);
    let kv_max_seqs = compute_kv_max_seqs(
        report.baseline.as_ref().and_then(|b| b.kv_headroom_gb),
        input.ctx.config.max_model_len,
        &input.ctx.model,
        input.ctx.config.kv_cache_dtype.as_deref(),
    );
    let any_issue = !report.groups.is_empty();
    let baseline_ref = report.baseline.as_ref();
    let tps = snapshot.vllm.generation_tokens_per_sec;

    let fired_names: std::collections::HashSet<&'static str> =
        report.groups.iter().map(|g| g.primary.rule_name).collect();

    let mut out = Vec::new();

    for g in &report.groups {
        append_display_block(
            &mut out,
            rule_display_block(g, verbose_rules, report.r2_suppressed_by_r4),
        );
    }

    append_waste_line(&mut out, &report.groups, baseline_ref, tps);

    let r2_adv = if !fired_names.contains("kv_cache_pressure") && !report.r2_suppressed_by_r4 {
        r2_kv_cache_advisory(snapshot, metrics_url)
    } else {
        None
    };
    let r4_adv = r4_advisory(
        report.baseline.as_ref().and_then(|b| b.kv_headroom_gb),
        input.ctx.gpu.vram_gb,
        report.baseline.as_ref().map(|b| b.weight_gb),
    );

    let any_advisory = r2_adv.is_some() || r4_adv.is_some();
    let r2_adv_present = r2_adv.is_some();
    let r4_adv_present = r4_adv.is_some();
    if let Some(lines) = r2_adv {
        append_display_block(&mut out, lines);
    }
    if let Some(lines) = r4_adv {
        append_display_block(&mut out, lines);
    }

    let not_fired = not_triggered_from_fired_names(
        &fired_names,
        report.r2_suppressed_by_r4,
        r2_adv_present,
        r4_adv_present,
    );
    if verbose_rules {
        append_not_triggered_lines(
            &mut out,
            &not_fired,
            verbose_rules,
            Some((
                snapshot,
                input.ctx.config.max_num_seqs,
                baseline_ref.and_then(|b| b.efficiency_pct),
            )),
        );
    }

    if !any_issue && !any_advisory && !verbose_rules {
        out.push(NO_ISSUES_LINE.to_string());
    }
    if let Some(line) = kv_ceiling_unknown_verbose_line(kv_max_seqs, verbose_rules) {
        append_display_block(&mut out, vec![line]);
    }

    trim_trailing_blank_lines(&mut out);
    out
}

struct WindowRuleEval {
    total: usize,
    skipped: usize,
    n_eval: usize,
    r1_fired: usize,
    r1_details: Vec<UnderBatchingDetail>,
    r2_fired: usize,
    r2_details: Vec<KvCachePressureDetail>,
    r2_backlog_fired: usize,
    r2_backlog_details: Vec<KvAdmissionBacklogDetail>,
    r3_fired: usize,
    r3_details: Vec<LowPrefixReuseDetail>,
    r5_fired: usize,
    r5_details: Vec<ConcurrencySaturationDetail>,
    session_kv_peak: Option<f64>,
    groups: Vec<IssueGroup>,
}

impl WindowRuleEval {
    fn no_fires(&self) -> bool {
        self.r1_fired + self.r2_fired + self.r2_backlog_fired + self.r3_fired + self.r5_fired == 0
    }

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
}

fn r3_display_args(snap: &VllmRawMetrics, d: &LowPrefixReuseDetail) -> (f64, Option<f64>) {
    let qps = snap
        .request_success_per_sec
        .filter(|x| x.is_finite())
        .unwrap_or(0.0);
    let prompt_mean = d
        .prompt_tokens_mean
        .or_else(|| snap.prompt_tokens_mean.filter(|x| x.is_finite()));
    (qps, prompt_mean)
}

pub(crate) fn aggregate_prefix_hit_rate_for_windows(windows: &[RuntimeWindow]) -> Option<f64> {
    // Average hit rate across ALL evaluable windows — not just windows where r3
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
) -> Option<WindowRuleEval> {
    if windows.is_empty() {
        return None;
    }
    let total = windows.len();
    let skipped = windows
        .iter()
        .filter(|w| !window_is_evaluable(&w.snapshot))
        .count();
    let evaluable: Vec<&RuntimeWindow> = windows
        .iter()
        .filter(|w| window_is_evaluable(&w.snapshot))
        .collect();
    let n_eval = evaluable.len();
    let summary_efficiency_pct = baseline::compute(summary).and_then(|b| b.efficiency_pct);
    let session_kv_peak = evaluable
        .iter()
        .filter_map(|w| {
            w.snapshot
                .vllm
                .kv_cache_peak_perc
                .or(w.snapshot.vllm.kv_cache_usage_perc)
                .filter(|v| v.is_finite())
        })
        .reduce(f64::max);

    let mut eval = WindowRuleEval {
        total,
        skipped,
        n_eval,
        r1_fired: 0,
        r1_details: Vec::new(),
        r2_fired: 0,
        r2_details: Vec::new(),
        r2_backlog_fired: 0,
        r2_backlog_details: Vec::new(),
        r3_fired: 0,
        r3_details: Vec::new(),
        r5_fired: 0,
        r5_details: Vec::new(),
        session_kv_peak,
        groups: Vec::new(),
    };

    for w in &evaluable {
        match rule1_under_batching_with_efficiency(
            &w.snapshot,
            summary.ctx.config.max_num_seqs,
            summary_efficiency_pct,
        ) {
            Rule1Outcome::Fired(d) => {
                eval.r1_fired += 1;
                eval.r1_details.push(d);
            }
            Rule1Outcome::NotFired(_) => {}
        }
        match rule2_kv_cache_pressure(&w.snapshot) {
            Rule2Outcome::Fired(d) => {
                eval.r2_fired += 1;
                eval.r2_details.push(d);
            }
            Rule2Outcome::NotFired => {}
        }
        if let Some(d) = rule2_kv_admission_backlog(&w.snapshot) {
            eval.r2_backlog_fired += 1;
            eval.r2_backlog_details.push(d);
        }
        match rule3_low_prefix_reuse(&w.snapshot) {
            Rule3Outcome::Fired(d) => {
                eval.r3_fired += 1;
                eval.r3_details.push(d);
            }
            Rule3Outcome::NotFired => {}
        }
        if let Some(d) = rule5_concurrency_saturation(
            &w.snapshot,
            w.snapshot
                .vllm
                .kv_cache_peak_perc
                .or(w.snapshot.vllm.kv_cache_usage_perc),
            summary.ctx.config.max_num_seqs,
        ) {
            eval.r5_fired += 1;
            eval.r5_details.push(d);
        }
    }
    Some(eval)
}

// session_hit_rate: all-evaluable-windows average hit rate for display in r3 recommendation body.
// Caller must compute this from the full window slice — not from r3-fired windows only.
// Pass None on the single-window path (no session to average).
fn build_report_from_eval(
    eval: &WindowRuleEval,
    summary: AnalysisInput<'_>,
    session_hit_rate: Option<f64>,
) -> super::Report {
    let baseline = baseline::compute(&summary);
    if eval.n_eval == 0 {
        return super::Report {
            baseline,
            groups: Vec::new(),
            r2_suppressed_by_r4: false,
        };
    }

    let summary_snap = &summary.window.snapshot;
    let max_model_len = summary.ctx.config.max_model_len;
    let kv_headroom_gb = baseline.as_ref().and_then(|b| b.kv_headroom_gb);
    let kv_max_seqs: Option<u32> = compute_kv_max_seqs(
        kv_headroom_gb,
        max_model_len,
        &summary.ctx.model,
        summary.ctx.config.kv_cache_dtype.as_deref(),
    );
    let r2_significant = eval.r2_significant();
    let r2_backlog_significant = eval.r2_backlog_significant();

    let mut recs: Vec<Recommendation> = Vec::new();

    if eval.r1_significant() {
        let d = aggregate_r1_detail(&eval.r1_details);
        let display_lines =
            format_under_batching_window_issue(&d, pct(eval.r1_fired, eval.n_eval), 0.8);
        recs.push(Recommendation {
            rule_name: "under_batching",
            impact: 4,
            confidence: 0.8,
            action: "Increase client concurrency".to_string(),
            short_action: r1_short_action(),
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
            summary_snap,
            eval.r2_fired,
            eval.n_eval,
            KvPressureFormatOpts {
                max_model_len,
                kv_headroom_gb,
                kv_max_seqs,
            },
        );
        recs.push(Recommendation {
            rule_name: "kv_cache_pressure",
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
            max_model_len,
            kv_headroom_gb,
            eval.r2_backlog_fired,
            eval.n_eval,
        );
        recs.push(Recommendation {
            rule_name: "kv_admission_backlog",
            impact: 5,
            confidence: kv_pressure_confidence(eval.r2_backlog_fired, eval.n_eval),
            action: r2_action(false, kv_max_seqs, max_model_len),
            short_action: r2_backlog_short_action().to_string(),
            expected_impact: "Wait queue drains, TTFT recovers.".to_string(),
            display_lines,
        });
    }

    if eval.r5_significant() && !r2_significant && !r2_backlog_significant {
        if let Some(agg) =
            aggregate_concurrency_saturation_detail(&eval.r5_details, eval.session_kv_peak)
        {
            let display_lines = format_concurrency_saturation_window_issue(
                &agg,
                pct(eval.r5_fired, eval.n_eval),
                max_model_len,
                kv_max_seqs,
            );
            recs.push(Recommendation {
                rule_name: "concurrency_saturation",
                impact: 4,
                confidence: match (agg.ttft_ms.or(agg.ttft_p99_ms), agg.kv_cache_usage_perc) {
                    (Some(_), Some(_)) => 0.9,
                    _ => 0.6,
                },
                action: r5_action(&agg, kv_max_seqs, max_model_len),
                short_action: r5_short_action(&agg, kv_max_seqs, max_model_len),
                expected_impact: "Queue drains, TTFT recovers.".to_string(),
                display_lines,
            });
        }
    }

    if eval.r3_significant() {
        let d = aggregate_r3_detail(&eval.r3_details, summary_snap);
        let enable_prefix = summary_snap.vllm.cache_config.enable_prefix_caching;
        let (qps, prompt_mean) = r3_display_args(&summary_snap.vllm, &d);
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
            rule_name: "low_prefix_reuse",
            impact,
            confidence,
            action,
            short_action,
            expected_impact: "Higher prefix cache hit rate and lower TTFT".to_string(),
            display_lines: format_low_prefix_window_issue(
                &d,
                pct(eval.r3_fired, eval.n_eval),
                enable_prefix,
                qps,
                prompt_mean,
                session_hit_rate,
            ),
        });
    }

    if let Some(r4) = r4_recommendation(
        baseline.as_ref().and_then(|b| b.kv_headroom_gb),
        summary.ctx.config.tensor_parallel_size,
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

    finalize_report_groups(recs, baseline)
}

fn finalize_report_groups(
    mut recs: Vec<Recommendation>,
    baseline: Option<baseline::PhysicsBaseline>,
) -> super::Report {
    let r2_present_before = recs.iter().any(|r| r.rule_name == "kv_cache_pressure");
    let r4_fired = recs.iter().any(|r| r.rule_name == "oom_risk");
    let r2_suppressed_by_r4 = r4_fired && r2_present_before;
    if r4_fired {
        recs.retain(|r| r.rule_name != "kv_cache_pressure");
    }
    recs.sort_by(|a, b| {
        let sa = a.impact as f64 * a.confidence;
        let sb = b.impact as f64 * b.confidence;
        sb.total_cmp(&sa)
    });
    let groups = recs
        .into_iter()
        .map(|r| IssueGroup {
            primary: r,
            secondary: Vec::new(),
        })
        .collect();
    super::Report {
        baseline,
        groups,
        r2_suppressed_by_r4,
    }
}

/// Multi-window rule evaluation — same significance gates as `format_diagnose_rules_for_windows`.
pub fn build_report_for_windows(
    windows: &[RuntimeWindow],
    summary: AnalysisInput<'_>,
) -> super::Report {
    let baseline = baseline::compute(&summary);
    let Some(eval) = eval_window_rules(windows, &summary) else {
        return super::Report {
            baseline,
            groups: Vec::new(),
            r2_suppressed_by_r4: false,
        };
    };
    build_report_from_eval(&eval, summary, None)
}

pub fn format_diagnose_rules_for_windows(
    windows: &[RuntimeWindow],
    summary: AnalysisInput<'_>,
    verbose_rules: bool,
    metrics_url: &str,
) -> Vec<String> {
    let Some(mut eval) = eval_window_rules(windows, &summary) else {
        return no_evaluable_diagnose_lines(verbose_rules, &[]);
    };
    let session_hit_rate = aggregate_prefix_hit_rate_for_windows(windows);

    if eval.n_eval > 0 {
        eval.groups = build_report_from_eval(&eval, summary, session_hit_rate).groups;
    }

    if eval.n_eval == 0 {
        return no_evaluable_diagnose_lines(verbose_rules, windows);
    }

    if eval.n_eval < ENGINE_MIN_PERSISTENT_WINDOWS {
        let mut out = vec![
            "[!] Insufficient Sustained Load".to_string(),
            String::new(),
            format!(
                "  Traffic detected but too brief for reliable diagnosis. \
                 Required: {} evaluable windows. Captured: {}{}.",
                ENGINE_MIN_PERSISTENT_WINDOWS,
                eval.n_eval,
                if eval.skipped > 0 {
                    format!(" ({} windows dropped)", eval.skipped)
                } else {
                    String::new()
                }
            ),
            String::new(),
            "  Fix:".to_string(),
            "    • Maintain steady traffic for the full diagnostic duration.".to_string(),
        ];
        trim_trailing_blank_lines(&mut out);
        return out;
    }

    let total = eval.total;
    let skipped = eval.skipped;
    let n_eval = eval.n_eval;
    let r1_fired = eval.r1_fired;
    let r2_fired = eval.r2_fired;
    let r2_backlog_fired = eval.r2_backlog_fired;
    let r3_fired = eval.r3_fired;
    let r5_fired = eval.r5_fired;
    let r1_details = &eval.r1_details;
    let r2_details = &eval.r2_details;
    let r2_backlog_details = &eval.r2_backlog_details;
    let r3_details = &eval.r3_details;
    let r5_details = &eval.r5_details;

    let summary_baseline = baseline::compute(&summary);
    let summary_snap = &summary.window.snapshot;
    let baseline_ref = summary_baseline.as_ref();
    let tps = summary_snap.vllm.generation_tokens_per_sec;

    if eval.no_fires() {
        let mut out = Vec::new();
        let r2_adv = r2_kv_cache_advisory(summary_snap, metrics_url);
        let r4_adv = r4_advisory(
            summary_baseline.as_ref().and_then(|b| b.kv_headroom_gb),
            summary.ctx.gpu.vram_gb,
            summary_baseline.as_ref().map(|b| b.weight_gb),
        );
        let any_advisory = r2_adv.is_some() || r4_adv.is_some();
        let r2_adv_present = r2_adv.is_some();
        let r4_adv_present = r4_adv.is_some();
        if let Some(lines) = r2_adv {
            out.extend(lines);
            out.push(String::new());
        }
        if let Some(lines) = r4_adv {
            out.extend(lines);
            out.push(String::new());
        }
        if verbose_rules {
            let not_fired = not_triggered_from_fired_names(
                &std::collections::HashSet::new(),
                false,
                r2_adv_present,
                r4_adv_present,
            );
            append_not_triggered_lines(
                &mut out,
                &not_fired,
                verbose_rules,
                Some((
                    summary_snap,
                    summary.ctx.config.max_num_seqs,
                    summary_baseline.as_ref().and_then(|b| b.efficiency_pct),
                )),
            );
        }
        if !any_advisory && !verbose_rules {
            out.push(NO_ISSUES_LINE.to_string());
        }
        if skipped > 0 {
            out.push(format!(
                "Note: {skipped} of {total} windows dropped. Telemetry failure. Diagnosis may be incomplete."
            ));
        }
        trim_trailing_blank_lines(&mut out);
        return out;
    }

    let r1_significant = rule_is_significant(r1_fired, n_eval);
    let r2_significant = rule_is_significant(r2_fired, n_eval);
    let r2_backlog_significant = rule_is_significant(r2_backlog_fired, n_eval);
    let r3_significant = rule_is_significant(r3_fired, n_eval);
    let r5_significant = rule_is_significant(r5_fired, n_eval);

    let r4 = r4_recommendation(
        summary_baseline.as_ref().and_then(|b| b.kv_headroom_gb),
        summary.ctx.config.tensor_parallel_size,
        summary_baseline.as_ref().map(|b| b.weight_gb),
        summary.ctx.gpu.vram_gb,
        summary.ctx.config.gpu_memory_utilization,
        summary_baseline
            .as_ref()
            .map(|b| b.weight_dtype_source)
            .unwrap_or(WeightDtypeSource::Fallback),
    );
    let r2_suppressed_by_r4_display = r4.is_some();
    let r4_groups: Vec<_> = r4
        .into_iter()
        .map(|r| IssueGroup {
            primary: r,
            secondary: Vec::new(),
        })
        .collect();

    let mut warnings = Vec::new();
    let kv_max_seqs = compute_kv_max_seqs(
        summary_baseline.as_ref().and_then(|b| b.kv_headroom_gb),
        summary.ctx.config.max_model_len,
        &summary.ctx.model,
        summary.ctx.config.kv_cache_dtype.as_deref(),
    );

    if r1_significant {
        let block = format_under_batching_window_issue(
            &aggregate_r1_detail(r1_details),
            pct(r1_fired, n_eval),
            0.8,
        );
        warnings.extend(block);
        warnings.push(String::new());
    }

    if r2_significant && !r2_suppressed_by_r4_display {
        let r2_agg = aggregate_r2_detail(r2_details);
        let block = format_kv_cache_window_issue(
            &r2_agg,
            pct(r2_fired, n_eval),
            summary_snap,
            r2_fired,
            n_eval,
            KvPressureFormatOpts {
                max_model_len: summary.ctx.config.max_model_len,
                kv_headroom_gb: summary_baseline.as_ref().and_then(|b| b.kv_headroom_gb),
                kv_max_seqs,
            },
        );
        warnings.extend(block);
        warnings.push(String::new());
    } else if r2_backlog_significant {
        let agg = aggregate_backlog_detail(r2_backlog_details);
        let block = format_kv_admission_backlog_issue(
            &agg,
            pct(r2_backlog_fired, n_eval),
            summary.ctx.config.max_model_len,
            summary_baseline.as_ref().and_then(|b| b.kv_headroom_gb),
            r2_backlog_fired,
            n_eval,
        );
        warnings.extend(block);
        warnings.push(String::new());
    }

    if r5_significant && !r2_significant && !r2_backlog_significant {
        if let Some(agg) = aggregate_concurrency_saturation_detail(r5_details, eval.session_kv_peak)
        {
            let block = format_concurrency_saturation_window_issue(
                &agg,
                pct(r5_fired, n_eval),
                summary.ctx.config.max_model_len,
                kv_max_seqs,
            );
            warnings.extend(block);
            warnings.push(String::new());
        }
    }

    if r3_significant {
        let d = aggregate_r3_detail(r3_details, summary_snap);
        let (qps, prompt_mean) = r3_display_args(&summary_snap.vllm, &d);
        warnings.extend(format_low_prefix_window_issue(
            &d,
            pct(r3_fired, n_eval),
            summary_snap.vllm.cache_config.enable_prefix_caching,
            qps,
            prompt_mean,
            None,
        ));
        warnings.push(String::new());
    }

    for g in &r4_groups {
        if !warnings.is_empty() && !warnings.last().is_some_and(|l| l.is_empty()) {
            warnings.push(String::new());
        }
        warnings.extend(rule_display_block(
            g,
            verbose_rules,
            r2_suppressed_by_r4_display,
        ));
        warnings.push(String::new());
    }

    append_waste_line(&mut warnings, &eval.groups, baseline_ref, tps);

    let r2_adv = if !r2_significant && !r2_backlog_significant {
        r2_kv_cache_advisory(summary_snap, metrics_url)
    } else {
        None
    };
    let r4_adv = r4_advisory(
        summary_baseline.as_ref().and_then(|b| b.kv_headroom_gb),
        summary.ctx.gpu.vram_gb,
        summary_baseline.as_ref().map(|b| b.weight_gb),
    );
    let r2_adv_present = r2_adv.is_some();
    let r4_adv_present = r4_adv.is_some();
    let mut advisories = Vec::new();
    if let Some(lines) = r2_adv {
        advisories.extend(lines);
        advisories.push(String::new());
    }
    if let Some(lines) = r4_adv {
        advisories.extend(lines);
        advisories.push(String::new());
    }

    let advisories_present = !advisories.is_empty();
    let mut out = warnings;
    out.append(&mut advisories);
    if let Some(line) = kv_ceiling_unknown_verbose_line(kv_max_seqs, verbose_rules) {
        append_display_block(&mut out, vec![line]);
    }

    let mut not_fired = Vec::new();
    if !r1_significant {
        not_fired.push("Under-batching");
    }
    if !r2_significant && !r2_backlog_significant && !r2_adv_present {
        not_fired.push("KV cache pressure");
    }
    if r4_groups.is_empty() && !r4_adv_present {
        not_fired.push("OOM risk");
    }
    if !r5_significant {
        not_fired.push("Concurrency saturation");
    }
    if !r3_significant {
        not_fired.push("Low prefix reuse");
    }
    let r5_warning = r5_significant && !r2_significant && !r2_backlog_significant;
    let any_warning = r1_significant
        || r2_significant
        || r2_backlog_significant
        || r3_significant
        || r5_warning
        || !r4_groups.is_empty();
    if verbose_rules {
        append_not_triggered_lines(
            &mut out,
            &not_fired,
            verbose_rules,
            Some((
                summary_snap,
                summary.ctx.config.max_num_seqs,
                summary_baseline.as_ref().and_then(|b| b.efficiency_pct),
            )),
        );
    }
    if !any_warning && !advisories_present && !verbose_rules {
        out.push(NO_ISSUES_LINE.to_string());
    }
    if skipped > 0 {
        out.push(String::new());
        out.push(format!(
            "Note: {skipped} of {total} windows dropped. Telemetry failure. Diagnosis may be incomplete."
        ));
    }

    trim_trailing_blank_lines(&mut out);
    out
}

pub(super) fn skew_secs(a: SystemTime, b: SystemTime) -> f64 {
    match a.duration_since(b) {
        Ok(d) => d.as_secs_f64(),
        Err(e) => -e.duration().as_secs_f64(),
    }
    .abs()
}

fn pct(fired: usize, total: usize) -> u32 {
    if total == 0 {
        return 0;
    }
    ((fired as f64 / total as f64) * 100.0).round() as u32
}

fn append_display_block(out: &mut Vec<String>, block: Vec<String>) {
    if !out.is_empty() && !block.is_empty() {
        out.push(String::new());
    }
    out.extend(block);
}

fn trim_trailing_blank_lines(lines: &mut Vec<String>) {
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{GpuRawMetrics, RawSnapshot, VllmConfig, VllmRawMetrics};
    use crate::context::{AnalysisInput, RuntimeWindow, StaticContext};
    use std::time::{Duration, SystemTime};

    fn snap(
        gpu_at: SystemTime,
        vllm_at: SystemTime,
        vllm: VllmRawMetrics,
        gpu: GpuRawMetrics,
    ) -> RawSnapshot {
        RawSnapshot {
            gpu_observed_at: gpu_at,
            vllm_observed_at: vllm_at,
            timestamp: gpu_at,
            vllm,
            gpu,
        }
    }

    fn mk_ctx() -> StaticContext {
        StaticContext::default()
    }

    #[test]
    fn compute_kv_max_seqs_uses_kv_layers_over_total_layers() {
        let hybrid = crate::context::ModelArch {
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_layers: Some(64),
            num_kv_layers: Some(32), // hybrid: only half the layers use KV cache
            ..Default::default()
        };
        // 2^34 byte budget → integer-clean seq counts at 4096 ctx (20 GB truncates to 37 vs 36)
        let headroom_gb = (1u64 << 34) as f64 / 1e9;
        let with_kv_layers = compute_kv_max_seqs(Some(headroom_gb), Some(4096), &hybrid, None);

        let dense = crate::context::ModelArch {
            num_kv_layers: None, // pure-attention: all 64 layers count
            ..hybrid
        };
        let without_kv_layers = compute_kv_max_seqs(Some(headroom_gb), Some(4096), &dense, None);

        assert!(with_kv_layers.is_some() && without_kv_layers.is_some());
        // 32 KV layers → half the bytes per token → fits 2× as many seqs
        assert_eq!(with_kv_layers.unwrap(), without_kv_layers.unwrap() * 2);
    }

    fn mk_win(s: RawSnapshot) -> RuntimeWindow {
        RuntimeWindow::from_snapshot(s)
    }

    fn ai<'a>(ctx: &'a StaticContext, win: &'a RuntimeWindow) -> AnalysisInput<'a> {
        AnalysisInput { ctx, window: win }
    }

    fn input_r4_suppresses_r2() -> (StaticContext, RuntimeWindow) {
        let t = SystemTime::UNIX_EPOCH;
        let snap = RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: VllmRawMetrics {
                model_name: Some("meta-llama/Llama-3.1-70B-Instruct".to_string()),
                num_requests_running: Some(3.0),
                num_requests_waiting: Some(0.0),
                max_num_seqs: Some(256),
                kv_cache_usage_perc: Some(89.0),
                num_preemptions_per_sec: Some(0.05),
                generation_tokens_per_sec: Some(50.0),
                request_success_per_sec: Some(10.0),
                window_duration_secs: Some(2.0),
                ..Default::default()
            },
            gpu: GpuRawMetrics {
                gpu_name: Some("NVIDIA H100 80GB HBM3".to_string()),
                vram_total_mb: Some(80 * 1024),
                gpu_util_pct: Some(58.0),
                ..Default::default()
            },
        };
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let ctx = StaticContext::from_snapshot(&snap, cfg);
        let win = RuntimeWindow::from_snapshot(snap);
        (ctx, win)
    }

    fn vllm_base() -> VllmRawMetrics {
        VllmRawMetrics {
            num_requests_running: Some(3.1),
            num_requests_waiting: Some(0.0),
            max_num_seqs: Some(256),
            kv_cache_usage_perc: Some(50.0),
            prefix_cache_hit_rate: Some(0.5),
            request_success_per_sec: Some(10.0),
            window_duration_secs: Some(2.0),
            ..Default::default()
        }
    }

    fn gpu_low() -> GpuRawMetrics {
        GpuRawMetrics {
            gpu_util_pct: Some(58.0),
            ..Default::default()
        }
    }

    fn gpu_busy() -> GpuRawMetrics {
        GpuRawMetrics {
            gpu_util_pct: Some(75.0),
            ..Default::default()
        }
    }

    #[test]
    fn under_batching_fires_when_gates_pass() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.tpot_ms = Some(35.0);
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        let r = r1_recommendation(&win.snapshot, None, None).expect("r1 fired");
        assert_eq!(r.rule_name, "under_batching");
        assert_eq!(r.impact, 4);
        assert!((r.confidence - 0.8).abs() < 1e-9);
        match rule1_under_batching(&win.snapshot, None) {
            Rule1Outcome::Fired(d) => {
                assert!((d.running - 3.1).abs() < 1e-9);
                assert_eq!(d.max_num_seqs, Some(256));
                assert!(d.occupancy_pct < 25.0);
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired"),
        }
    }

    #[test]
    fn waiting_none_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_waiting = None;
        v.tpot_ms = Some(35.0);
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot, None, None).is_none());
    }

    #[test]
    fn waiting_at_two_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_waiting = Some(2.0);
        v.tpot_ms = Some(35.0);
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot, None, None).is_none());
    }

    #[test]
    fn running_at_occupancy_threshold_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(64.0);
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot, None, None).is_none());
    }

    #[test]
    fn max_seqs_zero_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.max_num_seqs = Some(0);
        v.tpot_ms = Some(35.0);
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot, None, None).is_none());
    }

    #[test]
    fn nan_running_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(f64::NAN);
        v.tpot_ms = Some(35.0);
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot, None, None).is_none());
    }

    #[test]
    fn r2_advisory_requires_active_traffic() {
        let url = "http://127.0.0.1:8000/metrics";
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.kv_cache_usage_perc = None;
        v.num_requests_running = None;
        let raw = snap(t, t, v.clone(), gpu_busy());
        assert!(r2_kv_cache_advisory(&raw, url).is_none());

        v.num_requests_running = Some(0.0);
        let raw = snap(t, t, v.clone(), gpu_busy());
        assert!(r2_kv_cache_advisory(&raw, url).is_none());

        v.num_requests_running = Some(3.0);
        let raw = snap(t, t, v, gpu_busy());
        let lines = r2_kv_cache_advisory(&raw, url).expect("r2 advisory");
        assert!(lines[0].contains("core metric unavailable"));
    }

    #[test]
    fn format_under_batching_fired_matches_template() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(5.0);
        v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
        let mut g = gpu_low();
        g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
        let s = snap(t, t, v, g);
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let ctx = StaticContext::from_snapshot(&s, cfg);
        let win = mk_win(s);
        let lines = format_diagnose_rules(ai(&ctx, &win), false, "http://127.0.0.1:8000/metrics");
        let text = lines.join("\n");
        assert!(text.contains("[!] Under-batching: Insufficient Concurrency"));
        assert!(text.contains("Occupancy"));
        assert!(text.contains("threshold: < 25%"));
        assert!(text.contains("  Cause:"));
        assert!(text.contains("under-fed by client"));
        assert!(text.contains("    • Increase client concurrency"));
        assert!(!text.contains("slots idle"));
        assert!(text.contains("Expected: Higher throughput, stable TPOT."));
        assert!(
            text.contains("Confidence: High") || text.contains("Confidence: Medium"),
            "confidence reflects efficiency availability: {text}"
        );
    }

    #[test]
    fn format_diagnose_rules_for_windows_r4_suppresses_r2_when_both_significant() {
        let (ctx, _) = input_r4_suppresses_r2();
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        for w in windows.iter_mut().take(8) {
            *w = mk_evaluable_kv_window(89.0, true);
        }
        let summary_win = windows.last().expect("windows");
        let summary = ai(&ctx, summary_win);
        let text = format_diagnose_rules_for_windows(
            &windows,
            summary,
            true,
            "http://127.0.0.1:8000/metrics",
        )
        .join("\n");
        assert!(text.contains("[!] OOM Risk"));
        assert!(!text.contains("[!] KV Cache Pressure"));
        assert!(text.contains(R2_SUPPRESSED_BY_R4_VERBOSE_LINE));
    }

    #[test]
    fn format_diagnose_verbose_shows_r2_suppression_note_on_r4() {
        let (ctx, win) = input_r4_suppresses_r2();
        let text =
            format_diagnose_rules(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics").join("\n");
        assert!(text.contains("[!] OOM Risk"));
        assert!(text.contains(R2_SUPPRESSED_BY_R4_VERBOSE_LINE));
        assert!(!text.contains("KV cache pressure: not triggered"));
        assert!(!text.contains("[!] KV Cache Pressure"));
    }

    #[test]
    fn format_diagnose_non_verbose_omits_r2_suppression_note() {
        let (ctx, win) = input_r4_suppresses_r2();
        let text = format_diagnose_rules(ai(&ctx, &win), false, "http://127.0.0.1:8000/metrics")
            .join("\n");
        assert!(text.contains("[!] OOM Risk"));
        assert!(!text.contains(R2_SUPPRESSED_BY_R4_VERBOSE_LINE));
        assert!(!text.contains("[!] KV Cache Pressure"));
    }

    #[test]
    fn format_diagnose_verbose_r1_shows_prefill_saturation_when_gate_suppresses() {
        use crate::collectors::HistogramWindowMass;
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(5.0);
        v.num_requests_waiting = Some(0.0);
        v.prefill_window_mass = Some(HistogramWindowMass {
            sum_delta: 1.6,
            count_delta: 2.0,
        });
        v.window_duration_secs = Some(1.0);
        let s = snap(t, t, v, gpu_busy());
        let ctx = mk_ctx();
        let win = mk_win(s);
        let text =
            format_diagnose_rules(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics").join("\n");
        assert!(text.contains("Under-batching: not triggered (prefill saturated at 80%)"));
    }

    #[test]
    fn format_diagnose_verbose_shows_not_indicated_when_no_issue() {
        let t = SystemTime::UNIX_EPOCH;
        let mut g = gpu_low();
        g.gpu_util_pct = Some(75.0);
        let mut v = vllm_base();
        v.num_requests_running = Some(64.0);
        let s = snap(t, t, v, g);
        let ctx = mk_ctx();
        let win = mk_win(s);
        let text =
            format_diagnose_rules(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics").join("\n");
        assert!(text.contains("Under-batching: not triggered"));
        assert!(text.contains("KV cache pressure: not triggered"));
        assert!(text.contains("Low prefix reuse: not triggered"));
        assert!(text.contains("Concurrency saturation: not triggered"));
        assert!(!text.contains("No issues detected in this snapshot."));
    }

    fn vllm_high_kv() -> VllmRawMetrics {
        VllmRawMetrics {
            kv_cache_usage_perc: Some(89.0),
            ..vllm_base()
        }
    }

    fn mk_evaluable_kv_window(kv_pct: f64, preemptions: bool) -> RuntimeWindow {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.kv_cache_usage_perc = Some(kv_pct);
        v.generation_tokens_per_sec = Some(100.0);
        v.num_requests_running = Some(100.0);
        if preemptions {
            v.num_preemptions_per_sec = Some(0.05);
        }
        mk_win(snap(t, t, v, gpu_busy()))
    }

    fn mk_evaluable_backlog_window(
        kv_pct: f64,
        wait: f64,
        run: f64,
        prompt_mean: f64,
        num_gpu_blocks: u32,
        block_size: u32,
    ) -> RuntimeWindow {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.kv_cache_usage_perc = Some(kv_pct);
        v.num_requests_waiting = Some(wait);
        v.num_requests_running = Some(run);
        v.prompt_tokens_mean = Some(prompt_mean);
        v.generation_tokens_per_sec = Some(100.0);
        v.cache_config = crate::collectors::CacheConfigLabels {
            num_gpu_blocks: Some(num_gpu_blocks),
            block_size: Some(block_size),
            ..Default::default()
        };
        mk_win(snap(t, t, v, gpu_busy()))
    }

    fn r2_issue_lines(windows: Vec<RuntimeWindow>) -> Vec<String> {
        let ctx = mk_ctx();
        let summary = ai(&ctx, windows.last().expect("windows"));
        format_diagnose_rules_for_windows(&windows, summary, false, "http://127.0.0.1:8000/metrics")
    }

    #[test]
    fn r2_recommendation_confidence_from_density_counts() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_high_kv();
        v.num_preemptions_per_sec = Some(0.05);
        let s = snap(t, t, v, gpu_low());
        let r = r2_recommendation(&s, None, None, None, 1, 4).expect("fired");
        assert_eq!(r.rule_name, "kv_cache_pressure");
        assert_eq!(r.impact, 5);
        assert!((r.confidence - 0.5).abs() < 1e-9);
    }

    #[test]
    fn r2_recommendation_includes_peak_from_detail() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_high_kv();
        v.kv_cache_usage_perc = Some(89.0);
        v.kv_cache_peak_perc = Some(99.4);
        v.num_preemptions_per_sec = Some(0.05);
        let s = snap(t, t, v, gpu_low());
        let r = r2_recommendation(&s, None, None, None, 1, 1).expect("fired");
        let text = r.display_lines.join("\n");
        assert!(text.contains("KV cache hit 99.4% peak (threshold: 88%)"));
    }

    #[test]
    fn kv_cache_pressure_fires_at_88_boundary_with_stress() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.kv_cache_usage_perc = Some(88.0);
        v.num_preemptions_per_sec = Some(0.05);
        let s = snap(t, t, v, gpu_low());
        match rule2_kv_cache_pressure(&s) {
            Rule2Outcome::Fired(d) => {
                assert!((d.kv_cache_usage_perc - 88.0).abs() < 1e-9);
                assert!(d.preemptions_active);
            }
            Rule2Outcome::NotFired => panic!("expected fired at 88% with stress"),
        }
    }

    #[test]
    fn kv_cache_pressure_suppressed_below_88() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.kv_cache_usage_perc = Some(87.9);
        v.num_preemptions_per_sec = Some(0.05);
        let s = snap(t, t, v, gpu_low());
        assert!(matches!(
            rule2_kv_cache_pressure(&s),
            Rule2Outcome::NotFired
        ));
    }

    #[test]
    fn kv_cache_pressure_skew_suppresses() {
        let t0 = SystemTime::UNIX_EPOCH;
        let t1 = t0 + Duration::from_secs(2);
        let mut v = vllm_high_kv();
        v.num_requests_running = Some(64.0);
        let s = snap(t0, t1, v, gpu_low());
        assert!(matches!(
            rule2_kv_cache_pressure(&s),
            Rule2Outcome::NotFired
        ));
        let ctx = mk_ctx();
        let win = mk_win(s);
        let text =
            format_diagnose_rules(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics").join("\n");
        assert!(text.contains("Under-batching: not triggered"));
        assert!(text.contains("KV cache pressure: not triggered"));
        assert!(text.contains("Low prefix reuse: not triggered"));
        assert!(text.contains("Concurrency saturation: not triggered"));
        assert!(!text.contains("No issues detected in this snapshot."));
    }

    fn vllm_high_kv_stressed() -> VllmRawMetrics {
        VllmRawMetrics {
            kv_cache_usage_perc: Some(89.0),
            num_preemptions_per_sec: Some(0.05),
            ..vllm_base()
        }
    }

    #[test]
    fn kv_cache_pressure_preemption_displays_without_premature_confidence() {
        let t = SystemTime::UNIX_EPOCH;
        let s_kv_only = snap(t, t, vllm_high_kv_stressed(), gpu_busy());
        let ctx2 = mk_ctx();
        let win_kv_only = mk_win(s_kv_only);
        let r2_text = r2_recommendation(&win_kv_only.snapshot, None, None, None, 1, 1)
            .expect("r2 fired")
            .display_lines
            .join("\n");
        assert!(!r2_text.contains("Confidence:"));
        let text = format_diagnose_rules(
            ai(&ctx2, &win_kv_only),
            false,
            "http://127.0.0.1:8000/metrics",
        )
        .join("\n");
        assert!(text.contains("Cause:"));
        assert!(text.contains("KV cache hit 89.0% peak (threshold: 88%)"));
        assert!(text.contains("Expected: TTFT and TPOT recover once evictions stop."));
        assert!(text.contains("Lower --max-num-seqs to stop evictions"));
        assert!(text.contains("Switch --kv-cache-dtype fp8"));
    }

    #[test]
    fn kv_cache_miss_unavailable_without_gauge_verbose() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(64.0);
        v.kv_cache_usage_perc = None;
        let s = snap(t, t, v, gpu_busy());
        let ctx = mk_ctx();
        let win = mk_win(s);
        let text =
            format_diagnose_rules(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics").join("\n");
        assert!(text.contains("Under-batching: not triggered"));
        assert!(text.contains("[i] KV Cache Pressure: core metric unavailable"));
        assert!(!text.contains("KV cache pressure: not triggered"));
        assert!(text.contains("Low prefix reuse: not triggered"));
        assert!(text.contains("Concurrency saturation: not triggered"));
        assert!(!text.contains("No issues detected in this snapshot."));
    }

    #[test]
    fn rule3_fires_when_hit_below_35_and_gates_pass() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.prefix_cache_hit_rate = Some(0.34);
        v.prompt_tokens_mean = Some(25.0);
        v.request_success_per_sec = Some(40.0);
        v.num_requests_running = Some(1.0);
        let s = snap(t, t, v, gpu_busy());
        let win = mk_win(s);
        match rule3_low_prefix_reuse(&win.snapshot) {
            Rule3Outcome::Fired(d) => {
                assert_eq!(d.hit_rate, Some(0.34));
                assert_eq!(d.prompt_tokens_mean, Some(25.0));
            }
            Rule3Outcome::NotFired => panic!("expected fired"),
        }
        let r = r3_recommendation(&win.snapshot).expect("r3 fired");
        assert_eq!(r.rule_name, "low_prefix_reuse");
        assert_eq!(r.impact, 2);
        assert!((r.confidence - 0.9).abs() < 1e-9);
    }

    #[test]
    fn rule3_suppressed_at_or_above_35() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.prefix_cache_hit_rate = Some(0.35);
        v.prompt_tokens_mean = Some(25.0);
        v.request_success_per_sec = Some(40.0);
        v.num_requests_running = Some(1.0);
        let s = snap(t, t, v, gpu_busy());
        assert!(matches!(rule3_low_prefix_reuse(&s), Rule3Outcome::NotFired));
    }

    #[test]
    fn format_low_prefix_hit_rate_fired_matches_template() {
        let d = LowPrefixReuseDetail {
            hit_rate: Some(0.24),
            prompt_tokens_mean: Some(128.0),
            queries_delta: None,
        };
        let lines = format_low_prefix_hit_rate_fired(&d, Some(true), 10.0, Some(128.0), None);
        let text = lines.join("\n");
        assert!(text.contains("[!] Low Prefix Cache"));
        assert!(text.contains("  Cause:"));
        assert!(text.contains("  - Prefix hit rate 24.0% (threshold: 35%)"));
        assert!(text.contains("  - Prompt throughput: 1280 tok/s (threshold: 1000)"));
        assert!(text.contains("Restructure prompts to share common prefixes"));
        assert!(text.contains("  Fix:"));
        assert!(text.contains("Move shared instructions/system prompts to the very start"));
        assert!(text.contains("Standardize prompt templates across requests"));
        assert!(text.contains("Avoid unique tokens"));
        assert!(text.contains("Expected: Lower TTFT on repeated prefixes"));
        assert!(text.contains("Confidence: High"));
    }

    #[test]
    fn format_diagnose_rule3_verbose_working_effectively_when_rate_healthy() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.prefix_cache_hit_rate = Some(0.50);
        let s = snap(t, t, v, gpu_busy());
        let ctx = mk_ctx();
        let win = mk_win(s);
        let text =
            format_diagnose_rules(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics").join("\n");
        assert!(text.contains("Low prefix reuse: not triggered"));
    }

    #[test]
    fn format_diagnose_rule3_verbose_not_indicated_when_rate_low_but_prompt_below_floor() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.prefix_cache_hit_rate = Some(0.20);
        v.prompt_tokens_mean = Some(10.0);
        let s = snap(t, t, v, gpu_busy());
        let ctx = mk_ctx();
        let win = mk_win(s);
        let text =
            format_diagnose_rules(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics").join("\n");
        assert!(text.contains("Low prefix reuse: not triggered"));
        assert!(!text.contains("working effectively"));
    }

    #[test]
    fn format_diagnose_rules_no_fires_default_is_only_no_issues_line() {
        let t = SystemTime::UNIX_EPOCH;
        let mut g = gpu_low();
        g.gpu_util_pct = Some(75.0);
        g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
        g.vram_total_mb = Some(80 * 1024);
        let mut v = vllm_base();
        v.num_requests_running = Some(64.0);
        v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
        let s = snap(t, t, v, g);
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let ctx = StaticContext::from_snapshot(&s, cfg);
        let win = mk_win(s);
        let lines = format_diagnose_rules(ai(&ctx, &win), false, "http://127.0.0.1:8000/metrics");
        assert_eq!(
            lines,
            vec!["No issues detected in this snapshot.".to_string()]
        );
    }

    #[test]
    fn format_diagnose_rules_inserts_blank_between_rule_blocks() {
        let (ctx, win) = {
            let mut v = vllm_high_kv();
            v.num_preemptions_per_sec = Some(0.05);
            v.tpot_ms = Some(35.0);
            v.generation_tokens_per_sec = Some(30.0);
            v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
            let mut g = gpu_low();
            g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
            let t = SystemTime::UNIX_EPOCH;
            let snap = snap(t, t, v, g);
            let cfg = VllmConfig {
                dtype: Some("bf16".to_string()),
                max_model_len: Some(2048),
                ..Default::default()
            };
            let ctx = StaticContext::from_snapshot(&snap, cfg);
            let win = mk_win(snap);
            (ctx, win)
        };
        let lines = format_diagnose_rules(ai(&ctx, &win), false, "http://127.0.0.1:8000/metrics");
        let idx_under = lines
            .iter()
            .position(|l| l.contains("[!] Under-batching: Insufficient Concurrency"))
            .expect("rule1");
        let idx_kv = lines
            .iter()
            .position(|l| l.contains("[!] KV Cache Pressure"))
            .expect("rule2");
        let (lo, hi) = if idx_under < idx_kv {
            (idx_under, idx_kv)
        } else {
            (idx_kv, idx_under)
        };
        let between = &lines[lo..hi];
        assert!(
            between.iter().any(|l| l.is_empty()),
            "expected blank line between rule blocks: {between:?}"
        );
        assert!(
            !lines.iter().any(|l| l.contains("No issues detected")),
            "should not append no-issues line when at least one rule fired"
        );
        let waste_lines: Vec<_> = lines.iter().filter(|l| l.contains("/hr ")).collect();
        assert_eq!(
            waste_lines.len(),
            1,
            "expected one shared waste line: {lines:?}"
        );
        assert!(waste_lines[0].contains("lost to compounding bottlenecks"));
    }

    #[test]
    fn waste_label_r1_only() {
        assert_eq!(
            waste_label_suffix(&["under_batching"]),
            Some("wasted on idle compute")
        );
    }

    #[test]
    fn waste_label_r2_only() {
        assert_eq!(
            waste_label_suffix(&["kv_cache_pressure"]),
            Some("lost to memory thrashing")
        );
    }

    #[test]
    fn waste_label_r3_only() {
        assert_eq!(
            waste_label_suffix(&["low_prefix_reuse"]),
            Some("wasted on redundant prefill")
        );
    }

    #[test]
    fn waste_label_r5_only() {
        assert_eq!(
            waste_label_suffix(&["concurrency_saturation"]),
            Some("lost to scheduler queuing")
        );
    }

    #[test]
    fn waste_label_multi_rule() {
        assert_eq!(
            waste_label_suffix(&["under_batching", "kv_cache_pressure"]),
            Some("lost to compounding bottlenecks")
        );
    }

    #[test]
    fn waste_label_unknown_rule() {
        assert_eq!(
            waste_label_suffix(&["oom_risk"]),
            Some("unclassified overhead")
        );
    }

    #[test]
    fn rule_is_significant_six_of_ten_windows_passes() {
        assert!(rule_is_significant(6, 10));
    }

    #[test]
    fn rule_is_significant_three_of_fifteen_fails_density_gate() {
        assert!(!rule_is_significant(3, 15));
    }

    #[test]
    fn rule_is_significant_four_of_fifteen_passes() {
        assert!(rule_is_significant(4, 15));
    }

    #[test]
    fn rule_is_significant_zero_evaluable_windows_is_false() {
        assert!(!rule_is_significant(3, 0));
    }

    #[test]
    fn r2_fires_on_single_preemption_window() {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        windows[0] = mk_evaluable_kv_window(89.0, true);
        let text = r2_issue_lines(windows).join("\n");
        assert!(!text.contains("[!] KV Cache Pressure"));
    }

    #[test]
    fn r2_fires_on_two_critical_kv_windows_without_preemptions() {
        let mut windows: Vec<_> = (0..10)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        windows[0] = mk_evaluable_kv_window(96.0, false);
        windows[1] = mk_evaluable_kv_window(97.0, false);
        let text = r2_issue_lines(windows).join("\n");
        assert!(!text.contains("[!] KV Cache Pressure"));
    }

    #[test]
    fn r2_does_not_fire_when_kv_high_but_tpot_stable_and_no_preemptions() {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        for w in windows.iter_mut().take(4) {
            *w = mk_evaluable_kv_window(89.0, false);
        }
        let text = r2_issue_lines(windows).join("\n");
        assert!(!text.contains("[!] KV Cache Pressure"));
    }

    #[test]
    fn r2_confidence_equals_duration_density() {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        for w in windows.iter_mut().take(4) {
            *w = mk_evaluable_kv_window(89.0, true);
        }
        let ctx = mk_ctx();
        let mut summary_win = windows.last().expect("windows").clone();
        summary_win.snapshot.vllm.kv_cache_usage_perc = Some(72.5);
        summary_win.snapshot.vllm.kv_cache_peak_perc = Some(99.4);
        let summary = ai(&ctx, &summary_win);
        let report = build_report_for_windows(&windows, summary);
        let r2 = report
            .groups
            .iter()
            .find(|g| g.primary.rule_name == "kv_cache_pressure")
            .expect("r2 group");
        assert!((r2.primary.confidence - (4.0 / 15.0)).abs() < 1e-9);
        let text = format_diagnose_rules_for_windows(
            &windows,
            summary,
            false,
            "http://127.0.0.1:8000/metrics",
        )
        .join("\n");
        assert!(text.contains("KV cache hit 99.4% peak (threshold: 88%)"));
        assert!(text.contains("Seen in 27% of windows"));
        assert!(text.contains("Confidence: Medium"));
    }

    #[test]
    fn cause_line_peak_matches_summary_snapshot() {
        // 4 windows fired at 95% KV; 11 windows below threshold (30%).
        // Summary snapshot carries kv_cache_peak_perc=95.0 (realistic: profiler takes
        // MAX across windows) and kv_cache_usage_perc=92.0 (different value so we can
        // confirm the cause line reads kv_cache_peak_perc, not the usage fallback).
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(30.0, true))
            .collect();
        for w in windows.iter_mut().take(4) {
            *w = mk_evaluable_kv_window(95.0, true);
        }
        let ctx = mk_ctx();
        let mut summary_win = windows.last().expect("windows").clone();
        summary_win.snapshot.vllm.kv_cache_usage_perc = Some(92.0);
        summary_win.snapshot.vllm.kv_cache_peak_perc = Some(95.0);
        let summary = ai(&ctx, &summary_win);
        let text = format_diagnose_rules_for_windows(
            &windows,
            summary,
            false,
            "http://127.0.0.1:8000/metrics",
        )
        .join("\n");
        assert!(text.contains("KV cache hit 95.0% peak (threshold: 88%)"));
        assert!(!text.contains("92.0% peak"));
    }

    #[test]
    fn cause_kv_line_precedes_preemptions_and_queue() {
        // Verifies output ordering: KV peak line must appear before preemptions
        // and queue backpressure lines (#1 fix).
        // Uses a summary snapshot that has both signals active so all three
        // cause lines appear, then checks position order.
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(30.0, false))
            .collect();
        for w in windows.iter_mut().take(6) {
            *w = mk_evaluable_kv_window(91.0, true);
            w.snapshot.vllm.num_requests_waiting = Some(5.0);
        }
        let ctx = mk_ctx();
        let mut summary_win = windows.last().expect("windows").clone();
        summary_win.snapshot.vllm.kv_cache_usage_perc = Some(91.0);
        summary_win.snapshot.vllm.kv_cache_peak_perc = Some(91.0);
        summary_win.snapshot.vllm.num_preemptions_per_sec = Some(0.05);
        summary_win.snapshot.vllm.num_requests_waiting = Some(5.0);
        let summary = ai(&ctx, &summary_win);
        let text = format_diagnose_rules_for_windows(
            &windows,
            summary,
            false,
            "http://127.0.0.1:8000/metrics",
        )
        .join("\n");
        let pos_kv = text.find("KV cache hit").expect("KV peak line missing");
        let pos_preempt = text
            .find("Active preemptions")
            .expect("preemptions line missing");
        let pos_queue = text.find("Queue backpressure").expect("queue line missing");
        assert!(
            pos_kv < pos_preempt,
            "KV line must precede preemptions line"
        );
        assert!(pos_kv < pos_queue, "KV line must precede queue line");
    }

    #[test]
    fn r2_does_not_fire_on_single_critical_kv_window_without_preemptions() {
        let mut windows: Vec<_> = (0..10)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        windows[0] = mk_evaluable_kv_window(96.0, false);
        let text = r2_issue_lines(windows).join("\n");
        assert!(!text.contains("[!] KV Cache Pressure"));
        assert!(!text.contains("KV cache pressure: not triggered"));
        assert!(!text.contains("Low prefix reuse: not triggered"));
        assert!(!text.contains("Seen in"));
    }

    #[test]
    fn r2_backlog_fires_when_sustained_admission_pressure() {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_backlog_window(10.0, 1.0, 9.0, 10.0, 10_000, 16))
            .collect();
        for w in windows.iter_mut().take(4) {
            // KV 70% (< 88% standard r2 gate); free = 100×16×0.30 = 480; demand = 15×40 = 600
            *w = mk_evaluable_backlog_window(70.0, 15.0, 5.0, 40.0, 100, 16);
        }
        let text = r2_issue_lines(windows).join("\n");
        assert!(text.contains("[!] KV Cache Pressure: Admission Backlog"));
        assert!(text.contains("Free KV tokens"));
        assert!(!text.contains("threshold: 88%"));
    }

    #[test]
    fn backlog_short_action_matches_spec() {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_backlog_window(10.0, 1.0, 9.0, 10.0, 10_000, 16))
            .collect();
        for w in windows.iter_mut().take(4) {
            *w = mk_evaluable_backlog_window(70.0, 15.0, 5.0, 40.0, 100, 16);
        }
        let ctx = mk_ctx();
        let summary = ai(&ctx, windows.last().expect("windows"));
        let report = build_report_for_windows(&windows, summary);
        let r = report
            .groups
            .iter()
            .find(|g| g.primary.rule_name == "kv_admission_backlog")
            .expect("backlog kv recommendation")
            .primary
            .clone();
        assert_eq!(r.short_action, "raise --gpu-memory-utilization");
        let display = r.display_lines.join("\n");
        assert!(display.contains("[!] KV Cache Pressure: Admission Backlog"));
    }

    #[test]
    fn r2_backlog_suppressed_when_standard_r2_fires() {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        for w in windows.iter_mut().take(4) {
            *w = mk_evaluable_backlog_window(89.0, 15.0, 15.0, 20.0, 100, 16);
        }
        let text = r2_issue_lines(windows).join("\n");
        assert!(text.contains("[!] KV Cache Pressure"));
        assert!(!text.contains("Admission Backlog"));
    }

    #[test]
    fn r2_fires_on_sustained_warning_level_kv() {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        for w in windows.iter_mut().take(4) {
            *w = mk_evaluable_kv_window(89.0, true);
        }
        let text = r2_issue_lines(windows).join("\n");
        assert!(text.contains("[!] KV Cache Pressure"));
    }

    #[test]
    fn format_diagnose_rules_for_windows_matches_requested_style_when_some_rules_fire() {
        let t = SystemTime::UNIX_EPOCH;
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let mut windows = Vec::new();
        for i in 0..10 {
            let mut v = vllm_base();
            v.max_num_seqs = Some(256);
            v.num_requests_waiting = Some(1.0);
            v.kv_cache_usage_perc = Some(71.2);
            v.prefix_cache_hit_rate = Some(0.524);
            v.prompt_tokens_mean = Some(128.0);
            v.generation_tokens_per_sec = Some(1580.0);
            let mut g = gpu_busy();
            g.power_watts = Some(312.0);
            g.vram_used_mb = Some(62 * 1024);
            g.vram_total_mb = Some(80 * 1024);
            v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
            g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
            if i < 6 {
                v.num_requests_running = Some(3.2);
                v.tpot_ms = Some(35.0);
                g.gpu_util_pct = Some(50.0);
            } else {
                v.num_requests_running = Some(100.0);
                g.gpu_util_pct = Some(74.0);
            }
            windows.push(mk_win(snap(t, t, v, g)));
        }
        let ctx = StaticContext::from_snapshot(&windows[0].snapshot, cfg);
        let summary = ai(&ctx, windows.last().expect("summary source"));
        let lines = format_diagnose_rules_for_windows(
            &windows,
            summary,
            false,
            "http://127.0.0.1:8000/metrics",
        );
        let text = lines.join("\n");
        assert!(text.contains("Under-batching: Insufficient Concurrency"));
        assert!(text.contains("Seen in 100% of windows"));
        assert!(text.contains("Efficiency"));
        assert!(text.contains("threshold: < 60%"));
        assert!(text.contains("  Cause:"));
        assert!(text.contains("Increase client concurrency"));
        assert!(!text.contains("KV cache pressure: not triggered"));
        assert!(!text.contains("Low prefix reuse: not triggered"));
        assert!(!text.contains("Concurrency saturation: not triggered"));
    }

    #[test]
    fn insufficient_load_returns_advisory_not_no_issues() {
        let windows = vec![
            mk_evaluable_kv_window(89.0, true),
            mk_evaluable_kv_window(89.0, true),
        ];
        let ctx = mk_ctx();
        let summary = ai(&ctx, windows.last().expect("windows"));
        let text = format_diagnose_rules_for_windows(
            &windows,
            summary,
            false,
            "http://127.0.0.1:8000/metrics",
        )
        .join("\n");
        assert!(text.contains("Insufficient Sustained Load"));
        assert!(!text.contains("No issues detected"));
    }

    #[test]
    fn format_diagnose_rules_for_windows_no_fires_is_single_no_issues_line() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(20.0);
        v.num_requests_waiting = Some(3.0);
        v.kv_cache_usage_perc = Some(71.2);
        v.prefix_cache_hit_rate = Some(0.524);
        v.prompt_tokens_mean = Some(128.0);
        v.generation_tokens_per_sec = Some(100.0);
        v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
        let mut g = gpu_busy();
        g.gpu_util_pct = Some(74.0);
        g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
        g.vram_total_mb = Some(80 * 1024);
        let snap = snap(t, t, v, g);
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let ctx = StaticContext::from_snapshot(&snap, cfg);
        let win = mk_win(snap);
        let windows = vec![win.clone(), win.clone(), win];
        let summary = ai(&ctx, windows.last().unwrap());
        let lines = format_diagnose_rules_for_windows(
            &windows,
            summary,
            false,
            "http://127.0.0.1:8000/metrics",
        );
        assert_eq!(
            lines,
            vec!["No issues detected in this snapshot.".to_string()]
        );
    }

    #[test]
    fn format_diagnose_rules_non_evaluable_snapshot_shows_note() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = None;
        let s = snap(t, t, v, gpu_busy());
        let ctx = mk_ctx();
        let win = mk_win(s);
        let lines = format_diagnose_rules(ai(&ctx, &win), false, "http://127.0.0.1:8000/metrics");
        assert_eq!(
            lines,
            no_evaluable_diagnose_lines(false, std::slice::from_ref(&win))
        );
        let vlines = format_diagnose_rules(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics");
        assert!(vlines
            .iter()
            .any(|l| l.contains("1 of 1 collected windows")));
    }

    fn mk_evaluable_concurrency_saturation_window(
        run: f64,
        wait: f64,
        max_num_seqs: u32,
    ) -> RuntimeWindow {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(run);
        v.num_requests_waiting = Some(wait);
        v.max_num_seqs = Some(max_num_seqs);
        v.generation_tokens_per_sec = Some(100.0);
        mk_win(snap(t, t, v, gpu_busy()))
    }

    #[test]
    fn r5_concurrency_saturation_fires_on_sustained_saturation() {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        for w in windows.iter_mut().take(4) {
            *w = mk_evaluable_concurrency_saturation_window(32.0, 15.0, 32);
        }
        let ctx = mk_ctx();
        let summary = ai(&ctx, windows.last().expect("windows"));
        let text = format_diagnose_rules_for_windows(
            &windows,
            summary,
            false,
            "http://127.0.0.1:8000/metrics",
        )
        .join("\n");
        assert!(
            text.contains("[!] Concurrency Saturation"),
            "expected r5: {text}"
        );
        assert!(text.contains("--max-num-seqs=32 hit:"));
    }

    #[test]
    fn build_report_for_windows_r5_when_aggregate_snapshot_misses() {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        for w in windows.iter_mut().take(4) {
            *w = mk_evaluable_concurrency_saturation_window(32.0, 15.0, 32);
        }
        let ctx = mk_ctx();
        let summary = ai(&ctx, windows.last().expect("windows"));
        let aggregate_report = crate::engine::build_report(summary);
        assert!(
            !aggregate_report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == "concurrency_saturation"),
            "aggregate snapshot should not reproduce r5: {:?}",
            aggregate_report
                .groups
                .iter()
                .map(|g| g.primary.rule_name)
                .collect::<Vec<_>>()
        );
        let multi_report = build_report_for_windows(&windows, summary);
        assert!(
            multi_report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == "concurrency_saturation"),
            "multi-window report should include r5: {:?}",
            multi_report
                .groups
                .iter()
                .map(|g| g.primary.rule_name)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn r5_suppressed_when_r2_fires() {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_concurrency_saturation_window(32.0, 15.0, 32))
            .collect();
        for w in windows.iter_mut().take(4) {
            *w = mk_evaluable_kv_window(89.0, true);
        }
        let ctx = mk_ctx();
        let summary = ai(&ctx, windows.last().expect("windows"));
        let text = format_diagnose_rules_for_windows(
            &windows,
            summary,
            false,
            "http://127.0.0.1:8000/metrics",
        )
        .join("\n");
        assert!(text.contains("KV Cache Pressure"), "expected r2: {text}");
        assert!(!text.contains("[!] Concurrency Saturation"));
    }

    #[test]
    fn r5_uses_session_kv_peak_from_non_r5_window() {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        // r5-significant windows with moderate KV.
        for w in windows.iter_mut().take(4) {
            *w = mk_evaluable_concurrency_saturation_window(32.0, 15.0, 32);
            w.snapshot.vllm.kv_cache_usage_perc = Some(70.0);
        }
        // One high-KV r2 window (non-significant for r2), should still set session peak.
        windows[10] = mk_evaluable_kv_window(95.0, true);
        // Simulate gauge drift: peak captures the spike, avg usage does not.
        windows[10].snapshot.vllm.kv_cache_usage_perc = Some(60.0);
        windows[10].snapshot.vllm.kv_cache_peak_perc = Some(95.0);
        windows[10].snapshot.vllm.num_requests_running = Some(20.0);
        windows[10].snapshot.vllm.max_num_seqs = Some(32);
        windows[10].snapshot.vllm.num_requests_waiting = Some(1.0);

        let ctx = mk_ctx();
        let summary = ai(&ctx, windows.last().expect("windows"));
        let text = format_diagnose_rules_for_windows(
            &windows,
            summary,
            false,
            "http://127.0.0.1:8000/metrics",
        )
        .join("\n");

        assert!(
            text.contains("[!] Concurrency Saturation"),
            "expected r5: {text}"
        );
        assert!(text.contains("KV at 95%: scheduler at cap, pool full."));
        assert!(!text.contains("KV pool has room (70%)"));
    }

    #[test]
    fn session_kv_peak_from_non_r5_window_reaches_build_report_from_eval() {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        for w in windows.iter_mut().take(4) {
            *w = mk_evaluable_concurrency_saturation_window(32.0, 15.0, 32);
            w.snapshot.vllm.kv_cache_usage_perc = Some(70.0);
        }
        // Non-r5 window carries session spike via peak metric.
        windows[10] = mk_evaluable_kv_window(95.0, true);
        windows[10].snapshot.vllm.kv_cache_usage_perc = Some(60.0);
        windows[10].snapshot.vllm.kv_cache_peak_perc = Some(95.0);
        windows[10].snapshot.vllm.num_requests_running = Some(20.0);
        windows[10].snapshot.vllm.max_num_seqs = Some(32);
        windows[10].snapshot.vllm.num_requests_waiting = Some(1.0);

        let ctx = mk_ctx();
        let summary = ai(&ctx, windows.last().expect("windows"));
        let report = build_report_for_windows(&windows, summary);
        let r5 = report
            .groups
            .iter()
            .find(|g| g.primary.rule_name == "concurrency_saturation")
            .expect("r5 group");
        let text = r5.primary.display_lines.join("\n");
        assert!(text.contains("KV at 95%: scheduler at cap, pool full."));
        assert!(!text.contains("KV pool has room (70%)"));
        assert_eq!(r5.primary.action, "Add a replica to scale out.");
    }

    #[test]
    fn format_diagnose_rules_for_windows_all_non_evaluable() {
        let t = SystemTime::UNIX_EPOCH;
        let ctx = mk_ctx();
        let mut v = vllm_base();
        v.num_requests_running = None;
        let w1 = mk_win(snap(t, t, v.clone(), gpu_busy()));
        let w2 = mk_win(snap(t, t, v, gpu_busy()));
        let windows = vec![w1, w2];
        let summary = ai(&ctx, &windows[0]);
        let lines = format_diagnose_rules_for_windows(
            &windows,
            summary,
            false,
            "http://127.0.0.1:8000/metrics",
        );
        assert_eq!(lines, no_evaluable_diagnose_lines(false, &windows));
        let summary2 = ai(&ctx, &windows[0]);
        let vlines = format_diagnose_rules_for_windows(
            &windows,
            summary2,
            true,
            "http://127.0.0.1:8000/metrics",
        );
        assert!(vlines
            .iter()
            .any(|l| l.contains("2 of 2 collected windows")));
    }

    use crate::engine::baseline::{CeilingEstimate, CostEstimate, WeightDtypeSource};

    fn baseline_for_waste(eff: f64, source: CostSource, cpm: f64) -> PhysicsBaseline {
        PhysicsBaseline {
            decode: CeilingEstimate {
                lower: 90.0,
                expected: 100.0,
                upper: 110.0,
            },
            prefill: None,
            efficiency_pct: Some(eff),
            headroom_pct: Some(100.0 - eff),
            weight_dtype_source: WeightDtypeSource::Fallback,
            weight_gb: 1.0,
            kv_headroom_gb: None,
            tpot_floor_ms: 10.0,
            prefill_latency_floor_ms: None,
            ridge_batch_size: 1.0,
            cost: Some(CostEstimate {
                tok_per_watt: None,
                joules_per_token: None,
                cost_per_million_tokens: Some(cpm),
                cost_source: source,
            }),
        }
    }

    #[test]
    fn waste_line_appended_for_r1_r2_r3_r5() {
        let b = baseline_for_waste(32.0, CostSource::Catalog, 1.84);
        let tps = Some(14.2_f64);
        let cases = [
            (
                vec![issue_group("under_batching")],
                "wasted on idle compute",
            ),
            (
                vec![issue_group("kv_cache_pressure")],
                "lost to memory thrashing",
            ),
            (
                vec![issue_group("low_prefix_reuse")],
                "wasted on redundant prefill",
            ),
            (
                vec![issue_group("concurrency_saturation")],
                "lost to scheduler queuing",
            ),
        ];
        for (groups, suffix) in cases {
            let mut lines = vec!["issue".to_string()];
            append_waste_line(&mut lines, &groups, Some(&b), tps);
            let waste = lines.iter().find(|l| l.contains("/hr ")).expect(suffix);
            assert!(waste.ends_with(suffix), "got {waste}");
        }
    }

    #[test]
    fn waste_line_multi_rule_compounding() {
        let b = baseline_for_waste(32.0, CostSource::Catalog, 1.84);
        let groups = vec![
            issue_group("under_batching"),
            issue_group("kv_cache_pressure"),
        ];
        let mut lines = vec!["issue".to_string()];
        append_waste_line(&mut lines, &groups, Some(&b), Some(14.2));
        assert!(lines
            .iter()
            .any(|l| l.contains("lost to compounding bottlenecks")));
    }

    #[test]
    fn waste_line_unknown_rule_name_unclassified() {
        let groups = vec![issue_group("oom_risk")];

        let b = baseline_for_waste(32.0, CostSource::Catalog, 1.84);
        let mut lines = vec!["issue".to_string()];
        append_waste_line(&mut lines, &groups, Some(&b), Some(14.2));
        assert!(lines.iter().any(|l| l.contains("unclassified overhead")));

        // UserProvided source is accepted; label still falls through to unclassified.
        let b = baseline_for_waste(32.0, CostSource::UserProvided, 1.0);
        let mut lines = vec!["issue".to_string()];
        append_waste_line(&mut lines, &groups, Some(&b), Some(100.0));
        assert!(lines.iter().any(|l| l.contains("unclassified overhead")));
    }

    fn issue_group(rule_name: &'static str) -> IssueGroup {
        IssueGroup {
            primary: Recommendation {
                rule_name,
                impact: 4,
                confidence: 0.8,
                action: String::new(),
                short_action: String::new(),
                expected_impact: String::new(),
                display_lines: Vec::new(),
            },
            secondary: Vec::new(),
        }
    }

    #[test]
    fn waste_line_efficiency_over_100_omitted() {
        let b = baseline_for_waste(110.0, CostSource::Catalog, 1.84);
        let mut lines = vec!["issue".to_string()];
        append_waste_line(
            &mut lines,
            &[issue_group("under_batching")],
            Some(&b),
            Some(14.2),
        );
        assert_eq!(lines.len(), 1);
        assert!(!lines.iter().any(|l| l.contains("/hr ")));
    }

    #[test]
    fn waste_line_absent_without_cost_or_efficiency() {
        let mut b = baseline_for_waste(32.0, CostSource::Catalog, 1.84);
        b.efficiency_pct = None;
        let mut lines = vec!["issue".to_string()];
        append_waste_line(
            &mut lines,
            &[issue_group("under_batching")],
            Some(&b),
            Some(10.0),
        );
        assert_eq!(lines.len(), 1);

        b.efficiency_pct = Some(32.0);
        b.cost = None;
        append_waste_line(
            &mut lines,
            &[issue_group("under_batching")],
            Some(&b),
            Some(10.0),
        );
        assert_eq!(lines.len(), 1);
    }
}
