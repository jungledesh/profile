use std::time::SystemTime;

use crate::collectors::{
    RawSnapshot, VllmRawMetrics, effective_tensor_parallel, window_is_evaluable,
};
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine::baseline::{self, CostSource, PhysicsBaseline, WeightDtypeSource};

mod r1_under_batching;
mod r2_kv_cache_pressure;
mod r3_low_prefix_reuse;
mod r4_oom_risk;
mod r5_concurrency_saturation;
mod r6_prefill_bound;
mod r7_config_headroom;

pub use r1_under_batching::{
    R1MissReport, Rule1Outcome, UnderBatchingDetail, r1_recommendation, r1_verbose_miss_line,
};
pub use r2_kv_cache_pressure::{
    KvAdmissionBacklogDetail, KvCachePressureDetail, Rule2Outcome, r2_recommendation,
    rule2_kv_admission_backlog, rule2_kv_cache_pressure,
};
pub use r3_low_prefix_reuse::{
    LowPrefixReuseDetail, Rule3Outcome, r3_recommendation, rule3_low_prefix_reuse,
};
pub use r4_oom_risk::{r4_advisory, r4_recommendation};
pub use r5_concurrency_saturation::{
    ConcurrencySaturationDetail, r5_recommendation, rule5_concurrency_saturation,
};
pub use r6_prefill_bound::{
    PrefillBoundDetail, Rule6Outcome, r6_recommendation, r6_verbose_miss_line,
};
pub use r7_config_headroom::{ConfigHeadroomDetail, rule7_config_headroom};

use r1_under_batching::{
    aggregate_r1_detail, format_under_batching_window_issue, r1_short_action,
    rule1_under_batching_with_efficiency,
};
use r2_kv_cache_pressure::{
    KvFormatCtx, aggregate_backlog_detail, aggregate_r2_detail, format_kv_admission_backlog_issue,
    format_kv_cache_window_issue, kv_pressure_confidence, r2_action, r2_backlog_short_action,
    r2_kv_pressure_short_action,
};
#[cfg(test)]
use r3_low_prefix_reuse::format_low_prefix_hit_rate_fired;
use r3_low_prefix_reuse::{aggregate_r3_detail, format_low_prefix_window_issue};
use r5_concurrency_saturation::{
    aggregate_concurrency_saturation_detail, format_concurrency_saturation_window_issue, r5_action,
    r5_short_action,
};
use r6_prefill_bound::{
    aggregate_r6_detail, confidence as r6_confidence, evaluate as r6_evaluate,
    format_prefill_bound_window_issue, impact as r6_impact,
    prefill_fix_lines as r6_prefill_fix_lines, severity as r6_severity,
};
use r7_config_headroom::{aggregate_r7_detail, format_config_headroom_window_issue};

fn resolve_prefill_time_fraction(
    baseline: Option<&PhysicsBaseline>,
    snapshot: &RawSnapshot,
) -> Option<f64> {
    baseline
        .and_then(|b| b.prefill_time_fraction)
        .or_else(|| baseline::prefill_time_fraction_from_snapshot(snapshot))
}

pub(super) const MAX_OBSERVATION_SKEW_SECS: f64 = 1.0;
/// Enforces >= 6s temporal substance (3 windows × 2s).
pub(super) const ENGINE_MIN_PERSISTENT_WINDOWS: usize = 3;
/// Enforces >= 25% density floor across evaluable windows.
pub(super) const ENGINE_MIN_WINDOW_PCT: f64 = 0.25;

/// Push a max_model_len shrink suggestion into `lines`.
/// Hard number only when `total_count >= 100` and both p99s are present.
/// No-op when `max_model_len` is None.
pub(super) fn push_model_len_shrink_suggestion(
    lines: &mut Vec<String>,
    max_model_len: Option<u32>,
    prompt_p99: Option<f64>,
    generation_p99: Option<f64>,
    total_count: f64,
    indent: &str,
) {
    let Some(m) = max_model_len else { return };

    if total_count >= 100.0 {
        let Some(pp) = prompt_p99 else {
            lines.push(format!(
                "{indent}• Lower --max-model-len (current: {m}) to safely raise concurrency."
            ));
            return;
        };
        let Some(gp) = generation_p99 else {
            lines.push(format!(
                "{indent}• Lower --max-model-len (current: {m}) to safely raise concurrency."
            ));
            return;
        };
        let suggested = (pp as u32).saturating_add(gp as u32);
        // Suppress if reduction is < 5% - not a meaningful change (avoids "5464 → 5465" no-ops)
        if suggested >= m.saturating_sub(m / 20) {
            return;
        }
        lines.push(format!(
            "{indent}• Lower --max-model-len (current: {m}) to ~{suggested} \
             (prompt p99 {pp:.0} tok + output p99 {gp:.0} tok), to shrink KV footprint."
        ));
        lines.push(format!(
            "{indent}  Warning: max_model_len is total context (prompt + completion). Truncation risk!"
        ));
    } else {
        lines.push(format!(
            "{indent}• Lower --max-model-len (current: {m}) to safely raise concurrency."
        ));
    }
}

pub(super) fn compute_kv_max_seqs(
    kv_headroom_gb: Option<f64>,
    max_model_len: Option<u32>,
    model: &crate::context::ModelArch,
    kv_cache_dtype: Option<&str>,
    tp: Option<u32>,
) -> Option<u32> {
    use crate::engine::baseline::{kv_bytes_per_element, kv_max_concurrent_seqs};
    let headroom = kv_headroom_gb?;
    let max_len = max_model_len?;
    let num_layers = model.num_kv_layers.or(model.num_layers)?;
    let num_kv_heads = model.num_kv_heads?;
    let head_dim = model.head_dim?;
    let sharded_kv_heads = tp
        .map(|t| num_kv_heads / num_kv_heads.min(t))
        .unwrap_or(num_kv_heads);
    let kv_bpp = kv_bytes_per_element(kv_cache_dtype, 2);
    kv_max_concurrent_seqs(
        headroom,
        max_len,
        num_layers,
        sharded_kv_heads,
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

/// Inserts "Seen in N% of windows" after the rule title line in multi-window display blocks.
pub(super) fn with_seen_pct(mut lines: Vec<String>, seen_pct: u32) -> Vec<String> {
    lines.insert(1, format!("  Seen in {seen_pct}% of windows"));
    lines
}

#[derive(Debug, Clone, PartialEq)]
pub struct Recommendation {
    pub rule_name: &'static str,
    pub layer: u8,
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

/// Canonical `Recommendation.rule_name` values - single source of truth for DAG + output coupling.
pub mod rule_names {
    pub const UNDER_BATCHING: &str = "under_batching";
    pub const KV_CACHE_PRESSURE: &str = "kv_cache_pressure";
    pub const KV_ADMISSION_BACKLOG: &str = "kv_admission_backlog";
    pub const OOM_RISK: &str = "oom_risk";
    pub const CONCURRENCY_SATURATION: &str = "concurrency_saturation";
    pub const LOW_PREFIX_REUSE: &str = "low_prefix_reuse";
    pub const PREFILL_BOUND: &str = "prefill_bound";
    pub const CONFIG_HEADROOM: &str = "config_headroom";
    pub const MASSIVE_UNDERUTILIZATION: &str = "massive_underutilization";

    /// Human-readable label for a rule name - used in journey UI output.
    pub fn display_name(rule_name: &str) -> &str {
        match rule_name {
            UNDER_BATCHING => "Under-batching",
            KV_CACHE_PRESSURE => "KV Cache Pressure",
            KV_ADMISSION_BACKLOG => "KV Admission Backlog",
            OOM_RISK => "OOM Risk",
            CONCURRENCY_SATURATION => "Concurrency Saturation",
            LOW_PREFIX_REUSE => "Low Prefix Reuse",
            PREFILL_BOUND => "Prefill-Bound",
            CONFIG_HEADROOM => "Config Headroom",
            MASSIVE_UNDERUTILIZATION => "Massive Under-utilization",
            _ => rule_name,
        }
    }
}

const SUPPRESSION_TABLE: &[(&str, &str)] = &[
    (rule_names::OOM_RISK, rule_names::KV_CACHE_PRESSURE),
    (rule_names::OOM_RISK, rule_names::KV_ADMISSION_BACKLOG),
    (rule_names::UNDER_BATCHING, rule_names::PREFILL_BOUND),
    (rule_names::CONFIG_HEADROOM, rule_names::UNDER_BATCHING),
];

const KV_RULE_NAMES: &[&str] = &[
    rule_names::KV_CACHE_PRESSURE,
    rule_names::KV_ADMISSION_BACKLOG,
];

const KV_NOT_TRIGGERED_LABEL: &str = "KV cache pressure";

struct NotTriggeredRule {
    rule_name: &'static str,
    label: &'static str,
}

const NOT_TRIGGERED_SINGLES: &[NotTriggeredRule] = &[
    NotTriggeredRule {
        rule_name: rule_names::UNDER_BATCHING,
        label: "Under-batching",
    },
    NotTriggeredRule {
        rule_name: rule_names::OOM_RISK,
        label: "OOM risk",
    },
    NotTriggeredRule {
        rule_name: rule_names::CONCURRENCY_SATURATION,
        label: "Concurrency saturation",
    },
    NotTriggeredRule {
        rule_name: rule_names::CONFIG_HEADROOM,
        label: "Config headroom",
    },
    NotTriggeredRule {
        rule_name: rule_names::LOW_PREFIX_REUSE,
        label: "Low prefix reuse",
    },
    NotTriggeredRule {
        rule_name: rule_names::PREFILL_BOUND,
        label: "Prefill-bound",
    },
];

const NOT_TRIGGERED_KV: NotTriggeredRule = NotTriggeredRule {
    rule_name: rule_names::KV_CACHE_PRESSURE,
    label: KV_NOT_TRIGGERED_LABEL,
};

fn kv_rules_absent_from_fired(fired_names: &std::collections::HashSet<&'static str>) -> bool {
    !KV_RULE_NAMES.iter().any(|name| fired_names.contains(name))
}

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

fn rule_display_block(g: &IssueGroup) -> Vec<String> {
    g.primary.display_lines.clone()
}

struct CollectedAdvisories {
    r2_present: bool,
    r4_present: bool,
    lines: Vec<String>,
}

impl CollectedAdvisories {
    fn any(&self) -> bool {
        !self.lines.is_empty()
    }
}

fn collect_advisories(
    fired_names: &std::collections::HashSet<&'static str>,
    snapshot: &RawSnapshot,
    metrics_url: &str,
    kv_headroom_gb: Option<f64>,
    gpu_vram_gb: Option<f64>,
    weight_gb: Option<f64>,
) -> CollectedAdvisories {
    let r2_adv = if kv_rules_absent_from_fired(fired_names) {
        r2_kv_cache_advisory(snapshot, metrics_url)
    } else {
        None
    };
    let r4_adv = r4_advisory(kv_headroom_gb, gpu_vram_gb, weight_gb);
    let r2_present = r2_adv.is_some();
    let r4_present = r4_adv.is_some();
    let mut lines = Vec::new();
    if let Some(block) = r2_adv {
        lines.extend(block);
    }
    if let Some(block) = r4_adv {
        if !lines.is_empty() {
            lines.push(String::new());
        }
        lines.extend(block);
    }
    CollectedAdvisories {
        r2_present,
        r4_present,
        lines,
    }
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

// COUPLING: keys must match `rule_names` constants.
pub(super) fn waste_label_suffix(rule_names_list: &[&str]) -> Option<&'static str> {
    match rule_names_list.len() {
        0 => None,
        1 => match rule_names_list[0] {
            rule_names::UNDER_BATCHING => Some("wasted on idle compute"),
            rule_names::KV_CACHE_PRESSURE => Some("lost to memory thrashing"),
            rule_names::LOW_PREFIX_REUSE => Some("wasted on redundant prefill"),
            rule_names::PREFILL_BOUND => Some("lost to prefill interference"),
            rule_names::CONCURRENCY_SATURATION => Some("lost to scheduler queuing"),
            rule_names::CONFIG_HEADROOM => Some("wasted on config-limited batching"),
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

struct R1VerboseContext<'a> {
    snapshot: &'a RawSnapshot,
    max_num_seqs: Option<u32>,
    efficiency_pct: Option<f64>,
    config_relative_efficiency_pct: Option<f64>,
    prefill_time_fraction: Option<f64>,
}

fn append_not_triggered_lines(
    out: &mut Vec<String>,
    rules: &[&NotTriggeredRule],
    verbose_rules: bool,
    r1_context: Option<R1VerboseContext<'_>>,
    baseline: Option<&PhysicsBaseline>,
) {
    if rules.is_empty() {
        return;
    }
    if !out.is_empty() && !out.last().is_some_and(|l| l.is_empty()) {
        out.push(String::new());
    }
    for rule in rules {
        let line = if rule.rule_name == rule_names::UNDER_BATCHING && verbose_rules {
            match r1_context.as_ref() {
                Some(R1VerboseContext {
                    snapshot,
                    max_num_seqs,
                    efficiency_pct,
                    config_relative_efficiency_pct,
                    prefill_time_fraction,
                }) => r1_verbose_miss_line(
                    snapshot,
                    *max_num_seqs,
                    *efficiency_pct,
                    *config_relative_efficiency_pct,
                    *prefill_time_fraction,
                ),
                None => format!("{}: not triggered", rule.label),
            }
        } else if rule.rule_name == rule_names::PREFILL_BOUND && verbose_rules {
            r6_verbose_miss_line(
                r1_context.as_ref().and_then(|c| c.prefill_time_fraction),
                baseline.and_then(|b| b.efficiency_pct),
            )
            .unwrap_or_else(|| format!("{}: not triggered", rule.label))
        } else {
            format!("{}: not triggered", rule.label)
        };
        out.push(line);
    }
}

fn not_triggered_from_fired_names(
    fired_names: &std::collections::HashSet<&'static str>,
    suppressed_rules: &[&'static str],
    r2_adv_present: bool,
    r4_adv_present: bool,
) -> Vec<&'static NotTriggeredRule> {
    let suppressed = |name: &str| suppressed_rules.contains(&name);
    let mut rules = Vec::new();
    for entry in NOT_TRIGGERED_SINGLES {
        if entry.rule_name == rule_names::OOM_RISK && r4_adv_present {
            continue;
        }
        if !fired_names.contains(entry.rule_name) && !suppressed(entry.rule_name) {
            rules.push(entry);
        }
    }
    if !KV_RULE_NAMES.iter().any(|n| fired_names.contains(n))
        && !KV_RULE_NAMES.iter().any(|n| suppressed(n))
        && !r2_adv_present
    {
        rules.push(&NOT_TRIGGERED_KV);
    }
    rules
}

pub fn format_diagnose_rules(
    input: AnalysisInput<'_>,
    report: &super::Report,
    verbose_rules: bool,
    metrics_url: &str,
) -> Vec<String> {
    let snapshot = &input.window.snapshot;
    if !window_is_evaluable(snapshot) {
        return no_evaluable_diagnose_lines(verbose_rules, std::slice::from_ref(input.window));
    }

    let any_issue = !report.groups.is_empty();
    let baseline_ref = report.baseline.as_ref();
    let tps = snapshot.vllm.generation_tokens_per_sec;

    let fired_names: std::collections::HashSet<&'static str> = report
        .groups
        .iter()
        .flat_map(|g| {
            std::iter::once(g.primary.rule_name).chain(g.secondary.iter().map(|r| r.rule_name))
        })
        .collect();

    let mut out = Vec::new();

    for g in &report.groups {
        append_display_block(&mut out, rule_display_block(g));
    }

    append_waste_line(&mut out, &report.groups, baseline_ref, tps);

    let advisories = collect_advisories(
        &fired_names,
        snapshot,
        metrics_url,
        report.baseline.as_ref().and_then(|b| b.kv_headroom_gb),
        input.ctx.gpu.vram_gb,
        report.baseline.as_ref().map(|b| b.weight_gb),
    );
    let any_advisory = advisories.any();
    append_display_block(&mut out, advisories.lines);

    let not_fired = not_triggered_from_fired_names(
        &fired_names,
        &report.suppressed_rules,
        advisories.r2_present,
        advisories.r4_present,
    );
    if verbose_rules {
        append_not_triggered_lines(
            &mut out,
            &not_fired,
            verbose_rules,
            Some(R1VerboseContext {
                snapshot,
                max_num_seqs: input.ctx.config.max_num_seqs,
                efficiency_pct: baseline_ref.and_then(|b| b.efficiency_pct),
                config_relative_efficiency_pct: baseline_ref
                    .and_then(|b| b.config_relative_efficiency_pct),
                prefill_time_fraction: resolve_prefill_time_fraction(baseline_ref, snapshot),
            }),
            baseline_ref,
        );
    }

    if !any_issue && !any_advisory && !verbose_rules {
        out.push(NO_ISSUES_LINE.to_string());
    }
    if let Some(line) = kv_ceiling_unknown_verbose_line(report.kv_max_seqs, verbose_rules) {
        append_display_block(&mut out, vec![line]);
    }

    trim_trailing_blank_lines(&mut out);
    out
}

struct WindowRuleEval {
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
    // Average hit rate across ALL evaluable windows - not just windows where r3
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

        match rule1_under_batching_with_efficiency(
            snap,
            summary.ctx.config.max_num_seqs,
            summary_efficiency_pct,
            win_baseline
                .as_ref()
                .and_then(|b| b.config_relative_efficiency_pct),
            resolve_prefill_time_fraction(win_baseline.as_ref(), snap),
        ) {
            Rule1Outcome::Fired(d) => {
                eval.r1_fired += 1;
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

        match r6_evaluate(
            resolve_prefill_time_fraction(win_baseline.as_ref(), snap),
            win_baseline.as_ref().and_then(|b| b.efficiency_pct),
            win_baseline.as_ref().and_then(|b| b.prefill_efficiency_pct),
            snap,
            summary.ctx.config.enable_chunked_prefill,
        ) {
            Rule6Outcome::Fired(d) => {
                eval.r6_fired += 1;
                eval.r6_details.push(d);
            }
            Rule6Outcome::NotFired => {}
        }

        let ridge = win_baseline.as_ref().map(|b| b.ridge_batch_size);
        let win_kv_max = compute_kv_max_seqs(
            win_baseline.as_ref().and_then(|b| b.kv_headroom_gb),
            summary.ctx.config.max_model_len,
            &summary.ctx.model,
            summary.ctx.config.kv_cache_dtype.as_deref(),
            effective_tensor_parallel(
                summary.ctx.config.tensor_parallel_size,
                w.snapshot.collected_gpu_count(),
            ),
        );
        if let Some(d) =
            rule7_config_headroom(snap, summary.ctx.config.max_num_seqs, ridge, win_kv_max)
        {
            eval.r7_fired += 1;
            eval.r7_details.push(d);
        }
    }

    eval.skipped = skipped;
    Some(eval)
}

// session_hit_rate: all-evaluable-windows average hit rate for display in r3 recommendation body.
// Caller must compute this from the full window slice - not from r3-fired windows only.
// Pass None on the single-window path (no session to average).
fn build_report_from_eval(
    eval: &WindowRuleEval,
    summary: AnalysisInput<'_>,
    session_hit_rate: Option<f64>,
    baseline: Option<baseline::PhysicsBaseline>,
) -> super::Report {
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
        return super::Report {
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
        let display_lines = format_under_batching_window_issue(
            &d,
            pct(eval.r1_fired, eval.n_eval),
            summary_snap,
            confidence,
        );
        recs.push(Recommendation {
            rule_name: rule_names::UNDER_BATCHING,
            layer: 4,
            impact: 4,
            confidence,
            action: "Batch more requests or increase client concurrency".to_string(),
            short_action: r1_short_action(d.running, d.max_num_seqs),
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
        let conf = if d.kv_affordable_seqs.is_some() {
            0.8
        } else {
            0.6
        };
        let display_lines =
            format_config_headroom_window_issue(&d, pct(eval.r7_fired, eval.n_eval));
        recs.push(Recommendation {
            rule_name: rule_names::CONFIG_HEADROOM,
            layer: 3,
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
                qps,
                prompt_mean,
                session_hit_rate,
            ),
        });
    }

    if eval.r6_significant() {
        let d = aggregate_r6_detail(&eval.r6_details);
        let sev = r6_severity(d.prefill_time_fraction);
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
) -> super::Report {
    let mut suppressed_rules = Vec::new();
    let Some(min_layer) = recs.iter().map(|r| r.layer).min() else {
        return super::Report {
            baseline,
            groups: Vec::new(),
            suppressed_rules,
            kv_max_seqs,
            n_eval,
            skipped,
        };
    };

    let mut recs: Vec<Recommendation> = recs
        .into_iter()
        .filter(|r| {
            if r.layer == min_layer {
                true
            } else {
                suppressed_rules.push(r.rule_name);
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
                suppressed_rules.push(suppressed);
            }
        }
    }

    recs.sort_by(|a, b| {
        let sa = a.impact as f64 * a.confidence;
        let sb = b.impact as f64 * b.confidence;
        sb.total_cmp(&sa)
    });

    let groups = {
        let mut iter = recs.into_iter();
        match iter.next() {
            None => Vec::new(),
            Some(primary) => vec![IssueGroup {
                primary,
                secondary: iter.collect(),
            }],
        }
    };

    super::Report {
        baseline,
        groups,
        suppressed_rules,
        kv_max_seqs,
        n_eval,
        skipped,
    }
}

/// Multi-window rule evaluation - same significance gates as `format_diagnose_rules_for_windows`.
pub fn build_report_for_windows(
    windows: &[RuntimeWindow],
    summary: AnalysisInput<'_>,
) -> super::Report {
    let baseline = baseline::compute(&summary);
    let summary_efficiency_pct = baseline.as_ref().and_then(|b| b.efficiency_pct);
    let Some(eval) = eval_window_rules(windows, &summary, summary_efficiency_pct) else {
        return super::Report {
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

pub fn format_diagnose_rules_for_windows(
    windows: &[RuntimeWindow],
    summary: AnalysisInput<'_>,
    report: &super::Report,
    verbose_rules: bool,
    metrics_url: &str,
) -> Vec<String> {
    if report.n_eval == 0 {
        return no_evaluable_diagnose_lines(verbose_rules, windows);
    }

    if report.n_eval < ENGINE_MIN_PERSISTENT_WINDOWS {
        let mut out = vec![
            "[!] Insufficient Sustained Load".to_string(),
            String::new(),
            format!(
                "  Traffic detected but too brief for reliable diagnosis. \
                 Required: {} evaluable windows. Captured: {}{}.",
                ENGINE_MIN_PERSISTENT_WINDOWS,
                report.n_eval,
                if report.skipped > 0 {
                    format!(" ({} windows dropped)", report.skipped)
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

    let total = report.n_eval + report.skipped;
    let skipped = report.skipped;

    let summary_snap = &summary.window.snapshot;
    let baseline_ref = report.baseline.as_ref();
    let tps = summary_snap.vllm.generation_tokens_per_sec;

    if report.groups.is_empty() {
        let mut out = Vec::new();
        let advisories = collect_advisories(
            &std::collections::HashSet::new(),
            summary_snap,
            metrics_url,
            report.baseline.as_ref().and_then(|b| b.kv_headroom_gb),
            summary.ctx.gpu.vram_gb,
            report.baseline.as_ref().map(|b| b.weight_gb),
        );
        let any_advisory = advisories.any();
        append_display_block(&mut out, advisories.lines);
        if verbose_rules {
            let not_fired = not_triggered_from_fired_names(
                &std::collections::HashSet::new(),
                &[],
                advisories.r2_present,
                advisories.r4_present,
            );
            append_not_triggered_lines(
                &mut out,
                &not_fired,
                verbose_rules,
                Some(R1VerboseContext {
                    snapshot: summary_snap,
                    max_num_seqs: summary.ctx.config.max_num_seqs,
                    efficiency_pct: report.baseline.as_ref().and_then(|b| b.efficiency_pct),
                    config_relative_efficiency_pct: report
                        .baseline
                        .as_ref()
                        .and_then(|b| b.config_relative_efficiency_pct),
                    prefill_time_fraction: resolve_prefill_time_fraction(
                        report.baseline.as_ref(),
                        summary_snap,
                    ),
                }),
                report.baseline.as_ref(),
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

    let fired_names: std::collections::HashSet<&'static str> = report
        .groups
        .iter()
        .flat_map(|g| {
            std::iter::once(g.primary.rule_name).chain(g.secondary.iter().map(|r| r.rule_name))
        })
        .collect();

    let mut warnings: Vec<String> = Vec::new();
    for g in &report.groups {
        if !warnings.is_empty() && !warnings.last().is_some_and(|l| l.is_empty()) {
            warnings.push(String::new());
        }
        warnings.extend(rule_display_block(g));
        warnings.push(String::new());
    }

    append_waste_line(&mut warnings, &report.groups, baseline_ref, tps);

    let advisories = collect_advisories(
        &fired_names,
        summary_snap,
        metrics_url,
        report.baseline.as_ref().and_then(|b| b.kv_headroom_gb),
        summary.ctx.gpu.vram_gb,
        report.baseline.as_ref().map(|b| b.weight_gb),
    );
    let any_advisory = advisories.any();
    let mut out = warnings;
    append_display_block(&mut out, advisories.lines);

    if let Some(line) = kv_ceiling_unknown_verbose_line(report.kv_max_seqs, verbose_rules) {
        append_display_block(&mut out, vec![line]);
    }

    let not_fired = not_triggered_from_fired_names(
        &fired_names,
        &report.suppressed_rules,
        advisories.r2_present,
        advisories.r4_present,
    );
    let any_warning = !report.groups.is_empty();
    if verbose_rules {
        append_not_triggered_lines(
            &mut out,
            &not_fired,
            verbose_rules,
            Some(R1VerboseContext {
                snapshot: summary_snap,
                max_num_seqs: summary.ctx.config.max_num_seqs,
                efficiency_pct: report.baseline.as_ref().and_then(|b| b.efficiency_pct),
                config_relative_efficiency_pct: report
                    .baseline
                    .as_ref()
                    .and_then(|b| b.config_relative_efficiency_pct),
                prefill_time_fraction: resolve_prefill_time_fraction(
                    report.baseline.as_ref(),
                    summary_snap,
                ),
            }),
            report.baseline.as_ref(),
        );
    }
    if !any_warning && !any_advisory && !verbose_rules {
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
    fn format_diagnose_rules_test(
        input: AnalysisInput<'_>,
        verbose: bool,
        metrics_url: &str,
    ) -> Vec<String> {
        let report = super::super::build_report(input);
        super::format_diagnose_rules(input, &report, verbose, metrics_url)
    }

    fn format_diagnose_rules_for_windows_test(
        windows: &[RuntimeWindow],
        summary: AnalysisInput<'_>,
        verbose: bool,
        metrics_url: &str,
    ) -> Vec<String> {
        let report = super::super::build_report_for_windows(windows, summary);
        super::format_diagnose_rules_for_windows(windows, summary, &report, verbose, metrics_url)
    }

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
            gpus: vec![gpu],
            nvml_host_gpu_count: None,
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
        let with_kv_layers =
            compute_kv_max_seqs(Some(headroom_gb), Some(4096), &hybrid, None, None);

        let dense = crate::context::ModelArch {
            num_kv_layers: None, // pure-attention: all 64 layers count
            ..hybrid
        };
        let without_kv_layers =
            compute_kv_max_seqs(Some(headroom_gb), Some(4096), &dense, None, None);

        assert!(with_kv_layers.is_some() && without_kv_layers.is_some());
        // 32 KV layers → half the bytes per token → fits 2× as many seqs
        assert_eq!(with_kv_layers.unwrap(), without_kv_layers.unwrap() * 2);
    }

    #[test]
    fn compute_kv_max_seqs_tp2_doubles_capacity() {
        let model = crate::context::ModelArch {
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_layers: Some(32),
            ..Default::default()
        };
        let headroom_gb = 20.0;
        let tp1 = compute_kv_max_seqs(Some(headroom_gb), Some(4096), &model, None, Some(1));
        let tp2 = compute_kv_max_seqs(Some(headroom_gb), Some(4096), &model, None, Some(2));
        assert_eq!(tp2.unwrap(), tp1.unwrap() * 2);
    }

    #[test]
    fn compute_kv_max_seqs_tp_greater_than_kv_heads_no_benefit() {
        let model = crate::context::ModelArch {
            num_kv_heads: Some(2),
            head_dim: Some(128),
            num_layers: Some(32),
            ..Default::default()
        };
        let headroom_gb = 20.0;
        let tp2 = compute_kv_max_seqs(Some(headroom_gb), Some(4096), &model, None, Some(2));
        let tp4 = compute_kv_max_seqs(Some(headroom_gb), Some(4096), &model, None, Some(4));
        assert_eq!(tp2, tp4);
    }

    #[test]
    fn compute_kv_max_seqs_tp_none_uses_full_heads() {
        let model = crate::context::ModelArch {
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_layers: Some(32),
            ..Default::default()
        };
        let headroom_gb = 20.0;
        let none = compute_kv_max_seqs(Some(headroom_gb), Some(4096), &model, None, None);
        let one = compute_kv_max_seqs(Some(headroom_gb), Some(4096), &model, None, Some(1));
        assert_eq!(none, one);
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
            gpus: vec![GpuRawMetrics {
                gpu_name: Some("NVIDIA H100 80GB HBM3".to_string()),
                vram_total_mb: Some(80 * 1024),
                gpu_util_pct: Some(58.0),
                ..Default::default()
            }],

            nvml_host_gpu_count: None,
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
        let r = r1_recommendation(&win.snapshot, None, None, None, None).expect("r1 fired");
        assert_eq!(r.rule_name, rule_names::UNDER_BATCHING);
        assert_eq!(r.impact, 4);
        assert!((r.confidence - 0.5).abs() < 1e-9);
        match rule1_under_batching_with_efficiency(&win.snapshot, None, None, None, None) {
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
        assert!(r1_recommendation(&win.snapshot, None, None, None, None).is_none());
    }

    #[test]
    fn waiting_at_two_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_waiting = Some(2.0);
        v.tpot_ms = Some(35.0);
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot, None, None, None, None).is_none());
    }

    #[test]
    fn running_at_occupancy_threshold_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(64.0);
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot, None, None, None, None).is_none());
    }

    #[test]
    fn max_seqs_zero_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.max_num_seqs = Some(0);
        v.tpot_ms = Some(35.0);
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot, None, None, None, None).is_none());
    }

    #[test]
    fn nan_running_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(f64::NAN);
        v.tpot_ms = Some(35.0);
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot, None, None, None, None).is_none());
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
        let lines =
            format_diagnose_rules_test(ai(&ctx, &win), false, "http://127.0.0.1:8000/metrics");
        let text = lines.join("\n");
        assert!(text.contains("[!] Under-batching: Insufficient Concurrency"));
        assert!(text.contains("Occupancy"));
        assert!(text.contains("threshold: < 25%"));
        assert!(text.contains("  Cause:"));
        assert!(text.contains("under-fed by client"));
        assert!(
            text.contains(
                "    • Batch more requests or increase client concurrency (251 slots idle)"
            )
        );
        assert!(text.contains("Expected: Higher throughput, stable TPOT."));
        assert!(
            text.contains("Confidence: Low"),
            "unknown GPU path uses low confidence: {text}"
        );
        assert!(text.contains("low confidence"));
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
        let report = super::build_report_for_windows(&windows, summary);
        assert_eq!(report.groups[0].primary.rule_name, rule_names::OOM_RISK);
        assert!(
            !report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::KV_CACHE_PRESSURE)
        );
        assert!(
            report
                .suppressed_rules
                .contains(&rule_names::KV_CACHE_PRESSURE)
        );
    }

    #[test]
    fn format_diagnose_verbose_omits_kv_pressure_when_r4_fires() {
        let (ctx, win) = input_r4_suppresses_r2();
        let report = crate::engine::build_report(ai(&ctx, &win));
        assert_eq!(report.groups[0].primary.rule_name, rule_names::OOM_RISK);
        assert!(
            !report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::KV_CACHE_PRESSURE)
        );
        let text =
            format_diagnose_rules_test(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics")
                .join("\n");
        assert!(!text.contains("KV cache pressure: not triggered"));
    }

    #[test]
    fn format_diagnose_non_verbose_omits_kv_pressure_when_r4_fires() {
        let (ctx, win) = input_r4_suppresses_r2();
        let report = crate::engine::build_report(ai(&ctx, &win));
        assert_eq!(report.groups[0].primary.rule_name, rule_names::OOM_RISK);
        assert!(
            !report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::KV_CACHE_PRESSURE)
        );
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
            format_diagnose_rules_test(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics")
                .join("\n");
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
            format_diagnose_rules_test(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics")
                .join("\n");
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

    fn r2_report(windows: &[RuntimeWindow]) -> crate::engine::Report {
        let ctx = mk_ctx();
        let summary = ai(&ctx, windows.last().expect("windows"));
        build_report_for_windows(windows, summary)
    }

    fn r2_issue_lines(windows: Vec<RuntimeWindow>) -> Vec<String> {
        let ctx = mk_ctx();
        let summary = ai(&ctx, windows.last().expect("windows"));
        format_diagnose_rules_for_windows_test(
            &windows,
            summary,
            false,
            "http://127.0.0.1:8000/metrics",
        )
    }

    #[test]
    fn r2_recommendation_confidence_from_density_counts() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_high_kv();
        v.num_preemptions_per_sec = Some(0.05);
        let s = snap(t, t, v, gpu_low());
        let r = r2_recommendation(&s, None, None, None, 1, 4, false).expect("fired");
        assert_eq!(r.rule_name, rule_names::KV_CACHE_PRESSURE);
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
        let r = r2_recommendation(&s, None, None, None, 1, 1, false).expect("fired");
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
            format_diagnose_rules_test(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics")
                .join("\n");
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
        let r2_text = r2_recommendation(&win_kv_only.snapshot, None, None, None, 1, 1, false)
            .expect("r2 fired")
            .display_lines
            .join("\n");
        assert!(!r2_text.contains("Confidence:"));
        let text = format_diagnose_rules_test(
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
            format_diagnose_rules_test(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics")
                .join("\n");
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
        assert_eq!(r.rule_name, rule_names::LOW_PREFIX_REUSE);
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
            format_diagnose_rules_test(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics")
                .join("\n");
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
            format_diagnose_rules_test(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics")
                .join("\n");
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
        let lines =
            format_diagnose_rules_test(ai(&ctx, &win), false, "http://127.0.0.1:8000/metrics");
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
        let lines =
            format_diagnose_rules_test(ai(&ctx, &win), false, "http://127.0.0.1:8000/metrics");
        let idx_kv = lines
            .iter()
            .position(|l| l.contains("[!] KV Cache Pressure"))
            .expect("rule2");
        assert!(
            !lines.iter().any(|l| l.contains("[!] Under-batching")),
            "layer 2 suppresses layer 4: {lines:?}"
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
        assert!(waste_lines[0].contains("lost to memory thrashing"));
        let _ = idx_kv;
    }

    #[test]
    fn waste_label_r1_only() {
        assert_eq!(
            waste_label_suffix(&[rule_names::UNDER_BATCHING]),
            Some("wasted on idle compute")
        );
    }

    #[test]
    fn waste_label_r2_only() {
        assert_eq!(
            waste_label_suffix(&[rule_names::KV_CACHE_PRESSURE]),
            Some("lost to memory thrashing")
        );
    }

    #[test]
    fn waste_label_r3_only() {
        assert_eq!(
            waste_label_suffix(&[rule_names::LOW_PREFIX_REUSE]),
            Some("wasted on redundant prefill")
        );
    }

    #[test]
    fn waste_label_r5_only() {
        assert_eq!(
            waste_label_suffix(&[rule_names::CONCURRENCY_SATURATION]),
            Some("lost to scheduler queuing")
        );
    }

    #[test]
    fn waste_label_multi_rule() {
        assert_eq!(
            waste_label_suffix(&[rule_names::UNDER_BATCHING, rule_names::KV_CACHE_PRESSURE]),
            Some("lost to compounding bottlenecks")
        );
    }

    #[test]
    fn waste_label_unknown_rule() {
        assert_eq!(
            waste_label_suffix(&[rule_names::OOM_RISK]),
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
        let report = r2_report(&windows);
        assert!(
            !report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::KV_CACHE_PRESSURE)
        );
    }

    #[test]
    fn r2_fires_on_two_critical_kv_windows_without_preemptions() {
        let mut windows: Vec<_> = (0..10)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        windows[0] = mk_evaluable_kv_window(96.0, false);
        windows[1] = mk_evaluable_kv_window(97.0, false);
        let report = r2_report(&windows);
        assert!(
            !report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::KV_CACHE_PRESSURE)
        );
    }

    #[test]
    fn r2_does_not_fire_when_kv_high_but_tpot_stable_and_no_preemptions() {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        for w in windows.iter_mut().take(4) {
            *w = mk_evaluable_kv_window(89.0, false);
        }
        let report = r2_report(&windows);
        assert!(
            !report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::KV_CACHE_PRESSURE)
        );
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
            .find(|g| g.primary.rule_name == rule_names::KV_CACHE_PRESSURE)
            .expect("r2 group");
        assert!((r2.primary.confidence - (4.0 / 15.0)).abs() < 1e-9);
        let text = format_diagnose_rules_for_windows_test(
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
        let text = format_diagnose_rules_for_windows_test(
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
        let text = format_diagnose_rules_for_windows_test(
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
            .find(|g| g.primary.rule_name == rule_names::KV_ADMISSION_BACKLOG)
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
            max_num_seqs: Some(256),
            ..Default::default()
        };
        let mut windows = Vec::new();
        for _i in 0..10 {
            let mut v = vllm_base();
            v.max_num_seqs = Some(256);
            v.num_requests_waiting = Some(1.0);
            v.kv_cache_usage_perc = Some(71.2);
            v.prefix_cache_hit_rate = Some(0.524);
            v.prompt_tokens_mean = Some(128.0);
            v.generation_tokens_per_sec = Some(1580.0);
            v.num_requests_running = Some(3.2);
            v.tpot_ms = Some(35.0);
            let mut g = gpu_busy();
            g.gpu_util_pct = Some(50.0);
            g.power_watts = Some(312.0);
            g.vram_used_mb = Some(62 * 1024);
            g.vram_total_mb = Some(80 * 1024);
            v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
            g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
            windows.push(mk_win(snap(t, t, v, g)));
        }
        let ctx = StaticContext::from_snapshot(&windows[0].snapshot, cfg);
        let summary = ai(&ctx, windows.last().expect("summary source"));
        let lines = format_diagnose_rules_for_windows_test(
            &windows,
            summary,
            false,
            "http://127.0.0.1:8000/metrics",
        );
        let text = lines.join("\n");
        assert!(text.contains("Under-batching: Insufficient Concurrency"));
        assert!(text.contains("Seen in 100% of windows"));
        assert!(text.contains("Config efficiency"));
        assert!(text.contains("threshold: < 60%"));
        assert!(text.contains("  Cause:"));
        assert!(
            text.contains("Batch more requests or increase client concurrency (253 slots idle)")
        );
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
        let text = format_diagnose_rules_for_windows_test(
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
        let summary = ai(&ctx, windows.last().expect("windows"));
        let lines = format_diagnose_rules_for_windows_test(
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
        let lines =
            format_diagnose_rules_test(ai(&ctx, &win), false, "http://127.0.0.1:8000/metrics");
        assert_eq!(
            lines,
            no_evaluable_diagnose_lines(false, std::slice::from_ref(&win))
        );
        let vlines =
            format_diagnose_rules_test(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics");
        assert!(
            vlines
                .iter()
                .any(|l| l.contains("1 of 1 collected windows"))
        );
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
        let text = format_diagnose_rules_for_windows_test(
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
                .any(|g| g.primary.rule_name == rule_names::CONCURRENCY_SATURATION),
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
                .any(|g| g.primary.rule_name == rule_names::CONCURRENCY_SATURATION),
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
        let report = build_report_for_windows(&windows, summary);
        assert!(
            report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::KV_CACHE_PRESSURE)
        );
        assert!(
            !report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::CONCURRENCY_SATURATION)
        );
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
        let text = format_diagnose_rules_for_windows_test(
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
        // Fix line uses summary snapshot KV (50%) for branch selection - scale-out, not session peak.
        assert!(
            text.contains("Raise --max-num-seqs above 32"),
            "expected raise-cap fix from summary KV: {text}"
        );
        assert!(!text.contains("KV at 95%: scheduler at cap, pool full."));
        assert!(!text.contains("Add a replica"));
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
            .find(|g| g.primary.rule_name == rule_names::CONCURRENCY_SATURATION)
            .expect("r5 group");
        let text = r5.primary.display_lines.join("\n");
        assert!(
            text.contains("Raise --max-num-seqs above 32"),
            "display fix line must use summary snapshot KV branch: {text}"
        );
        assert!(!text.contains("KV at 95%: scheduler at cap, pool full."));
        assert!(!text.contains("Add a replica"));
        assert!(!text.contains("KV pool has room (70%)"));
        // action still uses aggregate session peak from eval.
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
        let lines = format_diagnose_rules_for_windows_test(
            &windows,
            summary,
            false,
            "http://127.0.0.1:8000/metrics",
        );
        assert_eq!(lines, no_evaluable_diagnose_lines(false, &windows));
        let summary2 = ai(&ctx, &windows[0]);
        let vlines = format_diagnose_rules_for_windows_test(
            &windows,
            summary2,
            true,
            "http://127.0.0.1:8000/metrics",
        );
        assert!(
            vlines
                .iter()
                .any(|l| l.contains("2 of 2 collected windows"))
        );
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
            prefill_efficiency_pct: None,
            prefill_time_fraction: None,
            config_relative_efficiency_pct: None,
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
                vec![issue_group(rule_names::UNDER_BATCHING)],
                "wasted on idle compute",
            ),
            (
                vec![issue_group(rule_names::KV_CACHE_PRESSURE)],
                "lost to memory thrashing",
            ),
            (
                vec![issue_group(rule_names::LOW_PREFIX_REUSE)],
                "wasted on redundant prefill",
            ),
            (
                vec![issue_group(rule_names::CONCURRENCY_SATURATION)],
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
            issue_group(rule_names::UNDER_BATCHING),
            issue_group(rule_names::KV_CACHE_PRESSURE),
        ];
        let mut lines = vec!["issue".to_string()];
        append_waste_line(&mut lines, &groups, Some(&b), Some(14.2));
        assert!(
            lines
                .iter()
                .any(|l| l.contains("lost to compounding bottlenecks"))
        );
    }

    #[test]
    fn waste_line_unknown_rule_name_unclassified() {
        let groups = vec![issue_group(rule_names::OOM_RISK)];

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
                layer: 4,
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
    fn dag_layer2_suppresses_layer4_when_r2_fires() {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        for w in windows.iter_mut().take(4) {
            *w = {
                let mut win = mk_evaluable_kv_window(89.0, true);
                win.snapshot.vllm.num_requests_running = Some(3.1);
                win.snapshot.vllm.num_requests_waiting = Some(0.0);
                win.snapshot.vllm.tpot_ms = Some(35.0);
                win.snapshot.gpus[0].gpu_util_pct = Some(58.0);
                win
            };
        }
        let ctx = mk_ctx();
        let summary = ai(&ctx, windows.last().expect("windows"));
        let report = build_report_for_windows(&windows, summary);
        assert!(
            report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::KV_CACHE_PRESSURE)
        );
        assert!(
            !report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::UNDER_BATCHING)
        );
    }

    #[test]
    fn dag_layer2_suppresses_layer3_when_r2_fires() {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_concurrency_saturation_window(32.0, 15.0, 32))
            .collect();
        for w in windows.iter_mut().take(4) {
            *w = mk_evaluable_kv_window(89.0, true);
        }
        let ctx = mk_ctx();
        let summary = ai(&ctx, windows.last().expect("windows"));
        let report = build_report_for_windows(&windows, summary);
        assert!(
            report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::KV_CACHE_PRESSURE)
        );
        assert!(
            !report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::CONCURRENCY_SATURATION)
        );
    }

    #[test]
    fn waste_line_efficiency_over_100_omitted() {
        let b = baseline_for_waste(110.0, CostSource::Catalog, 1.84);
        let mut lines = vec!["issue".to_string()];
        append_waste_line(
            &mut lines,
            &[issue_group(rule_names::UNDER_BATCHING)],
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
            &[issue_group(rule_names::UNDER_BATCHING)],
            Some(&b),
            Some(10.0),
        );
        assert_eq!(lines.len(), 1);

        b.efficiency_pct = Some(32.0);
        b.cost = None;
        append_waste_line(
            &mut lines,
            &[issue_group(rule_names::UNDER_BATCHING)],
            Some(&b),
            Some(10.0),
        );
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn model_len_suggestion_uses_p99_sum_when_count_sufficient() {
        let mut lines = Vec::new();
        push_model_len_shrink_suggestion(
            &mut lines,
            Some(8192),
            Some(6000.0),
            Some(450.0),
            150.0,
            "    ",
        );
        let text = lines.join("\n");
        assert!(text.contains("to ~6450"));
        assert!(text.contains("prompt p99 6000 tok + output p99 450 tok"));
        assert!(text.contains("Truncation risk"));
    }

    #[test]
    fn model_len_suggestion_no_op_when_count_below_threshold() {
        let mut lines = Vec::new();
        push_model_len_shrink_suggestion(
            &mut lines,
            Some(8192),
            Some(6000.0),
            Some(450.0),
            50.0,
            "    ",
        );
        let text = lines.join("\n");
        assert!(text.contains("to safely raise concurrency"));
        assert!(!text.contains("to ~"));
    }

    #[test]
    fn model_len_suggestion_no_op_when_p99_missing() {
        let mut lines = Vec::new();
        push_model_len_shrink_suggestion(&mut lines, Some(8192), Some(6000.0), None, 150.0, "    ");
        let text = lines.join("\n");
        assert!(text.contains("to safely raise concurrency"));
        assert!(!text.contains("to ~"));
    }

    #[test]
    fn model_len_suggestion_suppressed_when_delta_below_5pct() {
        let mut lines = Vec::new();
        push_model_len_shrink_suggestion(
            &mut lines,
            Some(5464),
            Some(5400.0),
            Some(65.0),
            150.0,
            "    ",
        );
        assert!(lines.is_empty());
    }

    fn mk_llama8b_h100_ctx(s: &RawSnapshot) -> StaticContext {
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(8192),
            max_num_seqs: Some(256),
            ..Default::default()
        };
        StaticContext::from_snapshot(s, cfg)
    }

    fn mk_r7_headroom_window(
        running: f64,
        max_num_seqs: u32,
        waiting: f64,
        tps: f64,
    ) -> RuntimeWindow {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
        v.generation_tokens_per_sec = Some(tps);
        v.num_requests_running = Some(running);
        v.num_requests_waiting = Some(waiting);
        v.max_num_seqs = Some(max_num_seqs);
        v.window_duration_secs = Some(2.0);
        let mut g = gpu_busy();
        g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
        mk_win(snap(t, t, v, g))
    }

    fn mk_r7_ctx(max_num_seqs: u32) -> StaticContext {
        let snap = mk_r7_headroom_window(5.0, max_num_seqs, 0.0, 50.0).snapshot;
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(8192),
            max_num_seqs: Some(max_num_seqs),
            ..Default::default()
        };
        StaticContext::from_snapshot(&snap, cfg)
    }

    #[test]
    fn r7_suppresses_r1() {
        let windows: Vec<_> = (0..10)
            .map(|_| mk_r7_headroom_window(20.0, 32, 0.0, 10.0))
            .collect();
        let ctx = mk_r7_ctx(32);
        let summary = ai(&ctx, windows.last().expect("windows"));
        let report = build_report_for_windows(&windows, summary);
        assert!(
            report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::CONFIG_HEADROOM)
        );
        assert!(
            !report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::UNDER_BATCHING)
        );
        assert!(
            report
                .suppressed_rules
                .contains(&rule_names::UNDER_BATCHING)
        );
    }

    #[test]
    fn r7_silent_when_waiting_nonzero_r5_territory() {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        for w in windows.iter_mut().take(6) {
            *w = mk_evaluable_concurrency_saturation_window(32.0, 15.0, 32);
        }
        let ctx = mk_r7_ctx(32);
        let summary = ai(&ctx, windows.last().expect("windows"));
        let report = build_report_for_windows(&windows, summary);
        assert!(
            report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::CONCURRENCY_SATURATION)
        );
        assert!(
            !report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::CONFIG_HEADROOM)
        );
    }

    #[test]
    fn r7_fires_as_primary_when_alone() {
        let windows: Vec<_> = (0..10)
            .map(|_| mk_r7_headroom_window(20.0, 32, 0.0, 50.0))
            .collect();
        let ctx = mk_r7_ctx(32);
        let summary = ai(&ctx, windows.last().expect("windows"));
        let report = build_report_for_windows(&windows, summary);
        assert_eq!(report.groups.len(), 1);
        assert_eq!(
            report.groups[0].primary.rule_name,
            rule_names::CONFIG_HEADROOM
        );
    }

    fn mk_r6_prefill_window(prefill_fraction: f64, tps: f64, running: f64) -> RuntimeWindow {
        use crate::collectors::HistogramWindowMass;
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
        v.generation_tokens_per_sec = Some(tps);
        v.num_requests_running = Some(running);
        v.num_requests_waiting = Some(0.0);
        let window_secs = 2.0;
        let mean_prefill = prefill_fraction * window_secs;
        v.prefill_window_mass = Some(HistogramWindowMass {
            sum_delta: mean_prefill * 4.0,
            count_delta: 4.0,
        });
        v.window_duration_secs = Some(window_secs);
        v.prompt_tokens_mean = Some(2048.0);
        v.cache_config.enable_prefix_caching = Some(false);
        let mut g = gpu_busy();
        g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
        mk_win(snap(t, t, v, g))
    }

    #[test]
    fn r6_suppressed_when_r1_fires() {
        // Prefill below R1 physics gate (0.30): R1 owns under-batching; R6 stays quiet.
        let windows: Vec<_> = (0..10)
            .map(|_| mk_r6_prefill_window(0.25, 10.0, 5.0))
            .collect();
        let s = windows[0].snapshot.clone();
        let ctx = mk_llama8b_h100_ctx(&s);
        let summary = ai(&ctx, windows.last().expect("windows"));
        let report = build_report_for_windows(&windows, summary);
        assert!(
            report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::UNDER_BATCHING)
        );
        assert!(
            !report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::PREFILL_BOUND)
        );
    }

    #[test]
    fn r6_fires_when_r1_prefill_gate_suppresses_r1() {
        let windows: Vec<_> = (0..10)
            .map(|_| mk_r6_prefill_window(0.55, 10.0, 5.0))
            .collect();
        let s = windows[0].snapshot.clone();
        let ctx = mk_llama8b_h100_ctx(&s);
        let summary = ai(&ctx, windows.last().expect("windows"));
        let report = build_report_for_windows(&windows, summary);
        assert!(
            report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::PREFILL_BOUND)
        );
    }

    #[test]
    fn r6_not_primary_when_r2_outscores() {
        let mut windows: Vec<_> = (0..10)
            .map(|_| mk_r6_prefill_window(0.55, 10.0, 50.0))
            .collect();
        for w in windows.iter_mut().take(4) {
            *w = mk_evaluable_kv_window(89.0, true);
        }
        let s = windows[0].snapshot.clone();
        let ctx = mk_llama8b_h100_ctx(&s);
        let summary = ai(&ctx, windows.last().expect("windows"));
        let report = build_report_for_windows(&windows, summary);
        assert!(
            report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::KV_CACHE_PRESSURE)
        );
        assert!(
            !report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::PREFILL_BOUND)
        );
    }

    #[test]
    fn r6_fires_as_primary_when_no_other_rules() {
        let windows: Vec<_> = (0..10)
            .map(|_| mk_r6_prefill_window(0.55, 10.0, 50.0))
            .collect();
        let s = windows[0].snapshot.clone();
        let ctx = mk_llama8b_h100_ctx(&s);
        let summary = ai(&ctx, windows.last().expect("windows"));
        let report = build_report_for_windows(&windows, summary);
        assert_eq!(
            report.groups[0].primary.rule_name,
            rule_names::PREFILL_BOUND
        );
    }

    #[test]
    fn r6_verbose_miss_line_when_not_triggered() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
        v.num_requests_running = Some(50.0);
        v.generation_tokens_per_sec = Some(10.0);
        v.prefill_window_mass = Some(crate::collectors::HistogramWindowMass {
            sum_delta: 0.44,
            count_delta: 2.0,
        });
        v.window_duration_secs = Some(2.0);
        let mut g = gpu_busy();
        g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
        let s = snap(t, t, v, g);
        let ctx = mk_llama8b_h100_ctx(&s);
        let win = mk_win(s);
        let text =
            format_diagnose_rules_test(ai(&ctx, &win), true, "http://127.0.0.1:8000/metrics")
                .join("\n");
        assert!(text.contains("Prefill-bound: not triggered"));
        assert!(text.contains("below 30% threshold"));
    }
}
