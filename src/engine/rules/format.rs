use std::collections::HashSet;

use crate::collectors::{
    RawSnapshot, effective_tensor_parallel, window_is_evaluable, window_is_idle,
};
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine::Report;
use crate::engine::baseline::{CostSource, PhysicsBaseline, WeightDtypeSource};

use super::r4_oom_risk::r4_advisory;
use super::{
    ACHIEVABLE_EFFICIENCY_CEILING, ENGINE_MIN_PERSISTENT_WINDOWS, NO_ISSUES_LINE, Recommendation,
    rule_names,
};

const KV_RULE_NAMES: &[&str] = &[
    rule_names::KV_CACHE_PRESSURE,
    rule_names::KV_ADMISSION_BACKLOG,
];

const NOT_TRIGGERED_SINGLES: &[&str] = &[
    rule_names::UNDER_BATCHING,
    rule_names::OOM_RISK,
    rule_names::CONCURRENCY_SATURATION,
    rule_names::CONFIG_HEADROOM,
    rule_names::LOW_PREFIX_REUSE,
    rule_names::PREFILL_BOUND,
    rule_names::MASSIVE_UNDERUTILIZATION,
];

fn kv_rules_absent_from_fired(fired_names: &HashSet<&'static str>) -> bool {
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
pub(super) fn r2_kv_cache_advisory(
    snapshot: &RawSnapshot,
    metrics_url: &str,
) -> Option<Vec<String>> {
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

fn rule_display_block(rec: &Recommendation) -> Vec<String> {
    rec.display_lines.clone()
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

#[derive(Clone, Copy)]
struct R4AdvisoryInput {
    kv_headroom_gb: Option<f64>,
    gpu_vram_gb: Option<f64>,
    weight_gb: Option<f64>,
    weight_dtype_source: WeightDtypeSource,
    tensor_parallel_size: Option<u32>,
}

fn build_r4_advisory_input(summary: &AnalysisInput<'_>, report: &Report) -> R4AdvisoryInput {
    R4AdvisoryInput {
        kv_headroom_gb: report.baseline.as_ref().and_then(|b| b.kv_headroom_gb),
        gpu_vram_gb: summary.ctx.gpu.vram_gb,
        weight_gb: report.baseline.as_ref().map(|b| b.weight_gb),
        weight_dtype_source: report
            .baseline
            .as_ref()
            .map(|b| b.weight_dtype_source)
            .unwrap_or(WeightDtypeSource::Fallback),
        tensor_parallel_size: effective_tensor_parallel(
            summary.ctx.config.tensor_parallel_size,
            summary.window.snapshot.collected_gpu_count(),
        ),
    }
}

fn collect_advisories(
    fired_names: &HashSet<&'static str>,
    snapshot: &RawSnapshot,
    metrics_url: &str,
    r4_input: R4AdvisoryInput,
) -> CollectedAdvisories {
    let r2_adv = if kv_rules_absent_from_fired(fired_names) {
        r2_kv_cache_advisory(snapshot, metrics_url)
    } else {
        None
    };
    let r4_adv = r4_advisory(
        r4_input.kv_headroom_gb,
        r4_input.gpu_vram_gb,
        r4_input.weight_gb,
        r4_input.weight_dtype_source,
        r4_input.tensor_parallel_size,
    );
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
    let waste_fraction = (ACHIEVABLE_EFFICIENCY_CEILING - eff / 100.0).max(0.0);
    let waste = cost_per_hr * waste_fraction;
    if !waste.is_finite() || waste <= 0.0 {
        return None;
    }
    Some(waste)
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
    recommendations: &[Recommendation],
    baseline: Option<&PhysicsBaseline>,
    tps: Option<f64>,
) {
    let rule_names: Vec<&str> = recommendations.iter().map(|r| r.rule_name).collect();
    let Some(suffix) = waste_label_suffix(&rule_names) else {
        return;
    };
    let Some(waste_per_hr) = compute_waste_per_hr(baseline, tps) else {
        return;
    };
    if !lines.is_empty() && !lines.last().is_some_and(|l| l.is_empty()) {
        lines.push(String::new());
    }
    lines.push(format!("    ~${waste_per_hr:.2}/hr {suffix}"));
}

/// Parameters for the idle-state load generation hint.
pub struct LoadHintParams<'a> {
    pub model_name: Option<&'a str>,
    pub metrics_url: &'a str,
    pub max_num_seqs: Option<u32>,
    pub duration_secs: u64,
}

fn base_url_from_metrics(metrics_url: &str) -> &str {
    metrics_url.strip_suffix("/metrics").unwrap_or(metrics_url)
}

fn load_hint_lines(hint: &LoadHintParams<'_>) -> Vec<String> {
    let model = hint.model_name.unwrap_or("your-model-name");
    let base_url = base_url_from_metrics(hint.metrics_url);
    let rate = hint.max_num_seqs.map(|m| m / 2).unwrap_or(10).max(1);
    let num_prompts = rate as u64 * hint.duration_secs * 2;

    vec![
        "Server is idle. Profile diagnoses under load, not at rest.".to_string(),
        String::new(),
        "Generate traffic in a separate terminal with vLLM's built-in benchmark:".to_string(),
        String::new(),
        "    python3 benchmarks/benchmark_serving.py \\".to_string(),
        "        --backend vllm \\".to_string(),
        format!("        --model \"{model}\" \\"),
        format!("        --base-url \"{base_url}\" \\"),
        "        --dataset-name sharegpt \\".to_string(),
        format!("        --num-prompts {num_prompts} \\"),
        format!("        --request-rate {rate}"),
        String::new(),
        "Then re-run: profile diagnose".to_string(),
    ]
}

fn telemetry_failure_lines(metrics_url: &str) -> Vec<String> {
    vec![
        "[!] Telemetry Failure".to_string(),
        String::new(),
        "    Profile could not read metrics from the target server.".to_string(),
        String::new(),
        "    Fix:".to_string(),
        "      Verify the endpoint is reachable:".to_string(),
        String::new(),
        format!("        curl -s {metrics_url}"),
    ]
}

fn append_verbose_window_note(
    out: &mut Vec<String>,
    verbose: bool,
    windows: &[RuntimeWindow],
    idle: bool,
) {
    if !verbose {
        return;
    }
    let total = windows.len();
    if total == 0 {
        out.push("Note: No collection windows were recorded.".to_string());
        return;
    }
    let skipped_broken = windows
        .iter()
        .filter(|w| !window_is_evaluable(&w.snapshot))
        .count();
    let skipped_idle = windows
        .iter()
        .filter(|w| window_is_idle(&w.snapshot))
        .count();
    if skipped_broken == 0 && skipped_idle == 0 {
        return;
    }
    if skipped_broken > 0 {
        if idle {
            out.push(format!(
                "Note: {skipped_broken} of {total} collected windows were not evaluable."
            ));
        } else {
            out.push(format!(
                "Note: {skipped_broken} of {total} windows dropped (telemetry failure). Diagnosis may be incomplete."
            ));
        }
    }
    if skipped_idle > 0 {
        out.push(format!(
            "Note: {skipped_idle} of {total} windows were idle (excluded from analysis)."
        ));
    }
}

fn append_report_skip_notes(out: &mut Vec<String>, report: &Report, blank_before: bool) {
    let total = report.n_eval + report.skipped_broken + report.skipped_idle;
    if report.skipped_broken == 0 && report.skipped_idle == 0 {
        return;
    }
    if blank_before {
        out.push(String::new());
    }
    if report.skipped_broken > 0 {
        out.push(format!(
            "Note: {} of {} windows dropped (telemetry failure). Diagnosis may be incomplete.",
            report.skipped_broken, total
        ));
    }
    if report.skipped_idle > 0 {
        out.push(format!(
            "Note: {} of {} windows were idle (excluded from analysis).",
            report.skipped_idle, total
        ));
    }
}

/// Skip accounting for "Captured: N" lines. Empty when nothing was skipped.
fn captured_skip_suffix(skipped_broken: usize, skipped_idle: usize) -> String {
    let mut parts = Vec::new();
    if skipped_broken > 0 {
        parts.push(format!("{skipped_broken} dropped"));
    }
    if skipped_idle > 0 {
        parts.push(format!("{skipped_idle} idle"));
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!(" ({})", parts.join(", "))
    }
}

/// e.g. `Captured: 2 (3 dropped, 10 idle).` or `Captured: 2.`
pub fn format_captured_windows(
    n_eval: usize,
    skipped_broken: usize,
    skipped_idle: usize,
) -> String {
    format!(
        "Captured: {n_eval}{}.",
        captured_skip_suffix(skipped_broken, skipped_idle)
    )
}

/// Evidence state for the post-DAG MU safety net. Engine selects; this module renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MuVariant {
    Starved,
    BlockedAdmission { kv_measured: bool },
    GaugeMissing,
}

/// Post-DAG traffic/admission safety net (massive under-utilization).
pub(crate) fn mu_diagnose_lines(
    eff: f64,
    run: Option<f64>,
    wait: Option<f64>,
    max_num_seqs: Option<u32>,
    variant: MuVariant,
) -> Vec<String> {
    // Truncate (not round): mean waiting 0.6 must not print as "1 waiting".
    let count_str = |v: Option<f64>| {
        v.filter(|x| x.is_finite())
            .map(|x| format!("{:.0}", x.trunc()))
            .unwrap_or_else(|| "-".to_string())
    };
    let run_str = count_str(run);
    let wait_str = count_str(wait);
    let requests_line = match variant {
        MuVariant::Starved => {
            format!("    Requests  {run_str} running, {wait_str} waiting  (server not saturated)")
        }
        MuVariant::BlockedAdmission { .. } => {
            let seats = match (max_num_seqs, run.filter(|r| r.is_finite())) {
                (Some(max_n), Some(run_v)) => {
                    let free = (f64::from(max_n) - run_v.trunc()).max(0.0);
                    format!("{:.0} of {max_n} seats free", free.trunc())
                }
                _ => "seats free unknown".to_string(),
            };
            format!("    Requests  {run_str} running, {wait_str} waiting  ({seats})")
        }
        MuVariant::GaugeMissing => {
            format!(
                "    Requests  {run_str} running, {wait_str} waiting  (waiting gauge unavailable)"
            )
        }
    };
    let (cause, fix, expected, confidence) = match variant {
        MuVariant::Starved => (
            "      Server is under-fed. No config bottleneck detected - client traffic is too low to utilize the hardware.",
            "      • Batch more requests or increase client concurrency until a wait queue forms.",
            "    Expected: Efficiency climbs as the GPU is fed more work.",
            "    Confidence: Medium",
        ),
        MuVariant::BlockedAdmission { kv_measured: true } => (
            "      Requests queue while seats are free and KV cache is low. Scheduler admission is blocked - likely the prefill token budget.",
            "      • Raise --max-num-batched-tokens or enable chunked prefill.",
            "    Expected: Queue drains as admission unblocks.",
            "    Confidence: Low (cause inferred, token budget not observed)",
        ),
        MuVariant::BlockedAdmission { kv_measured: false } => (
            "      Requests queue while seats are free. Scheduler admission is blocked - likely the prefill token budget.",
            "      • Raise --max-num-batched-tokens or enable chunked prefill.",
            "    Expected: Queue drains as admission unblocks.",
            "    Confidence: Low (cause inferred; KV gauge unavailable)",
        ),
        MuVariant::GaugeMissing => (
            "      Server is under-fed. No config bottleneck detected - client traffic is too low to utilize the hardware.",
            "      • Batch more requests or increase client concurrency until a wait queue forms.",
            "    Expected: Efficiency climbs as the GPU is fed more work.",
            "    Confidence: Low (waiting gauge unavailable)",
        ),
    };
    vec![
        "[!] Massive Under-utilization".to_string(),
        String::new(),
        format!(
            "    Decode eff. {eff:.1}%  (threshold: < {:.0}%)",
            crate::engine::MASSIVE_UNDERUTIL_THRESHOLD_PCT
        ),
        requests_line,
        String::new(),
        "    Cause:".to_string(),
        cause.to_string(),
        String::new(),
        "    Fix:".to_string(),
        fix.to_string(),
        String::new(),
        expected.to_string(),
        confidence.to_string(),
    ]
}

/// Idle server with working telemetry: show load-generation hint.
pub fn idle_diagnose_lines(
    verbose: bool,
    windows: &[RuntimeWindow],
    hint: &LoadHintParams<'_>,
) -> Vec<String> {
    let mut out = load_hint_lines(hint);
    append_verbose_window_note(&mut out, verbose, windows, true);
    out
}

/// Unreachable or broken telemetry: show connectivity diagnostic.
pub fn unreachable_diagnose_lines(
    verbose: bool,
    windows: &[RuntimeWindow],
    metrics_url: &str,
) -> Vec<String> {
    let mut out = telemetry_failure_lines(metrics_url);
    append_verbose_window_note(&mut out, verbose, windows, false);
    out
}

fn append_not_triggered_lines(
    out: &mut Vec<String>,
    rules: &[&str],
    suppressed_rules: &[(&'static str, &'static str)],
) {
    if rules.is_empty() && suppressed_rules.is_empty() {
        return;
    }
    if !out.is_empty() && !out.last().is_some_and(|l| l.is_empty()) {
        out.push(String::new());
    }
    for rule in rules {
        out.push(format!("{}: not triggered", rule_names::display_name(rule)));
    }
    for &(suppressed_name, suppressor_name) in suppressed_rules {
        let label = rule_names::display_name(suppressed_name);
        let suppressor_label = rule_names::display_name(suppressor_name);
        out.push(format!("{label}: suppressed by {suppressor_label}"));
    }
}

fn not_triggered_from_fired_names(
    fired_names: &HashSet<&'static str>,
    suppressed_rules: &[(&'static str, &'static str)],
    r2_adv_present: bool,
    r4_adv_present: bool,
) -> Vec<&'static str> {
    let suppressed = |name: &str| suppressed_rules.iter().any(|(s, _)| *s == name);
    let mut rules = Vec::new();
    for &entry in NOT_TRIGGERED_SINGLES {
        if entry == rule_names::OOM_RISK && r4_adv_present {
            continue;
        }
        if !fired_names.contains(entry) && !suppressed(entry) {
            rules.push(entry);
        }
    }
    if !KV_RULE_NAMES.iter().any(|n| fired_names.contains(n))
        && !KV_RULE_NAMES.iter().any(|n| suppressed(n))
        && !r2_adv_present
    {
        rules.push(rule_names::KV_CACHE_PRESSURE);
    }
    rules
}

/// Single-window path: test-only. Production always uses `format_diagnose_rules_for_windows`.
#[cfg(test)]
pub fn format_diagnose_rules(
    input: AnalysisInput<'_>,
    report: &Report,
    verbose_rules: bool,
    metrics_url: &str,
) -> Vec<String> {
    let snapshot = &input.window.snapshot;
    if !window_is_evaluable(snapshot) {
        return unreachable_diagnose_lines(
            verbose_rules,
            std::slice::from_ref(input.window),
            metrics_url,
        );
    }
    if window_is_idle(snapshot) {
        let hint = LoadHintParams {
            model_name: input.window.snapshot.vllm.model_name.as_deref(),
            metrics_url,
            max_num_seqs: input.ctx.config.max_num_seqs,
            duration_secs: 30,
        };
        return idle_diagnose_lines(verbose_rules, std::slice::from_ref(input.window), &hint);
    }

    let any_issue = !report.recommendations.is_empty();
    let baseline_ref = report.baseline.as_ref();
    let tps = snapshot.vllm.generation_tokens_per_sec;

    let fired_names: HashSet<&'static str> =
        report.recommendations.iter().map(|r| r.rule_name).collect();

    let mut out = Vec::new();

    for rec in &report.recommendations {
        append_display_block(&mut out, rule_display_block(rec));
    }

    append_waste_line(&mut out, &report.recommendations, baseline_ref, tps);

    let advisories = collect_advisories(
        &fired_names,
        snapshot,
        metrics_url,
        build_r4_advisory_input(&input, report),
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
        append_not_triggered_lines(&mut out, &not_fired, &report.suppressed_rules);
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

pub fn format_diagnose_rules_for_windows(
    windows: &[RuntimeWindow],
    summary: AnalysisInput<'_>,
    report: &Report,
    verbose_rules: bool,
    metrics_url: &str,
    duration_secs: u64,
) -> Vec<String> {
    if report.n_eval == 0 {
        let any_evaluable = windows.iter().any(|w| window_is_evaluable(&w.snapshot));
        if any_evaluable {
            let hint = LoadHintParams {
                model_name: summary.window.snapshot.vllm.model_name.as_deref(),
                metrics_url,
                max_num_seqs: summary.ctx.config.max_num_seqs,
                duration_secs,
            };
            return idle_diagnose_lines(verbose_rules, windows, &hint);
        }
        return unreachable_diagnose_lines(verbose_rules, windows, metrics_url);
    }

    if report.n_eval < ENGINE_MIN_PERSISTENT_WINDOWS {
        let mut out = vec![
            "[!] Insufficient Sustained Load".to_string(),
            String::new(),
            format!(
                "    Traffic detected but too brief for reliable diagnosis. \
                 Required: {} evaluable windows. {}",
                ENGINE_MIN_PERSISTENT_WINDOWS,
                format_captured_windows(report.n_eval, report.skipped_broken, report.skipped_idle)
            ),
            String::new(),
            "    Fix:".to_string(),
            "      • Maintain steady traffic for the full diagnostic duration.".to_string(),
        ];
        trim_trailing_blank_lines(&mut out);
        return out;
    }

    let summary_snap = &summary.window.snapshot;
    let baseline_ref = report.baseline.as_ref();
    let tps = summary_snap.vllm.generation_tokens_per_sec;

    if report.recommendations.is_empty() {
        let mut out = Vec::new();
        let advisories = collect_advisories(
            &HashSet::new(),
            summary_snap,
            metrics_url,
            build_r4_advisory_input(&summary, report),
        );
        let any_advisory = advisories.any();
        append_display_block(&mut out, advisories.lines);
        if verbose_rules {
            let not_fired = not_triggered_from_fired_names(
                &HashSet::new(),
                &report.suppressed_rules,
                advisories.r2_present,
                advisories.r4_present,
            );
            append_not_triggered_lines(&mut out, &not_fired, &report.suppressed_rules);
        }
        if !any_advisory && !verbose_rules {
            out.push(NO_ISSUES_LINE.to_string());
        }
        append_report_skip_notes(&mut out, report, false);
        trim_trailing_blank_lines(&mut out);
        return out;
    }

    let fired_names: HashSet<&'static str> =
        report.recommendations.iter().map(|r| r.rule_name).collect();

    let mut warnings: Vec<String> = Vec::new();
    for rec in &report.recommendations {
        if !warnings.is_empty() && !warnings.last().is_some_and(|l| l.is_empty()) {
            warnings.push(String::new());
        }
        warnings.extend(rule_display_block(rec));
        warnings.push(String::new());
    }

    append_waste_line(&mut warnings, &report.recommendations, baseline_ref, tps);

    let advisories = collect_advisories(
        &fired_names,
        summary_snap,
        metrics_url,
        build_r4_advisory_input(&summary, report),
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
    let any_warning = !report.recommendations.is_empty();
    if verbose_rules {
        append_not_triggered_lines(&mut out, &not_fired, &report.suppressed_rules);
    }
    if !any_warning && !any_advisory && !verbose_rules {
        out.push(NO_ISSUES_LINE.to_string());
    }
    append_report_skip_notes(&mut out, report, true);

    trim_trailing_blank_lines(&mut out);
    out
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
mod load_hint_tests {
    use super::*;
    use crate::collectors::{GpuRawMetrics, VllmRawMetrics};
    use std::time::SystemTime;

    fn idle_window() -> RuntimeWindow {
        let t = SystemTime::UNIX_EPOCH;
        RuntimeWindow::from_snapshot(RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: VllmRawMetrics {
                num_requests_running: Some(0.0),
                generation_tokens_per_sec: Some(0.0),
                window_duration_secs: Some(2.0),
                ..Default::default()
            },
            gpus: vec![GpuRawMetrics::default()],
        })
    }

    fn broken_window() -> RuntimeWindow {
        let t = SystemTime::UNIX_EPOCH;
        RuntimeWindow::from_snapshot(RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: VllmRawMetrics {
                num_requests_running: None,
                generation_tokens_per_sec: Some(0.0),
                ..Default::default()
            },
            gpus: vec![GpuRawMetrics::default()],
        })
    }

    fn join_hint(hint: &LoadHintParams<'_>) -> String {
        idle_diagnose_lines(false, &[], hint).join("\n")
    }

    #[test]
    fn load_hint_contains_model_name() {
        let hint = LoadHintParams {
            model_name: Some("Qwen/Qwen3-27B"),
            metrics_url: "http://localhost:8000/metrics",
            max_num_seqs: Some(256),
            duration_secs: 30,
        };
        assert!(join_hint(&hint).contains("--model \"Qwen/Qwen3-27B\""));
    }

    #[test]
    fn load_hint_model_fallback() {
        let hint = LoadHintParams {
            model_name: None,
            metrics_url: "http://localhost:8000/metrics",
            max_num_seqs: Some(256),
            duration_secs: 30,
        };
        assert!(join_hint(&hint).contains("--model \"your-model-name\""));
    }

    #[test]
    fn load_hint_strips_metrics_suffix() {
        let hint = LoadHintParams {
            model_name: Some("m"),
            metrics_url: "http://myhost:9000/metrics",
            max_num_seqs: Some(256),
            duration_secs: 30,
        };
        assert!(join_hint(&hint).contains("--base-url \"http://myhost:9000\""));
    }

    #[test]
    fn load_hint_strips_metrics_no_suffix() {
        let hint = LoadHintParams {
            model_name: Some("m"),
            metrics_url: "http://myhost:9000",
            max_num_seqs: Some(256),
            duration_secs: 30,
        };
        assert!(join_hint(&hint).contains("--base-url \"http://myhost:9000\""));
    }

    #[test]
    fn load_hint_model_with_spaces_is_quoted() {
        let hint = LoadHintParams {
            model_name: Some("/mnt/models/my model"),
            metrics_url: "http://localhost:8000/metrics",
            max_num_seqs: Some(256),
            duration_secs: 30,
        };
        assert!(join_hint(&hint).contains("--model \"/mnt/models/my model\""));
    }

    #[test]
    fn load_hint_rate_from_max_num_seqs() {
        let hint = LoadHintParams {
            model_name: Some("m"),
            metrics_url: "http://localhost:8000/metrics",
            max_num_seqs: Some(256),
            duration_secs: 30,
        };
        assert!(join_hint(&hint).contains("--request-rate 128"));
    }

    #[test]
    fn load_hint_rate_fallback() {
        let hint = LoadHintParams {
            model_name: Some("m"),
            metrics_url: "http://localhost:8000/metrics",
            max_num_seqs: None,
            duration_secs: 30,
        };
        assert!(join_hint(&hint).contains("--request-rate 10"));
    }

    #[test]
    fn load_hint_num_prompts_scaled() {
        let hint = LoadHintParams {
            model_name: Some("m"),
            metrics_url: "http://localhost:8000/metrics",
            max_num_seqs: Some(256),
            duration_secs: 30,
        };
        assert!(join_hint(&hint).contains("--num-prompts 7680"));
    }

    #[test]
    fn load_hint_num_prompts_fallback() {
        let hint = LoadHintParams {
            model_name: Some("m"),
            metrics_url: "http://localhost:8000/metrics",
            max_num_seqs: None,
            duration_secs: 30,
        };
        assert!(join_hint(&hint).contains("--num-prompts 600"));
    }

    #[test]
    fn load_hint_verbose_appends_idle_note() {
        let windows = vec![idle_window(), idle_window()];
        let hint = LoadHintParams {
            model_name: Some("m"),
            metrics_url: "http://localhost:8000/metrics",
            max_num_seqs: Some(256),
            duration_secs: 30,
        };
        let text = idle_diagnose_lines(true, &windows, &hint).join("\n");
        assert!(text.contains("Server is idle"));
        assert!(text.contains("benchmark_serving.py"));
        assert!(text.contains("Note: 2 of 2 windows were idle (excluded from analysis)."));
        assert!(!text.contains("(telemetry failure)"));
    }

    #[test]
    fn test_unreachable_shows_curl_command() {
        let url = "http://broken:8000/metrics";
        let text = unreachable_diagnose_lines(false, &[], url).join("\n");
        assert!(text.contains("[!] Telemetry Failure"));
        assert!(text.contains("Fix:"));
        assert!(text.contains(&format!("curl -s {url}")));
        assert!(!text.contains("Server is idle"));
        assert!(!text.contains("benchmark_serving.py"));
    }

    #[test]
    fn test_idle_shows_benchmark_hint() {
        let hint = LoadHintParams {
            model_name: Some("m"),
            metrics_url: "http://localhost:8000/metrics",
            max_num_seqs: Some(256),
            duration_secs: 30,
        };
        let text = idle_diagnose_lines(false, &[], &hint).join("\n");
        assert!(text.contains("Server is idle"));
        assert!(text.contains("benchmark_serving.py"));
        assert!(!text.contains("[!] Telemetry Failure"));
    }

    #[test]
    fn test_unreachable_verbose_shows_dropped_count() {
        let windows = vec![broken_window(), broken_window()];
        let text =
            unreachable_diagnose_lines(true, &windows, "http://localhost:8000/metrics").join("\n");
        assert!(text.contains("[!] Telemetry Failure"));
        assert!(text.contains(
            "Note: 2 of 2 windows dropped (telemetry failure). Diagnosis may be incomplete."
        ));
    }

    #[test]
    fn test_unreachable_verbose_idle_windows_not_labeled_failure() {
        let windows = vec![idle_window(), idle_window()];
        let text =
            unreachable_diagnose_lines(true, &windows, "http://localhost:8000/metrics").join("\n");
        assert!(text.contains("[!] Telemetry Failure"));
        assert!(text.contains("Note: 2 of 2 windows were idle (excluded from analysis)."));
        assert!(!text.contains("(telemetry failure)"));
    }

    #[test]
    fn load_hint_separate_terminal_mentioned() {
        let hint = LoadHintParams {
            model_name: Some("m"),
            metrics_url: "http://localhost:8000/metrics",
            max_num_seqs: Some(256),
            duration_secs: 30,
        };
        assert!(join_hint(&hint).contains("separate terminal"));
    }

    #[test]
    fn load_hint_rate_floor_at_one() {
        let hint = LoadHintParams {
            model_name: Some("m"),
            metrics_url: "http://localhost:8000/metrics",
            max_num_seqs: Some(1),
            duration_secs: 30,
        };
        assert!(join_hint(&hint).contains("--request-rate 1"));
        assert!(!join_hint(&hint).contains("--request-rate 0"));
    }

    #[test]
    fn mu_diagnose_lines_starved_variant() {
        let text = mu_diagnose_lines(12.5, Some(64.0), Some(0.0), Some(256), MuVariant::Starved)
            .join("\n");
        assert!(text.contains("Requests  64 running, 0 waiting  (server not saturated)"));
        assert!(text.contains("Server is under-fed"));
        assert!(text.contains("Confidence: Medium"));
        assert!(!text.contains("GPU is idle"));
        let blip = mu_diagnose_lines(12.5, Some(64.0), Some(0.6), Some(256), MuVariant::Starved)
            .join("\n");
        assert!(blip.contains("Requests  64 running, 0 waiting  (server not saturated)"));
        assert!(!blip.contains("1 waiting"));
    }

    #[test]
    fn mu_diagnose_lines_blocked_admission_variant() {
        let measured = mu_diagnose_lines(
            12.0,
            Some(10.0),
            Some(5.0),
            Some(256),
            MuVariant::BlockedAdmission { kv_measured: true },
        )
        .join("\n");
        assert!(measured.contains("Requests  10 running, 5 waiting  (246 of 256 seats free)"));
        assert!(measured.contains("seats are free and KV cache is low"));
        assert!(measured.contains("Scheduler admission is blocked"));
        assert!(measured.contains("Raise --max-num-batched-tokens"));
        assert!(measured.contains("Confidence: Low (cause inferred, token budget not observed)"));
        assert!(!measured.contains("server not saturated"));
        assert!(!measured.contains("KV gauge unavailable"));

        let unmeasured = mu_diagnose_lines(
            12.0,
            Some(10.0),
            Some(5.0),
            Some(256),
            MuVariant::BlockedAdmission { kv_measured: false },
        )
        .join("\n");
        assert!(unmeasured.contains("Requests  10 running, 5 waiting  (246 of 256 seats free)"));
        assert!(unmeasured.contains("Requests queue while seats are free."));
        assert!(!unmeasured.contains("KV cache is low"));
        assert!(unmeasured.contains("Scheduler admission is blocked"));
        assert!(unmeasured.contains("Raise --max-num-batched-tokens"));
        assert!(unmeasured.contains("Confidence: Low (cause inferred; KV gauge unavailable)"));

        let no_max = mu_diagnose_lines(
            12.0,
            Some(10.0),
            Some(5.0),
            None,
            MuVariant::BlockedAdmission { kv_measured: true },
        )
        .join("\n");
        assert!(no_max.contains("(seats free unknown)"));
        assert!(!no_max.contains("server not saturated"));
    }

    #[test]
    fn mu_diagnose_lines_gauge_missing_variant() {
        let text = mu_diagnose_lines(12.5, Some(64.0), None, Some(256), MuVariant::GaugeMissing)
            .join("\n");
        assert!(text.contains("Requests  64 running, - waiting  (waiting gauge unavailable)"));
        assert!(text.contains("Server is under-fed"));
        assert!(text.contains("Confidence: Low (waiting gauge unavailable)"));
        assert!(!text.contains("server not saturated"));
    }

    #[test]
    fn format_captured_windows_variants() {
        assert_eq!(format_captured_windows(2, 0, 0), "Captured: 2.");
        assert_eq!(format_captured_windows(2, 3, 0), "Captured: 2 (3 dropped).");
        assert_eq!(format_captured_windows(2, 0, 10), "Captured: 2 (10 idle).");
        assert_eq!(
            format_captured_windows(2, 3, 10),
            "Captured: 2 (3 dropped, 10 idle)."
        );
    }

    #[test]
    fn idle_evaluable_window_is_detected() {
        let w = idle_window();
        assert!(window_is_evaluable(&w.snapshot));
        assert!(window_is_idle(&w.snapshot));
    }

    #[test]
    fn active_window_is_not_idle() {
        let mut w = idle_window();
        w.snapshot.vllm.num_requests_running = Some(5.0);
        w.snapshot.vllm.generation_tokens_per_sec = Some(100.0);
        assert!(window_is_evaluable(&w.snapshot));
        assert!(!window_is_idle(&w.snapshot));
    }
}
