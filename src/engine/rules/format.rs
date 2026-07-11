use std::collections::HashSet;

use crate::collectors::{
    RawSnapshot, effective_tensor_parallel, window_is_evaluable, window_is_idle,
};
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine::Report;
use crate::engine::baseline::{CostSource, PhysicsBaseline, WeightDtypeSource};

use super::r4_oom_risk::r4_advisory;
use super::{
    ACHIEVABLE_EFFICIENCY_CEILING, ENGINE_MIN_PERSISTENT_WINDOWS, IssueGroup, NO_ISSUES_LINE,
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

#[derive(Clone, Copy)]
struct R4AdvisoryInput {
    kv_headroom_gb: Option<f64>,
    gpu_vram_gb: Option<f64>,
    weight_gb: Option<f64>,
    weight_dtype_source: WeightDtypeSource,
    tensor_parallel_size: Option<u32>,
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

/// User-facing lines when no window met `window_is_evaluable` (shared by stdout and rule formatters).
pub fn no_evaluable_diagnose_lines(
    verbose: bool,
    windows: &[RuntimeWindow],
    hint: Option<&LoadHintParams<'_>>,
) -> Vec<String> {
    let mut out = if let Some(h) = hint {
        load_hint_lines(h)
    } else {
        vec![
            "No qualifying load was detected during this run. Profile only diagnoses behavior under active traffic.".to_string(),
            "Run diagnose again while the server is handling requests (raise concurrency or wait for steady load).".to_string(),
        ]
    };
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
                if hint.is_some() {
                    out.push(format!(
                        "Note: {skipped} of {total} collected windows were not evaluable."
                    ));
                } else {
                    out.push(format!(
                        "Note: {skipped} of {total} collected windows dropped. Telemetry failure. Diagnosis may be incomplete."
                    ));
                }
            }
        }
    }
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

pub fn format_diagnose_rules(
    input: AnalysisInput<'_>,
    report: &Report,
    verbose_rules: bool,
    metrics_url: &str,
) -> Vec<String> {
    let snapshot = &input.window.snapshot;
    if !window_is_evaluable(snapshot) || window_is_idle(snapshot) {
        let hint = LoadHintParams {
            model_name: input.window.snapshot.vllm.model_name.as_deref(),
            metrics_url,
            max_num_seqs: input.ctx.config.max_num_seqs,
            duration_secs: 30,
        };
        return no_evaluable_diagnose_lines(
            verbose_rules,
            std::slice::from_ref(input.window),
            Some(&hint),
        );
    }

    let any_issue = !report.groups.is_empty();
    let baseline_ref = report.baseline.as_ref();
    let tps = snapshot.vllm.generation_tokens_per_sec;

    let fired_names: HashSet<&'static str> = report
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
        R4AdvisoryInput {
            kv_headroom_gb: report.baseline.as_ref().and_then(|b| b.kv_headroom_gb),
            gpu_vram_gb: input.ctx.gpu.vram_gb,
            weight_gb: report.baseline.as_ref().map(|b| b.weight_gb),
            weight_dtype_source: report
                .baseline
                .as_ref()
                .map(|b| b.weight_dtype_source)
                .unwrap_or(WeightDtypeSource::Fallback),
            tensor_parallel_size: effective_tensor_parallel(
                input.ctx.config.tensor_parallel_size,
                input.window.snapshot.collected_gpu_count(),
            ),
        },
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
) -> Vec<String> {
    if report.n_eval == 0 {
        let hint = LoadHintParams {
            model_name: summary.window.snapshot.vllm.model_name.as_deref(),
            metrics_url,
            max_num_seqs: summary.ctx.config.max_num_seqs,
            duration_secs: 30,
        };
        return no_evaluable_diagnose_lines(verbose_rules, windows, Some(&hint));
    }

    let all_idle = windows.iter().all(|w| window_is_idle(&w.snapshot));
    if all_idle {
        let hint = LoadHintParams {
            model_name: summary.window.snapshot.vllm.model_name.as_deref(),
            metrics_url,
            max_num_seqs: summary.ctx.config.max_num_seqs,
            duration_secs: 30,
        };
        return no_evaluable_diagnose_lines(verbose_rules, windows, Some(&hint));
    }

    if report.n_eval < ENGINE_MIN_PERSISTENT_WINDOWS {
        let mut out = vec![
            "[!] Insufficient Sustained Load".to_string(),
            String::new(),
            format!(
                "    Traffic detected but too brief for reliable diagnosis. \
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
            "    Fix:".to_string(),
            "      • Maintain steady traffic for the full diagnostic duration.".to_string(),
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
            &HashSet::new(),
            summary_snap,
            metrics_url,
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
            },
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
        if skipped > 0 {
            out.push(format!(
                "Note: {skipped} of {total} windows dropped. Telemetry failure. Diagnosis may be incomplete."
            ));
        }
        trim_trailing_blank_lines(&mut out);
        return out;
    }

    let fired_names: HashSet<&'static str> = report
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
        },
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
        append_not_triggered_lines(&mut out, &not_fired, &report.suppressed_rules);
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
                ..Default::default()
            },
            gpus: vec![GpuRawMetrics::default()],
            host_gpu_count: None,
        })
    }

    fn join_hint(hint: &LoadHintParams<'_>) -> String {
        no_evaluable_diagnose_lines(false, &[], Some(hint)).join("\n")
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
    fn load_hint_verbose_appends_drop_note() {
        let windows = vec![idle_window(), idle_window()];
        let hint = LoadHintParams {
            model_name: Some("m"),
            metrics_url: "http://localhost:8000/metrics",
            max_num_seqs: Some(256),
            duration_secs: 30,
        };
        let text = no_evaluable_diagnose_lines(true, &windows, Some(&hint)).join("\n");
        assert!(text.contains("Server is idle"));
        assert!(text.contains("benchmark_serving.py"));
        assert!(text.contains("Note: 2 of 2 collected windows were not evaluable."));
        assert!(!text.contains("Telemetry failure"));
    }

    #[test]
    fn verbose_without_hint_still_says_telemetry_failure() {
        let windows = vec![idle_window(), idle_window()];
        let text = no_evaluable_diagnose_lines(true, &windows, None).join("\n");
        assert!(text.contains(
            "Note: 2 of 2 collected windows dropped. Telemetry failure. Diagnosis may be incomplete."
        ));
    }

    #[test]
    fn load_hint_none_falls_back_to_generic() {
        let text = no_evaluable_diagnose_lines(false, &[], None).join("\n");
        assert!(text.contains("No qualifying load was detected"));
        assert!(!text.contains("Server is idle"));
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
    fn idle_evaluable_window_is_detected() {
        let mut w = idle_window();
        w.snapshot.vllm.window_duration_secs = Some(2.0);
        w.snapshot.vllm.num_requests_running = Some(0.0);
        w.snapshot.vllm.generation_tokens_per_sec = Some(0.0);
        assert!(window_is_evaluable(&w.snapshot));
        assert!(window_is_idle(&w.snapshot));
    }

    #[test]
    fn active_window_is_not_idle() {
        let mut w = idle_window();
        w.snapshot.vllm.window_duration_secs = Some(2.0);
        w.snapshot.vllm.num_requests_running = Some(5.0);
        w.snapshot.vllm.generation_tokens_per_sec = Some(100.0);
        assert!(window_is_evaluable(&w.snapshot));
        assert!(!window_is_idle(&w.snapshot));
    }
}
