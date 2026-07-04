use std::collections::HashSet;

use crate::collectors::{RawSnapshot, effective_tensor_parallel, window_is_evaluable};
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine::Report;
use crate::engine::baseline::{CostSource, PhysicsBaseline, WeightDtypeSource};

use super::r1_under_batching::{R1EvalInput, r1_verbose_miss_line};
use super::r4_oom_risk::r4_advisory;
use super::r6_prefill_bound::{R6GateInput, r6_verbose_miss_line};
use super::{
    ACHIEVABLE_EFFICIENCY_CEILING, ENGINE_MIN_PERSISTENT_WINDOWS, IssueGroup, NO_ISSUES_LINE,
    rule_names,
};

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
        label: "Configured batch limit",
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
    eval: R1EvalInput<'a>,
    tpot_ms: Option<f64>,
    tpot_floor_ms: Option<f64>,
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
                Some(ctx) => r1_verbose_miss_line(ctx.eval),
                None => format!("{}: not triggered", rule.label),
            }
        } else if rule.rule_name == rule_names::PREFILL_BOUND && verbose_rules {
            let gate = r1_context.as_ref().map(|ctx| R6GateInput {
                prompt_tokens_per_sec: ctx.eval.prompt_tokens_per_sec,
                generation_tokens_per_sec: ctx.eval.generation_tokens_per_sec,
                decode_efficiency_pct: baseline.and_then(|b| b.efficiency_pct),
                tpot_ms: ctx.tpot_ms,
                tpot_floor_ms: ctx
                    .tpot_floor_ms
                    .or_else(|| baseline.map(|b| b.tpot_floor_ms)),
                prefix_cache_hit_rate: ctx.eval.prefix_cache_hit_rate,
            });
            gate.and_then(r6_verbose_miss_line)
                .unwrap_or_else(|| format!("{}: not triggered", rule.label))
        } else {
            format!("{}: not triggered", rule.label)
        };
        out.push(line);
    }
}

fn not_triggered_from_fired_names(
    fired_names: &HashSet<&'static str>,
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
    report: &Report,
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
        append_not_triggered_lines(
            &mut out,
            &not_fired,
            verbose_rules,
            Some(R1VerboseContext {
                eval: R1EvalInput {
                    snapshot,
                    config_max_num_seqs: input.ctx.config.max_num_seqs,
                    efficiency_pct: baseline_ref.and_then(|b| b.efficiency_pct),
                    config_relative_efficiency_pct: baseline_ref
                        .and_then(|b| b.config_relative_efficiency_pct),
                    prompt_tokens_per_sec: snapshot.vllm.prompt_tokens_per_sec,
                    generation_tokens_per_sec: snapshot.vllm.generation_tokens_per_sec,
                    prefix_cache_hit_rate: snapshot.vllm.prefix_cache_hit_rate,
                    ridge_batch_size: baseline_ref.map(|b| b.ridge_batch_size),
                },
                tpot_ms: snapshot.vllm.tpot_ms,
                tpot_floor_ms: baseline_ref.map(|b| b.tpot_floor_ms),
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

pub fn format_diagnose_rules_for_windows(
    windows: &[RuntimeWindow],
    summary: AnalysisInput<'_>,
    report: &Report,
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
                &[],
                advisories.r2_present,
                advisories.r4_present,
            );
            append_not_triggered_lines(
                &mut out,
                &not_fired,
                verbose_rules,
                Some(R1VerboseContext {
                    eval: R1EvalInput {
                        snapshot: summary_snap,
                        config_max_num_seqs: summary.ctx.config.max_num_seqs,
                        efficiency_pct: report.baseline.as_ref().and_then(|b| b.efficiency_pct),
                        config_relative_efficiency_pct: report
                            .baseline
                            .as_ref()
                            .and_then(|b| b.config_relative_efficiency_pct),
                        prompt_tokens_per_sec: summary_snap.vllm.prompt_tokens_per_sec,
                        generation_tokens_per_sec: summary_snap.vllm.generation_tokens_per_sec,
                        prefix_cache_hit_rate: summary_snap.vllm.prefix_cache_hit_rate,
                        ridge_batch_size: report.baseline.as_ref().map(|b| b.ridge_batch_size),
                    },
                    tpot_ms: summary_snap.vllm.tpot_ms,
                    tpot_floor_ms: report.baseline.as_ref().map(|b| b.tpot_floor_ms),
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
        append_not_triggered_lines(
            &mut out,
            &not_fired,
            verbose_rules,
            Some(R1VerboseContext {
                eval: R1EvalInput {
                    snapshot: summary_snap,
                    config_max_num_seqs: summary.ctx.config.max_num_seqs,
                    efficiency_pct: report.baseline.as_ref().and_then(|b| b.efficiency_pct),
                    config_relative_efficiency_pct: report
                        .baseline
                        .as_ref()
                        .and_then(|b| b.config_relative_efficiency_pct),
                    prompt_tokens_per_sec: summary_snap.vllm.prompt_tokens_per_sec,
                    generation_tokens_per_sec: summary_snap.vllm.generation_tokens_per_sec,
                    prefix_cache_hit_rate: summary_snap.vllm.prefix_cache_hit_rate,
                    ridge_batch_size: report.baseline.as_ref().map(|b| b.ridge_batch_size),
                },
                tpot_ms: summary_snap.vllm.tpot_ms,
                tpot_floor_ms: report.baseline.as_ref().map(|b| b.tpot_floor_ms),
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
