use std::collections::HashSet;

use crate::collectors::{
    RawSnapshot, effective_tensor_parallel, window_is_evaluable, window_is_idle,
};
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine::Report;
use crate::engine::baseline::WeightDtypeSource;

use super::r4_oom_risk::r4_advisory;
use super::{ENGINE_MIN_PERSISTENT_WINDOWS, NO_ISSUES_LINE, Recommendation, rule_names};

/// Revealed secondary whose Fix fights the primary. Rank ME already suppresses;
/// this annotates Fix when that secondary is shown under reveal (2nd same primary).
/// Bound path: R6 primary, R1 revealed → neutralize R1 add-load.
/// Soft field: R1 primary, R6 revealed → neutralize Prefill scale-out; warn to
/// read REQUESTS/CACHE in the header.
const REVEAL_FIX_CONFLICT_TABLE: &[(&str, &str, &str)] = &[
    (
        rule_names::PREFILL_BOUND,
        rule_names::UNDER_BATCHING,
        "        Conflicts with primary above: adds load to a server at its compute wall.",
    ),
    (
        rule_names::UNDER_BATCHING,
        rule_names::PREFILL_BOUND,
        "        Conflicts with primary above: scale-out while the box is still under-fed. Check REQUESTS and CACHE in the header.",
    ),
];

fn reveal_fix_conflict_subline(
    primary: &'static str,
    secondary: &'static str,
) -> Option<&'static str> {
    REVEAL_FIX_CONFLICT_TABLE
        .iter()
        .find(|&&(p, s, _)| p == primary && s == secondary)
        .map(|&(_, _, sub)| sub)
}

/// Keep Cause and Fix bullets; mark header `[?]`; attach conflict subline under Fix.
/// Drop Expected (it describes the conflicting action).
fn neutralize_revealed_fix(display_lines: &[String], conflict_subline: &str) -> Vec<String> {
    let Some(fix_at) = display_lines.iter().position(|l| l == "    Fix:") else {
        return display_lines.to_vec();
    };
    let after_fix = fix_at + 1;
    let resume_at = display_lines[after_fix..]
        .iter()
        .position(|l| {
            l.starts_with("    Expected:")
                || l.starts_with("    Confidence:")
                || l.starts_with("    Note:")
        })
        .map(|i| after_fix + i)
        .unwrap_or(display_lines.len());

    let mut out = Vec::with_capacity(display_lines.len() + 2);
    for (i, line) in display_lines.iter().enumerate() {
        if i > fix_at {
            break;
        }
        if i == 0 && line.starts_with("[!] ") {
            out.push(format!(
                "[?] {}",
                line.trim_start_matches("[!] ").trim_start()
            ));
        } else {
            out.push(line.clone());
        }
    }
    // Fix bullets (skip trailing blanks before Expected/Confidence).
    let mut fix_end = resume_at;
    while fix_end > after_fix && display_lines[fix_end - 1].is_empty() {
        fix_end -= 1;
    }
    out.extend_from_slice(&display_lines[after_fix..fix_end]);
    out.push(conflict_subline.to_string());

    let mut i = resume_at;
    if i < display_lines.len() && display_lines[i].starts_with("    Expected:") {
        i += 1;
        if i < display_lines.len() && display_lines[i].is_empty() {
            i += 1;
        }
    }
    if i < display_lines.len() {
        if !out.last().is_some_and(|l| l.is_empty())
            && (display_lines[i].starts_with("    Confidence:")
                || display_lines[i].starts_with("    Note:"))
        {
            out.push(String::new());
        }
        out.extend_from_slice(&display_lines[i..]);
    }
    out
}

fn revealed_secondary_block(primary_name: &'static str, rec: &Recommendation) -> Vec<String> {
    match reveal_fix_conflict_subline(primary_name, rec.rule_name) {
        Some(sub) => neutralize_revealed_fix(&rec.display_lines, sub),
        None => rec.display_lines.clone(),
    }
}

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

/// Verbose (-v) only. Label uncertainty tracks the printed number's source, not
/// the existence of disagreement between sources; this note does not change
/// `(est)` labeling on observed-derived capacity.
fn catalog_state_mismatch_verbose_line(
    mismatch: Option<(u64, u64)>,
    verbose_rules: bool,
) -> Option<String> {
    if !verbose_rules {
        return None;
    }
    let (catalog_pages, observed_pages) = mismatch?;
    Some(format!(
        "Note: Catalog state {catalog_pages} pages, observed {observed_pages}; entry may be stale."
    ))
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum MuVariant {
    Starved,
    /// `kv_pct` is Some when the usage gauge was present (and below R2's 88% gate).
    BlockedAdmission {
        kv_pct: Option<f64>,
    },
    GaugeMissing,
}

/// KV below this: the cause line may say "low". At or above: state the number only.
const MU_KV_LOW_PCT: f64 = 50.0;

fn blocked_admission_kv_clause(kv_pct: f64) -> String {
    if kv_pct < MU_KV_LOW_PCT {
        format!("KV cache at {kv_pct:.0}% (low)")
    } else {
        format!("KV cache at {kv_pct:.0}%")
    }
}

/// Post-DAG traffic/admission safety net (massive under-utilization).
///
/// `chunked_prefill_enabled`: Enable only on `Some(false)`; Confirm on `None`;
/// omit on `Some(true)`. Unknown is not off.
pub(crate) fn mu_diagnose_lines(
    eff: f64,
    run: Option<f64>,
    wait: Option<f64>,
    max_num_seqs: Option<u32>,
    variant: MuVariant,
    chunked_prefill_enabled: Option<bool>,
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
    let (cause, mut fix_bullets, expected, confidence) = match variant {
        MuVariant::Starved => (
            "      Server is under-fed. No config bottleneck detected - client traffic is too low to utilize the hardware."
                .to_string(),
            vec![
                "      • Batch more requests or increase client concurrency until a wait queue forms."
                    .to_string(),
            ],
            "    Expected: Efficiency climbs as the GPU is fed more work.".to_string(),
            "    Confidence: Medium".to_string(),
        ),
        MuVariant::BlockedAdmission {
            kv_pct: Some(pct),
        } => (
            format!(
                "      Requests queue while seats are free and {}. Scheduler admission is blocked, likely the prefill token budget.",
                blocked_admission_kv_clause(pct)
            ),
            blocked_admission_fix_bullets(chunked_prefill_enabled),
            "    Expected: Queue drains as admission unblocks.".to_string(),
            "    Confidence: Low (cause inferred, token budget not observed)".to_string(),
        ),
        MuVariant::BlockedAdmission { kv_pct: None } => (
            "      Requests queue while seats are free. Scheduler admission is blocked, likely the prefill token budget."
                .to_string(),
            blocked_admission_fix_bullets(chunked_prefill_enabled),
            "    Expected: Queue drains as admission unblocks.".to_string(),
            "    Confidence: Low (cause inferred; KV gauge unavailable)".to_string(),
        ),
        MuVariant::GaugeMissing => (
            "      Server is under-fed. No config bottleneck detected - client traffic is too low to utilize the hardware."
                .to_string(),
            vec![
                "      • Batch more requests or increase client concurrency until a wait queue forms."
                    .to_string(),
            ],
            "    Expected: Efficiency climbs as the GPU is fed more work.".to_string(),
            "    Confidence: Low (waiting gauge unavailable)".to_string(),
        ),
    };
    let mut out = vec![
        "[!] Massive Under-utilization".to_string(),
        String::new(),
        format!(
            "    Decode eff. {eff:.1}%  (threshold: < {:.0}%)",
            crate::engine::MASSIVE_UNDERUTIL_THRESHOLD_PCT
        ),
        requests_line,
        String::new(),
        "    Cause:".to_string(),
        cause,
        String::new(),
        "    Fix:".to_string(),
    ];
    out.append(&mut fix_bullets);
    out.push(String::new());
    out.push(expected);
    out.push(confidence);
    out
}

fn blocked_admission_fix_bullets(chunked_prefill_enabled: Option<bool>) -> Vec<String> {
    let mut bullets = vec!["      • Raise --max-num-batched-tokens.".to_string()];
    match chunked_prefill_enabled {
        Some(false) => {
            bullets.push("      • Enable chunked prefill (--enable-chunked-prefill).".to_string());
        }
        None => {
            bullets.push(
                "      • Confirm chunked prefill is enabled (--enable-chunked-prefill)."
                    .to_string(),
            );
        }
        Some(true) => {}
    }
    bullets
}

/// Idle server with working telemetry: show load-generation hint.
fn idle_diagnose_lines(
    verbose: bool,
    windows: &[RuntimeWindow],
    hint: &LoadHintParams<'_>,
) -> Vec<String> {
    let mut out = load_hint_lines(hint);
    append_verbose_window_note(&mut out, verbose, windows, true);
    out
}

/// Unreachable or broken telemetry: show connectivity diagnostic.
fn unreachable_diagnose_lines(
    verbose: bool,
    windows: &[RuntimeWindow],
    metrics_url: &str,
) -> Vec<String> {
    let mut out = telemetry_failure_lines(metrics_url);
    append_verbose_window_note(&mut out, verbose, windows, false);
    out
}

/// Single crash-vs-idle chooser for empty diagnose runs (`n_eval == 0` / all-idle).
pub fn empty_run_diagnose_lines(
    verbose: bool,
    windows: &[RuntimeWindow],
    any_evaluable: bool,
    hint: &LoadHintParams<'_>,
    metrics_url: &str,
) -> Vec<String> {
    if any_evaluable {
        idle_diagnose_lines(verbose, windows, hint)
    } else {
        unreachable_diagnose_lines(verbose, windows, metrics_url)
    }
}

fn append_not_triggered_lines(
    out: &mut Vec<String>,
    rules: &[&str],
    suppressed_rules: &[(&'static str, &'static str)],
    gauge_missing: &crate::engine::GaugeMissingCounts,
    n_eval: usize,
) {
    if rules.is_empty() && suppressed_rules.is_empty() {
        return;
    }
    if !out.is_empty() && !out.last().is_some_and(|l| l.is_empty()) {
        out.push(String::new());
    }
    for rule in rules {
        let missing = match *rule {
            rule_names::UNDER_BATCHING => (gauge_missing.under_batching, "waiting gauge missing"),
            rule_names::KV_CACHE_PRESSURE => (gauge_missing.kv_cache_pressure, "KV gauge missing"),
            rule_names::LOW_PREFIX_REUSE => {
                (gauge_missing.low_prefix_reuse, "hit-rate gauge missing")
            }
            rule_names::CONCURRENCY_SATURATION => (
                gauge_missing.concurrency_saturation,
                "waiting gauge missing",
            ),
            _ => (0, ""),
        };
        let label = rule_names::display_name(rule);
        let (k, reason) = missing;
        out.push(if k > 0 && n_eval > 0 && k >= n_eval {
            format!("{label}: not evaluated ({reason}).")
        } else if k > 0 && n_eval > 0 {
            format!("{label}: not triggered ({reason} in {k}/{n_eval} windows).")
        } else {
            format!("{label}: not triggered")
        });
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

/// Quiet no-issues path: scoreboard already printed [`SPEC_GUARD_WARNING_LINE`]
/// when speculation is flagged. Skip the duplicate here. Healthy exit still uses
/// [`crate::engine::limiter::limiter_line`] directly.
fn limiter_line_for_quiet_report(ev: &crate::engine::limiter::LimiterEvidence) -> Option<String> {
    let line = crate::engine::limiter::limiter_line(ev)?;
    if matches!(
        crate::engine::limiter::identify(ev).verdict,
        Some(crate::engine::limiter::LimiterVerdict::SpecSuspected)
    ) {
        return None;
    }
    Some(line)
}

pub fn format_diagnose_rules_for_windows(
    windows: &[RuntimeWindow],
    summary: AnalysisInput<'_>,
    report: &Report,
    verbose_rules: bool,
    metrics_url: &str,
    duration_secs: u64,
    reveal_suppressed: bool,
) -> Vec<String> {
    if report.n_eval == 0 {
        let any_evaluable = windows.iter().any(|w| window_is_evaluable(&w.snapshot));
        let hint = LoadHintParams {
            model_name: summary.window.snapshot.vllm.model_name.as_deref(),
            metrics_url,
            max_num_seqs: summary.ctx.config.max_num_seqs,
            duration_secs,
        };
        return empty_run_diagnose_lines(verbose_rules, windows, any_evaluable, &hint, metrics_url);
    }

    if report.n_eval < ENGINE_MIN_PERSISTENT_WINDOWS {
        let mut out = vec![
            "[!] Insufficient Sustained Load".to_string(),
            String::new(),
            format!(
                "    Load too brief for reliable diagnosis. \
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
            append_not_triggered_lines(
                &mut out,
                &not_fired,
                &report.suppressed_rules,
                &report.gauge_missing,
                report.n_eval,
            );
        }
        if !any_advisory {
            out.push(NO_ISSUES_LINE.to_string());
            if let Some(ev) = report.limiter_evidence.as_ref()
                && let Some(line) = limiter_line_for_quiet_report(ev)
            {
                out.push(line);
            }
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

    // Reveal suppressed alternatives: same-primary re-fire, or terminal primary
    // on first fire (no server-local knob left).
    let primary_terminal = report.recommendations.first().is_some_and(|r| r.terminal);
    let primary_name = report.recommendations.first().map(|r| r.rule_name);
    if (reveal_suppressed || primary_terminal) && !report.suppressed_recs.is_empty() {
        if !warnings.last().is_some_and(|l| l.is_empty()) {
            warnings.push(String::new());
        }
        let header = if primary_terminal {
            "The primary issue has no tuning fix on this server. Other possible causes shown below."
        } else {
            "Same issue after remeasure. Other possible causes shown below."
        };
        warnings.push(header.to_string());
        warnings.push(String::new());
        for rec in &report.suppressed_recs {
            let block = match primary_name {
                Some(p) => revealed_secondary_block(p, rec),
                None => rec.display_lines.clone(),
            };
            warnings.extend(block);
            warnings.push(String::new());
        }
    }

    let advisories = collect_advisories(
        &fired_names,
        summary_snap,
        metrics_url,
        build_r4_advisory_input(&summary, report),
    );
    let mut out = warnings;
    append_display_block(&mut out, advisories.lines);

    if let Some(line) = kv_ceiling_unknown_verbose_line(report.kv_max_seqs, verbose_rules) {
        append_display_block(&mut out, vec![line]);
    }
    if let Some(line) =
        catalog_state_mismatch_verbose_line(report.catalog_state_mismatch, verbose_rules)
    {
        append_display_block(&mut out, vec![line]);
    }

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
            &report.suppressed_rules,
            &report.gauge_missing,
            report.n_eval,
        );
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

            host_memory: None,
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

            host_memory: None,
        })
    }

    fn join_hint(hint: &LoadHintParams<'_>) -> String {
        empty_run_diagnose_lines(false, &[], true, hint, hint.metrics_url).join("\n")
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
        let text =
            empty_run_diagnose_lines(true, &windows, true, &hint, hint.metrics_url).join("\n");
        assert!(text.contains("Server is idle"));
        assert!(text.contains("benchmark_serving.py"));
        assert!(text.contains("Note: 2 of 2 windows were idle (excluded from analysis)."));
        assert!(!text.contains("(telemetry failure)"));
    }

    #[test]
    fn test_unreachable_shows_curl_command() {
        let url = "http://broken:8000/metrics";
        let hint = LoadHintParams {
            model_name: None,
            metrics_url: url,
            max_num_seqs: None,
            duration_secs: 30,
        };
        let text = empty_run_diagnose_lines(false, &[], false, &hint, url).join("\n");
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
        let text = empty_run_diagnose_lines(false, &[], true, &hint, hint.metrics_url).join("\n");
        assert!(text.contains("Server is idle"));
        assert!(text.contains("benchmark_serving.py"));
        assert!(!text.contains("[!] Telemetry Failure"));
    }

    #[test]
    fn test_unreachable_verbose_shows_dropped_count() {
        let windows = vec![broken_window(), broken_window()];
        let url = "http://localhost:8000/metrics";
        let hint = LoadHintParams {
            model_name: None,
            metrics_url: url,
            max_num_seqs: None,
            duration_secs: 30,
        };
        let text = empty_run_diagnose_lines(true, &windows, false, &hint, url).join("\n");
        assert!(text.contains("[!] Telemetry Failure"));
        assert!(text.contains(
            "Note: 2 of 2 windows dropped (telemetry failure). Diagnosis may be incomplete."
        ));
    }

    #[test]
    fn test_unreachable_verbose_idle_windows_not_labeled_failure() {
        let windows = vec![idle_window(), idle_window()];
        let url = "http://localhost:8000/metrics";
        let hint = LoadHintParams {
            model_name: None,
            metrics_url: url,
            max_num_seqs: None,
            duration_secs: 30,
        };
        let text = empty_run_diagnose_lines(true, &windows, false, &hint, url).join("\n");
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
        let text = mu_diagnose_lines(
            12.5,
            Some(64.0),
            Some(0.0),
            Some(256),
            MuVariant::Starved,
            None,
        )
        .join("\n");
        assert!(text.contains("Requests  64 running, 0 waiting  (server not saturated)"));
        assert!(text.contains("Server is under-fed"));
        assert!(text.contains("Confidence: Medium"));
        assert!(!text.contains("GPU is idle"));
        let blip = mu_diagnose_lines(
            12.5,
            Some(64.0),
            Some(0.6),
            Some(256),
            MuVariant::Starved,
            None,
        )
        .join("\n");
        assert!(blip.contains("Requests  64 running, 0 waiting  (server not saturated)"));
        assert!(!blip.contains("1 waiting"));
    }

    #[test]
    fn mu_diagnose_lines_blocked_admission_variant() {
        let measured_low = mu_diagnose_lines(
            12.0,
            Some(10.0),
            Some(5.0),
            Some(256),
            MuVariant::BlockedAdmission { kv_pct: Some(10.0) },
            None,
        )
        .join("\n");
        assert!(measured_low.contains("Requests  10 running, 5 waiting  (246 of 256 seats free)"));
        assert!(measured_low.contains("seats are free and KV cache at 10% (low)"));
        assert!(measured_low.contains("Scheduler admission is blocked"));
        assert!(measured_low.contains("Raise --max-num-batched-tokens."));
        assert!(measured_low.contains("Confirm chunked prefill is enabled"));
        assert!(!measured_low.contains("or enable chunked prefill"));
        assert!(!measured_low.contains("Enable chunked prefill (--enable-chunked-prefill)."));
        assert!(
            measured_low.contains("Confidence: Low (cause inferred, token budget not observed)")
        );
        assert!(!measured_low.contains("server not saturated"));
        assert!(!measured_low.contains("KV gauge unavailable"));

        let measured_mid_low = mu_diagnose_lines(
            12.0,
            Some(10.0),
            Some(5.0),
            Some(256),
            MuVariant::BlockedAdmission { kv_pct: Some(30.0) },
            Some(true),
        )
        .join("\n");
        assert!(measured_mid_low.contains("KV cache at 30% (low)"));
        assert!(measured_mid_low.contains("Raise --max-num-batched-tokens."));
        assert!(!measured_mid_low.contains("chunked prefill"));

        let measured_near_full = mu_diagnose_lines(
            12.0,
            Some(10.0),
            Some(5.0),
            Some(256),
            MuVariant::BlockedAdmission { kv_pct: Some(85.0) },
            Some(false),
        )
        .join("\n");
        assert!(measured_near_full.contains("KV cache at 85%"));
        assert!(!measured_near_full.contains("(low)"));
        assert!(!measured_near_full.contains("KV cache is low"));
        assert!(measured_near_full.contains("Raise --max-num-batched-tokens."));
        assert!(measured_near_full.contains("Enable chunked prefill (--enable-chunked-prefill)."));
        assert!(!measured_near_full.contains("Confirm chunked prefill"));

        let unmeasured = mu_diagnose_lines(
            12.0,
            Some(10.0),
            Some(5.0),
            Some(256),
            MuVariant::BlockedAdmission { kv_pct: None },
            None,
        )
        .join("\n");
        assert!(unmeasured.contains("Requests  10 running, 5 waiting  (246 of 256 seats free)"));
        assert!(unmeasured.contains("Requests queue while seats are free."));
        assert!(!unmeasured.contains("KV cache"));
        assert!(unmeasured.contains("Scheduler admission is blocked"));
        assert!(unmeasured.contains("Raise --max-num-batched-tokens."));
        assert!(unmeasured.contains("Confirm chunked prefill is enabled"));
        assert!(unmeasured.contains("Confidence: Low (cause inferred; KV gauge unavailable)"));

        let no_max = mu_diagnose_lines(
            12.0,
            Some(10.0),
            Some(5.0),
            None,
            MuVariant::BlockedAdmission { kv_pct: Some(10.0) },
            Some(true),
        )
        .join("\n");
        assert!(no_max.contains("(seats free unknown)"));
        assert!(!no_max.contains("server not saturated"));
        assert!(!no_max.contains("chunked prefill"));
    }

    #[test]
    fn mu_diagnose_lines_gauge_missing_variant() {
        let text = mu_diagnose_lines(
            12.5,
            Some(64.0),
            None,
            Some(256),
            MuVariant::GaugeMissing,
            None,
        )
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

#[cfg(test)]
mod catalog_mismatch_note_tests {
    use super::catalog_state_mismatch_verbose_line;

    #[test]
    fn verbose_only_on_mismatch() {
        assert!(catalog_state_mismatch_verbose_line(Some((7, 3)), false).is_none());
        assert!(catalog_state_mismatch_verbose_line(None, true).is_none());
        let line = catalog_state_mismatch_verbose_line(Some((7, 3)), true).unwrap();
        assert_eq!(
            line,
            "Note: Catalog state 7 pages, observed 3; entry may be stale."
        );
    }
}

#[cfg(test)]
mod stuck_fix_reveal_tests {
    use super::{
        format_diagnose_rules_for_windows, neutralize_revealed_fix, reveal_fix_conflict_subline,
    };
    use crate::collectors::RawSnapshot;
    use crate::collectors::{GpuRawMetrics, VllmRawMetrics};
    use crate::context::{AnalysisInput, RuntimeWindow, StaticContext};
    use crate::engine::rules::{ENGINE_MIN_PERSISTENT_WINDOWS, rule_names};
    use crate::engine::{Recommendation, Report};
    use std::time::SystemTime;

    fn win() -> RuntimeWindow {
        let t = SystemTime::UNIX_EPOCH;
        RuntimeWindow::from_snapshot(RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: VllmRawMetrics {
                num_requests_running: Some(4.0),
                generation_tokens_per_sec: Some(50.0),
                window_duration_secs: Some(2.0),
                ..Default::default()
            },
            gpus: vec![GpuRawMetrics::default()],

            host_memory: None,
        })
    }

    fn rec(name: &'static str, impact: u8, confidence: f64, header: &str) -> Recommendation {
        rec_terminal(name, impact, confidence, header, false)
    }

    fn rec_terminal(
        name: &'static str,
        impact: u8,
        confidence: f64,
        header: &str,
        terminal: bool,
    ) -> Recommendation {
        Recommendation {
            rule_name: name,
            layer: 2,
            impact,
            confidence,
            display_lines: vec![header.to_string(), "    body".to_string()],
            terminal,
        }
    }

    fn report_with(primary: Recommendation, suppressed: Vec<Recommendation>) -> Report {
        Report {
            baseline: None,
            recommendations: vec![primary],
            suppressed_rules: suppressed
                .iter()
                .map(|r| (r.rule_name, rule_names::OOM_RISK))
                .collect(),
            suppressed_recs: suppressed,
            kv_max_seqs: None,
            catalog_state_mismatch: None,
            n_eval: ENGINE_MIN_PERSISTENT_WINDOWS,
            skipped_broken: 0,
            skipped_idle: 0,
            energy_skew_skipped: 0,
            gauge_missing: Default::default(),
            limiter_evidence: None,
        }
    }

    fn render(report: &Report, reveal: bool) -> String {
        let w = win();
        let windows = vec![w.clone(); ENGINE_MIN_PERSISTENT_WINDOWS];
        let ctx = StaticContext::default();
        let summary = AnalysisInput::new(&ctx, &windows[0]);
        format_diagnose_rules_for_windows(
            &windows,
            summary,
            report,
            false,
            "http://127.0.0.1:8000/metrics",
            30,
            reveal,
        )
        .join("\n")
    }

    #[test]
    fn limiter_line_for_quiet_report_skips_spec_suspected() {
        use crate::engine::limiter::{LimiterEvidence, SPEC_GUARD_LIMITER_LINE};
        let mut e = LimiterEvidence {
            kv_cache_mean_perc: Some(20.0),
            kv_cache_peak_perc: Some(20.0),
            mean_running: Some(50.0),
            mean_waiting: Some(0.0),
            ridge_batch_size: Some(100.0),
            mean_tpot_ms: Some(50.0),
            tpot_floor_ms: Some(10.0),
            effective_prompt_decode_ratio: Some(0.2),
            chunked_prefill_enabled: Some(false),
            headroom_pct: Some(50.0),
            n_eval: ENGINE_MIN_PERSISTENT_WINDOWS,
            ceiling_unknown_reason: None,
            spec_suspected: true,
        };
        assert!(super::limiter_line_for_quiet_report(&e).is_none());
        assert_eq!(
            crate::engine::limiter::limiter_line(&e).as_deref(),
            Some(SPEC_GUARD_LIMITER_LINE),
            "healthy exit / raw limiter_line still emits the decline line"
        );
        e.spec_suspected = false;
        e.headroom_pct = Some(2.0);
        let line = super::limiter_line_for_quiet_report(&e).expect("physics line");
        assert!(line.contains("Capped by hardware"));
    }

    #[test]
    fn rules_fired_path_does_not_print_spec_note_in_rules_block() {
        use crate::engine::limiter::{LimiterEvidence, SPEC_GUARD_WARNING_LINE};
        let mut report = report_with(
            rec(rule_names::KV_CACHE_PRESSURE, 5, 0.9, "KV Cache Pressure"),
            vec![],
        );
        report.limiter_evidence = Some(LimiterEvidence {
            kv_cache_mean_perc: Some(20.0),
            kv_cache_peak_perc: Some(20.0),
            mean_running: Some(50.0),
            mean_waiting: Some(0.0),
            ridge_batch_size: Some(100.0),
            mean_tpot_ms: Some(50.0),
            tpot_floor_ms: Some(10.0),
            effective_prompt_decode_ratio: Some(0.2),
            chunked_prefill_enabled: Some(false),
            headroom_pct: Some(50.0),
            n_eval: ENGINE_MIN_PERSISTENT_WINDOWS,
            ceiling_unknown_reason: None,
            spec_suspected: true,
        });
        let text = render(&report, false);
        assert!(text.contains("KV Cache Pressure"));
        assert!(
            !text.contains(SPEC_GUARD_WARNING_LINE),
            "rules block never owned the note; scoreboard does: {text}"
        );
    }

    #[test]
    fn triggered_warning_at_column_0_with_blanks_and_rank_order() {
        let report = report_with(
            rec(rule_names::OOM_RISK, 5, 0.9, "[!] OOM Risk"),
            vec![
                rec(
                    rule_names::KV_CACHE_PRESSURE,
                    5,
                    0.9,
                    "[!] KV Cache Pressure",
                ),
                rec(rule_names::UNDER_BATCHING, 3, 0.5, "[!] Under-batching"),
            ],
        );
        let text = render(&report, true);
        let warn = "Same issue after remeasure. Other possible causes shown below.";
        assert!(
            text.lines().any(|l| l == warn),
            "warning must be at column 0; got:\n{text}"
        );
        let oom = text.find("[!] OOM Risk").expect("primary");
        let warn_at = text.find(warn).expect("warning");
        let kv = text.find("[!] KV Cache Pressure").expect("suppressed hi");
        let ub = text.find("[!] Under-batching").expect("suppressed lo");
        assert!(oom < warn_at && warn_at < kv && kv < ub);
        // blank line before and after warning
        let before = &text[..warn_at];
        assert!(
            before.ends_with("\n\n") || before.ends_with("body\n\n"),
            "blank line before warning"
        );
        let after_warn = &text[warn_at + warn.len()..];
        assert!(after_warn.starts_with("\n\n"), "blank line after warning");
    }

    #[test]
    fn untriggered_identical_when_reveal_false() {
        let report = report_with(
            rec(rule_names::OOM_RISK, 5, 0.9, "[!] OOM Risk"),
            vec![rec(
                rule_names::KV_CACHE_PRESSURE,
                5,
                0.9,
                "[!] KV Cache Pressure",
            )],
        );
        let hidden = render(&report, false);
        assert!(!hidden.contains("Same issue after remeasure"));
        assert!(!hidden.contains("[!] KV Cache Pressure"));
        assert!(hidden.contains("[!] OOM Risk"));
    }

    #[test]
    fn triggered_but_no_suppressed_prints_nothing_extra() {
        let report = report_with(rec(rule_names::OOM_RISK, 5, 0.9, "[!] OOM Risk"), vec![]);
        let text = render(&report, true);
        assert!(!text.contains("Same issue after remeasure"));
    }

    #[test]
    fn first_iteration_reveal_false_hides_suppressed() {
        // diagnose.rs always passes reveal=false on the initial print.
        let report = report_with(
            rec(rule_names::OOM_RISK, 5, 0.9, "[!] OOM Risk"),
            vec![rec(
                rule_names::KV_CACHE_PRESSURE,
                5,
                0.9,
                "[!] KV Cache Pressure",
            )],
        );
        let text = render(&report, false);
        assert!(!text.contains("Same issue after remeasure"));
        assert!(!text.contains("[!] KV Cache Pressure"));
    }

    #[test]
    fn terminal_primary_reveals_on_first_fire() {
        let report = report_with(
            rec_terminal(rule_names::PREFILL_BOUND, 5, 0.9, "[!] Prefill-Bound", true),
            vec![rec(
                rule_names::UNDER_BATCHING,
                3,
                0.5,
                "[!] Under-batching",
            )],
        );
        let text = render(&report, false);
        let header = "The primary issue has no tuning fix on this server. Other possible causes shown below.";
        assert!(text.contains(header));
        assert!(text.contains("[!] Under-batching"));
        assert!(!text.contains("Same issue after remeasure"));
    }

    #[test]
    fn terminal_primary_empty_suppressed_no_header() {
        let report = report_with(
            rec_terminal(rule_names::PREFILL_BOUND, 5, 0.9, "[!] Prefill-Bound", true),
            vec![],
        );
        let text = render(&report, false);
        assert!(!text.contains("Other possible causes shown below"));
        assert!(text.contains("[!] Prefill-Bound"));
    }

    #[test]
    fn terminal_plus_reveal_true_one_section_terminal_header() {
        let report = report_with(
            rec_terminal(rule_names::PREFILL_BOUND, 5, 0.9, "[!] Prefill-Bound", true),
            vec![rec(
                rule_names::UNDER_BATCHING,
                3,
                0.5,
                "[!] Under-batching",
            )],
        );
        let text = render(&report, true);
        let terminal_header = "The primary issue has no tuning fix on this server. Other possible causes shown below.";
        assert_eq!(text.matches(terminal_header).count(), 1);
        assert!(!text.contains("Same issue after remeasure"));
        assert!(text.contains("[!] Under-batching"));
    }

    fn r1_style_rec() -> Recommendation {
        Recommendation {
            rule_name: rule_names::UNDER_BATCHING,
            layer: 4,
            impact: 3,
            confidence: 0.8,
            display_lines: vec![
                "[!] Under-batching: Insufficient Concurrency".to_string(),
                String::new(),
                "    Cause:".to_string(),
                "      Hardware capacity under-fed by client.".to_string(),
                String::new(),
                "    Fix:".to_string(),
                "      • Batch more requests or increase client concurrency (107 slots idle)."
                    .to_string(),
                String::new(),
                "    Expected: Higher throughput. TPOT stable until the GPU is fully fed, then it starts to rise."
                    .to_string(),
                "    Confidence: High".to_string(),
            ],
            terminal: false,
        }
    }

    #[test]
    fn r6_reveal_annotates_r1_fix_as_conflict() {
        let report = report_with(
            rec_terminal(rule_names::PREFILL_BOUND, 5, 0.9, "[!] Prefill-Bound", true),
            vec![r1_style_rec()],
        );
        let text = render(&report, false);
        assert!(text.contains("[?] Under-batching"));
        assert!(!text.contains("[!] Under-batching"));
        assert!(text.contains("Hardware capacity under-fed by client."));
        assert!(text.contains("increase client concurrency"));
        let bound_sub =
            reveal_fix_conflict_subline(rule_names::PREFILL_BOUND, rule_names::UNDER_BATCHING)
                .expect("bound conflict subline");
        assert!(text.contains(bound_sub.trim_start()));
        let fix = text.find("increase client concurrency").expect("fix");
        let conflict = text.find("Conflicts with primary above").expect("conflict");
        assert!(fix < conflict, "conflict subline trails the Fix bullet");
        assert!(!text.contains("Expected: Higher throughput"));
        assert!(text.contains("Confidence: High"));
    }

    #[test]
    fn soft_field_reveal_r6_under_r1_warns_scale_out_conflict() {
        // Soft reveal: R1 primary, Prefill secondary. Neutralize scale-out and
        // warn to read REQUESTS/CACHE before acting.
        let prefill = Recommendation {
            rule_name: rule_names::PREFILL_BOUND,
            layer: 5,
            impact: 5,
            confidence: 0.9,
            display_lines: vec![
                "[!] Prefill-Bound: Decode Starved by Prefill".to_string(),
                String::new(),
                "    Cause:".to_string(),
                "      Prefill FLOPs dominate.".to_string(),
                String::new(),
                "    Fix:".to_string(),
                "      • Disaggregate prefill and decode onto separate workers.".to_string(),
                "      • Add a replica to scale out.".to_string(),
                String::new(),
                "    Expected: Scale out.".to_string(),
                "    Confidence: High".to_string(),
            ],
            terminal: true,
        };
        let report = report_with(r1_style_rec(), vec![prefill]);
        let text = render(&report, true);
        assert!(text.contains("[?] Prefill-Bound"));
        assert!(!text.contains("[!] Prefill-Bound"));
        let soft_sub =
            reveal_fix_conflict_subline(rule_names::UNDER_BATCHING, rule_names::PREFILL_BOUND)
                .expect("soft conflict subline");
        assert!(text.contains(soft_sub.trim_start()));
        assert!(text.contains("REQUESTS and CACHE"));
        assert!(text.contains("Disaggregate prefill and decode"));
        assert!(!text.contains("Expected: Scale out"));
        assert!(!text.contains("adds load to a server at its compute wall"));
    }

    #[test]
    fn non_conflict_reveal_keeps_fix_bullets() {
        let report = report_with(
            rec(rule_names::OOM_RISK, 5, 0.9, "[!] OOM Risk"),
            vec![r1_style_rec()],
        );
        let text = render(&report, true);
        assert!(text.contains("[!] Under-batching"));
        assert!(text.contains("increase client concurrency"));
        assert!(!text.contains("Conflicts with primary above"));
    }

    #[test]
    fn neutralize_without_fix_header_is_passthrough() {
        let lines = vec!["[!] Under-batching".to_string(), "    body".to_string()];
        assert_eq!(neutralize_revealed_fix(&lines, "        conflict"), lines);
    }
}
