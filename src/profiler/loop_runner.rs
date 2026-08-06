use std::time::Duration;

use super::{DiagnoseResult, MaxNumSeqsPrompt, delta, drift, poll, run_diagnose, state::LoopState};
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine;
use crate::engine::rule_names;
use crate::output;

const EFFICIENCY_DISPLAY_MIN_PP: f64 = 0.05;
/// Absolute efficiency drop (pp) required before appending `worse`.
const EFFICIENCY_WORSE_MIN_PP: f64 = 1.0;
/// Relative throughput drop (%) required before appending `worse`.
const THROUGHPUT_WORSE_MIN_PCT: f64 = 5.0;
/// Absolute USD rise required before Cost/1M appends `worse`.
const COST_WORSE_THRESHOLD_USD: f64 = 0.01;
/// Absolute J/tok rise required before the energy line appends `worse`.
const JTOK_WORSE_THRESHOLD: f64 = 0.02;
/// TTFT avg materiality gate (ms).
const TTFT_WORSE_MIN_MS: f64 = 5.0;
/// TPOT avg materiality gate (ms).
const TPOT_WORSE_MIN_MS: f64 = 0.5;
/// Left label width for remeasure before→after lines (excludes leading `"  "`).
/// Sized for the longest label so Throughput, latency, and economics share one value column.
const DELTA_LABEL_WIDTH: usize = 18; // "Cost/1M output tok"

fn worse_suffix(material_regression: bool) -> &'static str {
    if material_regression { "  worse" } else { "" }
}

/// Whether the remeasure path prints an efficiency delta line after a baseline reset.
fn include_efficiency_delta(config_drifted: bool) -> bool {
    !config_drifted
}

/// Reveal suppressed alternatives when the same primary re-fires.
/// No delta condition: the operator applied a fix and the same diagnosis
/// returned; the alternatives are information regardless of what moved.
pub(crate) fn should_reveal_suppressed(
    previous_recommendation: &'static str,
    new_primary: Option<&'static str>,
    suppressed_recs_non_empty: bool,
) -> bool {
    suppressed_recs_non_empty && new_primary == Some(previous_recommendation)
}

/// Inputs for the interactive diagnose closed loop.
pub struct LoopRunnerInput<'a> {
    pub url: &'a str,
    pub max_num_seqs: u32,
    pub cost_per_hour: Option<f64>,
    pub tensor_parallel_size: u32,
    pub gpu_indices: Vec<u32>,
    pub duration: Duration,
    pub initial_result: DiagnoseResult,
    pub initial_report: engine::Report,
    pub verbose_rules: bool,
    pub max_num_seqs_prompt: &'a mut dyn MaxNumSeqsPrompt,
}

pub fn run(input: LoopRunnerInput<'_>) -> anyhow::Result<()> {
    let LoopRunnerInput {
        url,
        max_num_seqs,
        cost_per_hour,
        tensor_parallel_size,
        gpu_indices,
        duration,
        initial_result,
        initial_report,
        verbose_rules,
        max_num_seqs_prompt,
    } = input;

    let mut last_fingerprint = initial_result.snapshot.fingerprint();
    let mut state = LoopState::new(initial_result, initial_report);
    let stdin_rx = poll::spawn_stdin_watcher();
    let mut current_max_num_seqs = max_num_seqs;

    loop {
        let (rule_name, prev_result, prev_report, primary_terminal, suppressed_empty) = {
            let Some(last_state) = state.last() else {
                break;
            };
            let Some(rule_name) = last_state
                .report
                .recommendations
                .first()
                .map(|r| r.rule_name)
            else {
                let baseline = last_state.report.baseline.as_ref();
                let efficiency = baseline.and_then(|b| b.efficiency_pct);
                let enforce_eager = last_state.result.static_ctx.config.enforce_eager;
                let enable_prefix_caching =
                    last_state.result.static_ctx.config.enable_prefix_caching;
                let quantization = last_state.result.static_ctx.config.quantization.clone();
                let msg = healthy_exit_message(HealthyExitInput {
                    efficiency,
                    limiter_evidence: last_state.report.limiter_evidence.unwrap_or_default(),
                    n_eval: last_state.report.n_eval,
                    enforce_eager,
                    enable_prefix_caching,
                    quantization,
                });
                println!("\n{msg}");
                break;
            };
            let primary_terminal = last_state
                .report
                .recommendations
                .first()
                .is_some_and(|r| r.terminal);
            let suppressed_empty = last_state.report.suppressed_recs.is_empty();
            (
                rule_name,
                last_state.result.clone(),
                last_state.report.clone(),
                primary_terminal,
                suppressed_empty,
            )
        };

        if should_exit_terminal_wall(primary_terminal, suppressed_empty) {
            // Terminal wall with no alternatives: table already printed; exit.
            // No operator prompt: there is no server-local knob to apply.
            println!();
            let limiter = prev_report
                .limiter_evidence
                .as_ref()
                .and_then(crate::engine::limiter::limiter_line);
            for line in terminal_wall_close_lines(limiter.as_deref()) {
                println!("{line}");
            }
            break;
        }

        if state.is_oscillating() {
            match state.oscillating_pair() {
                Some((a, b)) if is_kv_r5_oscillation_pair(a, b) => {
                    let known = kv_r5_known_bracket(state.history());
                    let bounds = kv_r5_midpoint_bounds(state.history());
                    match bounds {
                        Some((lo, hi)) if state.should_suggest_midpoint(lo, hi) => {
                            let (bound_line, try_line) = kv_r5_midpoint_suggestion(lo, hi);
                            println!("{bound_line}");
                            println!("{try_line}");
                            state.record_midpoint_suggestion(lo, hi);
                        }
                        _ => {
                            let reason = kv_r5_dead_end_reason(bounds, known, &state);
                            let (dead_line, replica_line) = kv_r5_dead_end_lines(reason);
                            println!("{dead_line}");
                            println!("{replica_line}");
                            break;
                        }
                    }
                }
                _ => {}
            }
        }

        if state.iteration_count() >= super::state::MAX_LOOP_ITERATIONS {
            println!("{}", iteration_limit_message());
            break;
        }

        state.record_recommendation(rule_name);

        let _outcome = poll::wait_for_restart_or_skip(url, &stdin_rx);

        println!();
        current_max_num_seqs = max_num_seqs_prompt.ask(current_max_num_seqs, &stdin_rx)?;

        println!("\nMeasuring delta...\n");
        let new_result = run_diagnose(
            url,
            Some(current_max_num_seqs),
            cost_per_hour,
            tensor_parallel_size,
            gpu_indices.clone(),
            duration,
        )?;
        let agg_win = RuntimeWindow::from_snapshot(new_result.snapshot.clone());
        let summary = AnalysisInput::new(&new_result.static_ctx, &agg_win);
        let new_report = engine::build_report_for_diagnose(&new_result.windows, summary);
        if let Some(msg) = mid_loop_abort_message(
            new_result.any_evaluable,
            new_result.all_idle,
            new_report.n_eval,
            new_report.skipped_broken,
            new_report.skipped_idle,
        ) {
            println!("{msg}");
            break;
        }
        last_fingerprint = verified_pass_fingerprint(&last_fingerprint, &new_result.snapshot)?;

        let drifted = drift::config_changed(&prev_result.static_ctx, &new_result.static_ctx);
        let non_baseline =
            drift::non_baseline_drifted(&prev_result.static_ctx, &new_result.static_ctx);
        let prescribed_lines = prev_report
            .recommendations
            .first()
            .map(|r| r.display_lines.as_slice());
        let unverifiable = !drifted
            && !non_baseline
            && drift::change_unverifiable(
                &prev_result.static_ctx.config,
                &new_result.static_ctx.config,
                prescribed_lines,
            );

        let d = delta::compute(
            &prev_result,
            &prev_report,
            &new_result,
            &new_report,
            drifted,
            non_baseline,
            unverifiable,
        );
        print_delta(&d);
        println!();
        let new_primary = new_report.recommendations.first().map(|r| r.rule_name);
        let reveal_suppressed = should_reveal_suppressed(
            rule_name,
            new_primary,
            !new_report.suppressed_recs.is_empty(),
        );
        output::stdout::print_diagnose_table_with_report(
            &new_result,
            &new_report,
            &agg_win,
            verbose_rules,
            reveal_suppressed,
        );

        state.push(new_result, new_report, Some(rule_name));
    }

    Ok(())
}

/// Mid-loop remeasure abort copy. Pure so the gate is unit-testable.
pub(crate) fn mid_loop_abort_message(
    any_evaluable: bool,
    all_idle: bool,
    n_eval: usize,
    skipped_broken: usize,
    skipped_idle: usize,
) -> Option<String> {
    let captured = crate::engine::format_captured_windows(n_eval, skipped_broken, skipped_idle);
    if !any_evaluable {
        return Some(format!(
            "\nProfile could not extract metrics mid-loop. {captured} Is the server still running?"
        ));
    }
    if all_idle {
        return Some(format!(
            "\nServer went idle mid-loop. {captured} Send continuous traffic."
        ));
    }
    if n_eval < crate::engine::ENGINE_MIN_PERSISTENT_WINDOWS {
        return Some(format!(
            "\nInsufficient data to verify. Required: {} windows. {captured}",
            crate::engine::ENGINE_MIN_PERSISTENT_WINDOWS
        ));
    }
    None
}

fn iteration_limit_message() -> String {
    format!(
        "\nIteration limit ({}) reached.",
        super::state::MAX_LOOP_ITERATIONS
    )
}

pub(crate) fn terminal_exit_message() -> &'static str {
    "Exiting: the primary bottleneck has no tuning fix on this server. Scale out or accept the ceiling."
}

/// Terminal primary with no suppressed alternatives: exit; do not prompt.
pub(crate) fn should_exit_terminal_wall(primary_terminal: bool, suppressed_empty: bool) -> bool {
    primary_terminal && suppressed_empty
}

/// Limiter line (context) then exit close (conclusion). No fabricated "Capped by".
pub(crate) fn terminal_wall_close_lines(limiter_line: Option<&str>) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(line) = limiter_line {
        lines.push(line.to_string());
    }
    lines.push(terminal_exit_message().to_string());
    lines
}

fn is_kv_r5_oscillation_pair(a: &str, b: &str) -> bool {
    matches!(
        (a, b),
        (
            rule_names::KV_CACHE_PRESSURE,
            rule_names::CONCURRENCY_SATURATION
        ) | (
            rule_names::CONCURRENCY_SATURATION,
            rule_names::KV_CACHE_PRESSURE
        )
    )
}

/// Shared bound-line wording for midpoint suggestion and named dead end.
fn kv_r5_bracket_line(lo: u32, hi: u32) -> String {
    format!("\n--max-num-seqs={hi} filled KV cache. --max-num-seqs={lo} saturated the queue.")
}

/// Bound line and try line for the R2/R5 oscillation midpoint escape.
pub(crate) fn kv_r5_midpoint_suggestion(lo: u32, hi: u32) -> (String, String) {
    let mid = (lo + hi) / 2;
    (
        kv_r5_bracket_line(lo, hi),
        format!("Try --max-num-seqs={mid}."),
    )
}

/// Why the R2/R5 oscillation path exits without another midpoint suggestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvR5DeadEndReason {
    /// Same `(lo, hi)` bracket already got a midpoint (case C).
    RepeatBracket { lo: u32, hi: u32 },
    /// Different bracket, same midpoint as last Try line (case D).
    RepeatMidpoint { lo: u32, hi: u32, mid: u32 },
    /// Session cap reached with a wide bracket still available (case E).
    CapExhausted { lo: u32, hi: u32 },
    /// Bracket too tight after at least one midpoint was tried (case F).
    TooCloseAfterMidpoint { lo: u32, hi: u32 },
    /// First oscillation already too tight to suggest a midpoint (case G).
    TooCloseFirst { lo: u32, hi: u32 },
    /// Could not resolve both seat counts from history (case H).
    SeatsUnknown,
}

/// Classify the dead-end path from bounds, known bracket, and session state.
pub(crate) fn kv_r5_dead_end_reason(
    bounds: Option<(u32, u32)>,
    known: Option<(u32, u32)>,
    state: &super::state::LoopState,
) -> KvR5DeadEndReason {
    if let Some((lo, hi)) = bounds {
        if state.last_bracket() == Some((lo, hi)) {
            return KvR5DeadEndReason::RepeatBracket { lo, hi };
        }
        let mid = lo.saturating_add(hi) / 2;
        if state.last_midpoint() == Some(mid) {
            return KvR5DeadEndReason::RepeatMidpoint { lo, hi, mid };
        }
        return KvR5DeadEndReason::CapExhausted { lo, hi };
    }
    if state.midpoint_count() > 0
        && let Some((lo, hi)) = known
    {
        return KvR5DeadEndReason::TooCloseAfterMidpoint { lo, hi };
    }
    if let Some((lo, hi)) = known {
        return KvR5DeadEndReason::TooCloseFirst { lo, hi };
    }
    KvR5DeadEndReason::SeatsUnknown
}

/// Dead-end lines when no further midpoint is offered.
pub(crate) fn kv_r5_dead_end_lines(reason: KvR5DeadEndReason) -> (String, String) {
    match reason {
        KvR5DeadEndReason::RepeatBracket { lo, hi } => {
            let mid = lo.saturating_add(hi) / 2;
            (
                kv_r5_bracket_line(lo, hi),
                format!("Bracket unchanged after midpoint {mid}. Add a replica to scale out."),
            )
        }
        KvR5DeadEndReason::RepeatMidpoint { lo, hi, mid } => (
            kv_r5_bracket_line(lo, hi),
            format!("Midpoint {mid} already suggested. Add a replica to scale out."),
        ),
        KvR5DeadEndReason::CapExhausted { lo, hi } => (
            kv_r5_bracket_line(lo, hi),
            format!(
                "Midpoint cap ({}) reached. Add a replica to scale out.",
                super::state::MAX_MIDPOINT_SUGGESTIONS
            ),
        ),
        KvR5DeadEndReason::TooCloseAfterMidpoint { lo, hi } => (
            kv_r5_bracket_line(lo, hi),
            "Too close to try another value between them. Add a replica to scale out.".to_string(),
        ),
        KvR5DeadEndReason::TooCloseFirst { lo, hi } => {
            let gap = hi - lo;
            (
                kv_r5_bracket_line(lo, hi),
                format!(
                    "Only {gap} seat{} apart; no midpoint to try. Add a replica to scale out.",
                    if gap == 1 { "" } else { "s" }
                ),
            )
        }
        KvR5DeadEndReason::SeatsUnknown => (
            "\nNo --max-num-seqs value resolves both KV pressure and queue saturation.".to_string(),
            "Add a replica to scale out.".to_string(),
        ),
    }
}

/// Last seat counts from R5 (low) and R2 (high), when both are known.
pub(crate) fn kv_r5_known_bracket(
    history: &std::collections::VecDeque<super::state::IterationRecord>,
) -> Option<(u32, u32)> {
    let lo = history
        .iter()
        .rev()
        .find(|r| {
            r.report
                .recommendations
                .first()
                .is_some_and(|rec| rec.rule_name == rule_names::CONCURRENCY_SATURATION)
        })
        .and_then(|r| r.result.static_ctx.config.max_num_seqs);
    let hi = history
        .iter()
        .rev()
        .find(|r| {
            r.report
                .recommendations
                .first()
                .is_some_and(|rec| rec.rule_name == rule_names::KV_CACHE_PRESSURE)
        })
        .and_then(|r| r.result.static_ctx.config.max_num_seqs);
    match (lo, hi) {
        (Some(lo), Some(hi)) if hi > lo => Some((lo, hi)),
        _ => None,
    }
}

/// Bisectable bracket only: both seats known and `hi > lo + 2`.
pub(crate) fn kv_r5_midpoint_bounds(
    history: &std::collections::VecDeque<super::state::IterationRecord>,
) -> Option<(u32, u32)> {
    kv_r5_known_bracket(history).filter(|&(lo, hi)| hi > lo + 2)
}

fn capacity_levers(enable_prefix_caching: Option<bool>) -> String {
    let mut levers = Vec::new();
    if enable_prefix_caching != Some(true) {
        levers.push("enable prefix caching");
    }
    levers.push("apply KV quantization (FP8; affects output quality)");
    levers.push("add TP to split KV cache");
    format!("Levers: {}", levers.join(", "))
}

fn physics_levers(quantization: &Option<String>) -> String {
    let mut levers = Vec::new();
    if quantization.is_none() {
        levers.push("quantize further (FP16→FP8/AWQ)");
    }
    levers.push("speculative decoding");
    levers.push("scale out with TP");
    format!("Levers: {}", levers.join(", "))
}

fn framework_overhead_levers(enforce_eager: Option<bool>) -> String {
    let mut levers = Vec::new();
    if enforce_eager != Some(true) {
        levers.push("test --enforce-eager");
    }
    levers.push("verify CPU/PCIe bottlenecks");
    levers.push("evaluate SGLang for this workload");
    format!("Levers: {}", levers.join(", "))
}

struct HealthyExitInput {
    efficiency: Option<f64>,
    limiter_evidence: engine::limiter::LimiterEvidence,
    n_eval: usize,
    enforce_eager: Option<bool>,
    enable_prefix_caching: Option<bool>,
    quantization: Option<String>,
}

fn healthy_exit_message(input: HealthyExitInput) -> String {
    let HealthyExitInput {
        efficiency,
        mut limiter_evidence,
        n_eval,
        enforce_eager,
        enable_prefix_caching,
        quantization,
    } = input;
    // Single source of truth for the window trust bar (identify reads evidence.n_eval).
    limiter_evidence.n_eval = n_eval;
    let limiter_line = engine::limiter::limiter_line(&limiter_evidence);
    let limiter = engine::limiter::identify(&limiter_evidence).verdict;

    let eff_str = efficiency
        .map(|e| format!("Efficiency: {e:.1}%"))
        .unwrap_or_else(|| "Efficiency: unavailable".to_string());

    let limiter_block = match limiter {
        Some(engine::limiter::LimiterVerdict::Known(engine::limiter::PrimaryLimiter::Capacity)) => {
            let levers = capacity_levers(enable_prefix_caching);
            format!(
                "Primary Limiter: KV Cache Capacity\n\
                 {eff_str}\n\
                 {}\n\
                 {levers}",
                limiter_line.as_deref().unwrap_or_default()
            )
        }
        Some(engine::limiter::LimiterVerdict::Known(engine::limiter::PrimaryLimiter::Physics)) => {
            let levers = physics_levers(&quantization);
            format!(
                "Primary Limiter: Physics (Hardware Ceiling)\n\
                 {eff_str}\n\
                 {}\n\
                 {levers}",
                limiter_line.as_deref().unwrap_or_default()
            )
        }
        Some(engine::limiter::LimiterVerdict::Known(
            engine::limiter::PrimaryLimiter::PrefillInterference,
        )) => {
            format!(
                "Primary Limiter: Prefill Interference\n\
                 {eff_str}\n\
                 {}\n\
                 Levers: disaggregate prefill/decode onto separate workers, or tune chunk size.",
                limiter_line.as_deref().unwrap_or_default()
            )
        }
        Some(engine::limiter::LimiterVerdict::Known(
            engine::limiter::PrimaryLimiter::FrameworkOverhead,
        )) => {
            let levers = framework_overhead_levers(enforce_eager);
            format!(
                "Primary Limiter: Framework Overhead\n\
                 {eff_str}\n\
                 {}\n\
                 {levers}",
                limiter_line.as_deref().unwrap_or_default()
            )
        }
        Some(engine::limiter::LimiterVerdict::Known(engine::limiter::PrimaryLimiter::Traffic)) => {
            format!(
                "Primary Limiter: Traffic\n\
                 {eff_str}\n\
                 {}",
                limiter_line.as_deref().unwrap_or_default()
            )
        }
        Some(engine::limiter::LimiterVerdict::CeilingUnknown(_)) => {
            limiter_line.unwrap_or_else(|| {
                "Hardware ceiling unknown (hardware ceiling inputs incomplete).".to_string()
            })
        }
        None => {
            format!("{eff_str} - insufficient data to identify primary limiter.")
        }
    };

    let prefix = match limiter {
        Some(_) => "Rules clear. No actionable config fix identified.",
        None => "Rules clear.",
    };
    format!("{prefix}\n\n{limiter_block}")
}

fn format_efficiency_delta_line(delta_pp: Option<f64>) -> Option<String> {
    match delta_pp {
        Some(v) if v >= EFFICIENCY_DISPLAY_MIN_PP => Some(format!("  Decode eff. +{v:.1}pp")),
        Some(v) if v <= -EFFICIENCY_DISPLAY_MIN_PP => Some(format!(
            "  Decode eff. {v:.1}pp{}",
            worse_suffix(v <= -EFFICIENCY_WORSE_MIN_PP)
        )),
        _ => None,
    }
}

fn format_throughput_delta_line(before: f64, after: f64) -> String {
    let drop_pct = if before > 0.0 {
        (before - after) / before * 100.0
    } else {
        0.0
    };
    let worse = after < before && drop_pct >= THROUGHPUT_WORSE_MIN_PCT;
    format!(
        "  {:<DELTA_LABEL_WIDTH$}  {before:.0} → {after:.0} tok/s{}",
        "Throughput",
        worse_suffix(worse)
    )
}

fn format_ttft_delta_line(
    before: f64,
    after: f64,
    p95_before: Option<f64>,
    p95_after: Option<f64>,
) -> Option<String> {
    let delta = after - before;
    if delta.abs() <= TTFT_WORSE_MIN_MS {
        return None;
    }
    let p95_suffix = match (p95_before, p95_after) {
        (Some(pb), Some(pa)) if pb.is_finite() && pa.is_finite() => {
            format!(" (p95 {pb:.0} → {pa:.0}ms)")
        }
        _ => String::new(),
    };
    Some(format!(
        "  {:<DELTA_LABEL_WIDTH$}  {before:.0} → {after:.0}ms{p95_suffix}{}",
        "TTFT",
        worse_suffix(delta > TTFT_WORSE_MIN_MS)
    ))
}

fn format_tpot_delta_line(
    before: f64,
    after: f64,
    p95_before: Option<f64>,
    p95_after: Option<f64>,
) -> Option<String> {
    let delta = after - before;
    if delta.abs() <= TPOT_WORSE_MIN_MS {
        return None;
    }
    let p95_suffix = match (p95_before, p95_after) {
        (Some(pb), Some(pa)) if pb.is_finite() && pa.is_finite() => {
            format!(" (p95 {pb:.1} → {pa:.1}ms)")
        }
        _ => String::new(),
    };
    Some(format!(
        "  {:<DELTA_LABEL_WIDTH$}  {before:.1} → {after:.1}ms{p95_suffix}{}",
        "TPOT",
        worse_suffix(delta > TPOT_WORSE_MIN_MS)
    ))
}

fn format_jtok_delta_line(before: f64, after: f64) -> String {
    let worse = after - before > JTOK_WORSE_THRESHOLD;
    format!(
        "  {:<DELTA_LABEL_WIDTH$}  {before:.2} → {after:.2}{}",
        "J/tok",
        worse_suffix(worse)
    )
}

fn format_cost_delta_line(before: f64, after: f64, est_suffix: &str) -> String {
    let worse = after - before > COST_WORSE_THRESHOLD_USD;
    format!(
        "  {:<DELTA_LABEL_WIDTH$}  ${before:.2} → ${after:.2}{est_suffix}{}",
        "Cost/1M output tok",
        worse_suffix(worse)
    )
}

/// One status line for the remeasure delta header.
fn config_status_lines(
    config_drifted: bool,
    non_baseline_drifted: bool,
    change_unverifiable: bool,
) -> Vec<String> {
    let line = if config_drifted {
        "  Config changed. Baseline reset.".to_string()
    } else if non_baseline_drifted {
        "  Config changed.".to_string()
    } else if change_unverifiable {
        "  Change unverifiable; delta may include it.".to_string()
    } else {
        "  No change detected.".to_string()
    };
    vec![line, String::new()]
}

fn remeasure_delta_lines(d: &delta::Delta) -> Vec<String> {
    let mut lines = config_status_lines(
        d.config_drifted,
        d.non_baseline_drifted,
        d.change_unverifiable,
    );
    if let (Some(before), Some(after)) = (d.throughput_before, d.throughput_after)
        && before.is_finite()
        && after.is_finite()
    {
        lines.push(format_throughput_delta_line(before, after));
    }
    if let (Some(before), Some(after)) = (d.ttft_before_ms, d.ttft_after_ms)
        && before.is_finite()
        && after.is_finite()
        && let Some(line) =
            format_ttft_delta_line(before, after, d.ttft_p95_before_ms, d.ttft_p95_after_ms)
    {
        lines.push(line);
    }
    if let (Some(before), Some(after)) = (d.tpot_before_ms, d.tpot_after_ms)
        && before.is_finite()
        && after.is_finite()
        && let Some(line) =
            format_tpot_delta_line(before, after, d.tpot_p95_before_ms, d.tpot_p95_after_ms)
    {
        lines.push(line);
    }
    if include_efficiency_delta(d.config_drifted)
        && let Some(line) = format_efficiency_delta_line(d.efficiency_delta_pp)
    {
        lines.push(line);
    }
    lines
}

fn print_delta(d: &delta::Delta) {
    for line in remeasure_delta_lines(d) {
        println!("{line}");
    }
    if economics_section_active(d) {
        println!();
        println!("ECONOMICS:");
        if let (Some(before), Some(after)) = (d.joules_per_token_before, d.joules_per_token_after) {
            println!("{}", format_jtok_delta_line(before, after));
        }
        if let (Some(before), Some(after)) = (d.cost_per_million_before, d.cost_per_million_after) {
            let est = match d.cost_source_after {
                Some(engine::CostSource::Catalog) | None => " (est)",
                _ => "",
            };
            println!("{}", format_cost_delta_line(before, after, est));
        }
    }
}

fn economics_section_active(d: &delta::Delta) -> bool {
    let cost_line = d.cost_per_million_before.is_some() && d.cost_per_million_after.is_some();
    let jtok_line = d.joules_per_token_before.is_some() && d.joules_per_token_after.is_some();
    cost_line || jtok_line
}

/// Closed-loop pass gate: fingerprint must match the prior iteration.
fn verified_pass_fingerprint(
    last: &crate::collectors::GpuFingerprint,
    snapshot: &crate::collectors::RawSnapshot,
) -> anyhow::Result<crate::collectors::GpuFingerprint> {
    let current = snapshot.fingerprint();
    crate::collectors::types::check_topology_drift(last, &current)?;
    Ok(current)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;
    use crate::collectors::test_fixtures::snap_with_gpu_indices;

    #[test]
    fn iteration_limit_message_states_cap_not_outcome() {
        let msg = iteration_limit_message();
        assert!(msg.contains("Iteration limit (20) reached."));
        assert!(!msg.contains("No further improvement"));
    }

    #[test]
    fn terminal_exit_message_snapshot() {
        assert_eq!(
            terminal_exit_message(),
            "Exiting: the primary bottleneck has no tuning fix on this server. Scale out or accept the ceiling."
        );
    }

    #[test]
    fn terminal_wall_exits_only_when_terminal_and_no_alternatives() {
        assert!(should_exit_terminal_wall(true, true));
        assert!(
            !should_exit_terminal_wall(true, false),
            "alternatives present: keep looping to the prompt"
        );
        assert!(!should_exit_terminal_wall(false, true));
        assert!(!should_exit_terminal_wall(false, false));
    }

    #[test]
    fn terminal_wall_close_lines_limiter_then_exit() {
        let lines = terminal_wall_close_lines(Some("Capped by Physics (Hardware Ceiling)."));
        assert_eq!(
            lines,
            vec![
                "Capped by Physics (Hardware Ceiling).".to_string(),
                terminal_exit_message().to_string(),
            ]
        );
        let lines = terminal_wall_close_lines(None);
        assert_eq!(lines, vec![terminal_exit_message().to_string()]);
    }

    #[test]
    fn healthy_exit_low_headroom_names_physics_ceiling_without_tpot_near_floor() {
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: Some(98.0),
            limiter_evidence: engine::limiter::LimiterEvidence {
                kv_cache_mean_perc: Some(50.0),
                kv_cache_peak_perc: Some(50.0),
                mean_running: Some(50.0),
                ridge_batch_size: Some(40.0),
                mean_tpot_ms: Some(50.0),
                tpot_floor_ms: Some(10.0),
                effective_prompt_decode_ratio: None,
                chunked_prefill_enabled: Some(false),
                headroom_pct: Some(2.0),
                n_eval: 0,
                ceiling_unknown_reason: None,
            },
            n_eval: 3,
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(msg.contains("Physics (Hardware Ceiling)"));
        assert!(msg.contains("headroom below 10%"));
        assert!(msg.contains("scale out"));
    }

    fn framework_overhead_input(enforce_eager: Option<bool>) -> HealthyExitInput {
        HealthyExitInput {
            efficiency: Some(60.0),
            limiter_evidence: engine::limiter::LimiterEvidence {
                kv_cache_mean_perc: Some(50.0),
                kv_cache_peak_perc: Some(50.0),
                mean_running: Some(50.0),
                ridge_batch_size: Some(40.0),
                mean_tpot_ms: Some(50.0),
                tpot_floor_ms: Some(10.0),
                effective_prompt_decode_ratio: None,
                chunked_prefill_enabled: Some(false),
                headroom_pct: None,
                n_eval: 0,
                ceiling_unknown_reason: None,
            },
            n_eval: 3,
            enforce_eager,
            enable_prefix_caching: None,
            quantization: None,
        }
    }

    fn capacity_input(enable_prefix_caching: Option<bool>) -> HealthyExitInput {
        HealthyExitInput {
            efficiency: Some(42.5),
            limiter_evidence: engine::limiter::LimiterEvidence {
                kv_cache_mean_perc: Some(85.0),
                kv_cache_peak_perc: Some(85.0),
                mean_running: Some(50.0),
                ridge_batch_size: Some(40.0),
                mean_tpot_ms: Some(20.0),
                tpot_floor_ms: Some(5.0),
                effective_prompt_decode_ratio: None,
                chunked_prefill_enabled: Some(false),
                headroom_pct: None,
                n_eval: 0,
                ceiling_unknown_reason: None,
            },
            n_eval: 3,
            enforce_eager: None,
            enable_prefix_caching,
            quantization: None,
        }
    }

    fn physics_input(quantization: Option<String>) -> HealthyExitInput {
        HealthyExitInput {
            efficiency: Some(91.0),
            limiter_evidence: engine::limiter::LimiterEvidence {
                kv_cache_mean_perc: Some(50.0),
                kv_cache_peak_perc: Some(50.0),
                mean_running: Some(50.0),
                ridge_batch_size: Some(40.0),
                mean_tpot_ms: Some(11.0),
                tpot_floor_ms: Some(10.0),
                effective_prompt_decode_ratio: None,
                chunked_prefill_enabled: Some(false),
                headroom_pct: None,
                n_eval: 0,
                ceiling_unknown_reason: None,
            },
            n_eval: 3,
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization,
        }
    }

    #[test]
    fn healthy_exit_capacity_limiter() {
        let msg = healthy_exit_message(capacity_input(None));
        assert!(msg.contains("KV Cache Capacity"));
        assert!(msg.contains("Efficiency: 42.5%"));
        assert!(msg.contains("KV cache at 85%"));
        assert!(msg.contains("No actionable config fix identified."));
    }

    #[test]
    fn healthy_exit_traffic_limiter() {
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: Some(34.0),
            limiter_evidence: engine::limiter::LimiterEvidence {
                kv_cache_mean_perc: Some(50.0),
                kv_cache_peak_perc: Some(50.0),
                mean_running: Some(5.0),
                ridge_batch_size: Some(100.0),
                mean_tpot_ms: Some(20.0),
                tpot_floor_ms: Some(5.0),
                effective_prompt_decode_ratio: None,
                chunked_prefill_enabled: Some(false),
                headroom_pct: None,
                n_eval: 0,
                ceiling_unknown_reason: None,
            },
            n_eval: 3,
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(msg.contains("Primary Limiter: Traffic"));
        assert!(msg.contains(
            "Capped by traffic: 5 requests running, hardware has room for ~100. More concurrent requests raises throughput."
        ));
        assert!(!msg.contains("--max-num-seqs"));
    }

    #[test]
    fn healthy_exit_traffic_with_queue_does_not_claim_traffic_gap() {
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: Some(12.4),
            limiter_evidence: engine::limiter::LimiterEvidence {
                kv_cache_mean_perc: Some(10.0),
                kv_cache_peak_perc: Some(10.0),
                mean_running: Some(10.0),
                ridge_batch_size: Some(100.0),
                mean_tpot_ms: Some(20.0),
                tpot_floor_ms: Some(5.0),
                effective_prompt_decode_ratio: None,
                chunked_prefill_enabled: Some(false),
                headroom_pct: None,
                n_eval: 0,
                ceiling_unknown_reason: None,
            },
            n_eval: 3,
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(msg.contains("Primary Limiter: Traffic"));
        assert!(msg.contains("Capped by traffic: 10 requests running"));
        assert!(!msg.contains("--max-num-seqs"));
    }

    #[test]
    fn traffic_limiter_does_not_say_optimally_tuned() {
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: Some(34.0),
            limiter_evidence: engine::limiter::LimiterEvidence {
                kv_cache_mean_perc: Some(50.0),
                kv_cache_peak_perc: Some(50.0),
                mean_running: Some(5.0),
                ridge_batch_size: Some(100.0),
                mean_tpot_ms: Some(20.0),
                tpot_floor_ms: Some(5.0),
                effective_prompt_decode_ratio: None,
                chunked_prefill_enabled: Some(false),
                headroom_pct: None,
                n_eval: 0,
                ceiling_unknown_reason: None,
            },
            n_eval: 3,
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(msg.contains("Capped by traffic:"));
        assert!(!msg.contains("--max-num-seqs"));
    }

    #[test]
    fn healthy_exit_physics_limiter() {
        let msg = healthy_exit_message(physics_input(None));
        assert!(msg.contains("Physics (Hardware Ceiling)"));
        assert!(msg.contains("Efficiency: 91.0%"));
        assert!(msg.contains("Capped by hardware: TPOT 11.0ms vs ~10.0ms floor."));
    }

    #[test]
    fn healthy_exit_prefill_interference_limiter() {
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: Some(55.0),
            limiter_evidence: engine::limiter::LimiterEvidence {
                kv_cache_mean_perc: Some(50.0),
                kv_cache_peak_perc: Some(50.0),
                mean_running: Some(50.0),
                ridge_batch_size: Some(40.0),
                mean_tpot_ms: Some(50.0),
                tpot_floor_ms: Some(10.0),
                effective_prompt_decode_ratio: Some(0.6),
                chunked_prefill_enabled: Some(true),
                headroom_pct: None,
                n_eval: 0,
                ceiling_unknown_reason: None,
            },
            n_eval: 3,
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(msg.contains("Prefill Interference"));
        assert!(msg.contains("Efficiency: 55.0%"));
        assert!(msg.contains("Capped by prefill: prompt work at 0.6x of decode (effective)."));
    }

    #[test]
    fn healthy_exit_framework_overhead_limiter() {
        let msg = healthy_exit_message(framework_overhead_input(None));
        assert!(msg.contains("Framework Overhead"));
        assert!(msg.contains("Efficiency: 60.0%"));
    }

    #[test]
    fn healthy_exit_reuses_same_limiter_line_as_quiet_report() {
        let ev = engine::limiter::LimiterEvidence {
            kv_cache_mean_perc: Some(84.0),
            kv_cache_peak_perc: Some(84.0),
            mean_running: Some(50.0),
            ridge_batch_size: Some(153.0),
            mean_tpot_ms: Some(20.0),
            tpot_floor_ms: Some(10.0),
            effective_prompt_decode_ratio: Some(0.2),
            chunked_prefill_enabled: Some(false),
            headroom_pct: None,
            n_eval: 3,
            ceiling_unknown_reason: None,
        };
        let line = engine::limiter::limiter_line(&ev).expect("limiter line");
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: Some(42.0),
            limiter_evidence: ev,
            n_eval: 3,
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(msg.contains(&line));
    }

    #[test]
    fn framework_overhead_levers_hide_enforce_eager_when_on() {
        let msg = healthy_exit_message(framework_overhead_input(Some(true)));
        assert!(msg.contains("Framework Overhead"));
        assert!(!msg.contains("--enforce-eager"));
        assert!(msg.contains("verify CPU/PCIe bottlenecks"));
        assert!(msg.contains("evaluate SGLang for this workload"));
    }

    #[test]
    fn framework_overhead_levers_show_enforce_eager_when_off_or_unknown() {
        for enforce_eager in [Some(false), None] {
            let msg = healthy_exit_message(framework_overhead_input(enforce_eager));
            assert!(
                msg.contains("--enforce-eager"),
                "expected enforce-eager lever for {enforce_eager:?}"
            );
        }
    }

    #[test]
    fn healthy_exit_insufficient_data_when_limiter_unknown() {
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: None,
            limiter_evidence: engine::limiter::LimiterEvidence::default(),
            n_eval: 1,
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(msg.contains("insufficient data to identify primary limiter"));
        assert!(msg.contains("Efficiency: unavailable"));
        assert!(!msg.contains("No actionable config fix identified."));
        assert_eq!(msg.matches("Rules clear.").count(), 1);
    }

    #[test]
    fn healthy_exit_sparse_n_eval_prints_no_limiter_verdict() {
        // Same evidence that would name Capacity at n_eval>=3; one window must decline.
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: Some(42.5),
            limiter_evidence: capacity_input(None).limiter_evidence,
            n_eval: 1,
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(!msg.contains("Primary Limiter:"));
        assert!(!msg.contains("Capped by"));
        assert!(msg.contains("insufficient data to identify primary limiter"));
    }

    #[test]
    fn capacity_levers_hide_prefix_caching_when_on() {
        let msg = healthy_exit_message(capacity_input(Some(true)));
        assert!(msg.contains("KV Cache Capacity"));
        assert!(!msg.contains("enable prefix caching"));
        assert!(msg.contains("KV quantization (FP8; affects output quality)"));
        assert!(msg.contains("add TP"));
    }

    #[test]
    fn capacity_levers_show_prefix_caching_when_off() {
        let msg = healthy_exit_message(capacity_input(None));
        assert!(msg.contains("enable prefix caching"));
    }

    #[test]
    fn physics_levers_hide_quantize_when_already_quantized() {
        let msg = healthy_exit_message(physics_input(Some("awq".to_string())));
        assert!(msg.contains("Physics (Hardware Ceiling)"));
        assert!(!msg.contains("quantize further"));
        assert!(msg.contains("speculative decoding"));
        assert!(msg.contains("scale out"));
    }

    #[test]
    fn physics_levers_show_quantize_when_unquantized() {
        let msg = healthy_exit_message(physics_input(None));
        assert!(msg.contains("quantize further"));
    }

    #[test]
    fn non_baseline_max_num_seqs_not_baseline_drift() {
        use crate::collectors::VllmConfig;
        use crate::context::StaticContext;
        let prev_cfg = VllmConfig {
            max_num_seqs: Some(32),
            ..Default::default()
        };
        let curr_cfg = VllmConfig {
            max_num_seqs: Some(98),
            ..Default::default()
        };
        let prev = StaticContext {
            config: prev_cfg,
            ..StaticContext::default()
        };
        let curr = StaticContext {
            config: curr_cfg,
            ..StaticContext::default()
        };
        assert!(!drift::config_changed(&prev, &curr));
        assert!(drift::non_baseline_drifted(&prev, &curr));
    }

    #[test]
    fn config_status_lines_baseline_beats_non_baseline() {
        let lines = config_status_lines(true, true, true);
        assert_eq!(lines[0], "  Config changed. Baseline reset.");
    }

    #[test]
    fn config_status_lines_non_baseline_beats_no_change() {
        let lines = config_status_lines(false, true, true);
        assert_eq!(lines[0], "  Config changed.");
    }

    #[test]
    fn config_status_lines_no_changes() {
        let lines = config_status_lines(false, false, false);
        assert_eq!(lines[0], "  No change detected.");
    }

    #[test]
    fn config_status_lines_unverifiable_when_no_drift() {
        let lines = config_status_lines(false, false, true);
        assert_eq!(lines[0], "  Change unverifiable; delta may include it.");
    }

    #[test]
    fn remeasure_delta_no_config_change_large_throughput_still_no_change_detected() {
        let mut d = flat_delta();
        d.throughput_before = Some(100.0);
        d.throughput_after = Some(300.0);
        let text = remeasure_delta_lines(&d).join("\n");
        assert!(text.starts_with("  No change detected."));
        assert!(text.contains("Throughput"));
    }

    #[test]
    fn remeasure_delta_unverifiable_header() {
        let mut d = flat_delta();
        d.change_unverifiable = true;
        let text = remeasure_delta_lines(&d).join("\n");
        assert!(text.starts_with("  Change unverifiable; delta may include it."));
        assert!(!text.contains("No change detected."));
    }

    #[test]
    fn efficiency_delta_near_zero_suppressed() {
        assert!(format_efficiency_delta_line(Some(-0.04)).is_none());
        assert!(format_efficiency_delta_line(Some(0.03)).is_none());
    }

    #[test]
    fn efficiency_delta_skipped_when_baseline_reset() {
        assert!(!include_efficiency_delta(true));
        assert!(include_efficiency_delta(false));
    }

    #[test]
    fn efficiency_delta_small_drop_prints_without_worse() {
        let line = format_efficiency_delta_line(Some(-0.06)).expect("line");
        assert!(!line.contains("worse"));
        assert!(line.contains("-0.1pp"));
        assert!(!line.contains('↓') && !line.contains('↑'));
    }

    #[test]
    fn worse_appears_on_material_regression() {
        let ttft = format_ttft_delta_line(98.0, 5793.0, Some(228.0), Some(10556.0)).unwrap();
        assert!(ttft.ends_with("  worse"));
        let tpot = format_tpot_delta_line(26.3, 47.8, Some(48.8), Some(85.0)).unwrap();
        assert!(tpot.ends_with("  worse"));
        assert!(format_throughput_delta_line(1000.0, 900.0).ends_with("  worse"));
        assert!(format_jtok_delta_line(0.70, 0.80).ends_with("  worse"));
        assert!(format_cost_delta_line(0.92, 1.00, " (est)").ends_with("  worse"));
        let eff = format_efficiency_delta_line(Some(-4.2)).unwrap();
        assert!(eff.ends_with("  worse"));
    }

    #[test]
    fn worse_absent_on_improvement() {
        assert!(!format_throughput_delta_line(230.0, 902.0).contains("worse"));
        let ttft = format_ttft_delta_line(5793.0, 98.0, Some(10556.0), Some(228.0)).unwrap();
        assert!(!ttft.contains("worse"));
        let tpot = format_tpot_delta_line(47.8, 26.3, Some(85.0), Some(48.8)).unwrap();
        assert!(!tpot.contains("worse"));
        assert!(!format_jtok_delta_line(2.23, 0.70).contains("worse"));
        assert!(!format_cost_delta_line(3.60, 0.92, " (est)").contains("worse"));
        let eff = format_efficiency_delta_line(Some(4.2)).unwrap();
        assert!(!eff.contains("worse"));
        assert_eq!(eff, "  Decode eff. +4.2pp");
    }

    #[test]
    fn worse_absent_below_gate() {
        // Throughput: 4% drop < 5% gate → silent.
        assert!(!format_throughput_delta_line(1000.0, 960.0).contains("worse"));
        // TTFT/TPOT: below materiality → no line.
        assert!(format_ttft_delta_line(100.0, 104.0, None, None).is_none());
        assert!(format_tpot_delta_line(26.0, 26.4, None, None).is_none());
        // Cost/Jtok: rise below absolute gate → silent.
        assert!(!format_cost_delta_line(2.00, 2.005, " (est)").contains("worse"));
        assert!(!format_jtok_delta_line(0.31, 0.32).contains("worse"));
        // Eff below display min → no line.
        assert!(format_efficiency_delta_line(Some(-0.04)).is_none());
        // Eff displayed but below 1.0pp worse gate → silent.
        let small = format_efficiency_delta_line(Some(-0.5)).unwrap();
        assert!(!small.contains("worse"));
        assert!(
            format_efficiency_delta_line(Some(-1.5))
                .unwrap()
                .ends_with("  worse")
        );
    }

    #[test]
    fn before_after_delta_labels_share_value_column() {
        // "  " + label(18) + "  " → value starts at byte index 22.
        const VALUE_COL: usize = 2 + DELTA_LABEL_WIDTH + 2;
        let lines = [
            format_throughput_delta_line(1000.0, 900.0),
            format_ttft_delta_line(98.0, 5793.0, None, None).unwrap(),
            format_tpot_delta_line(26.3, 47.8, None, None).unwrap(),
            format_jtok_delta_line(1.19, 1.28),
            format_cost_delta_line(2.04, 2.27, " (est)"),
        ];
        for line in &lines {
            assert_eq!(
                line.as_bytes().get(VALUE_COL - 1),
                Some(&b' '),
                "space before value: {line:?}"
            );
            assert_ne!(
                line.as_bytes().get(VALUE_COL),
                Some(&b' '),
                "value starts at col {VALUE_COL}: {line:?}"
            );
        }
        assert!(lines[3].as_bytes()[VALUE_COL].is_ascii_digit());
        assert_eq!(lines[4].as_bytes()[VALUE_COL], b'$');
    }

    #[test]
    fn p95_has_no_own_label() {
        let line = format_ttft_delta_line(98.0, 5793.0, Some(228.0), Some(10556.0)).unwrap();
        assert!(line.contains("(p95 228 → 10556ms)"));
        assert_eq!(line.matches("worse").count(), 1);
        assert!(line.ends_with("  worse"));
        let start = line.find("(p95").unwrap();
        let end = line.find(')').unwrap();
        let p95 = &line[start..=end];
        assert!(p95.ends_with("ms)"));
        assert!(!p95.contains("worse"));
        assert!(!p95.contains('↑') && !p95.contains('↓'));
        assert!(!p95.contains("improved") && !p95.contains("regressed"));
    }

    #[test]
    fn economics_header_shown_when_only_jtok_available() {
        let mut d = economics_delta_base();
        d.joules_per_token_before = Some(0.31);
        d.joules_per_token_after = Some(0.28);
        assert!(economics_section_active(&d));
    }

    #[test]
    fn economics_header_shown_when_only_cost_available() {
        let mut d = economics_delta_base();
        d.cost_per_million_before = Some(2.50);
        d.cost_per_million_after = Some(2.00);
        assert!(economics_section_active(&d));
    }

    #[test]
    fn economics_header_shown_when_both_metrics_available() {
        let mut d = economics_delta_base();
        d.joules_per_token_before = Some(0.31);
        d.joules_per_token_after = Some(0.28);
        d.cost_per_million_before = Some(2.50);
        d.cost_per_million_after = Some(2.00);
        assert!(economics_section_active(&d));
    }

    fn economics_delta_base() -> delta::Delta {
        delta::Delta {
            throughput_before: None,
            throughput_after: None,
            efficiency_delta_pp: None,
            efficiency_pct_before: None,
            efficiency_pct_after: None,
            cost_per_million_before: None,
            cost_per_million_after: None,
            joules_per_token_before: None,
            joules_per_token_after: None,
            cost_source_after: None,
            ttft_before_ms: None,
            ttft_after_ms: None,
            tpot_before_ms: None,
            tpot_after_ms: None,
            ttft_p95_before_ms: None,
            ttft_p95_after_ms: None,
            tpot_p95_before_ms: None,
            tpot_p95_after_ms: None,
            config_drifted: false,
            non_baseline_drifted: false,
            change_unverifiable: false,
        }
    }

    #[test]
    fn remeasure_delta_output_excludes_capacity_prescription() {
        let mut d = flat_delta();
        d.ttft_before_ms = Some(120.0);
        d.ttft_after_ms = Some(95.0);
        d.tpot_before_ms = Some(45.0);
        d.tpot_after_ms = Some(38.0);
        d.efficiency_delta_pp = Some(3.5);
        let text = remeasure_delta_lines(&d).join("\n");
        assert!(!text.contains("Capacity:"));
        assert!(!text.contains("prescribed"));
    }

    #[test]
    fn verified_pass_fingerprint_allows_unchanged_topology() {
        let last = snap_with_gpu_indices(&[0, 1]).fingerprint();
        let next = snap_with_gpu_indices(&[0, 1]);
        assert!(verified_pass_fingerprint(&last, &next).is_ok());
    }

    #[test]
    fn verified_pass_fingerprint_errors_on_topology_drift() {
        let last = snap_with_gpu_indices(&[0, 1]).fingerprint();
        let drifted = snap_with_gpu_indices(&[0, 2]);
        let err = verified_pass_fingerprint(&last, &drifted).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Topology drift detected"));
        assert!(msg.contains("missing [GPU-1]"));
        assert!(msg.contains("new [GPU-2]"));
    }

    #[test]
    fn mid_loop_abort_message_tests() {
        let crash = mid_loop_abort_message(false, false, 0, 15, 0).unwrap();
        assert!(crash.contains("could not extract metrics mid-loop"));
        assert!(crash.contains("Captured: 0 (15 dropped)."));
        let idle_msg = mid_loop_abort_message(true, true, 0, 0, 3).unwrap();
        assert!(idle_msg.contains("Server went idle mid-loop. Captured: 0 (3 idle)."));
        let sparse_msg = mid_loop_abort_message(true, false, 2, 3, 10).unwrap();
        assert!(sparse_msg.contains("Insufficient data to verify. Required: 3 windows."));
        assert!(sparse_msg.contains("Captured: 2 (3 dropped, 10 idle)."));
        assert_eq!(mid_loop_abort_message(true, false, 3, 0, 0), None);
    }

    fn flat_delta() -> delta::Delta {
        delta::Delta {
            throughput_before: Some(100.0),
            throughput_after: Some(101.0),  // < 5%
            efficiency_delta_pp: Some(0.5), // < plateau 2.0
            efficiency_pct_before: Some(10.0),
            efficiency_pct_after: Some(10.5),
            cost_per_million_before: None,
            cost_per_million_after: None,
            joules_per_token_before: None,
            joules_per_token_after: None,
            cost_source_after: None,
            ttft_before_ms: None,
            ttft_after_ms: None,
            tpot_before_ms: None,
            tpot_after_ms: None,
            ttft_p95_before_ms: None,
            ttft_p95_after_ms: None,
            tpot_p95_before_ms: None,
            tpot_p95_after_ms: None,
            config_drifted: false,
            non_baseline_drifted: false,
            change_unverifiable: false,
        }
    }

    #[test]
    fn reveal_when_same_primary_and_has_suppressed() {
        assert!(should_reveal_suppressed("oom_risk", Some("oom_risk"), true));
    }

    #[test]
    fn no_reveal_when_new_primary_none() {
        assert!(!should_reveal_suppressed("oom_risk", None, true));
    }

    #[test]
    fn no_reveal_when_primary_changed() {
        assert!(!should_reveal_suppressed(
            "oom_risk",
            Some("kv_cache_pressure"),
            true
        ));
    }

    #[test]
    fn no_reveal_when_no_suppressed() {
        assert!(!should_reveal_suppressed(
            "oom_risk",
            Some("oom_risk"),
            false
        ));
    }

    fn kv_bounds_empty_report() -> engine::Report {
        engine::Report {
            baseline: None,
            recommendations: Vec::new(),
            suppressed_rules: Vec::new(),
            suppressed_recs: Vec::new(),
            kv_max_seqs: None,
            catalog_state_mismatch: None,
            n_eval: 1,
            skipped_broken: 0,
            skipped_idle: 0,
            energy_skew_skipped: 0,
            gauge_missing: Default::default(),
            limiter_evidence: None,
        }
    }

    fn kv_bounds_report_firing(rule_name: &'static str) -> engine::Report {
        let mut report = kv_bounds_empty_report();
        report.recommendations.push(engine::Recommendation {
            rule_name,
            layer: 1,
            impact: 5,
            confidence: 0.9,
            display_lines: Vec::new(),
            terminal: false,
        });
        report
    }

    fn kv_bounds_diagnose(max_num_seqs: u32) -> DiagnoseResult {
        use crate::collectors::VllmRawMetrics;
        use crate::context::StaticContext;
        use std::time::{Duration, SystemTime};

        DiagnoseResult {
            snapshot: crate::collectors::RawSnapshot {
                gpu_observed_at: SystemTime::UNIX_EPOCH,
                vllm_observed_at: SystemTime::UNIX_EPOCH,
                timestamp: SystemTime::UNIX_EPOCH,
                vllm: VllmRawMetrics::default(),
                gpus: vec![],

                host_memory: None,
            },
            windows: Vec::new(),
            static_ctx: StaticContext {
                config: crate::collectors::VllmConfig {
                    max_num_seqs: Some(max_num_seqs),
                    ..Default::default()
                },
                ..Default::default()
            },
            duration: Duration::from_secs(2),
            started_at: SystemTime::UNIX_EPOCH,
            any_evaluable: true,
            all_idle: false,
            metrics_input: String::new(),
            energy_active_windows: 0,
            energy_pair_windows: 0,
        }
    }

    fn kv_bounds_record(
        max_num_seqs: u32,
        fired: &'static str,
        recommendation_shown: Option<&'static str>,
    ) -> super::super::state::IterationRecord {
        super::super::state::IterationRecord {
            result: kv_bounds_diagnose(max_num_seqs),
            report: kv_bounds_report_firing(fired),
            recommendation_shown,
        }
    }

    #[test]
    fn kv_r5_midpoint_bounds_kv_high_r5_low() {
        use super::super::state::IterationRecord;
        use std::collections::VecDeque;

        let mut history = VecDeque::new();
        history.push_back(IterationRecord {
            result: kv_bounds_diagnose(100),
            report: kv_bounds_empty_report(),
            recommendation_shown: None,
        });
        // Operator raised seats after R5; remeasure fired R2 at the higher cap.
        history.push_back(kv_bounds_record(
            345,
            rule_names::KV_CACHE_PRESSURE,
            Some(rule_names::CONCURRENCY_SATURATION),
        ));
        // Operator lowered seats after R2; remeasure fired R5 at the lower cap.
        history.push_back(kv_bounds_record(
            45,
            rule_names::CONCURRENCY_SATURATION,
            Some(rule_names::KV_CACHE_PRESSURE),
        ));

        let (lo, hi) = kv_r5_midpoint_bounds(&history).expect("bounds");
        assert_eq!(lo, 45);
        assert_eq!(hi, 345);
        assert_eq!((lo + hi) / 2, 195);
    }

    #[test]
    fn kv_r5_midpoint_bounds_saturation_before_kv_in_history() {
        use super::super::state::LoopState;
        use std::collections::VecDeque;

        let mut history = VecDeque::new();
        history.push_back(kv_bounds_record(
            45,
            rule_names::CONCURRENCY_SATURATION,
            Some(rule_names::KV_CACHE_PRESSURE),
        ));
        history.push_back(kv_bounds_record(
            345,
            rule_names::KV_CACHE_PRESSURE,
            Some(rule_names::CONCURRENCY_SATURATION),
        ));

        let (lo, hi) = kv_r5_midpoint_bounds(&history).expect("bounds");
        assert_eq!((lo, hi), (45, 345));

        let mut state = LoopState::new(kv_bounds_diagnose(345), kv_bounds_empty_report());
        state.record_recommendation(rule_names::CONCURRENCY_SATURATION);
        state.record_recommendation(rule_names::KV_CACHE_PRESSURE);
        state.record_recommendation(rule_names::CONCURRENCY_SATURATION);
        assert_eq!(
            state.oscillating_pair(),
            Some((
                rule_names::CONCURRENCY_SATURATION,
                rule_names::KV_CACHE_PRESSURE
            ))
        );
    }

    #[test]
    fn kv_r5_midpoint_bounds_uses_firing_rule_not_recommendation_shown() {
        use std::collections::VecDeque;

        let mut history = VecDeque::new();
        history.push_back(kv_bounds_record(
            345,
            rule_names::KV_CACHE_PRESSURE,
            Some(rule_names::CONCURRENCY_SATURATION),
        ));
        history.push_back(kv_bounds_record(
            45,
            rule_names::CONCURRENCY_SATURATION,
            Some(rule_names::KV_CACHE_PRESSURE),
        ));

        let (lo, hi) = kv_r5_midpoint_bounds(&history).expect("bounds");
        assert_eq!((lo, hi), (45, 345));
    }

    #[test]
    fn kv_r5_known_bracket_when_too_tight_to_bisect() {
        use std::collections::VecDeque;

        let mut history = VecDeque::new();
        history.push_back(kv_bounds_record(
            150,
            rule_names::CONCURRENCY_SATURATION,
            None,
        ));
        history.push_back(kv_bounds_record(152, rule_names::KV_CACHE_PRESSURE, None));

        assert_eq!(kv_r5_known_bracket(&history), Some((150, 152)));
        assert!(kv_r5_midpoint_bounds(&history).is_none());
    }

    #[test]
    fn kv_r5_midpoint_bounds_none_when_hi_within_margin_of_lo() {
        use std::collections::VecDeque;

        let mut history = VecDeque::new();
        history.push_back(kv_bounds_record(
            10,
            rule_names::CONCURRENCY_SATURATION,
            None,
        ));
        history.push_back(kv_bounds_record(12, rule_names::KV_CACHE_PRESSURE, None));

        assert_eq!(kv_r5_known_bracket(&history), Some((10, 12)));
        assert!(kv_r5_midpoint_bounds(&history).is_none());
    }

    #[test]
    fn kv_r5_midpoint_message_maps_hi_to_kv_and_lo_to_queue() {
        let (bound_line, try_line) = kv_r5_midpoint_suggestion(45, 345);
        assert!(bound_line.contains("--max-num-seqs=345 filled KV cache"));
        assert!(bound_line.contains("--max-num-seqs=45 saturated the queue"));
        assert_eq!(try_line, "Try --max-num-seqs=195.");
    }

    #[test]
    fn kv_r5_dead_end_repeat_bracket_case_c() {
        let (dead, replica) =
            kv_r5_dead_end_lines(KvR5DeadEndReason::RepeatBracket { lo: 150, hi: 180 });
        assert!(dead.contains("--max-num-seqs=180 filled KV cache"));
        assert!(dead.contains("--max-num-seqs=150 saturated the queue"));
        assert_eq!(
            replica,
            "Bracket unchanged after midpoint 165. Add a replica to scale out."
        );
    }

    #[test]
    fn kv_r5_dead_end_repeat_midpoint_case_d() {
        let (dead, replica) = kv_r5_dead_end_lines(KvR5DeadEndReason::RepeatMidpoint {
            lo: 151,
            hi: 179,
            mid: 165,
        });
        assert!(dead.contains("--max-num-seqs=179 filled KV cache"));
        assert!(dead.contains("--max-num-seqs=151 saturated the queue"));
        assert_eq!(
            replica,
            "Midpoint 165 already suggested. Add a replica to scale out."
        );
    }

    #[test]
    fn kv_r5_dead_end_cap_exhausted_case_e() {
        let (dead, replica) =
            kv_r5_dead_end_lines(KvR5DeadEndReason::CapExhausted { lo: 125, hi: 137 });
        assert!(dead.contains("--max-num-seqs=137 filled KV cache"));
        assert!(dead.contains("--max-num-seqs=125 saturated the queue"));
        assert_eq!(
            replica,
            "Midpoint cap (3) reached. Add a replica to scale out."
        );
    }

    #[test]
    fn dead_end_reason_cap_exhausted_case_e() {
        use super::super::state::{LoopState, MAX_MIDPOINT_SUGGESTIONS};

        let mut state = LoopState::new(kv_bounds_diagnose(137), kv_bounds_empty_report());
        state.record_midpoint_suggestion(100, 200);
        state.record_midpoint_suggestion(120, 200);
        state.record_midpoint_suggestion(140, 200);
        assert_eq!(state.midpoint_count(), MAX_MIDPOINT_SUGGESTIONS);

        let bounds = Some((125, 137));
        let known = Some((125, 137));
        assert!(!state.should_suggest_midpoint(125, 137));
        assert_eq!(
            kv_r5_dead_end_reason(bounds, known, &state),
            KvR5DeadEndReason::CapExhausted { lo: 125, hi: 137 }
        );
    }

    #[test]
    fn kv_r5_dead_end_too_close_after_midpoint_case_f() {
        let (dead, replica) =
            kv_r5_dead_end_lines(KvR5DeadEndReason::TooCloseAfterMidpoint { lo: 150, hi: 152 });
        assert!(dead.contains("--max-num-seqs=152 filled KV cache"));
        assert!(dead.contains("--max-num-seqs=150 saturated the queue"));
        assert_eq!(
            replica,
            "Too close to try another value between them. Add a replica to scale out."
        );
    }

    #[test]
    fn kv_r5_dead_end_too_close_first_gap_one() {
        let (dead, replica) =
            kv_r5_dead_end_lines(KvR5DeadEndReason::TooCloseFirst { lo: 150, hi: 151 });
        assert!(dead.contains("--max-num-seqs=151 filled KV cache"));
        assert!(dead.contains("--max-num-seqs=150 saturated the queue"));
        assert_eq!(
            replica,
            "Only 1 seat apart; no midpoint to try. Add a replica to scale out."
        );
    }

    #[test]
    fn kv_r5_dead_end_too_close_first_gap_two() {
        let (dead, replica) =
            kv_r5_dead_end_lines(KvR5DeadEndReason::TooCloseFirst { lo: 150, hi: 152 });
        assert!(dead.contains("--max-num-seqs=152 filled KV cache"));
        assert!(dead.contains("--max-num-seqs=150 saturated the queue"));
        assert_eq!(
            replica,
            "Only 2 seats apart; no midpoint to try. Add a replica to scale out."
        );
    }

    #[test]
    fn kv_r5_dead_end_generic_when_seats_unknown_case_h() {
        let (dead, replica) = kv_r5_dead_end_lines(KvR5DeadEndReason::SeatsUnknown);
        assert!(
            dead.contains(
                "No --max-num-seqs value resolves both KV pressure and queue saturation."
            )
        );
        assert_eq!(replica, "Add a replica to scale out.");
    }

    #[test]
    fn dead_end_reason_repeat_bracket_case_c() {
        use super::super::state::LoopState;

        let mut state = LoopState::new(kv_bounds_diagnose(180), kv_bounds_empty_report());
        state.record_midpoint_suggestion(150, 180);
        let bounds = Some((150, 180));
        let known = Some((150, 180));
        assert_eq!(
            kv_r5_dead_end_reason(bounds, known, &state),
            KvR5DeadEndReason::RepeatBracket { lo: 150, hi: 180 }
        );
    }

    #[test]
    fn dead_end_reason_repeat_midpoint_case_d() {
        use super::super::state::LoopState;

        let mut state = LoopState::new(kv_bounds_diagnose(180), kv_bounds_empty_report());
        state.record_midpoint_suggestion(150, 180);
        let bounds = Some((151, 179));
        let known = Some((151, 179));
        assert_eq!(
            kv_r5_dead_end_reason(bounds, known, &state),
            KvR5DeadEndReason::RepeatMidpoint {
                lo: 151,
                hi: 179,
                mid: 165,
            }
        );
    }

    #[test]
    fn dead_end_reason_too_close_after_midpoint_case_f() {
        use super::super::state::LoopState;
        use std::collections::VecDeque;

        let mut state = LoopState::new(kv_bounds_diagnose(180), kv_bounds_empty_report());
        state.record_midpoint_suggestion(150, 180);

        let mut history = VecDeque::new();
        history.push_back(kv_bounds_record(
            150,
            rule_names::CONCURRENCY_SATURATION,
            None,
        ));
        history.push_back(kv_bounds_record(152, rule_names::KV_CACHE_PRESSURE, None));
        let known = kv_r5_known_bracket(&history);
        let bounds = kv_r5_midpoint_bounds(&history);
        assert!(bounds.is_none());
        assert_eq!(known, Some((150, 152)));

        assert_eq!(
            kv_r5_dead_end_reason(bounds, known, &state),
            KvR5DeadEndReason::TooCloseAfterMidpoint { lo: 150, hi: 152 }
        );
    }

    #[test]
    fn dead_end_generic_when_known_absent_after_midpoint() {
        use super::super::state::LoopState;

        let mut state = LoopState::new(kv_bounds_diagnose(180), kv_bounds_empty_report());
        state.record_midpoint_suggestion(150, 180);
        assert_eq!(
            kv_r5_dead_end_reason(None, None, &state),
            KvR5DeadEndReason::SeatsUnknown
        );
    }

    #[test]
    fn dead_end_reason_too_close_first_case_g() {
        use super::super::state::LoopState;
        use std::collections::VecDeque;

        let state = LoopState::new(kv_bounds_diagnose(152), kv_bounds_empty_report());
        let mut history = VecDeque::new();
        history.push_back(kv_bounds_record(
            150,
            rule_names::CONCURRENCY_SATURATION,
            None,
        ));
        history.push_back(kv_bounds_record(152, rule_names::KV_CACHE_PRESSURE, None));
        let known = kv_r5_known_bracket(&history);
        let bounds = kv_r5_midpoint_bounds(&history);
        assert_eq!(known, Some((150, 152)));
        assert!(bounds.is_none());

        assert_eq!(
            kv_r5_dead_end_reason(bounds, known, &state),
            KvR5DeadEndReason::TooCloseFirst { lo: 150, hi: 152 }
        );
    }
}
