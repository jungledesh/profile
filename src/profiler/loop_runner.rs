use std::io::{self, Write};
use std::time::Duration;

use super::{delta, drift, poll, run_diagnose, state::LoopState, DiagnoseResult};
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine;
use crate::output;

const CEILING_HEADROOM_THRESHOLD_PCT: f64 = 10.0;
const LOW_OCCUPANCY_THRESHOLD: f64 = 0.25;
const EFFICIENCY_DISPLAY_MIN_PP: f64 = 0.05;

pub fn run(
    url: &str,
    max_num_seqs: Option<u32>,
    cost_per_hour: Option<f64>,
    tensor_parallel_size: Option<u32>,
    duration: Duration,
    initial_result: DiagnoseResult,
    initial_report: engine::Report,
) -> anyhow::Result<()> {
    let mut state = LoopState::new(initial_result, initial_report);
    let stdin_rx = poll::spawn_stdin_watcher();
    let mut current_max_num_seqs = max_num_seqs;

    loop {
        let Some(rule_name) = state
            .last()
            .report
            .groups
            .first()
            .map(|g| g.primary.rule_name)
        else {
            let baseline = state.last().report.baseline.as_ref();
            let efficiency = baseline.and_then(|b| b.efficiency_pct);
            let headroom = baseline.and_then(|b| b.headroom_pct);

            let num_running = state.last().result.snapshot.vllm.num_requests_running;
            let max_num_seqs = state.last().result.static_ctx.config.max_num_seqs;
            let msg = healthy_exit_message(efficiency, headroom, num_running, max_num_seqs);
            println!("\n{msg}");
            break;
        };

        if state.is_oscillating() {
            match state.oscillating_pair() {
                Some((a, b))
                    if (a == "kv_cache_pressure" && b == "concurrency_saturation")
                        || (a == "concurrency_saturation" && b == "kv_cache_pressure") =>
                {
                    if state.midpoint_suggested() {
                        println!(
                            "\nNo --max-num-seqs value resolves both KV pressure and queue saturation."
                        );
                        println!("Add a replica to scale out.");
                        break;
                    }

                    let lo = state
                        .history()
                        .iter()
                        .rev()
                        .find(|r| r.recommendation_shown == Some("kv_cache_pressure"))
                        .and_then(|r| r.result.static_ctx.config.max_num_seqs);
                    let hi = state
                        .history()
                        .iter()
                        .rev()
                        .find(|r| r.recommendation_shown == Some("concurrency_saturation"))
                        .and_then(|r| r.result.static_ctx.config.max_num_seqs);

                    match (lo, hi) {
                        (Some(lo), Some(hi)) if hi > lo + 2 => {
                            let mid = (lo + hi) / 2;
                            println!(
                                "\n--max-num-seqs={hi} filled KV cache. --max-num-seqs={lo} saturated the queue."
                            );
                            println!("Try --max-num-seqs={mid}.");
                            state.set_midpoint_suggested();
                        }
                        _ => {
                            println!(
                                "\nNo --max-num-seqs value resolves both KV pressure and queue saturation."
                            );
                            println!("Add a replica to scale out.");
                            break;
                        }
                    }
                }
                Some((a, b)) => {
                    println!(
                        "\nCycling between [{a}] and [{b}]. No further improvement found. Stopping."
                    );
                    break;
                }
                None => {
                    println!(
                        "\nCycling between recommendations. No further improvement found. Stopping."
                    );
                    break;
                }
            }
        }

        if state.iteration_count() >= super::state::MAX_LOOP_ITERATIONS {
            println!(
                "\nNo further improvement found after {} iterations. Stopping.",
                super::state::MAX_LOOP_ITERATIONS
            );
            break;
        }

        state.record_recommendation(rule_name);

        let _outcome = poll::wait_for_restart_or_skip(url, &stdin_rx);

        if needs_max_num_seqs_prompt_for_groups(&state.last().report.groups) {
            println!();
            let current = current_max_num_seqs.unwrap_or(256);
            current_max_num_seqs = Some(crate::cli::prompt_for_updated_max_num_seqs(
                current, &stdin_rx,
            )?);
        }

        println!("\nMeasuring delta...\n");
        let new_result = run_diagnose(
            url,
            current_max_num_seqs,
            cost_per_hour,
            tensor_parallel_size,
            duration,
        )?;
        let agg_win = RuntimeWindow::from_snapshot(new_result.snapshot.clone());
        let summary = AnalysisInput::new(&new_result.static_ctx, &agg_win);
        let new_report = engine::build_report_for_diagnose(&new_result.windows, summary);

        let drifted =
            drift::config_changed(&state.last().result.static_ctx, &new_result.static_ctx);
        if drifted {
            println!("Config change detected, re-baselined.");
        }

        let d = delta::compute(
            &state.last().result,
            &state.last().report,
            &new_result,
            &new_report,
            drifted,
        );
        print_delta(&d);
        println!();
        output::stdout::print_direction_line(&d);
        println!();
        output::stdout::print_diagnose_table(&new_result, false);

        let headroom = new_report.baseline.as_ref().and_then(|b| b.headroom_pct);
        if at_hardware_ceiling(headroom) {
            println!(
                "\nHardware ceiling reached. Headroom < {CEILING_HEADROOM_THRESHOLD_PCT:.0}%: further gains require scaling hardware."
            );
            break;
        }

        let applied_short_action = state
            .last()
            .report
            .groups
            .first()
            .map(|g| g.primary.short_action.as_str());
        let next_fix = new_report
            .groups
            .first()
            .map(|g| g.primary.short_action.as_str());

        if d.direction == delta::Direction::Worse {
            let choice = read_worse_regression_choice(&stdin_rx, &mut io::stdout())?;
            match choice {
                WorseRegressionChoice::Continue => {
                    for line in direction_followup_lines(delta::Direction::NoChange, next_fix) {
                        println!("{line}");
                    }
                    let _ = apply_worse_regression_choice(
                        &mut state, new_result, new_report, rule_name, choice,
                    );
                }
                WorseRegressionChoice::Revert => {
                    if let Some(action) = applied_short_action {
                        println!("Revert: undo {action}, then re-measure when ready.");
                    } else {
                        println!("Revert: undo the last change, then re-measure when ready.");
                    }
                }
            }
        } else {
            for line in direction_followup_lines(d.direction, next_fix) {
                println!("{line}");
            }
            state.push(new_result, new_report, Some(rule_name));
        }
    }

    Ok(())
}

fn at_hardware_ceiling(headroom_pct: Option<f64>) -> bool {
    headroom_pct.is_some_and(|h| h < CEILING_HEADROOM_THRESHOLD_PCT)
}

// COUPLING: matches "--max-num-seqs" in rule action strings; update if flag name changes.
fn needs_max_num_seqs_prompt(action: &str) -> bool {
    action.contains("--max-num-seqs")
}

fn needs_max_num_seqs_prompt_for_groups(groups: &[engine::IssueGroup]) -> bool {
    groups
        .iter()
        .any(|g| needs_max_num_seqs_prompt(&g.primary.short_action))
}

fn healthy_exit_message(
    efficiency: Option<f64>,
    headroom: Option<f64>,
    num_running: Option<f64>,
    max_num_seqs: Option<u32>,
) -> String {
    let load_is_low = match (num_running, max_num_seqs) {
        (Some(r), Some(m)) if m > 0 => (r / f64::from(m)) < LOW_OCCUPANCY_THRESHOLD,
        _ => false,
    };

    match (efficiency, headroom) {
        (Some(e), _) if headroom.is_some_and(|h| h < CEILING_HEADROOM_THRESHOLD_PCT) => {
            format!("No issues detected. Server is at hardware capacity: {e:.1}% of ceiling.")
        }
        (Some(e), _) if load_is_low => {
            format!("No issues detected. Efficiency: {e:.1}% (hardware is under-fed, not misconfigured).")
        }
        (Some(e), _) => {
            format!("No significant change. Efficiency: {e:.1}% of hardware ceiling.")
        }
        (None, _) => {
            "No issues detected. Efficiency unavailable; GPU or model data missing.".to_string()
        }
    }
}

fn direction_followup_lines(
    direction: delta::Direction,
    short_action: Option<&str>,
) -> Vec<String> {
    let mut lines = Vec::new();
    match direction {
        delta::Direction::Worse => {}
        delta::Direction::Mixed => {
            // Direction line carries the reason. Nothing to add.
        }
        delta::Direction::NoChange => {
            if let Some(fix) = short_action {
                lines.push("No significant change.".to_string());
                lines.push(format!("Apply fix: {fix}, then re-measure."));
            }
        }
        // Better: no followup needed. The table shows the improved state; the
        // Direction line already carries the signal. Silence is the right output.
        delta::Direction::Better => {}
    }
    lines
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorseRegressionChoice {
    Continue,
    Revert,
}

pub(crate) fn parse_worse_regression_choice(line: &str) -> Option<WorseRegressionChoice> {
    match line.trim() {
        "c" | "C" => Some(WorseRegressionChoice::Continue),
        "r" | "R" => Some(WorseRegressionChoice::Revert),
        _ => None,
    }
}

pub(crate) fn read_worse_regression_choice(
    stdin_rx: &std::sync::mpsc::Receiver<String>,
    prompt: &mut dyn Write,
) -> io::Result<WorseRegressionChoice> {
    writeln!(prompt, "  [r] revert   [c] continue")?;
    loop {
        write!(prompt, "> ")?;
        prompt.flush()?;
        let line = stdin_rx
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "stdin closed"))?;
        if let Some(choice) = parse_worse_regression_choice(&line) {
            return Ok(choice);
        }
        writeln!(prompt, " r = revert, c = continue")?;
    }
}

pub(crate) fn apply_worse_regression_choice(
    state: &mut LoopState,
    new_result: DiagnoseResult,
    new_report: engine::Report,
    rule_name: &'static str,
    choice: WorseRegressionChoice,
) -> bool {
    match choice {
        WorseRegressionChoice::Continue => {
            state.push(new_result, new_report, Some(rule_name));
            true
        }
        WorseRegressionChoice::Revert => false,
    }
}

fn format_efficiency_delta_line(delta_pp: Option<f64>) -> Option<String> {
    match delta_pp {
        Some(v) if v >= EFFICIENCY_DISPLAY_MIN_PP => Some(format!("  Efficiency  +{v:.1}pp ↑")),
        Some(v) if v <= -EFFICIENCY_DISPLAY_MIN_PP => Some(format!("  Efficiency  {v:.1}pp ↓")),
        _ => None,
    }
}

fn print_delta(d: &delta::Delta) {
    if d.config_drifted {
        println!("  Config changed, baseline reset.");
    }
    if let (Some(before), Some(after)) = (d.throughput_before, d.throughput_after) {
        if before.is_finite() && after.is_finite() {
            let arrow = throughput_arrow(before, after);
            println!("  Throughput  {before:.0} → {after:.0} tok/s {arrow}");
        }
    }
    if let (Some(before), Some(after)) = (d.ttft_before_ms, d.ttft_after_ms) {
        if before.is_finite() && after.is_finite() {
            let delta = after - before;
            if delta.abs() > 5.0 {
                let arrow = latency_arrow(delta);
                let p99_suffix = match (d.ttft_p99_before_ms, d.ttft_p99_after_ms) {
                    (Some(pb), Some(pa)) if pb.is_finite() && pa.is_finite() => {
                        format!("  (p99 {pb:.0} → {pa:.0}ms {})", latency_arrow(pa - pb))
                    }
                    _ => String::new(),
                };
                println!("  TTFT        {before:.0} → {after:.0}ms {arrow}{p99_suffix}");
            }
        }
    }
    if let (Some(before), Some(after)) = (d.tpot_before_ms, d.tpot_after_ms) {
        if before.is_finite() && after.is_finite() {
            let delta = after - before;
            if delta.abs() > 0.5 {
                let arrow = latency_arrow(delta);
                let p99_suffix = match (d.tpot_p99_before_ms, d.tpot_p99_after_ms) {
                    (Some(pb), Some(pa)) if pb.is_finite() && pa.is_finite() => {
                        format!("  (p99 {pb:.1} → {pa:.1}ms {})", latency_arrow(pa - pb))
                    }
                    _ => String::new(),
                };
                println!("  TPOT        {before:.1} → {after:.1}ms {arrow}{p99_suffix}");
            }
        }
    }
    if let Some(line) = format_efficiency_delta_line(d.efficiency_delta_pp) {
        println!("{line}");
    }
    let has_cost = economics_section_active(d);
    if has_cost {
        println!();
        println!("ECONOMICS:");
    }
    match (d.joules_per_token_before, d.joules_per_token_after) {
        (Some(before), Some(after)) if before.is_finite() && after.is_finite() => {
            let arrow = jtok_change_arrow(before, after);
            println!("  J/tok         {before:.2} → {after:.2} {arrow}");
        }
        _ => {}
    }
    match (d.cost_per_million_before, d.cost_per_million_after) {
        (Some(before), Some(after)) if before.is_finite() && after.is_finite() => {
            let arrow = cost_change_arrow(before, after);
            let est = match d.cost_source_after {
                Some(engine::CostSource::Catalog) | None => " (est)",
                _ => "",
            };
            println!("  Cost/1M tok   ${before:.2} → ${after:.2} {arrow}{est}");
        }
        _ => {}
    }
    if let (Some(cpm_b), Some(cpm_a), Some(tps_b), Some(tps_a), Some(eff_b), Some(eff_a)) = (
        d.cost_per_million_before,
        d.cost_per_million_after,
        d.throughput_before,
        d.throughput_after,
        d.efficiency_pct_before,
        d.efficiency_pct_after,
    ) {
        if cpm_b.is_finite() && cpm_a.is_finite() && tps_b > 0.0 && tps_a > 0.0 {
            let waste_b = (cpm_b * tps_b * 3600.0 / 1_000_000.0) * (1.0 - eff_b / 100.0).max(0.0);
            let waste_a = (cpm_a * tps_a * 3600.0 / 1_000_000.0) * (1.0 - eff_a / 100.0).max(0.0);
            if waste_b.is_finite() && waste_a.is_finite() {
                let arrow = recoverable_waste_arrow(waste_b, waste_a);
                println!("  Waste         ${waste_b:.2} → ${waste_a:.2}/hr {arrow}");
            }
        }
    }
}

const COST_ARROW_THRESHOLD_USD: f64 = 0.01;
const RECOVERABLE_ARROW_THRESHOLD_USD_PER_HR: f64 = 0.05;
const JTOK_ARROW_THRESHOLD: f64 = 0.02;

fn recoverable_waste_available(d: &delta::Delta) -> bool {
    matches!(
        (
            d.cost_per_million_before,
            d.cost_per_million_after,
            d.throughput_before,
            d.throughput_after,
            d.efficiency_pct_before,
            d.efficiency_pct_after,
        ),
        (
            Some(cpm_b),
            Some(cpm_a),
            Some(tps_b),
            Some(tps_a),
            Some(eff_b),
            Some(eff_a),
        ) if cpm_b.is_finite()
            && cpm_a.is_finite()
            && tps_b > 0.0
            && tps_a > 0.0
            && eff_b.is_finite()
            && eff_a.is_finite()
    )
}

fn economics_section_active(d: &delta::Delta) -> bool {
    (d.cost_per_million_before.is_some() && d.cost_per_million_after.is_some())
        || (d.joules_per_token_before.is_some() && d.joules_per_token_after.is_some())
        || recoverable_waste_available(d)
}

fn jtok_change_arrow(before: f64, after: f64) -> &'static str {
    let diff = after - before;
    if diff < -JTOK_ARROW_THRESHOLD {
        "↓"
    } else if diff > JTOK_ARROW_THRESHOLD {
        "↑"
    } else {
        ""
    }
}

fn cost_change_arrow(before: f64, after: f64) -> &'static str {
    let diff = after - before;
    if diff < -COST_ARROW_THRESHOLD_USD {
        "↓"
    } else if diff > COST_ARROW_THRESHOLD_USD {
        "↑"
    } else {
        ""
    }
}

fn recoverable_waste_arrow(waste_before: f64, waste_after: f64) -> &'static str {
    let waste_diff = waste_after - waste_before;
    if waste_diff < -RECOVERABLE_ARROW_THRESHOLD_USD_PER_HR {
        "↓"
    } else if waste_diff > RECOVERABLE_ARROW_THRESHOLD_USD_PER_HR {
        "↑"
    } else {
        ""
    }
}

fn throughput_arrow(before: f64, after: f64) -> &'static str {
    if after > before {
        "↑"
    } else if after < before {
        "↓"
    } else {
        ""
    }
}

fn latency_arrow(delta_ms: f64) -> &'static str {
    if delta_ms < 0.0 {
        "↓"
    } else if delta_ms > 0.0 {
        "↑"
    } else {
        ""
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::IssueGroup;

    #[test]
    fn at_hardware_ceiling_below_threshold() {
        assert!(at_hardware_ceiling(Some(9.0)));
    }

    #[test]
    fn at_hardware_ceiling_at_threshold_not_reached() {
        assert!(!at_hardware_ceiling(Some(10.0)));
    }

    #[test]
    fn at_hardware_ceiling_none_not_reached() {
        assert!(!at_hardware_ceiling(None));
    }

    #[test]
    fn healthy_exit_no_issues_shows_no_significant_change_with_efficiency() {
        let msg = healthy_exit_message(Some(42.5), Some(57.5), None, None);
        assert_eq!(
            msg,
            "No significant change. Efficiency: 42.5% of hardware ceiling."
        );
    }

    #[test]
    fn healthy_exit_at_capacity_when_headroom_low() {
        let msg = healthy_exit_message(Some(91.0), Some(9.0), None, None);
        assert!(msg.contains("at hardware capacity: 91.0% of ceiling."));
    }

    #[test]
    fn healthy_exit_unavailable_when_efficiency_missing() {
        let msg = healthy_exit_message(None, Some(50.0), None, None);
        assert!(msg.contains("Efficiency unavailable"));
    }

    #[test]
    fn healthy_exit_under_fed_when_occupancy_low() {
        let msg = healthy_exit_message(Some(34.0), Some(66.0), Some(2.0), Some(256));
        assert!(msg.contains("under-fed, not misconfigured"));
        assert!(msg.contains("34.0%"));
    }

    #[test]
    fn healthy_exit_no_under_fed_label_when_occupancy_high() {
        let msg = healthy_exit_message(Some(34.0), Some(66.0), Some(200.0), Some(256));
        assert!(msg.contains("of hardware ceiling"));
        assert!(!msg.contains("under-fed"));
    }

    #[test]
    fn efficiency_delta_near_zero_suppressed() {
        assert!(format_efficiency_delta_line(Some(-0.04)).is_none());
        assert!(format_efficiency_delta_line(Some(0.03)).is_none());
    }

    #[test]
    fn efficiency_delta_below_threshold_prints_down() {
        let line = format_efficiency_delta_line(Some(-0.06)).expect("line");
        assert!(line.contains('↓'));
        assert!(line.contains("-0.1pp"));
    }

    #[test]
    fn direction_followup_includes_short_action_on_no_change() {
        let lines = direction_followup_lines(
            delta::Direction::NoChange,
            Some("raise --max-num-seqs above 32"),
        );
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "No significant change.");
        assert_eq!(
            lines[1],
            "Apply fix: raise --max-num-seqs above 32, then re-measure."
        );
    }

    #[test]
    fn direction_followup_mixed_is_empty() {
        let lines = direction_followup_lines(
            delta::Direction::Mixed,
            Some("raise --max-num-seqs above 32"),
        );
        assert!(lines.is_empty());
    }

    #[test]
    fn direction_followup_worse_is_empty() {
        let lines =
            direction_followup_lines(delta::Direction::Worse, Some("increase client concurrency"));
        assert!(lines.is_empty());
    }

    #[test]
    fn parse_worse_regression_choice_accepts_r_and_c() {
        assert_eq!(
            parse_worse_regression_choice("c\n"),
            Some(WorseRegressionChoice::Continue)
        );
        assert_eq!(
            parse_worse_regression_choice("R"),
            Some(WorseRegressionChoice::Revert)
        );
        assert_eq!(parse_worse_regression_choice("x"), None);
    }

    #[test]
    fn read_worse_regression_choice_reprompts_on_invalid_input() {
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send("x\n".to_string()).unwrap();
        tx.send("r\n".to_string()).unwrap();
        let mut out = Vec::new();
        let choice = read_worse_regression_choice(&rx, &mut out).unwrap();
        assert_eq!(choice, WorseRegressionChoice::Revert);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains(" r = revert, c = continue"));
    }

    #[test]
    fn worse_pauses_loop_before_next_recommendation() {
        assert!(!apply_worse_regression_choice(
            &mut LoopState::new(minimal_diagnose(), empty_report()),
            minimal_diagnose(),
            empty_report(),
            "under_batching",
            WorseRegressionChoice::Revert,
        ));
    }

    #[test]
    fn acknowledge_sets_prev_to_curr() {
        let mut s = LoopState::new(minimal_diagnose(), empty_report());
        let mut new = minimal_diagnose();
        new.snapshot.vllm.generation_tokens_per_sec = Some(80.0);
        assert!(apply_worse_regression_choice(
            &mut s,
            new,
            empty_report(),
            "under_batching",
            WorseRegressionChoice::Continue,
        ));
        assert_eq!(
            s.last().result.snapshot.vllm.generation_tokens_per_sec,
            Some(80.0)
        );
    }

    #[test]
    fn revert_does_not_update_prev() {
        let mut prev = minimal_diagnose();
        prev.snapshot.vllm.generation_tokens_per_sec = Some(100.0);
        let mut s = LoopState::new(prev, empty_report());
        let mut degraded = minimal_diagnose();
        degraded.snapshot.vllm.generation_tokens_per_sec = Some(80.0);
        assert!(!apply_worse_regression_choice(
            &mut s,
            degraded,
            empty_report(),
            "under_batching",
            WorseRegressionChoice::Revert,
        ));
        assert_eq!(
            s.last().result.snapshot.vllm.generation_tokens_per_sec,
            Some(100.0)
        );
    }

    fn minimal_diagnose() -> DiagnoseResult {
        DiagnoseResult {
            snapshot: crate::collectors::RawSnapshot {
                gpu_observed_at: std::time::SystemTime::UNIX_EPOCH,
                vllm_observed_at: std::time::SystemTime::UNIX_EPOCH,
                timestamp: std::time::SystemTime::UNIX_EPOCH,
                vllm: crate::collectors::VllmRawMetrics::default(),
                gpu: crate::collectors::GpuRawMetrics::default(),
            },
            windows: Vec::new(),
            static_ctx: crate::context::StaticContext::default(),
            duration: Duration::from_secs(2),
            started_at: std::time::SystemTime::UNIX_EPOCH,
            any_evaluable: true,
            metrics_input: String::new(),
        }
    }

    fn empty_report() -> engine::Report {
        engine::Report {
            baseline: None,
            groups: Vec::new(),
            r2_suppressed_by_r4: false,
        }
    }

    #[test]
    fn direction_followup_omits_fix_when_short_action_missing() {
        let lines = direction_followup_lines(delta::Direction::NoChange, None);
        assert!(lines.is_empty());
    }

    #[test]
    fn cost_arrow_suppressed_when_delta_below_threshold() {
        assert_eq!(cost_change_arrow(2.00, 2.005), "");
    }

    #[test]
    fn cost_arrow_down_when_improvement_above_threshold() {
        assert_eq!(cost_change_arrow(2.00, 1.98), "↓");
    }

    #[test]
    fn recoverable_arrow_suppressed_when_delta_below_threshold() {
        assert_eq!(recoverable_waste_arrow(10.0, 9.97), "");
    }

    #[test]
    fn recoverable_arrow_down_when_waste_reduced_above_threshold() {
        assert_eq!(recoverable_waste_arrow(10.0, 9.94), "↓");
    }

    #[test]
    fn jtok_arrow_down_when_energy_improves() {
        assert_eq!(jtok_change_arrow(0.31, 0.28), "↓");
    }

    #[test]
    fn jtok_arrow_suppressed_below_threshold() {
        assert_eq!(jtok_change_arrow(0.31, 0.30), "");
    }

    #[test]
    fn economics_header_shown_when_only_jtok_available() {
        let d = delta::Delta {
            throughput_delta_pct: None,
            throughput_before: None,
            throughput_after: None,
            efficiency_delta_pp: None,
            efficiency_pct_before: None,
            efficiency_pct_after: None,
            cost_per_million_before: None,
            cost_per_million_after: None,
            joules_per_token_before: Some(0.31),
            joules_per_token_after: Some(0.28),
            cost_source_after: None,
            ttft_before_ms: None,
            ttft_after_ms: None,
            tpot_before_ms: None,
            tpot_after_ms: None,
            ttft_p99_before_ms: None,
            ttft_p99_after_ms: None,
            tpot_p99_before_ms: None,
            tpot_p99_after_ms: None,
            direction: delta::Direction::NoChange,
            direction_reason: None,
            ttft_p99_delta_pct: None,
            veto_fired: false,
            config_drifted: false,
        };
        assert!(economics_section_active(&d));
    }

    #[test]
    fn economics_header_shown_for_recoverable_waste() {
        let d = delta::Delta {
            throughput_delta_pct: Some(160.8),
            throughput_before: Some(1580.0),
            throughput_after: Some(4120.0),
            efficiency_delta_pp: Some(18.4),
            efficiency_pct_before: Some(36.0),
            efficiency_pct_after: Some(54.4),
            cost_per_million_before: Some(0.16),
            cost_per_million_after: Some(0.16),
            joules_per_token_before: None,
            joules_per_token_after: None,
            cost_source_after: Some(engine::CostSource::Catalog),
            ttft_before_ms: None,
            ttft_after_ms: None,
            tpot_before_ms: None,
            tpot_after_ms: None,
            ttft_p99_before_ms: None,
            ttft_p99_after_ms: None,
            tpot_p99_before_ms: None,
            tpot_p99_after_ms: None,
            direction: delta::Direction::Better,
            direction_reason: None,
            ttft_p99_delta_pct: None,
            veto_fired: false,
            config_drifted: false,
        };
        assert!(recoverable_waste_available(&d));
        assert!(economics_section_active(&d));
    }

    #[test]
    fn max_num_seqs_prompt_ignored_when_only_in_secondary() {
        let groups = vec![IssueGroup {
            primary: test_rec("raise --gpu-memory-utilization"),
            secondary: vec![test_rec("lower --max-num-seqs")],
        }];
        assert!(!needs_max_num_seqs_prompt_for_groups(&groups));
    }

    #[test]
    fn any_max_num_seqs_prompt_skipped_when_absent_from_all() {
        let groups = vec![IssueGroup {
            primary: test_rec("Switch to fp8 KV cache (--kv-cache-dtype fp8)"),
            secondary: vec![test_rec("raise --gpu-memory-utilization")],
        }];
        assert!(!needs_max_num_seqs_prompt_for_groups(&groups));
    }

    fn test_rec(short_action: &str) -> engine::Recommendation {
        engine::Recommendation {
            rule_name: "test",
            impact: 1,
            confidence: 0.5,
            action: String::new(),
            short_action: short_action.to_string(),
            expected_impact: String::new(),
            display_lines: Vec::new(),
        }
    }
}
