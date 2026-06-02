use std::time::Duration;

use super::{delta, drift, poll, run_diagnose, state::LoopState, DiagnoseResult};
use crate::collectors::window_is_evaluable;
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine;
use crate::output;

const CEILING_HEADROOM_THRESHOLD_PCT: f64 = 10.0;

pub fn run(
    url: &str,
    max_num_seqs: u32,
    duration: Duration,
    initial_result: DiagnoseResult,
    initial_report: engine::Report,
) -> anyhow::Result<()> {
    let mut state = LoopState::new(initial_result, initial_report);
    let stdin_rx = poll::spawn_stdin_watcher();

    loop {
        let rule_name = match primary_window_rule(&state.last().result)
            .or_else(|| state.current_primary_recommendation().map(|r| r.rule_name))
        {
            None => {
                let baseline = state.last().report.baseline.as_ref();
                let efficiency = baseline.and_then(|b| b.efficiency_pct);
                let headroom = baseline.and_then(|b| b.headroom_pct);

                let msg = if efficiency.is_some_and(|e| e > 100.0) {
                    "No issues detected. Server is healthy under current load."
                } else if headroom.is_some_and(|h| h < CEILING_HEADROOM_THRESHOLD_PCT) {
                    "No issues detected. Server is at hardware capacity."
                } else {
                    "No issues detected. Server is healthy under current load."
                };

                println!("\n{msg}");
                break;
            }
            Some(rule_name) => rule_name,
        };

        if state.is_oscillating() {
            println!(
                "\nOscillating between recommendations — hardware likely at Pareto frontier for this workload."
            );
            println!("Stopping.");
            break;
        }

        state.record_recommendation(rule_name);

        let _outcome = poll::wait_for_restart_or_skip(url, &stdin_rx);

        println!("\nMeasuring...");
        let new_result = run_diagnose(url, max_num_seqs, duration)?;
        let agg_win = RuntimeWindow::from_snapshot(new_result.snapshot.clone());
        let summary = AnalysisInput::new(&new_result.static_ctx, &agg_win);
        let new_report = engine::build_report(summary);

        let drifted =
            drift::config_changed(&state.last().result.static_ctx, &new_result.static_ctx);
        if drifted {
            println!("Config change detected — re-baselined.");
        }

        let d = delta::compute(
            &state.last().result,
            &state.last().report,
            &new_result,
            &new_report,
            drifted,
        );
        print_delta(&d);

        output::stdout::print_diagnose_table(&new_result, false);

        match d.direction {
            delta::Direction::Worse => {
                println!("Throughput dropped. Check if workload changed before acting on this.");
                println!("Trying next recommendation.");
            }
            delta::Direction::Plateau => {
                println!("No significant change. Trying next recommendation.");
            }
            delta::Direction::Better => {}
        }

        state.push(new_result, new_report, Some(rule_name));
    }

    Ok(())
}

/// Returns the rule name that met window-significance thresholds, if any.
/// Aligned with `engine::rule_is_significant` used by the diagnose UI formatter.
fn primary_window_rule(result: &DiagnoseResult) -> Option<&'static str> {
    let evaluable: Vec<_> = result
        .windows
        .iter()
        .filter(|w| window_is_evaluable(&w.snapshot))
        .collect();
    let n = evaluable.len();
    if n == 0 {
        return None;
    }
    let r1_count = evaluable
        .iter()
        .filter(|w| {
            matches!(
                engine::rule1_under_batching(&w.snapshot),
                engine::Rule1Outcome::Fired(_)
            )
        })
        .count();
    if engine::rule_is_significant(r1_count, n) {
        return Some("under_batching");
    }
    None
}

fn print_delta(d: &delta::Delta) {
    if d.config_drifted {
        println!("  Config changed — baseline reset.");
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
                println!("  TTFT        {before:.0} → {after:.0}ms {arrow}");
            }
        }
    }
    if let (Some(before), Some(after)) = (d.tpot_before_ms, d.tpot_after_ms) {
        if before.is_finite() && after.is_finite() {
            let delta = after - before;
            if delta.abs() > 0.5 {
                let arrow = latency_arrow(delta);
                println!("  TPOT        {before:.1} → {after:.1}ms {arrow}");
            }
        }
    }
    match d.efficiency_delta_pp {
        Some(v) if v > 0.0 => println!("  Efficiency  +{v:.1}pp ↑"),
        Some(v) if v < 0.0 => println!("  Efficiency  {v:.1}pp ↓"),
        _ => {}
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
