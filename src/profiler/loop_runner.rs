use std::time::Duration;

use super::{delta, drift, poll, run_diagnose, state::LoopState, DiagnoseResult};
use crate::cli::goal::{check_feasibility, FeasibilityResult, Goal};
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine;
use crate::output;

pub fn run(
    url: &str,
    max_num_seqs: u32,
    duration: Duration,
    initial_result: DiagnoseResult,
    initial_report: engine::Report,
    goal: Goal,
) -> anyhow::Result<()> {
    let mut state = LoopState::new(initial_result, initial_report);
    let mut iteration: u32 = 1;
    let stdin_rx = poll::spawn_stdin_watcher();

    loop {
        let (rule_name, display_lines) = match state.current_primary_recommendation() {
            None => {
                println!("\nNo issues detected. Server looks healthy.");
                break;
            }
            Some(rec) => (rec.rule_name, rec.display_lines.clone()),
        };

        if state.is_oscillating() {
            println!(
                "\nOscillating between recommendations — hardware likely at Pareto frontier for this workload."
            );
            println!("Stopping.");
            break;
        }

        println!("\n━━━ Iteration {iteration} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        for line in &display_lines {
            println!("{line}");
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

        match check_feasibility(&goal, new_report.baseline.as_ref()) {
            FeasibilityResult::AtCeiling { headroom_pct } => {
                println!(
                    "\nAt hardware ceiling ({:.1}% headroom). No further improvement possible.",
                    headroom_pct
                );
                break;
            }
            FeasibilityResult::Reachable => {}
        }

        match d.direction {
            delta::Direction::Worse => {
                println!("Performance regressed. Trying next recommendation.");
            }
            delta::Direction::Plateau => {
                println!("No significant change. Trying next recommendation.");
            }
            delta::Direction::Better => {}
        }

        state.push(new_result, new_report, Some(rule_name));
        iteration += 1;
    }

    Ok(())
}

fn print_delta(d: &delta::Delta) {
    if d.config_drifted {
        println!("  Config changed — baseline reset.");
    }
    match d.throughput_delta_pct {
        Some(v) if v > 0.0 => println!("  Throughput  +{v:.1}% ↑"),
        Some(v) if v < 0.0 => println!("  Throughput  {v:.1}% ↓"),
        Some(_) => println!("  Throughput  no change"),
        None => {}
    }
    match d.efficiency_delta_pp {
        Some(v) if v > 0.0 => println!("  Efficiency  +{v:.1}pp ↑"),
        Some(v) if v < 0.0 => println!("  Efficiency  {v:.1}pp ↓"),
        _ => {}
    }
}
