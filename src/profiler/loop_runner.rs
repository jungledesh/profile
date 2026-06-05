use std::time::Duration;

use super::{delta, drift, poll, run_diagnose, state::LoopState, DiagnoseResult};
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine;
use crate::output;

const CEILING_HEADROOM_THRESHOLD_PCT: f64 = 10.0;

pub fn run(
    url: &str,
    max_num_seqs: u32,
    cost_per_hour: Option<f64>,
    duration: Duration,
    initial_result: DiagnoseResult,
    initial_report: engine::Report,
) -> anyhow::Result<()> {
    let mut state = LoopState::new(initial_result, initial_report);
    let stdin_rx = poll::spawn_stdin_watcher();

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

            let msg = if efficiency.is_some_and(|e| e > 100.0) {
                "No issues detected. Server is healthy under current load."
            } else if headroom.is_some_and(|h| h < CEILING_HEADROOM_THRESHOLD_PCT) {
                "No issues detected. Server is at hardware capacity."
            } else {
                "No issues detected. Server is healthy under current load."
            };

            println!("\n{msg}");
            break;
        };

        if state.is_oscillating() {
            if let Some((a, b)) = state.oscillating_pair() {
                println!(
                    "\nCycling between [{a}] and [{b}]. No further improvement found. Stopping."
                );
            } else {
                println!(
                    "\nCycling between recommendations. No further improvement found. Stopping."
                );
            }
            break;
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

        println!("\nMeasuring...");
        let new_result = run_diagnose(url, max_num_seqs, cost_per_hour, duration)?;
        let agg_win = RuntimeWindow::from_snapshot(new_result.snapshot.clone());
        let summary = AnalysisInput::new(&new_result.static_ctx, &agg_win);
        let new_report = engine::build_report_for_diagnose(&new_result.windows, summary);

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

        let headroom = new_report.baseline.as_ref().and_then(|b| b.headroom_pct);
        if at_hardware_ceiling(headroom) {
            println!(
                "\nHardware ceiling reached. Headroom < {CEILING_HEADROOM_THRESHOLD_PCT:.0}% — further gains require scaling hardware."
            );
            break;
        }

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

fn at_hardware_ceiling(headroom_pct: Option<f64>) -> bool {
    headroom_pct.is_some_and(|h| h < CEILING_HEADROOM_THRESHOLD_PCT)
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
    match (d.cost_per_million_before, d.cost_per_million_after) {
        (Some(before), Some(after)) if before.is_finite() && after.is_finite() => {
            let arrow = if after < before {
                "↓"
            } else if after > before {
                "↑"
            } else {
                ""
            };
            let est = match d.cost_source_after {
                Some(engine::CostSource::Catalog) | None => " (est)",
                _ => "",
            };
            println!("  Cost/1M tok  ${before:.2} → ${after:.2} {arrow}{est}");
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
                let arrow = if waste_a < waste_b {
                    "↓"
                } else if waste_a > waste_b {
                    "↑"
                } else {
                    ""
                };
                println!("  Recoverable  ${waste_b:.2} → ${waste_a:.2}/hr {arrow}");
            }
        }
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
}
