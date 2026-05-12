use std::time::Duration;

use crate::cli::goal;
use crate::{context, engine, output, profiler};

pub fn execute(
    vllm_metrics_input: &str,
    max_num_seqs: u32,
    verbose_rules: bool,
    duration: Duration,
) -> anyhow::Result<()> {
    let result = profiler::run_diagnose(vllm_metrics_input, max_num_seqs, duration)?;
    output::stdout::print_diagnose_table(&result, verbose_rules);

    if !result.any_evaluable {
        return Ok(());
    }

    // Build report for goal inference.
    // Note: print_diagnose_table builds this internally too — minor double-build,
    // acceptable until output API is refactored in Step 7.
    let aggregate_win = context::RuntimeWindow::from_snapshot(result.snapshot.clone());
    let summary_input = context::AnalysisInput::new(&result.static_ctx, &aggregate_win);
    let report = engine::build_report(summary_input);

    let inferred = goal::infer_objective(&report.groups, report.baseline.as_ref());
    let chosen = goal::prompt_goal(&inferred)?;

    match goal::check_feasibility(&chosen, report.baseline.as_ref()) {
        goal::FeasibilityResult::Reachable => {
            println!(
                "\nGoal: {}. Optimization loop coming in Step 6.",
                chosen.objective.label()
            );
        }
        goal::FeasibilityResult::AtCeiling { headroom_pct } => {
            println!(
                "\nAlready within {:.1}% of the hardware ceiling. No headroom to improve.",
                headroom_pct
            );
            println!("To go further: upgrade GPU, reduce model size, or use a smaller dtype.");
        }
    }

    Ok(())
}
