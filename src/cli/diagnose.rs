use std::time::Duration;

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

    let aggregate_win = context::RuntimeWindow::from_snapshot(result.snapshot.clone());
    let summary_input = context::AnalysisInput::new(&result.static_ctx, &aggregate_win);
    let report = engine::build_report_for_diagnose(&result.windows, summary_input);

    profiler::loop_runner::run(vllm_metrics_input, max_num_seqs, duration, result, report)?;

    Ok(())
}
