use std::time::Duration;

use crate::{output, profiler};

pub fn execute(
    vllm_metrics_input: &str,
    max_num_seqs: u32,
    verbose_rules: bool,
    duration: Duration,
) -> anyhow::Result<()> {
    let result = profiler::run_diagnose(vllm_metrics_input, max_num_seqs, duration)?;
    output::stdout::print_diagnose_table(&result, verbose_rules);
    Ok(())
}
