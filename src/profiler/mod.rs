//! Profiler: orchestrate collectors for `diagnose`.

use crate::collectors;

#[derive(Debug, Clone)]
pub struct DiagnoseResult {
    pub snapshot: collectors::RawSnapshot,
}

pub fn run_diagnose(vllm_metrics_input: &str, max_num_seqs: u32) -> anyhow::Result<DiagnoseResult> {
    let snapshot = collectors::collect_snapshot(vllm_metrics_input, max_num_seqs)?;

    Ok(DiagnoseResult { snapshot })
}
