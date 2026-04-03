//! Profiler: orchestrate collectors for `diagnose`.

use crate::collectors;

#[derive(Debug, Clone)]
pub struct DiagnoseResult {
    pub snapshot: collectors::RawSnapshot,
}

pub fn run_diagnose(vllm_base: &str, max_num_seqs: u32) -> anyhow::Result<DiagnoseResult> {
    let snapshot = collectors::collect_snapshot(vllm_base, max_num_seqs)?;

    Ok(DiagnoseResult { snapshot })
}
