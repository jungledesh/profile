use anyhow::Result;

use super::VllmRawMetrics;

pub fn collect_vllm_metrics(_base: &str) -> Result<VllmRawMetrics> {
    Ok(VllmRawMetrics::default())
}
