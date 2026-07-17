//! Shared RawSnapshot builders for unit tests.

use std::time::SystemTime;

use super::{GpuRawMetrics, RawSnapshot, VllmRawMetrics};

/// Epoch-timestamped snapshot with the given vLLM metrics and no GPUs.
pub(crate) fn snap_vllm(vllm: VllmRawMetrics) -> RawSnapshot {
    RawSnapshotFixture::default().vllm(vllm).build()
}

/// Builder for tests that need custom timestamps or GPU rows.
#[derive(Clone)]
pub(crate) struct RawSnapshotFixture {
    pub gpu_observed_at: SystemTime,
    pub vllm_observed_at: SystemTime,
    pub timestamp: SystemTime,
    pub vllm: VllmRawMetrics,
    pub gpus: Vec<GpuRawMetrics>,
}

impl Default for RawSnapshotFixture {
    fn default() -> Self {
        let t = SystemTime::UNIX_EPOCH;
        Self {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: VllmRawMetrics::default(),
            gpus: vec![],
        }
    }
}

impl RawSnapshotFixture {
    pub fn vllm(mut self, vllm: VllmRawMetrics) -> Self {
        self.vllm = vllm;
        self
    }

    pub fn gpus(mut self, gpus: Vec<GpuRawMetrics>) -> Self {
        self.gpus = gpus;
        self
    }

    pub fn observed_at(mut self, gpu_at: SystemTime, vllm_at: SystemTime) -> Self {
        self.gpu_observed_at = gpu_at;
        self.vllm_observed_at = vllm_at;
        self.timestamp = gpu_at;
        self
    }

    pub fn build(self) -> RawSnapshot {
        RawSnapshot {
            gpu_observed_at: self.gpu_observed_at,
            vllm_observed_at: self.vllm_observed_at,
            timestamp: self.timestamp,
            vllm: self.vllm,
            gpus: self.gpus,
        }
    }
}
