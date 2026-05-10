use std::time::SystemTime;

use crate::collectors::{traffic_from_snapshot, RawSnapshot, TrafficState, VllmConfig};

#[derive(Debug, Clone, Default)]
pub struct ModelArch {
    pub name: Option<String>,
    /// Total parameter count. None until Step 3 fills it from a model catalog.
    pub param_count: Option<u64>,
}

#[derive(Debug, Clone, Default)]
pub struct GPUModel {
    pub name: Option<String>,
    pub vram_gb: Option<f64>,
    /// Peak FP32 TFLOPS. None until Step 3 fills it from a GPU catalog.
    pub peak_flops_f32_tflops: Option<f64>,
    /// Peak memory bandwidth GB/s. None until Step 3.
    pub peak_bw_gbps: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct StaticContext {
    pub model: ModelArch,
    pub gpu: GPUModel,
    pub config: VllmConfig,
}

impl StaticContext {
    pub fn from_snapshot(snapshot: &RawSnapshot, config: VllmConfig) -> Self {
        StaticContext {
            model: ModelArch {
                name: snapshot.vllm.model_name.clone(),
                param_count: None,
            },
            gpu: GPUModel {
                name: snapshot.gpu.gpu_name.clone(),
                vram_gb: snapshot.gpu.vram_total_mb.map(|m| m as f64 / 1024.0),
                peak_flops_f32_tflops: None,
                peak_bw_gbps: None,
            },
            config,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeWindow {
    pub snapshot: RawSnapshot,
    pub traffic: TrafficState,
    pub captured_at: SystemTime,
}

impl RuntimeWindow {
    pub fn from_snapshot(snapshot: RawSnapshot) -> Self {
        let traffic = traffic_from_snapshot(&snapshot);
        let captured_at = snapshot.timestamp;
        RuntimeWindow {
            snapshot,
            traffic,
            captured_at,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AnalysisInput<'a> {
    pub ctx: &'a StaticContext,
    pub window: &'a RuntimeWindow,
}

impl<'a> AnalysisInput<'a> {
    pub fn new(ctx: &'a StaticContext, window: &'a RuntimeWindow) -> Self {
        AnalysisInput { ctx, window }
    }
}
