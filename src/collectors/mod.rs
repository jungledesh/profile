//! Collectors: GPU util, power draw, token stats. Stubs until vLLM/NVML (or agent).

mod gpu;
mod power;
mod tokens;

#[derive(Debug, Default, Clone)]
pub struct Snapshot {
    pub gpu_name: Option<String>, // NEW: V1 table header (friendly name only)
    pub gpu_util: Option<f32>,
    pub power_w: Option<f32>,
    pub tokens_per_sec: Option<f32>,
}

pub fn snapshot() -> Snapshot {
    Snapshot {
        gpu_name: gpu::gpu_name(),
        gpu_util: gpu::gpu_utilization(),
        power_w: power::power_draw(),
        tokens_per_sec: tokens::token_stats(),
    }
}
