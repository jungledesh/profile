use std::time::SystemTime;

use crate::collectors::{traffic_from_snapshot, RawSnapshot, TrafficState, VllmConfig};

use super::{gpu_catalog, model_catalog};

#[derive(Debug, Clone, Default)]
pub struct ModelArch {
    pub name: Option<String>,
    pub family: Option<String>,
    /// Total parameter count (billions × 1e9 as u64). Dense and MoE total.
    pub param_count: Option<u64>,
    /// Active parameter count for MoE models. None for dense.
    pub active_param_count: Option<u64>,
    pub num_layers: Option<u32>,
    pub hidden_dim: Option<u32>,
    pub is_moe: bool,
    pub default_weight_dtype: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct GPUModel {
    pub name: Option<String>,
    pub arch: Option<String>,
    pub vram_gb: Option<f64>,
    /// Non-tensor-core FP32 TFLOPS. Conservative roofline input.
    pub peak_flops_f32_tflops: Option<f64>,
    /// Peak memory bandwidth GB/s.
    pub peak_bw_gbps: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct StaticContext {
    pub model: ModelArch,
    pub gpu: GPUModel,
    pub config: VllmConfig,
}

/// Last non-empty `/`-delimited segment; `None` if `s` is empty or whitespace only.
fn catalog_path_basename(s: &str) -> Option<&str> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    t.rsplit('/').find(|seg| !seg.is_empty())
}

fn catalog_model_lookup(s: &str) -> Option<&'static model_catalog::CatalogEntry> {
    model_catalog::lookup_model(s.trim())
}

fn lookup_model_catalog(
    config: &VllmConfig,
    snapshot_model_name: Option<&str>,
) -> Option<&'static model_catalog::CatalogEntry> {
    let root = config.model_root.as_deref();
    let scrape = snapshot_model_name;
    root.and_then(catalog_model_lookup)
        .or_else(|| scrape.and_then(catalog_model_lookup))
        .or_else(|| {
            root.and_then(catalog_path_basename)
                .and_then(catalog_model_lookup)
        })
        .or_else(|| {
            scrape
                .and_then(catalog_path_basename)
                .and_then(catalog_model_lookup)
        })
}

impl StaticContext {
    pub fn from_snapshot(snapshot: &RawSnapshot, config: VllmConfig) -> Self {
        let model_name = snapshot.vllm.model_name.clone();
        let catalog_entry = lookup_model_catalog(&config, model_name.as_deref());
        let model = match catalog_entry {
            Some(e) => ModelArch {
                name: model_name,
                family: Some(e.family.to_string()),
                param_count: Some(e.param_count),
                active_param_count: e.active_param_count,
                num_layers: Some(e.num_layers),
                hidden_dim: Some(e.hidden_dim),
                is_moe: e.is_moe,
                default_weight_dtype: Some(e.default_weight_dtype.to_string()),
            },
            None => ModelArch {
                name: model_name,
                ..Default::default()
            },
        };
        let gpu_name = snapshot.gpu.gpu_name.clone();
        let vram_gb = snapshot.gpu.vram_total_mb.map(|m| m as f64 / 1024.0);
        let gpu_entry = gpu_name.as_deref().and_then(gpu_catalog::lookup_gpu);
        let gpu = match gpu_entry {
            Some(e) => GPUModel {
                name: gpu_name,
                arch: Some(e.arch.to_string()),
                vram_gb,
                peak_flops_f32_tflops: Some(e.peak_flops_f32_tflops),
                peak_bw_gbps: Some(e.peak_bw_gbps),
            },
            None => GPUModel {
                name: gpu_name,
                arch: None,
                vram_gb,
                peak_flops_f32_tflops: None,
                peak_bw_gbps: None,
            },
        };
        StaticContext { model, gpu, config }
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

#[cfg(test)]
mod tests {
    use super::catalog_path_basename;

    #[test]
    fn catalog_path_basename_hf_repo() {
        assert_eq!(
            catalog_path_basename("meta-llama/Meta-Llama-3-8B-Instruct"),
            Some("Meta-Llama-3-8B-Instruct")
        );
    }

    #[test]
    fn catalog_path_basename_local_path() {
        assert_eq!(
            catalog_path_basename("/workspace/models/Meta-Llama-3-8B-Instruct"),
            Some("Meta-Llama-3-8B-Instruct")
        );
    }

    #[test]
    fn catalog_path_basename_plain_name() {
        assert_eq!(catalog_path_basename("llama3"), Some("llama3"));
    }

    #[test]
    fn catalog_path_basename_empty() {
        assert_eq!(catalog_path_basename(""), None);
        assert_eq!(catalog_path_basename("   "), None);
    }
}
