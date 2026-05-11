use crate::context::AnalysisInput;

use super::math;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CeilingEstimate {
    pub lower: f64,
    pub expected: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightDtypeSource {
    EnvVar,
    KvCacheDtype,
    Catalog,
    Fallback,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsBaseline {
    pub decode: CeilingEstimate,
    pub prefill: Option<CeilingEstimate>,
    pub efficiency_pct: Option<f64>,
    pub headroom_pct: Option<f64>,
    pub weight_dtype_source: WeightDtypeSource,
}

pub fn compute(input: &AnalysisInput<'_>) -> Option<PhysicsBaseline> {
    let ctx = input.ctx;
    let peak_flops = ctx.gpu.peak_flops_f32_tflops?;
    let peak_bw = ctx.gpu.peak_bw_gbps?;

    let model_params = ctx.model.active_param_count.or(ctx.model.param_count)?;
    let catalog_default_dtype = ctx.model.default_weight_dtype.as_deref();

    let (bytes_per_param, weight_dtype_source) = resolve_bytes_per_param(
        ctx.config.dtype.as_deref(),
        ctx.config.kv_cache_dtype.as_deref(),
        catalog_default_dtype,
    );

    let decode_expected = math::decode_ceiling_tps(peak_bw, model_params, bytes_per_param);
    let decode = make_estimate(decode_expected)?;

    let seq_len = resolve_seq_len(
        ctx.config.max_model_len,
        input.window.snapshot.vllm.prompt_tokens_mean,
    );
    let prefill = seq_len.and_then(|len| {
        let expected = math::prefill_ceiling_tps(peak_flops, model_params, len);
        make_estimate(expected)
    });

    let efficiency_pct = input
        .window
        .snapshot
        .vllm
        .generation_tokens_per_sec
        .filter(|v| v.is_finite())
        .map(|actual| math::efficiency_pct(actual, decode.expected))
        .filter(|v| v.is_finite());

    let headroom_pct = efficiency_pct.map(|raw| 100.0 - raw.min(100.0));

    Some(PhysicsBaseline {
        decode,
        prefill,
        efficiency_pct,
        headroom_pct,
        weight_dtype_source,
    })
}

fn make_estimate(expected: f64) -> Option<CeilingEstimate> {
    if !expected.is_finite() {
        return None;
    }
    let lower = expected * 0.85;
    let upper = expected * 1.05;
    if !(lower.is_finite() && upper.is_finite()) {
        return None;
    }
    Some(CeilingEstimate {
        lower,
        expected,
        upper,
    })
}

fn resolve_seq_len(max_model_len: Option<u32>, prompt_tokens_mean: Option<f64>) -> Option<u32> {
    if let Some(v) = max_model_len.filter(|v| *v > 0) {
        return Some(v);
    }
    prompt_tokens_mean
        .filter(|v| v.is_finite())
        .map(|v| v.round())
        .filter(|v| *v > 0.0 && *v <= u32::MAX as f64)
        .map(|v| v as u32)
}

fn resolve_bytes_per_param(
    dtype_env: Option<&str>,
    kv_cache_dtype: Option<&str>,
    catalog_default_dtype: Option<&str>,
) -> (u8, WeightDtypeSource) {
    if let Some(v) = dtype_env.and_then(dtype_to_bytes) {
        return (v, WeightDtypeSource::EnvVar);
    }
    if let Some(v) = kv_cache_dtype.and_then(dtype_to_bytes) {
        return (v, WeightDtypeSource::KvCacheDtype);
    }
    if let Some(v) = catalog_default_dtype.and_then(dtype_to_bytes) {
        return (v, WeightDtypeSource::Catalog);
    }
    (2, WeightDtypeSource::Fallback)
}

fn dtype_to_bytes(dtype: &str) -> Option<u8> {
    let d = dtype.trim().to_ascii_lowercase();
    if d.is_empty() {
        return None;
    }

    if d.contains("fp8") || d.contains("e4m3") || d.contains("e5m2") {
        return Some(1);
    }
    if d.contains("bf16") || d.contains("fp16") || d.contains("float16") || d == "half" {
        return Some(2);
    }
    if d.contains("fp32") || d.contains("float32") || d == "f32" {
        return Some(4);
    }
    None
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use crate::{
        collectors::{GpuRawMetrics, RawSnapshot, VllmConfig, VllmRawMetrics},
        context::{AnalysisInput, RuntimeWindow, StaticContext},
    };

    use super::*;

    fn baseline_input(
        model_params: Option<u64>,
        active_params: Option<u64>,
        default_dtype: Option<&str>,
        peak_flops: Option<f64>,
        peak_bw: Option<f64>,
        cfg: VllmConfig,
        snapshot: VllmRawMetrics,
    ) -> (StaticContext, RuntimeWindow) {
        let ctx = StaticContext {
            model: crate::context::ModelArch {
                name: Some("model".to_string()),
                family: Some("family".to_string()),
                param_count: model_params,
                active_param_count: active_params,
                num_layers: Some(1),
                hidden_dim: Some(1),
                is_moe: active_params.is_some(),
                default_weight_dtype: default_dtype.map(str::to_string),
            },
            gpu: crate::context::GPUModel {
                name: Some("gpu".to_string()),
                arch: Some("arch".to_string()),
                vram_gb: Some(80.0),
                peak_flops_f32_tflops: peak_flops,
                peak_bw_gbps: peak_bw,
            },
            config: cfg,
        };
        let win = RuntimeWindow::from_snapshot(RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: snapshot,
            gpu: GpuRawMetrics::default(),
        });
        (ctx, win)
    }

    #[test]
    fn compute_none_when_catalog_fields_missing() {
        let (ctx, win) = baseline_input(
            None,
            None,
            None,
            Some(67.0),
            Some(3350.0),
            VllmConfig::default(),
            VllmRawMetrics::default(),
        );
        let input = AnalysisInput::new(&ctx, &win);
        assert!(compute(&input).is_none());
    }

    #[test]
    fn compute_prefers_active_params_and_dtype_priority() {
        let cfg = VllmConfig {
            dtype: Some("fp32".to_string()),
            kv_cache_dtype: Some("fp8".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            prompt_tokens_mean: Some(100.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(100_000_000_000),
            Some(10_000_000_000),
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let out = compute(&input);
        assert!(out.is_some());
        let b = match out {
            Some(v) => v,
            None => panic!("expected baseline"),
        };
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::EnvVar);
        let expected_decode = math::decode_ceiling_tps(3350.0, 10_000_000_000, 4);
        assert!((b.decode.expected - expected_decode).abs() < 1e-9);
    }

    #[test]
    fn compute_prefill_none_when_no_seq_len() {
        let cfg = VllmConfig::default();
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            prompt_tokens_mean: None,
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let out = compute(&input);
        assert!(out.is_some());
        let b = match out {
            Some(v) => v,
            None => panic!("expected baseline"),
        };
        assert!(b.prefill.is_none());
    }

    #[test]
    fn compute_efficiency_and_headroom_rules() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(300.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let out = compute(&input);
        assert!(out.is_some());
        let b = match out {
            Some(v) => v,
            None => panic!("expected baseline"),
        };
        assert!(b.efficiency_pct.unwrap_or(0.0) > 100.0);
        assert_eq!(b.headroom_pct, Some(0.0));
    }

    #[test]
    fn compute_filters_non_finite_estimates() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(0),
            ..Default::default()
        };
        let snap = VllmRawMetrics::default();
        let (ctx, win) = baseline_input(
            Some(0),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        assert!(compute(&input).is_none());
    }

    #[test]
    fn dtype_source_fallback_when_unrecognized_everywhere() {
        let cfg = VllmConfig {
            dtype: Some("mystery".to_string()),
            kv_cache_dtype: Some("unknown".to_string()),
            max_model_len: Some(1024),
            ..Default::default()
        };
        let snap = VllmRawMetrics::default();
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("??"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let out = compute(&input);
        assert!(out.is_some());
        let b = match out {
            Some(v) => v,
            None => panic!("expected baseline"),
        };
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::Fallback);
    }

    #[test]
    fn fallback_when_catalog_dtype_none_and_no_env_vars() {
        // StaticContext built with param_count but no default_weight_dtype (e.g. model not in
        // catalog but GPU is). Previously this caused compute() to return None; now hits Fallback.
        let cfg = VllmConfig {
            max_model_len: Some(1024),
            ..Default::default()
        };
        let snap = VllmRawMetrics::default();
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            None, // no catalog dtype
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let out = compute(&input);
        assert!(out.is_some());
        let b = out.unwrap();
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::Fallback);
    }
}
