use crate::collectors::config::DEFAULT_GPU_MEMORY_UTILIZATION;
use crate::collectors::effective_tensor_parallel;
use crate::context::{AnalysisInput, gpu_prices};

use super::math::{self, KvCacheDtypeSource};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostEstimate {
    /// Tokens generated per watt of power draw. None if power_watts unavailable.
    pub tok_per_watt: Option<f64>,
    /// Energy per generated token (J/tok). None if power or throughput unavailable.
    pub joules_per_token: Option<f64>,
    /// Estimated cost per 1M tokens (USD). None if no price source available.
    pub cost_per_million_tokens: Option<f64>,
    /// Source of the cost estimate.
    pub cost_source: CostSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostSource {
    /// Operator-supplied via --cost-per-hour flag.
    UserProvided,
    /// From gpu_prices.json catalog. Always labeled (est).
    Catalog,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CeilingEstimate {
    pub lower: f64,
    pub expected: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightDtypeSource {
    VllmInfoQuantization,
    EnvVarQuantization,
    VllmConfig,
    VllmInfoEndpoint,
    EnvVar,
    Catalog,
    Fallback,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsBaseline {
    pub decode: CeilingEstimate,
    /// Prefill ceiling in **prompts/s** (full forward passes at `seq_len`), not tok/s.
    /// See [`math::prefill_ops_per_sec`].
    pub prefill: Option<CeilingEstimate>,
    pub efficiency_pct: Option<f64>,
    pub headroom_pct: Option<f64>,
    pub weight_dtype_source: WeightDtypeSource,
    /// Model weight memory footprint in GB. Derived from bits_per_param (quantization chain or dtype chain).
    pub weight_gb: f64,
    /// Bytes per weight parameter (`bits_per_param / 8`). Used for weight footprint only.
    pub weight_bytes_per_param: u8,
    /// KV element width from kv_cache_dtype resolution (never weight width).
    pub kv_bytes_per_element: u8,
    pub kv_cache_dtype_source: KvCacheDtypeSource,
    /// VRAM remaining after weights, if total VRAM is known. Negative means weights alone exceed VRAM.
    pub kv_headroom_gb: Option<f64>,
    /// Theoretical minimum time-per-output-token at decode ceiling (ms).
    pub tpot_floor_ms: f64,
    /// Theoretical minimum prefill latency (ms per full prompt) at prefill ceiling.
    /// None when seq_len unknown.
    pub prefill_latency_floor_ms: Option<f64>,
    /// Concurrent batch size at which decode crosses from BW-bound to compute-bound (roofline ridge).
    pub ridge_batch_size: f64,
    /// Efficiency relative to max_num_seqs config ceiling (not ridge). None when GPU unknown or inputs missing.
    pub config_relative_efficiency_pct: Option<f64>,
    pub cost: Option<CostEstimate>,
}

/// Lower/upper band around roofline ceiling `expected` (conservative / optimistic).
const CEILING_LOWER_BAND: f64 = 0.85;
const CEILING_UPPER_BAND: f64 = 1.05;

pub fn compute(input: &AnalysisInput<'_>) -> Option<PhysicsBaseline> {
    let ctx = input.ctx;
    let peak_flops = ctx.gpu.peak_flops_tc_tflops?;
    let peak_bw = ctx.gpu.peak_bw_gbps?;
    let collected = input.window.snapshot.collected_gpu_count();
    let tp = effective_tensor_parallel(ctx.config.tensor_parallel_size, collected)? as f64;

    let roofline_params = ctx.model.active_param_count.or(ctx.model.param_count)?;
    let weight_params = ctx.model.param_count.or(ctx.model.active_param_count)?;
    let catalog_default_dtype = ctx.model.default_weight_dtype.as_deref();

    let (bits_per_param, weight_dtype_source) = resolve_bits_per_param(
        ctx.config.vllm_reported_dtype.as_deref(),
        ctx.config.vllm_reported_dtype_resolved,
        ctx.config.vllm_reported_quantization.as_deref(),
        ctx.config.dtype.as_deref(),
        ctx.config.quantization.as_deref(),
        catalog_default_dtype,
    );

    let ridge_batch_size = math::ridge_batch_size(peak_flops, peak_bw, bits_per_param);

    let decode_expected = math::decode_ceiling_tps(peak_bw * tp, roofline_params, bits_per_param);
    let decode = make_estimate(decode_expected)?;

    let seq_len = resolve_seq_len(
        ctx.config.max_model_len,
        input.window.snapshot.vllm.prompt_tokens_mean,
    );
    // Note: Prefill ceiling assumes standard BF16 GEMM throughput. Fused dequantization
    // kernels (e.g., Marlin/AWQ) introduce instruction overhead that typically lowers
    // achievable compute utilization.
    let attn_coeff = ctx
        .model
        .attn_flops_coeff
        .unwrap_or_else(|| ctx.model.hidden_dim.map(|h| 2 * h as u64).unwrap_or(0));
    let num_layers = ctx.model.num_layers.unwrap_or(0);

    let prefill = seq_len.and_then(|len| {
        let expected = math::prefill_ops_per_sec(
            peak_flops * tp,
            roofline_params,
            len,
            num_layers,
            attn_coeff,
        );
        make_estimate(expected)
    });

    let ceiling = decode.expected;

    // Fraction of the absolute hardware ceiling in use. Denominator is ceiling × ridge_batch_size
    // (the compute ceiling), independent of current traffic. An idle server reads low, correctly.
    let efficiency_pct = input
        .window
        .snapshot
        .vllm
        .generation_tokens_per_sec
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|actual| {
            let absolute_ceiling = ceiling * ridge_batch_size;
            let pct = math::efficiency_pct(actual, absolute_ceiling);
            pct.min(100.0) // clamp: actual cannot exceed compute ceiling in practice
        })
        .filter(|pct| pct.is_finite());

    let config_relative_efficiency_pct = input
        .window
        .snapshot
        .vllm
        .generation_tokens_per_sec
        .filter(|v| v.is_finite() && *v > 0.0)
        .and_then(|actual| {
            let max_seqs = ctx.config.max_num_seqs?;
            Some(
                math::config_relative_efficiency_pct(actual, ceiling, max_seqs, ridge_batch_size)
                    .min(100.0),
            )
        })
        .filter(|pct| pct.is_finite());

    let headroom_pct = efficiency_pct.map(|raw| 100.0 - raw.min(100.0));

    let weight_gb = math::weight_gb(weight_params, bits_per_param);
    let weight_bytes_per_param = (bits_per_param / 8).max(1);
    let (kv_bytes_per_element, kv_cache_dtype_source) =
        math::resolve_kv_cache_element(ctx.config.kv_cache_dtype.as_deref());
    let kv_headroom_gb = ctx.gpu.vram_gb.map(|vram| {
        let gpu_util = ctx
            .config
            .gpu_memory_utilization
            .unwrap_or(DEFAULT_GPU_MEMORY_UTILIZATION);
        (vram * gpu_util) - math::ACTIVATION_KV_BUFFER_GB - (weight_gb / tp)
    });
    let tpot_floor_ms = math::latency_floor_ms(decode.expected);
    let prefill_latency_floor_ms = prefill.map(|p| math::latency_floor_ms(p.expected));

    let snap = &input.window.snapshot;
    let tps = snap
        .vllm
        .generation_tokens_per_sec
        .filter(|v| v.is_finite() && *v > 0.0);
    // Energy: aligned_power only. Never divide misaligned NVML/vLLM clocks.
    // $/1M tok joins cost/hr (config or catalog) with vLLM tok/s only; no GPU clock.
    let total_power: f64 = snap.gpus.iter().filter_map(|g| g.aligned_power_watts).sum();
    let power_watts = (total_power > 0.0).then_some(total_power);

    let tok_per_watt = match (tps, power_watts) {
        (Some(t), Some(p)) => Some(t / p),
        _ => None,
    };

    let joules_per_token = match (power_watts, tps) {
        (Some(p), Some(t)) if p > 0.0 && t > 0.0 => Some(p / t),
        _ => None,
    };

    let cost = if let Some(hr) = ctx
        .config
        .cost_per_hour
        .filter(|v| v.is_finite() && *v > 0.0)
    {
        build_cost_estimate(
            tok_per_watt,
            joules_per_token,
            hr,
            tps,
            CostSource::UserProvided,
        )
    } else if let Some(gpu_name) = ctx.gpu.name.as_deref() {
        gpu_prices::lookup_gpu_price(gpu_name).and_then(|p| {
            build_cost_estimate(
                tok_per_watt,
                joules_per_token,
                p.on_demand_per_hr * tp,
                tps,
                CostSource::Catalog,
            )
        })
    } else {
        None
    };

    Some(PhysicsBaseline {
        decode,
        prefill,
        efficiency_pct,
        headroom_pct,
        weight_dtype_source,
        weight_gb,
        weight_bytes_per_param,
        kv_bytes_per_element,
        kv_cache_dtype_source,
        kv_headroom_gb,
        tpot_floor_ms,
        prefill_latency_floor_ms,
        ridge_batch_size,
        config_relative_efficiency_pct,
        cost,
    })
}

fn build_cost_estimate(
    tok_per_watt: Option<f64>,
    joules_per_token: Option<f64>,
    cost_per_hr: f64,
    tps: Option<f64>,
    cost_source: CostSource,
) -> Option<CostEstimate> {
    let cost_per_million_tokens = tps.filter(|t| *t > 0.0).and_then(|t| {
        let cpm = cost_per_hr * 1_000_000.0 / (t * 3600.0);
        cpm.is_finite().then_some(cpm)
    });
    if tok_per_watt.is_none() && joules_per_token.is_none() && cost_per_million_tokens.is_none() {
        return None;
    }
    Some(CostEstimate {
        tok_per_watt,
        joules_per_token,
        cost_per_million_tokens,
        cost_source,
    })
}

fn make_estimate(expected: f64) -> Option<CeilingEstimate> {
    if !expected.is_finite() {
        return None;
    }
    let lower = expected * CEILING_LOWER_BAND;
    let upper = expected * CEILING_UPPER_BAND;
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
    if let Some(v) = prompt_tokens_mean
        .filter(|v| v.is_finite())
        .map(|v| v.round())
        .filter(|v| *v > 0.0 && *v <= u32::MAX as f64)
        .map(|v| v as u32)
    {
        return Some(v);
    }
    max_model_len.filter(|v| *v > 0)
}

fn resolve_bits_per_param(
    vllm_reported_dtype: Option<&str>,
    vllm_reported_dtype_resolved: bool,
    vllm_reported_quantization: Option<&str>,
    dtype_env: Option<&str>,
    quant_env: Option<&str>,
    catalog_default_dtype: Option<&str>,
) -> (u8, WeightDtypeSource) {
    if let Some(bits) = vllm_reported_quantization.and_then(quantization_to_bits) {
        return (bits, WeightDtypeSource::VllmInfoQuantization);
    }
    if let Some(bits) = quant_env.and_then(quantization_to_bits) {
        return (bits, WeightDtypeSource::EnvVarQuantization);
    }
    if let Some(bits) = vllm_reported_dtype.and_then(dtype_to_bits) {
        let source = if vllm_reported_dtype_resolved {
            WeightDtypeSource::VllmInfoEndpoint
        } else {
            WeightDtypeSource::VllmConfig
        };
        return (bits, source);
    }
    if let Some(bits) = dtype_env.and_then(dtype_to_bits) {
        return (bits, WeightDtypeSource::EnvVar);
    }
    if let Some(bits) = catalog_default_dtype.and_then(dtype_to_bits) {
        return (bits, WeightDtypeSource::Catalog);
    }
    (16, WeightDtypeSource::Fallback)
}

fn dtype_to_bits(dtype: &str) -> Option<u8> {
    let d = dtype.trim().to_ascii_lowercase();
    if d.is_empty() {
        return None;
    }

    if d.contains("fp8") || d.contains("e4m3") || d.contains("e5m2") {
        return Some(8);
    }
    if d.contains("bf16") || d.contains("fp16") || d.contains("float16") || d == "half" {
        return Some(16);
    }
    if d.contains("fp32") || d.contains("float32") || d == "f32" {
        return Some(32);
    }
    None
}

fn quantization_to_bits(scheme: &str) -> Option<u8> {
    match scheme.trim().to_ascii_lowercase().as_str() {
        "awq" | "awq_marlin" | "gptq" | "gptq_marlin" | "marlin" => Some(4),
        "int8" | "w8a8" | "fp8" => Some(8),
        _ => None,
    }
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
        let n_gpus = cfg.tensor_parallel_size.unwrap_or(1).max(1) as usize;
        let ctx = StaticContext {
            model: crate::context::ModelArch {
                param_count: model_params,
                active_param_count: active_params,
                num_layers: Some(1),
                hidden_dim: Some(1),
                default_weight_dtype: default_dtype.map(str::to_string),
                num_kv_heads: None,
                head_dim: None,
                num_kv_layers: None,
                attn_flops_coeff: None,
                linear_num_layers: None,
                linear_key_heads: None,
                linear_value_heads: None,
                linear_key_head_dim: None,
                linear_value_head_dim: None,
                linear_conv_kernel_dim: None,
                state_dtype: None,
                swa_window: None,
                num_swa_layers: None,
            },
            gpu: crate::context::GPUModel {
                name: Some("gpu".to_string()),
                vram_gb: Some(80.0),
                peak_flops_tc_tflops: peak_flops,
                peak_bw_gbps: peak_bw,
            },
            config: cfg,
            fp8_compiler_available: false,
        };
        let win = RuntimeWindow::from_snapshot(RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: snapshot,
            gpus: vec![GpuRawMetrics::default(); n_gpus],
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
        let b = out.expect("expected baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::EnvVar);
        let expected_decode = math::decode_ceiling_tps(3350.0, 10_000_000_000, 32);
        assert!((b.decode.expected - expected_decode).abs() < 1e-9);
        assert!(
            (b.weight_gb - 400.0).abs() < 1e-3,
            "weight_gb uses total param_count (100B × 32 bits / 8), got {}",
            b.weight_gb
        );
    }

    #[test]
    fn vllm_reported_dtype_beats_catalog() {
        let cfg = VllmConfig {
            vllm_reported_dtype: Some("fp8".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
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
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::VllmConfig);
        let expected_decode = math::decode_ceiling_tps(3350.0, 8_000_000_000, 8);
        assert!((b.decode.expected - expected_decode).abs() < 1e-9);
    }

    #[test]
    fn vllm_info_endpoint_used_when_resolved_from_info() {
        let cfg = VllmConfig {
            vllm_reported_dtype: Some("bfloat16".to_string()),
            vllm_reported_dtype_resolved: true,
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("fp32"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::VllmInfoEndpoint);
    }

    #[test]
    fn vllm_reported_dtype_beats_env_and_catalog() {
        let cfg = VllmConfig {
            vllm_reported_dtype: Some("fp8".to_string()),
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
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
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::VllmConfig);
        let expected_decode = math::decode_ceiling_tps(3350.0, 8_000_000_000, 8);
        assert!((b.decode.expected - expected_decode).abs() < 1e-9);
    }

    #[test]
    fn moe_weight_uses_total_params_roofline_uses_active() {
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            prompt_tokens_mean: Some(512.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(671_000_000_000),
            Some(37_000_000_000),
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input).expect("baseline");
        let expected_weight = math::weight_gb(671_000_000_000, 16);
        assert!((b.weight_gb - expected_weight).abs() < 1e-3);
        let expected_decode = math::decode_ceiling_tps(3350.0, 37_000_000_000, 16);
        assert!((b.decode.expected - expected_decode).abs() < 1e-6);
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
        let b = out.expect("expected baseline");
        assert!(b.prefill.is_none());
    }

    #[test]
    fn efficiency_clamped_at_100_when_above_hardware_ceiling() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(10_000.0),
            num_requests_running: Some(1.0),
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
        let b = compute(&input).expect("baseline");
        let absolute_ceiling = b.decode.expected * b.ridge_batch_size;
        assert!(
            10_000.0 > absolute_ceiling,
            "test setup: actual must exceed hardware ceiling (ceiling={absolute_ceiling})"
        );
        let eff = b.efficiency_pct.expect("efficiency");
        assert!((eff - 100.0).abs() < 1e-9);
        assert!((b.headroom_pct.expect("headroom") - 0.0).abs() < 1e-9);
    }

    #[test]
    fn efficiency_some_when_actual_below_decode_ceiling() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
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
        let b = compute(&input).expect("baseline");
        let eff = b.efficiency_pct.expect("efficiency");
        assert!((0.0..=100.0).contains(&eff));
        assert!(b.headroom_pct.is_some());
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
            vllm_reported_dtype: Some("unknown".to_string()),
            dtype: Some("mystery".to_string()),
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
        let b = out.expect("expected baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::Fallback);
    }

    #[test]
    fn resolve_seq_len_prefers_prompt_tokens_mean_when_both_present() {
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(8192),
            ..Default::default()
        };
        let snap_prompt = VllmRawMetrics {
            prompt_tokens_mean: Some(512.0),
            generation_tokens_per_sec: Some(10.0),
            ..Default::default()
        };
        let snap_no_prompt = VllmRawMetrics {
            prompt_tokens_mean: None,
            generation_tokens_per_sec: Some(10.0),
            ..Default::default()
        };
        let (ctx, win_short) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg.clone(),
            snap_prompt,
        );
        let (_, win_long) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap_no_prompt,
        );
        let input_short = AnalysisInput::new(&ctx, &win_short);
        let input_long = AnalysisInput::new(&ctx, &win_long);
        let b_short = compute(&input_short).expect("baseline");
        let b_long = compute(&input_long).expect("baseline");
        let ps = b_short.prefill.expect("prefill");
        let pl = b_long.prefill.expect("prefill");
        assert!(
            ps.expected > pl.expected,
            "shorter seq_len from prompt mean should raise prefill ceiling"
        );
    }

    #[test]
    fn resolve_seq_len_falls_back_to_max_model_len_when_prompt_missing() {
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(1024),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            prompt_tokens_mean: None,
            generation_tokens_per_sec: Some(10.0),
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
        let b = compute(&input);
        assert!(b.is_some());
        let expected = math::prefill_ops_per_sec(67.0, 8_000_000_000, 1024, 1, 2);
        let got = b.and_then(|x| x.prefill).expect("prefill").expected;
        assert!((got - expected).abs() < 1e-6);
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
        let b = out.expect("expected baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::Fallback);
    }

    #[test]
    fn hw_efficiency_uses_absolute_hardware_ceiling() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(4000.0),
            num_requests_running: Some(64.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(4800.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input).expect("baseline");
        let absolute_ceiling = b.decode.expected * b.ridge_batch_size;
        let expected = math::efficiency_pct(4000.0, absolute_ceiling);
        let eff = b.efficiency_pct.expect("efficiency");
        assert!(
            (eff - expected).abs() < 0.05,
            "expected ~{expected:.1}%, got {eff:.1}%"
        );
    }

    #[test]
    fn hw_efficiency_is_traffic_independent() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let actual_tps = 4000.0;
        let mut efficiencies = Vec::new();
        for running in [1.0, 4.0, 8.0, 16.0, 32.0] {
            let snap = VllmRawMetrics {
                generation_tokens_per_sec: Some(actual_tps),
                num_requests_running: Some(running),
                ..Default::default()
            };
            let (ctx, win) = baseline_input(
                Some(8_000_000_000),
                None,
                Some("bf16"),
                Some(67.0),
                Some(4800.0),
                cfg.clone(),
                snap,
            );
            let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
            efficiencies.push(b.efficiency_pct.expect("efficiency"));
        }
        let first = efficiencies[0];
        for (i, eff) in efficiencies.iter().enumerate().skip(1) {
            assert!(
                (eff - first).abs() < 1e-9,
                "running index {i}: expected {first}, got {eff}"
            );
        }
    }

    #[test]
    fn efficiency_some_when_num_running_zero() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(6043.0),
            num_requests_running: Some(0.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(4800.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input).expect("baseline");
        assert!(b.efficiency_pct.is_some());
        assert!(b.headroom_pct.is_some());
    }

    #[test]
    fn efficiency_some_when_num_running_missing() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(6043.0),
            num_requests_running: None,
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(4800.0),
            cfg,
            snap,
        );
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input).expect("baseline");
        assert!(b.efficiency_pct.is_some());
        assert!(b.headroom_pct.is_some());
    }

    #[test]
    fn tok_per_watt_from_power_and_generation_tps() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(100.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (mut ctx, mut win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        ctx.gpu.name = Some("NVIDIA H100 80GB HBM3".to_string());
        win.snapshot.gpus.first_mut().expect("gpu").power_watts = Some(50.0);
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = Some(50.0);
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input).expect("baseline");
        let cost = b.cost.expect("cost block");
        let tpw = cost.tok_per_watt.expect("tok/W");
        assert!((tpw - 2.0).abs() < 1e-9);
        assert_eq!(cost.joules_per_token, Some(0.5));
    }

    #[test]
    fn joules_per_token_none_when_power_missing() {
        // No power telemetry: energy fields absent; $/1M still fires from cost_per_hour.
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            cost_per_hour: Some(2.0),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(100.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx, mut win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        win.snapshot.gpus.first_mut().expect("gpu").power_watts = None;
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = None;
        let cost = compute(&AnalysisInput::new(&ctx, &win))
            .expect("baseline")
            .cost
            .expect("cost block");
        assert!(cost.tok_per_watt.is_none());
        assert!(cost.joules_per_token.is_none());
        assert!(cost.cost_per_million_tokens.is_some());
        assert_eq!(cost.cost_source, CostSource::UserProvided);
    }

    #[test]
    fn dollar_cost_survives_when_aligned_power_missing_but_raw_present() {
        // All windows skewed: raw power exists, aligned is None. Energy refuses the
        // bad join; $/1M tok still uses cost_per_hour × tok/s.
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            cost_per_hour: Some(3.6),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(100.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx, mut win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        win.snapshot.gpus.first_mut().expect("gpu").power_watts = Some(400.0);
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = None;
        let cost = compute(&AnalysisInput::new(&ctx, &win))
            .expect("baseline")
            .cost
            .expect("cost block");
        assert!(cost.tok_per_watt.is_none());
        assert!(cost.joules_per_token.is_none());
        let cpm = cost.cost_per_million_tokens.expect("$/1M");
        // 3.6 $/hr × 1e6 / (100 tok/s × 3600) = 10.0
        assert!((cpm - 10.0).abs() < 1e-9);
    }

    #[test]
    fn joules_per_token_none_when_tps_zero() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(0.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (mut ctx, mut win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        ctx.gpu.name = Some("NVIDIA H100 80GB HBM3".to_string());
        win.snapshot.gpus.first_mut().expect("gpu").power_watts = Some(50.0);
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = Some(50.0);
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert!(b.cost.is_none_or(|c| c.joules_per_token.is_none()));
    }

    #[test]
    fn joules_per_token_is_inverse_of_tok_per_watt() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(200.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (mut ctx, mut win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        ctx.gpu.name = Some("NVIDIA H100 80GB HBM3".to_string());
        win.snapshot.gpus.first_mut().expect("gpu").power_watts = Some(100.0);
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = Some(100.0);
        let cost = compute(&AnalysisInput::new(&ctx, &win))
            .expect("baseline")
            .cost
            .expect("cost");
        let tpw = cost.tok_per_watt.expect("tok/W");
        let jpt = cost.joules_per_token.expect("J/tok");
        assert!((tpw * jpt - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cost_per_million_tokens_catalog_h100() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(100.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (mut ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        ctx.gpu.name = Some("NVIDIA H100 80GB HBM3".to_string());
        // Cost join requires aligned power even for $/1M tok.
        let mut win = win;
        win.snapshot.gpus.first_mut().expect("gpu").power_watts = Some(400.0);
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = Some(400.0);
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input).expect("baseline");
        let cost = b.cost.expect("cost");
        assert_eq!(cost.cost_source, CostSource::Catalog);
        let cpm = cost.cost_per_million_tokens.expect("cpm");
        let expected = 2.99 * 1_000_000.0 / (100.0 * 3600.0);
        assert!((cpm - expected).abs() < 1e-6);
    }

    #[test]
    fn cost_source_user_provided_overrides_catalog() {
        let cfg = VllmConfig {
            cost_per_hour: Some(5.0),
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(200.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (mut ctx, mut win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        ctx.gpu.name = Some("NVIDIA H100 80GB HBM3".to_string());
        win.snapshot.gpus.first_mut().expect("gpu").power_watts = Some(400.0);
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = Some(400.0);
        let input = AnalysisInput::new(&ctx, &win);
        let cost = compute(&input).expect("baseline").cost.expect("cost");
        assert_eq!(cost.cost_source, CostSource::UserProvided);
        let expected = 5.0 * 1_000_000.0 / (200.0 * 3600.0);
        assert!((cost.cost_per_million_tokens.unwrap() - expected).abs() < 1e-6);
    }

    #[test]
    fn tp2_doubles_decode_ceiling() {
        let cfg_tp1 = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            tensor_parallel_size: Some(1),
            ..Default::default()
        };
        let cfg_tp2 = VllmConfig {
            tensor_parallel_size: Some(2),
            ..cfg_tp1.clone()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx1, win1) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg_tp1,
            snap.clone(),
        );
        let (ctx2, win2) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg_tp2,
            snap,
        );
        let b1 = compute(&AnalysisInput::new(&ctx1, &win1)).expect("tp1 baseline");
        let b2 = compute(&AnalysisInput::new(&ctx2, &win2)).expect("tp2 baseline");
        assert!(
            (b2.decode.expected - b1.decode.expected * 2.0).abs() < 1e-6,
            "tp2 decode {} expected ~2× tp1 {}",
            b2.decode.expected,
            b1.decode.expected
        );
    }

    #[test]
    fn kv_headroom_accounts_for_tp() {
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            tensor_parallel_size: Some(2),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(70_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert!((b.weight_gb - 140.0).abs() < 1e-3);
        let headroom = b.kv_headroom_gb.expect("kv headroom");
        let expected =
            (80.0 * DEFAULT_GPU_MEMORY_UTILIZATION) - math::ACTIVATION_KV_BUFFER_GB - (140.0 / 2.0);
        assert!(
            (headroom - expected).abs() < 1e-3,
            "expected {expected}GB kv headroom per GPU, got {headroom}"
        );
    }

    #[test]
    fn cost_none_when_tps_missing() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let (mut ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            VllmRawMetrics {
                generation_tokens_per_sec: None,
                num_requests_running: Some(1.0),
                ..Default::default()
            },
        );
        ctx.gpu.name = Some("NVIDIA H100 80GB HBM3".to_string());
        let no_tps = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert!(no_tps.cost.is_none());
    }

    #[test]
    fn catalog_dollar_cost_without_aligned_power() {
        // Catalog price + tok/s yields $/1M even when power telemetry is absent.
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let (mut ctx2, mut win2) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            VllmRawMetrics {
                generation_tokens_per_sec: Some(100.0),
                num_requests_running: Some(1.0),
                ..Default::default()
            },
        );
        ctx2.gpu.name = Some("NVIDIA H100 80GB HBM3".to_string());
        win2.snapshot.gpus.first_mut().expect("gpu").power_watts = None;
        win2.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = None;
        let cost = compute(&AnalysisInput::new(&ctx2, &win2))
            .expect("baseline")
            .cost
            .expect("catalog $/1M");
        assert!(cost.tok_per_watt.is_none());
        assert!(cost.joules_per_token.is_none());
        assert!(cost.cost_per_million_tokens.is_some());
        assert_eq!(cost.cost_source, CostSource::Catalog);
    }

    #[test]
    fn consumer_amd_no_price_omits_dollar_per_million() {
        // gpu_prices.json has no RX 7900 XTX row → no $/1M tok.
        assert!(crate::context::gpu_prices::lookup_gpu_price("AMD Radeon RX 7900 XTX").is_none());
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(100.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (mut ctx, mut win) = baseline_input(
            Some(8_000_000_000),
            None,
            Some("bf16"),
            Some(61.4),
            Some(960.0),
            cfg,
            snap,
        );
        ctx.gpu.name = Some("AMD Radeon RX 7900 XTX".to_string());
        win.snapshot.gpus.first_mut().expect("gpu").power_watts = Some(300.0);
        win.snapshot
            .gpus
            .first_mut()
            .expect("gpu")
            .aligned_power_watts = Some(300.0);
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert!(
            b.cost
                .as_ref()
                .and_then(|c| c.cost_per_million_tokens)
                .is_none()
        );
    }

    #[test]
    fn awq_quantization_source_used_when_info_provides_scheme() {
        let cfg = VllmConfig {
            vllm_reported_quantization: Some("awq".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(70_000_000_000),
            None,
            Some("bf16"),
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert_eq!(
            b.weight_dtype_source,
            WeightDtypeSource::VllmInfoQuantization
        );
        assert!((b.weight_gb - 35.0).abs() < 1e-3);
        let expected_decode = math::decode_ceiling_tps(3350.0, 70_000_000_000, 4);
        assert!(
            (b.decode.expected - expected_decode).abs() < 1e-6,
            "AWQ 4-bit decode ceiling should be 4× higher than bf16; got {}",
            b.decode.expected
        );
    }

    #[test]
    fn env_var_quantization_used_when_info_missing() {
        let cfg = VllmConfig {
            quantization: Some("gptq".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
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
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::EnvVarQuantization);
    }

    #[test]
    fn unrecognized_quantization_falls_through_to_dtype_chain() {
        let cfg = VllmConfig {
            vllm_reported_quantization: Some("gguf".to_string()),
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
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
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::EnvVar);
    }

    #[test]
    fn quantization_beats_prometheus_dtype_for_awq() {
        let cfg = VllmConfig {
            vllm_reported_dtype: Some("bfloat16".to_string()),
            vllm_reported_quantization: Some("awq".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(1.0),
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
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert_eq!(
            b.weight_dtype_source,
            WeightDtypeSource::VllmInfoQuantization
        );
        let expected_decode = math::decode_ceiling_tps(3350.0, 8_000_000_000, 4);
        assert!((b.decode.expected - expected_decode).abs() < 1e-9);
    }

    fn baseline_input_llama8b(
        snapshot: VllmRawMetrics,
        cfg: VllmConfig,
    ) -> (StaticContext, RuntimeWindow) {
        let ctx = StaticContext {
            model: crate::context::ModelArch {
                param_count: Some(8_000_000_000),
                active_param_count: None,
                num_layers: Some(32),
                hidden_dim: Some(4096),
                default_weight_dtype: Some("bf16".to_string()),
                num_kv_heads: Some(8),
                head_dim: Some(128),
                num_kv_layers: None,
                attn_flops_coeff: None,
                linear_num_layers: None,
                linear_key_heads: None,
                linear_value_heads: None,
                linear_key_head_dim: None,
                linear_value_head_dim: None,
                linear_conv_kernel_dim: None,
                state_dtype: None,
                swa_window: None,
                num_swa_layers: None,
            },
            gpu: crate::context::GPUModel {
                name: Some("NVIDIA H100 80GB HBM3".to_string()),
                vram_gb: Some(80.0),
                peak_flops_tc_tflops: Some(67.0),
                peak_bw_gbps: Some(3350.0),
            },
            config: cfg,
            fp8_compiler_available: false,
        };
        let win = RuntimeWindow::from_snapshot(RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: snapshot,
            gpus: vec![GpuRawMetrics::default()],
        });
        (ctx, win)
    }

    #[test]
    fn prefill_ceiling_uses_attention_correction_when_hidden_dim_known() {
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(8192),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            prompt_tokens_mean: Some(2048.0),
            generation_tokens_per_sec: Some(10.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input_llama8b(snap, cfg);
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        let p = b.prefill.expect("prefill");
        let linear_only = math::prefill_ops_per_sec(67.0, 8_000_000_000, 2048, 0, 0);
        assert!(p.expected < linear_only);
    }

    #[test]
    fn prefill_ceiling_uses_explicit_attn_coeff_for_mla() {
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(4096),
            ..Default::default()
        };
        let mut ctx = baseline_input_llama8b(VllmRawMetrics::default(), cfg.clone()).0;
        ctx.model.attn_flops_coeff = Some(139_264);
        ctx.model.param_count = Some(37_000_000_000);
        ctx.model.active_param_count = Some(37_000_000_000);
        ctx.model.num_layers = Some(61);
        let win = RuntimeWindow::from_snapshot(RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: VllmRawMetrics {
                prompt_tokens_mean: Some(4096.0),
                generation_tokens_per_sec: Some(10.0),
                ..Default::default()
            },
            gpus: vec![GpuRawMetrics::default()],
        });
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        let expected = math::prefill_ops_per_sec(67.0, 37_000_000_000, 4096, 61, 139_264);
        assert!((b.prefill.expect("prefill").expected - expected).abs() < 1e-6);
    }

    #[test]
    fn prefill_ceiling_falls_back_to_linear_when_hidden_dim_and_coeff_both_none() {
        let cfg = VllmConfig {
            max_model_len: Some(1024),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(10.0),
            ..Default::default()
        };
        let (ctx, win) = baseline_input(
            Some(8_000_000_000),
            None,
            None,
            Some(67.0),
            Some(3350.0),
            cfg,
            snap,
        );
        let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        let expected = math::prefill_ops_per_sec(67.0, 8_000_000_000, 1024, 0, 0);
        assert!((b.prefill.expect("prefill").expected - expected).abs() < 1e-6);
    }
}
