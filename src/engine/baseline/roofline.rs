use crate::context::{gpu_prices, AnalysisInput};

use super::math;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostEstimate {
    /// Tokens generated per watt of power draw. None if power_watts unavailable.
    pub tok_per_watt: Option<f64>,
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
    /// No cost data available.
    None,
}

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
    /// Model weight memory footprint in GB (params × bytes_per_param).
    pub weight_gb: f64,
    /// VRAM remaining after weights, if total VRAM is known. Negative means weights alone exceed VRAM.
    pub kv_headroom_gb: Option<f64>,
    /// Theoretical minimum time-per-output-token at decode ceiling (ms).
    pub tpot_floor_ms: f64,
    /// Theoretical minimum prefill latency at prefill ceiling (ms). None when seq_len unknown.
    pub prefill_latency_floor_ms: Option<f64>,
    /// Concurrent batch size at which decode crosses from BW-bound to compute-bound (roofline ridge).
    pub ridge_batch_size: f64,
    pub cost: Option<CostEstimate>,
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

    let ridge_batch_size = math::ridge_batch_size(peak_flops, peak_bw, bytes_per_param);

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

    let ceiling = decode.expected;
    let num_running = input
        .window
        .snapshot
        .vllm
        .num_requests_running
        .filter(|v| v.is_finite() && *v > 0.0);

    let efficiency_pct = input
        .window
        .snapshot
        .vllm
        .generation_tokens_per_sec
        .filter(|v| v.is_finite() && *v > 0.0)
        .zip(num_running)
        .and_then(|(actual, running)| {
            let aggregate_ceiling = ceiling * running;
            if actual <= aggregate_ceiling {
                let pct = math::efficiency_pct(actual, aggregate_ceiling);
                pct.is_finite().then_some(pct)
            } else {
                None
            }
        });

    let headroom_pct = efficiency_pct.map(|raw| 100.0 - raw.min(100.0));

    let weight_gb = math::weight_gb(model_params, bytes_per_param);
    let kv_headroom_gb = ctx.gpu.vram_gb.map(|vram| vram - weight_gb);
    let tpot_floor_ms = math::latency_floor_ms(decode.expected);
    let prefill_latency_floor_ms = prefill.map(|p| math::latency_floor_ms(p.expected));

    let snap = &input.window.snapshot;
    let tps = snap
        .vllm
        .generation_tokens_per_sec
        .filter(|v| v.is_finite() && *v > 0.0);
    let power_watts = snap.gpu.power_watts.filter(|v| v.is_finite() && *v > 0.0);

    let tok_per_watt = match (tps, power_watts) {
        (Some(t), Some(p)) => Some(t / p),
        _ => None,
    };

    let (cost_per_hr, cost_source) = if let Some(hr) = ctx
        .config
        .cost_per_hour
        .filter(|v| v.is_finite() && *v > 0.0)
    {
        (Some(hr), CostSource::UserProvided)
    } else if let Some(gpu_name) = ctx.gpu.name.as_deref() {
        gpu_prices::lookup_gpu_price(gpu_name)
            .map(|p| (Some(p.on_demand_per_hr), CostSource::Catalog))
            .unwrap_or((None, CostSource::None))
    } else {
        (None, CostSource::None)
    };

    let cost_per_million_tokens = match (cost_per_hr, tps, cost_source) {
        (Some(hr), Some(t), CostSource::UserProvided | CostSource::Catalog) if t > 0.0 => {
            let cpm = hr * 1_000_000.0 / (t * 3600.0);
            cpm.is_finite().then_some(cpm)
        }
        _ => None,
    };

    let cost = if tok_per_watt.is_some() || cost_per_million_tokens.is_some() {
        Some(CostEstimate {
            tok_per_watt,
            cost_per_million_tokens,
            cost_source,
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
        kv_headroom_gb,
        tpot_floor_ms,
        prefill_latency_floor_ms,
        ridge_batch_size,
        cost,
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
    fn efficiency_none_when_actual_above_decode_ceiling() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(300.0),
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
        let ceiling = b.decode.expected;
        assert!(
            300.0 > ceiling,
            "test setup: actual must exceed ceiling (ceiling={ceiling})"
        );
        assert!(b.efficiency_pct.is_none());
        assert!(b.headroom_pct.is_none());
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
        let expected = math::prefill_ceiling_tps(67.0, 8_000_000_000, 1024);
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
        let b = out.unwrap();
        assert_eq!(b.weight_dtype_source, WeightDtypeSource::Fallback);
    }

    #[test]
    fn efficiency_accounts_for_batch_size() {
        let cfg = VllmConfig {
            kv_cache_dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let snap = VllmRawMetrics {
            generation_tokens_per_sec: Some(6043.0),
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
        let per_seq_ceiling = b.decode.expected;
        let aggregate_ceiling = per_seq_ceiling * 64.0;
        let expected = math::efficiency_pct(6043.0, aggregate_ceiling);
        let eff = b.efficiency_pct.expect("efficiency");
        assert!(
            (eff - expected).abs() < 0.05,
            "expected ~{expected:.1}%, got {eff:.1}%"
        );
        assert!((eff - 31.5).abs() < 0.5, "expected ~31.5%, got {eff:.1}%");
    }

    #[test]
    fn efficiency_none_when_num_running_zero() {
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
        assert!(b.efficiency_pct.is_none());
        assert!(b.headroom_pct.is_none());
    }

    #[test]
    fn efficiency_none_when_num_running_missing() {
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
        assert!(b.efficiency_pct.is_none());
        assert!(b.headroom_pct.is_none());
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
        win.snapshot.gpu.power_watts = Some(50.0);
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input).expect("baseline");
        let cost = b.cost.expect("cost block");
        let tpw = cost.tok_per_watt.expect("tok/W");
        assert!((tpw - 2.0).abs() < 1e-9);
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
        let input = AnalysisInput::new(&ctx, &win);
        let b = compute(&input).expect("baseline");
        let cost = b.cost.expect("cost");
        assert_eq!(cost.cost_source, CostSource::Catalog);
        let cpm = cost.cost_per_million_tokens.expect("cpm");
        let expected = 2.80 * 1_000_000.0 / (100.0 * 3600.0);
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
        let input = AnalysisInput::new(&ctx, &win);
        let cost = compute(&input).expect("baseline").cost.expect("cost");
        assert_eq!(cost.cost_source, CostSource::UserProvided);
        let expected = 5.0 * 1_000_000.0 / (200.0 * 3600.0);
        assert!((cost.cost_per_million_tokens.unwrap() - expected).abs() < 1e-6);
    }

    #[test]
    fn cost_none_when_power_or_tps_missing() {
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
            cfg.clone(),
            VllmRawMetrics {
                generation_tokens_per_sec: None,
                num_requests_running: Some(1.0),
                ..Default::default()
            },
        );
        ctx.gpu.name = Some("NVIDIA H100 80GB HBM3".to_string());
        let no_tps = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
        assert!(no_tps.cost.is_none());

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
        win2.snapshot.gpu.power_watts = None;
        let cost = compute(&AnalysisInput::new(&ctx2, &win2))
            .expect("baseline")
            .cost
            .expect("cost without power");
        assert!(cost.tok_per_watt.is_none());
        assert!(cost.cost_per_million_tokens.is_some());
    }
}
