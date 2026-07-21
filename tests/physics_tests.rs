use std::time::SystemTime;

use profile::{
    collectors::{GpuRawMetrics, RawSnapshot, VllmConfig, VllmRawMetrics},
    context::{AnalysisInput, ModelArch, RuntimeWindow, StaticContext},
    engine::baseline::compute,
};

fn build_input(
    model_name: &str,
    gpu_name: &str,
    cfg: VllmConfig,
    prompt_tokens_mean: Option<f64>,
    generation_tokens_per_sec: Option<f64>,
    num_requests_running: Option<f64>,
) -> (StaticContext, RuntimeWindow) {
    let snap = RawSnapshot {
        gpu_observed_at: SystemTime::UNIX_EPOCH,
        vllm_observed_at: SystemTime::UNIX_EPOCH,
        timestamp: SystemTime::UNIX_EPOCH,
        vllm: VllmRawMetrics {
            model_name: Some(model_name.to_string()),
            prompt_tokens_mean,
            generation_tokens_per_sec,
            num_requests_running,
            ..Default::default()
        },
        gpus: vec![GpuRawMetrics {
            gpu_name: Some(gpu_name.to_string()),
            ..Default::default()
        }],
    };
    let mut ctx = StaticContext::from_snapshot(&snap, cfg);
    if model_name == "test/Llama-3.1-70B" {
        ctx.model = ModelArch {
            param_count: Some(70_000_000_000),
            num_layers: Some(80),
            hidden_dim: Some(8192),
            default_weight_dtype: Some("bf16".to_string()),
            num_kv_heads: Some(8),
            head_dim: Some(128),
            ..Default::default()
        };
    }
    let win = RuntimeWindow::from_snapshot(snap);
    (ctx, win)
}

#[test]
fn h100_sxm_llama3_70b_decode_ceiling_is_about_23_9_tok_s() {
    // H100 SXM bandwidth = 3350 GB/s; synthetic Llama-3 70B geometry; bf16 = 2 bytes.
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let (ctx, win) = build_input(
        "test/Llama-3.1-70B",
        "NVIDIA H100 80GB HBM3",
        cfg,
        Some(2048.0),
        Some(10.0),
        None,
    );
    let input = AnalysisInput::new(&ctx, &win);
    let baseline = compute(&input);
    assert!(baseline.is_some());
    let b = match baseline {
        Some(v) => v,
        None => panic!("baseline missing"),
    };
    let expected = 23.9285714286;
    let err = ((b.decode.expected - expected) / expected).abs();
    assert!(err < 0.01);
    let prefill = b.prefill;
    assert!(prefill.is_some());
    let p = match prefill {
        Some(v) => v,
        None => panic!("prefill missing"),
    };
    assert!(p.expected.is_finite() && p.expected > 0.0);
}

#[test]
fn h100_sxm_llama3_8b_decode_ceiling_is_about_209_tok_s() {
    // Assumes catalog: H100 SXM bandwidth = 3350 GB/s, Llama-3 8B params, bf16 = 2 bytes.
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let (ctx, win) = build_input(
        "meta-llama/Llama-3.1-8B-Instruct",
        "NVIDIA H100 80GB HBM3",
        cfg,
        Some(2048.0),
        Some(10.0),
        None,
    );
    let input = AnalysisInput::new(&ctx, &win);
    let baseline = compute(&input);
    assert!(baseline.is_some());
    let b = match baseline {
        Some(v) => v,
        None => panic!("baseline missing"),
    };
    let expected = 209.375;
    let err = ((b.decode.expected - expected) / expected).abs();
    assert!(err < 0.01);
}

#[test]
fn a100_80gb_llama3_70b_decode_ceiling_is_about_14_6_tok_s() {
    // A100 80GB bandwidth = 2039 GB/s; synthetic Llama-3 70B geometry; bf16 = 2 bytes.
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let (ctx, win) = build_input(
        "test/Llama-3.1-70B",
        "NVIDIA A100-SXM4-80GB",
        cfg,
        Some(2048.0),
        Some(10.0),
        None,
    );
    let input = AnalysisInput::new(&ctx, &win);
    let baseline = compute(&input);
    assert!(baseline.is_some());
    let b = match baseline {
        Some(v) => v,
        None => panic!("baseline missing"),
    };
    let expected = 14.5642857143;
    let err = ((b.decode.expected - expected) / expected).abs();
    assert!(err < 0.01);
}

#[test]
fn compute_returns_none_when_gpu_not_in_catalog() {
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let (ctx, win) = build_input(
        "test/Llama-3.1-70B",
        "NVIDIA Tesla V100",
        cfg,
        Some(2048.0),
        Some(10.0),
        None,
    );
    let input = AnalysisInput::new(&ctx, &win);
    assert!(compute(&input).is_none());
}

#[test]
fn compute_returns_none_when_model_not_in_catalog() {
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let (ctx, win) = build_input(
        "vendor/unknown-13b",
        "NVIDIA H100 80GB HBM3",
        cfg,
        Some(2048.0),
        Some(10.0),
        None,
    );
    let input = AnalysisInput::new(&ctx, &win);
    assert!(compute(&input).is_none());
}

#[test]
fn prefill_is_none_when_seq_len_unavailable() {
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: None,
        ..Default::default()
    };
    let (ctx, win) = build_input(
        "test/Llama-3.1-70B",
        "NVIDIA H100 80GB HBM3",
        cfg,
        None,
        Some(10.0),
        None,
    );
    let input = AnalysisInput::new(&ctx, &win);
    let baseline = compute(&input);
    assert!(baseline.is_some());
    let b = match baseline {
        Some(v) => v,
        None => panic!("baseline missing"),
    };
    assert!(b.prefill.is_none());
}

#[test]
fn efficiency_is_none_when_actual_tps_missing() {
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let (ctx, win) = build_input(
        "test/Llama-3.1-70B",
        "NVIDIA H100 80GB HBM3",
        cfg,
        Some(2048.0),
        None,
        None,
    );
    let input = AnalysisInput::new(&ctx, &win);
    let baseline = compute(&input);
    assert!(baseline.is_some());
    let b = match baseline {
        Some(v) => v,
        None => panic!("baseline missing"),
    };
    assert!(b.efficiency_pct.is_none());
    assert!(b.headroom_pct.is_none());
}

#[test]
fn efficiency_clamped_at_100_when_above_hardware_ceiling() {
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let (ctx, win) = build_input(
        "test/Llama-3.1-70B",
        "NVIDIA H100 80GB HBM3",
        cfg,
        Some(2048.0),
        Some(10_000.0),
        Some(1.0),
    );
    let input = AnalysisInput::new(&ctx, &win);
    let baseline = compute(&input);
    assert!(baseline.is_some());
    let b = match baseline {
        Some(v) => v,
        None => panic!("baseline missing"),
    };
    let absolute_ceiling = b.decode.expected * b.ridge_batch_size;
    assert!(
        10_000.0 > absolute_ceiling,
        "test setup: actual must exceed hardware ceiling"
    );
    let eff = b.efficiency_pct.expect("efficiency");
    assert!((eff - 100.0).abs() < 1e-9);
    assert!((b.headroom_pct.expect("headroom") - 0.0).abs() < 1e-9);
}

#[test]
fn efficiency_some_in_zero_to_100_when_below_ceiling() {
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let (ctx, win) = build_input(
        "test/Llama-3.1-70B",
        "NVIDIA H100 80GB HBM3",
        cfg,
        Some(2048.0),
        Some(10.0),
        Some(1.0),
    );
    let input = AnalysisInput::new(&ctx, &win);
    let b = compute(&input).expect("baseline");
    assert!(
        10.0 <= b.decode.expected * 1.0,
        "test setup: actual must be at or below decode ceiling"
    );
    let eff = b.efficiency_pct.expect("efficiency");
    assert!((0.0..=100.0).contains(&eff));
    assert!(b.headroom_pct.is_some());
}
