use std::time::SystemTime;

use profile::{
    collectors::{GpuRawMetrics, RawSnapshot, VllmConfig, VllmRawMetrics},
    context::{AnalysisInput, RuntimeWindow, StaticContext},
    engine::baseline::compute,
};

fn build_input(
    model_name: &str,
    gpu_name: &str,
    cfg: VllmConfig,
    prompt_tokens_mean: Option<f64>,
    generation_tokens_per_sec: Option<f64>,
) -> (StaticContext, RuntimeWindow) {
    let snap = RawSnapshot {
        gpu_observed_at: SystemTime::UNIX_EPOCH,
        vllm_observed_at: SystemTime::UNIX_EPOCH,
        timestamp: SystemTime::UNIX_EPOCH,
        vllm: VllmRawMetrics {
            model_name: Some(model_name.to_string()),
            prompt_tokens_mean,
            generation_tokens_per_sec,
            ..Default::default()
        },
        gpu: GpuRawMetrics {
            gpu_name: Some(gpu_name.to_string()),
            ..Default::default()
        },
    };
    let ctx = StaticContext::from_snapshot(&snap, cfg);
    let win = RuntimeWindow::from_snapshot(snap);
    (ctx, win)
}

#[test]
fn h100_sxm_llama3_70b_decode_ceiling_is_about_23_9_tok_s() {
    // Assumes catalog: H100 SXM bandwidth = 3350 GB/s, Llama-3 70B params, bf16 = 2 bytes.
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let (ctx, win) = build_input(
        "meta-llama/Llama-3.1-70B-Instruct",
        "NVIDIA H100 80GB HBM3",
        cfg,
        Some(2048.0),
        Some(10.0),
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
    // Assumes catalog: A100 80GB bandwidth = 2039 GB/s, Llama-3 70B params, bf16 = 2 bytes.
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let (ctx, win) = build_input(
        "meta-llama/Llama-3.1-70B-Instruct",
        "NVIDIA A100-SXM4-80GB",
        cfg,
        Some(2048.0),
        Some(10.0),
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
        "meta-llama/Llama-3.1-70B-Instruct",
        "NVIDIA Tesla V100",
        cfg,
        Some(2048.0),
        Some(10.0),
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
        "meta-llama/Llama-3.1-70B-Instruct",
        "NVIDIA H100 80GB HBM3",
        cfg,
        None,
        Some(10.0),
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
        "meta-llama/Llama-3.1-70B-Instruct",
        "NVIDIA H100 80GB HBM3",
        cfg,
        Some(2048.0),
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
fn efficiency_can_exceed_100_and_headroom_floors_to_zero() {
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let (ctx, win) = build_input(
        "meta-llama/Llama-3.1-70B-Instruct",
        "NVIDIA H100 80GB HBM3",
        cfg,
        Some(2048.0),
        Some(1000.0),
    );
    let input = AnalysisInput::new(&ctx, &win);
    let baseline = compute(&input);
    assert!(baseline.is_some());
    let b = match baseline {
        Some(v) => v,
        None => panic!("baseline missing"),
    };
    let raw = b.efficiency_pct.unwrap_or(0.0);
    assert!(raw > 100.0);
    assert_eq!(b.headroom_pct, Some(0.0));
}
