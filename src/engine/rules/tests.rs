use super::format::r2_kv_cache_advisory;
use super::r1_under_batching::{r1_recommendation, rule1_under_batching_with_efficiency};
use super::r2_kv_cache_pressure::{
    KvCapacityLabel, R2RecommendationInput, Rule2Outcome, r2_recommendation,
    rule2_kv_cache_pressure,
};
use super::r3_low_prefix_reuse::{format_low_prefix_hit_rate_fired, rule3_low_prefix_reuse};
use super::*;
use crate::collectors::{GpuRawMetrics, RawSnapshot, VllmConfig, VllmRawMetrics};
use crate::context::{AnalysisInput, ModelArch, RuntimeWindow, StaticContext};
use std::time::{Duration, SystemTime};

fn format_diagnose_rules_test(
    ctx: &StaticContext,
    win: &RuntimeWindow,
    verbose: bool,
    metrics_url: &str,
) -> Vec<String> {
    let windows: Vec<_> = (0..ENGINE_MIN_PERSISTENT_WINDOWS)
        .map(|_| win.clone())
        .collect();
    let summary = ai(ctx, &windows[0]);
    format_diagnose_rules_for_windows_test(&windows, summary, verbose, metrics_url)
}

fn format_diagnose_rules_for_windows_test(
    windows: &[RuntimeWindow],
    summary: AnalysisInput<'_>,
    verbose: bool,
    metrics_url: &str,
) -> Vec<String> {
    let report = build_report_for_windows(windows, summary);
    format_diagnose_rules_for_windows(windows, summary, &report, verbose, metrics_url, 30, false)
}

fn hint_for_empty<'a>(
    ctx: &'a StaticContext,
    win: &'a RuntimeWindow,
    metrics_url: &'a str,
    duration_secs: u64,
) -> LoadHintParams<'a> {
    LoadHintParams {
        model_name: win.snapshot.vllm.model_name.as_deref(),
        metrics_url,
        max_num_seqs: ctx.config.max_num_seqs,
        duration_secs,
    }
}

fn snap(
    gpu_at: SystemTime,
    vllm_at: SystemTime,
    vllm: VllmRawMetrics,
    gpu: GpuRawMetrics,
) -> RawSnapshot {
    crate::collectors::RawSnapshotFixture::default()
        .observed_at(gpu_at, vllm_at)
        .vllm(vllm)
        .gpus(vec![gpu])
        .build()
}

fn mk_ctx() -> StaticContext {
    StaticContext::default()
}

fn mk_win(s: RawSnapshot) -> RuntimeWindow {
    RuntimeWindow::from_snapshot(s)
}

fn ai<'a>(ctx: &'a StaticContext, win: &'a RuntimeWindow) -> AnalysisInput<'a> {
    AnalysisInput { ctx, window: win }
}

fn input_r4_suppresses_r2() -> (StaticContext, RuntimeWindow) {
    let t = SystemTime::UNIX_EPOCH;
    let snap = RawSnapshot {
        gpu_observed_at: t,
        vllm_observed_at: t,
        timestamp: t,
        vllm: VllmRawMetrics {
            model_name: Some("test/oversized-70b".to_string()),
            num_requests_running: Some(3.0),
            num_requests_waiting: Some(0.0),
            max_num_seqs: Some(256),
            kv_cache_usage_perc: Some(89.0),
            num_preemptions_per_sec: Some(0.05),
            generation_tokens_per_sec: Some(50.0),
            request_success_per_sec: Some(10.0),
            window_duration_secs: Some(2.0),
            ..Default::default()
        },
        gpus: vec![GpuRawMetrics {
            gpu_name: Some("NVIDIA H100 80GB HBM3".to_string()),
            vram_total_mb: Some(80 * 1024),
            gpu_util_pct: Some(58.0),
            ..Default::default()
        }],
    };
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let mut ctx = StaticContext::from_snapshot(&snap, cfg);
    // Synthetic oversized model keeps this suppression test independent of
    // which models the single-GPU launch catalog supports.
    ctx.model.param_count = Some(70_000_000_000);
    let win = RuntimeWindow::from_snapshot(snap);
    (ctx, win)
}

fn vllm_base() -> VllmRawMetrics {
    VllmRawMetrics {
        num_requests_running: Some(3.1),
        num_requests_waiting: Some(0.0),
        max_num_seqs: Some(256),
        kv_cache_usage_perc: Some(50.0),
        prefix_cache_hit_rate: Some(0.5),
        request_success_per_sec: Some(10.0),
        window_duration_secs: Some(2.0),
        ..Default::default()
    }
}

fn gpu_low() -> GpuRawMetrics {
    GpuRawMetrics {
        gpu_util_pct: Some(58.0),
        ..Default::default()
    }
}

fn gpu_busy() -> GpuRawMetrics {
    GpuRawMetrics {
        gpu_util_pct: Some(75.0),
        ..Default::default()
    }
}

fn gpu_busy_with_vram() -> GpuRawMetrics {
    GpuRawMetrics {
        gpu_name: Some("NVIDIA H100 80GB HBM3".to_string()),
        gpu_util_pct: Some(75.0),
        vram_used_mb: Some(40 * 1024),
        vram_total_mb: Some(80 * 1024),
        ..Default::default()
    }
}

fn r1_test_input(snapshot: &RawSnapshot) -> R1EvalInput<'_> {
    R1EvalInput {
        snapshot,
        config_max_num_seqs: None,
        efficiency_pct: None,
        config_relative_efficiency_pct: None,
        prompt_tokens_per_sec: None,
        generation_tokens_per_sec: None,
        prefix_cache_hit_rate: None,
        ridge_batch_size: None,
        r6_fired: false,
    }
}

fn mk_r1_window_with_kv(kv_pct: f64) -> RuntimeWindow {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.kv_cache_usage_perc = Some(kv_pct);
    mk_win(snap(t, t, v, gpu_low()))
}

fn vllm_high_kv() -> VllmRawMetrics {
    VllmRawMetrics {
        kv_cache_usage_perc: Some(89.0),
        ..vllm_base()
    }
}

fn r2_report(windows: &[RuntimeWindow]) -> crate::engine::Report {
    let ctx = mk_ctx();
    let summary = ai(&ctx, windows.last().expect("windows"));
    build_report_for_windows(windows, summary)
}

fn vllm_high_kv_stressed() -> VllmRawMetrics {
    VllmRawMetrics {
        kv_cache_usage_perc: Some(89.0),
        num_preemptions_per_sec: Some(0.05),
        ..vllm_base()
    }
}

fn mk_evaluable_kv_window(kv_pct: f64, preemptions: bool) -> RuntimeWindow {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.kv_cache_usage_perc = Some(kv_pct);
    v.generation_tokens_per_sec = Some(100.0);
    v.num_requests_running = Some(100.0);
    if preemptions {
        v.num_preemptions_per_sec = Some(0.05);
    }
    mk_win(snap(t, t, v, gpu_busy()))
}

fn mk_evaluable_backlog_window(
    kv_pct: f64,
    wait: f64,
    run: f64,
    prompt_mean: f64,
    num_gpu_blocks: u32,
    block_size: u32,
) -> RuntimeWindow {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.kv_cache_usage_perc = Some(kv_pct);
    v.num_requests_waiting = Some(wait);
    v.num_requests_running = Some(run);
    v.prompt_tokens_mean = Some(prompt_mean);
    v.generation_tokens_per_sec = Some(100.0);
    v.cache_config = crate::collectors::CacheConfigLabels {
        num_gpu_blocks: Some(num_gpu_blocks),
        block_size: Some(block_size),
        ..Default::default()
    };
    mk_win(snap(t, t, v, gpu_busy_with_vram()))
}

fn r2_issue_lines(windows: &[RuntimeWindow]) -> Vec<String> {
    let ctx = mk_ctx();
    let summary = ai(&ctx, windows.last().expect("windows"));
    format_diagnose_rules_for_windows_test(windows, summary, false, "http://127.0.0.1:8000/metrics")
}

fn mk_evaluable_concurrency_saturation_window(
    run: f64,
    wait: f64,
    max_num_seqs: u32,
) -> RuntimeWindow {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.num_requests_running = Some(run);
    v.num_requests_waiting = Some(wait);
    v.max_num_seqs = Some(max_num_seqs);
    v.generation_tokens_per_sec = Some(100.0);
    mk_win(snap(t, t, v, gpu_busy()))
}

fn mk_llama8b_h100_ctx(s: &RawSnapshot) -> StaticContext {
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(8192),
        max_num_seqs: Some(256),
        ..Default::default()
    };
    StaticContext::from_snapshot(s, cfg)
}

fn mk_r7_headroom_window(running: f64, max_num_seqs: u32, waiting: f64, tps: f64) -> RuntimeWindow {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
    v.generation_tokens_per_sec = Some(tps);
    v.num_requests_running = Some(running);
    v.num_requests_waiting = Some(waiting);
    v.max_num_seqs = Some(max_num_seqs);
    v.kv_cache_usage_perc = Some(3.3);
    v.window_duration_secs = Some(2.0);
    let mut g = gpu_busy();
    g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
    mk_win(snap(t, t, v, g))
}

fn mk_r7_ctx(max_num_seqs: u32) -> StaticContext {
    let snap = mk_r7_headroom_window(5.0, max_num_seqs, 0.0, 50.0).snapshot;
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(8192),
        max_num_seqs: Some(max_num_seqs),
        ..Default::default()
    };
    StaticContext::from_snapshot(&snap, cfg)
}

fn mk_r6_prefill_window(
    prompt_gen_ratio: f64,
    gen_tps: f64,
    running: f64,
    tpot_ms: Option<f64>,
) -> RuntimeWindow {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
    v.generation_tokens_per_sec = Some(gen_tps);
    v.prompt_tokens_per_sec = Some(gen_tps * prompt_gen_ratio);
    v.num_requests_running = Some(running);
    v.num_requests_waiting = Some(0.0);
    v.tpot_ms = tpot_ms.or(Some(50.0));
    v.window_duration_secs = Some(2.0);
    v.prompt_tokens_mean = Some(2048.0);
    v.cache_config.enable_prefix_caching = Some(true);
    v.prefix_cache_hit_rate = Some(0.5);
    let mut g = gpu_busy();
    g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
    mk_win(snap(t, t, v, g))
}

#[test]
fn compute_kv_max_seqs_uses_kv_layers_over_total_layers() {
    // Qwen3.6-27B hybrid shape: 16/64 full-attention layers → 4× KV capacity vs num_layers.
    let hybrid = ModelArch {
        num_kv_heads: Some(4),
        head_dim: Some(256),
        num_layers: Some(64),
        num_kv_layers: Some(16),
        ..Default::default()
    };
    #[allow(clippy::cast_precision_loss)]
    let headroom_gb = (1u64 << 34) as f64 / 1e9;
    let with_kv_layers = compute_kv_max_seqs_for_cache(
        Some(headroom_gb),
        Some(4096),
        &hybrid,
        None,
        None,
        &crate::collectors::CacheConfigLabels::default(),
    )
    .max_seqs;

    let dense = ModelArch {
        num_kv_layers: None,
        ..hybrid
    };
    let without_kv_layers = compute_kv_max_seqs_for_cache(
        Some(headroom_gb),
        Some(4096),
        &dense,
        None,
        None,
        &crate::collectors::CacheConfigLabels::default(),
    )
    .max_seqs;

    assert!(with_kv_layers.is_some() && without_kv_layers.is_some());
    assert_eq!(with_kv_layers.unwrap(), without_kv_layers.unwrap() * 4);
}

#[test]
fn compute_kv_max_seqs_tp2_doubles_capacity() {
    let model = ModelArch {
        num_kv_heads: Some(8),
        head_dim: Some(128),
        num_layers: Some(32),
        ..Default::default()
    };
    let headroom_gb = 20.0;
    let tp1 = compute_kv_max_seqs_with_mode::<true>(
        Some(headroom_gb),
        Some(4096),
        &model,
        None,
        Some(1),
        None,
    )
    .max_seqs;
    let tp2 = compute_kv_max_seqs_with_mode::<true>(
        Some(headroom_gb),
        Some(4096),
        &model,
        None,
        Some(2),
        None,
    )
    .max_seqs;
    assert_eq!(tp2.unwrap(), tp1.unwrap() * 2);
}

#[test]
fn compute_kv_max_seqs_tp2_uses_one_model_view_for_budget_and_cost() {
    let model = ModelArch {
        num_kv_heads: Some(8),
        head_dim: Some(128),
        num_layers: Some(32),
        ..Default::default()
    };
    let cache = crate::collectors::CacheConfigLabels {
        num_gpu_blocks: Some(1024),
        block_size: Some(16),
        ..Default::default()
    };
    let result = compute_kv_max_seqs_with_mode::<true>(
        None,
        Some(4096),
        &model,
        None,
        Some(2),
        Some(&cache),
    );

    // Shared sharding cancels from budget / request cost:
    // 1024 blocks × 16 tokens / 4096 tokens per request = 4.
    assert_eq!(result.max_seqs, Some(4));
}

#[test]
fn compute_kv_max_seqs_non_divisible_tp_declines() {
    let model = ModelArch {
        num_kv_heads: Some(8),
        head_dim: Some(128),
        num_layers: Some(32),
        ..Default::default()
    };
    let headroom_gb = 20.0;
    // 8 % 3 != 0: must not truncate to 2 heads/GPU.
    let tp3 = compute_kv_max_seqs_with_mode::<true>(
        Some(headroom_gb),
        Some(4096),
        &model,
        None,
        Some(3),
        None,
    )
    .max_seqs;
    assert!(tp3.is_none());

    // tp > heads is also non-divisible (2 % 4 != 0).
    let few_heads = ModelArch {
        num_kv_heads: Some(2),
        head_dim: Some(128),
        num_layers: Some(32),
        ..Default::default()
    };
    let tp4 = compute_kv_max_seqs_with_mode::<true>(
        Some(headroom_gb),
        Some(4096),
        &few_heads,
        None,
        Some(4),
        None,
    )
    .max_seqs;
    assert!(tp4.is_none());

    // Divisible case still prices: 2 heads / tp 2 → 1 head/GPU.
    let tp2 = compute_kv_max_seqs_with_mode::<true>(
        Some(headroom_gb),
        Some(4096),
        &few_heads,
        None,
        Some(2),
        None,
    )
    .max_seqs;
    assert!(tp2.is_some());
}

#[test]
fn compute_kv_max_seqs_declines_tp2_when_launch_flag_off() {
    let model = ModelArch {
        num_kv_heads: Some(8),
        head_dim: Some(128),
        num_layers: Some(32),
        ..Default::default()
    };
    let derived =
        compute_kv_max_seqs_with_mode::<false>(Some(20.0), Some(4096), &model, None, Some(2), None);
    assert_eq!(derived.max_seqs, None);
    let (bound, source, floor) =
        resolve_kv_bound(None, derived.max_seqs, false, Some(4.0), Some(50.0), None);
    assert_eq!(bound, None);
    assert_eq!(source, None);
    assert_eq!(floor, Some(8.0));
}

#[test]
fn compute_kv_max_seqs_zero_kv_heads_declines_under_tp() {
    let model = ModelArch {
        num_kv_heads: Some(0),
        head_dim: Some(128),
        num_layers: Some(32),
        ..Default::default()
    };
    let derived =
        compute_kv_max_seqs_with_mode::<true>(Some(20.0), Some(4096), &model, None, Some(2), None);
    assert_eq!(derived, DerivedCapacity::default());
}

#[test]
fn compute_kv_max_seqs_tp_none_uses_full_heads() {
    let model = ModelArch {
        num_kv_heads: Some(8),
        head_dim: Some(128),
        num_layers: Some(32),
        ..Default::default()
    };
    let headroom_gb = 20.0;
    let none = compute_kv_max_seqs_for_cache(
        Some(headroom_gb),
        Some(4096),
        &model,
        None,
        None,
        &crate::collectors::CacheConfigLabels::default(),
    )
    .max_seqs;
    let one = compute_kv_max_seqs_for_cache(
        Some(headroom_gb),
        Some(4096),
        &model,
        None,
        Some(1),
        &crate::collectors::CacheConfigLabels::default(),
    )
    .max_seqs;
    assert_eq!(none, one);
}

#[test]
fn compute_kv_max_seqs_auto_kv_uses_activation_dtype_not_weight_width() {
    let model = ModelArch {
        num_kv_heads: Some(8),
        head_dim: Some(128),
        num_layers: Some(32),
        ..Default::default()
    };
    let headroom_gb = 20.0;
    let bf16_weights = compute_kv_max_seqs_for_cache(
        Some(headroom_gb),
        Some(4096),
        &model,
        Some("auto"),
        None,
        &crate::collectors::CacheConfigLabels::default(),
    )
    .max_seqs;
    let quantized_weights = compute_kv_max_seqs_for_cache(
        Some(headroom_gb),
        Some(4096),
        &model,
        Some("auto"),
        None,
        &crate::collectors::CacheConfigLabels::default(),
    )
    .max_seqs;
    assert_eq!(quantized_weights, bf16_weights);
}

#[test]
fn effective_kv_dtype_baseline_capacity_and_r2_agree() {
    use crate::engine::baseline::{
        KvCacheDtypeSource, compute, effective_kv_cache_dtype, kv_bytes_per_element,
    };
    use crate::engine::rules::r2_kv_cache_pressure::fp8_kv_cache_fix_bullet;

    let model = ModelArch {
        num_kv_heads: Some(8),
        head_dim: Some(128),
        num_layers: Some(32),
        param_count: Some(7_000_000_000),
        default_weight_dtype: Some("bf16".to_string()),
        ..Default::default()
    };
    let headroom_gb = 20.0;

    // runtime fp8 + config None → 1 byte at capacity; no fp8-switch advice
    let runtime_fp8 = effective_kv_cache_dtype(Some("fp8"), None);
    assert_eq!(kv_bytes_per_element(runtime_fp8), 1);
    let cap_fp8 = compute_kv_max_seqs_for_cache(
        Some(headroom_gb),
        Some(4096),
        &model,
        runtime_fp8,
        None,
        &crate::collectors::CacheConfigLabels::default(),
    )
    .max_seqs;
    let cap_bf16 = compute_kv_max_seqs_for_cache(
        Some(headroom_gb),
        Some(4096),
        &model,
        Some("bf16"),
        None,
        &crate::collectors::CacheConfigLabels::default(),
    )
    .max_seqs;
    assert_eq!(cap_fp8.unwrap(), cap_bf16.unwrap() * 2);
    assert!(
        fp8_kv_cache_fix_bullet(runtime_fp8, true).is_none(),
        "already-fp8 runtime must not advise switching to fp8"
    );

    // config fp8 + runtime None → 1 byte (fallback)
    assert_eq!(
        kv_bytes_per_element(effective_kv_cache_dtype(None, Some("fp8"))),
        1
    );

    // runtime bf16 + config fp8 → runtime wins, 2 bytes
    assert_eq!(
        kv_bytes_per_element(effective_kv_cache_dtype(Some("bf16"), Some("fp8"))),
        2
    );

    // both None → Auto, 2 bytes
    let both_none = effective_kv_cache_dtype(None, None);
    assert_eq!(kv_bytes_per_element(both_none), 2);

    // Seam: baseline and R2 price the same snapshot's runtime fp8 the same way.
    let t = SystemTime::UNIX_EPOCH;
    let snap = RawSnapshot {
        gpu_observed_at: t,
        vllm_observed_at: t,
        timestamp: t,
        vllm: VllmRawMetrics {
            model_name: Some("meta-llama/Llama-3.1-8B-Instruct".to_string()),
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(4.0),
            window_duration_secs: Some(2.0),
            cache_config: crate::collectors::CacheConfigLabels {
                cache_dtype: Some("fp8".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        gpus: vec![GpuRawMetrics {
            gpu_name: Some("NVIDIA A100-SXM4-80GB".to_string()),
            vram_total_mb: Some(80 * 1024),
            ..Default::default()
        }],
    };
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        kv_cache_dtype: None,
        max_model_len: Some(4096),
        ..Default::default()
    };
    let ctx = StaticContext::from_snapshot(&snap, cfg);
    let win = RuntimeWindow::from_snapshot(snap);
    let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
    assert_eq!(b.kv_bytes_per_element, 1);
    assert_eq!(b.kv_cache_dtype_source, KvCacheDtypeSource::ExplicitFp8);
    let runtime = win.snapshot.vllm.cache_config.cache_dtype.as_deref();
    assert_eq!(kv_bytes_per_element(runtime), b.kv_bytes_per_element);
    assert!(fp8_kv_cache_fix_bullet(runtime, true).is_none());
}

#[test]
fn unknown_kv_dtype_request_floor_fires_hedged_weights_overflow_unaffected() {
    use crate::engine::baseline::{KvCacheDtypeSource, compute};

    let t = SystemTime::UNIX_EPOCH;
    // Tiny weights + huge KV geometry → positive headroom but one request won't fit.
    let snap = RawSnapshot {
        gpu_observed_at: t,
        vllm_observed_at: t,
        timestamp: t,
        vllm: VllmRawMetrics {
            model_name: Some("meta-llama/Llama-3.1-8B-Instruct".to_string()),
            generation_tokens_per_sec: Some(50.0),
            num_requests_running: Some(4.0),
            window_duration_secs: Some(2.0),
            cache_config: crate::collectors::CacheConfigLabels {
                cache_dtype: Some("mystery_dtype".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        gpus: vec![GpuRawMetrics {
            gpu_name: Some("NVIDIA A100-SXM4-80GB".to_string()),
            vram_total_mb: Some(80 * 1024),
            ..Default::default()
        }],
    };
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        kv_cache_dtype: None,
        max_model_len: Some(131072),
        gpu_memory_utilization: Some(0.9),
        ..Default::default()
    };
    let mut ctx = StaticContext::from_snapshot(&snap, cfg);
    ctx.model.param_count = Some(1_000_000_000);
    ctx.model.num_layers = Some(128);
    ctx.model.num_kv_heads = Some(64);
    ctx.model.head_dim = Some(256);
    ctx.model.default_weight_dtype = Some("bf16".to_string());
    let win = RuntimeWindow::from_snapshot(snap);
    let b = compute(&AnalysisInput::new(&ctx, &win)).expect("baseline");
    assert_eq!(b.kv_cache_dtype_source, KvCacheDtypeSource::Unknown);
    assert!(b.kv_headroom_gb.expect("headroom") > 0.0);

    let windows: Vec<_> = (0..ENGINE_MIN_PERSISTENT_WINDOWS)
        .map(|_| win.clone())
        .collect();
    let summary = ai(&ctx, &windows[0]);
    let report = build_report_for_windows(&windows, summary);
    let floor = report
        .recommendations
        .iter()
        .find(|r| r.rule_name == rule_names::OOM_RISK)
        .expect("Unknown KV dtype still fires one-request floor");
    let text = floor.display_lines.join("\n");
    assert!(text.contains("(est)"));
    assert!(text.contains("KV cache dtype unrecognized; sized at 2 bytes/element."));
    assert!(text.contains("Confidence: Low"));
    assert!(text.contains("cannot hold a single request"));

    // Explicit bf16: same floor, no provenance hedge, High confidence.
    let snap_known = RawSnapshot {
        vllm: VllmRawMetrics {
            cache_config: crate::collectors::CacheConfigLabels {
                cache_dtype: Some("bf16".to_string()),
                ..Default::default()
            },
            ..windows[0].snapshot.vllm.clone()
        },
        ..windows[0].snapshot.clone()
    };
    let win_known = RuntimeWindow::from_snapshot(snap_known);
    let windows_known: Vec<_> = (0..ENGINE_MIN_PERSISTENT_WINDOWS)
        .map(|_| win_known.clone())
        .collect();
    let summary_known = ai(&ctx, &windows_known[0]);
    let report_known = build_report_for_windows(&windows_known, summary_known);
    let known = report_known
        .recommendations
        .iter()
        .find(|r| r.rule_name == rule_names::OOM_RISK)
        .expect("explicit bf16 fires floor");
    let known_text = known.display_lines.join("\n");
    assert!(known_text.contains("cannot hold a single request"));
    assert!(!known_text.contains("KV cache dtype unrecognized"));
    assert!(known_text.contains("Confidence: High"));

    // Weights overflow path ignores KV Unknown: oversized weights still fire.
    let mut ctx_overflow = ctx.clone();
    ctx_overflow.model.param_count = Some(200_000_000_000);
    let report_overflow = build_report_for_windows(&windows, ai(&ctx_overflow, &windows[0]));
    assert!(
        report_overflow.recommendations.iter().any(|r| {
            r.rule_name == rule_names::OOM_RISK
                && r.display_lines
                    .iter()
                    .any(|l| l.contains("Model weights exceed GPU VRAM"))
                && !r
                    .display_lines
                    .iter()
                    .any(|l| l.contains("KV cache dtype unrecognized"))
        }),
        "weights-overflow must fire unchanged when KV dtype is Unknown"
    );
}

#[test]
fn compute_kv_max_seqs_whiteboard_reduces_hybrid_capacity() {
    let hybrid = qwen36_hybrid_model_with_attention();
    let attention_only = ModelArch {
        linear_num_layers: None,
        linear_key_heads: None,
        linear_value_heads: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_conv_kernel_dim: None,
        state_dtype: None,
        ..hybrid.clone()
    };
    let with_state = compute_kv_max_seqs_for_cache(
        Some(2.0),
        Some(4096),
        &hybrid,
        None,
        None,
        &crate::collectors::CacheConfigLabels::default(),
    )
    .max_seqs;
    let without_state = compute_kv_max_seqs_for_cache(
        Some(2.0),
        Some(4096),
        &attention_only,
        None,
        None,
        &crate::collectors::CacheConfigLabels::default(),
    )
    .max_seqs;
    assert_eq!(with_state, Some(6));
    assert_eq!(without_state, Some(7));
}

#[test]
fn compute_kv_max_seqs_windowed_gemma_uses_capped_price() {
    let gemma = ModelArch {
        num_layers: Some(62),
        num_kv_heads: Some(16),
        head_dim: Some(128),
        swa_window: Some(1024),
        num_swa_layers: Some(52),
        ..Default::default()
    };
    let all_full = ModelArch {
        swa_window: None,
        num_swa_layers: None,
        ..gemma.clone()
    };
    assert_eq!(
        compute_kv_max_seqs_for_cache(
            Some(20.0),
            Some(8192),
            &gemma,
            None,
            None,
            &crate::collectors::CacheConfigLabels::default(),
        )
        .max_seqs,
        Some(18)
    );
    assert_eq!(
        compute_kv_max_seqs_for_cache(
            Some(20.0),
            Some(8192),
            &all_full,
            None,
            None,
            &crate::collectors::CacheConfigLabels::default(),
        )
        .max_seqs,
        Some(4)
    );
}

#[test]
fn capacity_budget_rungs_prefer_observed_and_refuse_windowed_blocks() {
    let dense = ModelArch {
        num_layers: Some(32),
        num_kv_heads: Some(8),
        head_dim: Some(128),
        ..Default::default()
    };
    let dense_cache = crate::collectors::CacheConfigLabels {
        num_gpu_blocks: Some(10_000),
        block_size: Some(16),
        ..Default::default()
    };
    let dense_result =
        compute_kv_max_seqs_for_cache(Some(20.0), Some(4096), &dense, None, None, &dense_cache);
    assert_eq!(dense_result.max_seqs, Some(39));

    let hybrid = qwen36_hybrid_model_with_attention();
    let hybrid_cache = crate::collectors::CacheConfigLabels {
        num_gpu_blocks: Some(390),
        mamba_page_size_padded: Some(25_690_112),
        ..Default::default()
    };
    let hybrid_result =
        compute_kv_max_seqs_for_cache(Some(20.0), Some(4096), &hybrid, None, None, &hybrid_cache);
    assert_eq!(hybrid_result.max_seqs, Some(30));

    let windowed = ModelArch {
        swa_window: Some(1024),
        num_swa_layers: Some(26),
        ..dense.clone()
    };
    let windowed_result =
        compute_kv_max_seqs_for_cache(Some(20.0), Some(4096), &windowed, None, None, &dense_cache);
    assert_eq!(windowed_result.max_seqs, Some(95));

    let no_labels = compute_kv_max_seqs_for_cache(
        Some(20.0),
        Some(4096),
        &dense,
        None,
        None,
        &crate::collectors::CacheConfigLabels::default(),
    );
    assert_eq!(no_labels.max_seqs, Some(37));

    let no_budget = compute_kv_max_seqs_for_cache(
        None,
        Some(4096),
        &dense,
        None,
        None,
        &crate::collectors::CacheConfigLabels::default(),
    );
    assert_eq!(no_budget.max_seqs, None);
}

#[test]
fn unpriced_currency_declines_to_empirical_low_confidence() {
    let unpriced = ModelArch {
        num_layers: Some(32),
        num_kv_heads: Some(8),
        head_dim: Some(128),
        linear_num_layers: Some(1),
        ..Default::default()
    };
    let derived = compute_kv_max_seqs_for_cache(
        Some(20.0),
        Some(4096),
        &unpriced,
        None,
        None,
        &crate::collectors::CacheConfigLabels::default(),
    )
    .max_seqs;
    assert_eq!(derived, None);
    let (bound, source, floor) = resolve_kv_bound(None, derived, true, Some(4.0), Some(50.0), None);
    let rec =
        recommended_seqs(None, bound, source, floor, Some(4), None).expect("empirical fallback");
    assert_eq!(bound, None);
    assert_eq!(source, None);
    assert!(floor.is_some());
    assert!(rec.empirical);
    assert!(rec.floor_capped);
}

#[test]
fn rule_is_significant_six_of_ten_windows_passes() {
    assert!(rule_is_significant(6, 10));
}

#[test]
fn rule_is_significant_three_of_fifteen_fails_density_gate() {
    assert!(!rule_is_significant(3, 15));
}

#[test]
fn rule_is_significant_four_of_fifteen_passes() {
    assert!(rule_is_significant(4, 15));
}

#[test]
fn rule_is_significant_zero_evaluable_windows_is_false() {
    assert!(!rule_is_significant(3, 0));
}

#[test]
fn model_len_suggestion_uses_p99_sum_when_count_sufficient() {
    let mut lines = Vec::new();
    extend_with_shrink_suggestion(
        &mut lines,
        model_len_shrink_suggestion_lines(
            Some(8192),
            &ShrinkEvidence {
                prompt_p99: Some(6000.0),
                generation_p99: Some(450.0),
                prompt_mean: None,
                generation_mean: None,
                total_count: 150.0,
            },
            "      ",
            false,
        ),
    );
    let text = lines
        .iter()
        .map(|(b, _)| b.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("Lower --max-model-len 8192 → 6450"));
    assert!(text.contains("Observed p99 6.5k tokens per request (prompt + generation p99)"));
    assert!(!text.contains("prompt p99"));
    assert_eq!(lines[0].1, Some(SHRINK_REJECTION_WARNING));
}

#[test]
fn shrink_rejection_warning_on_all_four_forms() {
    const WARN: &str = SHRINK_REJECTION_WARNING;

    let named = model_len_shrink_suggestion_lines(
        Some(262144),
        &ShrinkEvidence {
            prompt_p99: Some(17000.0),
            generation_p99: Some(1500.0),
            prompt_mean: None,
            generation_mean: None,
            total_count: 150.0,
        },
        "      ",
        false,
    );
    assert_eq!(named.subline, Some(WARN));
    assert!(named.lines[0].contains("262144 → 18500"));
    assert!(
        named.lines[0].contains("Observed p99 18.5k tokens per request (prompt + generation p99)")
    );

    let both_halves = model_len_shrink_suggestion_lines(
        Some(262144),
        &ShrinkEvidence {
            prompt_p99: None,
            generation_p99: None,
            prompt_mean: Some(1100.0),
            generation_mean: Some(4000.0),
            total_count: 48.0,
        },
        "      ",
        false,
    );
    assert_eq!(both_halves.subline, Some(WARN));
    assert!(
        both_halves.lines[0].contains("Observed 5.1k tokens per request, prompt plus generation.")
    );

    let single_half = model_len_shrink_suggestion_lines(
        Some(262144),
        &ShrinkEvidence {
            prompt_p99: None,
            generation_p99: None,
            prompt_mean: Some(1100.0),
            generation_mean: None,
            total_count: 48.0,
        },
        "      ",
        false,
    );
    assert_eq!(single_half.subline, Some(WARN));
    assert!(single_half.lines[0].contains("Observed prompt 1.1k tokens per request."));

    let no_max = model_len_shrink_suggestion_lines(
        None,
        &ShrinkEvidence {
            prompt_p99: None,
            generation_p99: None,
            prompt_mean: None,
            generation_mean: None,
            total_count: 0.0,
        },
        "      ",
        false,
    );
    assert_eq!(no_max.subline, Some(WARN));
    assert!(no_max.lines[0].contains("to safely raise concurrency."));
}

#[test]
fn model_len_suggestion_no_op_when_count_below_threshold() {
    let mut lines = Vec::new();
    extend_with_shrink_suggestion(
        &mut lines,
        model_len_shrink_suggestion_lines(
            Some(8192),
            &ShrinkEvidence {
                prompt_p99: Some(6000.0),
                generation_p99: Some(450.0),
                prompt_mean: None,
                generation_mean: None,
                total_count: 50.0,
            },
            "      ",
            false,
        ),
    );
    let text = lines
        .iter()
        .map(|(b, _)| b.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("to safely raise concurrency"));
    assert!(!text.contains('→'));
}

#[test]
fn model_len_suggestion_missing_generation_p99_has_rejection_warning() {
    let mut lines = Vec::new();
    extend_with_shrink_suggestion(
        &mut lines,
        model_len_shrink_suggestion_lines(
            Some(8192),
            &ShrinkEvidence {
                prompt_p99: Some(6000.0),
                generation_p99: None,
                prompt_mean: None,
                generation_mean: None,
                total_count: 150.0,
            },
            "      ",
            false,
        ),
    );
    let text = lines
        .iter()
        .map(|(b, sub)| sub.map_or_else(|| b.clone(), |s| format!("{b}\n        {s}")))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("to safely raise concurrency"));
    assert!(!text.contains('→'));
    assert!(text.contains(SHRINK_REJECTION_WARNING));
    assert_eq!(lines[0].1, Some(SHRINK_REJECTION_WARNING));
}

#[test]
fn model_len_suggestion_missing_prompt_p99_has_rejection_warning() {
    let mut lines = Vec::new();
    extend_with_shrink_suggestion(
        &mut lines,
        model_len_shrink_suggestion_lines(
            Some(8192),
            &ShrinkEvidence {
                prompt_p99: None,
                generation_p99: Some(450.0),
                prompt_mean: None,
                generation_mean: None,
                total_count: 150.0,
            },
            "      ",
            false,
        ),
    );
    let text = lines
        .iter()
        .map(|(b, sub)| sub.map_or_else(|| b.clone(), |s| format!("{b}\n        {s}")))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("to safely raise concurrency"));
    assert!(!text.contains('→'));
    assert!(text.contains(SHRINK_REJECTION_WARNING));
    assert_eq!(lines[0].1, Some(SHRINK_REJECTION_WARNING));
}

#[test]
fn shrink_rejection_warning_on_all_nonempty_return_paths() {
    const WARN: &str = SHRINK_REJECTION_WARNING;

    let paths = [
        model_len_shrink_suggestion_lines(
            None,
            &ShrinkEvidence {
                prompt_p99: None,
                generation_p99: None,
                prompt_mean: None,
                generation_mean: None,
                total_count: 0.0,
            },
            "      ",
            false,
        ),
        model_len_shrink_suggestion_lines(
            Some(8192),
            &ShrinkEvidence {
                prompt_p99: Some(6000.0),
                generation_p99: None,
                prompt_mean: None,
                generation_mean: None,
                total_count: 150.0,
            },
            "      ",
            false,
        ),
        model_len_shrink_suggestion_lines(
            Some(8192),
            &ShrinkEvidence {
                prompt_p99: None,
                generation_p99: Some(450.0),
                prompt_mean: None,
                generation_mean: None,
                total_count: 150.0,
            },
            "      ",
            false,
        ),
        model_len_shrink_suggestion_lines(
            Some(8192),
            &ShrinkEvidence {
                prompt_p99: Some(6000.0),
                generation_p99: Some(450.0),
                prompt_mean: None,
                generation_mean: None,
                total_count: 150.0,
            },
            "      ",
            false,
        ),
        model_len_shrink_suggestion_lines(
            Some(262144),
            &ShrinkEvidence {
                prompt_p99: None,
                generation_p99: None,
                prompt_mean: Some(1100.0),
                generation_mean: Some(4000.0),
                total_count: 48.0,
            },
            "      ",
            false,
        ),
    ];
    for suggestion in paths {
        assert!(
            !suggestion.lines.is_empty(),
            "expected non-empty shrink path"
        );
        assert_eq!(suggestion.subline, Some(WARN));
    }

    let noop = model_len_shrink_suggestion_lines(
        Some(5464),
        &ShrinkEvidence {
            prompt_p99: Some(5400.0),
            generation_p99: Some(65.0),
            prompt_mean: None,
            generation_mean: None,
            total_count: 150.0,
        },
        "      ",
        false,
    );
    assert!(noop.lines.is_empty());
    assert_eq!(noop.subline, None);
}

#[test]
fn model_len_suggestion_no_op_when_p99_missing() {
    model_len_suggestion_missing_generation_p99_has_rejection_warning();
}

#[test]
fn model_len_suggestion_suppressed_when_delta_below_5pct() {
    let mut lines = Vec::new();
    extend_with_shrink_suggestion(
        &mut lines,
        model_len_shrink_suggestion_lines(
            Some(5464),
            &ShrinkEvidence {
                prompt_p99: Some(5400.0),
                generation_p99: Some(65.0),
                prompt_mean: None,
                generation_mean: None,
                total_count: 150.0,
            },
            "      ",
            false,
        ),
    );
    assert!(lines.is_empty());
}

#[test]
fn model_len_suggestion_projects_capacity_from_observed_geometry() {
    // Source: H100 ladder 2026-07-17 — 390 blocks, mamba_block_size 784,
    // obs 8.667 @ 32768. Suggested 16384 → floor(16.25)=16 concurrent (est).
    let mut lines = Vec::new();
    extend_with_shrink_suggestion(
        &mut lines,
        model_len_shrink_suggestion_lines(
            Some(32768),
            &ShrinkEvidence {
                prompt_p99: Some(15000.0),
                generation_p99: Some(1384.0),
                prompt_mean: None,
                generation_mean: None,
                total_count: 150.0,
            },
            "      ",
            false,
        ),
    );
    let text = lines
        .iter()
        .map(|(b, _)| b.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        text.contains("Lower --max-model-len 32768 → 16384"),
        "got: {text}"
    );
    assert!(!text.contains("fits at least"), "got: {text}");
    assert!(!text.contains("worst-case"));
    assert!(
        !text.contains("fits 8 concurrent"),
        "must not use current observed"
    );
}

#[test]
fn model_len_suggestion_live_run_projection_at_5465_is_39_not_observed_8() {
    // Source: live pressure run 2026-07-17 05:53 UTC — 390 blocks / 784
    // block_size / state_pages 3. At suggested 5465:
    // 390 ÷ (ceil(5465/784) + 3) = 390 ÷ 10 = 39. Never current observed (8).
    let cache = crate::collectors::CacheConfigLabels {
        block_size: Some(16),
        num_gpu_blocks: Some(390),
        mamba_block_size: Some(784),
        kv_cache_max_concurrency: Some(8.667),
        ..Default::default()
    };
    let hyp = HypCapacityCtx {
        cache: &cache,
        kv_headroom_gb: None,
        model: None,
        kv_cache_dtype: None,
        tp: None,
    };
    assert_eq!(
        capacity_at_hypothetical_max_len(5465, Some(32768), &hyp),
        Some(39)
    );
    let mut lines = Vec::new();
    extend_with_shrink_suggestion(
        &mut lines,
        model_len_shrink_suggestion_lines(
            Some(32768),
            &ShrinkEvidence {
                prompt_p99: Some(5000.0),
                generation_p99: Some(465.0),
                prompt_mean: None,
                generation_mean: None,
                total_count: 150.0,
            },
            "      ",
            false,
        ),
    );
    let text = lines
        .iter()
        .map(|(b, _)| b.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        text.contains("Lower --max-model-len 32768 → 5465"),
        "got: {text}"
    );
    assert!(!text.contains("fits at least"), "got: {text}");
    assert!(!text.contains("worst-case"));
    assert!(
        !text.contains("fits 8 concurrent"),
        "suffix must not be current observed concurrency"
    );
}

#[test]
fn operator_text_has_no_capacity_numbers_in_r1_r2_r5_r7() {
    use super::r5_concurrency_saturation::{
        ConcurrencySaturationDetail, format_concurrency_saturation_issue, r5_recommendation,
    };
    use super::r7_config_headroom::{ConfigHeadroomDetail, format_config_headroom_window_issue};
    use super::{BindingWall, KvBoundSource, RecommendedSeqs, recommended_seqs};

    let assert_clean = |label: &str, text: &str| {
        for term in ["vLLM-reported", "worst-case requests"] {
            assert!(
                !text.contains(term),
                "{label}: forbidden {term} in:\n{text}"
            );
        }
        for line in text.lines() {
            if line.contains("Batch more requests") || line.contains("Raise --max-num-seqs") {
                assert!(
                    !line.contains('≤'),
                    "{label}: fix line must not contain ≤ capacity: {line}"
                );
            }
        }
    };

    let t = SystemTime::UNIX_EPOCH;
    let r1_win = mk_r1_window_with_kv(50.0);
    let r1 = r1_recommendation(r1_test_input(&r1_win.snapshot)).expect("r1");
    assert_clean("r1", &r1.display_lines.join("\n"));

    let mut v = vllm_high_kv();
    v.num_requests_waiting = Some(5.0);
    v.prompt_tokens_p99 = Some(6000.0);
    v.generation_tokens_p99 = Some(450.0);
    v.generation_tokens_completed = Some(150.0);
    let s = snap(t, t, v, gpu_low());
    let r2 = r2_recommendation(R2RecommendationInput {
        snapshot: &s,
        max_model_len: Some(8192),
        kv_headroom_gb: None,
        kv_max_seqs: Some(24),
        capacity_label: KvCapacityLabel::Observed,
        windows_fired: 1,
        total_evaluable: 1,
        fp8_compiler_available: false,
    })
    .expect("r2");
    assert_clean("r2", &r2.display_lines.join("\n"));

    let r5_rec = recommended_seqs(
        Some(153.0),
        Some(120.0),
        Some(KvBoundSource::Observed),
        None,
        Some(32),
        None,
    )
    .expect("r5 rec");
    let r5_raise = r5_recommendation(
        &snap(
            t,
            t,
            VllmRawMetrics {
                num_requests_running: Some(32.0),
                num_requests_waiting: Some(15.0),
                max_num_seqs: Some(32),
                kv_cache_usage_perc: Some(70.0),
                generation_tokens_per_sec: Some(100.0),
                ..Default::default()
            },
            gpu_low(),
        ),
        Some(70.0),
        None,
        None,
        Some(r5_rec),
    )
    .expect("r5 raise");
    assert_clean("r5 raise", &r5_raise.display_lines.join("\n"));

    let at_wall_rec = recommended_seqs(
        None,
        Some(15.0),
        Some(KvBoundSource::Derived),
        None,
        Some(15),
        None,
    )
    .expect("r5 at wall");
    let r5_wall = format_concurrency_saturation_issue(
        &ConcurrencySaturationDetail {
            requests_running: 15.0,
            requests_waiting: 10.0,
            max_num_seqs: Some(15),
            queue_ratio: 10.0 / 25.0,
            ttft_ms: None,
            ttft_p99_ms: None,
            ttft_p99_clamped: false,
            ttft_p99_buckets: vec![],
            kv_cache_usage_perc: Some(50.0),
        },
        Some(8192),
        Some(&at_wall_rec),
        &snap(t, t, VllmRawMetrics::default(), gpu_low()),
    )
    .join("\n");
    assert_clean("r5 at wall", &r5_wall);

    for (label, src) in [
        ("r7 observed", KvBoundSource::Observed),
        ("r7 derived", KvBoundSource::Derived),
    ] {
        let rec = RecommendedSeqs {
            target: 96,
            wall: 120.0,
            binder: BindingWall::Memory { cap: 120 },
            source: Some(src),
            empirical: false,
            floor_capped: false,
            wall_is_capacity: true,
        };
        let d = ConfigHeadroomDetail {
            max_num_seqs: 32,
            recommended_seqs: 96,
            ridge_batch_size: Some(153.0),
            kv_cache_usage_perc: None,
            occupancy_pct: 62.5,
            running: 20.0,
        };
        let r7 = format_config_headroom_window_issue(&d, 100, 0.8, Some(&rec), true).join("\n");
        assert_clean(label, &r7);
    }
}

#[test]
fn capacity_at_hypothetical_falls_to_catalog_when_labels_absent() {
    // All labels absent: older vLLM with no cache_config_info scrape.
    let cache = crate::collectors::CacheConfigLabels::default();
    let model = ModelArch {
        num_kv_heads: Some(8),
        head_dim: Some(128),
        num_layers: Some(32),
        ..Default::default()
    };
    let catalog = compute_kv_max_seqs_for_cache(
        Some(20.0),
        Some(4096),
        &model,
        None,
        None,
        &crate::collectors::CacheConfigLabels::default(),
    )
    .max_seqs;
    assert!(catalog.is_some());
    let hyp = HypCapacityCtx {
        cache: &cache,
        kv_headroom_gb: Some(20.0),
        model: Some(&model),
        kv_cache_dtype: None,
        tp: None,
    };
    assert_eq!(
        capacity_at_hypothetical_max_len(4096, Some(8192), &hyp),
        catalog
    );
}

#[test]
fn capacity_at_hypothetical_falls_to_catalog_when_num_gpu_blocks_absent() {
    // Incomplete scrape: other labels present, num_gpu_blocks missing.
    // Must route to tier 2 without consulting the gate.
    let cache = crate::collectors::CacheConfigLabels {
        block_size: Some(16),
        kv_cache_max_concurrency: Some(10.0),
        ..Default::default()
    };
    let model = ModelArch {
        num_kv_heads: Some(8),
        head_dim: Some(128),
        num_layers: Some(32),
        ..Default::default()
    };
    let catalog = compute_kv_max_seqs_for_cache(
        Some(20.0),
        Some(4096),
        &model,
        None,
        None,
        &crate::collectors::CacheConfigLabels::default(),
    )
    .max_seqs;
    assert!(catalog.is_some());
    let hyp = HypCapacityCtx {
        cache: &cache,
        kv_headroom_gb: Some(20.0),
        model: Some(&model),
        kv_cache_dtype: None,
        tp: None,
    };
    assert_eq!(
        capacity_at_hypothetical_max_len(4096, Some(8192), &hyp),
        catalog
    );
}

#[test]
fn capacity_at_hypothetical_gate_suppresses_both_tiers() {
    let cache = crate::collectors::CacheConfigLabels {
        block_size: Some(16),
        num_gpu_blocks: Some(2000),
        // round(2000 / obs) = 2374 pages; transcript = 512; state = 1862.
        kv_cache_max_concurrency: Some(2000.0 / 2374.0),
        ..Default::default()
    };
    let model = ModelArch {
        num_kv_heads: Some(8),
        head_dim: Some(128),
        num_layers: Some(32),
        ..Default::default()
    };
    let hyp = HypCapacityCtx {
        cache: &cache,
        kv_headroom_gb: Some(20.0),
        model: Some(&model),
        kv_cache_dtype: None,
        tp: None,
    };
    assert!(
        compute_kv_max_seqs_for_cache(
            Some(20.0),
            Some(4096),
            &model,
            None,
            None,
            &crate::collectors::CacheConfigLabels::default(),
        )
        .max_seqs
        .is_some(),
        "catalog tier must be viable so the test proves it was suppressed"
    );
    assert_eq!(
        capacity_at_hypothetical_max_len(4096, Some(8192), &hyp),
        None
    );

    let text = model_len_shrink_suggestion_lines(
        Some(8192),
        &ShrinkEvidence {
            prompt_p99: Some(3500.0),
            generation_p99: Some(596.0),
            prompt_mean: None,
            generation_mean: None,
            total_count: 150.0,
        },
        "      ",
        false,
    )
    .lines
    .join("\n");
    assert!(text.contains("Lower --max-model-len"), "got: {text}");
    assert!(!text.contains("; fits"), "got: {text}");
    assert!(!text.contains("concurrent requests (est)"), "got: {text}");
}

#[test]
fn capacity_at_hypothetical_fits_none_does_not_fall_to_catalog() {
    let cache = crate::collectors::CacheConfigLabels {
        block_size: Some(16),
        num_gpu_blocks: Some(100),
        kv_cache_max_concurrency: Some(1.0),
        ..Default::default()
    };
    let model = ModelArch {
        num_kv_heads: Some(8),
        head_dim: Some(128),
        num_layers: Some(32),
        ..Default::default()
    };
    let hyp = HypCapacityCtx {
        cache: &cache,
        kv_headroom_gb: Some(20.0),
        model: Some(&model),
        kv_cache_dtype: None,
        tp: None,
    };
    assert!(
        compute_kv_max_seqs_for_cache(
            Some(20.0),
            Some(2048),
            &model,
            None,
            None,
            &crate::collectors::CacheConfigLabels::default(),
        )
        .max_seqs
        .is_some(),
        "catalog tier must be viable so the test proves geometry overruled it"
    );
    assert!(
        crate::engine::baseline::counterfactual_concurrency(2048, 16, 100, 1.0, 1600)
            .is_some_and(|c| (c - 100.0 / 128.0).abs() < f64::EPSILON)
    );
    assert_eq!(
        capacity_at_hypothetical_max_len(2048, Some(1600), &hyp),
        None
    );
}

#[test]
fn capacity_at_hypothetical_dense_zero_state_projects_as_before() {
    let cache = crate::collectors::CacheConfigLabels {
        block_size: Some(16),
        num_gpu_blocks: Some(2560),
        kv_cache_max_concurrency: Some(10.0),
        ..Default::default()
    };
    let hyp = HypCapacityCtx {
        cache: &cache,
        kv_headroom_gb: None,
        model: None,
        kv_cache_dtype: None,
        tp: None,
    };
    assert_eq!(
        capacity_at_hypothetical_max_len(2048, Some(4096), &hyp),
        Some(20)
    );
}

#[test]
fn capacity_at_hypothetical_prefers_geometry_over_catalog() {
    // Geometry predicts floor(16.25)=16; catalog with huge headroom would differ.
    let cache = crate::collectors::CacheConfigLabels {
        block_size: Some(16),
        num_gpu_blocks: Some(390),
        mamba_block_size: Some(784),
        kv_cache_max_concurrency: Some(8.667),
        ..Default::default()
    };
    let model = ModelArch {
        num_kv_heads: Some(8),
        head_dim: Some(128),
        num_layers: Some(32),
        ..Default::default()
    };
    let hyp = HypCapacityCtx {
        cache: &cache,
        kv_headroom_gb: Some(80.0),
        model: Some(&model),
        kv_cache_dtype: None,
        tp: None,
    };
    assert_eq!(
        capacity_at_hypothetical_max_len(16384, Some(32768), &hyp),
        Some(16)
    );
}

fn qwen36_hybrid_model() -> ModelArch {
    ModelArch {
        linear_num_layers: Some(48),
        linear_key_heads: Some(16),
        linear_value_heads: Some(48),
        linear_key_head_dim: Some(128),
        linear_value_head_dim: Some(128),
        linear_conv_kernel_dim: Some(4),
        state_dtype: Some("fp32".to_string()),
        ..Default::default()
    }
}

fn qwen36_hybrid_model_with_attention() -> ModelArch {
    ModelArch {
        num_layers: Some(64),
        num_kv_layers: Some(16),
        num_kv_heads: Some(4),
        head_dim: Some(256),
        ..qwen36_hybrid_model()
    }
}

#[test]
fn catalog_state_mismatch_none_when_qwen36_agrees_with_ladder() {
    // Source: H100 ladder 2026-07-17 — observed state_pages=3; fixed formula agrees.
    let cache = crate::collectors::CacheConfigLabels {
        block_size: Some(16),
        num_gpu_blocks: Some(390),
        mamba_block_size: Some(784),
        mamba_page_size_padded: Some(25_690_112),
        kv_cache_max_concurrency: Some(8.667),
        ..Default::default()
    };
    let model = qwen36_hybrid_model();
    assert_eq!(
        catalog_state_pages_mismatch(&cache, Some(32768), &model),
        None
    );
}

#[test]
fn catalog_state_mismatch_when_synthetic_catalog_differs() {
    // Synthetic: inflate key heads so catalog pages diverge from ladder-observed 3.
    let cache = crate::collectors::CacheConfigLabels {
        block_size: Some(16),
        num_gpu_blocks: Some(390),
        mamba_block_size: Some(784),
        mamba_page_size_padded: Some(25_690_112),
        kv_cache_max_concurrency: Some(8.667),
        ..Default::default()
    };
    let mut model = qwen36_hybrid_model();
    model.linear_key_heads = Some(64);
    let mismatch = catalog_state_pages_mismatch(&cache, Some(32768), &model);
    assert!(mismatch.is_some());
    let (catalog_pages, observed_pages) = mismatch.unwrap();
    assert_eq!(observed_pages, 3);
    assert_ne!(catalog_pages, observed_pages);
}

#[test]
fn catalog_state_mismatch_none_when_labels_absent() {
    let cache = crate::collectors::CacheConfigLabels::default();
    assert!(catalog_state_pages_mismatch(&cache, Some(32768), &qwen36_hybrid_model()).is_none());
}

#[test]
fn catalog_state_mismatch_none_when_catalog_hybrid_absent() {
    let cache = crate::collectors::CacheConfigLabels {
        block_size: Some(16),
        num_gpu_blocks: Some(390),
        mamba_block_size: Some(784),
        mamba_page_size_padded: Some(25_690_112),
        kv_cache_max_concurrency: Some(8.667),
        ..Default::default()
    };
    let dense = ModelArch::default();
    assert!(catalog_state_pages_mismatch(&cache, Some(32768), &dense).is_none());
}

#[test]
fn under_batching_fires_when_gates_pass() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.tpot_ms = Some(35.0);
    let s = snap(t, t, v, gpu_low());
    let win = mk_win(s);
    let r = r1_recommendation(r1_test_input(&win.snapshot)).expect("r1 fired");
    assert_eq!(r.rule_name, rule_names::UNDER_BATCHING);
    match rule1_under_batching_with_efficiency(r1_test_input(&win.snapshot)) {
        Rule1Outcome::Fired(d) => assert!(d.occupancy_pct < 25.0),
        Rule1Outcome::NotFired => panic!("expected fired"),
    }
}

#[test]
fn r2_advisory_requires_active_traffic() {
    let url = "http://127.0.0.1:8000/metrics";
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.kv_cache_usage_perc = None;
    v.num_requests_running = None;
    let raw = snap(t, t, v.clone(), gpu_busy());
    assert!(r2_kv_cache_advisory(&raw, url).is_none());

    v.num_requests_running = Some(0.0);
    let raw = snap(t, t, v.clone(), gpu_busy());
    assert!(r2_kv_cache_advisory(&raw, url).is_none());

    v.num_requests_running = Some(3.0);
    let raw = snap(t, t, v, gpu_busy());
    let lines = r2_kv_cache_advisory(&raw, url).expect("r2 advisory");
    assert!(lines[0].contains("core metric unavailable"));
}

#[test]
fn format_under_batching_fired_matches_template() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.num_requests_running = Some(5.0);
    v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
    let mut g = gpu_low();
    g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
    let s = snap(t, t, v, g);
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let ctx = StaticContext::from_snapshot(&s, cfg);
    let win = mk_win(s);
    let text =
        format_diagnose_rules_test(&ctx, &win, false, "http://127.0.0.1:8000/metrics").join("\n");
    assert!(text.contains("[!] Under-batching: Insufficient Concurrency"));
    assert!(text.contains("Batch more requests or increase client concurrency (251 slots idle)"));
    assert!(!text.contains("/hr "));
}

#[test]
fn issue_blocks_r1_through_r7_have_no_waste_per_hr() {
    let url = "http://127.0.0.1:8000/metrics";
    let assert_no_waste = |label: &str, text: &str| {
        assert!(
            !text.contains("/hr "),
            "{label} still has $/hr waste line: {text}"
        );
        for fragment in [
            "wasted on idle compute",
            "lost to memory thrashing",
            "wasted on redundant prefill",
            "lost to prefill interference",
            "lost to scheduler queuing",
            "wasted on config-limited batching",
            "lost to compounding bottlenecks",
            "unclassified overhead",
        ] {
            assert!(
                !text.contains(fragment),
                "{label} still has waste label `{fragment}`: {text}"
            );
        }
    };

    // R1
    {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(5.0);
        v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
        v.generation_tokens_per_sec = Some(100.0);
        let mut g = gpu_low();
        g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
        g.power_watts = Some(400.0);
        g.aligned_power_watts = Some(400.0);
        let s = snap(t, t, v, g);
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let ctx = StaticContext::from_snapshot(&s, cfg);
        let win = mk_win(s);
        let text = format_diagnose_rules_test(&ctx, &win, false, url).join("\n");
        assert!(text.contains("[!] Under-batching"));
        assert_no_waste("R1", &text);
    }

    // R2
    {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        for w in windows.iter_mut().take(4) {
            *w = mk_evaluable_kv_window(89.0, true);
            w.snapshot.gpus[0].gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
            w.snapshot.gpus[0].power_watts = Some(400.0);
            w.snapshot.gpus[0].aligned_power_watts = Some(400.0);
        }
        let text = r2_issue_lines(&windows).join("\n");
        assert!(text.contains("[!] KV Cache Pressure"));
        assert_no_waste("R2", &text);
    }

    // R3 (raise occupancy so R1 does not win the layer filter)
    {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.prefix_cache_hit_rate = Some(0.20);
        v.prompt_tokens_mean = Some(25.0);
        v.request_success_per_sec = Some(40.0);
        v.num_requests_running = Some(200.0);
        v.num_requests_waiting = Some(0.0);
        v.max_num_seqs = Some(256);
        v.generation_tokens_per_sec = Some(80.0);
        v.kv_cache_usage_perc = Some(40.0);
        let mut g = gpu_busy();
        g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
        g.power_watts = Some(400.0);
        g.aligned_power_watts = Some(400.0);
        let s = snap(t, t, v, g);
        let ctx = mk_ctx();
        let win = mk_win(s);
        let text = format_diagnose_rules_test(&ctx, &win, false, url).join("\n");
        assert!(
            text.contains("[!] Low Prefix"),
            "expected R3 primary: {text}"
        );
        assert_no_waste("R3", &text);
    }

    // R4
    {
        let (ctx, win) = input_r4_suppresses_r2();
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        for w in windows.iter_mut().take(8) {
            *w = win.clone();
        }
        let summary = ai(&ctx, windows.last().expect("windows"));
        let text = format_diagnose_rules_for_windows_test(&windows, summary, false, url).join("\n");
        assert!(text.contains("[!] OOM Risk") || text.contains("OOM"));
        assert_no_waste("R4", &text);
    }

    // R5
    {
        let mut windows: Vec<_> = (0..15)
            .map(|_| mk_evaluable_kv_window(50.0, false))
            .collect();
        for w in windows.iter_mut().take(4) {
            *w = mk_evaluable_concurrency_saturation_window(32.0, 15.0, 32);
            w.snapshot.vllm.kv_cache_usage_perc = Some(70.0);
        }
        let ctx = mk_ctx();
        let summary = ai(&ctx, windows.last().expect("windows"));
        let text = format_diagnose_rules_for_windows_test(&windows, summary, false, url).join("\n");
        assert!(text.contains("[!] Concurrency Saturation"));
        assert_no_waste("R5", &text);
    }

    // R6
    {
        let windows: Vec<_> = (0..10)
            .map(|_| mk_r6_prefill_window(12.0, 10.0, 5.0, Some(50.0)))
            .collect();
        let ctx = mk_llama8b_h100_ctx(&windows[0].snapshot);
        let summary = ai(&ctx, windows.last().expect("windows"));
        let text = format_diagnose_rules_for_windows_test(&windows, summary, false, url).join("\n");
        assert!(text.contains("[!] Prefill-Bound") || text.contains("Prefill"));
        assert_no_waste("R6", &text);
    }

    // R7
    {
        let windows: Vec<_> = (0..10)
            .map(|_| mk_r7_headroom_window(60.0, 64, 0.0, 50.0))
            .collect();
        let ctx = mk_r7_ctx(64);
        let summary = ai(&ctx, windows.last().expect("windows"));
        let text = format_diagnose_rules_for_windows_test(&windows, summary, false, url).join("\n");
        assert!(text.contains("Configured Batch Limit") || text.contains("config"));
        assert_no_waste("R7", &text);
    }
}

#[test]
fn format_diagnose_rules_for_windows_r4_suppresses_r2_when_both_significant() {
    let (ctx, _) = input_r4_suppresses_r2();
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_kv_window(50.0, false))
        .collect();
    for w in windows.iter_mut().take(8) {
        *w = mk_evaluable_kv_window(89.0, true);
    }
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    assert_eq!(report.recommendations[0].rule_name, rule_names::OOM_RISK);
    assert!(
        report
            .suppressed_rules
            .iter()
            .any(|(s, _)| *s == rule_names::KV_CACHE_PRESSURE)
    );
    assert!(
        report
            .suppressed_recs
            .iter()
            .any(|r| r.rule_name == rule_names::KV_CACHE_PRESSURE && !r.display_lines.is_empty()),
        "suppressed KV body must be retained for stuck-fix reveal"
    );
}

#[test]
fn format_diagnose_verbose_shows_kv_pressure_suppressed_when_r4_fires() {
    let (ctx, win) = input_r4_suppresses_r2();
    let text =
        format_diagnose_rules_test(&ctx, &win, true, "http://127.0.0.1:8000/metrics").join("\n");
    assert!(text.contains("KV Cache Pressure: suppressed by OOM Risk"));
    assert!(!text.contains("KV Cache Pressure: not triggered"));
}

#[test]
fn not_triggered_shows_plain_label() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
    v.prompt_tokens_per_sec = Some(600.0);
    v.generation_tokens_per_sec = Some(100.0);
    v.prefix_cache_hit_rate = None;
    let mut g = gpu_busy();
    g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
    let s = snap(t, t, v, g);
    let ctx = mk_llama8b_h100_ctx(&s);
    let win = mk_win(s);
    let text =
        format_diagnose_rules_test(&ctx, &win, true, "http://127.0.0.1:8000/metrics").join("\n");
    assert!(text.contains("Under-batching: not triggered"));
    assert!(!text.contains("Under-batching: not triggered ("));
}

#[test]
fn suppressed_rule_shows_suppressor_in_verbose() {
    // Mixed run: both R1 and R6 significant → ME puts R6 over R1.
    let mut windows: Vec<_> = (0..10)
        .map(|_| mk_r6_prefill_window(2.5, 10.0, 5.0, Some(50.0)))
        .collect();
    for w in windows.iter_mut().skip(5) {
        *w = mk_r6_prefill_window(12.0, 10.0, 5.0, Some(80.0));
    }
    let ctx = mk_llama8b_h100_ctx(&windows[0].snapshot);
    let summary = ai(&ctx, windows.last().expect("windows"));
    let text = format_diagnose_rules_for_windows_test(
        &windows,
        summary,
        true,
        "http://127.0.0.1:8000/metrics",
    )
    .join("\n");
    assert!(text.contains("Under-batching: suppressed by Prefill-Bound"));
}

#[test]
fn suppression_table_shows_suppressor_in_verbose() {
    let (ctx, _) = input_r4_suppresses_r2();
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_kv_window(50.0, false))
        .collect();
    for w in windows.iter_mut().take(8) {
        *w = mk_evaluable_kv_window(89.0, true);
    }
    let summary = ai(&ctx, windows.last().expect("windows"));
    let text = format_diagnose_rules_for_windows_test(
        &windows,
        summary,
        true,
        "http://127.0.0.1:8000/metrics",
    )
    .join("\n");
    assert!(text.contains("KV Cache Pressure: suppressed by OOM Risk"));
}

#[test]
fn format_diagnose_verbose_r1_shows_plain_not_triggered_when_gate_suppresses() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
    v.prompt_tokens_per_sec = Some(600.0);
    v.generation_tokens_per_sec = Some(100.0);
    v.prefix_cache_hit_rate = None;
    let mut g = gpu_busy();
    g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
    let s = snap(t, t, v, g);
    let ctx = mk_llama8b_h100_ctx(&s);
    let win = mk_win(s);
    let text =
        format_diagnose_rules_test(&ctx, &win, true, "http://127.0.0.1:8000/metrics").join("\n");
    assert!(text.contains("Under-batching: not triggered"));
    assert!(!text.contains("prompt/gen ratio"));
}

#[test]
fn format_diagnose_verbose_shows_not_indicated_when_no_issue() {
    let t = SystemTime::UNIX_EPOCH;
    let mut g = gpu_low();
    g.gpu_util_pct = Some(75.0);
    let mut v = vllm_base();
    v.num_requests_running = Some(64.0);
    let s = snap(t, t, v, g);
    let ctx = mk_ctx();
    let win = mk_win(s);
    let text =
        format_diagnose_rules_test(&ctx, &win, true, "http://127.0.0.1:8000/metrics").join("\n");
    assert!(text.contains("Under-batching: not triggered"));
    assert!(text.contains("KV Cache Pressure: not triggered"));
    assert!(text.contains("Low Prefix Reuse: not triggered"));
}

#[test]
fn format_diagnose_verbose_healthy_shows_not_triggered_and_limiter() {
    let t = SystemTime::UNIX_EPOCH;
    let mut g = gpu_low();
    g.gpu_util_pct = Some(75.0);
    g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
    g.vram_total_mb = Some(80 * 1024);
    let mut v = vllm_base();
    v.num_requests_running = Some(64.0);
    v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
    let s = snap(t, t, v, g);
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let ctx = StaticContext::from_snapshot(&s, cfg);
    let win = mk_win(s);
    let lines = format_diagnose_rules_test(&ctx, &win, true, "http://127.0.0.1:8000/metrics");
    let text = lines.join("\n");
    assert!(text.contains("Under-batching: not triggered"));
    assert!(text.contains("No issues detected."));
    assert!(
        text.contains("Capped by "),
        "verbose healthy must render limiter after not-triggered list: {text}"
    );
    let not_triggered = text
        .find("Under-batching: not triggered")
        .expect("not triggered list");
    let capped = text.find("Capped by ").expect("limiter line");
    assert!(
        not_triggered < capped,
        "not-triggered list must precede limiter verdict: {text}"
    );
}

#[test]
fn format_diagnose_quiet_healthy_renders_no_issues_and_limiter_only() {
    let t = SystemTime::UNIX_EPOCH;
    let mut g = gpu_low();
    g.gpu_util_pct = Some(75.0);
    g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
    g.vram_total_mb = Some(80 * 1024);
    let mut v = vllm_base();
    v.num_requests_running = Some(64.0);
    v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
    let s = snap(t, t, v, g);
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let ctx = StaticContext::from_snapshot(&s, cfg);
    let win = mk_win(s);
    let lines = format_diagnose_rules_test(&ctx, &win, false, "http://127.0.0.1:8000/metrics");
    assert_eq!(
        lines.first().map(String::as_str),
        Some("No issues detected.")
    );
    assert!(
        lines.get(1).is_some_and(|l| l.starts_with("Capped by ")),
        "expected limiter line, got: {lines:?}"
    );
    assert!(
        !lines.iter().any(|l| l.contains(": not triggered")),
        "quiet healthy must not list not-triggered rules: {lines:?}"
    );
}

#[test]
fn format_diagnose_verbose_advisory_suppresses_no_issues_and_limiter() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.num_requests_running = Some(64.0);
    v.kv_cache_usage_perc = None;
    let s = snap(t, t, v, gpu_busy());
    let ctx = mk_ctx();
    let win = mk_win(s);
    let text =
        format_diagnose_rules_test(&ctx, &win, true, "http://127.0.0.1:8000/metrics").join("\n");
    assert!(text.contains("[i] KV Cache Pressure: core metric unavailable"));
    assert!(!text.contains("No issues detected."));
    assert!(!text.contains("Capped by "));
}

#[test]
fn kv_cache_pressure_fires_despite_observation_skew() {
    // r2 is vLLM-only; GPU/vLLM clock divergence must not suppress KV pressure.
    let t0 = SystemTime::UNIX_EPOCH;
    let t1 = t0 + Duration::from_secs(5);
    let mut v = vllm_high_kv();
    v.num_requests_running = Some(64.0);
    v.num_requests_waiting = Some(5.0); // queue corroboration
    let s = snap(t0, t1, v, gpu_low());
    assert!(
        matches!(rule2_kv_cache_pressure(&s), Rule2Outcome::Fired(_)),
        "skew must not gate r2"
    );
}

#[test]
fn kv_cache_miss_unavailable_without_gauge_verbose() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.num_requests_running = Some(64.0);
    v.kv_cache_usage_perc = None;
    let s = snap(t, t, v, gpu_busy());
    let ctx = mk_ctx();
    let win = mk_win(s);
    let text =
        format_diagnose_rules_test(&ctx, &win, true, "http://127.0.0.1:8000/metrics").join("\n");
    assert!(text.contains("[i] KV Cache Pressure: core metric unavailable"));
    assert!(!text.contains("KV Cache Pressure: not triggered"));
}

#[test]
fn format_low_prefix_hit_rate_fired_matches_template() {
    let d = LowPrefixReuseDetail {
        hit_rate: Some(0.24),
        prompt_tokens_mean: Some(128.0),
        queries_delta: None,
    };
    let text = format_low_prefix_hit_rate_fired(&d, Some(true), None).join("\n");
    assert!(text.contains("[!] Low Prefix Cache"));
    assert!(text.contains("Prefix hit rate 24.0%"));
}

#[test]
fn format_diagnose_rules_no_fires_default_is_only_no_issues_line() {
    let t = SystemTime::UNIX_EPOCH;
    let mut g = gpu_low();
    g.gpu_util_pct = Some(75.0);
    g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
    g.vram_total_mb = Some(80 * 1024);
    let mut v = vllm_base();
    v.num_requests_running = Some(64.0);
    v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
    let s = snap(t, t, v, g);
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let ctx = StaticContext::from_snapshot(&s, cfg);
    let win = mk_win(s);
    let lines = format_diagnose_rules_test(&ctx, &win, false, "http://127.0.0.1:8000/metrics");
    assert_eq!(
        lines.first().map(String::as_str),
        Some("No issues detected.")
    );
    assert!(
        lines.get(1).is_some_and(|l| l.starts_with("Capped by ")),
        "expected limiter line, got: {lines:?}"
    );
}

#[test]
fn format_diagnose_rules_non_evaluable_snapshot_shows_note() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.num_requests_running = None;
    let s = snap(t, t, v, gpu_busy());
    let ctx = mk_ctx();
    let win = mk_win(s);
    let metrics_url = "http://127.0.0.1:8000/metrics";
    let windows: Vec<_> = (0..ENGINE_MIN_PERSISTENT_WINDOWS)
        .map(|_| win.clone())
        .collect();
    let lines = format_diagnose_rules_test(&ctx, &win, false, metrics_url);
    assert_eq!(
        lines,
        empty_run_diagnose_lines(
            false,
            &windows,
            false,
            &hint_for_empty(&ctx, &win, metrics_url, 30),
            metrics_url
        )
    );
}

#[test]
fn r2_confidence_equals_duration_density() {
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_kv_window(50.0, false))
        .collect();
    for w in windows.iter_mut().take(4) {
        *w = mk_evaluable_kv_window(89.0, true);
    }
    let ctx = mk_ctx();
    let mut summary_win = windows.last().expect("windows").clone();
    summary_win.snapshot.vllm.kv_cache_usage_perc = Some(72.5);
    summary_win.snapshot.vllm.kv_cache_peak_perc = Some(99.4);
    let summary = ai(&ctx, &summary_win);
    let report = build_report_for_windows(&windows, summary);
    let r2 = report
        .recommendations
        .iter()
        .find(|r| r.rule_name == rule_names::KV_CACHE_PRESSURE)
        .expect("r2 recommendation");
    assert!((r2.confidence - (4.0 / 15.0)).abs() < 1e-9);
}

#[test]
fn r2_backlog_fires_when_sustained_admission_pressure() {
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_backlog_window(10.0, 1.0, 9.0, 10.0, 10_000, 16))
        .collect();
    for w in windows.iter_mut().take(4) {
        // 90% KV satisfies kv_near_full; wait=2 avoids R2 queue_backpressure (>2).
        *w = mk_evaluable_backlog_window(90.0, 2.0, 4.0, 100.0, 100, 16);
    }
    let text = r2_issue_lines(&windows).join("\n");
    assert!(text.contains("[!] KV Cache Pressure: Admission Backlog"));
    assert!(
        !text.lines().any(|l| l.trim() == "[!] KV Cache Pressure"),
        "standard R2 must not co-fire: {text}"
    );
}

#[test]
fn r2_backlog_suppressed_when_standard_r2_fires() {
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_kv_window(50.0, false))
        .collect();
    for w in windows.iter_mut().take(4) {
        *w = mk_evaluable_backlog_window(89.0, 15.0, 15.0, 20.0, 100, 16);
    }
    let text = r2_issue_lines(&windows).join("\n");
    assert!(text.contains("[!] KV Cache Pressure"));
    assert!(!text.contains("Admission Backlog"));
}

#[test]
fn r5_concurrency_saturation_fires_on_sustained_saturation() {
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_kv_window(50.0, false))
        .collect();
    for w in windows.iter_mut().take(4) {
        *w = mk_evaluable_concurrency_saturation_window(32.0, 15.0, 32);
    }
    let ctx = mk_ctx();
    let summary = ai(&ctx, windows.last().expect("windows"));
    let text = format_diagnose_rules_for_windows_test(
        &windows,
        summary,
        false,
        "http://127.0.0.1:8000/metrics",
    )
    .join("\n");
    assert!(text.contains("[!] Concurrency Saturation"));
}

#[test]
fn build_report_for_windows_fires_r5_when_sustained() {
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_kv_window(50.0, false))
        .collect();
    for w in windows.iter_mut().take(4) {
        *w = mk_evaluable_concurrency_saturation_window(32.0, 15.0, 32);
    }
    let ctx = mk_ctx();
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    assert!(
        report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::CONCURRENCY_SATURATION)
    );
}

#[test]
fn r5_suppressed_when_r2_fires() {
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_concurrency_saturation_window(32.0, 15.0, 32))
        .collect();
    for w in windows.iter_mut().take(4) {
        *w = mk_evaluable_kv_window(89.0, true);
    }
    let ctx = mk_ctx();
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    assert!(
        report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::KV_CACHE_PRESSURE)
    );
    assert!(
        !report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::CONCURRENCY_SATURATION)
    );
}

#[test]
fn format_diagnose_rules_for_windows_all_non_evaluable() {
    let t = SystemTime::UNIX_EPOCH;
    let ctx = mk_ctx();
    let mut v = vllm_base();
    v.num_requests_running = None;
    let w1 = mk_win(snap(t, t, v.clone(), gpu_busy()));
    let w2 = mk_win(snap(t, t, v, gpu_busy()));
    let windows = vec![w1, w2];
    let summary = ai(&ctx, &windows[0]);
    let metrics_url = "http://127.0.0.1:8000/metrics";
    let lines = format_diagnose_rules_for_windows_test(&windows, summary, false, metrics_url);
    assert_eq!(
        lines,
        empty_run_diagnose_lines(
            false,
            &windows,
            false,
            &hint_for_empty(&ctx, &windows[0], metrics_url, 30),
            metrics_url
        )
    );
}

#[test]
fn empty_run_stdout_and_format_sites_byte_identical() {
    let t = SystemTime::UNIX_EPOCH;
    let metrics_url = "http://127.0.0.1:8000/metrics";
    let duration_secs = 30u64;
    let ctx = mk_ctx();

    let idle_v = {
        let mut v = vllm_base();
        v.num_requests_running = Some(0.0);
        v.generation_tokens_per_sec = Some(0.0);
        v
    };
    let broken_v = {
        let mut v = vllm_base();
        v.num_requests_running = None;
        v
    };

    let cases: [(bool, VllmRawMetrics); 2] = [(true, idle_v), (false, broken_v)];
    for (any_evaluable, vllm) in cases {
        let win = mk_win(snap(t, t, vllm, gpu_busy()));
        let windows = vec![win.clone(), win.clone()];
        let summary = ai(&ctx, &windows[0]);
        let hint = hint_for_empty(&ctx, &windows[0], metrics_url, duration_secs);
        for verbose in [false, true] {
            // stdout empty-run site: calls chooser directly with its any_evaluable flag.
            let from_stdout_site =
                empty_run_diagnose_lines(verbose, &windows, any_evaluable, &hint, metrics_url);
            // format.rs empty-run site: n_eval == 0 branch → same chooser.
            let from_format_site = format_diagnose_rules_for_windows(
                &windows,
                summary,
                &crate::engine::Report {
                    baseline: None,
                    recommendations: Vec::new(),
                    suppressed_rules: Vec::new(),
                    suppressed_recs: Vec::new(),
                    kv_max_seqs: None,
                    catalog_state_mismatch: None,
                    n_eval: 0,
                    skipped_broken: if any_evaluable { 0 } else { windows.len() },
                    skipped_idle: if any_evaluable { windows.len() } else { 0 },
                    energy_skew_skipped: 0,
                    gauge_missing: Default::default(),
                    limiter_evidence: None,
                },
                verbose,
                metrics_url,
                duration_secs,
                false,
            );
            assert_eq!(
                from_stdout_site, from_format_site,
                "any_evaluable={any_evaluable} verbose={verbose}"
            );
        }
    }
}

#[test]
fn idle_windows_skipped_in_rule_evaluation() {
    let t = SystemTime::UNIX_EPOCH;
    // High occupancy: R1 stays quiet on real traffic windows.
    let mut active = vllm_base();
    active.num_requests_running = Some(200.0);
    active.generation_tokens_per_sec = Some(200.0);
    active.max_num_seqs = Some(256);
    // Evaluable idle. Tiny running would false-positive R1 if the engine scored it.
    let mut idle = vllm_base();
    idle.num_requests_running = Some(0.5);
    idle.generation_tokens_per_sec = Some(0.0);
    idle.max_num_seqs = Some(256);

    let mut windows = Vec::with_capacity(15);
    for _ in 0..10 {
        windows.push(mk_win(snap(t, t, active.clone(), gpu_busy())));
    }
    for _ in 0..5 {
        windows.push(mk_win(snap(t, t, idle.clone(), gpu_busy())));
    }

    let ctx = mk_ctx();
    let summary = ai(&ctx, &windows[0]);
    let report = build_report_for_windows(&windows, summary);
    assert_eq!(report.n_eval, 10);
    assert_eq!(report.skipped_idle, 5);
    assert_eq!(report.skipped_broken, 0);
    assert!(
        !report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::UNDER_BATCHING)
    );
}

#[test]
fn idle_and_broken_skip_counts_preserved_when_n_eval_zero() {
    let t = SystemTime::UNIX_EPOCH;
    let mut idle = vllm_base();
    idle.num_requests_running = Some(0.5);
    idle.generation_tokens_per_sec = Some(0.0);
    idle.max_num_seqs = Some(256);
    let mut broken = vllm_base();
    broken.num_requests_running = None;

    let mut all_idle = Vec::with_capacity(15);
    for _ in 0..15 {
        all_idle.push(mk_win(snap(t, t, idle.clone(), gpu_busy())));
    }
    let ctx = mk_ctx();
    let summary = ai(&ctx, &all_idle[0]);
    let report = build_report_for_windows(&all_idle, summary);
    assert_eq!(report.n_eval, 0);
    assert_eq!(report.skipped_idle, 15);
    assert_eq!(report.skipped_broken, 0);

    let mut mixed = Vec::with_capacity(15);
    for _ in 0..14 {
        mixed.push(mk_win(snap(t, t, idle.clone(), gpu_busy())));
    }
    mixed.push(mk_win(snap(t, t, broken, gpu_busy())));
    let summary = ai(&ctx, &mixed[0]);
    let report = build_report_for_windows(&mixed, summary);
    assert_eq!(report.n_eval, 0);
    assert_eq!(report.skipped_idle, 14);
    assert_eq!(report.skipped_broken, 1);
}

#[test]
fn format_notes_split_idle_vs_telemetry_failure() {
    let t = SystemTime::UNIX_EPOCH;
    let mut active = vllm_base();
    active.num_requests_running = Some(200.0);
    active.generation_tokens_per_sec = Some(200.0);
    active.max_num_seqs = Some(256);
    active.num_requests_waiting = Some(3.0);
    active.kv_cache_usage_perc = Some(71.2);
    active.prefix_cache_hit_rate = Some(0.524);
    active.prompt_tokens_mean = Some(128.0);
    let mut idle = vllm_base();
    idle.num_requests_running = Some(0.5);
    idle.generation_tokens_per_sec = Some(0.0);
    idle.max_num_seqs = Some(256);
    let mut broken = vllm_base();
    broken.num_requests_running = None;

    // 10 active + 5 idle: idle note only, no telemetry failure.
    let mut windows = Vec::with_capacity(15);
    for _ in 0..10 {
        windows.push(mk_win(snap(t, t, active.clone(), gpu_busy())));
    }
    for _ in 0..5 {
        windows.push(mk_win(snap(t, t, idle.clone(), gpu_busy())));
    }
    let ctx = mk_ctx();
    let summary = ai(&ctx, &windows[0]);
    let text = format_diagnose_rules_for_windows_test(
        &windows,
        summary,
        false,
        "http://127.0.0.1:8000/metrics",
    )
    .join("\n");
    assert!(
        text.contains("windows were idle (excluded from analysis)"),
        "expected idle note, got:\n{text}"
    );
    assert!(
        !text.contains("(telemetry failure)"),
        "idle skips must not claim telemetry failure:\n{text}"
    );

    // 10 active + 2 non-evaluable: telemetry failure for exactly 2.
    let mut windows = Vec::with_capacity(12);
    for _ in 0..10 {
        windows.push(mk_win(snap(t, t, active.clone(), gpu_busy())));
    }
    for _ in 0..2 {
        windows.push(mk_win(snap(t, t, broken.clone(), gpu_busy())));
    }
    let summary = ai(&ctx, &windows[0]);
    let text = format_diagnose_rules_for_windows_test(
        &windows,
        summary,
        false,
        "http://127.0.0.1:8000/metrics",
    )
    .join("\n");
    assert!(
        text.contains("2 of 12 windows dropped (telemetry failure)"),
        "expected broken note for 2, got:\n{text}"
    );
    assert!(!text.contains("were idle"));

    // All idle: no telemetry failure.
    let mut windows = Vec::with_capacity(15);
    for _ in 0..15 {
        windows.push(mk_win(snap(t, t, idle.clone(), gpu_busy())));
    }
    let summary = ai(&ctx, &windows[0]);
    let text = format_diagnose_rules_for_windows_test(
        &windows,
        summary,
        false,
        "http://127.0.0.1:8000/metrics",
    )
    .join("\n");
    assert!(!text.contains("(telemetry failure)"));
}

#[test]
fn dag_layer2_suppresses_layer4_when_r2_fires() {
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_kv_window(50.0, false))
        .collect();
    for w in windows.iter_mut().take(4) {
        *w = {
            let mut win = mk_evaluable_kv_window(89.0, true);
            win.snapshot.vllm.num_requests_running = Some(3.1);
            win.snapshot.vllm.num_requests_waiting = Some(0.0);
            win.snapshot.vllm.tpot_ms = Some(35.0);
            win.snapshot.gpus[0].gpu_util_pct = Some(58.0);
            win
        };
    }
    let ctx = mk_ctx();
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    assert!(
        report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::KV_CACHE_PRESSURE)
    );
    assert!(
        !report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::UNDER_BATCHING)
    );
}

#[test]
fn r1_suppresses_r7() {
    let windows: Vec<_> = (0..10)
        .map(|_| mk_r7_headroom_window(20.0, 32, 0.0, 10.0))
        .collect();
    let ctx = mk_r7_ctx(32);
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    assert!(
        report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::UNDER_BATCHING)
    );
    assert!(
        !report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::CONFIG_HEADROOM)
    );
}

#[test]
fn kv_warning_requires_significance() {
    let mut windows: Vec<_> = (0..10).map(|_| mk_r1_window_with_kv(50.0)).collect();
    windows[0] = mk_r1_window_with_kv(78.0);
    let ctx = mk_ctx();
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    let text = report.recommendations[0].display_lines.join("\n");
    assert!(!text.contains("Monitor KV cache when scaling up."));
    assert!(!text.contains("• Monitor"));
}

#[test]
fn kv_warning_fires_when_significant() {
    let mut windows: Vec<_> = (0..10).map(|_| mk_r1_window_with_kv(50.0)).collect();
    for w in windows.iter_mut().take(4) {
        *w = mk_r1_window_with_kv(78.0);
    }
    let ctx = mk_ctx();
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    let text = report.recommendations[0].display_lines.join("\n");
    assert!(text.contains("        Monitor KV cache when scaling up."));
    assert!(!text.contains("• Monitor"));
}

#[test]
fn r2_suppresses_r1_in_table() {
    let t = SystemTime::UNIX_EPOCH;
    let windows: Vec<_> = (0..10)
        .map(|_| {
            let mut v = vllm_base();
            v.kv_cache_usage_perc = Some(89.0);
            v.num_preemptions_per_sec = Some(0.05);
            mk_win(snap(t, t, v, gpu_busy()))
        })
        .collect();
    let ctx = mk_ctx();
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    assert!(
        report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::KV_CACHE_PRESSURE)
    );
    assert!(
        !report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::UNDER_BATCHING)
    );
}

#[test]
fn r7_silent_when_waiting_nonzero_r5_territory() {
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_kv_window(50.0, false))
        .collect();
    for w in windows.iter_mut().take(6) {
        *w = mk_evaluable_concurrency_saturation_window(32.0, 15.0, 32);
    }
    let ctx = mk_r7_ctx(32);
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    assert!(
        !report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::CONFIG_HEADROOM)
    );
}

#[test]
fn mixed_run_r6_suppresses_r1() {
    // Half under-batched (R1), half prefill-bound (R6). Both significant;
    // ME before layer filter → R6 primary, R1 suppressed.
    let mut windows: Vec<_> = (0..10)
        .map(|_| mk_r6_prefill_window(2.5, 10.0, 5.0, Some(50.0)))
        .collect();
    for w in windows.iter_mut().skip(5) {
        *w = mk_r6_prefill_window(12.0, 10.0, 5.0, Some(80.0));
    }
    let ctx = mk_llama8b_h100_ctx(&windows[0].snapshot);
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    assert!(
        report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::PREFILL_BOUND)
    );
    assert!(
        !report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::UNDER_BATCHING)
    );
    assert!(
        report
            .suppressed_rules
            .iter()
            .any(|(suppressed, suppressor)| {
                *suppressed == rule_names::UNDER_BATCHING
                    && *suppressor == rule_names::PREFILL_BOUND
            })
    );
}

#[test]
fn r1_fires_when_r6_muted_by_tpot() {
    // High prompt/gen + low occupancy, but TPOT near floor so R6 declines.
    // Must not leave the window silent: R1 owns under-batching.
    let windows: Vec<_> = (0..10)
        .map(|_| mk_r6_prefill_window(12.0, 10.0, 5.0, Some(5.0)))
        .collect();
    let ctx = mk_llama8b_h100_ctx(&windows[0].snapshot);
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    assert!(
        report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::UNDER_BATCHING),
        "expected R1 when R6 TPOT-muted; got {:?}",
        report
            .recommendations
            .iter()
            .map(|g| g.rule_name)
            .collect::<Vec<_>>()
    );
    assert!(
        !report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::PREFILL_BOUND)
    );
}

#[test]
fn r6_fires_when_r1_prefill_gate_suppresses_r1() {
    let windows: Vec<_> = (0..10)
        .map(|_| mk_r6_prefill_window(12.0, 10.0, 5.0, Some(50.0)))
        .collect();
    let ctx = mk_llama8b_h100_ctx(&windows[0].snapshot);
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    assert!(
        report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::PREFILL_BOUND)
    );
}

#[test]
fn r6_not_primary_when_r2_outscores() {
    let mut windows: Vec<_> = (0..10)
        .map(|_| mk_r6_prefill_window(12.0, 10.0, 50.0, Some(50.0)))
        .collect();
    for w in windows.iter_mut().take(4) {
        *w = mk_evaluable_kv_window(89.0, true);
    }
    let ctx = mk_llama8b_h100_ctx(&windows[0].snapshot);
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    assert!(
        report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::KV_CACHE_PRESSURE)
    );
    assert!(
        !report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::PREFILL_BOUND)
    );
}

#[test]
fn waiting_none_suppresses() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.num_requests_waiting = None;
    v.tpot_ms = Some(35.0);
    let s = snap(t, t, v, gpu_low());
    let win = mk_win(s);
    assert!(r1_recommendation(r1_test_input(&win.snapshot)).is_none());
}

#[test]
fn waiting_at_two_suppresses() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.num_requests_waiting = Some(2.0);
    v.tpot_ms = Some(35.0);
    let s = snap(t, t, v, gpu_low());
    let win = mk_win(s);
    assert!(r1_recommendation(r1_test_input(&win.snapshot)).is_none());
}

#[test]
fn running_at_occupancy_threshold_suppresses() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.num_requests_running = Some(64.0);
    let s = snap(t, t, v, gpu_low());
    let win = mk_win(s);
    assert!(r1_recommendation(r1_test_input(&win.snapshot)).is_none());
}

#[test]
fn max_seqs_zero_suppresses() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.max_num_seqs = Some(0);
    v.tpot_ms = Some(35.0);
    let s = snap(t, t, v, gpu_low());
    let win = mk_win(s);
    assert!(r1_recommendation(r1_test_input(&win.snapshot)).is_none());
}

#[test]
fn nan_running_suppresses() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.num_requests_running = Some(f64::NAN);
    v.tpot_ms = Some(35.0);
    let s = snap(t, t, v, gpu_low());
    let win = mk_win(s);
    assert!(r1_recommendation(r1_test_input(&win.snapshot)).is_none());
}

#[test]
fn format_diagnose_non_verbose_omits_kv_pressure_when_r4_fires() {
    let (ctx, win) = input_r4_suppresses_r2();
    let windows: Vec<_> = (0..ENGINE_MIN_PERSISTENT_WINDOWS)
        .map(|_| win.clone())
        .collect();
    let summary = ai(&ctx, &windows[0]);
    let report = build_report_for_windows(&windows, summary);
    assert_eq!(report.recommendations[0].rule_name, rule_names::OOM_RISK);
    assert!(
        !report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::KV_CACHE_PRESSURE)
    );
}

#[test]
fn r2_recommendation_confidence_from_density_counts() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_high_kv();
    v.num_preemptions_per_sec = Some(0.05);
    let s = snap(t, t, v, gpu_low());
    let r = r2_recommendation(R2RecommendationInput {
        snapshot: &s,
        max_model_len: None,
        kv_headroom_gb: None,
        kv_max_seqs: None,
        capacity_label: KvCapacityLabel::Derived,
        windows_fired: 1,
        total_evaluable: 4,
        fp8_compiler_available: false,
    })
    .expect("fired");
    assert_eq!(r.rule_name, rule_names::KV_CACHE_PRESSURE);
    assert_eq!(r.impact, 5);
    assert!((r.confidence - 0.5).abs() < 1e-9);
}

#[test]
fn r2_recommendation_includes_peak_from_detail() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_high_kv();
    v.kv_cache_usage_perc = Some(89.0);
    v.kv_cache_peak_perc = Some(99.4);
    v.num_preemptions_per_sec = Some(0.05);
    let s = snap(t, t, v, gpu_low());
    let r = r2_recommendation(R2RecommendationInput {
        snapshot: &s,
        max_model_len: None,
        kv_headroom_gb: None,
        kv_max_seqs: None,
        capacity_label: KvCapacityLabel::Derived,
        windows_fired: 1,
        total_evaluable: 1,
        fp8_compiler_available: false,
    })
    .expect("fired");
    let text = r.display_lines.join("\n");
    assert!(text.contains("KV cache 89% avg, 99% peak (threshold: 88%)."));
}

#[test]
fn kv_cache_pressure_fires_at_88_boundary_with_stress() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.kv_cache_usage_perc = Some(88.0);
    v.num_preemptions_per_sec = Some(0.05);
    let s = snap(t, t, v, gpu_low());
    match rule2_kv_cache_pressure(&s) {
        Rule2Outcome::Fired(d) => {
            assert!((d.kv_cache_usage_perc.unwrap() - 88.0).abs() < 1e-9);
            assert!(d.preemptions_active);
        }
        Rule2Outcome::NotFired => panic!("expected fired at 88% with stress"),
    }
}

#[test]
fn kv_cache_pressure_suppressed_below_88() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.kv_cache_usage_perc = Some(87.9);
    v.num_preemptions_per_sec = Some(0.05);
    let s = snap(t, t, v, gpu_low());
    assert!(matches!(
        rule2_kv_cache_pressure(&s),
        Rule2Outcome::NotFired
    ));
}

#[test]
fn kv_cache_pressure_preemption_displays_without_premature_confidence() {
    let t = SystemTime::UNIX_EPOCH;
    let s_kv_only = snap(t, t, vllm_high_kv_stressed(), gpu_busy());
    let ctx2 = mk_ctx();
    let win_kv_only = mk_win(s_kv_only);
    let r2_text = r2_recommendation(R2RecommendationInput {
        snapshot: &win_kv_only.snapshot,
        max_model_len: None,
        kv_headroom_gb: None,
        kv_max_seqs: None,
        capacity_label: KvCapacityLabel::Derived,
        windows_fired: 1,
        total_evaluable: 1,
        fp8_compiler_available: false,
    })
    .expect("r2 fired")
    .display_lines
    .join("\n");
    assert!(!r2_text.contains("Confidence:"));
    let text =
        format_diagnose_rules_test(&ctx2, &win_kv_only, false, "http://127.0.0.1:8000/metrics")
            .join("\n");
    assert!(text.contains("Cause:"));
    assert!(text.contains("KV cache 89% avg, 89% peak (threshold: 88%)."));
    assert!(text.contains("Expected: TTFT and TPOT recover once evictions stop."));
    assert!(text.contains("Lower --max-num-seqs to reduce KV demand"));
    assert!(text.contains("Switch --kv-cache-dtype fp8"));
}

#[test]
fn r2_fires_on_single_preemption_window() {
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_kv_window(50.0, false))
        .collect();
    windows[0] = mk_evaluable_kv_window(89.0, true);
    let report = r2_report(&windows);
    assert!(
        !report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::KV_CACHE_PRESSURE)
    );
}

#[test]
fn r2_fires_on_two_critical_kv_windows_without_preemptions() {
    let mut windows: Vec<_> = (0..10)
        .map(|_| mk_evaluable_kv_window(50.0, false))
        .collect();
    windows[0] = mk_evaluable_kv_window(96.0, false);
    windows[1] = mk_evaluable_kv_window(97.0, false);
    let report = r2_report(&windows);
    assert!(
        !report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::KV_CACHE_PRESSURE)
    );
}

#[test]
fn r2_does_not_fire_when_kv_high_but_tpot_stable_and_no_preemptions() {
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_kv_window(50.0, false))
        .collect();
    for w in windows.iter_mut().take(4) {
        *w = mk_evaluable_kv_window(89.0, false);
    }
    let report = r2_report(&windows);
    assert!(
        !report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::KV_CACHE_PRESSURE)
    );
}

#[test]
fn r2_does_not_fire_on_single_critical_kv_window_without_preemptions() {
    let mut windows: Vec<_> = (0..10)
        .map(|_| mk_evaluable_kv_window(50.0, false))
        .collect();
    windows[0] = mk_evaluable_kv_window(96.0, false);
    let text = r2_issue_lines(&windows).join("\n");
    assert!(!text.contains("[!] KV Cache Pressure"));
    assert!(!text.contains("KV Cache Pressure: not triggered"));
    assert!(!text.contains("Low Prefix Reuse: not triggered"));
    assert!(!text.contains("Seen in"));
}

#[test]
fn r2_fires_on_sustained_warning_level_kv() {
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_kv_window(50.0, false))
        .collect();
    for w in windows.iter_mut().take(4) {
        *w = mk_evaluable_kv_window(89.0, true);
    }
    let text = r2_issue_lines(&windows).join("\n");
    assert!(text.contains("[!] KV Cache Pressure"));
}

#[test]
fn cause_line_peak_matches_summary_snapshot() {
    // 4 windows fired at 95% KV; 11 windows below threshold (30%).
    // Summary snapshot carries kv_cache_peak_perc=95.0 (realistic: profiler takes
    // MAX across windows) and kv_cache_usage_perc=92.0 (different value so we can
    // confirm the cause line reads kv_cache_peak_perc, not the usage fallback).
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_kv_window(30.0, true))
        .collect();
    for w in windows.iter_mut().take(4) {
        *w = mk_evaluable_kv_window(95.0, true);
    }
    let ctx = mk_ctx();
    let mut summary_win = windows.last().expect("windows").clone();
    summary_win.snapshot.vllm.kv_cache_usage_perc = Some(92.0);
    summary_win.snapshot.vllm.kv_cache_peak_perc = Some(95.0);
    let summary = ai(&ctx, &summary_win);
    let text = format_diagnose_rules_for_windows_test(
        &windows,
        summary,
        false,
        "http://127.0.0.1:8000/metrics",
    )
    .join("\n");
    assert!(text.contains("KV cache 95% avg, 95% peak (threshold: 88%)."));
    assert!(!text.contains("92% avg"));
}

#[test]
fn cause_kv_line_precedes_preemptions_and_queue() {
    // Verifies output ordering: KV peak line must appear before preemptions
    // and queue backpressure lines (#1 fix).
    // Uses a summary snapshot that has both signals active so all three
    // cause lines appear, then checks position order.
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_kv_window(30.0, false))
        .collect();
    for w in windows.iter_mut().take(6) {
        *w = mk_evaluable_kv_window(91.0, true);
        w.snapshot.vllm.num_requests_waiting = Some(5.0);
    }
    let ctx = mk_ctx();
    let mut summary_win = windows.last().expect("windows").clone();
    summary_win.snapshot.vllm.kv_cache_usage_perc = Some(91.0);
    summary_win.snapshot.vllm.kv_cache_peak_perc = Some(91.0);
    summary_win.snapshot.vllm.num_preemptions_per_sec = Some(0.05);
    summary_win.snapshot.vllm.num_requests_waiting = Some(5.0);
    let summary = ai(&ctx, &summary_win);
    let text = format_diagnose_rules_for_windows_test(
        &windows,
        summary,
        false,
        "http://127.0.0.1:8000/metrics",
    )
    .join("\n");
    let pos_kv = text.find("KV cache").expect("KV peak line missing");
    let pos_evidence = text
        .find("Scheduler evicting")
        .expect("evidence line missing");
    assert!(pos_kv < pos_evidence, "KV line must precede evidence line");
}

#[test]
fn backlog_display_matches_spec() {
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_backlog_window(10.0, 1.0, 9.0, 10.0, 10_000, 16))
        .collect();
    for w in windows.iter_mut().take(4) {
        *w = mk_evaluable_backlog_window(90.0, 2.0, 4.0, 100.0, 100, 16);
    }
    let snap = windows.last().expect("windows").snapshot.clone();
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(8192),
        max_num_seqs: Some(256),
        ..Default::default()
    };
    let mut ctx = StaticContext::from_snapshot(&snap, cfg);
    ctx.model.param_count = Some(8_000_000_000);
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    let r = report
        .recommendations
        .iter()
        .find(|g| g.rule_name == rule_names::KV_ADMISSION_BACKLOG)
        .expect("backlog kv recommendation")
        .clone();
    let display = r.display_lines.join("\n");
    assert!(display.contains("[!] KV Cache Pressure: Admission Backlog"));
    assert!(display.contains("KV cache 90% avg, 90% peak (threshold: 88%)"));
    assert!(display.contains("Raise --gpu-memory-utilization"));
    assert!(
        display.contains("Lower --max-num-seqs to reduce KV demand"),
        "backlog must prescribe seat guidance: {display}"
    );
    if display.contains("Lower --max-model-len") {
        let cuts = display
            .find("    Cuts throughput:")
            .expect("shrink requires Cuts throughput header");
        let shrink = display.find("Lower --max-model-len").expect("shrink");
        assert!(cuts < shrink, "shrink must sit under Cuts throughput:");
    }
}

#[test]
fn rule3_fires_when_hit_below_35_and_gates_pass() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.prefix_cache_hit_rate = Some(0.34);
    v.prompt_tokens_mean = Some(25.0);
    v.request_success_per_sec = Some(40.0);
    v.num_requests_running = Some(1.0);
    let s = snap(t, t, v, gpu_busy());
    let win = mk_win(s);
    match rule3_low_prefix_reuse(&win.snapshot) {
        Rule3Outcome::Fired(d) => {
            assert_eq!(d.hit_rate, Some(0.34));
            assert_eq!(d.prompt_tokens_mean, Some(25.0));
        }
        Rule3Outcome::NotFired => panic!("expected fired"),
    }
    let r = r3_recommendation(&win.snapshot).expect("r3 fired");
    assert_eq!(r.rule_name, rule_names::LOW_PREFIX_REUSE);
    assert_eq!(r.impact, 2);
    assert!((r.confidence - 0.9).abs() < 1e-9);
}

#[test]
fn rule3_suppressed_at_or_above_35() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.prefix_cache_hit_rate = Some(0.35);
    v.prompt_tokens_mean = Some(25.0);
    v.request_success_per_sec = Some(40.0);
    v.num_requests_running = Some(1.0);
    let s = snap(t, t, v, gpu_busy());
    assert!(matches!(rule3_low_prefix_reuse(&s), Rule3Outcome::NotFired));
}

#[test]
fn format_diagnose_rule3_verbose_working_effectively_when_rate_healthy() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.prefix_cache_hit_rate = Some(0.50);
    let s = snap(t, t, v, gpu_busy());
    let ctx = mk_ctx();
    let win = mk_win(s);
    let text =
        format_diagnose_rules_test(&ctx, &win, true, "http://127.0.0.1:8000/metrics").join("\n");
    assert!(text.contains("Low Prefix Reuse: not triggered"));
}

#[test]
fn format_diagnose_rule3_verbose_not_indicated_when_rate_low_but_prompt_below_floor() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.prefix_cache_hit_rate = Some(0.20);
    v.prompt_tokens_mean = Some(10.0);
    let s = snap(t, t, v, gpu_busy());
    let ctx = mk_ctx();
    let win = mk_win(s);
    let text =
        format_diagnose_rules_test(&ctx, &win, true, "http://127.0.0.1:8000/metrics").join("\n");
    assert!(text.contains("Low Prefix Reuse: not triggered"));
    assert!(!text.contains("working effectively"));
}

#[test]
fn format_diagnose_rules_inserts_blank_between_rule_blocks() {
    let (ctx, win) = {
        let mut v = vllm_high_kv();
        v.num_preemptions_per_sec = Some(0.05);
        v.tpot_ms = Some(35.0);
        v.generation_tokens_per_sec = Some(30.0);
        v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
        let mut g = gpu_low();
        g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
        g.power_watts = Some(400.0);
        g.aligned_power_watts = Some(400.0);
        let t = SystemTime::UNIX_EPOCH;
        let snap = snap(t, t, v, g);
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let ctx = StaticContext::from_snapshot(&snap, cfg);
        let win = mk_win(snap);
        (ctx, win)
    };
    let lines = format_diagnose_rules_test(&ctx, &win, false, "http://127.0.0.1:8000/metrics");
    let idx_kv = lines
        .iter()
        .position(|l| l.contains("[!] KV Cache Pressure"))
        .expect("rule2");
    assert!(
        !lines.iter().any(|l| l.contains("[!] Under-batching")),
        "layer 2 suppresses layer 4: {lines:?}"
    );
    assert!(
        !lines.iter().any(|l| l.contains("No issues detected")),
        "should not append no-issues line when at least one rule fired"
    );
    let waste_lines: Vec<_> = lines.iter().filter(|l| l.contains("/hr ")).collect();
    assert!(
        waste_lines.is_empty(),
        "waste lines removed from diagnose output: {lines:?}"
    );
    let _ = idx_kv;
}

#[test]
fn format_diagnose_rules_for_windows_matches_requested_style_when_some_rules_fire() {
    let t = SystemTime::UNIX_EPOCH;
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        max_num_seqs: Some(256),
        ..Default::default()
    };
    let mut windows = Vec::new();
    for _i in 0..10 {
        let mut v = vllm_base();
        v.max_num_seqs = Some(256);
        v.num_requests_waiting = Some(1.0);
        v.kv_cache_usage_perc = Some(71.2);
        v.prefix_cache_hit_rate = Some(0.524);
        v.prompt_tokens_mean = Some(128.0);
        v.generation_tokens_per_sec = Some(1580.0);
        v.num_requests_running = Some(3.2);
        v.tpot_ms = Some(35.0);
        let mut g = gpu_busy();
        g.gpu_util_pct = None;
        g.power_watts = Some(312.0);
        g.vram_used_mb = Some(62 * 1024);
        g.vram_total_mb = Some(80 * 1024);
        v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
        g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
        windows.push(mk_win(snap(t, t, v, g)));
    }
    let ctx = StaticContext::from_snapshot(&windows[0].snapshot, cfg);
    let summary = ai(&ctx, windows.last().expect("summary source"));
    let lines = format_diagnose_rules_for_windows_test(
        &windows,
        summary,
        false,
        "http://127.0.0.1:8000/metrics",
    );
    let text = lines.join("\n");
    assert!(text.contains("Under-batching: Insufficient Concurrency"));
    assert!(text.contains("Seen in 100% of windows"));
    assert!(text.contains("Requests"));
    assert!(text.contains("running"));
    assert!(text.contains("    Cause:"));
    assert!(text.contains("Batch more requests or increase client concurrency (253 slots idle)"));
    assert!(!text.contains("KV Cache Pressure: not triggered"));
    assert!(!text.contains("Low Prefix Reuse: not triggered"));
    assert!(!text.contains("Concurrency Saturation: not triggered"));
}

#[test]
fn format_diagnose_rules_for_windows_no_fires_is_single_no_issues_line() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.num_requests_running = Some(20.0);
    v.num_requests_waiting = Some(3.0);
    v.kv_cache_usage_perc = Some(71.2);
    v.prefix_cache_hit_rate = Some(0.524);
    v.prompt_tokens_mean = Some(128.0);
    v.generation_tokens_per_sec = Some(100.0);
    v.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
    let mut g = gpu_busy();
    g.gpu_util_pct = Some(74.0);
    g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
    g.vram_total_mb = Some(80 * 1024);
    let snap = snap(t, t, v, g);
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let ctx = StaticContext::from_snapshot(&snap, cfg);
    let win = mk_win(snap);
    let windows = vec![win.clone(), win.clone(), win];
    let summary = ai(&ctx, windows.last().expect("windows"));
    let lines = format_diagnose_rules_for_windows_test(
        &windows,
        summary,
        false,
        "http://127.0.0.1:8000/metrics",
    );
    assert_eq!(
        lines.first().map(String::as_str),
        Some("No issues detected.")
    );
    assert!(
        lines.get(1).is_some_and(|l| l.starts_with("Capped by ")),
        "expected limiter line, got: {lines:?}"
    );
}

#[test]
fn no_rules_limiter_uses_aggregates_not_last_idle_snapshot() {
    let t = SystemTime::UNIX_EPOCH;
    let mut g = gpu_busy();
    g.gpu_util_pct = Some(80.0);
    g.gpu_name = Some("NVIDIA H100 80GB HBM3".to_string());
    g.vram_total_mb = Some(80 * 1024);

    let mut busy = vllm_base();
    busy.model_name = Some("meta-llama/Llama-3.1-8B-Instruct".to_string());
    busy.num_requests_running = Some(128.0);
    busy.num_requests_waiting = Some(0.0);
    busy.generation_tokens_per_sec = Some(180.0);
    busy.prompt_tokens_per_sec = Some(30.0);
    busy.window_duration_secs = Some(2.0);
    busy.tpot_ms = Some(20.0);

    let mut idle = busy.clone();
    idle.num_requests_running = Some(0.0);
    idle.generation_tokens_per_sec = Some(0.0);

    let windows = vec![
        mk_win(snap(t, t, busy.clone(), g.clone())),
        mk_win(snap(t, t, busy.clone(), g.clone())),
        mk_win(snap(t, t, busy, g.clone())),
        mk_win(snap(t, t, idle, g)),
    ];
    let cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        ..Default::default()
    };
    let ctx = StaticContext::from_snapshot(&windows[0].snapshot, cfg);
    let summary = ai(&ctx, windows.last().expect("windows"));
    let lines = format_diagnose_rules_for_windows_test(
        &windows,
        summary,
        false,
        "http://127.0.0.1:8000/metrics",
    );
    assert_eq!(
        lines.first().map(String::as_str),
        Some("No issues detected.")
    );
    assert!(
        lines.get(1).is_some_and(|l| l.starts_with("Capped by ")),
        "expected limiter line, got: {lines:?}"
    );
    assert!(
        !lines
            .get(1)
            .is_some_and(|l| l.starts_with("Capped by traffic")),
        "must not use last idle snapshot as traffic evidence: {lines:?}"
    );
}

#[test]
fn insufficient_load_returns_advisory_not_no_issues() {
    let windows = vec![
        mk_evaluable_kv_window(89.0, true),
        mk_evaluable_kv_window(89.0, true),
    ];
    let ctx = mk_ctx();
    let summary = ai(&ctx, windows.last().expect("windows"));
    let text = format_diagnose_rules_for_windows_test(
        &windows,
        summary,
        false,
        "http://127.0.0.1:8000/metrics",
    )
    .join("\n");
    assert!(text.contains("Insufficient Sustained Load"));
    assert!(!text.contains("No issues detected"));
}

#[test]
fn r5_uses_session_kv_peak_from_non_r5_window() {
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_kv_window(50.0, false))
        .collect();
    // r5-significant windows with moderate KV.
    for w in windows.iter_mut().take(4) {
        *w = mk_evaluable_concurrency_saturation_window(32.0, 15.0, 32);
        w.snapshot.vllm.kv_cache_usage_perc = Some(70.0);
    }
    // One high-KV r2 window (non-significant for r2), should still set session peak.
    windows[10] = mk_evaluable_kv_window(95.0, true);
    // Simulate gauge drift: peak captures the spike, avg usage does not.
    windows[10].snapshot.vllm.kv_cache_usage_perc = Some(60.0);
    windows[10].snapshot.vllm.kv_cache_peak_perc = Some(95.0);
    windows[10].snapshot.vllm.num_requests_running = Some(20.0);
    windows[10].snapshot.vllm.max_num_seqs = Some(32);
    windows[10].snapshot.vllm.num_requests_waiting = Some(1.0);

    let ctx = mk_ctx();
    let summary = ai(&ctx, windows.last().expect("windows"));
    let text = format_diagnose_rules_for_windows_test(
        &windows,
        summary,
        false,
        "http://127.0.0.1:8000/metrics",
    )
    .join("\n");

    assert!(
        text.contains("[!] Concurrency Saturation"),
        "expected r5: {text}"
    );
    // Session peak 95% (from non-r5 window) blocks raise even though landing KV is low.
    assert!(
        text.contains("KV at 95%: scheduler at cap, pool full; no config change helps"),
        "session peak must gate the fix: {text}"
    );
    assert!(text.contains("Add a replica"));
    assert!(!text.contains("Raise --max-num-seqs"));
}

#[test]
fn session_kv_peak_from_non_r5_window_reaches_build_report_from_eval() {
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_kv_window(50.0, false))
        .collect();
    for w in windows.iter_mut().take(4) {
        *w = mk_evaluable_concurrency_saturation_window(32.0, 15.0, 32);
        w.snapshot.vllm.kv_cache_usage_perc = Some(70.0);
    }
    // Non-r5 window carries session spike via peak metric.
    windows[10] = mk_evaluable_kv_window(95.0, true);
    windows[10].snapshot.vllm.kv_cache_usage_perc = Some(60.0);
    windows[10].snapshot.vllm.kv_cache_peak_perc = Some(95.0);
    windows[10].snapshot.vllm.num_requests_running = Some(20.0);
    windows[10].snapshot.vllm.max_num_seqs = Some(32);
    windows[10].snapshot.vllm.num_requests_waiting = Some(1.0);

    let ctx = mk_ctx();
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    let r5 = report
        .recommendations
        .iter()
        .find(|g| g.rule_name == rule_names::CONCURRENCY_SATURATION)
        .expect("r5 group");
    let text = r5.display_lines.join("\n");
    assert!(
        text.contains("KV at 95%: scheduler at cap, pool full; no config change helps"),
        "display fix line must use session peak: {text}"
    );
    assert!(text.contains("Add a replica"));
    assert!(!text.contains("Raise --max-num-seqs"));
}

#[test]
fn r6_fires_as_primary_when_no_other_rules() {
    let windows: Vec<_> = (0..10)
        .map(|_| mk_r6_prefill_window(12.0, 10.0, 50.0, Some(50.0)))
        .collect();
    let s = windows[0].snapshot.clone();
    let ctx = mk_llama8b_h100_ctx(&s);
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    assert_eq!(
        report.recommendations[0].rule_name,
        rule_names::PREFILL_BOUND
    );
}

#[test]
fn r6_suppresses_r7_when_both_fire() {
    let mut windows: Vec<_> = (0..10)
        .map(|_| mk_r6_prefill_window(12.0, 10.0, 50.0, Some(50.0)))
        .collect();
    for w in &mut windows {
        w.snapshot.vllm.max_num_seqs = Some(32);
    }
    let s = windows[0].snapshot.clone();
    let mut cfg = VllmConfig {
        dtype: Some("bf16".to_string()),
        max_model_len: Some(2048),
        max_num_seqs: Some(32),
        ..Default::default()
    };
    cfg.enable_chunked_prefill = Some(false);
    let ctx = StaticContext::from_snapshot(&s, cfg);
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    assert!(
        report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::PREFILL_BOUND)
    );
    assert!(
        !report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::CONFIG_HEADROOM)
    );
    assert!(
        report
            .suppressed_rules
            .iter()
            .any(|(suppressed, suppressor)| {
                *suppressed == rule_names::CONFIG_HEADROOM
                    && *suppressor == rule_names::PREFILL_BOUND
            })
    );
}

#[test]
fn r7_fires_as_primary_when_alone() {
    let windows: Vec<_> = (0..10)
        .map(|_| mk_r7_headroom_window(60.0, 64, 0.0, 50.0))
        .collect();
    let ctx = mk_r7_ctx(64);
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    assert_eq!(report.recommendations.len(), 1);
    assert_eq!(
        report.recommendations[0].rule_name,
        rule_names::CONFIG_HEADROOM
    );
}

#[test]
fn r7_dropped_when_run_level_target_at_or_below_current() {
    // Per-window R7 fires on ridge (no Observed). Landing snapshot reports a tight
    // Observed concurrency so run-level target is 18 while current max is 20.
    let mut windows: Vec<_> = (0..10)
        .map(|_| mk_r7_headroom_window(15.0, 20, 0.0, 50.0))
        .collect();
    windows
        .last_mut()
        .expect("windows")
        .snapshot
        .vllm
        .cache_config
        .kv_cache_max_concurrency = Some(22.5);
    let ctx = mk_r7_ctx(20);
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    assert!(
        !report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::CONFIG_HEADROOM),
        "run-level target <= current must drop R7; got {:?}",
        report
            .recommendations
            .iter()
            .map(|g| g.rule_name)
            .collect::<Vec<_>>()
    );
}

#[test]
fn dag_layer2_suppresses_layer3_when_r2_fires() {
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_concurrency_saturation_window(32.0, 15.0, 32))
        .collect();
    for w in windows.iter_mut().take(4) {
        *w = mk_evaluable_kv_window(89.0, true);
    }
    let ctx = mk_ctx();
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    assert!(
        report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::KV_CACHE_PRESSURE)
    );
    assert!(
        !report
            .recommendations
            .iter()
            .any(|g| g.rule_name == rule_names::CONCURRENCY_SATURATION)
    );
}

#[test]
fn verbose_not_evaluated_when_waiting_gauge_missing_all_windows() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.num_requests_waiting = None;
    v.num_requests_running = Some(64.0);
    v.generation_tokens_per_sec = Some(100.0);
    v.kv_cache_usage_perc = Some(10.0);
    v.prefix_cache_hit_rate = Some(0.9);
    let s = snap(t, t, v, gpu_busy());
    let ctx = mk_ctx();
    let win = mk_win(s);
    let text =
        format_diagnose_rules_test(&ctx, &win, true, "http://127.0.0.1:8000/metrics").join("\n");
    assert!(
        text.contains("Under-batching: not evaluated (waiting gauge missing)."),
        "{text}"
    );
    assert!(
        text.contains("Concurrency Saturation: not evaluated (waiting gauge missing)."),
        "{text}"
    );
    assert!(!text.contains("Under-batching: not triggered"));
}

#[test]
fn verbose_not_triggered_suffix_when_waiting_missing_in_some_windows() {
    let t = SystemTime::UNIX_EPOCH;
    let mut present = vllm_base();
    present.num_requests_running = Some(64.0);
    present.generation_tokens_per_sec = Some(100.0);
    present.num_requests_waiting = Some(0.0);
    present.kv_cache_usage_perc = Some(10.0);
    present.prefix_cache_hit_rate = Some(0.9);
    let mut missing = present.clone();
    missing.num_requests_waiting = None;
    let ctx = mk_ctx();
    let windows = vec![
        mk_win(snap(t, t, present.clone(), gpu_busy())),
        mk_win(snap(t, t, present.clone(), gpu_busy())),
        mk_win(snap(t, t, missing, gpu_busy())),
        mk_win(snap(t, t, present, gpu_busy())),
    ];
    let summary = ai(&ctx, &windows[0]);
    let text = format_diagnose_rules_for_windows_test(
        &windows,
        summary,
        true,
        "http://127.0.0.1:8000/metrics",
    )
    .join("\n");
    assert!(
        text.contains("Under-batching: not triggered (waiting gauge missing in 1/4 windows)."),
        "{text}"
    );
}

#[test]
fn verbose_not_triggered_byte_identical_when_no_gauge_gaps() {
    let t = SystemTime::UNIX_EPOCH;
    let mut v = vllm_base();
    v.num_requests_running = Some(64.0);
    v.generation_tokens_per_sec = Some(100.0);
    v.num_requests_waiting = Some(0.0);
    v.kv_cache_usage_perc = Some(10.0);
    v.prefix_cache_hit_rate = Some(0.9);
    let s = snap(t, t, v, gpu_busy());
    let ctx = mk_ctx();
    let win = mk_win(s);
    let text =
        format_diagnose_rules_test(&ctx, &win, true, "http://127.0.0.1:8000/metrics").join("\n");
    assert!(text.contains("Under-batching: not triggered"));
    assert!(!text.contains("waiting gauge missing"));
    assert!(!text.contains("not evaluated"));
}

#[test]
fn energy_skew_skipped_counted_on_report() {
    let t0 = SystemTime::UNIX_EPOCH;
    let t1 = t0 + Duration::from_secs(5);
    let mut v = vllm_base();
    v.num_requests_running = Some(64.0);
    v.generation_tokens_per_sec = Some(100.0);
    v.num_requests_waiting = Some(0.0);
    let aligned = mk_win(snap(t0, t0, v.clone(), gpu_busy()));
    let skewed = mk_win(snap(t0, t1, v, gpu_busy()));
    let windows = vec![aligned.clone(), aligned.clone(), skewed];
    let ctx = mk_ctx();
    let summary = ai(&ctx, &windows[0]);
    let report = build_report_for_windows(&windows, summary);
    assert_eq!(report.n_eval, 3);
    assert_eq!(report.energy_skew_skipped, 1);
}
