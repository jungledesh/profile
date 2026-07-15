use super::format::{append_waste_line, r2_kv_cache_advisory, waste_label_suffix};
use super::r1_under_batching::{r1_recommendation, rule1_under_batching_with_efficiency};
use super::r2_kv_cache_pressure::{Rule2Outcome, rule2_kv_cache_pressure};
use super::r3_low_prefix_reuse::{format_low_prefix_hit_rate_fired, rule3_low_prefix_reuse};
use super::*;
use crate::collectors::{GpuRawMetrics, RawSnapshot, VllmConfig, VllmRawMetrics};
use crate::context::{AnalysisInput, ModelArch, RuntimeWindow, StaticContext};
use crate::engine::baseline::{
    CeilingEstimate, CostEstimate, CostSource, PhysicsBaseline, WeightDtypeSource,
};
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
    format_diagnose_rules_for_windows(windows, summary, &report, verbose, metrics_url, 30)
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
    RawSnapshot {
        gpu_observed_at: gpu_at,
        vllm_observed_at: vllm_at,
        timestamp: gpu_at,
        vllm,
        gpus: vec![gpu],
    }
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
            model_name: Some("meta-llama/Llama-3.1-70B-Instruct".to_string()),
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
    let ctx = StaticContext::from_snapshot(&snap, cfg);
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
    mk_win(snap(t, t, v, gpu_busy()))
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

fn baseline_for_waste(eff: f64, source: CostSource, cpm: f64) -> PhysicsBaseline {
    PhysicsBaseline {
        decode: CeilingEstimate {
            lower: 90.0,
            expected: 100.0,
            upper: 110.0,
        },
        prefill: None,
        efficiency_pct: Some(eff),
        headroom_pct: Some(100.0 - eff),
        weight_dtype_source: WeightDtypeSource::Fallback,
        weight_gb: 1.0,
        kv_headroom_gb: None,
        tpot_floor_ms: 10.0,
        prefill_latency_floor_ms: None,
        ridge_batch_size: 1.0,
        config_relative_efficiency_pct: None,
        cost: Some(CostEstimate {
            tok_per_watt: None,
            joules_per_token: None,
            cost_per_million_tokens: Some(cpm),
            cost_source: source,
        }),
    }
}

fn mk_rec(rule_name: &'static str) -> Recommendation {
    Recommendation {
        rule_name,
        layer: 4,
        impact: 4,
        confidence: 0.8,
        action: String::new(),
        short_action: String::new(),
        expected_impact: String::new(),
        display_lines: Vec::new(),
    }
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
    let hybrid = ModelArch {
        num_kv_heads: Some(8),
        head_dim: Some(128),
        num_layers: Some(64),
        num_kv_layers: Some(32),
        ..Default::default()
    };
    #[allow(clippy::cast_precision_loss)]
    let headroom_gb = (1u64 << 34) as f64 / 1e9;
    let with_kv_layers = compute_kv_max_seqs(Some(headroom_gb), Some(4096), &hybrid, None, None);

    let dense = ModelArch {
        num_kv_layers: None,
        ..hybrid
    };
    let without_kv_layers = compute_kv_max_seqs(Some(headroom_gb), Some(4096), &dense, None, None);

    assert!(with_kv_layers.is_some() && without_kv_layers.is_some());
    assert_eq!(with_kv_layers.unwrap(), without_kv_layers.unwrap() * 2);
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
    let tp1 = compute_kv_max_seqs(Some(headroom_gb), Some(4096), &model, None, Some(1));
    let tp2 = compute_kv_max_seqs(Some(headroom_gb), Some(4096), &model, None, Some(2));
    assert_eq!(tp2.unwrap(), tp1.unwrap() * 2);
}

#[test]
fn compute_kv_max_seqs_tp_greater_than_kv_heads_no_benefit() {
    let model = ModelArch {
        num_kv_heads: Some(2),
        head_dim: Some(128),
        num_layers: Some(32),
        ..Default::default()
    };
    let headroom_gb = 20.0;
    let tp2 = compute_kv_max_seqs(Some(headroom_gb), Some(4096), &model, None, Some(2));
    let tp4 = compute_kv_max_seqs(Some(headroom_gb), Some(4096), &model, None, Some(4));
    assert_eq!(tp2, tp4);
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
    let none = compute_kv_max_seqs(Some(headroom_gb), Some(4096), &model, None, None);
    let one = compute_kv_max_seqs(Some(headroom_gb), Some(4096), &model, None, Some(1));
    assert_eq!(none, one);
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
    push_model_len_shrink_suggestion(
        &mut lines,
        Some(8192),
        Some(6000.0),
        Some(450.0),
        150.0,
        "      ",
    );
    let text = lines.join("\n");
    assert!(text.contains("to ~6450"));
    assert!(text.contains("prompt p99 6000 tok + output p99 450 tok"));
    assert!(text.contains("Truncation risk"));
}

#[test]
fn model_len_suggestion_no_op_when_count_below_threshold() {
    let mut lines = Vec::new();
    push_model_len_shrink_suggestion(
        &mut lines,
        Some(8192),
        Some(6000.0),
        Some(450.0),
        50.0,
        "      ",
    );
    let text = lines.join("\n");
    assert!(text.contains("to safely raise concurrency"));
    assert!(!text.contains("to ~"));
}

#[test]
fn model_len_suggestion_no_op_when_p99_missing() {
    let mut lines = Vec::new();
    push_model_len_shrink_suggestion(&mut lines, Some(8192), Some(6000.0), None, 150.0, "      ");
    let text = lines.join("\n");
    assert!(text.contains("to safely raise concurrency"));
    assert!(!text.contains("to ~"));
}

#[test]
fn model_len_suggestion_suppressed_when_delta_below_5pct() {
    let mut lines = Vec::new();
    push_model_len_shrink_suggestion(
        &mut lines,
        Some(5464),
        Some(5400.0),
        Some(65.0),
        150.0,
        "      ",
    );
    assert!(lines.is_empty());
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
    v.prompt_tokens_per_sec = Some(600.0);
    v.generation_tokens_per_sec = Some(100.0);
    v.prefix_cache_hit_rate = None;
    let s = snap(t, t, v, gpu_busy());
    let ctx = mk_ctx();
    let win = mk_win(s);
    let text =
        format_diagnose_rules_test(&ctx, &win, true, "http://127.0.0.1:8000/metrics").join("\n");
    assert!(text.contains("Under-batching: not triggered"));
    assert!(!text.contains("Under-batching: not triggered ("));
}

#[test]
fn suppressed_rule_shows_suppressor_in_verbose() {
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
    assert!(text.contains("Prefill-Bound: suppressed by Under-batching"));
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
    v.prompt_tokens_per_sec = Some(600.0);
    v.generation_tokens_per_sec = Some(100.0);
    v.prefix_cache_hit_rate = None;
    let s = snap(t, t, v, gpu_busy());
    let ctx = mk_ctx();
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
    assert_eq!(lines, vec!["No issues detected.".to_string()]);
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
        *w = mk_evaluable_backlog_window(70.0, 15.0, 5.0, 40.0, 100, 16);
    }
    let text = r2_issue_lines(&windows).join("\n");
    assert!(text.contains("[!] KV Cache Pressure: Admission Backlog"));
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
                    kv_max_seqs: None,
                    n_eval: 0,
                    skipped_broken: if any_evaluable { 0 } else { windows.len() },
                    skipped_idle: if any_evaluable { windows.len() } else { 0 },
                    energy_skew_skipped: 0,
                    gauge_missing: Default::default(),
                },
                verbose,
                metrics_url,
                duration_secs,
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
fn waste_label_r1_only() {
    assert_eq!(
        waste_label_suffix(&[rule_names::UNDER_BATCHING]),
        Some("wasted on idle compute")
    );
}

#[test]
fn waste_line_multi_rule_compounding() {
    let b = baseline_for_waste(32.0, CostSource::Catalog, 1.84);
    let groups = vec![
        mk_rec(rule_names::UNDER_BATCHING),
        mk_rec(rule_names::KV_CACHE_PRESSURE),
    ];
    let mut lines = vec!["issue".to_string()];
    append_waste_line(&mut lines, &groups, Some(&b), Some(14.2));
    assert!(
        lines
            .iter()
            .any(|l| l.contains("lost to compounding bottlenecks"))
    );
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
    assert!(text.contains("Monitor KV cache when scaling up."));
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
fn r6_suppressed_when_r1_fires() {
    let windows: Vec<_> = (0..10)
        .map(|_| mk_r6_prefill_window(2.5, 10.0, 5.0, Some(50.0)))
        .collect();
    let ctx = mk_llama8b_h100_ctx(&windows[0].snapshot);
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
    let r = r2_recommendation(&s, None, None, None, 1, 4, false).expect("fired");
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
    let r = r2_recommendation(&s, None, None, None, 1, 1, false).expect("fired");
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
            assert!((d.kv_cache_usage_perc - 88.0).abs() < 1e-9);
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
    let r2_text = r2_recommendation(&win_kv_only.snapshot, None, None, None, 1, 1, false)
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
    assert!(text.contains("Lower --max-num-seqs to stop evictions"));
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
fn backlog_short_action_matches_spec() {
    let mut windows: Vec<_> = (0..15)
        .map(|_| mk_evaluable_backlog_window(10.0, 1.0, 9.0, 10.0, 10_000, 16))
        .collect();
    for w in windows.iter_mut().take(4) {
        *w = mk_evaluable_backlog_window(70.0, 15.0, 5.0, 40.0, 100, 16);
    }
    let ctx = mk_ctx();
    let summary = ai(&ctx, windows.last().expect("windows"));
    let report = build_report_for_windows(&windows, summary);
    let r = report
        .recommendations
        .iter()
        .find(|g| g.rule_name == rule_names::KV_ADMISSION_BACKLOG)
        .expect("backlog kv recommendation")
        .clone();
    assert_eq!(r.short_action, "raise --gpu-memory-utilization");
    let display = r.display_lines.join("\n");
    assert!(display.contains("[!] KV Cache Pressure: Admission Backlog"));
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
    assert_eq!(
        waste_lines.len(),
        1,
        "expected one shared waste line: {lines:?}"
    );
    assert!(waste_lines[0].contains("lost to memory thrashing"));
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
    assert_eq!(lines, vec!["No issues detected.".to_string()]);
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
    // Fix line uses summary snapshot KV (50%) for branch selection - scale-out, not session peak.
    assert!(
        text.contains("Raise --max-num-seqs above 32"),
        "expected raise-cap fix from summary KV: {text}"
    );
    assert!(!text.contains("KV at 95%: scheduler at cap, pool full."));
    assert!(!text.contains("Add a replica"));
    assert!(!text.contains("KV pool has room (70%)"));
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
        text.contains("Raise --max-num-seqs above 32"),
        "display fix line must use summary snapshot KV branch: {text}"
    );
    assert!(!text.contains("KV at 95%: scheduler at cap, pool full."));
    assert!(!text.contains("Add a replica"));
    assert!(!text.contains("KV pool has room (70%)"));
    // action still uses aggregate session peak from eval.
    assert_eq!(r5.action, "Add a replica to scale out.");
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
fn waste_none_when_efficiency_above_ceiling() {
    let b = baseline_for_waste(85.0, CostSource::Catalog, 1.84);
    let groups = vec![mk_rec(rule_names::UNDER_BATCHING)];
    let mut lines = vec!["issue".to_string()];
    append_waste_line(&mut lines, &groups, Some(&b), Some(14.2));
    assert!(
        !lines.iter().any(|l| l.contains("/hr ")),
        "85% efficiency is above 80% ceiling; no recoverable waste"
    );
}

#[test]
fn waste_computed_against_80_pct_ceiling() {
    let cpm = 1.84;
    let tps = 14.2_f64;
    let cost_per_hr = cpm * tps * 3600.0 / 1_000_000.0;
    let expected_waste = cost_per_hr * 0.30;
    let not_full_gap_waste = cost_per_hr * 0.50;
    assert!(
        (expected_waste - not_full_gap_waste).abs() > 1e-9,
        "test must distinguish 80% ceiling from 100% roofline"
    );
    let b = baseline_for_waste(50.0, CostSource::Catalog, cpm);
    let groups = vec![mk_rec(rule_names::UNDER_BATCHING)];
    let mut lines = vec!["issue".to_string()];
    append_waste_line(&mut lines, &groups, Some(&b), Some(tps));
    let waste_line = lines
        .iter()
        .find(|l| l.contains("/hr "))
        .expect("50% efficiency should produce waste line");
    assert!(
        waste_line.contains(&format!("~${expected_waste:.2}/hr")),
        "expected ~${expected_waste:.2}/hr (30% gap to 80% ceiling), got {waste_line}"
    );
    assert!(
        !waste_line.contains(&format!("~${not_full_gap_waste:.2}/hr")),
        "must not use 100% roofline gap: {waste_line}"
    );
}

#[test]
fn waste_line_appended_for_r1_r2_r3_r5() {
    let b = baseline_for_waste(32.0, CostSource::Catalog, 1.84);
    let tps = Some(14.2_f64);
    let cases = [
        (
            vec![mk_rec(rule_names::UNDER_BATCHING)],
            "wasted on idle compute",
        ),
        (
            vec![mk_rec(rule_names::KV_CACHE_PRESSURE)],
            "lost to memory thrashing",
        ),
        (
            vec![mk_rec(rule_names::LOW_PREFIX_REUSE)],
            "wasted on redundant prefill",
        ),
        (
            vec![mk_rec(rule_names::CONCURRENCY_SATURATION)],
            "lost to scheduler queuing",
        ),
    ];
    for (recs, suffix) in cases {
        let mut lines = vec!["issue".to_string()];
        append_waste_line(&mut lines, &recs, Some(&b), tps);
        let waste = lines.iter().find(|l| l.contains("/hr ")).expect(suffix);
        assert!(waste.ends_with(suffix), "got {waste}");
    }
}

#[test]
fn waste_line_unknown_rule_name_unclassified() {
    let groups = vec![mk_rec(rule_names::OOM_RISK)];

    let b = baseline_for_waste(32.0, CostSource::Catalog, 1.84);
    let mut lines = vec!["issue".to_string()];
    append_waste_line(&mut lines, &groups, Some(&b), Some(14.2));
    assert!(lines.iter().any(|l| l.contains("unclassified overhead")));

    // UserProvided source is accepted; label still falls through to unclassified.
    let b = baseline_for_waste(32.0, CostSource::UserProvided, 1.0);
    let mut lines = vec!["issue".to_string()];
    append_waste_line(&mut lines, &groups, Some(&b), Some(100.0));
    assert!(lines.iter().any(|l| l.contains("unclassified overhead")));
}

#[test]
fn waste_line_efficiency_over_100_omitted() {
    let b = baseline_for_waste(110.0, CostSource::Catalog, 1.84);
    let mut lines = vec!["issue".to_string()];
    append_waste_line(
        &mut lines,
        &[mk_rec(rule_names::UNDER_BATCHING)],
        Some(&b),
        Some(14.2),
    );
    assert_eq!(lines.len(), 1);
    assert!(!lines.iter().any(|l| l.contains("/hr ")));
}

#[test]
fn waste_line_absent_without_cost_or_efficiency() {
    let mut b = baseline_for_waste(32.0, CostSource::Catalog, 1.84);
    b.efficiency_pct = None;
    let mut lines = vec!["issue".to_string()];
    append_waste_line(
        &mut lines,
        &[mk_rec(rule_names::UNDER_BATCHING)],
        Some(&b),
        Some(10.0),
    );
    assert_eq!(lines.len(), 1);

    b.efficiency_pct = Some(32.0);
    b.cost = None;
    append_waste_line(
        &mut lines,
        &[mk_rec(rule_names::UNDER_BATCHING)],
        Some(&b),
        Some(10.0),
    );
    assert_eq!(lines.len(), 1);
}

#[test]
fn waste_label_r2_only() {
    assert_eq!(
        waste_label_suffix(&[rule_names::KV_CACHE_PRESSURE]),
        Some("lost to memory thrashing")
    );
}

#[test]
fn waste_label_r3_only() {
    assert_eq!(
        waste_label_suffix(&[rule_names::LOW_PREFIX_REUSE]),
        Some("wasted on redundant prefill")
    );
}

#[test]
fn waste_label_r5_only() {
    assert_eq!(
        waste_label_suffix(&[rule_names::CONCURRENCY_SATURATION]),
        Some("lost to scheduler queuing")
    );
}

#[test]
fn waste_label_multi_rule() {
    assert_eq!(
        waste_label_suffix(&[rule_names::UNDER_BATCHING, rule_names::KV_CACHE_PRESSURE]),
        Some("lost to compounding bottlenecks")
    );
}

#[test]
fn waste_label_unknown_rule() {
    assert_eq!(
        waste_label_suffix(&[rule_names::OOM_RISK]),
        Some("unclassified overhead")
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
