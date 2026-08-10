pub mod baseline;
pub mod limiter;
mod rules;

use crate::context::{AnalysisInput, RuntimeWindow};

pub use baseline::{
    CeilingEstimate, CostEstimate, CostSource, KvCacheDtypeSource, PhysicsBaseline,
    WeightDtypeSource, baseline_missing_reason, catalog_model_weight_gb,
};
pub use rules::*;

/// Launch scope: single GPU, no tensor parallelism. TP machinery stays behind this.
pub const MULTI_GPU_TP: bool = false;

pub(crate) const MASSIVE_UNDERUTIL_THRESHOLD_PCT: f64 = 60.0;
/// Occupancy at or above this: server is config-capped, not traffic-starved. Skip traffic fallback.
const MASSIVE_UNDERUTIL_OCCUPANCY_CEILING: f64 = 0.75;
/// Shared confidence for MU variants that infer cause without a direct observation.
const MU_INFERRED_CONFIDENCE: f64 = 0.4;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GaugeMissingCounts {
    pub under_batching: usize,
    pub kv_cache_pressure: usize,
    pub low_prefix_reuse: usize,
    pub concurrency_saturation: usize,
}

/// Window-level skip/gap counts threaded into `Report` construction.
#[derive(Debug, Clone, Default)]
pub struct EvalSkipStats {
    pub skipped_broken: usize,
    pub skipped_idle: usize,
    /// Active windows excluded from energy/cost because GPU and vLLM clocks diverge.
    pub energy_skew_skipped: usize,
    /// Evaluable windows where a required gauge was absent (could not judge).
    pub gauge_missing: GaugeMissingCounts,
    pub limiter_evidence: Option<limiter::LimiterEvidence>,
}

#[derive(Debug, Clone)]
pub struct Report {
    pub baseline: Option<PhysicsBaseline>,
    pub recommendations: Vec<rules::Recommendation>,
    /// Rules that fired but were removed by layer filtering or the suppression table.
    pub suppressed_rules: Vec<(&'static str, &'static str)>,
    /// Full recommendation bodies removed by ME / layer filter, ranked like primaries.
    /// Rendered only when the loop reveals alternatives after a stuck fix.
    pub suppressed_recs: Vec<rules::Recommendation>,
    pub kv_max_seqs: Option<u32>,
    /// When labels and catalog hybrid facts both exist and disagree on state
    /// pages: `(catalog_pages, observed_pages)`. Verbose (-v) only. Default
    /// output is unaffected. Label uncertainty tracks the printed number's
    /// source, not the existence of disagreement between sources.
    pub catalog_state_mismatch: Option<(u64, u64)>,
    /// Evaluable window count. `engine::build_report_for_diagnose` gates MU
    /// inject on `ENGINE_MIN_PERSISTENT_WINDOWS`; stdout gates only the journey
    /// footer on the same threshold. `--json` (when emitted) should keep raw
    /// recommendations and let consumers apply the same `n_eval` gate.
    pub n_eval: usize,
    pub skipped_broken: usize,
    pub skipped_idle: usize,
    /// Active windows excluded from energy/cost because GPU and vLLM clocks diverge.
    pub energy_skew_skipped: usize,
    /// Evaluable windows where a required gauge was absent (could not judge).
    pub gauge_missing: GaugeMissingCounts,
    /// Run-level limiter evidence for no-rules "capped by" line.
    pub limiter_evidence: Option<limiter::LimiterEvidence>,
}

/// Multi-window diagnose report. Production always collects >= 15 windows (min duration 30s).
pub fn build_report_for_diagnose(windows: &[RuntimeWindow], input: AnalysisInput<'_>) -> Report {
    let mut report = rules::build_report_for_windows(windows, input);
    // MU is a traffic/efficiency judgment; below the sustained-load trust bar, skip.
    if report.n_eval >= ENGINE_MIN_PERSISTENT_WINDOWS {
        maybe_add_massive_underutilization(
            &mut report.recommendations,
            report.baseline.as_ref(),
            &input.window.snapshot,
            input.ctx.config.max_num_seqs,
            input.ctx.config.enable_chunked_prefill,
        );
    }
    report
}

fn maybe_add_massive_underutilization(
    recommendations: &mut Vec<rules::Recommendation>,
    baseline: Option<&PhysicsBaseline>,
    snapshot: &crate::collectors::RawSnapshot,
    config_max_num_seqs: Option<u32>,
    chunked_prefill_enabled: Option<bool>,
) {
    if !recommendations.is_empty() {
        return;
    }
    let Some(eff) = baseline.and_then(|b| b.efficiency_pct) else {
        return;
    };
    if eff >= MASSIVE_UNDERUTIL_THRESHOLD_PCT {
        return;
    }
    let running = snapshot
        .vllm
        .num_requests_running
        .filter(|v| v.is_finite() && *v >= 0.0);
    let waiting = snapshot
        .vllm
        .num_requests_waiting
        .filter(|v| v.is_finite() && *v >= 0.0);
    let max_num_seqs = snapshot
        .vllm
        .max_num_seqs
        .or(config_max_num_seqs)
        .filter(|&n| n > 0);
    let kv = snapshot
        .vllm
        .kv_cache_usage_perc
        .filter(|v| v.is_finite() && *v >= 0.0);

    // Occupancy against real capacity, not the knob. Same three walls R1 uses
    // (src/engine/rules/r1_under_batching.rs:99). Near-cap is R5 territory.
    if let Some(max_n) = max_num_seqs {
        let (effective_max, _) = rules::effective_max_and_binder(
            max_n,
            baseline.map(|b| b.ridge_batch_size),
            rules::usable_kv_concurrency(snapshot),
        );
        if running.is_some_and(|run| {
            run > 0.0 && run / effective_max >= MASSIVE_UNDERUTIL_OCCUPANCY_CEILING
        }) {
            return;
        }
    }

    // KV near full means memory is the wall, not the client. Never call this
    // starvation, whatever the queue looks like.
    if rules::kv_near_full(snapshot) {
        return;
    }

    let variant = match waiting {
        None => rules::MuVariant::GaugeMissing,
        Some(w) if w < 1.0 => rules::MuVariant::Starved,
        Some(_) => rules::MuVariant::BlockedAdmission { kv_pct: kv },
    };

    let confidence = match &variant {
        rules::MuVariant::Starved => 0.7,
        rules::MuVariant::BlockedAdmission { .. } | rules::MuVariant::GaugeMissing => {
            MU_INFERRED_CONFIDENCE
        }
    };

    let chunked = chunked_prefill_enabled.or(snapshot.vllm.cache_config.enable_chunked_prefill);

    recommendations.push(rules::Recommendation {
        rule_name: rule_names::MASSIVE_UNDERUTILIZATION,
        // Sentinel: post-DAG inject; not a DAG layer. Layer-min filtering does not apply.
        layer: 0,
        impact: 5,
        confidence,
        display_lines: rules::mu_diagnose_lines(
            eff,
            running,
            waiting,
            max_num_seqs,
            variant,
            chunked,
        ),
        terminal: false,
    });
}

pub fn aggregate_prefix_hit_rate_for_diagnose(windows: &[RuntimeWindow]) -> Option<f64> {
    rules::aggregate_prefix_hit_rate_for_windows(windows)
}

#[cfg(test)]
mod build_report_tests {
    use super::*;
    use crate::collectors::{GpuRawMetrics, RawSnapshot, VllmRawMetrics};
    use crate::context::{RuntimeWindow, StaticContext};
    use std::time::SystemTime;

    #[test]
    fn dag_layer2_surfaces_highest_scoring_rule_in_layer() {
        use crate::collectors::VllmConfig;

        let t = SystemTime::UNIX_EPOCH;
        let v = VllmRawMetrics {
            model_name: Some("meta-llama/Llama-3.1-8B-Instruct".to_string()),
            num_requests_running: Some(3.1),
            num_requests_waiting: Some(0.0),
            max_num_seqs: Some(256),
            kv_cache_usage_perc: Some(89.0),
            num_preemptions_per_sec: Some(0.05),
            tpot_ms: Some(35.0),
            generation_tokens_per_sec: Some(30.0),
            window_duration_secs: Some(2.0),
            ..Default::default()
        };
        let g = GpuRawMetrics {
            gpu_name: Some("NVIDIA H100 80GB HBM3".to_string()),
            gpu_util_pct: Some(58.0),
            ..Default::default()
        };
        let s = RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: v,
            gpus: vec![g],
            host_memory: None,
        };

        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let ctx = StaticContext::from_snapshot(&s, cfg);
        let win = RuntimeWindow::from_snapshot(s);
        let report = diagnose_windows(&ctx, &win);
        assert_eq!(report.recommendations.len(), 1);
        assert_eq!(
            report.recommendations[0].rule_name,
            rule_names::KV_CACHE_PRESSURE
        );
        assert!(
            !report
                .recommendations
                .iter()
                .any(|r| r.rule_name == rule_names::UNDER_BATCHING)
        );
    }

    #[test]
    fn dag_suppression_table_oom_drops_kv_pressure() {
        use crate::collectors::VllmConfig;

        let t = SystemTime::UNIX_EPOCH;
        let v = VllmRawMetrics {
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
        };
        let g = GpuRawMetrics {
            gpu_name: Some("NVIDIA H100 80GB HBM3".to_string()),
            vram_total_mb: Some(80 * 1024),
            gpu_util_pct: Some(58.0),
            ..Default::default()
        };
        let s = RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: v,
            gpus: vec![g],
            host_memory: None,
        };

        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let mut ctx = StaticContext::from_snapshot(&s, cfg);
        // Synthetic oversized model keeps this DAG test independent of the
        // single-GPU launch catalog's supported model set.
        ctx.model.param_count = Some(70_000_000_000);
        let win = RuntimeWindow::from_snapshot(s);
        let report = diagnose_windows(&ctx, &win);

        assert!(
            report
                .recommendations
                .iter()
                .any(|r| r.rule_name == rule_names::OOM_RISK)
        );
        assert!(
            !report
                .recommendations
                .iter()
                .any(|r| r.rule_name == rule_names::KV_CACHE_PRESSURE)
        );
    }

    #[test]
    fn dag_layer4_surfaces_when_no_higher_layer_fires() {
        use crate::collectors::VllmConfig;

        let t = SystemTime::UNIX_EPOCH;
        let v = VllmRawMetrics {
            model_name: Some("meta-llama/Llama-3.1-8B-Instruct".to_string()),
            num_requests_running: Some(3.1),
            num_requests_waiting: Some(0.0),
            max_num_seqs: Some(256),
            kv_cache_usage_perc: Some(50.0),
            tpot_ms: Some(35.0),
            generation_tokens_per_sec: Some(30.0),
            window_duration_secs: Some(2.0),
            ..Default::default()
        };
        let g = GpuRawMetrics {
            gpu_name: Some("NVIDIA H100 80GB HBM3".to_string()),
            gpu_util_pct: Some(58.0),
            ..Default::default()
        };
        let s = RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: v,
            gpus: vec![g],
            host_memory: None,
        };

        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let ctx = StaticContext::from_snapshot(&s, cfg);
        let win = RuntimeWindow::from_snapshot(s);
        let report = diagnose_windows(&ctx, &win);
        assert_eq!(report.recommendations.len(), 1);
        assert_eq!(
            report.recommendations[0].rule_name,
            rule_names::UNDER_BATCHING
        );
    }

    fn starved_no_rules_fixture() -> (StaticContext, RuntimeWindow) {
        use crate::collectors::VllmConfig;

        let t = SystemTime::UNIX_EPOCH;
        let v = VllmRawMetrics {
            model_name: Some("meta-llama/Llama-3.1-8B-Instruct".to_string()),
            num_requests_running: Some(64.0),
            num_requests_waiting: Some(0.0),
            max_num_seqs: Some(256),
            kv_cache_usage_perc: Some(10.0),
            prefix_cache_hit_rate: Some(0.5),
            generation_tokens_per_sec: Some(30.0),
            request_success_per_sec: Some(10.0),
            window_duration_secs: Some(2.0),
            ..Default::default()
        };
        let g = GpuRawMetrics {
            gpu_name: Some("NVIDIA H100 80GB HBM3".to_string()),
            gpu_util_pct: Some(50.0),
            ..Default::default()
        };
        let s = RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: v,
            gpus: vec![g],
            host_memory: None,
        };

        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let ctx = StaticContext::from_snapshot(&s, cfg);
        let win = RuntimeWindow::from_snapshot(s);
        (ctx, win)
    }

    fn sustained_windows(win: &RuntimeWindow, n: usize) -> Vec<RuntimeWindow> {
        (0..n).map(|_| win.clone()).collect()
    }

    /// Multi-window engine path without MU inject (matches production aggregate eval).
    fn diagnose_windows(ctx: &StaticContext, win: &RuntimeWindow) -> Report {
        let windows = sustained_windows(win, ENGINE_MIN_PERSISTENT_WINDOWS);
        let input = AnalysisInput::new(ctx, &windows[0]);
        rules::build_report_for_windows(&windows, input)
    }

    fn diagnose_mu(ctx: &StaticContext, win: &RuntimeWindow, n_windows: usize) -> Report {
        let windows = sustained_windows(win, n_windows);
        let input = AnalysisInput::new(ctx, &windows[0]);
        build_report_for_diagnose(&windows, input)
    }

    fn massive_underutilization_count(report: &Report) -> usize {
        report
            .recommendations
            .iter()
            .filter(|r| r.rule_name == rule_names::MASSIVE_UNDERUTILIZATION)
            .count()
    }

    fn mu_rec(report: &Report) -> &rules::Recommendation {
        report
            .recommendations
            .iter()
            .find(|r| r.rule_name == rule_names::MASSIVE_UNDERUTILIZATION)
            .expect("MU recommendation")
    }

    #[test]
    fn build_report_for_windows_does_not_add_massive_underutilization() {
        let (ctx, win) = starved_no_rules_fixture();
        let report = diagnose_windows(&ctx, &win);
        assert_eq!(
            massive_underutilization_count(&report),
            0,
            "windows path must not inject the diagnose-only safety net"
        );
        let eff = report
            .baseline
            .as_ref()
            .and_then(|b| b.efficiency_pct)
            .expect("baseline efficiency");
        assert!(
            eff < MASSIVE_UNDERUTIL_THRESHOLD_PCT,
            "fixture should be under-utilized: {eff}%"
        );
        assert!(
            report.recommendations.is_empty(),
            "fixture should fire no rules before safety net: {:?}",
            report
                .recommendations
                .iter()
                .map(|r| r.rule_name)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn build_report_for_diagnose_adds_massive_underutilization_once() {
        let (ctx, win) = starved_no_rules_fixture();
        let report = diagnose_mu(&ctx, &win, ENGINE_MIN_PERSISTENT_WINDOWS);
        assert_eq!(
            massive_underutilization_count(&report),
            1,
            "diagnose path should inject safety net exactly once"
        );
        let mu = mu_rec(&report);
        assert!((mu.confidence - 0.7).abs() < 1e-9);
        let text = mu.display_lines.join("\n");
        assert!(text.contains("Requests  64 running, 0 waiting  (server not saturated)"));
        assert!(text.contains("Server is under-fed"));
        assert!(!text.contains("GPU is idle"));
        assert!(text.contains("    Cause:"));
        assert!(text.contains("      Server is under-fed"));
    }

    #[test]
    fn massive_underutilization_blocked_admission_when_waiting_with_free_seats() {
        let (ctx, mut win) = starved_no_rules_fixture();
        win.snapshot.vllm.num_requests_running = Some(10.0);
        win.snapshot.vllm.num_requests_waiting = Some(5.0);
        win.snapshot.vllm.kv_cache_usage_perc = Some(10.0);
        let report = diagnose_mu(&ctx, &win, ENGINE_MIN_PERSISTENT_WINDOWS);
        assert_eq!(massive_underutilization_count(&report), 1);
        let mu = mu_rec(&report);
        assert!((mu.confidence - MU_INFERRED_CONFIDENCE).abs() < 1e-9);
        let text = mu.display_lines.join("\n");
        assert!(text.contains("Requests  10 running, 5 waiting  (246 of 256 seats free)"));
        assert!(text.contains("seats are free and KV cache at 10% (low)"));
        assert!(!text.contains("KV cache is low."));
        assert!(text.contains("Scheduler admission is blocked"));
        assert!(text.contains("Raise --max-num-batched-tokens."));
        assert!(text.contains("Confirm chunked prefill is enabled"));
        assert!(!text.contains("or enable chunked prefill"));
        assert!(!text.contains("Enable chunked prefill (--enable-chunked-prefill)."));
        assert!(!text.contains("server not saturated"));
        assert!(text.contains("Confidence: Low (cause inferred, token budget not observed)"));
    }

    #[test]
    fn massive_underutilization_blocked_admission_when_kv_gauge_missing() {
        let (ctx, mut win) = starved_no_rules_fixture();
        win.snapshot.vllm.num_requests_running = Some(10.0);
        win.snapshot.vllm.num_requests_waiting = Some(5.0);
        win.snapshot.vllm.kv_cache_usage_perc = None;
        let report = diagnose_mu(&ctx, &win, ENGINE_MIN_PERSISTENT_WINDOWS);
        assert_eq!(massive_underutilization_count(&report), 1);
        let mu = mu_rec(&report);
        assert!((mu.confidence - MU_INFERRED_CONFIDENCE).abs() < 1e-9);
        let text = mu.display_lines.join("\n");
        assert!(text.contains("Requests queue while seats are free."));
        assert!(!text.contains("KV cache is low"));
        assert!(text.contains("Confidence: Low (cause inferred; KV gauge unavailable)"));
    }

    #[test]
    fn massive_underutilization_fires_when_waiting_blip_below_one() {
        let (ctx, mut win) = starved_no_rules_fixture();
        win.snapshot.vllm.num_requests_waiting = Some(0.01);
        let report = diagnose_mu(&ctx, &win, ENGINE_MIN_PERSISTENT_WINDOWS);
        assert_eq!(
            massive_underutilization_count(&report),
            1,
            "waiting < 1.0 must fire Starved"
        );
        let text = mu_rec(&report).display_lines.join("\n");
        assert!(text.contains("server not saturated"));
        assert!(text.contains("Server is under-fed"));
    }

    #[test]
    fn massive_underutilization_suppressed_when_kv_pressure_shape() {
        // r2 must NOT fire here or this test proves nothing.
        let (ctx, base) = starved_no_rules_fixture();
        let mut high = base.clone();
        high.snapshot.vllm.num_requests_running = Some(10.0);
        high.snapshot.vllm.num_requests_waiting = Some(5.0);
        high.snapshot.vllm.kv_cache_usage_perc = Some(95.0);
        let mut mid = high.clone();
        mid.snapshot.vllm.kv_cache_usage_perc = Some(80.0);
        // 95, 95, 80 → r2 fires in 2 windows (< ENGINE_MIN_PERSISTENT_WINDOWS);
        // summary avg kv = 90 >= 88 so MU's KV veto (kv_near_full) silences.
        let windows = vec![high.clone(), high.clone(), mid];
        let mut summary = high.clone();
        summary.snapshot.vllm.kv_cache_usage_perc = Some(90.0);
        let input = AnalysisInput::new(&ctx, &summary);
        let report = build_report_for_diagnose(&windows, input);
        assert!(
            !report
                .recommendations
                .iter()
                .any(|r| r.rule_name == rule_names::KV_CACHE_PRESSURE),
            "r2 must be absent: silence must come from MU's KV veto"
        );
        assert_eq!(
            massive_underutilization_count(&report),
            0,
            "aggregate kv >= r2 threshold with free seats: MU must stay silent"
        );
    }

    #[test]
    fn massive_underutilization_suppressed_at_high_occupancy() {
        let (mut ctx, mut win) = starved_no_rules_fixture();
        win.snapshot.vllm.num_requests_running = Some(200.0);
        win.snapshot.vllm.num_requests_waiting = Some(5.0);
        win.snapshot.vllm.max_num_seqs = Some(256);
        ctx.config.max_num_seqs = Some(256);
        let report = diagnose_mu(&ctx, &win, ENGINE_MIN_PERSISTENT_WINDOWS);
        assert_eq!(
            massive_underutilization_count(&report),
            0,
            "high occupancy is R5 territory"
        );
    }

    #[test]
    fn massive_underutilization_gauge_missing_waiting() {
        let (ctx, mut win) = starved_no_rules_fixture();
        win.snapshot.vllm.num_requests_waiting = None;
        let report = diagnose_mu(&ctx, &win, ENGINE_MIN_PERSISTENT_WINDOWS);
        assert_eq!(massive_underutilization_count(&report), 1);
        let mu = mu_rec(&report);
        assert!((mu.confidence - MU_INFERRED_CONFIDENCE).abs() < 1e-9);
        let text = mu.display_lines.join("\n");
        assert!(text.contains("waiting gauge unavailable"));
        assert!(text.contains("Confidence: Low (waiting gauge unavailable)"));
        assert!(!text.contains("server not saturated"));
    }

    #[test]
    fn massive_underutilization_skipped_when_sparse_n_eval() {
        let (ctx, win) = starved_no_rules_fixture();
        let report = diagnose_mu(&ctx, &win, 2);
        assert_eq!(report.n_eval, 2);
        assert_eq!(
            massive_underutilization_count(&report),
            0,
            "MU must not inject below ENGINE_MIN_PERSISTENT_WINDOWS"
        );
    }

    #[test]
    fn massive_underutilization_silent_when_kv_peak_high_with_empty_queue() {
        // Journey iteration 1 KV shape on the R1-silent fixture seats (64/256).
        // Do not set ctx.config.max_num_seqs. That flips R1 to known-GPU and
        // R1 owns the window before MU reads KV.
        let (ctx, mut win) = starved_no_rules_fixture();
        win.snapshot.vllm.num_requests_waiting = Some(0.0);
        win.snapshot.vllm.kv_cache_usage_perc = Some(73.4);
        win.snapshot.vllm.kv_cache_peak_perc = Some(92.5);
        let report = diagnose_mu(&ctx, &win, ENGINE_MIN_PERSISTENT_WINDOWS);
        assert!(
            report.recommendations.is_empty(),
            "MU must be the only possible output here: {:?}",
            report
                .recommendations
                .iter()
                .map(|r| r.rule_name)
                .collect::<Vec<_>>()
        );
        let limiter_report = diagnose_windows(&ctx, &win);
        let ev = limiter_report.limiter_evidence.expect("limiter evidence");
        assert_eq!(
            limiter::identify(&ev).verdict,
            Some(limiter::LimiterVerdict::Known(
                limiter::PrimaryLimiter::Traffic
            )),
            "peak burst must not alone trigger Capacity; mean 73.4% is below the 80% bar"
        );
        let line = limiter::limiter_line(&ev).expect("limiter line");
        assert!(line.contains("Capped by traffic"));
        assert!(!line.contains("Capped by memory"));
    }

    #[test]
    fn limiter_capacity_when_mean_above_bar_shows_avg_and_peak() {
        let (ctx, mut win) = starved_no_rules_fixture();
        win.snapshot.vllm.kv_cache_usage_perc = Some(85.0);
        win.snapshot.vllm.kv_cache_peak_perc = Some(92.0);
        let report = diagnose_windows(&ctx, &win);
        let ev = report.limiter_evidence.expect("limiter evidence");
        assert_eq!(
            limiter::identify(&ev).verdict,
            Some(limiter::LimiterVerdict::Known(
                limiter::PrimaryLimiter::Capacity
            ))
        );
        let line = limiter::limiter_line(&ev).expect("limiter line");
        assert_eq!(
            line,
            "Capped by memory: KV cache at 85% avg, 92% peak (R2 fires at 88%). Concurrency cannot grow further on this pool."
        );
    }

    #[test]
    fn massive_underutilization_silent_when_kv_avg_high_with_empty_queue() {
        let (ctx, mut win) = starved_no_rules_fixture();
        win.snapshot.vllm.num_requests_waiting = Some(0.0);
        win.snapshot.vllm.kv_cache_usage_perc = Some(90.0);
        win.snapshot.vllm.kv_cache_peak_perc = None;
        let report = diagnose_mu(&ctx, &win, ENGINE_MIN_PERSISTENT_WINDOWS);
        assert!(
            report.recommendations.is_empty(),
            "MU must be the only possible output here: {:?}",
            report
                .recommendations
                .iter()
                .map(|r| r.rule_name)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn massive_underutilization_fires_when_kv_below_veto() {
        let (ctx, mut win) = starved_no_rules_fixture();
        win.snapshot.vllm.num_requests_waiting = Some(0.0);
        win.snapshot.vllm.kv_cache_usage_perc = Some(87.9);
        win.snapshot.vllm.kv_cache_peak_perc = Some(87.9);
        let report = diagnose_mu(&ctx, &win, ENGINE_MIN_PERSISTENT_WINDOWS);
        assert_eq!(report.recommendations.len(), 1);
        assert_eq!(
            report.recommendations[0].rule_name,
            rule_names::MASSIVE_UNDERUTILIZATION
        );
        let text = mu_rec(&report).display_lines.join("\n");
        assert!(text.contains("server not saturated"));
    }

    #[test]
    fn massive_underutilization_silent_when_waiting_with_peak_above_veto() {
        // waiting = 1: at/above 1 so pre-veto took BlockedAdmission; below R2's
        // w > 2 and R5's waiting >= 2 so neither owns the window. Peak 95 hits
        // kv_near_full → silence is the veto alone.
        let (ctx, mut win) = starved_no_rules_fixture();
        win.snapshot.vllm.num_requests_waiting = Some(1.0);
        win.snapshot.vllm.kv_cache_usage_perc = Some(70.0);
        win.snapshot.vllm.kv_cache_peak_perc = Some(95.0);
        let report = diagnose_mu(&ctx, &win, ENGINE_MIN_PERSISTENT_WINDOWS);
        assert!(
            report.recommendations.is_empty(),
            "MU must be the only possible output here: {:?}",
            report
                .recommendations
                .iter()
                .map(|r| r.rule_name)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn massive_underutilization_suppressed_when_kv_concurrency_cap_binds() {
        // Occupancy vs memory wall, not the knob. Keep KV gauges low so the
        // veto cannot mute. Leave ctx.config alone so R1 stays unknown-GPU.
        let (ctx, mut win) = starved_no_rules_fixture();
        win.snapshot.vllm.num_requests_running = Some(14.0);
        win.snapshot.vllm.num_requests_waiting = Some(0.0);
        win.snapshot.vllm.max_num_seqs = Some(345);
        win.snapshot.vllm.kv_cache_usage_perc = Some(10.0);
        win.snapshot.vllm.kv_cache_peak_perc = None;
        win.snapshot.vllm.cache_config.kv_cache_max_concurrency = Some(1.06);
        let report = diagnose_mu(&ctx, &win, ENGINE_MIN_PERSISTENT_WINDOWS);
        assert!(
            report.recommendations.is_empty(),
            "MU must be the only possible output here: {:?}",
            report
                .recommendations
                .iter()
                .map(|r| r.rule_name)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn massive_underutilization_unaffected_when_no_wall_reported() {
        // Same starved fixture as the MU fire test: unknown-GPU R1 bar is 25%,
        // and 64/256 sits on that line so R1 stays silent. Do not raise
        // max_num_seqs / drop running, which drops occupancy under 25% and R1
        // owns the window, so MU never runs (false fail for Fix 2).
        let (ctx, mut win) = starved_no_rules_fixture();
        win.snapshot.vllm.cache_config.kv_cache_max_concurrency = None;
        let report = diagnose_mu(&ctx, &win, ENGINE_MIN_PERSISTENT_WINDOWS);
        assert_eq!(report.recommendations.len(), 1);
        assert_eq!(
            report.recommendations[0].rule_name,
            rule_names::MASSIVE_UNDERUTILIZATION,
            "no concurrency wall: Fix 2 is inert; MU still fires"
        );
    }

    #[test]
    fn contradicted_kv_cap_does_not_bind_mu_occupancy_wall() {
        // Peak 64 vs floor(1.06)=1: usable declines. Occupancy must not be 64/1.
        // Keep fixture seats (64/256) so unknown-GPU R1 stays on the 25% line.
        let (ctx, mut win) = starved_no_rules_fixture();
        win.snapshot.vllm.num_requests_running_peak = Some(64.0);
        win.snapshot.vllm.kv_cache_usage_perc = Some(30.0);
        win.snapshot.vllm.kv_cache_peak_perc = None;
        win.snapshot.vllm.cache_config.kv_cache_max_concurrency = Some(1.06);
        let ridge = report_ridge(&ctx, &win);
        let (effective_max, wall) = rules::effective_max_and_binder(
            win.snapshot.vllm.max_num_seqs.unwrap_or(256),
            ridge,
            rules::usable_kv_concurrency(&win.snapshot),
        );
        assert!(
            (effective_max - 1.0).abs() > 0.5,
            "contradicted cap must not bind effective_max to 1, got {effective_max} wall={wall:?}"
        );
        assert!(rules::usable_kv_concurrency(&win.snapshot).is_none());
        let report = diagnose_mu(&ctx, &win, ENGINE_MIN_PERSISTENT_WINDOWS);
        assert_eq!(report.recommendations.len(), 1);
        assert_eq!(
            report.recommendations[0].rule_name,
            rule_names::MASSIVE_UNDERUTILIZATION,
            "contradicted 1-seat wall must not silence MU: {:?}",
            report
                .recommendations
                .iter()
                .map(|r| r.rule_name)
                .collect::<Vec<_>>()
        );
    }

    fn report_ridge(ctx: &StaticContext, win: &RuntimeWindow) -> Option<f64> {
        let input = AnalysisInput::new(ctx, win);
        crate::engine::baseline::compute(&input).map(|b| b.ridge_batch_size)
    }
}
