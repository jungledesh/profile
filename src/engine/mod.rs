pub mod baseline;
pub mod limiter;
mod rules;

use crate::context::{AnalysisInput, RuntimeWindow};

pub use baseline::{
    CeilingEstimate, CostEstimate, CostSource, KvCacheDtypeSource, PhysicsBaseline,
    WeightDtypeSource, catalog_model_weight_gb,
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
    /// Capacity value R2 prescribed this iteration (`≤N`). Used for self-grade after restart.
    pub prescribed_kv_capacity: Option<u32>,
    /// When labels and catalog hybrid facts both exist and disagree on state
    /// pages: `(catalog_pages, observed_pages)`. Verbose (-v) only. Default
    /// output is unaffected. Label uncertainty tracks the printed number's
    /// source, not the existence of disagreement between sources.
    pub catalog_state_mismatch: Option<(u64, u64)>,
    /// When allocator bytes and the 3 GB estimate both exist:
    /// `(observed_budget_bytes, estimated_budget_bytes)`. Verbose only.
    pub memory_budget_self_grade: Option<(u64, u64)>,
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
        );
    }
    report
}

fn maybe_add_massive_underutilization(
    recommendations: &mut Vec<rules::Recommendation>,
    baseline: Option<&PhysicsBaseline>,
    snapshot: &crate::collectors::RawSnapshot,
    config_max_num_seqs: Option<u32>,
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

    // Near-cap is R5 territory (occupancy ≈ 1.0 always clears 0.75).
    if let Some(max_n) = max_num_seqs
        && running.is_some_and(|run| {
            run > 0.0 && run / f64::from(max_n) >= MASSIVE_UNDERUTIL_OCCUPANCY_CEILING
        })
    {
        return;
    }

    let variant = match waiting {
        None => rules::MuVariant::GaugeMissing,
        Some(w) if w < 1.0 => rules::MuVariant::Starved,
        Some(_) if kv.is_some_and(|k| k >= rules::KV_CACHE_PRESSURE_MIN_PERC) => {
            // r2's shape; it did not sustain significance, don't mislabel as MU.
            return;
        }
        Some(_) => rules::MuVariant::BlockedAdmission { kv_pct: kv },
    };

    let confidence = match &variant {
        rules::MuVariant::Starved => 0.7,
        rules::MuVariant::BlockedAdmission { .. } | rules::MuVariant::GaugeMissing => {
            MU_INFERRED_CONFIDENCE
        }
    };

    recommendations.push(rules::Recommendation {
        rule_name: rule_names::MASSIVE_UNDERUTILIZATION,
        // Sentinel: post-DAG inject; not a DAG layer. Layer-min filtering does not apply.
        layer: 0,
        impact: 5,
        confidence,
        display_lines: rules::mu_diagnose_lines(eff, running, waiting, max_num_seqs, variant),
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
        assert!(text.contains("Raise --max-num-batched-tokens"));
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
        // aggregate mean kv = 90 >= 88 so MU's kv-pressure silence branch runs.
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
            "r2 must be absent: silence must come from MU's kv branch"
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
}
