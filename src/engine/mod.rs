pub mod baseline;
pub mod limiter;
mod rules;

#[cfg(test)]
use crate::collectors::{effective_tensor_parallel, window_is_evaluable};
use crate::context::{AnalysisInput, RuntimeWindow};

pub use baseline::{CeilingEstimate, CostEstimate, CostSource, PhysicsBaseline, WeightDtypeSource};
pub use rules::*;

const MASSIVE_UNDERUTIL_THRESHOLD_PCT: f64 = 60.0;
/// Occupancy at or above this: server is config-capped, not traffic-starved. Skip traffic fallback.
const MASSIVE_UNDERUTIL_OCCUPANCY_CEILING: f64 = 0.75;

#[derive(Debug, Clone)]
pub struct Report {
    pub baseline: Option<PhysicsBaseline>,
    pub groups: Vec<IssueGroup>,
    /// Rules that fired but were removed by layer filtering or the suppression table.
    pub suppressed_rules: Vec<(&'static str, &'static str)>,
    pub kv_max_seqs: Option<u32>,
    pub n_eval: usize,
    pub skipped: usize,
}

/// Multi-window diagnose report. Production always collects >= 15 windows (min duration 30s).
pub fn build_report_for_diagnose(windows: &[RuntimeWindow], input: AnalysisInput<'_>) -> Report {
    let mut report = rules::build_report_for_windows(windows, input);
    maybe_add_massive_underutilization(
        &mut report.groups,
        report.baseline.as_ref(),
        &input.window.snapshot,
        input.ctx.config.max_num_seqs,
    );
    report
}

/// Single-window path: test-only. Production always uses `build_report_for_windows`.
#[cfg(test)]
pub(crate) fn build_report(input: AnalysisInput<'_>) -> Report {
    let baseline = baseline::compute(&input);
    let snapshot = &input.window.snapshot;

    if !window_is_evaluable(snapshot) {
        return Report {
            baseline,
            groups: Vec::new(),
            suppressed_rules: Vec::new(),
            kv_max_seqs: None,
            n_eval: 0,
            skipped: 1,
        };
    }

    let n_eval = 1;
    let r2_fired = usize::from(matches!(
        rules::rule2_kv_cache_pressure(snapshot),
        rules::Rule2Outcome::Fired(_)
    ));
    let kv_max_seqs = rules::compute_kv_max_seqs(
        baseline.as_ref().and_then(|b| b.kv_headroom_gb),
        input.ctx.config.max_model_len,
        &input.ctx.model,
        input.ctx.config.kv_cache_dtype.as_deref(),
        effective_tensor_parallel(
            input.ctx.config.tensor_parallel_size,
            input.window.snapshot.collected_gpu_count(),
        ),
    );

    let mut recs: Vec<Recommendation> = Vec::new();
    if let Some(r) = rules::r4_recommendation(
        baseline.as_ref().and_then(|b| b.kv_headroom_gb),
        effective_tensor_parallel(
            input.ctx.config.tensor_parallel_size,
            input.window.snapshot.collected_gpu_count(),
        ),
        baseline.as_ref().map(|b| b.weight_gb),
        input.ctx.gpu.vram_gb,
        input.ctx.config.gpu_memory_utilization,
        baseline
            .as_ref()
            .map(|b| b.weight_dtype_source)
            .unwrap_or(WeightDtypeSource::Fallback),
    ) {
        recs.push(r);
    }
    if let Some(r) = rules::r2_recommendation(
        snapshot,
        input.ctx.config.max_model_len,
        baseline.as_ref().and_then(|b| b.kv_headroom_gb),
        kv_max_seqs,
        r2_fired,
        n_eval,
        input.ctx.fp8_compiler_available,
    ) {
        recs.push(r);
    }
    if let Some(r) = rules::r5_recommendation(
        snapshot,
        snapshot
            .vllm
            .kv_cache_peak_perc
            .or(snapshot.vllm.kv_cache_usage_perc),
        input.ctx.config.max_num_seqs,
        input.ctx.config.max_model_len,
        kv_max_seqs,
    ) {
        recs.push(r);
    }
    if let Some(r) = rules::r1_recommendation(rules::R1EvalInput {
        snapshot,
        config_max_num_seqs: input.ctx.config.max_num_seqs,
        efficiency_pct: baseline.as_ref().and_then(|b| b.efficiency_pct),
        config_relative_efficiency_pct: baseline
            .as_ref()
            .and_then(|b| b.config_relative_efficiency_pct),
        prompt_tokens_per_sec: snapshot.vllm.prompt_tokens_per_sec,
        generation_tokens_per_sec: snapshot.vllm.generation_tokens_per_sec,
        prefix_cache_hit_rate: snapshot.vllm.prefix_cache_hit_rate,
        ridge_batch_size: baseline.as_ref().map(|b| b.ridge_batch_size),
    }) {
        recs.push(r);
    }
    if let Some(r) = rules::r3_recommendation(snapshot) {
        recs.push(r);
    }
    if let Some(r) = rules::r6_recommendation(rules::PrefillBoundEvalInput {
        prompt_tokens_per_sec: snapshot.vllm.prompt_tokens_per_sec,
        generation_tokens_per_sec: snapshot.vllm.generation_tokens_per_sec,
        decode_efficiency_pct: baseline.as_ref().and_then(|b| b.efficiency_pct),
        tpot_ms: snapshot.vllm.tpot_ms,
        tpot_floor_ms: baseline.as_ref().map(|b| b.tpot_floor_ms),
        prefix_cache_hit_rate: snapshot.vllm.prefix_cache_hit_rate,
        snapshot,
        chunked_prefill_enabled: input.ctx.config.enable_chunked_prefill,
    }) {
        recs.push(r);
    }

    rules::finalize_report_groups(recs, baseline, kv_max_seqs, 1, 0)
}

fn maybe_add_massive_underutilization(
    groups: &mut Vec<IssueGroup>,
    baseline: Option<&PhysicsBaseline>,
    snapshot: &crate::collectors::RawSnapshot,
    config_max_num_seqs: Option<u32>,
) {
    if !groups.is_empty() {
        return;
    }
    let Some(eff) = baseline.and_then(|b| b.efficiency_pct) else {
        return;
    };
    if eff >= MASSIVE_UNDERUTIL_THRESHOLD_PCT {
        return;
    }
    if let (Some(run), Some(max_n)) = (
        snapshot
            .vllm
            .num_requests_running
            .filter(|v| v.is_finite() && *v > 0.0),
        snapshot
            .vllm
            .max_num_seqs
            .or(config_max_num_seqs)
            .filter(|&n| n > 0),
    ) {
        let occupancy = run / f64::from(max_n);
        if occupancy >= MASSIVE_UNDERUTIL_OCCUPANCY_CEILING {
            return;
        }
    }
    groups.push(IssueGroup {
        primary: Recommendation {
            rule_name: rule_names::MASSIVE_UNDERUTILIZATION,
            layer: 5,
            impact: 5,
            confidence: 0.7,
            action: "Batch more requests or increase client concurrency until a wait queue forms".to_string(),
            display_lines: vec![
                "[!] Massive Under-utilization".to_string(),
                String::new(),
                format!(
                    "  Decode eff. {eff:.1}%  (threshold: < {:.0}%)",
                    MASSIVE_UNDERUTIL_THRESHOLD_PCT
                ),
                "  Wait queue  0  (server not saturated)".to_string(),
                String::new(),
                "  Cause:".to_string(),
                "    GPU is idle. No config rule explains this - client traffic is too low."
                    .to_string(),
                String::new(),
                "  Fix:".to_string(),
                "    • Batch more requests or increase client concurrency until a wait queue forms.".to_string(),
                String::new(),
                "  Expected: Efficiency climbs as the GPU is fed more work.".to_string(),
                "  Confidence: Medium".to_string(),
            ],
            short_action: "batch more requests or increase client concurrency until a wait queue forms".to_string(),
            expected_impact: "Efficiency climbs as the GPU is fed more work.".to_string(),
        },
        secondary: Vec::new(),
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
            host_gpu_count: None,
        };
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let ctx = StaticContext::from_snapshot(&s, cfg);
        let win = RuntimeWindow::from_snapshot(s);
        let input = AnalysisInput::new(&ctx, &win);
        let report = build_report(input);
        assert_eq!(report.groups.len(), 1);
        assert_eq!(
            report.groups[0].primary.rule_name,
            rule_names::KV_CACHE_PRESSURE
        );
        assert!(
            !report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::UNDER_BATCHING)
        );
    }

    #[test]
    fn dag_suppression_table_oom_drops_kv_pressure() {
        use crate::collectors::VllmConfig;

        let t = SystemTime::UNIX_EPOCH;
        let v = VllmRawMetrics {
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
            host_gpu_count: None,
        };
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let ctx = StaticContext::from_snapshot(&s, cfg);
        let win = RuntimeWindow::from_snapshot(s);
        let input = AnalysisInput::new(&ctx, &win);
        let report = build_report(input);

        assert!(
            report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::OOM_RISK)
        );
        assert!(
            !report
                .groups
                .iter()
                .any(|g| g.primary.rule_name == rule_names::KV_CACHE_PRESSURE)
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
            host_gpu_count: None,
        };
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let ctx = StaticContext::from_snapshot(&s, cfg);
        let win = RuntimeWindow::from_snapshot(s);
        let input = AnalysisInput::new(&ctx, &win);
        let report = build_report(input);
        assert_eq!(report.groups.len(), 1);
        assert_eq!(
            report.groups[0].primary.rule_name,
            rule_names::UNDER_BATCHING
        );
    }

    fn starved_no_rules_fixture() -> (StaticContext, RuntimeWindow) {
        use crate::collectors::VllmConfig;

        let t = SystemTime::UNIX_EPOCH;
        let v = VllmRawMetrics {
            model_name: Some("meta-llama/Llama-3.1-8B-Instruct".to_string()),
            num_requests_running: Some(64.0),
            num_requests_waiting: Some(2.0),
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
            host_gpu_count: None,
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

    fn massive_underutilization_count(report: &Report) -> usize {
        report
            .groups
            .iter()
            .filter(|g| g.primary.rule_name == rule_names::MASSIVE_UNDERUTILIZATION)
            .count()
    }

    #[test]
    fn build_report_does_not_add_massive_underutilization() {
        let (ctx, win) = starved_no_rules_fixture();
        let input = AnalysisInput::new(&ctx, &win);
        let report = build_report(input);
        assert_eq!(
            massive_underutilization_count(&report),
            0,
            "build_report must not inject the safety net"
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
            report.groups.is_empty(),
            "fixture should fire no rules before safety net: {:?}",
            report
                .groups
                .iter()
                .map(|g| g.primary.rule_name)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn build_report_for_diagnose_adds_massive_underutilization_once() {
        let (ctx, win) = starved_no_rules_fixture();
        let windows = vec![win];
        let input = AnalysisInput::new(&ctx, &windows[0]);
        let report = build_report_for_diagnose(&windows, input);
        assert_eq!(
            massive_underutilization_count(&report),
            1,
            "diagnose path should inject safety net exactly once"
        );
    }

    #[test]
    fn massive_underutilization_suppressed_at_high_occupancy() {
        let (mut ctx, mut win) = starved_no_rules_fixture();
        win.snapshot.vllm.num_requests_running = Some(200.0);
        win.snapshot.vllm.max_num_seqs = Some(256);
        ctx.config.max_num_seqs = Some(256);
        let windows = vec![win.clone()];
        let input = AnalysisInput::new(&ctx, &win);
        let report = build_report_for_diagnose(&windows, input);
        assert_eq!(
            massive_underutilization_count(&report),
            0,
            "high occupancy should not trigger traffic-starvation fallback"
        );
        let eff = report
            .baseline
            .as_ref()
            .and_then(|b| b.efficiency_pct)
            .expect("baseline efficiency");
        assert!(
            eff < MASSIVE_UNDERUTIL_THRESHOLD_PCT,
            "fixture should still be under-utilized: {eff}%"
        );
    }
}
