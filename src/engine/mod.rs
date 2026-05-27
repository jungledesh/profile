pub mod baseline;
mod rules;

use crate::context::AnalysisInput;

pub use baseline::{CeilingEstimate, PhysicsBaseline, WeightDtypeSource};
pub use rules::*;

#[derive(Debug, Clone)]
pub struct Report {
    pub baseline: Option<PhysicsBaseline>,
    pub groups: Vec<IssueGroup>,
    /// True when r4 removed a `kv_cache_pressure` recommendation from this report.
    pub r2_suppressed_by_r4: bool,
}

pub fn build_report(input: AnalysisInput<'_>) -> Report {
    let baseline = baseline::compute(&input);
    let snapshot = &input.window.snapshot;
    let kv_headroom = baseline.as_ref().and_then(|b| b.kv_headroom_gb);
    let tp = input.ctx.config.tensor_parallel_size;

    let mut recs: Vec<Recommendation> = [
        rules::r1_recommendation(snapshot, baseline.as_ref()),
        rules::r2_recommendation(snapshot),
        rules::r3_recommendation(snapshot),
        rules::r4_recommendation(kv_headroom, tp),
    ]
    .into_iter()
    .flatten()
    .collect();

    let kv_pressure = recs.iter().any(|r| r.rule_name == "kv_cache_pressure");
    if !kv_pressure {
        if let Some(r5) = rules::r5_recommendation(snapshot) {
            recs.push(r5);
        }
    }

    let r2_present_before = recs.iter().any(|r| r.rule_name == "kv_cache_pressure");
    let r4_fired = recs.iter().any(|r| r.rule_name == "parallelism_mismatch");
    let r2_suppressed_by_r4 = r4_fired && r2_present_before;
    if r4_fired {
        recs.retain(|r| r.rule_name != "kv_cache_pressure");
    }

    recs.sort_by(|a, b| {
        let sa = a.impact as f64 * a.confidence;
        let sb = b.impact as f64 * b.confidence;
        sb.total_cmp(&sa)
    });

    let groups = recs
        .into_iter()
        .map(|r| IssueGroup {
            primary: r,
            secondary: Vec::new(),
        })
        .collect();

    Report {
        baseline,
        groups,
        r2_suppressed_by_r4,
    }
}

#[cfg(test)]
mod build_report_tests {
    use super::*;
    use crate::collectors::{GpuRawMetrics, RawSnapshot, VllmRawMetrics};
    use crate::context::{RuntimeWindow, StaticContext};
    use std::time::SystemTime;

    #[test]
    fn build_report_groups_sorted_by_impact_times_confidence() {
        use crate::collectors::VllmConfig;

        let t = SystemTime::UNIX_EPOCH;
        let v = VllmRawMetrics {
            model_name: Some("meta-llama/Llama-3.1-8B-Instruct".to_string()),
            num_requests_running: Some(3.1),
            num_requests_waiting: Some(0.0),
            max_num_seqs: Some(256),
            kv_cache_usage_perc: Some(86.0),
            tpot_ms: Some(35.0),
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
            gpu: g,
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
            report.groups.len() >= 2,
            "expected r1+r2 to fire; got {:?}",
            report
                .groups
                .iter()
                .map(|x| x.primary.rule_name)
                .collect::<Vec<_>>()
        );
        for w in report.groups.windows(2) {
            assert!(
                w[0].score() + f64::EPSILON >= w[1].score(),
                "groups not sorted by score: {:?} then {:?}",
                w[0].primary.rule_name,
                w[1].primary.rule_name
            );
        }
        assert_eq!(report.groups[0].primary.rule_name, "under_batching");
        assert!(!report.r2_suppressed_by_r4);
    }

    #[test]
    fn build_report_suppresses_r2_when_r4_fires() {
        use crate::collectors::VllmConfig;

        let t = SystemTime::UNIX_EPOCH;
        let v = VllmRawMetrics {
            model_name: Some("meta-llama/Llama-3.1-70B-Instruct".to_string()),
            num_requests_running: Some(3.0),
            num_requests_waiting: Some(0.0),
            max_num_seqs: Some(256),
            kv_cache_usage_perc: Some(86.0),
            generation_tokens_per_sec: Some(50.0),
            request_success_per_sec: Some(10.0),
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
            gpu: g,
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

        assert!(report.r2_suppressed_by_r4);
        assert!(report
            .groups
            .iter()
            .any(|g| g.primary.rule_name == "parallelism_mismatch"));
        assert!(!report
            .groups
            .iter()
            .any(|g| g.primary.rule_name == "kv_cache_pressure"));
    }
}
