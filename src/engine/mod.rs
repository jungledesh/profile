pub mod baseline;
mod rules;

use crate::context::AnalysisInput;

pub use baseline::{CeilingEstimate, PhysicsBaseline, WeightDtypeSource};
pub use rules::*;

#[derive(Debug, Clone)]
pub struct Report {
    pub baseline: Option<PhysicsBaseline>,
    pub groups: Vec<IssueGroup>,
}

pub fn build_report(input: AnalysisInput<'_>) -> Report {
    let baseline = baseline::compute(&input);
    let snapshot = &input.window.snapshot;
    let kv_headroom = baseline.as_ref().and_then(|b| b.kv_headroom_gb);
    let tp = input.ctx.config.tensor_parallel_size;

    let mut recs: Vec<Recommendation> = [
        rules::r1_recommendation(snapshot),
        rules::r2_recommendation(snapshot),
        rules::r3_recommendation(snapshot),
        rules::r4_recommendation(kv_headroom, tp),
    ]
    .into_iter()
    .flatten()
    .collect();

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

    Report { baseline, groups }
}

#[cfg(test)]
mod build_report_tests {
    use super::*;
    use crate::collectors::{GpuRawMetrics, RawSnapshot, VllmRawMetrics};
    use crate::context::{RuntimeWindow, StaticContext};
    use std::time::SystemTime;

    #[test]
    fn build_report_groups_sorted_by_impact_times_confidence() {
        let t = SystemTime::UNIX_EPOCH;
        let v = VllmRawMetrics {
            num_requests_running: Some(3.1),
            num_requests_waiting: Some(0.0),
            max_num_seqs: Some(256),
            kv_cache_usage_perc: Some(86.0),
            ..Default::default()
        };
        let g = GpuRawMetrics {
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
        let ctx = StaticContext::default();
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
        assert_eq!(report.groups[0].primary.rule_name, "kv_cache_pressure");
    }
}
