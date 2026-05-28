use crate::collectors::RawSnapshot;
use crate::engine::PhysicsBaseline;

use super::Recommendation;

const UNDER_BATCHING_OCCUPANCY_PCT: f64 = 0.10;
const UNDER_BATCHING_WAITING_LT: f64 = 2.0;

#[derive(Debug, Clone, PartialEq)]
pub struct UnderBatchingDetail {
    pub running: f64,
    pub waiting: f64,
    pub max_num_seqs: u32,
    pub occupancy_pct: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissReport {
    pub running: Option<f64>,
    pub max_num_seqs: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Rule1Outcome {
    Fired(UnderBatchingDetail),
    NotFired(MissReport),
}

pub fn rule1_under_batching(
    snapshot: &RawSnapshot,
    _baseline: Option<&PhysicsBaseline>,
) -> Rule1Outcome {
    let running = snapshot.vllm.num_requests_running;
    let max_num_seqs = snapshot.vllm.max_num_seqs;
    let waiting = snapshot.vllm.num_requests_waiting;

    let miss = || MissReport {
        running,
        max_num_seqs,
    };

    let Some(rv) = running.filter(|v| v.is_finite() && *v > 0.0) else {
        return Rule1Outcome::NotFired(miss());
    };
    let Some(max_n) = max_num_seqs.filter(|&n| n > 0) else {
        return Rule1Outcome::NotFired(miss());
    };
    let Some(wv) = waiting.filter(|v| v.is_finite()) else {
        return Rule1Outcome::NotFired(miss());
    };

    let occupancy = rv / f64::from(max_n);

    if occupancy < UNDER_BATCHING_OCCUPANCY_PCT && wv < UNDER_BATCHING_WAITING_LT {
        Rule1Outcome::Fired(UnderBatchingDetail {
            running: rv,
            waiting: wv,
            max_num_seqs: max_n,
            occupancy_pct: occupancy * 100.0,
        })
    } else {
        Rule1Outcome::NotFired(miss())
    }
}

pub fn r1_recommendation(
    snapshot: &RawSnapshot,
    baseline: Option<&PhysicsBaseline>,
) -> Option<Recommendation> {
    let Rule1Outcome::Fired(d) = rule1_under_batching(snapshot, baseline) else {
        return None;
    };
    Some(Recommendation {
        rule_name: "under_batching",
        impact: 4,
        confidence: 0.9,
        action: "Increase client concurrency".to_string(),
        expected_impact: "Higher throughput, stable TPOT".to_string(),
        display_lines: format_under_batching_fired(&d),
    })
}

pub(super) fn format_under_batching_fired(d: &UnderBatchingDetail) -> Vec<String> {
    let threshold = UNDER_BATCHING_OCCUPANCY_PCT * 100.0;
    let unused = f64::from(d.max_num_seqs) - d.running;
    vec![
        "[!] Under-batching — Low Occupancy".to_string(),
        String::new(),
        format!(
            "  Occupancy  {:.1}%  (threshold: < {threshold:.0}%)",
            d.occupancy_pct
        ),
        format!(
            "  Requests   {:.0} running, {:.0} waiting  (max: {})",
            d.running, d.waiting, d.max_num_seqs
        ),
        String::new(),
        "  Engine has unused capacity with no backlog. Batch is too small.".to_string(),
        String::new(),
        "  Fix:".to_string(),
        format!("    • Increase client concurrency ({unused:.0} slots unused)"),
        "    • Note: if your client is already at full capacity, the bottleneck is upstream of vLLM, not a server config issue.".to_string(),
        String::new(),
        "  Expected: Higher throughput, lower TPOT at scale.".to_string(),
        "  Confidence: High".to_string(),
    ]
}

pub(super) fn format_under_batching_window_issue(
    d: &UnderBatchingDetail,
    seen_pct: u32,
) -> Vec<String> {
    let mut lines = format_under_batching_fired(d);
    lines.insert(1, format!("  Seen in {seen_pct}% of windows"));
    lines
}

pub(super) fn aggregate_r1_detail(
    details: &[UnderBatchingDetail],
    _summary: &RawSnapshot,
    _baseline: Option<&PhysicsBaseline>,
) -> UnderBatchingDetail {
    debug_assert!(
        !details.is_empty(),
        "aggregate_r1_detail called with empty details"
    );
    let n = details.len() as f64;
    let running = details.iter().map(|d| d.running).sum::<f64>() / n;
    let waiting = details.iter().map(|d| d.waiting).sum::<f64>() / n;
    let occupancy_pct = details.iter().map(|d| d.occupancy_pct).sum::<f64>() / n;
    UnderBatchingDetail {
        running,
        waiting,
        max_num_seqs: details.first().map_or(0, |d| d.max_num_seqs),
        occupancy_pct,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::VllmRawMetrics;

    fn snap(running: Option<f64>, max_num_seqs: Option<u32>, waiting: Option<f64>) -> RawSnapshot {
        use std::time::SystemTime;
        let t = SystemTime::UNIX_EPOCH;
        RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: VllmRawMetrics {
                num_requests_running: running,
                num_requests_waiting: waiting,
                max_num_seqs,
                ..Default::default()
            },
            gpu: Default::default(),
        }
    }

    #[test]
    fn fires_when_occupancy_low() {
        let s = snap(Some(5.0), Some(256), Some(0.0));
        match rule1_under_batching(&s, None) {
            Rule1Outcome::Fired(d) => {
                assert!((d.occupancy_pct - (5.0 / 256.0 * 100.0)).abs() < 0.1);
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired"),
        }
    }

    #[test]
    fn fires_at_occupancy_boundary() {
        let s = snap(Some(25.0), Some(256), Some(0.0));
        match rule1_under_batching(&s, None) {
            Rule1Outcome::Fired(d) => {
                assert!(d.occupancy_pct < 10.0);
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired at 9.8% occupancy"),
        }
    }

    #[test]
    fn mutes_at_occupancy_threshold() {
        let s = snap(Some(26.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_no_traffic() {
        let s = snap(Some(0.0), Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_backpressure_at_two() {
        let s = snap(Some(5.0), Some(256), Some(2.0));
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn fires_when_waiting_one_below_backpressure_gate() {
        let s = snap(Some(5.0), Some(256), Some(1.0));
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::Fired(_)
        ));
    }

    #[test]
    fn mutes_when_max_num_seqs_missing() {
        let s = snap(Some(5.0), None, Some(0.0));
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::NotFired(_)
        ));
    }

    #[test]
    fn mutes_when_running_missing() {
        let s = snap(None, Some(256), Some(0.0));
        assert!(matches!(
            rule1_under_batching(&s, None),
            Rule1Outcome::NotFired(_)
        ));
    }
}
