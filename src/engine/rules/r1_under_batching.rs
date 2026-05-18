use crate::collectors::RawSnapshot;

use super::{skew_secs, Recommendation, MAX_OBSERVATION_SKEW_SECS};

const UNDER_BATCHING_GPU_UTIL_LT: f64 = 62.0;
const UNDER_BATCHING_RUNNING_GT: f64 = 0.75;
const UNDER_BATCHING_WAITING_LT: f64 = 2.0;
const UNDER_BATCHING_TPOT_HIGH_MS: f64 = 100.0;

#[derive(Debug, Clone, PartialEq)]
pub struct UnderBatchingDetail {
    pub running: f64,
    pub waiting: f64,
    pub max_num_seqs: u32,
    pub gpu_util: f64,
    pub tpot_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissReport {
    pub running: Option<f64>,
    pub gpu_util: Option<f64>,
    pub max_num_seqs: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Rule1Outcome {
    Fired(UnderBatchingDetail),
    NotFired(MissReport),
}

pub fn rule1_under_batching(snapshot: &RawSnapshot) -> Rule1Outcome {
    let skew = skew_secs(snapshot.gpu_observed_at, snapshot.vllm_observed_at);
    let running = snapshot.vllm.num_requests_running;
    let max_num_seqs = snapshot.vllm.max_num_seqs;
    let gpu_util = snapshot.gpu.gpu_util_pct;
    let waiting = snapshot.vllm.num_requests_waiting;
    let tpot = snapshot.vllm.tpot_ms;

    let miss = || MissReport {
        running,
        gpu_util,
        max_num_seqs,
    };

    if skew > MAX_OBSERVATION_SKEW_SECS {
        return Rule1Outcome::NotFired(miss());
    }

    let Some(rv) = running.filter(|v| v.is_finite()) else {
        return Rule1Outcome::NotFired(miss());
    };
    let Some(max_n) = max_num_seqs.filter(|&n| n > 0) else {
        return Rule1Outcome::NotFired(miss());
    };
    let Some(gpu) = gpu_util.filter(|v| v.is_finite()) else {
        return Rule1Outcome::NotFired(miss());
    };
    let Some(wv) = waiting.filter(|v| v.is_finite()) else {
        return Rule1Outcome::NotFired(miss());
    };
    let Some(tpot_ms) = tpot.filter(|v| v.is_finite() && *v >= UNDER_BATCHING_TPOT_HIGH_MS) else {
        return Rule1Outcome::NotFired(miss());
    };

    let fires = rv > UNDER_BATCHING_RUNNING_GT
        && gpu < UNDER_BATCHING_GPU_UTIL_LT
        && wv < UNDER_BATCHING_WAITING_LT;

    if fires {
        Rule1Outcome::Fired(UnderBatchingDetail {
            running: rv,
            waiting: wv,
            max_num_seqs: max_n,
            gpu_util: gpu,
            tpot_ms,
        })
    } else {
        Rule1Outcome::NotFired(miss())
    }
}

pub fn r1_recommendation(snapshot: &RawSnapshot) -> Option<Recommendation> {
    let Rule1Outcome::Fired(d) = rule1_under_batching(snapshot) else {
        return None;
    };
    Some(Recommendation {
        rule_name: "under_batching",
        impact: 4,
        confidence: 0.9,
        action: "Raise max_num_seqs or increase client concurrency".to_string(),
        expected_impact: "Lower TPOT, higher throughput at scale".to_string(),
        display_lines: format_under_batching_fired(&d),
    })
}

pub(super) fn format_under_batching_fired(d: &UnderBatchingDetail) -> Vec<String> {
    vec![
        "[!] Under-batching — Memory-Bandwidth Bottleneck".to_string(),
        String::new(),
        format!(
            "  GPU util   {:.1}%  (threshold: < {:.0}%)",
            d.gpu_util, UNDER_BATCHING_GPU_UTIL_LT
        ),
        format!(
            "  TPOT       {:.0}ms     (threshold: ≥ {:.0}ms)",
            d.tpot_ms, UNDER_BATCHING_TPOT_HIGH_MS
        ),
        format!(
            "  Requests   {:.0} running, {:.0} waiting",
            d.running, d.waiting
        ),
        String::new(),
        "  Small batch size is forcing memory-bandwidth-bound execution —".to_string(),
        "  GPU loads weights faster than it computes, stalling token output.".to_string(),
        String::new(),
        "  Fix:".to_string(),
        format!(
            "    • Raise --max-num-seqs (current: {}) if upstream traffic is queuing elsewhere",
            d.max_num_seqs
        ),
        "    • Increase client concurrency to feed the GPU larger batches".to_string(),
        "    • If this is peak traffic: switch to a quantized model (fp8/AWQ)".to_string(),
        "      to shrink weight footprint and lower TPOT at low concurrency".to_string(),
        String::new(),
        "  Expected: Lower TPOT, higher throughput at scale.".to_string(),
        "  Confidence: High".to_string(),
    ]
}

pub(super) fn format_under_batching_window_issue(
    d: &UnderBatchingDetail,
    seen_pct: u32,
) -> Vec<String> {
    vec![
        "[!] Under-batching — Memory-Bandwidth Bottleneck".to_string(),
        format!("  Seen in {seen_pct}% of windows"),
        String::new(),
        format!(
            "  GPU util   {:.1}%  (threshold: < {:.0}%)",
            d.gpu_util, UNDER_BATCHING_GPU_UTIL_LT
        ),
        format!(
            "  TPOT       {:.0}ms     (threshold: ≥ {:.0}ms)",
            d.tpot_ms, UNDER_BATCHING_TPOT_HIGH_MS
        ),
        format!(
            "  Requests   {:.0} running, {:.0} waiting",
            d.running, d.waiting
        ),
        String::new(),
        "  Fix:".to_string(),
        format!(
            "    • Raise --max-num-seqs (current: {}) if upstream traffic is queuing elsewhere",
            d.max_num_seqs
        ),
        "    • Increase client concurrency to feed the GPU larger batches".to_string(),
    ]
}

pub(super) fn aggregate_r1_detail(
    details: &[UnderBatchingDetail],
    summary: &RawSnapshot,
) -> UnderBatchingDetail {
    if details.is_empty() {
        return UnderBatchingDetail {
            running: summary.vllm.num_requests_running.unwrap_or(0.0),
            waiting: summary.vllm.num_requests_waiting.unwrap_or(0.0),
            max_num_seqs: summary.vllm.max_num_seqs.unwrap_or(256),
            gpu_util: summary.gpu.gpu_util_pct.unwrap_or(0.0),
            tpot_ms: summary.vllm.tpot_ms.unwrap_or(UNDER_BATCHING_TPOT_HIGH_MS),
        };
    }
    let n = details.len() as f64;
    let running = details.iter().map(|d| d.running).sum::<f64>() / n;
    let waiting = details.iter().map(|d| d.waiting).sum::<f64>() / n;
    let gpu = details.iter().map(|d| d.gpu_util).sum::<f64>() / n;
    let tpot = details.iter().map(|d| d.tpot_ms).sum::<f64>() / n;
    UnderBatchingDetail {
        running,
        waiting,
        max_num_seqs: details[0].max_num_seqs,
        gpu_util: gpu,
        tpot_ms: tpot,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{GpuRawMetrics, VllmRawMetrics};
    use std::time::{Duration, SystemTime};

    fn snap(
        gpu_at: SystemTime,
        vllm_at: SystemTime,
        running: f64,
        waiting: f64,
        gpu_util: f64,
        tpot_ms: Option<f64>,
    ) -> RawSnapshot {
        RawSnapshot {
            gpu_observed_at: gpu_at,
            vllm_observed_at: vllm_at,
            timestamp: gpu_at,
            vllm: VllmRawMetrics {
                num_requests_running: Some(running),
                num_requests_waiting: Some(waiting),
                max_num_seqs: Some(256),
                tpot_ms,
                ..Default::default()
            },
            gpu: GpuRawMetrics {
                gpu_util_pct: Some(gpu_util),
                ..Default::default()
            },
        }
    }

    #[test]
    fn fires_when_gates_and_tpot_high() {
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, 2.0, 0.0, 36.0, Some(120.0));
        let r = r1_recommendation(&s).expect("fired");
        assert_eq!(r.rule_name, "under_batching");
        assert!((r.confidence - 0.9).abs() < 1e-9);
        match rule1_under_batching(&s) {
            Rule1Outcome::Fired(d) => {
                assert!((d.running - 2.0).abs() < 1e-9);
                assert!((d.tpot_ms - 120.0).abs() < 1e-9);
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired"),
        }
    }

    #[test]
    fn mute_when_tpot_low() {
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, 2.0, 0.0, 36.0, Some(50.0));
        assert!(r1_recommendation(&s).is_none());
    }

    #[test]
    fn mute_when_tpot_missing() {
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, 2.0, 0.0, 36.0, None);
        assert!(r1_recommendation(&s).is_none());
    }

    #[test]
    fn mute_when_waiting_at_two() {
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, 2.0, 2.0, 36.0, Some(120.0));
        assert!(r1_recommendation(&s).is_none());
    }

    #[test]
    fn mute_when_skew_over_one_second() {
        let t0 = SystemTime::UNIX_EPOCH;
        let t1 = t0 + Duration::from_secs(2);
        let s = snap(t0, t1, 2.0, 0.0, 36.0, Some(120.0));
        assert!(r1_recommendation(&s).is_none());
    }
}
