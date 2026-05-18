use crate::collectors::RawSnapshot;
use crate::engine::PhysicsBaseline;

use super::{skew_secs, Recommendation, MAX_OBSERVATION_SKEW_SECS};

const UNDER_BATCHING_GPU_UTIL_LT: f64 = 62.0;
const UNDER_BATCHING_RUNNING_GT: f64 = 0.75;
const UNDER_BATCHING_WAITING_LT: f64 = 2.0;
const UNDER_BATCHING_TPOT_RATIO: f64 = 3.0;
const UNDER_BATCHING_TPOT_FLOOR_MIN_MS: f64 = 1.0;

#[derive(Debug, Clone, PartialEq)]
pub struct UnderBatchingDetail {
    pub running: f64,
    pub waiting: f64,
    pub max_num_seqs: u32,
    pub gpu_util: f64,
    pub tpot_ms: f64,
    pub tpot_floor_ms: f64,
    pub tpot_ratio: f64,
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

fn effective_tpot_floor_ms(baseline: &PhysicsBaseline) -> f64 {
    baseline.tpot_floor_ms.max(UNDER_BATCHING_TPOT_FLOOR_MIN_MS)
}

fn tpot_ratio(tpot_ms: f64, baseline: &PhysicsBaseline) -> f64 {
    tpot_ms / effective_tpot_floor_ms(baseline)
}

pub fn rule1_under_batching(
    snapshot: &RawSnapshot,
    baseline: Option<&PhysicsBaseline>,
) -> Rule1Outcome {
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

    let Some(base) = baseline else {
        return Rule1Outcome::NotFired(miss());
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
    let Some(tpot_ms) = tpot.filter(|v| v.is_finite()) else {
        return Rule1Outcome::NotFired(miss());
    };

    let floor_ms = effective_tpot_floor_ms(base);
    let ratio = tpot_ratio(tpot_ms, base);

    let structural = rv > UNDER_BATCHING_RUNNING_GT
        && gpu < UNDER_BATCHING_GPU_UTIL_LT
        && wv < UNDER_BATCHING_WAITING_LT;

    if structural && ratio >= UNDER_BATCHING_TPOT_RATIO {
        Rule1Outcome::Fired(UnderBatchingDetail {
            running: rv,
            waiting: wv,
            max_num_seqs: max_n,
            gpu_util: gpu,
            tpot_ms,
            tpot_floor_ms: floor_ms,
            tpot_ratio: ratio,
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
            "  TPOT       {:.0}ms     (floor: {:.0}ms, ratio: {:.1}x, threshold: ≥ {:.1}x)",
            d.tpot_ms, d.tpot_floor_ms, d.tpot_ratio, UNDER_BATCHING_TPOT_RATIO
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
            "  TPOT       {:.0}ms     (floor: {:.0}ms, ratio: {:.1}x, threshold: ≥ {:.1}x)",
            d.tpot_ms, d.tpot_floor_ms, d.tpot_ratio, UNDER_BATCHING_TPOT_RATIO
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
    baseline: Option<&PhysicsBaseline>,
) -> UnderBatchingDetail {
    if details.is_empty() {
        let floor_ms = baseline
            .map(effective_tpot_floor_ms)
            .unwrap_or(UNDER_BATCHING_TPOT_FLOOR_MIN_MS);
        let tpot_ms = summary
            .vllm
            .tpot_ms
            .unwrap_or(floor_ms * UNDER_BATCHING_TPOT_RATIO);
        let ratio = baseline
            .map(|b| tpot_ratio(tpot_ms, b))
            .unwrap_or(UNDER_BATCHING_TPOT_RATIO);
        return UnderBatchingDetail {
            running: summary.vllm.num_requests_running.unwrap_or(0.0),
            waiting: summary.vllm.num_requests_waiting.unwrap_or(0.0),
            max_num_seqs: summary.vllm.max_num_seqs.unwrap_or(256),
            gpu_util: summary.gpu.gpu_util_pct.unwrap_or(0.0),
            tpot_ms,
            tpot_floor_ms: floor_ms,
            tpot_ratio: ratio,
        };
    }
    let n = details.len() as f64;
    let running = details.iter().map(|d| d.running).sum::<f64>() / n;
    let waiting = details.iter().map(|d| d.waiting).sum::<f64>() / n;
    let gpu = details.iter().map(|d| d.gpu_util).sum::<f64>() / n;
    let tpot = details.iter().map(|d| d.tpot_ms).sum::<f64>() / n;
    let floor = details.iter().map(|d| d.tpot_floor_ms).sum::<f64>() / n;
    let ratio = details.iter().map(|d| d.tpot_ratio).sum::<f64>() / n;
    UnderBatchingDetail {
        running,
        waiting,
        max_num_seqs: details[0].max_num_seqs,
        gpu_util: gpu,
        tpot_ms: tpot,
        tpot_floor_ms: floor,
        tpot_ratio: ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{GpuRawMetrics, VllmRawMetrics};
    use crate::engine::baseline::{CeilingEstimate, WeightDtypeSource};
    use std::time::{Duration, SystemTime};

    fn mock_baseline(tpot_floor_ms: f64) -> PhysicsBaseline {
        PhysicsBaseline {
            decode: CeilingEstimate {
                lower: 1.0,
                expected: 1.0,
                upper: 1.0,
            },
            prefill: None,
            efficiency_pct: None,
            headroom_pct: None,
            weight_dtype_source: WeightDtypeSource::Fallback,
            weight_gb: 1.0,
            kv_headroom_gb: None,
            tpot_floor_ms,
            prefill_latency_floor_ms: None,
            ridge_batch_size: 1.0,
        }
    }

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
    fn fires_when_ratio_at_or_above_threshold() {
        let base = mock_baseline(10.0);
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, 2.0, 0.0, 36.0, Some(35.0));
        let r = r1_recommendation(&s, Some(&base)).expect("fired");
        assert_eq!(r.rule_name, "under_batching");
        assert!((r.confidence - 0.9).abs() < 1e-9);
        match rule1_under_batching(&s, Some(&base)) {
            Rule1Outcome::Fired(d) => {
                assert!((d.tpot_ratio - 3.5).abs() < 1e-9);
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired"),
        }
    }

    #[test]
    fn mute_when_ratio_below_threshold() {
        let base = mock_baseline(10.0);
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, 2.0, 0.0, 36.0, Some(15.0));
        assert!(r1_recommendation(&s, Some(&base)).is_none());
    }

    #[test]
    fn mute_when_tpot_missing() {
        let base = mock_baseline(10.0);
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, 2.0, 0.0, 36.0, None);
        assert!(r1_recommendation(&s, Some(&base)).is_none());
    }

    #[test]
    fn mute_when_baseline_none() {
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, 2.0, 0.0, 36.0, Some(35.0));
        assert!(r1_recommendation(&s, None).is_none());
    }

    #[test]
    fn mute_when_waiting_at_two() {
        let base = mock_baseline(10.0);
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, 2.0, 2.0, 36.0, Some(35.0));
        assert!(r1_recommendation(&s, Some(&base)).is_none());
    }

    #[test]
    fn mute_when_skew_over_one_second() {
        let base = mock_baseline(10.0);
        let t0 = SystemTime::UNIX_EPOCH;
        let t1 = t0 + Duration::from_secs(2);
        let s = snap(t0, t1, 2.0, 0.0, 36.0, Some(35.0));
        assert!(r1_recommendation(&s, Some(&base)).is_none());
    }

    #[test]
    fn fires_when_near_zero_floor_clamped_to_minimum() {
        let base = mock_baseline(0.1);
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, 2.0, 0.0, 36.0, Some(4.0));
        assert!(r1_recommendation(&s, Some(&base)).is_some());
        match rule1_under_batching(&s, Some(&base)) {
            Rule1Outcome::Fired(d) => assert!((d.tpot_ratio - 4.0).abs() < 1e-9),
            Rule1Outcome::NotFired(_) => panic!("expected fired"),
        }
    }
}
