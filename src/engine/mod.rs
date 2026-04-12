//! Pure rule evaluation on [`crate::collectors::RawSnapshot`] (no network, no NVML).

use std::time::SystemTime;

use crate::collectors::RawSnapshot;

/// Correlation gate: GPU vs vLLM observation times must be close.
const MAX_OBSERVATION_SKEW_SECS: f64 = 1.0;
/// Rule 1: fire only when NVML GPU util is strictly below this (percent).
const UNDER_BATCHING_GPU_UTIL_LT: f64 = 62.0;
/// Minimum mean `num_requests_running` (window) so we do not fire on an idle server.
const UNDER_BATCHING_RUNNING_GT: f64 = 0.75;
/// Mean running must stay strictly below this fraction of `max_num_seqs` to fire (8% — primary cap).
const UNDER_BATCHING_OCCUPANCY_FRAC: f64 = 0.08;
/// Fire only when mean waiting is strictly below this (no backlog).
const UNDER_BATCHING_WAITING_LT: f64 = 2.0;

#[derive(Debug, Clone, PartialEq)]
pub struct Issue {
    pub confidence: f64,
    pub evidence: Vec<String>,
}

/// Values that triggered rule 1 (under-batching).
#[derive(Debug, Clone, PartialEq)]
pub struct UnderBatchingDetail {
    pub running: f64,
    pub max_num_seqs: u32,
    pub gpu_util: f64,
}

pub fn evaluate_issues(snapshot: &RawSnapshot) -> Vec<Issue> {
    match rule1_under_batching(snapshot) {
        Rule1Outcome::Fired(d) => vec![issue_from_under_batching(&d)],
        Rule1Outcome::NotFired(_) => vec![],
    }
}

fn issue_from_under_batching(d: &UnderBatchingDetail) -> Issue {
    Issue {
        confidence: 0.85,
        evidence: vec![format!(
            "Under-batching: {:.1} running / max_num_seqs {} | GPU {:.1}%",
            d.running, d.max_num_seqs, d.gpu_util
        )],
    }
}

/// Rule 1 evaluation: either fired detail for diagnose, or a miss report.
#[derive(Debug, Clone, PartialEq)]
pub enum Rule1Outcome {
    Fired(UnderBatchingDetail),
    NotFired(MissReport),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissReport {
    pub running: Option<f64>,
    pub gpu_util: Option<f64>,
    pub max_num_seqs: Option<u32>,
}

/// Rule 1 lines for the diagnose table (fired or not).
pub fn format_rule1_diagnose(snapshot: &RawSnapshot) -> Vec<String> {
    match rule1_under_batching(snapshot) {
        Rule1Outcome::Fired(d) => format_under_batching_fired(&d),
        Rule1Outcome::NotFired(m) => format_rule1_miss(&m),
    }
}

pub fn rule1_under_batching(snapshot: &RawSnapshot) -> Rule1Outcome {
    let skew = skew_secs(snapshot.gpu_observed_at, snapshot.vllm_observed_at);
    let running = snapshot.vllm.num_requests_running;
    let max_num_seqs = snapshot.vllm.max_num_seqs;
    let gpu_util = snapshot.gpu.gpu_util_pct;
    let waiting = snapshot.vllm.num_requests_waiting;

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

    let max_f = f64::from(max_n);
    let occupancy_cap = UNDER_BATCHING_OCCUPANCY_FRAC * max_f;

    let fires = rv > UNDER_BATCHING_RUNNING_GT
        && rv < occupancy_cap
        && gpu < UNDER_BATCHING_GPU_UTIL_LT
        && wv < UNDER_BATCHING_WAITING_LT;

    if fires {
        Rule1Outcome::Fired(UnderBatchingDetail {
            running: rv,
            max_num_seqs: max_n,
            gpu_util: gpu,
        })
    } else {
        Rule1Outcome::NotFired(miss())
    }
}

fn skew_secs(a: SystemTime, b: SystemTime) -> f64 {
    match a.duration_since(b) {
        Ok(d) => d.as_secs_f64(),
        Err(e) => -e.duration().as_secs_f64(),
    }
    .abs()
}

fn format_rule1_miss(m: &MissReport) -> Vec<String> {
    let mut lines = vec!["Rule: Under-batching — Not triggered".to_string()];
    lines.extend(miss_bullet_lines(m));
    lines
}

fn miss_bullet_lines(m: &MissReport) -> Vec<String> {
    let run = m
        .running
        .filter(|x| x.is_finite())
        .map(|r| format!("{r:.1}"))
        .unwrap_or_else(|| "—".to_string());
    let maxs = m
        .max_num_seqs
        .map(|n| n.to_string())
        .unwrap_or_else(|| "—".to_string());
    let gpu = m
        .gpu_util
        .filter(|x| x.is_finite())
        .map(|g| format!("{g:.1}"))
        .unwrap_or_else(|| "—".to_string());

    vec![
        format!("  - Running {run} / {maxs} max_num_seqs (moderate occupancy)"),
        format!("  - GPU utilization {gpu}% — batching is not the primary bottleneck"),
    ]
}

fn format_under_batching_fired(d: &UnderBatchingDetail) -> Vec<String> {
    let pct = (d.running / f64::from(d.max_num_seqs)) * 100.0;
    vec![
        "ISSUE: Under-batching Detected".to_string(),
        format!(
            "Cause: Very low scheduler occupancy — {:.1} running requests vs max_num_seqs = {} ({:.1}%)",
            d.running, d.max_num_seqs, pct
        ),
        format!(
            "       GPU utilization only {:.1}% with large unused capacity",
            d.gpu_util
        ),
        String::new(),
        "Recommendation:".to_string(),
        "  • Increase client concurrency or request rate to better utilize the GPU".to_string(),
        "  • Consider raising max_num_seqs if it is currently limited".to_string(),
        "  • Verify continuous batching is properly enabled".to_string(),
        String::new(),
        "Expected Impact: Can significantly improve throughput when scheduler occupancy is the bottleneck"
            .to_string(),
        "Confidence: Medium-High".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{GpuRawMetrics, RawSnapshot, VllmRawMetrics};
    use std::time::{Duration, SystemTime};

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
            gpu,
        }
    }

    fn vllm_base() -> VllmRawMetrics {
        VllmRawMetrics {
            num_requests_running: Some(3.1),
            num_requests_waiting: Some(0.0),
            max_num_seqs: Some(256),
            ..Default::default()
        }
    }

    fn gpu_low() -> GpuRawMetrics {
        GpuRawMetrics {
            gpu_util_pct: Some(58.0),
            ..Default::default()
        }
    }

    #[test]
    fn under_batching_fires_when_gates_pass() {
        let t = SystemTime::UNIX_EPOCH;
        // 3.1 < 0.08 * 256 = 20.48, gpu 58 < 62, wait 0 < 2, running > 0.75
        let s = snap(t, t, vllm_base(), gpu_low());
        let issues = evaluate_issues(&s);
        assert_eq!(issues.len(), 1);
        assert!((issues[0].confidence - 0.85).abs() < 1e-9);
        match rule1_under_batching(&s) {
            Rule1Outcome::Fired(d) => {
                assert!((d.running - 3.1).abs() < 1e-9);
                assert_eq!(d.max_num_seqs, 256);
                assert!((d.gpu_util - 58.0).abs() < 1e-9);
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired"),
        }
    }

    #[test]
    fn skew_over_one_second_suppresses() {
        let t0 = SystemTime::UNIX_EPOCH;
        let t1 = t0 + Duration::from_secs(2);
        let s = snap(t0, t1, vllm_base(), gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn waiting_none_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_waiting = None;
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn waiting_at_two_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_waiting = Some(2.0);
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn running_at_floor_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(0.75);
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn running_below_activity_floor_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(0.6);
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn high_occupancy_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(40.0); // >= 8% of 256
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn occupancy_at_eight_percent_cap_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        // 8% * 256 = 20.48 — must be strictly below to fire
        v.num_requests_running = Some(21.0);
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn gpu_sixty_two_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut g = gpu_low();
        g.gpu_util_pct = Some(62.0);
        let s = snap(t, t, vllm_base(), g);
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn max_seqs_zero_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.max_num_seqs = Some(0);
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn nan_running_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(f64::NAN);
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn format_rule1_diagnose_fired_matches_template() {
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, vllm_base(), gpu_low());
        let lines = format_rule1_diagnose(&s);
        let text = lines.join("\n");
        assert!(text.contains("ISSUE: Under-batching Detected"));
        assert!(text.contains("Very low scheduler occupancy"));
        assert!(text.contains("3.1 running requests"));
        assert!(text.contains("max_num_seqs = 256"));
        assert!(text.contains("GPU utilization only 58.0% with large unused capacity"));
        assert!(text.contains("Recommendation:"));
        assert!(text.contains("continuous batching is properly enabled"));
        assert!(text.contains("scheduler occupancy is the bottleneck"));
        assert!(text.contains("Confidence: Medium-High"));
    }

    #[test]
    fn format_rule1_diagnose_miss_two_bullets() {
        let t = SystemTime::UNIX_EPOCH;
        let mut g = gpu_low();
        g.gpu_util_pct = Some(75.0);
        let s = snap(t, t, vllm_base(), g);
        let lines = format_rule1_diagnose(&s);
        let text = lines.join("\n");
        assert!(text.contains("Rule: Under-batching — Not triggered"));
        assert_eq!(lines.iter().filter(|l| l.starts_with("  - ")).count(), 2);
        assert!(text.contains("Running 3.1 / 256 max_num_seqs"));
        assert!(text.contains("GPU utilization 75.0%"));
    }
}
