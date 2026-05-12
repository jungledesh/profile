use crate::collectors::RawSnapshot;

use super::{skew_secs, Recommendation, MAX_OBSERVATION_SKEW_SECS};

const UNDER_BATCHING_GPU_UTIL_LT: f64 = 62.0;
const UNDER_BATCHING_RUNNING_GT: f64 = 0.75;
const UNDER_BATCHING_OCCUPANCY_FRAC: f64 = 0.04;
const UNDER_BATCHING_WAITING_LT: f64 = 2.0;

#[derive(Debug, Clone, PartialEq)]
pub struct UnderBatchingDetail {
    pub running: f64,
    pub max_num_seqs: u32,
    pub gpu_util: f64,
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

pub fn r1_recommendation(snapshot: &RawSnapshot) -> Option<Recommendation> {
    let Rule1Outcome::Fired(d) = rule1_under_batching(snapshot) else {
        return None;
    };
    Some(Recommendation {
        rule_name: "under_batching",
        impact: 4,
        confidence: if d.gpu_util < 30.0 { 0.9 } else { 0.75 },
        action: "Increase client concurrency or raise max_num_seqs".to_string(),
        expected_impact: "Higher GPU utilization and throughput".to_string(),
        display_lines: format_under_batching_fired(&d),
    })
}

pub(super) fn format_under_batching_fired(d: &UnderBatchingDetail) -> Vec<String> {
    let pct = (d.running / f64::from(d.max_num_seqs)) * 100.0;
    let run_s = fmt_f64_display(d.running);
    let gpu_s = fmt_f64_display(d.gpu_util);
    vec![
        "ISSUE: Under-batching".to_string(),
        format!(
            "Cause: Very low occupancy — {run_s} / {} ({pct:.1}%), avg GPU util {gpu_s}% with headroom",
            d.max_num_seqs,
        ),
        String::new(),
        "Recommendation:".to_string(),
        "  • Increase client concurrency or request rate".to_string(),
        "  • Raise max_num_seqs if VRAM allows".to_string(),
        String::new(),
        "Expected: Better throughput".to_string(),
        "Confidence: Medium-High".to_string(),
    ]
}

pub(super) fn format_under_batching_window_issue(
    d: &UnderBatchingDetail,
    seen_pct: u32,
) -> Vec<String> {
    let occupancy_pct = (d.running / f64::from(d.max_num_seqs)) * 100.0;
    vec![
        "Under-batching".to_string(),
        format!("Seen in {seen_pct}% of windows"),
        format!(
            "Cause: Very low occupancy — avg {:.1} / {} ({occupancy_pct:.1}%), avg GPU util {:.1}% with headroom",
            d.running, d.max_num_seqs, d.gpu_util
        ),
        String::new(),
        "For better efficiency:".to_string(),
        "  • Increase client concurrency or request rate".to_string(),
        "  • Raise max_num_seqs if VRAM allows".to_string(),
    ]
}

pub(super) fn aggregate_r1_detail(
    details: &[UnderBatchingDetail],
    summary: &RawSnapshot,
) -> UnderBatchingDetail {
    if details.is_empty() {
        return UnderBatchingDetail {
            running: summary.vllm.num_requests_running.unwrap_or(0.0),
            max_num_seqs: summary.vllm.max_num_seqs.unwrap_or(256),
            gpu_util: summary.gpu.gpu_util_pct.unwrap_or(0.0),
        };
    }
    let running = details.iter().map(|d| d.running).sum::<f64>() / details.len() as f64;
    let gpu = details.iter().map(|d| d.gpu_util).sum::<f64>() / details.len() as f64;
    UnderBatchingDetail {
        running,
        max_num_seqs: details[0].max_num_seqs,
        gpu_util: gpu,
    }
}

fn fmt_f64_display(x: f64) -> String {
    if (x - x.round()).abs() < 1e-6 {
        format!("{:.0}", x)
    } else {
        format!("{:.1}", x)
    }
}
