use crate::collectors::RawSnapshot;

use super::Recommendation;

const CONCURRENCY_SATURATION_QUEUE_RATIO_MIN: f64 = 0.30;

#[derive(Debug, Clone, PartialEq)]
pub struct ConcurrencySaturationDetail {
    pub requests_running: f64,
    pub requests_waiting: f64,
    pub max_num_seqs: u32,
    pub queue_ratio: f64,
}

pub fn rule5_concurrency_saturation(snapshot: &RawSnapshot) -> Option<ConcurrencySaturationDetail> {
    let run = snapshot
        .vllm
        .num_requests_running
        .filter(|v| v.is_finite() && *v > 0.0)?;
    let max_seqs = snapshot.vllm.max_num_seqs.filter(|&n| n > 0)?;
    // Exact equality: scheduler cap is the bottleneck.
    // run > max_seqs means chunked prefill is batching across steps — cap is not the constraint.
    if (run - f64::from(max_seqs)).abs() > 0.5 {
        return None;
    }
    let wait = snapshot
        .vllm
        .num_requests_waiting
        .filter(|v| v.is_finite())?;
    let total = wait + run;
    if total <= 0.0 {
        return None;
    }
    let ratio = wait / total;
    if ratio < CONCURRENCY_SATURATION_QUEUE_RATIO_MIN {
        return None;
    }
    Some(ConcurrencySaturationDetail {
        requests_running: run,
        requests_waiting: wait,
        max_num_seqs: max_seqs,
        queue_ratio: ratio,
    })
}

pub(super) fn format_concurrency_saturation_issue(
    d: &ConcurrencySaturationDetail,
    seen_pct: u32,
) -> Vec<String> {
    vec![
        "[!] Concurrency Saturation — Scheduler at max_num_seqs limit".to_string(),
        format!("  Seen in {seen_pct}% of windows"),
        String::new(),
        "  Cause:".to_string(),
        format!(
            "  - Running at --max-num-seqs={} — scheduler cannot admit more sequences",
            d.max_num_seqs
        ),
        format!(
            "  - {:.0}% of active requests waiting (threshold: ≥{:.0}%)",
            d.queue_ratio * 100.0,
            CONCURRENCY_SATURATION_QUEUE_RATIO_MIN * 100.0
        ),
        format!("  - Queue: {:.0} waiting, {:.0} running", d.requests_waiting, d.requests_running),
        String::new(),
        "  Recommendation:".to_string(),
        format!("  • Raise --max-num-seqs above {} — check VRAM headroom in header first", d.max_num_seqs),
        "  • If VRAM is at limit: add a replica or lower max_model_len to reduce KV footprint per sequence".to_string(),
        String::new(),
        "  Expected: Queue drains, TTFT recovers.".to_string(),
        "  Confidence: High".to_string(),
    ]
}

pub fn r5_recommendation(snapshot: &RawSnapshot) -> Option<Recommendation> {
    let d = rule5_concurrency_saturation(snapshot)?;
    Some(Recommendation {
        rule_name: "concurrency_saturation",
        impact: 4,
        confidence: 0.9,
        action: format!(
            "Raise --max-num-seqs above {} (scheduler at cap, {:.0}% of requests waiting)",
            d.max_num_seqs,
            d.queue_ratio * 100.0
        ),
        expected_impact: "Queue drains, TTFT recovers.".to_string(),
        display_lines: format_concurrency_saturation_issue(&d, 100),
    })
}

pub(super) fn aggregate_concurrency_saturation_detail(
    details: &[ConcurrencySaturationDetail],
) -> Option<ConcurrencySaturationDetail> {
    if details.is_empty() {
        return None;
    }
    let n = details.len() as f64;
    let run = details.iter().map(|d| d.requests_running).sum::<f64>() / n;
    let wait = details.iter().map(|d| d.requests_waiting).sum::<f64>() / n;
    let ratio = details.iter().map(|d| d.queue_ratio).sum::<f64>() / n;
    let max_seqs = details.iter().map(|d| d.max_num_seqs).max().unwrap_or(0);
    Some(ConcurrencySaturationDetail {
        requests_running: run,
        requests_waiting: wait,
        max_num_seqs: max_seqs,
        queue_ratio: ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{GpuRawMetrics, VllmRawMetrics};
    use std::time::SystemTime;

    fn snap(vllm: VllmRawMetrics) -> RawSnapshot {
        RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm,
            gpu: GpuRawMetrics::default(),
        }
    }

    fn sat_vllm(run: f64, wait: f64, max_num_seqs: Option<u32>) -> VllmRawMetrics {
        VllmRawMetrics {
            num_requests_running: Some(run),
            num_requests_waiting: Some(wait),
            max_num_seqs,
            generation_tokens_per_sec: Some(100.0),
            ..Default::default()
        }
    }

    #[test]
    fn fires_when_at_max_num_seqs_and_ratio_at_least_0_30() {
        let d = rule5_concurrency_saturation(&snap(sat_vllm(32.0, 15.0, Some(32)))).expect("fired");
        assert_eq!(d.max_num_seqs, 32);
        assert!((d.queue_ratio - (15.0 / 47.0)).abs() < 1e-9);
    }

    #[test]
    fn silent_when_run_below_max_num_seqs() {
        assert!(rule5_concurrency_saturation(&snap(sat_vllm(31.0, 15.0, Some(32)))).is_none());
    }

    #[test]
    fn silent_when_ratio_below_0_30() {
        assert!(rule5_concurrency_saturation(&snap(sat_vllm(32.0, 2.0, Some(32)))).is_none());
    }

    #[test]
    fn silent_when_max_num_seqs_missing() {
        assert!(rule5_concurrency_saturation(&snap(sat_vllm(32.0, 15.0, None))).is_none());
    }

    #[test]
    fn silent_when_num_requests_waiting_missing() {
        let mut v = sat_vllm(32.0, 15.0, Some(32));
        v.num_requests_waiting = None;
        assert!(rule5_concurrency_saturation(&snap(v)).is_none());
    }

    #[test]
    fn silent_when_run_exceeds_max_num_seqs_chunked_prefill() {
        // run=40 > max_num_seqs=32: chunked prefill is batching across steps.
        // Scheduler cap is not the bottleneck — r5 must not fire.
        assert!(rule5_concurrency_saturation(&snap(sat_vllm(40.0, 15.0, Some(32)))).is_none());
    }
}
