use crate::collectors::RawSnapshot;

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
    if run < f64::from(max_seqs) {
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
        "[!] Concurrency Saturation".to_string(),
        format!("  Seen in {seen_pct}% of windows"),
        "Cause:".to_string(),
        format!(
            "  - Running at --max-num-seqs limit ({:.0}/{}) — scheduler cannot admit more sequences",
            d.requests_running, d.max_num_seqs
        ),
        format!(
            "  - {:.0} requests queuing ({:.0}% of active requests waiting)",
            d.requests_waiting,
            d.queue_ratio * 100.0
        ),
        String::new(),
        "  Fix:".to_string(),
        "    • Raise --max-num-seqs if vRAM headroom exists (check vRAM in header)".to_string(),
        "    • Scale out horizontally if already at memory limit".to_string(),
        "    • Lower max_model_len to reduce per-sequence KV footprint".to_string(),
        String::new(),
        "  Expected: Wait queue drains, TTFT recovers.".to_string(),
        "  Confidence: High".to_string(),
    ]
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
    Some(ConcurrencySaturationDetail {
        requests_running: run,
        requests_waiting: wait,
        max_num_seqs: details[0].max_num_seqs,
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
}
