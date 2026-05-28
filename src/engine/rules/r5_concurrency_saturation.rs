use crate::collectors::RawSnapshot;

use super::{model_len_suffix, Recommendation};

const CONCURRENCY_SATURATION_QUEUE_RATIO_MIN: f64 = 0.30;
const VRAM_HEADROOM_MIN_GB: f64 = 20.0;

#[derive(Debug, Clone, PartialEq)]
pub struct ConcurrencySaturationDetail {
    pub requests_running: f64,
    pub requests_waiting: f64,
    pub max_num_seqs: u32,
    pub queue_ratio: f64,
    pub ttft_ms: Option<f64>,
    pub kv_headroom_gb: Option<f64>,
}

pub fn rule5_concurrency_saturation(
    snapshot: &RawSnapshot,
    kv_headroom_gb: Option<f64>,
) -> Option<ConcurrencySaturationDetail> {
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
        ttft_ms: snapshot.vllm.ttft_ms,
        kv_headroom_gb,
    })
}

pub(super) fn format_concurrency_saturation_issue(
    d: &ConcurrencySaturationDetail,
    seen_pct: u32,
    max_model_len: Option<u32>,
) -> Vec<String> {
    let mut lines = vec![
        "[!] Concurrency Saturation".to_string(),
        format!("  Seen in {seen_pct}% of windows"),
        String::new(),
        "  Cause:".to_string(),
        format!(
            "  - --max-num-seqs={} hit: scheduler won't admit more sequences",
            d.max_num_seqs
        ),
        format!(
            "  - {:.0}% of requests waiting ({:.0} of {:.0} active)",
            d.queue_ratio * 100.0,
            d.requests_waiting,
            d.requests_waiting + d.requests_running
        ),
    ];
    if let Some(ttft_ms) = d.ttft_ms.filter(|t| t.is_finite()) {
        lines.push(format!(
            "  - TTFT {:.1}s, {:.0} requests queued ahead",
            ttft_ms / 1000.0,
            d.requests_waiting
        ));
    }
    lines.push(String::new());
    lines.push("  Fix:".to_string());
    match d.kv_headroom_gb {
        Some(headroom) if headroom > VRAM_HEADROOM_MIN_GB => {
            lines.push(format!(
                "  • Raise --max-num-seqs above {} ({headroom:.0}GB VRAM available)",
                d.max_num_seqs
            ));
        }
        _ => {
            lines.push("  • Add a replica".to_string());
            lines.push(format!(
                "  • Or lower --max-model-len{} to reduce KV footprint per sequence",
                model_len_suffix(max_model_len)
            ));
        }
    }
    lines.push(String::new());
    lines.push("  Expected: Queue drains, TTFT recovers.".to_string());
    lines.push("  Confidence: High".to_string());
    lines
}

pub fn r5_recommendation(
    snapshot: &RawSnapshot,
    kv_headroom_gb: Option<f64>,
    max_model_len: Option<u32>,
) -> Option<Recommendation> {
    let d = rule5_concurrency_saturation(snapshot, kv_headroom_gb)?;
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
        display_lines: format_concurrency_saturation_issue(&d, 100, max_model_len),
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
    let (ttft_sum, ttft_count) = details
        .iter()
        .filter_map(|d| d.ttft_ms)
        .fold((0.0_f64, 0usize), |(s, c), v| (s + v, c + 1));
    let ttft_ms = (ttft_count > 0).then_some(ttft_sum / ttft_count as f64);
    let kv_headroom_gb = details.last().and_then(|d| d.kv_headroom_gb);
    Some(ConcurrencySaturationDetail {
        requests_running: run,
        requests_waiting: wait,
        max_num_seqs: max_seqs,
        queue_ratio: ratio,
        ttft_ms,
        kv_headroom_gb,
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

    fn fired_detail(
        ttft_ms: Option<f64>,
        kv_headroom_gb: Option<f64>,
    ) -> ConcurrencySaturationDetail {
        ConcurrencySaturationDetail {
            requests_running: 32.0,
            requests_waiting: 15.0,
            max_num_seqs: 32,
            queue_ratio: 15.0 / 47.0,
            ttft_ms,
            kv_headroom_gb,
        }
    }

    #[test]
    fn fires_when_at_max_num_seqs_and_ratio_at_least_0_30() {
        let d = rule5_concurrency_saturation(&snap(sat_vllm(32.0, 15.0, Some(32))), None)
            .expect("fired");
        assert_eq!(d.max_num_seqs, 32);
        assert!((d.queue_ratio - (15.0 / 47.0)).abs() < 1e-9);
        assert_eq!(d.kv_headroom_gb, None);
    }

    #[test]
    fn silent_when_run_below_max_num_seqs() {
        assert!(
            rule5_concurrency_saturation(&snap(sat_vllm(31.0, 15.0, Some(32))), None).is_none()
        );
    }

    #[test]
    fn silent_when_ratio_below_0_30() {
        assert!(rule5_concurrency_saturation(&snap(sat_vllm(32.0, 2.0, Some(32))), None).is_none());
    }

    #[test]
    fn silent_when_max_num_seqs_missing() {
        assert!(rule5_concurrency_saturation(&snap(sat_vllm(32.0, 15.0, None)), None).is_none());
    }

    #[test]
    fn silent_when_num_requests_waiting_missing() {
        let mut v = sat_vllm(32.0, 15.0, Some(32));
        v.num_requests_waiting = None;
        assert!(rule5_concurrency_saturation(&snap(v), None).is_none());
    }

    #[test]
    fn silent_when_run_exceeds_max_num_seqs_chunked_prefill() {
        assert!(
            rule5_concurrency_saturation(&snap(sat_vllm(40.0, 15.0, Some(32))), None).is_none()
        );
    }

    #[test]
    fn recommendation_raises_cap_when_headroom_available() {
        let text = format_concurrency_saturation_issue(&fired_detail(None, Some(63.0)), 100, None)
            .join("\n");
        assert!(text.contains("63GB VRAM available"));
    }

    #[test]
    fn recommendation_scales_out_when_vram_exhausted() {
        let text = format_concurrency_saturation_issue(&fired_detail(None, Some(5.0)), 100, None)
            .join("\n");
        assert!(text.contains("Add a replica"));
    }

    #[test]
    fn recommendation_scales_out_when_headroom_unknown() {
        let text =
            format_concurrency_saturation_issue(&fired_detail(None, None), 100, None).join("\n");
        assert!(text.contains("Add a replica"));
    }

    #[test]
    fn recommendation_shows_max_model_len_when_known() {
        let text =
            format_concurrency_saturation_issue(&fired_detail(None, Some(5.0)), 100, Some(8192))
                .join("\n");
        assert!(text.contains("--max-model-len (currently 8192)"));
    }

    #[test]
    fn cause_shows_ttft_when_available() {
        let text =
            format_concurrency_saturation_issue(&fired_detail(Some(5000.0), None), 100, None)
                .join("\n");
        assert!(text.contains("TTFT 5.0s"));
    }

    #[test]
    fn cause_omits_ttft_when_none() {
        let text =
            format_concurrency_saturation_issue(&fired_detail(None, None), 100, None).join("\n");
        assert!(!text.contains("requests queued ahead"));
    }
}
