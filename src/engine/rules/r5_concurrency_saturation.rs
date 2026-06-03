use crate::collectors::RawSnapshot;

use super::{model_len_suffix, Recommendation};

/// Minimum ratio of (waiting / total active) confirming the cap is structurally bottlenecking.
/// No industry standard exists — 0.30 is a judgment call. At 30%, nearly 1 in 3 active
/// requests is halted. Below this, queue depth is likely a transient micro-burst, not structural.
const CONCURRENCY_SATURATION_QUEUE_RATIO_MIN: f64 = 0.30;

/// KV cache usage below this means the pool has room to absorb new sequences safely.
/// Buffered 8 points below r2's 88% warning threshold — raising --max-num-seqs at 80%
/// won't immediately trigger KV pressure. At or above: hardware is near capacity, scale out.
const KV_CACHE_SAFE_TO_SCALE_PCT: f64 = 80.0;

#[derive(Debug, Clone, PartialEq)]
pub struct ConcurrencySaturationDetail {
    pub requests_running: f64,
    pub requests_waiting: f64,
    pub max_num_seqs: Option<u32>,
    pub queue_ratio: f64,
    pub ttft_ms: Option<f64>,
    pub kv_cache_usage_perc: Option<f64>,
}

pub fn rule5_concurrency_saturation(
    snapshot: &RawSnapshot,
    kv_cache_usage_perc: Option<f64>,
    config_max_num_seqs: Option<u32>,
) -> Option<ConcurrencySaturationDetail> {
    let run = snapshot
        .vllm
        .num_requests_running
        .filter(|v| v.is_finite() && *v > 0.0)?;
    let max_seqs = snapshot
        .vllm
        .max_num_seqs
        .or(config_max_num_seqs)
        .filter(|&n| n > 0)?;
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
        max_num_seqs: Some(max_seqs),
        queue_ratio: ratio,
        ttft_ms: snapshot.vllm.ttft_ms,
        kv_cache_usage_perc,
    })
}

pub(super) fn format_concurrency_saturation_issue(
    d: &ConcurrencySaturationDetail,
    max_model_len: Option<u32>,
) -> Vec<String> {
    let max_str = d
        .max_num_seqs
        .map(|n| n.to_string())
        .unwrap_or_else(|| "?".to_string());
    let confidence = match (d.ttft_ms, d.kv_cache_usage_perc) {
        (Some(_), Some(_)) => "High",
        _ => "Medium",
    };

    let mut lines = vec![
        "[!] Concurrency Saturation".to_string(),
        String::new(),
        "  Cause:".to_string(),
        format!("    • --max-num-seqs={max_str} hit: scheduler won't admit more sequences"),
        format!(
            "    • {:.0}% of requests waiting ({:.0} of {:.0} active)",
            d.queue_ratio * 100.0,
            d.requests_waiting,
            d.requests_waiting + d.requests_running
        ),
    ];
    if let Some(ttft_ms) = d.ttft_ms.filter(|t| t.is_finite()) {
        lines.push(format!("    • TTFT {:.1}s", ttft_ms / 1000.0,));
    }
    lines.push(String::new());
    lines.push("  Fix:".to_string());
    match d.kv_cache_usage_perc {
        Some(pct) if pct < KV_CACHE_SAFE_TO_SCALE_PCT => {
            lines.push(format!(
                "    • Raise --max-num-seqs above {max_str} (KV cache {pct:.0}% used, pool has room)"
            ));
        }
        Some(pct) => {
            lines.push(format!(
                "    • KV cache at {pct:.0}%. Raising --max-num-seqs will cause thrashing."
            ));
            lines.push("    • Add a replica".to_string());
            lines.push(format!(
                "    • Or lower --max-model-len{} to free KV blocks",
                model_len_suffix(max_model_len)
            ));
        }
        None => {
            lines.push(format!(
                "    • Raise --max-num-seqs above {max_str} if KV cache has headroom"
            ));
        }
    }
    lines.push(String::new());
    lines.push("  Expected: Queue drains, TTFT recovers.".to_string());
    lines.push(format!("  Confidence: {confidence}"));
    lines
}

pub(super) fn format_concurrency_saturation_window_issue(
    d: &ConcurrencySaturationDetail,
    seen_pct: u32,
    max_model_len: Option<u32>,
) -> Vec<String> {
    let mut lines = format_concurrency_saturation_issue(d, max_model_len);
    lines.insert(1, format!("  Seen in {seen_pct}% of windows"));
    lines
}

pub fn r5_recommendation(
    snapshot: &RawSnapshot,
    kv_cache_usage_perc: Option<f64>,
    config_max_num_seqs: Option<u32>,
    max_model_len: Option<u32>,
) -> Option<Recommendation> {
    let d = rule5_concurrency_saturation(snapshot, kv_cache_usage_perc, config_max_num_seqs)?;
    let max_label = d
        .max_num_seqs
        .map(|n| n.to_string())
        .unwrap_or_else(|| "?".to_string());
    Some(Recommendation {
        rule_name: "concurrency_saturation",
        impact: 4,
        confidence: match (d.ttft_ms, d.kv_cache_usage_perc) {
            (Some(_), Some(_)) => 0.9,
            _ => 0.6,
        },
        action: format!(
            "Raise --max-num-seqs above {} (scheduler at cap, {:.0}% of requests waiting)",
            max_label,
            d.queue_ratio * 100.0
        ),
        expected_impact: "Queue drains, TTFT recovers.".to_string(),
        display_lines: format_concurrency_saturation_issue(&d, max_model_len),
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
    let max_seqs = details.iter().filter_map(|d| d.max_num_seqs).max();
    let (ttft_sum, ttft_count) = details
        .iter()
        .filter_map(|d| d.ttft_ms)
        .fold((0.0_f64, 0usize), |(s, c), v| (s + v, c + 1));
    let ttft_ms = (ttft_count > 0).then_some(ttft_sum / ttft_count as f64);
    let kv_cache_usage_perc = details.last().and_then(|d| d.kv_cache_usage_perc);
    Some(ConcurrencySaturationDetail {
        requests_running: run,
        requests_waiting: wait,
        max_num_seqs: max_seqs,
        queue_ratio: ratio,
        ttft_ms,
        kv_cache_usage_perc,
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
        kv_cache_usage_perc: Option<f64>,
    ) -> ConcurrencySaturationDetail {
        ConcurrencySaturationDetail {
            requests_running: 32.0,
            requests_waiting: 15.0,
            max_num_seqs: Some(32),
            queue_ratio: 15.0 / 47.0,
            ttft_ms,
            kv_cache_usage_perc,
        }
    }

    #[test]
    fn fires_when_at_max_num_seqs_and_ratio_at_least_0_30() {
        let d = rule5_concurrency_saturation(&snap(sat_vllm(32.0, 15.0, Some(32))), None, None)
            .expect("fired");
        assert_eq!(d.max_num_seqs, Some(32));
        assert!((d.queue_ratio - (15.0 / 47.0)).abs() < 1e-9);
        assert_eq!(d.kv_cache_usage_perc, None);
    }

    #[test]
    fn silent_when_run_below_max_num_seqs() {
        assert!(
            rule5_concurrency_saturation(&snap(sat_vllm(31.0, 15.0, Some(32))), None, None)
                .is_none()
        );
    }

    #[test]
    fn silent_when_ratio_below_0_30() {
        assert!(
            rule5_concurrency_saturation(&snap(sat_vllm(32.0, 2.0, Some(32))), None, None)
                .is_none()
        );
    }

    #[test]
    fn silent_when_max_num_seqs_missing() {
        assert!(
            rule5_concurrency_saturation(&snap(sat_vllm(32.0, 15.0, None)), None, None).is_none()
        );
    }

    #[test]
    fn fires_when_max_num_seqs_from_config_fallback() {
        let d = rule5_concurrency_saturation(&snap(sat_vllm(32.0, 15.0, None)), None, Some(32))
            .expect("config max_num_seqs");
        assert_eq!(d.max_num_seqs, Some(32));
    }

    #[test]
    fn silent_when_num_requests_waiting_missing() {
        let mut v = sat_vllm(32.0, 15.0, Some(32));
        v.num_requests_waiting = None;
        assert!(rule5_concurrency_saturation(&snap(v), None, None).is_none());
    }

    #[test]
    fn silent_when_run_exceeds_max_num_seqs_chunked_prefill() {
        assert!(
            rule5_concurrency_saturation(&snap(sat_vllm(40.0, 15.0, Some(32))), None, None)
                .is_none()
        );
    }

    #[test]
    fn fix_raises_cap_when_kv_below_safe_threshold() {
        let text =
            format_concurrency_saturation_issue(&fired_detail(None, Some(70.0)), None).join("\n");
        assert!(text.contains("Raise --max-num-seqs above 32"));
        assert!(text.contains("KV cache 70% used, pool has room"));
    }

    #[test]
    fn fix_scales_out_when_kv_at_or_above_safe_threshold() {
        let text =
            format_concurrency_saturation_issue(&fired_detail(None, Some(85.0)), None).join("\n");
        assert!(text.contains("KV cache at 85%"));
        assert!(text.contains("will cause thrashing"));
        assert!(text.contains("Add a replica"));
    }

    #[test]
    fn fix_generic_when_kv_usage_unknown() {
        let text = format_concurrency_saturation_issue(&fired_detail(None, None), None).join("\n");
        assert!(text.contains("if KV cache has headroom"));
        assert!(!text.contains("Add a replica"));
    }

    #[test]
    fn confidence_high_when_ttft_and_kv_present() {
        let text =
            format_concurrency_saturation_issue(&fired_detail(Some(5000.0), Some(70.0)), None)
                .join("\n");
        assert!(text.contains("Confidence: High"));
    }

    #[test]
    fn confidence_medium_when_ttft_or_kv_missing() {
        let text =
            format_concurrency_saturation_issue(&fired_detail(Some(5000.0), None), None).join("\n");
        assert!(text.contains("Confidence: Medium"));
        let text2 =
            format_concurrency_saturation_issue(&fired_detail(None, Some(70.0)), None).join("\n");
        assert!(text2.contains("Confidence: Medium"));
    }

    #[test]
    fn fix_shows_max_model_len_when_kv_high() {
        let text = format_concurrency_saturation_issue(&fired_detail(None, Some(85.0)), Some(8192))
            .join("\n");
        assert!(text.contains("--max-model-len (currently 8192)"));
    }

    #[test]
    fn cause_shows_ttft_when_available() {
        let text =
            format_concurrency_saturation_issue(&fired_detail(Some(5000.0), None), None).join("\n");
        assert!(text.contains("TTFT 5.0s"));
    }

    #[test]
    fn cause_omits_ttft_when_none() {
        let text = format_concurrency_saturation_issue(&fired_detail(None, None), None).join("\n");
        assert!(!text.contains("requests queued ahead"));
    }

    #[test]
    fn aggregate_max_num_seqs_is_option_not_zero_sentinel() {
        let agg =
            aggregate_concurrency_saturation_detail(&[fired_detail(None, None)]).expect("agg");
        assert_eq!(agg.max_num_seqs, Some(32));
    }

    #[test]
    fn window_issue_inserts_seen_pct() {
        let lines = format_concurrency_saturation_window_issue(&fired_detail(None, None), 40, None);
        assert_eq!(lines[1], "  Seen in 40% of windows");
    }
}
