use crate::collectors::RawSnapshot;
use crate::fmt::fmt_seconds_from_ms;

use super::{model_len_suffix, Recommendation};

/// Minimum ratio of (waiting / total active) confirming the cap is structurally bottlenecking.
/// No industry standard exists — 0.30 is a judgment call. At 30%, nearly 1 in 3 active
/// requests is halted. Below this, queue depth is likely a transient micro-burst, not structural.
const CONCURRENCY_SATURATION_QUEUE_RATIO_MIN: f64 = 0.30;

/// KV cache usage below this means the pool has room to absorb new sequences safely.
/// Buffered 8 points below r2's 88% warning threshold — raising --max-num-seqs at 80%
/// won't immediately trigger KV pressure. At or above: hardware is near capacity, scale out.
const KV_CACHE_SAFE_TO_SCALE_PCT: f64 = 80.0;

#[derive(Debug, Clone)]
pub struct ConcurrencySaturationDetail {
    pub requests_running: f64,
    pub requests_waiting: f64,
    pub max_num_seqs: Option<u32>,
    pub queue_ratio: f64,
    pub ttft_ms: Option<f64>,
    pub ttft_p99_ms: Option<f64>,
    pub ttft_p99_buckets: Vec<crate::collectors::HistogramCount>,
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
        ttft_p99_ms: snapshot.vllm.ttft_p99_ms,
        ttft_p99_buckets: snapshot.vllm.ttft_p99_buckets.clone(),
        kv_cache_usage_perc,
    })
}

pub(super) fn r5_action(
    d: &ConcurrencySaturationDetail,
    kv_max_seqs: Option<u32>,
    max_model_len: Option<u32>,
) -> String {
    let max_label = d
        .max_num_seqs
        .map(|n| n.to_string())
        .unwrap_or_else(|| "?".to_string());

    if d.kv_cache_usage_perc
        .is_some_and(|p| p >= KV_CACHE_SAFE_TO_SCALE_PCT)
    {
        return "Add a replica to scale out.".to_string();
    }

    if let (Some(ceiling), Some(current)) = (kv_max_seqs, d.max_num_seqs) {
        if current >= ceiling {
            let m = max_model_len
                .map(|n| n.to_string())
                .unwrap_or_else(|| "?".to_string());
            let kv_note = match d.kv_cache_usage_perc {
                Some(pct) => format!("KV pool has room ({pct:.0}%), but"),
                None => "KV unknown, but".to_string(),
            };
            return format!(
                "{kv_note} --max-num-seqs is at the physics ceiling for max_model_len={m}. Lower --max-model-len to safely raise concurrency, or add a replica."
            );
        }
    }

    format!(
        "Raise --max-num-seqs above {max_label} (scheduler at cap, {:.0}% of requests waiting)",
        d.queue_ratio * 100.0
    )
}

pub(super) fn format_concurrency_saturation_issue(
    d: &ConcurrencySaturationDetail,
    max_model_len: Option<u32>,
    kv_max_seqs: Option<u32>,
) -> Vec<String> {
    let max_str = d
        .max_num_seqs
        .map(|n| n.to_string())
        .unwrap_or_else(|| "?".to_string());
    let confidence = match (d.ttft_ms.or(d.ttft_p99_ms), d.kv_cache_usage_perc) {
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
    match (
        d.ttft_p99_ms.filter(|t| t.is_finite()),
        d.ttft_ms.filter(|t| t.is_finite()),
    ) {
        (Some(p99), Some(avg)) => lines.push(format!(
            "    • TTFT {} p99 ({} avg)",
            fmt_seconds_from_ms(p99),
            fmt_seconds_from_ms(avg)
        )),
        (Some(p99), None) => lines.push(format!("    • TTFT {} p99", fmt_seconds_from_ms(p99))),
        (None, Some(avg)) => lines.push(format!("    • TTFT {}", fmt_seconds_from_ms(avg))),
        (None, None) => {}
    }
    lines.push(String::new());
    lines.push("  Fix:".to_string());
    match d.kv_cache_usage_perc {
        Some(pct) if pct < KV_CACHE_SAFE_TO_SCALE_PCT => {
            if kv_max_seqs
                .is_some_and(|ceiling| d.max_num_seqs.is_some_and(|current| current >= ceiling))
            {
                let m = max_model_len
                    .map(|n| format!("max_model_len={n}"))
                    .unwrap_or_else(|| "max_model_len=unknown".to_string());
                lines.push(format!(
                    "    • KV pool has room ({pct:.0}%), but --max-num-seqs is at the physics ceiling for {m}. Lower --max-model-len to safely raise concurrency, or add a replica."
                ));
            } else {
                match kv_max_seqs {
                    Some(_) => lines.push(format!(
                        "    • Raise --max-num-seqs above {max_str} (KV cache {pct:.0}% used, pool has room)"
                    )),
                    None => lines.push(format!(
                        "    • Raise --max-num-seqs above {max_str} if KV headroom confirmed (ceiling unknown)"
                    )),
                }
            }
        }
        Some(pct) => {
            lines.push(format!(
                "    • KV at {pct:.0}%: scheduler at cap, pool full. No config change helps."
            ));
            lines.push("    • Add a replica to scale out.".to_string());
            lines.push(format!(
                "    • Lower --max-model-len{} to free KV blocks.",
                model_len_suffix(max_model_len)
            ));
        }
        None => {
            if kv_max_seqs
                .is_some_and(|ceiling| d.max_num_seqs.is_some_and(|current| current >= ceiling))
            {
                let m = max_model_len
                    .map(|n| format!("max_model_len={n}"))
                    .unwrap_or_else(|| "max_model_len=unknown".to_string());
                lines.push(format!(
                    "    • KV unknown, but --max-num-seqs is at the physics ceiling for {m}. Lower --max-model-len to safely raise concurrency, or add a replica."
                ));
            } else {
                lines.push(format!(
                    "    • Raise --max-num-seqs above {max_str} if KV cache has headroom"
                ));
            }
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
    kv_max_seqs: Option<u32>,
) -> Vec<String> {
    let mut lines = format_concurrency_saturation_issue(d, max_model_len, kv_max_seqs);
    lines.insert(1, format!("  Seen in {seen_pct}% of windows"));
    lines
}

pub fn r5_recommendation(
    snapshot: &RawSnapshot,
    kv_cache_usage_perc: Option<f64>,
    config_max_num_seqs: Option<u32>,
    max_model_len: Option<u32>,
    kv_max_seqs: Option<u32>,
) -> Option<Recommendation> {
    let d = rule5_concurrency_saturation(snapshot, kv_cache_usage_perc, config_max_num_seqs)?;
    Some(Recommendation {
        rule_name: "concurrency_saturation",
        impact: 4,
        confidence: match (d.ttft_ms.or(d.ttft_p99_ms), d.kv_cache_usage_perc) {
            (Some(_), Some(_)) => 0.9,
            _ => 0.6,
        },
        action: r5_action(&d, kv_max_seqs, max_model_len),
        short_action: r5_short_action(&d, kv_max_seqs, max_model_len),
        expected_impact: "Queue drains, TTFT recovers.".to_string(),
        display_lines: format_concurrency_saturation_issue(&d, max_model_len, kv_max_seqs),
    })
}

pub(super) fn r5_short_action(
    d: &ConcurrencySaturationDetail,
    kv_max_seqs: Option<u32>,
    max_model_len: Option<u32>,
) -> String {
    let kv_safe = d
        .kv_cache_usage_perc
        .is_none_or(|pct| pct < KV_CACHE_SAFE_TO_SCALE_PCT);

    if !kv_safe {
        return "add a replica to scale out".to_string();
    }

    if let (Some(ceiling), Some(current)) = (kv_max_seqs, d.max_num_seqs) {
        if current >= ceiling {
            let m = max_model_len
                .map(|n| n.to_string())
                .unwrap_or_else(|| "?".to_string());
            return format!("lower --max-model-len (at physics ceiling for max_model_len={m})");
        }
    }

    let max_label = d
        .max_num_seqs
        .map(|n| n.to_string())
        .unwrap_or_else(|| "current cap".to_string());
    format!("raise --max-num-seqs above {max_label}")
}

pub(super) fn aggregate_concurrency_saturation_detail(
    details: &[ConcurrencySaturationDetail],
    session_kv_peak: Option<f64>,
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
    let ttft_p99_vecs: Vec<&[crate::collectors::HistogramCount]> = details
        .iter()
        .map(|d| d.ttft_p99_buckets.as_slice())
        .collect();
    let merged_ttft = crate::collectors::merge_p99_bucket_vecs(&ttft_p99_vecs);
    let ttft_p99_ms =
        crate::collectors::vllm::histogram_quantile(0.99, &merged_ttft).map(|s| s * 1000.0);
    // session_kv_peak: global peak across all evaluable windows (kv_cache_peak_perc preferred).
    // Supersedes the per-detail values which are bounded to r5-firing windows only.
    // Falls back to the r5-window peak if caller has no session data (e.g. single-window path).
    let kv_cache_usage_perc = session_kv_peak.or_else(|| {
        details
            .iter()
            .filter_map(|d| d.kv_cache_usage_perc)
            .reduce(f64::max)
    });
    Some(ConcurrencySaturationDetail {
        requests_running: run,
        requests_waiting: wait,
        max_num_seqs: max_seqs,
        queue_ratio: ratio,
        ttft_ms,
        ttft_p99_ms,
        ttft_p99_buckets: merged_ttft,
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
            ttft_p99_ms: None,
            ttft_p99_buckets: vec![],
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
        let text = format_concurrency_saturation_issue(&fired_detail(None, Some(70.0)), None, None)
            .join("\n");
        assert!(text.contains("ceiling unknown"));
        assert!(!text.contains("pool has room"));
    }

    #[test]
    fn fix_shows_physics_ceiling_when_kv_low_and_cap_at_ceiling() {
        let mut d = fired_detail(None, Some(8.0));
        d.max_num_seqs = Some(13);
        let text = format_concurrency_saturation_issue(&d, Some(8192), Some(13)).join("\n");
        assert!(text.contains("Lower --max-model-len"));
        assert!(text.contains("max_model_len=8192"));
    }

    #[test]
    fn fix_scales_out_when_kv_at_or_above_safe_threshold() {
        let text = format_concurrency_saturation_issue(&fired_detail(None, Some(85.0)), None, None)
            .join("\n");
        assert!(text.contains("KV at 85%"));
        assert!(text.contains("No config change helps"));
        assert!(text.contains("Add a replica to scale out"));
    }

    #[test]
    fn fix_generic_when_kv_usage_unknown() {
        let text =
            format_concurrency_saturation_issue(&fired_detail(None, None), None, None).join("\n");
        assert!(text.contains("if KV cache has headroom"));
        assert!(!text.contains("Add a replica"));
    }

    #[test]
    fn confidence_high_when_ttft_and_kv_present() {
        let text = format_concurrency_saturation_issue(
            &fired_detail(Some(5000.0), Some(70.0)),
            None,
            None,
        )
        .join("\n");
        assert!(text.contains("Confidence: High"));
    }

    #[test]
    fn confidence_medium_when_ttft_or_kv_missing() {
        let text =
            format_concurrency_saturation_issue(&fired_detail(Some(5000.0), None), None, None)
                .join("\n");
        assert!(text.contains("Confidence: Medium"));
        let text2 =
            format_concurrency_saturation_issue(&fired_detail(None, Some(70.0)), None, None)
                .join("\n");
        assert!(text2.contains("Confidence: Medium"));
    }

    #[test]
    fn fix_shows_max_model_len_when_kv_high() {
        let text =
            format_concurrency_saturation_issue(&fired_detail(None, Some(85.0)), Some(8192), None)
                .join("\n");
        assert!(text.contains("--max-model-len (currently 8192)"));
    }

    #[test]
    fn cause_shows_ttft_when_available() {
        let text =
            format_concurrency_saturation_issue(&fired_detail(Some(5000.0), None), None, None)
                .join("\n");
        assert!(text.contains("TTFT 5.0s"));
    }

    #[test]
    fn cause_shows_ttft_p99_primary_when_both_available() {
        let mut d = fired_detail(Some(5000.0), None);
        d.ttft_p99_ms = Some(12400.0);
        let text = format_concurrency_saturation_issue(&d, None, None).join("\n");
        assert!(text.contains("TTFT 12.4s p99 (5.0s avg)"));
    }

    #[test]
    fn cause_shows_ttft_p99_only_when_mean_missing() {
        let mut d = fired_detail(None, None);
        d.ttft_p99_ms = Some(12400.0);
        let text = format_concurrency_saturation_issue(&d, None, None).join("\n");
        assert!(text.contains("TTFT 12.4s p99"));
        assert!(!text.contains("avg"));
    }

    #[test]
    fn cause_omits_ttft_when_none() {
        let text =
            format_concurrency_saturation_issue(&fired_detail(None, None), None, None).join("\n");
        assert!(!text.contains("requests queued ahead"));
    }

    #[test]
    fn aggregate_max_num_seqs_is_option_not_zero_sentinel() {
        let agg = aggregate_concurrency_saturation_detail(&[fired_detail(None, None)], None)
            .expect("agg");
        assert_eq!(agg.max_num_seqs, Some(32));
    }

    #[test]
    fn aggregate_uses_merged_buckets_not_average() {
        use crate::collectors::HistogramCount;

        let mut d1 = fired_detail(None, None);
        d1.ttft_p99_ms = Some(99.0);
        d1.ttft_p99_buckets = vec![
            HistogramCount {
                less_than: 0.1,
                count: 100.0,
            },
            HistogramCount {
                less_than: 0.2,
                count: 100.0,
            },
            HistogramCount {
                less_than: f64::INFINITY,
                count: 100.0,
            },
        ];
        let mut d2 = fired_detail(None, None);
        d2.ttft_p99_ms = Some(199.0);
        d2.ttft_p99_buckets = vec![
            HistogramCount {
                less_than: 0.1,
                count: 0.0,
            },
            HistogramCount {
                less_than: 0.2,
                count: 100.0,
            },
            HistogramCount {
                less_than: f64::INFINITY,
                count: 100.0,
            },
        ];
        let agg = aggregate_concurrency_saturation_detail(&[d1, d2], None).expect("agg");
        let p99 = agg.ttft_p99_ms.expect("merged p99");
        // Merged: 200 obs, p99 ≈ 198ms. Simple average of 99ms and 199ms would be 149ms.
        assert!((p99 - 198.0).abs() < 1.0);
        assert!((p99 - 149.0).abs() > 10.0);
    }

    #[test]
    fn aggregate_r5_kv_falls_back_to_r5_detail_peak_without_session_context() {
        // Peak (not average) — a spike must block a "safe to raise" recommendation
        // even if KV drained by end of session.
        let d1 = fired_detail(None, Some(60.0));
        let d2 = fired_detail(None, Some(95.0));
        let d3 = fired_detail(None, Some(70.0));
        let agg = aggregate_concurrency_saturation_detail(&[d1, d2, d3], None).expect("agg");
        assert_eq!(agg.kv_cache_usage_perc, Some(95.0));
    }

    #[test]
    fn aggregate_r5_kv_prefers_session_peak_over_detail_peaks() {
        let d1 = fired_detail(None, Some(60.0));
        let d2 = fired_detail(None, Some(70.0));
        let agg = aggregate_concurrency_saturation_detail(&[d1, d2], Some(95.0)).expect("agg");
        assert_eq!(agg.kv_cache_usage_perc, Some(95.0));
    }

    #[test]
    fn window_issue_inserts_seen_pct() {
        let lines =
            format_concurrency_saturation_window_issue(&fired_detail(None, None), 40, None, None);
        assert_eq!(lines[1], "  Seen in 40% of windows");
    }

    #[test]
    fn short_action_raises_cap_when_kv_safe() {
        let d = fired_detail(None, Some(70.0));
        assert_eq!(
            r5_short_action(&d, None, None),
            "raise --max-num-seqs above 32"
        );
        let r = r5_recommendation(
            &snap(sat_vllm(32.0, 15.0, Some(32))),
            Some(70.0),
            None,
            None,
            None,
        )
        .expect("fired");
        assert_eq!(r.short_action, "raise --max-num-seqs above 32");
    }

    #[test]
    fn short_action_scales_out_when_kv_not_safe() {
        let d = fired_detail(None, Some(85.0));
        assert_eq!(
            r5_short_action(&d, None, None),
            "add a replica to scale out"
        );
        let r = r5_recommendation(
            &snap(sat_vllm(32.0, 15.0, Some(32))),
            Some(85.0),
            None,
            None,
            None,
        )
        .expect("fired");
        assert_eq!(r.short_action, "add a replica to scale out");
        assert_eq!(r.action, "Add a replica to scale out.");
    }

    #[test]
    fn action_at_physics_ceiling_when_kv_has_room_but_max_num_seqs_at_cap() {
        let r = r5_recommendation(
            &snap(sat_vllm(15.0, 10.0, Some(15))),
            Some(50.0),
            None,
            Some(8192),
            Some(15),
        )
        .expect("fired");
        assert!(r.action.contains("physics ceiling for max_model_len"));
    }

    #[test]
    fn action_raises_max_num_seqs_when_headroom_below_ceiling() {
        let r = r5_recommendation(
            &snap(sat_vllm(10.0, 10.0, Some(10))),
            Some(50.0),
            None,
            None,
            Some(15),
        )
        .expect("fired");
        assert!(r.action.contains("Raise --max-num-seqs"));
    }

    #[test]
    fn action_at_physics_ceiling_when_kv_unknown() {
        let d = ConcurrencySaturationDetail {
            requests_running: 15.0,
            requests_waiting: 10.0,
            max_num_seqs: Some(15),
            queue_ratio: 10.0 / 25.0,
            ttft_ms: None,
            ttft_p99_ms: None,
            ttft_p99_buckets: vec![],
            kv_cache_usage_perc: None,
        };
        let action = r5_action(&d, Some(15), Some(8192));
        assert!(action.contains("max_model_len=8192"));
        assert!(!action.contains("Raise --max-num-seqs"));
        let short = r5_short_action(&d, Some(15), Some(8192));
        assert!(short.contains("max_model_len=8192"));
    }

    #[test]
    fn short_action_at_physics_ceiling_recommends_lower_max_model_len() {
        let d = ConcurrencySaturationDetail {
            requests_running: 15.0,
            requests_waiting: 10.0,
            max_num_seqs: Some(15),
            queue_ratio: 10.0 / 25.0,
            ttft_ms: None,
            ttft_p99_ms: None,
            ttft_p99_buckets: vec![],
            kv_cache_usage_perc: Some(50.0),
        };
        let action = r5_short_action(&d, Some(15), Some(8192));
        assert!(
            action.contains("max_model_len=8192"),
            "short_action must not say 'raise --max-num-seqs' when at physics ceiling"
        );
        assert!(
            !action.contains("raise --max-num-seqs"),
            "contradicts r5_action physics ceiling message"
        );
    }

    #[test]
    fn short_action_raises_max_num_seqs_when_below_ceiling() {
        let d = ConcurrencySaturationDetail {
            requests_running: 10.0,
            requests_waiting: 10.0,
            max_num_seqs: Some(10),
            queue_ratio: 0.5,
            ttft_ms: None,
            ttft_p99_ms: None,
            ttft_p99_buckets: vec![],
            kv_cache_usage_perc: Some(50.0),
        };
        let action = r5_short_action(&d, Some(15), Some(8192));
        assert!(action.contains("raise --max-num-seqs above 10"));
    }
}
