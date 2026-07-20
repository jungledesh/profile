use crate::collectors::RawSnapshot;
use crate::fmt::fmt_seconds_from_ms;

#[cfg(test)]
use super::Recommendation;
#[cfg(test)]
use super::rule_names;

/// Minimum ratio of (waiting / total active) confirming the cap is structurally bottlenecking.
/// No industry standard exists; 0.30 is a judgment call. At 30%, nearly 1 in 3 active
/// requests is halted. Below this, queue depth is likely a transient micro-burst, not structural.
const CONCURRENCY_SATURATION_QUEUE_RATIO_MIN: f64 = 0.30;

/// Absolute floor on waiting requests. Below this, queue depth is gauge noise,
/// not structural saturation. Matches R1 and R2's established floor of 2.0.
const CONCURRENCY_SATURATION_WAITING_MIN: f64 = 2.0;

/// KV cache usage below this means the pool has room to absorb new sequences safely.
/// Buffered 8 points below r2's 88% warning threshold; raising --max-num-seqs at 80%
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

/// Chunked prefill can batch `running` above `max_num_seqs` across steps; cap is not the constraint.
const CHUNKED_PREFILL_TOLERANCE: f64 = 0.5;

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
    // run > max_seqs means chunked prefill is batching across steps; cap is not the constraint.
    if (run - f64::from(max_seqs)).abs() > CHUNKED_PREFILL_TOLERANCE {
        return None;
    }
    let wait = snapshot
        .vllm
        .num_requests_waiting
        .filter(|v| v.is_finite())?;
    if wait < CONCURRENCY_SATURATION_WAITING_MIN {
        return None;
    }
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

/// Confidence value for the R5 recommendation. Low (0.5) when the binding wall is
/// empirical; else the existing High/Medium from TTFT+KV evidence.
pub(super) fn r5_confidence(d: &ConcurrencySaturationDetail, empirical: bool) -> f64 {
    if empirical {
        return 0.5;
    }
    match (d.ttft_ms.or(d.ttft_p99_ms), d.kv_cache_usage_perc) {
        (Some(_), Some(_)) => 0.9,
        _ => 0.6,
    }
}

fn r5_confidence_label(empirical: bool, d: &ConcurrencySaturationDetail) -> &'static str {
    if empirical {
        return "Low";
    }
    match (d.ttft_ms.or(d.ttft_p99_ms), d.kv_cache_usage_perc) {
        (Some(_), Some(_)) => "High",
        _ => "Medium",
    }
}

/// Build the safe + cuts fix bullets from the resolved walls. `kv_usage` is the
/// known KV usage percent (A path) or `None` when KV usage is unavailable.
fn walls_fix_lines(
    d: &ConcurrencySaturationDetail,
    rec: Option<&super::RecommendedSeqs>,
    kv_usage: Option<f64>,
    max_model_len: Option<u32>,
    snapshot: &RawSnapshot,
    hyp: Option<&super::HypCapacityCtx<'_>>,
) -> (Vec<String>, Vec<String>) {
    use super::{BindingWall, KvBoundSource};

    let mut safe = Vec::new();
    let mut cuts = Vec::new();

    let cur = d.max_num_seqs;
    // No wall known: honest conditional line, never invent a ceiling number.
    let Some(rec) = rec else {
        match (cur, kv_usage) {
            (Some(c), Some(pct)) => safe.push(format!(
                "      • Raise --max-num-seqs above {c} (KV cache {pct:.0}% used, pool has room; no ceiling known)"
            )),
            (Some(c), None) => safe.push(format!(
                "      • Raise --max-num-seqs above {c} if KV headroom confirmed (no ceiling known)"
            )),
            (None, Some(pct)) => safe.push(format!(
                "      • Raise --max-num-seqs (KV cache {pct:.0}% used, pool has room; no ceiling known)"
            )),
            (None, None) => safe.push(
                "      • Raise --max-num-seqs if KV headroom confirmed (no ceiling known)".to_string(),
            ),
        }
        return (safe, cuts);
    };
    let Some(cur) = cur else {
        safe.push(
            "      • Raise --max-num-seqs if KV headroom confirmed (no ceiling known)".to_string(),
        );
        return (safe, cuts);
    };

    if rec.target > cur {
        // Bounded raise. Empirical: "(est)" only; step cap is silent.
        let reason = match rec.binder {
            BindingWall::Ridge | BindingWall::Config => {
                format!("80% of compute ridge ~{:.0}", rec.wall)
            }
            BindingWall::Memory { cap } => match rec.source {
                Some(KvBoundSource::Observed) => {
                    format!("80% of memory limit {cap}, vLLM-reported")
                }
                Some(KvBoundSource::Derived) | Some(KvBoundSource::DerivedHybrid) => {
                    format!("80% of memory limit ~{cap}, est")
                }
                Some(KvBoundSource::Empirical) => "est".to_string(),
                None => format!("80% of memory limit {cap}"),
            },
        };
        safe.push(format!(
            "      • Raise --max-num-seqs to {} ({reason})",
            rec.target
        ));
        if rec.empirical {
            safe.push("      • Monitor KV cache when scaling up.".to_string());
        }
        return (safe, cuts);
    }

    // target <= current: no knob. "at" only when current >= wall.
    let at_wall = f64::from(cur) >= rec.wall;
    match rec.binder {
        BindingWall::Ridge | BindingWall::Config => {
            let zone = if at_wall {
                format!("at compute ridge (~{:.0})", rec.wall)
            } else {
                format!("within safety margin of compute ridge (~{:.0})", rec.wall)
            };
            safe.push(format!(
                "      • --max-num-seqs {cur} is {zone}. Raising adds TPOT, not throughput."
            ));
            safe.push("      • Add a replica to scale out.".to_string());
        }
        BindingWall::Memory { cap } => {
            let limit = match rec.source {
                Some(KvBoundSource::Observed) => format!("memory limit ({cap}, vLLM-reported)"),
                Some(KvBoundSource::Derived) | Some(KvBoundSource::DerivedHybrid) => {
                    format!("memory limit (~{cap}, est)")
                }
                Some(KvBoundSource::Empirical) => format!("memory limit (~{cap}, est)"),
                None => format!("memory limit ({cap})"),
            };
            let zone = if at_wall {
                format!("at {limit}")
            } else {
                format!("within safety margin of {limit}")
            };
            let prefix = match kv_usage {
                Some(pct) => format!("KV pool has room ({pct:.0}%), but"),
                None => "KV unknown, but".to_string(),
            };
            if at_wall {
                safe.push(format!("      • {prefix} --max-num-seqs {cur} is {zone}."));
            } else {
                // Current stays in the cause line; within-margin memory omits it here.
                safe.push(format!("      • {prefix} --max-num-seqs is {zone}."));
            }
            let shrink = super::model_len_shrink_suggestion_lines(
                max_model_len,
                snapshot.vllm.prompt_tokens_p99,
                snapshot.vllm.generation_tokens_p99,
                snapshot.vllm.generation_tokens_completed.unwrap_or(0.0),
                "      ",
                hyp,
                false,
            );
            super::extend_with_shrink_suggestion(&mut cuts, shrink);
        }
    }
    (safe, cuts)
}

pub(super) fn format_concurrency_saturation_issue(
    d: &ConcurrencySaturationDetail,
    max_model_len: Option<u32>,
    rec: Option<&super::RecommendedSeqs>,
    snapshot: &RawSnapshot,
    hyp: Option<&super::HypCapacityCtx<'_>>,
) -> Vec<String> {
    let max_str = d
        .max_num_seqs
        .map(|n| n.to_string())
        .unwrap_or_else(|| "?".to_string());
    let empirical = rec.is_some_and(|r| r.empirical);

    // Prefer snapshot for display; header reads the same source.
    let display_run = snapshot
        .vllm
        .num_requests_running
        .filter(|v| v.is_finite())
        .unwrap_or(d.requests_running);
    let display_wait = snapshot
        .vllm
        .num_requests_waiting
        .filter(|v| v.is_finite())
        .unwrap_or(d.requests_waiting);
    let display_queue_pct = d.queue_ratio * 100.0;
    let display_kv = snapshot
        .vllm
        .kv_cache_usage_perc
        .filter(|v| v.is_finite())
        .or(d.kv_cache_usage_perc);
    // p95 from snapshot (matches header label); fall back to p99 from d if absent.
    let display_p_x = snapshot
        .vllm
        .ttft_p95_ms
        .filter(|t| t.is_finite())
        .map(|v| (v, "p95"))
        .or_else(|| d.ttft_p99_ms.filter(|t| t.is_finite()).map(|v| (v, "p99")));
    let display_avg = snapshot
        .vllm
        .ttft_ms
        .filter(|t| t.is_finite())
        .or_else(|| d.ttft_ms.filter(|t| t.is_finite()));

    let mut lines = vec![
        "[!] Concurrency Saturation".to_string(),
        String::new(),
        "    Cause:".to_string(),
        format!("      • --max-num-seqs={max_str} hit: scheduler won't admit more sequences"),
        format!(
            "      • {:.0}% of requests waiting ({:.0} waiting, {:.0} running)",
            display_queue_pct, display_wait, display_run
        ),
    ];
    match (display_p_x, display_avg) {
        (Some((px, label)), Some(avg)) => lines.push(format!(
            "      • TTFT ({} {}) ({} avg)",
            label,
            fmt_seconds_from_ms(px),
            fmt_seconds_from_ms(avg)
        )),
        (Some((px, label)), None) => {
            lines.push(format!(
                "      • TTFT ({} {})",
                label,
                fmt_seconds_from_ms(px)
            ));
        }
        (None, Some(avg)) => lines.push(format!("      • TTFT {}", fmt_seconds_from_ms(avg))),
        (None, None) => {}
    }
    lines.push(String::new());
    match display_kv {
        Some(pct) if pct < KV_CACHE_SAFE_TO_SCALE_PCT => {
            let (safe, cuts) = walls_fix_lines(d, rec, Some(pct), max_model_len, snapshot, hyp);
            super::push_grouped_fixes(&mut lines, safe, cuts, Vec::new());
        }
        Some(pct) => {
            // KV >= safe-to-scale gate: pool full, no config change helps. Scale out.
            super::push_grouped_fixes(
                &mut lines,
                vec![
                    format!(
                        "      • KV at {pct:.0}%: scheduler at cap, pool full. No config change helps."
                    ),
                    "      • Add a replica to scale out.".to_string(),
                ],
                Vec::new(),
                Vec::new(),
            );
        }
        None => {
            let (safe, cuts) = walls_fix_lines(d, rec, None, max_model_len, snapshot, hyp);
            super::push_grouped_fixes(&mut lines, safe, cuts, Vec::new());
        }
    }
    lines.push(String::new());
    lines.push("    Expected: Queue drains, TTFT recovers.".to_string());
    lines.push(format!(
        "    Confidence: {}",
        r5_confidence_label(empirical, d)
    ));
    lines
}

pub(super) fn format_concurrency_saturation_window_issue(
    d: &ConcurrencySaturationDetail,
    seen_pct: u32,
    max_model_len: Option<u32>,
    rec: Option<&super::RecommendedSeqs>,
    snapshot: &RawSnapshot,
    hyp: Option<&super::HypCapacityCtx<'_>>,
) -> Vec<String> {
    super::with_seen_pct(
        format_concurrency_saturation_issue(d, max_model_len, rec, snapshot, hyp),
        seen_pct,
    )
}

#[cfg(test)]
pub fn r5_recommendation(
    snapshot: &RawSnapshot,
    kv_cache_usage_perc: Option<f64>,
    config_max_num_seqs: Option<u32>,
    max_model_len: Option<u32>,
    rec: Option<super::RecommendedSeqs>,
) -> Option<Recommendation> {
    let d = rule5_concurrency_saturation(snapshot, kv_cache_usage_perc, config_max_num_seqs)?;
    let empirical = rec.is_some_and(|r| r.empirical);
    Some(Recommendation {
        rule_name: rule_names::CONCURRENCY_SATURATION,
        layer: 3,
        impact: 4,
        confidence: r5_confidence(&d, empirical),
        display_lines: format_concurrency_saturation_issue(
            &d,
            max_model_len,
            rec.as_ref(),
            snapshot,
            None,
        ),
    })
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
    use crate::collectors::VllmRawMetrics;
    use crate::engine::rules::{
        KvBoundSource, RecommendedSeqs, empirical_kv_max, recommended_seqs, resolve_kv_bound,
    };

    fn snap(vllm: VllmRawMetrics) -> RawSnapshot {
        crate::collectors::snap_vllm(vllm)
    }

    /// Build a margined recommendation from the two physical walls via the shared helper.
    fn rec_for(
        ridge: Option<f64>,
        kv_bound: Option<f64>,
        source: Option<KvBoundSource>,
        current: u32,
    ) -> RecommendedSeqs {
        recommended_seqs(ridge, kv_bound, source, Some(current)).expect("rec")
    }

    /// A fired R5 detail with an explicit current cap, KV usage, and TTFT.
    fn detail_at(
        max_num_seqs: u32,
        kv_cache_usage_perc: Option<f64>,
        ttft_ms: Option<f64>,
    ) -> ConcurrencySaturationDetail {
        let cur = f64::from(max_num_seqs);
        ConcurrencySaturationDetail {
            requests_running: cur,
            requests_waiting: 15.0,
            max_num_seqs: Some(max_num_seqs),
            queue_ratio: 15.0 / (15.0 + cur),
            ttft_ms,
            ttft_p99_ms: None,
            ttft_p99_buckets: vec![],
            kv_cache_usage_perc,
        }
    }

    fn blank_snap() -> RawSnapshot {
        snap(VllmRawMetrics::default())
    }

    fn model_len_snap(pp: f64, gp: f64, completed: f64) -> RawSnapshot {
        snap(VllmRawMetrics {
            prompt_tokens_p99: Some(pp),
            generation_tokens_p99: Some(gp),
            generation_tokens_completed: Some(completed),
            ..Default::default()
        })
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
    fn does_not_fire_when_waiting_below_2() {
        assert!(
            rule5_concurrency_saturation(&snap(sat_vllm(32.0, 1.5, Some(32))), Some(70.0), None)
                .is_none()
        );
    }

    #[test]
    fn fires_when_waiting_at_2() {
        let d = rule5_concurrency_saturation(&snap(sat_vllm(4.0, 2.0, Some(4))), Some(70.0), None)
            .expect("fired");
        assert_eq!(d.max_num_seqs, Some(4));
        assert!((d.queue_ratio - (2.0 / 6.0)).abs() < 1e-9);
        assert_eq!(d.kv_cache_usage_perc, Some(70.0));
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
        // No wall known: honest conditional line naming current + KV usage.
        let text = format_concurrency_saturation_issue(
            &fired_detail(None, Some(70.0)),
            None,
            None,
            &blank_snap(),
            None,
        )
        .join("\n");
        assert!(text.contains(
            "Raise --max-num-seqs above 32 (KV cache 70% used, pool has room; no ceiling known)"
        ));
    }

    #[test]
    fn fix_shows_physics_ceiling_when_kv_low_and_cap_at_ceiling() {
        // Derived memory wall, current at the wall: shrink pivot + memory limit (est).
        let d = detail_at(13, Some(8.0), None);
        let rec = rec_for(None, Some(13.0), Some(KvBoundSource::Derived), 13);
        let text = format_concurrency_saturation_issue(
            &d,
            Some(8192),
            Some(&rec),
            &model_len_snap(6000.0, 450.0, 150.0),
            None,
        )
        .join("\n");
        assert!(text.contains("at memory limit (~13, est)"));
        assert!(text.contains("Lower --max-model-len 8192 → 6450"));
        assert!(text.contains("Truncation risk"));
        assert!(!text.contains("free KV blocks"));
    }

    #[test]
    fn fix_scales_out_when_kv_at_or_above_safe_threshold() {
        let text = format_concurrency_saturation_issue(
            &fired_detail(None, Some(85.0)),
            Some(8192),
            None,
            &blank_snap(),
            None,
        )
        .join("\n");
        assert!(text.contains("KV at 85%"));
        assert!(text.contains("No config change helps"));
        assert!(text.contains("Add a replica to scale out"));
        assert!(!text.contains("free KV blocks"));
        assert!(!text.contains("Warning: Truncation risk!"));
    }

    #[test]
    fn fix_generic_when_kv_usage_unknown() {
        let text = format_concurrency_saturation_issue(
            &fired_detail(None, None),
            None,
            None,
            &blank_snap(),
            None,
        )
        .join("\n");
        assert!(text.contains("if KV headroom confirmed (no ceiling known)"));
        assert!(!text.contains("Add a replica"));
    }

    #[test]
    fn confidence_high_when_ttft_and_kv_present() {
        let text = format_concurrency_saturation_issue(
            &fired_detail(Some(5000.0), Some(70.0)),
            None,
            None,
            &blank_snap(),
            None,
        )
        .join("\n");
        assert!(text.contains("Confidence: High"));
    }

    #[test]
    fn confidence_medium_when_ttft_or_kv_missing() {
        let text = format_concurrency_saturation_issue(
            &fired_detail(Some(5000.0), None),
            None,
            None,
            &blank_snap(),
            None,
        )
        .join("\n");
        assert!(text.contains("Confidence: Medium"));
        let text2 = format_concurrency_saturation_issue(
            &fired_detail(None, Some(70.0)),
            None,
            None,
            &blank_snap(),
            None,
        )
        .join("\n");
        assert!(text2.contains("Confidence: Medium"));
    }

    #[test]
    fn fix_shows_max_model_len_when_kv_high() {
        let text = format_concurrency_saturation_issue(
            &fired_detail(None, Some(85.0)),
            Some(8192),
            None,
            &blank_snap(),
            None,
        )
        .join("\n");
        assert!(text.contains("Add a replica to scale out"));
        assert!(!text.contains("--max-model-len"));
    }

    #[test]
    fn cause_waiting_line_shows_wait_and_run_explicitly() {
        let mut d = fired_detail(None, None);
        d.max_num_seqs = Some(13);
        d.requests_running = 13.0;
        d.requests_waiting = 237.0;
        d.queue_ratio = 237.0 / 250.0;
        let s = snap(VllmRawMetrics {
            num_requests_running: Some(13.0),
            num_requests_waiting: Some(237.0),
            max_num_seqs: Some(13),
            ..Default::default()
        });
        let text = format_concurrency_saturation_issue(&d, None, None, &s, None).join("\n");
        assert!(text.contains("237 waiting, 13 running"));
        assert!(!text.contains("of 13 active"));
        assert!(!text.contains("of 250 active"));
    }

    #[test]
    fn cause_shows_ttft_when_available() {
        let text = format_concurrency_saturation_issue(
            &fired_detail(Some(5000.0), None),
            None,
            None,
            &blank_snap(),
            None,
        )
        .join("\n");
        assert!(text.contains("TTFT 5.0s"));
    }

    #[test]
    fn cause_falls_back_to_ttft_p99_when_snapshot_has_no_p95() {
        let mut d = fired_detail(Some(5000.0), None);
        d.ttft_p99_ms = Some(12400.0);
        let text =
            format_concurrency_saturation_issue(&d, None, None, &blank_snap(), None).join("\n");
        // snapshot has no p95 → falls back to d.ttft_p99_ms
        assert!(text.contains("TTFT (p99 12.4s) (5.0s avg)"));
    }

    #[test]
    fn cause_shows_snapshot_p95_over_d_ttft_p99() {
        let mut d = fired_detail(None, None);
        d.ttft_p99_ms = Some(12400.0); // aggregate: 12.4s p99
        let s = snap(VllmRawMetrics {
            ttft_p95_ms: Some(9500.0), // snapshot: 9.5s p95
            ttft_ms: Some(5000.0),     // snapshot avg: 5.0s
            ..Default::default()
        });
        let text = format_concurrency_saturation_issue(&d, None, None, &s, None).join("\n");
        assert!(
            text.contains("TTFT (p95 9.5s)"),
            "snapshot p95 must win over d.ttft_p99"
        );
        assert!(
            !text.contains("12.4s"),
            "d.ttft_p99 must not appear when snapshot has p95"
        );
    }

    #[test]
    fn fix_uses_snapshot_kv_not_aggregate_peak() {
        // d has KV 85% (>= safe threshold) → would say "Add a replica"
        // snapshot has KV 70% (< safe threshold) → snapshot wins, should say "Raise --max-num-seqs"
        let d = fired_detail(None, Some(85.0));
        let s = snap(VllmRawMetrics {
            kv_cache_usage_perc: Some(70.0),
            ..Default::default()
        });
        let text = format_concurrency_saturation_issue(&d, None, None, &s, None).join("\n");
        assert!(
            text.contains("Raise --max-num-seqs"),
            "snapshot KV (70%) should flip fix to raise cap"
        );
        assert!(
            !text.contains("Add a replica"),
            "aggregate KV (85%) must not override snapshot"
        );
    }

    #[test]
    fn cause_shows_ttft_p99_only_when_mean_missing() {
        let mut d = fired_detail(None, None);
        d.ttft_p99_ms = Some(12400.0);
        let text =
            format_concurrency_saturation_issue(&d, None, None, &blank_snap(), None).join("\n");
        assert!(text.contains("TTFT (p99 12.4s)"));
        assert!(!text.contains("avg"));
    }

    #[test]
    fn cause_omits_ttft_when_none() {
        let text = format_concurrency_saturation_issue(
            &fired_detail(None, None),
            None,
            None,
            &blank_snap(),
            None,
        )
        .join("\n");
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
        // Peak (not average); a spike must block a "safe to raise" recommendation
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
        let lines = format_concurrency_saturation_window_issue(
            &fired_detail(None, None),
            40,
            None,
            None,
            &blank_snap(),
            None,
        );
        assert_eq!(lines[1], "    Seen in 40% of windows");
    }

    #[test]
    fn display_raises_cap_when_kv_safe() {
        let r = r5_recommendation(
            &snap(sat_vllm(32.0, 15.0, Some(32))),
            Some(70.0),
            None,
            None,
            None,
        )
        .expect("fired");
        let text = r.display_lines.join("\n");
        assert!(text.contains(
            "Raise --max-num-seqs above 32 (KV cache 70% used, pool has room; no ceiling known)"
        ));
    }

    #[test]
    fn display_scales_out_when_kv_not_safe() {
        let r = r5_recommendation(
            &snap(sat_vllm(32.0, 15.0, Some(32))),
            Some(85.0),
            None,
            None,
            None,
        )
        .expect("fired");
        let text = r.display_lines.join("\n");
        assert!(text.contains("Add a replica to scale out."));
    }

    #[test]
    fn display_at_physics_ceiling_when_kv_has_room_but_max_num_seqs_at_cap() {
        let rec = rec_for(None, Some(15.0), Some(KvBoundSource::Derived), 15);
        let r = r5_recommendation(
            &snap(sat_vllm(15.0, 10.0, Some(15))),
            Some(50.0),
            None,
            Some(8192),
            Some(rec),
        )
        .expect("fired");
        let text = r.display_lines.join("\n");
        assert!(text.contains("at memory limit (~15, est)"));
        assert!(!text.contains("Raise --max-num-seqs to"));
    }

    #[test]
    fn display_raises_max_num_seqs_when_headroom_below_ceiling() {
        // Derived memory wall 15, current 10: target 12 > 10, bounded raise.
        let rec = rec_for(None, Some(15.0), Some(KvBoundSource::Derived), 10);
        let r = r5_recommendation(
            &snap(sat_vllm(10.0, 10.0, Some(10))),
            Some(50.0),
            None,
            None,
            Some(rec),
        )
        .expect("fired");
        let text = r.display_lines.join("\n");
        assert!(text.contains("Raise --max-num-seqs to 12 (80% of memory limit ~15, est)"));
    }

    #[test]
    fn display_at_physics_ceiling_when_kv_unknown() {
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
        let rec = rec_for(None, Some(15.0), Some(KvBoundSource::Derived), 15);
        let text =
            format_concurrency_saturation_issue(&d, Some(8192), Some(&rec), &blank_snap(), None)
                .join("\n");
        assert!(text.contains("at memory limit (~15, est)"));
        assert!(!text.contains("Raise --max-num-seqs to"));
    }

    #[test]
    fn display_at_physics_ceiling_does_not_raise_max_num_seqs() {
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
        let rec = rec_for(None, Some(15.0), Some(KvBoundSource::Derived), 15);
        let text =
            format_concurrency_saturation_issue(&d, Some(8192), Some(&rec), &blank_snap(), None)
                .join("\n");
        assert!(
            text.contains("at memory limit (~15, est)"),
            "display must name memory limit when at cap"
        );
        assert!(
            !text.contains("Raise --max-num-seqs to"),
            "must not raise max-num-seqs at memory wall"
        );
    }

    #[test]
    fn display_raises_max_num_seqs_when_below_ceiling() {
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
        let rec = rec_for(None, Some(15.0), Some(KvBoundSource::Derived), 10);
        let text =
            format_concurrency_saturation_issue(&d, Some(8192), Some(&rec), &blank_snap(), None)
                .join("\n");
        assert!(text.contains("Raise --max-num-seqs to 12 (80% of memory limit ~15, est)"));
    }

    #[test]
    fn ceiling_path_shows_shrink_suggestion_with_truncation_warning() {
        let d = detail_at(15, Some(50.0), None);
        let rec = rec_for(None, Some(15.0), Some(KvBoundSource::Derived), 15);
        let lines = format_concurrency_saturation_issue(
            &d,
            Some(8192),
            Some(&rec),
            &model_len_snap(6000.0, 450.0, 150.0),
            None,
        );
        let text = lines.join("\n");
        assert!(text.contains("at memory limit (~15, est)"));
        assert!(text.contains("    Cuts throughput:"));
        let cuts = text.find("    Cuts throughput:").unwrap();
        let shrink = text.find("Lower --max-model-len 8192 → 6450").unwrap();
        assert!(cuts < shrink);
        assert!(text.contains("Truncation risk"));
        let warn_idx = lines
            .iter()
            .position(|l| l.contains("Truncation risk"))
            .expect("warning");
        // Warning is last in Cuts group (trimmed); blank before Expected follows.
        assert!(lines[warn_idx + 1].is_empty());
        assert!(lines[warn_idx + 2].starts_with("    Expected"));
    }

    #[test]
    fn scale_out_path_omits_max_model_len_shrink() {
        let text = format_concurrency_saturation_issue(
            &fired_detail(None, Some(85.0)),
            Some(4096),
            None,
            &blank_snap(),
            None,
        )
        .join("\n");
        assert!(text.contains("Add a replica to scale out"));
        assert!(!text.contains("truncation risk"));
        assert!(!text.contains("free KV blocks"));
    }

    // --- Spec tests: three walls (specs/r5_three_walls.md) ---

    // 1. ridge 153 < kv 240: target 122, ridge phrasing.
    #[test]
    fn spec_ridge_binds_below_kv() {
        let rec = rec_for(Some(153.0), Some(240.0), Some(KvBoundSource::Observed), 32);
        assert_eq!(rec.target, 122);
        let text = format_concurrency_saturation_issue(
            &detail_at(32, Some(70.0), None),
            None,
            Some(&rec),
            &blank_snap(),
            None,
        )
        .join("\n");
        assert!(text.contains("Raise --max-num-seqs to 122 (80% of compute ridge ~153)"));
    }

    // 2. Observed kv 120 < ridge 153: target 96, "vLLM-reported", no tilde, no est.
    #[test]
    fn spec_memory_observed_no_tilde() {
        let rec = rec_for(Some(153.0), Some(120.0), Some(KvBoundSource::Observed), 32);
        assert_eq!(rec.target, 96);
        let text = format_concurrency_saturation_issue(
            &detail_at(32, Some(70.0), None),
            None,
            Some(&rec),
            &blank_snap(),
            None,
        )
        .join("\n");
        assert!(
            text.contains("Raise --max-num-seqs to 96 (80% of memory limit 120, vLLM-reported)")
        );
        assert!(!text.contains("~120"));
    }

    // 3. Derived kv 120 < ridge: target 96, "~120", "(est)".
    #[test]
    fn spec_memory_derived_tilde_and_est() {
        let rec = rec_for(Some(153.0), Some(120.0), Some(KvBoundSource::Derived), 32);
        assert_eq!(rec.target, 96);
        let text = format_concurrency_saturation_issue(
            &detail_at(32, Some(70.0), None),
            None,
            Some(&rec),
            &blank_snap(),
            None,
        )
        .join("\n");
        assert!(text.contains("Raise --max-num-seqs to 96 (80% of memory limit ~120, est)"));
    }

    // 4. current 130, ridge 153, target 122: margin zone, replica, no raise, no shrink.
    #[test]
    fn spec_ridge_margin_zone() {
        let rec = rec_for(Some(153.0), None, None, 130);
        assert_eq!(rec.target, 122);
        let text = format_concurrency_saturation_issue(
            &detail_at(130, Some(50.0), None),
            Some(8192),
            Some(&rec),
            &blank_snap(),
            None,
        )
        .join("\n");
        assert!(text.contains(
            "--max-num-seqs 130 is within safety margin of compute ridge (~153). Raising adds TPOT, not throughput."
        ));
        assert!(text.contains("Add a replica to scale out."));
        assert!(!text.contains("Raise --max-num-seqs to"));
        assert!(!text.contains("Lower --max-model-len"));
        assert!(!text.contains("the safety margin"));
    }

    // 5. current 153 == ridge: "at compute ridge", replica.
    #[test]
    fn spec_ridge_at_wall() {
        let rec = rec_for(Some(153.0), None, None, 153);
        let text = format_concurrency_saturation_issue(
            &detail_at(153, Some(50.0), None),
            None,
            Some(&rec),
            &blank_snap(),
            None,
        )
        .join("\n");
        assert!(text.contains(
            "--max-num-seqs 153 is at compute ridge (~153). Raising adds TPOT, not throughput."
        ));
        assert!(text.contains("Add a replica to scale out."));
        assert!(!text.contains("at the compute"));
    }

    // 6. memory margin Observed / derived labels; shrink under Cuts.
    #[test]
    fn spec_memory_margin_labels() {
        let snapshot = model_len_snap(6000.0, 450.0, 150.0);
        let observed = rec_for(Some(153.0), Some(120.0), Some(KvBoundSource::Observed), 110);
        let text = format_concurrency_saturation_issue(
            &detail_at(110, Some(50.0), None),
            Some(8192),
            Some(&observed),
            &snapshot,
            None,
        )
        .join("\n");
        assert!(text.contains(
            "KV pool has room (50%), but --max-num-seqs is within safety margin of memory limit (120, vLLM-reported)."
        ));
        assert!(!text.contains("physics ceiling"));
        assert!(text.contains("Lower --max-model-len 8192"));

        let derived = rec_for(Some(153.0), Some(120.0), Some(KvBoundSource::Derived), 110);
        let text2 = format_concurrency_saturation_issue(
            &detail_at(110, Some(50.0), None),
            Some(8192),
            Some(&derived),
            &snapshot,
            None,
        )
        .join("\n");
        assert!(text2.contains("within safety margin of memory limit (~120, est)."));
        assert!(!text2.contains("vLLM-reported"));
    }

    // 7. No ridge, no kv_bound: conditional fallback with current, no invented ceiling.
    #[test]
    fn spec_no_wall_conditional() {
        assert!(recommended_seqs(None, None, None, Some(32)).is_none());
        let text = format_concurrency_saturation_issue(
            &detail_at(32, None, None),
            None,
            None,
            &blank_snap(),
            None,
        )
        .join("\n");
        assert!(
            text.contains(
                "Raise --max-num-seqs above 32 if KV headroom confirmed (no ceiling known)"
            )
        );
    }

    // 8. Same inputs through R5 and R7 produce the same target (shared function).
    #[test]
    fn spec_r5_and_r7_same_target() {
        use crate::collectors::CacheConfigLabels;
        use crate::engine::rules::r7_config_headroom::rule7_config_headroom;

        let r5_rec = rec_for(Some(153.0), Some(120.0), Some(KvBoundSource::Observed), 32);
        let mut v = sat_vllm(20.0, 0.0, Some(32));
        v.kv_cache_usage_perc = Some(3.3);
        v.cache_config = CacheConfigLabels {
            kv_cache_max_concurrency: Some(120.0),
            ..Default::default()
        };
        let d = rule7_config_headroom(&snap(v), None, Some(153.0), Some(240), false).expect("r7");
        assert_eq!(r5_rec.target, d.recommended_seqs);
        assert_eq!(r5_rec.target, 96);
    }

    // 9. Empirical used only when Observed and derived are both absent.
    #[test]
    fn spec_empirical_last_resort() {
        let (_, s_obs) = resolve_kv_bound(Some(120.0), Some(200), false, Some(32.0), Some(8.0));
        assert_eq!(s_obs, Some(KvBoundSource::Observed));
        let (_, s_der) = resolve_kv_bound(None, Some(200), false, Some(32.0), Some(8.0));
        assert_eq!(s_der, Some(KvBoundSource::Derived));
        let (v_emp, s_emp) = resolve_kv_bound(None, None, false, Some(32.0), Some(8.0));
        assert_eq!(s_emp, Some(KvBoundSource::Empirical));
        assert!((v_emp.expect("emp") - 400.0).abs() < 1.0);
    }

    // 10. KV >= 80% scale-out path byte-identical regardless of resolved wall.
    #[test]
    fn spec_scale_out_ignores_wall() {
        let d = fired_detail(None, Some(85.0));
        let rec = rec_for(Some(153.0), Some(120.0), Some(KvBoundSource::Observed), 32);
        let with_rec =
            format_concurrency_saturation_issue(&d, Some(8192), Some(&rec), &blank_snap(), None);
        let without_rec =
            format_concurrency_saturation_issue(&d, Some(8192), None, &blank_snap(), None);
        assert_eq!(with_rec, without_rec);
        assert!(with_rec.join("\n").contains("KV at 85%"));
    }

    // 11. Empirical-bound: "(est)" only; step cap silent; Low + Monitor.
    #[test]
    fn spec_empirical_step_cap() {
        let kv_bound = empirical_kv_max(32.0, Some(8.0)).expect("empirical");
        assert!((kv_bound - 400.0).abs() < 1.0);
        let rec = rec_for(None, Some(kv_bound), Some(KvBoundSource::Empirical), 32);
        assert!(rec.empirical);
        assert_eq!(rec.target, 64);
        let d = detail_at(32, Some(8.0), Some(5000.0));
        let text = format_concurrency_saturation_issue(&d, None, Some(&rec), &blank_snap(), None)
            .join("\n");
        assert!(text.contains("Raise --max-num-seqs to 64 (est)"));
        assert!(!text.contains("bounded step"));
        assert!(!text.contains("2x"));
        assert!(text.contains("Monitor KV cache when scaling up."));
        assert!(text.contains("Confidence: Low"));
    }

    // 12. Empirical denominator is run-level peak KV fraction, not the mean.
    #[test]
    fn spec_empirical_uses_peak_not_mean() {
        // Peak 24% (not mean of 8% and 24% = 16%): 32 / 0.24 ~= 133, not 200.
        let (v, src) = resolve_kv_bound(None, None, false, Some(32.0), Some(24.0));
        assert_eq!(src, Some(KvBoundSource::Empirical));
        let bound = v.expect("bound");
        assert!((bound - 133.33).abs() < 1.0);
        assert!((bound - 200.0).abs() > 10.0);
    }

    // 13. Observed-bound target uncapped; normal confidence; no monitor line.
    #[test]
    fn spec_observed_uncapped() {
        let rec = rec_for(None, Some(120.0), Some(KvBoundSource::Observed), 32);
        assert!(!rec.empirical);
        assert_eq!(rec.target, 96); // > 2 x 32, uncapped
        let d = detail_at(32, Some(50.0), Some(5000.0));
        let text = format_concurrency_saturation_issue(&d, None, Some(&rec), &blank_snap(), None)
            .join("\n");
        assert!(
            text.contains("Raise --max-num-seqs to 96 (80% of memory limit 120, vLLM-reported)")
        );
        assert!(!text.contains("Monitor KV cache"));
        assert!(text.contains("Confidence: High"));
    }
}
