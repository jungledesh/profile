use crate::collectors::RawSnapshot;

#[cfg(test)]
use super::Recommendation;
#[cfg(test)]
use super::rule_names;

const PREFIX_HIT_RATE_LT: f64 = 0.35;
const PREFIX_RULE_PROMPT_TOKENS_GTE: f64 = 20.0;
const PREFIX_RULE_RUNNING_GT: f64 = 0.75;
/// Minimum prompt token throughput (QPS × mean_prompt_tokens) to gate R3.
/// Filters sparse cold-cache workloads where 0% hit rate is expected, not actionable.
/// Covers all real use cases: chat (20 QPS × 500 tok), agents (5 QPS × 2k tok),
/// RAG (0.5 QPS × 32k tok). Calibration constant - print live value in output for operator tuning.
const PREFIX_RULE_MIN_PROMPT_TPS: f64 = 1000.0;

#[derive(Debug, Clone, PartialEq)]
pub struct LowPrefixReuseDetail {
    /// None when prefix caching is disabled (path B).
    pub hit_rate: Option<f64>,
    pub prompt_tokens_mean: Option<f64>,
    pub queries_delta: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Rule3Outcome {
    Fired(LowPrefixReuseDetail),
    NotFired,
}

pub fn rule3_low_prefix_reuse(snapshot: &RawSnapshot) -> Rule3Outcome {
    let v = &snapshot.vllm;

    let running = v.num_requests_running.filter(|x| x.is_finite());
    let Some(pm) = v.prompt_tokens_mean.filter(|x| x.is_finite()) else {
        return Rule3Outcome::NotFired;
    };
    // physical constraint first: must be at least one full vLLM cache block (16 tok)
    if pm < PREFIX_RULE_PROMPT_TOKENS_GTE {
        return Rule3Outcome::NotFired;
    }
    let qps = v
        .request_success_per_sec
        .filter(|x| x.is_finite())
        .unwrap_or(0.0);
    if qps * pm < PREFIX_RULE_MIN_PROMPT_TPS {
        return Rule3Outcome::NotFired;
    }
    let Some(rv) = running else {
        return Rule3Outcome::NotFired;
    };
    if rv <= PREFIX_RULE_RUNNING_GT {
        return Rule3Outcome::NotFired;
    }

    let queries_delta = {
        let s = &v.prefix_cache_scrape_samples;
        match (s.first(), s.last()) {
            (Some(f), Some(l)) if s.len() >= 2 => {
                f.queries.zip(l.queries).map(|(q0, q1)| (q1 - q0).max(0.0))
            }
            _ => None,
        }
    };

    if v.cache_config.enable_prefix_caching == Some(false) {
        return Rule3Outcome::Fired(LowPrefixReuseDetail {
            hit_rate: None,
            prompt_tokens_mean: Some(pm),
            queries_delta,
        });
    }

    let Some(hit_rate) = v.prefix_cache_hit_rate.filter(|x| x.is_finite()) else {
        return Rule3Outcome::NotFired;
    };
    if hit_rate >= PREFIX_HIT_RATE_LT {
        return Rule3Outcome::NotFired;
    }
    Rule3Outcome::Fired(LowPrefixReuseDetail {
        hit_rate: Some(hit_rate),
        prompt_tokens_mean: Some(pm),
        queries_delta,
    })
}

#[cfg(test)]
pub fn r3_recommendation(snapshot: &RawSnapshot) -> Option<Recommendation> {
    let Rule3Outcome::Fired(d) = rule3_low_prefix_reuse(snapshot) else {
        return None;
    };
    let enable_prefix = snapshot.vllm.cache_config.enable_prefix_caching;
    let (impact, confidence) = if d.hit_rate.is_none() {
        (3, 0.95_f64)
    } else {
        (2, 0.9_f64)
    };
    Some(Recommendation {
        rule_name: rule_names::LOW_PREFIX_REUSE,
        layer: 5,
        impact,
        confidence,
        // Single-window path has no session context - use hit rate from this window only.
        display_lines: format_low_prefix_hit_rate_fired(&d, enable_prefix, None),
    })
}

pub(super) fn format_low_prefix_hit_rate_fired(
    d: &LowPrefixReuseDetail,
    enable_prefix_caching: Option<bool>,
    session_hit_rate: Option<f64>,
) -> Vec<String> {
    let cause_lines: Vec<String> = if d.hit_rate.is_none() {
        vec![
            "    Cause:".to_string(),
            "      Prefix caching is disabled (enable_prefix_caching=False).".to_string(),
        ]
    } else {
        let hit = session_hit_rate.or(d.hit_rate).unwrap_or(0.0) * 100.0;
        vec![
            "    Cause:".to_string(),
            format!(
                "      Prefix hit rate {hit:.1}% (threshold: {:.0}%).",
                PREFIX_HIT_RATE_LT * 100.0
            ),
        ]
    };

    let fix_lines: Vec<String> = if enable_prefix_caching == Some(false) {
        vec![
            "      • Enable prefix caching: --enable-prefix-caching".to_string(),
            "      • Move shared instructions/system prompts to the very start".to_string(),
            "      • Standardize prompt templates across requests".to_string(),
        ]
    } else {
        vec![
            "      • Move shared instructions/system prompts to the very start".to_string(),
            "      • Standardize prompt templates across requests".to_string(),
            "      • Avoid unique tokens (IDs, timestamps) at the beginning".to_string(),
        ]
    };

    let mut lines = vec!["[!] Low Prefix Cache".to_string(), String::new()];
    lines.extend(cause_lines);
    lines.push(String::new());
    lines.push("    Fix:".to_string());
    lines.extend(fix_lines);
    lines.push(String::new());
    lines.push("    Expected: Higher prefix cache hit rate and lower TTFT.".to_string());
    lines.push("    Confidence: High".to_string());
    lines
}

pub(super) fn format_low_prefix_window_issue(
    d: &LowPrefixReuseDetail,
    seen_pct: u32,
    enable_prefix_caching: Option<bool>,
    session_hit_rate: Option<f64>,
) -> Vec<String> {
    super::with_seen_pct(
        format_low_prefix_hit_rate_fired(d, enable_prefix_caching, session_hit_rate),
        seen_pct,
    )
}

pub(super) fn aggregate_r3_detail(
    details: &[LowPrefixReuseDetail],
    summary: &RawSnapshot,
) -> LowPrefixReuseDetail {
    if details.is_empty() {
        return LowPrefixReuseDetail {
            hit_rate: summary.vllm.prefix_cache_hit_rate.filter(|x| x.is_finite()),
            prompt_tokens_mean: summary.vllm.prompt_tokens_mean,
            queries_delta: None,
        };
    }
    let hit_rate = if details.iter().any(|d| d.hit_rate.is_none()) {
        None
    } else {
        // Weighted by query volume: Σ(rate × queries) / Σqueries
        // Falls back to unweighted if any window is missing query count.
        let weighted: Option<f64> = {
            let mut wsum = 0.0_f64;
            let mut wtotal = 0.0_f64;
            let mut all_have_weight = true;
            for d in details {
                match (d.hit_rate, d.queries_delta.filter(|&q| q > 0.0)) {
                    (Some(r), Some(q)) => {
                        wsum += r * q;
                        wtotal += q;
                    }
                    _ => {
                        all_have_weight = false;
                        break;
                    }
                }
            }
            (all_have_weight && wtotal > 0.0).then_some(wsum / wtotal)
        };
        weighted.or_else(|| {
            // Fallback: unweighted (query counts unavailable)
            let n = details.len() as f64;
            Some(details.iter().filter_map(|d| d.hit_rate).sum::<f64>() / n)
        })
    };
    let prompt_tokens_mean =
        super::mean_of_present(details.iter().filter_map(|d| d.prompt_tokens_mean));
    LowPrefixReuseDetail {
        hit_rate,
        prompt_tokens_mean,
        queries_delta: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{CacheConfigLabels, RawSnapshot, VllmRawMetrics};
    use std::time::SystemTime;

    fn traffic_gates_snap(v: VllmRawMetrics) -> RawSnapshot {
        let t = SystemTime::UNIX_EPOCH;
        RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: v,
            gpus: vec![],
            host_memory: None,
        }
    }

    fn path_b_base_vllm() -> VllmRawMetrics {
        VllmRawMetrics {
            prompt_tokens_mean: Some(64.0),
            num_requests_running: Some(5.0),
            request_success_per_sec: Some(20.0),
            cache_config: CacheConfigLabels {
                enable_prefix_caching: Some(false),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn path_b_fires_when_caching_disabled() {
        let snap = traffic_gates_snap(path_b_base_vllm());
        match rule3_low_prefix_reuse(&snap) {
            Rule3Outcome::Fired(d) => {
                assert!(d.hit_rate.is_none());
                assert_eq!(d.prompt_tokens_mean, Some(64.0));
            }
            Rule3Outcome::NotFired => panic!("expected path B fired"),
        }
    }

    #[test]
    fn path_b_does_not_fire_when_caching_unknown() {
        let mut v = path_b_base_vllm();
        v.cache_config.enable_prefix_caching = None;
        v.prefix_cache_hit_rate = None;
        assert!(matches!(
            rule3_low_prefix_reuse(&traffic_gates_snap(v)),
            Rule3Outcome::NotFired
        ));
    }

    #[test]
    fn path_b_does_not_fire_when_caching_enabled() {
        let mut v = path_b_base_vllm();
        v.cache_config.enable_prefix_caching = Some(true);
        v.prefix_cache_hit_rate = Some(0.50);
        assert!(matches!(
            rule3_low_prefix_reuse(&traffic_gates_snap(v)),
            Rule3Outcome::NotFired
        ));
    }

    #[test]
    fn path_b_recommendation_has_correct_impact_and_confidence() {
        let r = r3_recommendation(&traffic_gates_snap(path_b_base_vllm())).expect("fired");
        assert_eq!(r.impact, 3);
        assert!((r.confidence - 0.95).abs() < 1e-9);
    }

    #[test]
    fn path_b_display_shows_disabled_cause_not_hit_rate() {
        let d = LowPrefixReuseDetail {
            hit_rate: None,
            prompt_tokens_mean: Some(64.0),
            queries_delta: None,
        };
        let text = format_low_prefix_hit_rate_fired(&d, Some(false), None).join("\n");
        assert!(text.contains("Prefix caching is disabled (enable_prefix_caching=False)"));
        assert!(!text.contains("Prefix hit rate"));
    }

    #[test]
    fn path_a_detail_carries_hit_rate() {
        let mut v = path_b_base_vllm();
        v.cache_config.enable_prefix_caching = Some(true);
        v.prefix_cache_hit_rate = Some(0.10);
        match rule3_low_prefix_reuse(&traffic_gates_snap(v)) {
            Rule3Outcome::Fired(d) => {
                assert_eq!(d.hit_rate, Some(0.10));
            }
            Rule3Outcome::NotFired => panic!("expected path A fired"),
        }
    }

    #[test]
    fn fix_bullets_differ_when_prefix_caching_disabled() {
        let d = LowPrefixReuseDetail {
            hit_rate: Some(0.10),
            prompt_tokens_mean: Some(64.0),
            queries_delta: None,
        };
        let disabled = format_low_prefix_hit_rate_fired(&d, Some(false), None);
        let enabled = format_low_prefix_hit_rate_fired(&d, Some(true), None);
        assert!(
            disabled
                .iter()
                .any(|l| l.contains("--enable-prefix-caching"))
        );
        assert!(!disabled.iter().any(|l| l.contains("Avoid unique tokens")));
        assert!(enabled.iter().any(|l| l.contains("Avoid unique tokens")));
        assert!(
            !enabled
                .iter()
                .any(|l| l.contains("--enable-prefix-caching"))
        );
    }

    #[test]
    fn rag_low_qps_high_prompt_passes_gate() {
        // 0.5 QPS × 32_000 tok = 16_000 tok/s - RAG workload passes gate
        let mut v = path_b_base_vllm();
        v.request_success_per_sec = Some(0.5);
        v.prompt_tokens_mean = Some(32_000.0);
        v.cache_config.enable_prefix_caching = Some(false);
        assert!(matches!(
            rule3_low_prefix_reuse(&traffic_gates_snap(v)),
            Rule3Outcome::Fired(_)
        ));
    }

    #[test]
    fn sparse_cold_cache_does_not_fire() {
        // 0.003 QPS × 32_000 tok = 96 tok/s - sparse, gate stays closed
        let mut v = path_b_base_vllm();
        v.request_success_per_sec = Some(0.003);
        v.prompt_tokens_mean = Some(32_000.0);
        v.cache_config.enable_prefix_caching = Some(false);
        assert!(matches!(
            rule3_low_prefix_reuse(&traffic_gates_snap(v)),
            Rule3Outcome::NotFired
        ));
    }

    #[test]
    fn display_enable_prefix_when_caching_disabled() {
        let r = r3_recommendation(&traffic_gates_snap(path_b_base_vllm())).expect("fired");
        let text = r.display_lines.join("\n");
        assert!(text.contains("Enable prefix caching: --enable-prefix-caching"));
        assert!(text.contains("Higher prefix cache hit rate and lower TTFT"));
    }

    #[test]
    fn aggregate_r3_detail_empty_with_no_prompt_mean_returns_none() {
        let snap = traffic_gates_snap(VllmRawMetrics::default());
        let d = aggregate_r3_detail(&[], &snap);
        assert!(d.prompt_tokens_mean.is_none());
    }

    #[test]
    fn aggregate_r3_detail_weighted_by_query_volume() {
        let details = [
            LowPrefixReuseDetail {
                hit_rate: Some(0.9),
                prompt_tokens_mean: Some(64.0),
                queries_delta: Some(2.0),
            },
            LowPrefixReuseDetail {
                hit_rate: Some(0.1),
                prompt_tokens_mean: Some(64.0),
                queries_delta: Some(18.0),
            },
        ];
        let snap = traffic_gates_snap(VllmRawMetrics::default());
        let agg = aggregate_r3_detail(&details, &snap);
        let hit = agg.hit_rate.expect("hit rate");
        assert!((hit - 0.18).abs() < 1e-9, "weighted hit rate, got {hit}");
        assert!((hit - 0.5).abs() > 0.1, "should not be unweighted average");
    }

    #[test]
    fn session_hit_rate_overrides_detail_hit_rate_in_display() {
        let d = LowPrefixReuseDetail {
            hit_rate: Some(0.38),
            prompt_tokens_mean: Some(64.0),
            queries_delta: None,
        };
        let text = format_low_prefix_hit_rate_fired(&d, Some(true), Some(0.616)).join("\n");
        assert!(text.contains("Prefix hit rate 61.6%"));
        assert!(!text.contains("38.0%"));
    }
}
