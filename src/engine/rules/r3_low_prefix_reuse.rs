use crate::collectors::RawSnapshot;

use super::Recommendation;

const PREFIX_HIT_RATE_LT: f64 = 0.35;
const PREFIX_RULE_PROMPT_TOKENS_GTE: f64 = 20.0;
const PREFIX_RULE_RUNNING_GT: f64 = 0.75;
const PREFIX_RULE_MIN_QPS: f64 = 5.0;

#[derive(Debug, Clone, PartialEq)]
pub struct LowPrefixReuseDetail {
    /// None when prefix caching is disabled (path B).
    pub hit_rate: Option<f64>,
    pub prompt_tokens_mean: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Rule3Outcome {
    Fired(LowPrefixReuseDetail),
    NotFired,
}

pub fn rule3_low_prefix_reuse(snapshot: &RawSnapshot) -> Rule3Outcome {
    let v = &snapshot.vllm;

    let running = v.num_requests_running.filter(|x| x.is_finite());
    let prompt_mean = v.prompt_tokens_mean.filter(|x| x.is_finite());
    let qps = v.request_success_per_sec.filter(|x| x.is_finite());

    if qps.is_none_or(|q| q < PREFIX_RULE_MIN_QPS) {
        return Rule3Outcome::NotFired;
    }
    let Some(rv) = running else {
        return Rule3Outcome::NotFired;
    };
    if rv <= PREFIX_RULE_RUNNING_GT {
        return Rule3Outcome::NotFired;
    }
    let Some(pm) = prompt_mean else {
        return Rule3Outcome::NotFired;
    };
    if pm < PREFIX_RULE_PROMPT_TOKENS_GTE {
        return Rule3Outcome::NotFired;
    }

    if v.cache_config.enable_prefix_caching == Some(false) {
        return Rule3Outcome::Fired(LowPrefixReuseDetail {
            hit_rate: None,
            prompt_tokens_mean: pm,
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
        prompt_tokens_mean: pm,
    })
}

pub fn r3_recommendation(snapshot: &RawSnapshot) -> Option<Recommendation> {
    let Rule3Outcome::Fired(d) = rule3_low_prefix_reuse(snapshot) else {
        return None;
    };
    let enable_prefix = snapshot.vllm.cache_config.enable_prefix_caching;
    let (action, short_action, confidence) = if d.hit_rate.is_none() {
        (
            "Enable --enable-prefix-caching".to_string(),
            "enable prefix caching".to_string(),
            0.95_f64,
        )
    } else {
        (
            "Move shared context to prompt prefix; standardize prompt templates".to_string(),
            "standardize prompts to share prefix context".to_string(),
            0.9_f64,
        )
    };
    Some(Recommendation {
        rule_name: "low_prefix_reuse",
        impact: if d.hit_rate.is_none() { 3 } else { 2 },
        confidence,
        action,
        short_action,
        expected_impact: "Higher prefix cache hit rate and lower TTFT".to_string(),
        display_lines: format_low_prefix_hit_rate_fired(&d, enable_prefix),
    })
}

fn prefix_cause_bullet(enable_prefix_caching: Option<bool>) -> String {
    match enable_prefix_caching {
        Some(false) => {
            "  - Prefix caching is disabled — enable with --enable-prefix-caching".to_string()
        }
        Some(true) | None => {
            "  - Low prefix hit rate — restructure prompts to share common prefixes".to_string()
        }
    }
}

pub(super) fn format_low_prefix_hit_rate_fired(
    d: &LowPrefixReuseDetail,
    enable_prefix_caching: Option<bool>,
) -> Vec<String> {
    let cause_lines: Vec<String> = if d.hit_rate.is_none() {
        vec![
            "  Cause:".to_string(),
            "  - Prefix caching is disabled (enable_prefix_caching=False)".to_string(),
        ]
    } else {
        let hit = d.hit_rate.unwrap_or(0.0) * 100.0;
        vec![
            "  Cause:".to_string(),
            format!(
                "  - Prefix hit rate {hit:.1}% (threshold: {:.0}%)",
                PREFIX_HIT_RATE_LT * 100.0
            ),
            prefix_cause_bullet(enable_prefix_caching),
        ]
    };

    let fix_lines: Vec<String> = if enable_prefix_caching == Some(false) {
        vec![
            "  • Enable prefix caching: --enable-prefix-caching".to_string(),
            "  • Move shared instructions/system prompts to the very start".to_string(),
            "  • Standardize prompt templates across requests".to_string(),
        ]
    } else {
        vec![
            "  • Move shared instructions/system prompts to the very start".to_string(),
            "  • Standardize prompt templates across requests".to_string(),
            "  • Avoid unique tokens (IDs, timestamps) at the beginning".to_string(),
        ]
    };

    let mut lines = vec!["[!] Low Prefix Cache".to_string(), String::new()];
    lines.extend(cause_lines);
    lines.push(String::new());
    lines.push("  Fix:".to_string());
    lines.extend(fix_lines);
    lines.push(String::new());
    lines.push("  Expected: Lower TTFT on repeated prefixes".to_string());
    lines.push("  Confidence: High".to_string());
    lines
}

pub(super) fn format_low_prefix_window_issue(
    d: &LowPrefixReuseDetail,
    seen_pct: u32,
    enable_prefix_caching: Option<bool>,
) -> Vec<String> {
    let mut lines = format_low_prefix_hit_rate_fired(d, enable_prefix_caching);
    lines.insert(1, format!("  Seen in {seen_pct}% of windows"));
    lines
}

pub(super) fn aggregate_r3_detail(
    details: &[LowPrefixReuseDetail],
    summary: &RawSnapshot,
) -> LowPrefixReuseDetail {
    if details.is_empty() {
        return LowPrefixReuseDetail {
            hit_rate: summary.vllm.prefix_cache_hit_rate.filter(|x| x.is_finite()),
            prompt_tokens_mean: summary.vllm.prompt_tokens_mean.unwrap_or(0.0),
        };
    }
    let hit_rate = if details.iter().any(|d| d.hit_rate.is_none()) {
        None
    } else {
        let n = details.len() as f64;
        Some(details.iter().filter_map(|d| d.hit_rate).sum::<f64>() / n)
    };
    let prompt_tokens_mean =
        details.iter().map(|d| d.prompt_tokens_mean).sum::<f64>() / details.len() as f64;
    LowPrefixReuseDetail {
        hit_rate,
        prompt_tokens_mean,
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
            gpu: Default::default(),
        }
    }

    fn path_b_base_vllm() -> VllmRawMetrics {
        VllmRawMetrics {
            prompt_tokens_mean: Some(64.0),
            num_requests_running: Some(5.0),
            request_success_per_sec: Some(10.0),
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
                assert!((d.prompt_tokens_mean - 64.0).abs() < 1e-9);
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
            prompt_tokens_mean: 64.0,
        };
        let text = format_low_prefix_hit_rate_fired(&d, Some(false)).join("\n");
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
            prompt_tokens_mean: 64.0,
        };
        let disabled = format_low_prefix_hit_rate_fired(&d, Some(false));
        let enabled = format_low_prefix_hit_rate_fired(&d, Some(true));
        assert!(disabled
            .iter()
            .any(|l| l.contains("--enable-prefix-caching")));
        assert!(!disabled.iter().any(|l| l.contains("Avoid unique tokens")));
        assert!(enabled.iter().any(|l| l.contains("Avoid unique tokens")));
        assert!(!enabled
            .iter()
            .any(|l| l.contains("--enable-prefix-caching")));
    }

    #[test]
    fn short_action_is_enable_flag_when_prefix_caching_disabled() {
        let r = r3_recommendation(&traffic_gates_snap(path_b_base_vllm())).expect("fired");
        assert_eq!(r.short_action, "enable prefix caching");
        assert_eq!(r.action, "Enable --enable-prefix-caching");
    }
}
