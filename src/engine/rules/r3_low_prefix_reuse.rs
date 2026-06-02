use crate::collectors::RawSnapshot;

use super::Recommendation;

const PREFIX_HIT_RATE_LT: f64 = 0.35;
const PREFIX_RULE_PROMPT_TOKENS_GTE: f64 = 20.0;
const PREFIX_RULE_RUNNING_GT: f64 = 0.75;
const PREFIX_RULE_MIN_QPS: f64 = 5.0;

#[derive(Debug, Clone, PartialEq)]
pub struct LowPrefixReuseDetail {
    pub hit_rate: f64,
    pub prompt_tokens_mean: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Rule3Outcome {
    Fired(LowPrefixReuseDetail),
    NotFired,
}

pub fn rule3_low_prefix_reuse(snapshot: &RawSnapshot) -> Rule3Outcome {
    let v = &snapshot.vllm;
    let rate = v.prefix_cache_hit_rate.filter(|x| x.is_finite());
    let running = v.num_requests_running.filter(|x| x.is_finite());
    let prompt_mean = v.prompt_tokens_mean.filter(|x| x.is_finite());

    let qps = v.request_success_per_sec.filter(|x| x.is_finite());
    if qps.is_none_or(|q| q < PREFIX_RULE_MIN_QPS) {
        return Rule3Outcome::NotFired;
    }

    let Some(hit_rate) = rate else {
        return Rule3Outcome::NotFired;
    };
    let Some(rv) = running else {
        return Rule3Outcome::NotFired;
    };
    let Some(pm) = prompt_mean else {
        return Rule3Outcome::NotFired;
    };

    if rv <= PREFIX_RULE_RUNNING_GT {
        return Rule3Outcome::NotFired;
    }
    if pm < PREFIX_RULE_PROMPT_TOKENS_GTE {
        return Rule3Outcome::NotFired;
    }
    if hit_rate >= PREFIX_HIT_RATE_LT {
        return Rule3Outcome::NotFired;
    }

    Rule3Outcome::Fired(LowPrefixReuseDetail {
        hit_rate,
        prompt_tokens_mean: pm,
    })
}

pub fn r3_recommendation(snapshot: &RawSnapshot) -> Option<Recommendation> {
    let Rule3Outcome::Fired(d) = rule3_low_prefix_reuse(snapshot) else {
        return None;
    };
    let enable_prefix = snapshot.vllm.cache_config.enable_prefix_caching;
    Some(Recommendation {
        rule_name: "low_prefix_reuse",
        impact: 2,
        confidence: 0.9,
        action: "Move shared context to prompt prefix; standardize prompt templates".to_string(),
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
    let hit = d.hit_rate * 100.0;
    vec![
        "[!] Low Prefix Cache".to_string(),
        String::new(),
        "  Cause:".to_string(),
        format!(
            "  - Prefix hit rate {hit:.1}% (threshold: {:.0}%)",
            PREFIX_HIT_RATE_LT * 100.0
        ),
        prefix_cause_bullet(enable_prefix_caching),
        String::new(),
        "  Fix:".to_string(),
        "  • Move shared instructions/system prompts to the very start".to_string(),
        "  • Standardize prompt templates across requests".to_string(),
        "  • Avoid unique tokens (IDs, timestamps) at the beginning".to_string(),
        String::new(),
        "  Expected: Reduced prefill time".to_string(),
        "  Confidence: High".to_string(),
    ]
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

pub(super) fn format_rule3_verbose_miss(snapshot: &RawSnapshot) -> Vec<String> {
    let v = &snapshot.vllm;
    let Some(hr) = v.prefix_cache_hit_rate.filter(|x| x.is_finite()) else {
        return vec!["Prefix cache hit rate: not triggered".to_string()];
    };
    let pct = hr * 100.0;
    if hr >= PREFIX_HIT_RATE_LT {
        vec![format!("Prefix cache hit rate: {pct:.1}% (not triggered)")]
    } else {
        vec!["Prefix cache hit rate: not triggered".to_string()]
    }
}

pub(super) fn aggregate_r3_detail(
    details: &[LowPrefixReuseDetail],
    summary: &RawSnapshot,
) -> LowPrefixReuseDetail {
    if details.is_empty() {
        return LowPrefixReuseDetail {
            hit_rate: summary.vllm.prefix_cache_hit_rate.unwrap_or(0.0),
            prompt_tokens_mean: summary.vllm.prompt_tokens_mean.unwrap_or(0.0),
        };
    }
    let hit_rate = details.iter().map(|d| d.hit_rate).sum::<f64>() / details.len() as f64;
    let prompt_tokens_mean =
        details.iter().map(|d| d.prompt_tokens_mean).sum::<f64>() / details.len() as f64;
    LowPrefixReuseDetail {
        hit_rate,
        prompt_tokens_mean,
    }
}
