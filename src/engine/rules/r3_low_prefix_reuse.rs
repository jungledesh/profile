use crate::collectors::RawSnapshot;

use super::Recommendation;

const PREFIX_HIT_RATE_LT: f64 = 0.35;
const PREFIX_RULE_PROMPT_TOKENS_GTE: f64 = 20.0;
const PREFIX_RULE_RUNNING_GT: f64 = 0.75;

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
    Some(Recommendation {
        rule_name: "low_prefix_reuse",
        impact: 2,
        confidence: 0.6,
        action: "Move shared context to prompt prefix; standardize prompt templates".to_string(),
        expected_impact: "Higher prefix cache hit rate and lower TTFT".to_string(),
        display_lines: format_low_prefix_hit_rate_fired(&d),
    })
}

pub(super) fn format_low_prefix_hit_rate_fired(d: &LowPrefixReuseDetail) -> Vec<String> {
    let hit = d.hit_rate * 100.0;
    vec![
        "ISSUE: Low Prefix Cache".to_string(),
        "Cause:".to_string(),
        format!("  - Prefix hit rate {hit:.1}%"),
        "  - Prompts have no shared leading context".to_string(),
        String::new(),
        "Recommendation:".to_string(),
        "  • Workload shows no prefix reuse — cache is currently ineffective".to_string(),
        "  • If reuse is expected:".to_string(),
        "      - Move shared instructions/system prompts to the very start".to_string(),
        "      - Standardize prompt templates across requests".to_string(),
        "      - Avoid unique tokens (IDs, timestamps) at the beginning".to_string(),
        "  • Otherwise: no action needed".to_string(),
        String::new(),
        "Expected: Reduced prefill time".to_string(),
        "Confidence: Medium-High".to_string(),
    ]
}

pub(super) fn format_low_prefix_window_issue(
    d: &LowPrefixReuseDetail,
    seen_pct: u32,
) -> Vec<String> {
    vec![
        "Low Prefix Cache".to_string(),
        format!("Seen in {seen_pct}% of windows"),
        "Cause:".to_string(),
        format!("  - Prefix hit rate {:.1}%", d.hit_rate * 100.0),
        "  - Prompts have no shared leading context".to_string(),
        String::new(),
        "Recommendation:".to_string(),
        "  • Workload shows no prefix reuse — cache is currently ineffective".to_string(),
        "  • If reuse is expected:".to_string(),
        "      - Move shared instructions/system prompts to the very start".to_string(),
        "      - Standardize prompt templates across requests".to_string(),
        "      - Avoid unique tokens (IDs, timestamps) at the beginning".to_string(),
        "  • Otherwise: no action needed".to_string(),
    ]
}

pub(super) fn format_rule3_verbose_miss(snapshot: &RawSnapshot) -> Vec<String> {
    let v = &snapshot.vllm;
    let Some(hr) = v.prefix_cache_hit_rate.filter(|x| x.is_finite()) else {
        return vec!["Prefix cache hit rate: not indicated".to_string()];
    };
    let pct = hr * 100.0;
    if hr >= PREFIX_HIT_RATE_LT {
        vec![
            "Rule: Low Prefix Cache — Not triggered".to_string(),
            format!("  - Prefix cache hit rate {pct:.1}% — working effectively"),
        ]
    } else {
        vec!["Prefix cache hit rate: not indicated".to_string()]
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
