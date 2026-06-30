use crate::collectors::RawSnapshot;

use super::Recommendation;
use super::rule_names;

/// Minimum prefill time fraction to consider prefill as a bottleneck.
const PREFILL_FRACTION_MILD: f64 = 0.30;
const PREFILL_FRACTION_MODERATE: f64 = 0.45;
const PREFILL_FRACTION_SEVERE: f64 = 0.60;

/// Decode efficiency below this indicates underperformance that prefill might explain.
const DECODE_EFFICIENCY_GATE: f64 = 40.0;
const PROMPT_SKEW_RATIO: f64 = 5.0;

#[derive(Debug, Clone, PartialEq)]
pub struct PrefillBoundDetail {
    pub prefill_time_fraction: f64,
    pub decode_efficiency_pct: f64,
    pub prefill_efficiency_pct: Option<f64>,
    pub prefix_caching_enabled: Option<bool>,
    pub chunked_prefill_enabled: Option<bool>,
    pub prompt_tokens_mean: Option<f64>,
    pub prompt_tokens_p99: Option<f64>,
    pub prompt_skew_ratio: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Mild,
    Moderate,
    Severe,
}

pub fn severity(prefill_fraction: f64) -> Severity {
    if prefill_fraction >= PREFILL_FRACTION_SEVERE {
        Severity::Severe
    } else if prefill_fraction >= PREFILL_FRACTION_MODERATE {
        Severity::Moderate
    } else {
        Severity::Mild
    }
}

pub fn impact(sev: Severity) -> u8 {
    match sev {
        Severity::Severe => 5,
        Severity::Moderate => 4,
        Severity::Mild => 3,
    }
}

pub fn confidence(sev: Severity) -> f64 {
    match sev {
        Severity::Severe => 0.85,
        Severity::Moderate => 0.75,
        Severity::Mild => 0.65,
    }
}

fn confidence_label(conf: f64) -> &'static str {
    if conf >= 0.8 {
        "High"
    } else if conf >= 0.7 {
        "Medium"
    } else {
        "Low"
    }
}

fn severity_title(sev: Severity) -> &'static str {
    match sev {
        Severity::Severe => "Prefill-Dominated",
        Severity::Moderate => "Prefill-Heavy",
        Severity::Mild => "Prefill-Elevated",
    }
}

fn severity_subtitle(sev: Severity) -> &'static str {
    match sev {
        Severity::Severe => "GPU Time Consumed by Prompt Processing",
        Severity::Moderate => "High Prompt Processing Time",
        Severity::Mild => "Elevated Prompt Processing Time",
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Rule6Outcome {
    Fired(PrefillBoundDetail),
    NotFired,
}

/// Evaluate R6 for a single window.
pub fn evaluate(
    prefill_time_fraction: Option<f64>,
    decode_efficiency_pct: Option<f64>,
    prefill_efficiency_pct: Option<f64>,
    snapshot: &RawSnapshot,
    chunked_prefill_enabled: Option<bool>,
) -> Rule6Outcome {
    let Some(fraction) = prefill_time_fraction.filter(|f| f.is_finite()) else {
        return Rule6Outcome::NotFired;
    };
    let Some(eff) = decode_efficiency_pct.filter(|e| e.is_finite()) else {
        return Rule6Outcome::NotFired;
    };

    if fraction < PREFILL_FRACTION_MILD {
        return Rule6Outcome::NotFired;
    }

    if eff >= DECODE_EFFICIENCY_GATE {
        return Rule6Outcome::NotFired;
    }

    let prompt_skew_ratio = match (
        snapshot.vllm.prompt_tokens_p99,
        snapshot.vllm.prompt_tokens_mean,
    ) {
        (Some(p99), Some(mean)) if mean > 0.0 && p99.is_finite() && mean.is_finite() => {
            Some(p99 / mean)
        }
        _ => None,
    };

    Rule6Outcome::Fired(PrefillBoundDetail {
        prefill_time_fraction: fraction,
        decode_efficiency_pct: eff,
        prefill_efficiency_pct,
        prefix_caching_enabled: snapshot.vllm.cache_config.enable_prefix_caching,
        chunked_prefill_enabled,
        prompt_tokens_mean: snapshot.vllm.prompt_tokens_mean,
        prompt_tokens_p99: snapshot.vllm.prompt_tokens_p99,
        prompt_skew_ratio,
    })
}

pub(super) fn aggregate_r6_detail(details: &[PrefillBoundDetail]) -> PrefillBoundDetail {
    let n = details.len() as f64;
    PrefillBoundDetail {
        prefill_time_fraction: details.iter().map(|d| d.prefill_time_fraction).sum::<f64>() / n,
        decode_efficiency_pct: details.iter().map(|d| d.decode_efficiency_pct).sum::<f64>() / n,
        prefill_efficiency_pct: {
            let vals: Vec<f64> = details
                .iter()
                .filter_map(|d| d.prefill_efficiency_pct)
                .collect();
            if vals.is_empty() {
                None
            } else {
                Some(vals.iter().sum::<f64>() / vals.len() as f64)
            }
        },
        prefix_caching_enabled: details.last().and_then(|d| d.prefix_caching_enabled),
        chunked_prefill_enabled: details.last().and_then(|d| d.chunked_prefill_enabled),
        prompt_tokens_mean: {
            let vals: Vec<f64> = details
                .iter()
                .filter_map(|d| d.prompt_tokens_mean)
                .collect();
            if vals.is_empty() {
                None
            } else {
                Some(vals.iter().sum::<f64>() / vals.len() as f64)
            }
        },
        prompt_tokens_p99: {
            details
                .iter()
                .filter_map(|d| d.prompt_tokens_p99)
                .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v))))
        },
        prompt_skew_ratio: {
            details
                .iter()
                .filter_map(|d| d.prompt_skew_ratio)
                .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v))))
        },
    }
}

fn skewed_mode(d: &PrefillBoundDetail) -> bool {
    d.prompt_skew_ratio
        .filter(|r| r.is_finite())
        .is_some_and(|r| r >= PROMPT_SKEW_RATIO)
}

pub(super) fn prefill_fix_lines(
    d: &PrefillBoundDetail,
    sev: Severity,
) -> (Vec<String>, String, String, String) {
    if skewed_mode(d)
        && let (Some(p99), Some(mean)) = (
            d.prompt_tokens_p99.filter(|v| v.is_finite() && *v > 0.0),
            d.prompt_tokens_mean.filter(|v| v.is_finite() && *v > 0.0),
        )
    {
        return (
            vec![
                format!(
                    "    • Route long-context requests (p99: {p99:.0} tok) to a dedicated vLLM instance."
                ),
                format!(
                    "      Short requests ({mean:.0} tok mean) are blocked by outlier prefills."
                ),
                "    • Or cap --max-model-len to reject prompts above the p95 threshold, or truncate at the application layer.".to_string(),
            ],
            format!(
                "Route long-context requests (p99: {p99:.0} tok) to a dedicated instance, or cap --max-model-len above p95"
            ),
            "Route long-context requests to a dedicated vLLM instance".to_string(),
            "Eliminates head-of-line blocking from long-tail prompts.".to_string(),
        );
    }

    let prefix_off = d.prefix_caching_enabled == Some(false);
    let chunked_on = d.chunked_prefill_enabled == Some(true);
    let chunked_not_enabled = d.chunked_prefill_enabled != Some(true);

    if prefix_off {
        (
            vec![
                "    • Enable automatic prefix caching (--enable-prefix-caching).".to_string(),
                "      Repeated prompt prefixes are re-computed every request.".to_string(),
            ],
            "Enable automatic prefix caching (--enable-prefix-caching) to avoid re-computing shared prompt prefixes".to_string(),
            "Enable prefix caching (--enable-prefix-caching)".to_string(),
            "20-40% reduction in prefill time for workloads with shared prefixes.".to_string(),
        )
    } else if chunked_not_enabled {
        (
            vec![
                "    • Enable chunked prefill (--enable-chunked-prefill) with a --max-num-batched-tokens budget (start at 2048, tune up).".to_string(),
            ],
            "Enable chunked prefill (--enable-chunked-prefill) with --max-num-batched-tokens budget, start at 2048".to_string(),
            "Enable chunked prefill (--enable-chunked-prefill)".to_string(),
            "Decode batches interleave with prefill, reducing head-of-line blocking.".to_string(),
        )
    } else if sev == Severity::Severe && d.prefix_caching_enabled == Some(true) && chunked_on {
        (
            vec![
                "    • Disaggregate prefill and decode onto separate workers (vLLM disaggregated serving, requires 2+ nodes).".to_string(),
            ],
            "Disaggregate prefill and decode onto separate workers (vLLM disaggregated serving, requires 2+ nodes)".to_string(),
            "Disaggregate prefill and decode workers".to_string(),
            "Full separation of prefill and decode compute paths.".to_string(),
        )
    } else if chunked_on {
        (
            vec![
                "    • Reduce --max-num-batched-tokens to shrink prefill chunk size.".to_string(),
                "      Current chunks are too large, starving decode.".to_string(),
            ],
            "Reduce --max-num-batched-tokens to shrink prefill chunk size, current chunks are starving decode".to_string(),
            "Reduce --max-num-batched-tokens".to_string(),
            "Lower TTFT variance, steadier decode throughput.".to_string(),
        )
    } else {
        unreachable!("chunked prefill decision tree exhausted")
    }
}

pub(super) fn format_prefill_bound_window_issue(
    d: &PrefillBoundDetail,
    seen_pct: u32,
) -> Vec<String> {
    let sev = severity(d.prefill_time_fraction);
    let conf = confidence(sev);
    let (fix_bullets, _, _, expected) = prefill_fix_lines(d, sev);
    let skewed = skewed_mode(d);

    let mut lines = vec![
        format!(
            "[!] {}: {}",
            severity_title(sev),
            if skewed {
                "Skewed Prompt Distribution"
            } else {
                severity_subtitle(sev)
            }
        ),
        format!("  Seen in {seen_pct}% of windows"),
        String::new(),
        format!(
            "  Prefill time    ~{:.0}%  estimated GPU time",
            d.prefill_time_fraction * 100.0
        ),
        format!(
            "  Decode eff.      {:.1}%  of HW ceiling",
            d.decode_efficiency_pct
        ),
    ];

    if let Some(pe) = d.prefill_efficiency_pct {
        lines.push(format!("  Prefill eff.    {pe:.1}%  of compute ceiling"));
    }
    if skewed {
        if let Some(pm) = d.prompt_tokens_mean.filter(|v| v.is_finite() && *v > 0.0) {
            lines.push(format!("  Prompt mean    {pm:.0} tok"));
        }
        if let Some(p99) = d.prompt_tokens_p99.filter(|v| v.is_finite() && *v > 0.0) {
            let ratio = d.prompt_skew_ratio.unwrap_or(0.0);
            if ratio > 0.0 && ratio.is_finite() {
                lines.push(format!("  Prompt p99    {p99:.0} tok  ({ratio:.0}x mean)"));
            } else {
                lines.push(format!("  Prompt p99    {p99:.0} tok"));
            }
        }
    } else if let Some(pm) = d.prompt_tokens_mean.filter(|v| v.is_finite() && *v > 0.0) {
        lines.push(format!("  Avg prompt     {pm:.0} tok"));
    }

    lines.push(String::new());
    lines.push("  Cause:".to_string());
    if skewed {
        if let (Some(pm), Some(p99)) = (
            d.prompt_tokens_mean.filter(|v| v.is_finite() && *v > 0.0),
            d.prompt_tokens_p99.filter(|v| v.is_finite() && *v > 0.0),
        ) {
            lines.push(format!(
                "    Outlier prompts (p99: {p99:.0} tok) are monopolizing prefill compute."
            ));
            lines.push(format!(
                "    Short requests ({pm:.0} tok mean) are blocked behind long-tail prefills."
            ));
        } else {
            lines.push(
                "    Prompt length outliers are monopolizing prefill compute and blocking shorter requests."
                    .to_string(),
            );
        }
    } else {
        lines.push(format!(
            "    Prefill is consuming {:.0}% of GPU time, starving decode throughput.",
            d.prefill_time_fraction * 100.0
        ));
        if sev == Severity::Severe
            && d.prefix_caching_enabled == Some(true)
            && d.chunked_prefill_enabled == Some(true)
        {
            lines.push(
                "    Prefix caching and chunked prefill are enabled but insufficient for this workload."
                    .to_string(),
            );
        } else if sev == Severity::Severe {
            lines.push(
                "    The GPU is busy, but mostly doing prompt processing, not token generation."
                    .to_string(),
            );
        } else {
            lines.push("    The GPU is busy, but decode throughput is limited.".to_string());
        }
    }

    lines.push(String::new());
    lines.push("  Fix:".to_string());
    lines.extend(fix_bullets);
    if !skewed {
        lines.push("    • Reduce prompt length where possible.".to_string());
    }
    lines.push(String::new());
    lines.push(format!("  Expected: {expected}"));
    lines.push(format!("  Confidence: {}", confidence_label(conf)));
    lines
}

pub fn r6_recommendation(
    prefill_time_fraction: Option<f64>,
    decode_efficiency_pct: Option<f64>,
    prefill_efficiency_pct: Option<f64>,
    snapshot: &RawSnapshot,
    chunked_prefill_enabled: Option<bool>,
) -> Option<Recommendation> {
    let Rule6Outcome::Fired(d) = evaluate(
        prefill_time_fraction,
        decode_efficiency_pct,
        prefill_efficiency_pct,
        snapshot,
        chunked_prefill_enabled,
    ) else {
        return None;
    };
    let sev = severity(d.prefill_time_fraction);
    let conf = confidence(sev);
    let (_, action, short_action, expected) = prefill_fix_lines(&d, sev);
    Some(Recommendation {
        rule_name: rule_names::PREFILL_BOUND,
        layer: 5,
        impact: impact(sev),
        confidence: conf,
        action,
        short_action,
        expected_impact: expected,
        display_lines: format_prefill_bound_window_issue(&d, 100),
    })
}

pub fn r6_verbose_miss_line(
    prefill_time_fraction: Option<f64>,
    decode_efficiency_pct: Option<f64>,
) -> Option<String> {
    let fraction = prefill_time_fraction.filter(|f| f.is_finite())?;
    if fraction >= PREFILL_FRACTION_MILD {
        if let Some(eff) = decode_efficiency_pct.filter(|e| e.is_finite()) {
            if eff >= DECODE_EFFICIENCY_GATE {
                return Some(format!(
                    "Prefill-bound: not triggered (decode efficiency {eff:.1}%, above {:.0}% gate)",
                    DECODE_EFFICIENCY_GATE
                ));
            }
            return None;
        }
        return Some("Prefill-bound: not triggered (decode efficiency unavailable)".to_string());
    }
    Some(format!(
        "Prefill-bound: not triggered (prefill fraction {:.0}%, below {:.0}% threshold)",
        fraction * 100.0,
        PREFILL_FRACTION_MILD * 100.0
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{CacheConfigLabels, RawSnapshot, VllmRawMetrics};
    use std::time::SystemTime;

    fn test_snapshot() -> RawSnapshot {
        RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: VllmRawMetrics {
                cache_config: CacheConfigLabels::default(),
                ..Default::default()
            },
            gpus: vec![],

            nvml_host_gpu_count: None,
        }
    }

    #[test]
    fn fires_when_prefill_high_and_efficiency_low() {
        let s = test_snapshot();
        match evaluate(Some(0.55), Some(10.0), Some(68.0), &s, None) {
            Rule6Outcome::Fired(d) => {
                assert!((d.prefill_time_fraction - 0.55).abs() < 1e-9);
                assert!((d.decode_efficiency_pct - 10.0).abs() < 1e-9);
                assert_eq!(severity(d.prefill_time_fraction), Severity::Moderate);
            }
            Rule6Outcome::NotFired => panic!("expected fired"),
        }
    }

    #[test]
    fn skewed_mode_fires_when_p99_over_5x_mean() {
        let mut s = test_snapshot();
        s.vllm.prompt_tokens_mean = Some(2000.0);
        s.vllm.prompt_tokens_p99 = Some(50_000.0);
        match evaluate(Some(0.55), Some(10.0), None, &s, None) {
            Rule6Outcome::Fired(d) => {
                let ratio = d.prompt_skew_ratio.expect("skew ratio");
                assert!(ratio >= 5.0);
            }
            Rule6Outcome::NotFired => panic!("expected fired"),
        }
    }

    #[test]
    fn uniform_mode_when_skew_below_threshold() {
        let mut s = test_snapshot();
        s.vllm.prompt_tokens_mean = Some(2000.0);
        s.vllm.prompt_tokens_p99 = Some(4000.0);
        match evaluate(Some(0.55), Some(10.0), None, &s, None) {
            Rule6Outcome::Fired(d) => {
                assert!(d.prompt_skew_ratio.unwrap_or(0.0) < 5.0);
            }
            Rule6Outcome::NotFired => panic!("expected fired"),
        }
    }

    #[test]
    fn skew_ratio_none_when_mean_missing() {
        let mut s = test_snapshot();
        s.vllm.prompt_tokens_mean = None;
        s.vllm.prompt_tokens_p99 = Some(50_000.0);
        match evaluate(Some(0.55), Some(10.0), None, &s, None) {
            Rule6Outcome::Fired(d) => {
                assert!(d.prompt_skew_ratio.is_none());
            }
            Rule6Outcome::NotFired => panic!("expected fired"),
        }
    }

    #[test]
    fn skew_ratio_none_when_p99_missing() {
        let mut s = test_snapshot();
        s.vllm.prompt_tokens_mean = Some(2000.0);
        s.vllm.prompt_tokens_p99 = None;
        match evaluate(Some(0.55), Some(10.0), None, &s, None) {
            Rule6Outcome::Fired(d) => {
                assert!(d.prompt_skew_ratio.is_none());
            }
            Rule6Outcome::NotFired => panic!("expected fired"),
        }
    }

    #[test]
    fn not_fired_when_prefill_fraction_below_threshold() {
        let s = test_snapshot();
        assert!(matches!(
            evaluate(Some(0.20), Some(10.0), None, &s, None),
            Rule6Outcome::NotFired
        ));
    }

    #[test]
    fn not_fired_when_efficiency_above_gate() {
        let s = test_snapshot();
        assert!(matches!(
            evaluate(Some(0.60), Some(50.0), None, &s, None),
            Rule6Outcome::NotFired
        ));
    }

    #[test]
    fn not_fired_when_prefill_fraction_none() {
        let s = test_snapshot();
        assert!(matches!(
            evaluate(None, Some(10.0), None, &s, None),
            Rule6Outcome::NotFired
        ));
    }

    #[test]
    fn not_fired_when_efficiency_none() {
        let s = test_snapshot();
        assert!(matches!(
            evaluate(Some(0.60), None, None, &s, None),
            Rule6Outcome::NotFired
        ));
    }

    #[test]
    fn verbose_miss_when_efficiency_unavailable() {
        let line = r6_verbose_miss_line(Some(0.55), None);
        assert!(line.is_some());
        assert!(line.expect("line").contains("unavailable"));
    }

    #[test]
    fn severity_tiers_correct() {
        assert_eq!(severity(0.30), Severity::Mild);
        assert_eq!(severity(0.44), Severity::Mild);
        assert_eq!(severity(0.45), Severity::Moderate);
        assert_eq!(severity(0.59), Severity::Moderate);
        assert_eq!(severity(0.60), Severity::Severe);
        assert_eq!(severity(0.90), Severity::Severe);
    }

    #[test]
    fn impact_scales_with_severity() {
        assert!(impact(Severity::Severe) > impact(Severity::Moderate));
        assert!(impact(Severity::Moderate) > impact(Severity::Mild));
    }

    #[test]
    fn confidence_scales_with_severity() {
        assert!(confidence(Severity::Severe) > confidence(Severity::Moderate));
        assert!(confidence(Severity::Moderate) > confidence(Severity::Mild));
    }

    #[test]
    fn fires_at_exact_boundary() {
        let s = test_snapshot();
        match evaluate(Some(0.30), Some(39.9), None, &s, None) {
            Rule6Outcome::Fired(_) => {}
            Rule6Outcome::NotFired => panic!("should fire at boundary"),
        }
    }

    #[test]
    fn does_not_fire_at_efficiency_boundary() {
        let s = test_snapshot();
        assert!(matches!(
            evaluate(Some(0.30), Some(40.0), None, &s, None),
            Rule6Outcome::NotFired
        ));
    }

    #[test]
    fn prefill_efficiency_propagated_when_available() {
        let s = test_snapshot();
        match evaluate(Some(0.50), Some(10.0), Some(72.5), &s, None) {
            Rule6Outcome::Fired(d) => {
                assert_eq!(d.prefill_efficiency_pct, Some(72.5));
            }
            Rule6Outcome::NotFired => panic!("expected fired"),
        }
    }

    #[test]
    fn prefill_efficiency_none_when_unavailable() {
        let s = test_snapshot();
        match evaluate(Some(0.50), Some(10.0), None, &s, None) {
            Rule6Outcome::Fired(d) => {
                assert!(d.prefill_efficiency_pct.is_none());
            }
            Rule6Outcome::NotFired => panic!("expected fired"),
        }
    }

    #[test]
    fn nan_inputs_do_not_fire() {
        let s = test_snapshot();
        assert!(matches!(
            evaluate(Some(f64::NAN), Some(10.0), None, &s, None),
            Rule6Outcome::NotFired
        ));
        assert!(matches!(
            evaluate(Some(0.5), Some(f64::NAN), None, &s, None),
            Rule6Outcome::NotFired
        ));
    }

    #[test]
    fn fix_recommends_routing_when_skewed() {
        let d = PrefillBoundDetail {
            prefill_time_fraction: 0.62,
            decode_efficiency_pct: 6.7,
            prefill_efficiency_pct: Some(68.4),
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(2048.0),
            prompt_tokens_p99: Some(51_200.0),
            prompt_skew_ratio: Some(25.0),
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Route long-context requests"));
        assert!(!text.contains("Reduce --max-num-batched-tokens to shrink prefill chunk size"));
    }

    #[test]
    fn fix_recommends_prefix_caching_when_disabled_uniform() {
        let d = PrefillBoundDetail {
            prefill_time_fraction: 0.58,
            decode_efficiency_pct: 8.2,
            prefill_efficiency_pct: Some(72.1),
            prefix_caching_enabled: Some(false),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("--enable-prefix-caching"));
    }

    #[test]
    fn fix_recommends_enable_chunked_when_prefix_on_but_chunked_off() {
        let d = PrefillBoundDetail {
            prefill_time_fraction: 0.55,
            decode_efficiency_pct: 10.0,
            prefill_efficiency_pct: None,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(false),
            prompt_tokens_mean: Some(2048.0),
            prompt_tokens_p99: Some(4096.0),
            prompt_skew_ratio: Some(2.0),
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("--enable-chunked-prefill"));
    }

    #[test]
    fn fix_recommends_reduce_chunk_size_when_both_enabled() {
        let d = PrefillBoundDetail {
            prefill_time_fraction: 0.48,
            decode_efficiency_pct: 14.5,
            prefill_efficiency_pct: None,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(2048.0),
            prompt_tokens_p99: Some(4096.0),
            prompt_skew_ratio: Some(2.0),
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("--max-num-batched-tokens"));
    }

    #[test]
    fn fix_recommends_disaggregation_when_severe_and_all_mitigations_on() {
        let d = PrefillBoundDetail {
            prefill_time_fraction: 0.71,
            decode_efficiency_pct: 3.1,
            prefill_efficiency_pct: Some(81.3),
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(8192.0),
            prompt_tokens_p99: Some(10_000.0),
            prompt_skew_ratio: Some(1.22),
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Disaggregate prefill and decode"));
    }

    #[test]
    fn appends_reduce_prompt_length_in_uniform_mode() {
        let d = PrefillBoundDetail {
            prefill_time_fraction: 0.48,
            decode_efficiency_pct: 14.5,
            prefill_efficiency_pct: None,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(2048.0),
            prompt_tokens_p99: Some(4096.0),
            prompt_skew_ratio: Some(2.0),
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Reduce prompt length where possible."));
    }

    #[test]
    fn skewed_mode_omits_reduce_prompt_length() {
        let d = PrefillBoundDetail {
            prefill_time_fraction: 0.62,
            decode_efficiency_pct: 6.7,
            prefill_efficiency_pct: Some(68.4),
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(2048.0),
            prompt_tokens_p99: Some(51_200.0),
            prompt_skew_ratio: Some(25.0),
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(
            !text.contains("Reduce prompt length where possible"),
            "skewed mode should not include generic prompt length advice"
        );
    }
}
