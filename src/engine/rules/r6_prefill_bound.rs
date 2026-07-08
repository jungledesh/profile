use crate::collectors::RawSnapshot;

use super::Recommendation;
use super::rule_names;

/// Primary trigger: prompt-to-generation token ratio.
/// Break-even is ridge / decode_batch. For A100-H100 at batch 30+, this falls in
/// the 1.3-5.7 range. 5.0 catches meaningful prefill dominance without
/// false-positiving on agent workloads (~4:1 ratio). Calibrate with production data.
const PROMPT_GEN_RATIO_MILD: f64 = 5.0;
const PROMPT_GEN_RATIO_MODERATE: f64 = 10.0;
const PROMPT_GEN_RATIO_SEVERE: f64 = 20.0;

/// TPOT must be inflated above this multiple of the physics floor for R6 to fire.
/// Prefill ratio alone isn't a problem if TPOT isn't inflated (server is handling it).
const TPOT_INFLATION_GATE: f64 = 4.0;

/// Decode efficiency below this indicates underperformance that prefill might explain.
const DECODE_EFFICIENCY_GATE: f64 = 40.0;
const PROMPT_SKEW_RATIO: f64 = 5.0;

/// Fixed label column width for R6 metric rows (longest label: "Prefill ratio").
const R6_METRIC_LABEL_W: usize = 20;

fn r6_metric_line(label: &str, value: &str) -> String {
    format!("    {label:<R6_METRIC_LABEL_W$}{value}")
}

/// Fallback when prompt mean or running count unavailable for budget derivation.
const DEFAULT_BATCH_TOKEN_BUDGET: u64 = 2048;

/// Compute recommended --max-num-batched-tokens from workload.
/// Target: chunk average prompt into ~2 steps after decode overhead.
/// Round up to nearest 128 for readability.
fn recommended_batch_token_budget(prompt_mean: f64, running: f64) -> u64 {
    let raw = prompt_mean / 2.0 + running;
    ((raw / 128.0).ceil() as u64) * 128
}

fn batch_token_budget(d: &PrefillBoundDetail) -> u64 {
    match (d.prompt_tokens_mean, d.running_count) {
        (Some(pm), Some(rc)) if pm.is_finite() && pm > 0.0 && rc.is_finite() && rc >= 0.0 => {
            recommended_batch_token_budget(pm, rc)
        }
        _ => DEFAULT_BATCH_TOKEN_BUDGET,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrefillBoundDetail {
    pub prompt_gen_ratio: f64,
    pub decode_efficiency_pct: f64,
    pub tpot_ms: Option<f64>,
    pub tpot_floor_ms: Option<f64>,
    pub prefix_caching_enabled: Option<bool>,
    pub chunked_prefill_enabled: Option<bool>,
    pub prompt_tokens_mean: Option<f64>,
    pub prompt_tokens_p99: Option<f64>,
    pub prompt_skew_ratio: Option<f64>,
    pub running_count: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Mild,
    Moderate,
    Severe,
}

/// Adjust raw prompt tok/s to reflect actual prefill compute.
/// Cached tokens skip prefill, so subtract them.
/// When prefix_hit_rate is None or > 1.0 (bad data), return raw value (conservative).
pub(crate) fn effective_prompt_tps(raw_prompt_tps: f64, prefix_hit_rate: Option<f64>) -> f64 {
    match prefix_hit_rate.filter(|r| r.is_finite() && *r >= 0.0 && *r <= 1.0) {
        Some(rate) => raw_prompt_tps * (1.0 - rate),
        None => raw_prompt_tps,
    }
}

/// Gate inputs shared by evaluate() and verbose miss reporting.
#[derive(Debug, Clone, Copy)]
pub struct R6GateInput {
    pub prompt_tokens_per_sec: Option<f64>,
    pub generation_tokens_per_sec: Option<f64>,
    pub decode_efficiency_pct: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub tpot_floor_ms: Option<f64>,
    pub prefix_cache_hit_rate: Option<f64>,
}

/// Inputs for per-window R6 evaluation.
pub struct PrefillBoundEvalInput<'a> {
    pub prompt_tokens_per_sec: Option<f64>,
    pub generation_tokens_per_sec: Option<f64>,
    pub decode_efficiency_pct: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub tpot_floor_ms: Option<f64>,
    pub prefix_cache_hit_rate: Option<f64>,
    pub snapshot: &'a RawSnapshot,
    pub chunked_prefill_enabled: Option<bool>,
}

pub fn severity(prompt_gen_ratio: f64) -> Severity {
    if prompt_gen_ratio >= PROMPT_GEN_RATIO_SEVERE {
        Severity::Severe
    } else if prompt_gen_ratio >= PROMPT_GEN_RATIO_MODERATE {
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
pub fn evaluate(input: PrefillBoundEvalInput<'_>) -> Rule6Outcome {
    let PrefillBoundEvalInput {
        prompt_tokens_per_sec,
        generation_tokens_per_sec,
        decode_efficiency_pct,
        tpot_ms,
        tpot_floor_ms,
        prefix_cache_hit_rate,
        snapshot,
        chunked_prefill_enabled,
    } = input;
    let Some(raw_prompt_tps) = prompt_tokens_per_sec.filter(|v| v.is_finite() && *v > 0.0) else {
        return Rule6Outcome::NotFired;
    };
    let prompt_tps = effective_prompt_tps(raw_prompt_tps, prefix_cache_hit_rate);
    if prompt_tps <= 0.0 {
        return Rule6Outcome::NotFired;
    }
    let gen_tps = match generation_tokens_per_sec.filter(|v| v.is_finite()) {
        Some(v) => v,
        None => return Rule6Outcome::NotFired,
    };

    let ratio = if gen_tps > 0.0 {
        prompt_tps / gen_tps
    } else {
        f64::INFINITY
    };

    if ratio < PROMPT_GEN_RATIO_MILD {
        return Rule6Outcome::NotFired;
    }

    let Some(eff) = decode_efficiency_pct.filter(|e| e.is_finite()) else {
        return Rule6Outcome::NotFired;
    };
    if eff >= DECODE_EFFICIENCY_GATE {
        return Rule6Outcome::NotFired;
    }

    if let (Some(tpot), Some(floor)) = (
        tpot_ms.filter(|v| v.is_finite() && *v > 0.0),
        tpot_floor_ms.filter(|v| v.is_finite() && *v > 0.0),
    ) && tpot < floor * TPOT_INFLATION_GATE
    {
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
        prompt_gen_ratio: ratio,
        decode_efficiency_pct: eff,
        tpot_ms,
        tpot_floor_ms,
        prefix_caching_enabled: snapshot.vllm.cache_config.enable_prefix_caching,
        chunked_prefill_enabled,
        prompt_tokens_mean: snapshot.vllm.prompt_tokens_mean,
        prompt_tokens_p99: snapshot.vllm.prompt_tokens_p99,
        prompt_skew_ratio,
        running_count: snapshot.vllm.num_requests_running,
    })
}

pub(super) fn aggregate_r6_detail(details: &[PrefillBoundDetail]) -> PrefillBoundDetail {
    let n = details.len() as f64;
    PrefillBoundDetail {
        prompt_gen_ratio: {
            let finite: Vec<f64> = details
                .iter()
                .map(|d| d.prompt_gen_ratio)
                .filter(|r| r.is_finite())
                .collect();
            if finite.is_empty() {
                f64::INFINITY
            } else {
                finite.iter().sum::<f64>() / finite.len() as f64
            }
        },
        decode_efficiency_pct: details.iter().map(|d| d.decode_efficiency_pct).sum::<f64>() / n,
        tpot_ms: details
            .iter()
            .filter_map(|d| d.tpot_ms)
            .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v)))),
        tpot_floor_ms: details.last().and_then(|d| d.tpot_floor_ms),
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
        prompt_tokens_p99: details
            .iter()
            .filter_map(|d| d.prompt_tokens_p99)
            .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v)))),
        prompt_skew_ratio: details
            .iter()
            .filter_map(|d| d.prompt_skew_ratio)
            .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v)))),
        running_count: {
            let vals: Vec<f64> = details.iter().filter_map(|d| d.running_count).collect();
            if vals.is_empty() {
                None
            } else {
                Some(vals.iter().sum::<f64>() / vals.len() as f64)
            }
        },
    }
}

fn skewed_mode(d: &PrefillBoundDetail) -> bool {
    d.prompt_skew_ratio
        .filter(|r| r.is_finite())
        .is_some_and(|r| r >= PROMPT_SKEW_RATIO)
}

fn cause_severity_line(sev: Severity, d: &PrefillBoundDetail) -> String {
    if sev == Severity::Severe
        && d.prefix_caching_enabled == Some(true)
        && d.chunked_prefill_enabled == Some(true)
    {
        "      Prefix caching and chunked prefill are enabled but insufficient for this workload."
            .to_string()
    } else if sev == Severity::Severe {
        "      The GPU is busy, but mostly doing prompt processing, not token generation."
            .to_string()
    } else {
        "      The GPU is busy, but decode throughput is limited.".to_string()
    }
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
                    "      • Route long-context requests (p99: {p99:.0} tok) to a dedicated vLLM instance."
                ),
                format!(
                    "      Short requests ({mean:.0} tok mean) are blocked by outlier prefills."
                ),
                "      • Or cap --max-model-len to reject prompts above the p95 threshold, or truncate at the application layer.".to_string(),
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
                "      • Enable automatic prefix caching (--enable-prefix-caching).".to_string(),
                "      Repeated prompt prefixes are re-computed every request.".to_string(),
            ],
            "Enable automatic prefix caching (--enable-prefix-caching) to avoid re-computing shared prompt prefixes".to_string(),
            "Enable prefix caching (--enable-prefix-caching)".to_string(),
            "20-40% reduction in prefill time for workloads with shared prefixes.".to_string(),
        )
    } else if chunked_not_enabled {
        let budget = batch_token_budget(d);
        (
            vec![
                "      • Enable chunked prefill (--enable-chunked-prefill).".to_string(),
                format!(
                    "      • Set --max-num-batched-tokens to {budget}. Lower for smoother TPOT, raise for lower TTFT."
                ),
            ],
            format!(
                "Enable chunked prefill (--enable-chunked-prefill) and set --max-num-batched-tokens to {budget}"
            ),
            "Enable chunked prefill (--enable-chunked-prefill)".to_string(),
            "Decode batches interleave with prefill, reducing head-of-line blocking.".to_string(),
        )
    } else if sev == Severity::Severe && d.prefix_caching_enabled == Some(true) && chunked_on {
        (
            vec![
                "      • Disaggregate prefill and decode onto separate workers (vLLM disaggregated serving, requires 2+ nodes).".to_string(),
            ],
            "Disaggregate prefill and decode onto separate workers (vLLM disaggregated serving, requires 2+ nodes)".to_string(),
            "Disaggregate prefill and decode workers".to_string(),
            "Full separation of prefill and decode compute paths.".to_string(),
        )
    } else if chunked_on {
        let budget = batch_token_budget(d);
        (
            vec![format!(
                "      • Reduce --max-num-batched-tokens to {budget}. Lower for smoother TPOT, raise for lower TTFT."
            )],
            format!("Reduce --max-num-batched-tokens to {budget} to shrink prefill chunk size"),
            "Reduce --max-num-batched-tokens".to_string(),
            "Lower TTFT variance, steadier decode throughput.".to_string(),
        )
    } else {
        // Logically unreachable: branch 2 forces chunked_on=true, branch 4 catches it.
        // Safe fallback instead of panic in library code.
        let budget = batch_token_budget(d);
        (
            vec![format!(
                "      • Reduce --max-num-batched-tokens to {budget}. Lower for smoother TPOT, raise for lower TTFT."
            )],
            format!("Reduce --max-num-batched-tokens to {budget} to shrink prefill chunk size"),
            "Reduce --max-num-batched-tokens".to_string(),
            "Lower TTFT variance, steadier decode throughput.".to_string(),
        )
    }
}

pub(super) fn format_prefill_bound_window_issue(
    d: &PrefillBoundDetail,
    seen_pct: u32,
) -> Vec<String> {
    let sev = severity(d.prompt_gen_ratio);
    let conf = confidence(sev);
    let (fix_bullets, _, _, expected) = prefill_fix_lines(d, sev);
    let skewed = skewed_mode(d);

    let ratio_display = if d.prompt_gen_ratio.is_finite() {
        format!("{:.1}x", d.prompt_gen_ratio)
    } else {
        "inf".to_string()
    };

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
        String::new(),
        r6_metric_line(
            "Prefill ratio",
            &format!("{ratio_display}  prompt tok/s vs gen tok/s"),
        ),
        format!(
            "    {:<width$}(avg when prefill-bound)   {:.1}%  of HW ceiling",
            "Decode eff.",
            d.decode_efficiency_pct,
            width = R6_METRIC_LABEL_W
        ),
    ];

    if skewed {
        if let Some(pm) = d.prompt_tokens_mean.filter(|v| v.is_finite() && *v > 0.0) {
            lines.push(r6_metric_line("Prompt mean", &format!("{pm:.0} tok")));
        }
        if let Some(p99) = d.prompt_tokens_p99.filter(|v| v.is_finite() && *v > 0.0) {
            let ratio = d.prompt_skew_ratio.unwrap_or(0.0);
            if ratio > 0.0 && ratio.is_finite() {
                lines.push(r6_metric_line(
                    "Prompt p99",
                    &format!("{p99:.0} tok  ({ratio:.0}x mean)"),
                ));
            } else {
                lines.push(r6_metric_line("Prompt p99", &format!("{p99:.0} tok")));
            }
        }
    } else if let Some(pm) = d.prompt_tokens_mean.filter(|v| v.is_finite() && *v > 0.0) {
        lines.push(r6_metric_line("Avg prompt", &format!("{pm:.0} tok")));
    }

    lines.push(String::new());
    lines.push("    Cause:".to_string());
    if skewed {
        if let (Some(pm), Some(p99)) = (
            d.prompt_tokens_mean.filter(|v| v.is_finite() && *v > 0.0),
            d.prompt_tokens_p99.filter(|v| v.is_finite() && *v > 0.0),
        ) {
            lines.push(format!(
                "      Outlier prompts (p99: {p99:.0} tok) are monopolizing prefill compute."
            ));
            lines.push(format!(
                "      Short requests ({pm:.0} tok mean) are blocked behind long-tail prefills."
            ));
        } else {
            lines.push(
                "      Prompt length outliers are monopolizing prefill compute and blocking shorter requests."
                    .to_string(),
            );
        }
    } else {
        lines.push(format!(
            "      Prompt input rate is {ratio_display} generation output rate, starving decode throughput."
        ));
        lines.push(cause_severity_line(sev, d));
    }

    lines.push(String::new());
    lines.push("    Fix:".to_string());
    lines.extend(fix_bullets);
    if !skewed {
        lines.push("      • Reduce prompt length where possible.".to_string());
    }
    lines.push(String::new());
    lines.push(format!("    Expected: {expected}"));
    lines.push(format!("    Confidence: {}", confidence_label(conf)));
    super::with_seen_pct(lines, seen_pct)
}

pub fn r6_recommendation(input: PrefillBoundEvalInput<'_>) -> Option<Recommendation> {
    let Rule6Outcome::Fired(d) = evaluate(input) else {
        return None;
    };
    let sev = severity(d.prompt_gen_ratio);
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

pub fn r6_verbose_miss_line(input: R6GateInput) -> Option<String> {
    let R6GateInput {
        prompt_tokens_per_sec,
        generation_tokens_per_sec,
        decode_efficiency_pct,
        tpot_ms,
        tpot_floor_ms,
        prefix_cache_hit_rate,
    } = input;
    let Some(raw_prompt_tps) = prompt_tokens_per_sec.filter(|v| v.is_finite() && *v > 0.0) else {
        return Some("Prefill-bound: not triggered (prompt tok/s unavailable)".to_string());
    };
    let prompt_tps = effective_prompt_tps(raw_prompt_tps, prefix_cache_hit_rate);
    if prompt_tps <= 0.0 {
        return Some(
            "Prefill-bound: not triggered (effective prompt tok/s zero after prefix cache adjustment)"
                .to_string(),
        );
    }
    let gen_tps = match generation_tokens_per_sec.filter(|v| v.is_finite()) {
        Some(v) => v,
        None => {
            return Some("Prefill-bound: not triggered (gen tok/s unavailable)".to_string());
        }
    };
    let ratio = if gen_tps > 0.0 {
        prompt_tps / gen_tps
    } else {
        f64::INFINITY
    };

    if ratio < PROMPT_GEN_RATIO_MILD {
        let ratio_str = if ratio.is_finite() {
            format!("{ratio:.1}x")
        } else {
            "inf".to_string()
        };
        return Some(format!(
            "Prefill-bound: not triggered (prompt/gen ratio {ratio_str}, below {:.1}x threshold)",
            PROMPT_GEN_RATIO_MILD
        ));
    }

    if let Some(eff) = decode_efficiency_pct.filter(|e| e.is_finite()) {
        if eff >= DECODE_EFFICIENCY_GATE {
            return Some(format!(
                "Prefill-bound: not triggered (decode efficiency {eff:.1}%, above {:.0}% gate)",
                DECODE_EFFICIENCY_GATE
            ));
        }
    } else {
        return Some("Prefill-bound: not triggered (decode efficiency unavailable)".to_string());
    }

    if let (Some(tpot), Some(floor)) = (
        tpot_ms.filter(|v| v.is_finite() && *v > 0.0),
        tpot_floor_ms.filter(|v| v.is_finite() && *v > 0.0),
    ) && tpot < floor * TPOT_INFLATION_GATE
    {
        return Some(format!(
            "Prefill-bound: not triggered (TPOT {tpot:.1}ms below {:.1}x floor {:.1}ms)",
            TPOT_INFLATION_GATE,
            floor * TPOT_INFLATION_GATE
        ));
    }

    None
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

    #[derive(Copy, Clone)]
    struct EvalR6Params<'a> {
        prompt_tps: Option<f64>,
        gen_tps: Option<f64>,
        eff: Option<f64>,
        tpot_ms: Option<f64>,
        tpot_floor_ms: Option<f64>,
        prefix_cache_hit_rate: Option<f64>,
        snapshot: &'a RawSnapshot,
    }

    fn eval_r6(p: EvalR6Params<'_>) -> Rule6Outcome {
        evaluate(PrefillBoundEvalInput {
            prompt_tokens_per_sec: p.prompt_tps,
            generation_tokens_per_sec: p.gen_tps,
            decode_efficiency_pct: p.eff,
            tpot_ms: p.tpot_ms,
            tpot_floor_ms: p.tpot_floor_ms,
            prefix_cache_hit_rate: p.prefix_cache_hit_rate,
            snapshot: p.snapshot,
            chunked_prefill_enabled: None,
        })
    }

    fn eval_r6_default(
        snapshot: &RawSnapshot,
        prompt_tps: Option<f64>,
        gen_tps: Option<f64>,
        eff: Option<f64>,
        tpot_ms: Option<f64>,
        tpot_floor_ms: Option<f64>,
    ) -> Rule6Outcome {
        eval_r6(EvalR6Params {
            prompt_tps,
            gen_tps,
            eff,
            tpot_ms,
            tpot_floor_ms,
            prefix_cache_hit_rate: None,
            snapshot,
        })
    }

    #[test]
    fn not_fired_when_prefix_cache_deflates_ratio() {
        let s = test_snapshot();
        // Raw: 5500/968 = 5.68x (above threshold)
        // Effective: 5500 * (1 - 0.996) = 22 / 968 = 0.023x (below threshold)
        let result = evaluate(PrefillBoundEvalInput {
            prompt_tokens_per_sec: Some(5500.0),
            generation_tokens_per_sec: Some(968.0),
            decode_efficiency_pct: Some(10.0),
            tpot_ms: Some(66.0),
            tpot_floor_ms: Some(7.85),
            prefix_cache_hit_rate: Some(0.996),
            snapshot: &s,
            chunked_prefill_enabled: None,
        });
        assert!(matches!(result, Rule6Outcome::NotFired));
    }

    #[test]
    fn fires_on_prefill_heavy_workload() {
        let s = test_snapshot();
        match eval_r6_default(
            &s,
            Some(9243.0),
            Some(626.0),
            Some(3.2),
            Some(130.0),
            Some(7.85),
        ) {
            Rule6Outcome::Fired(d) => {
                assert!(d.prompt_gen_ratio > PROMPT_GEN_RATIO_MILD);
                assert_eq!(severity(d.prompt_gen_ratio), Severity::Moderate);
            }
            Rule6Outcome::NotFired => panic!("expected fired"),
        }
    }

    #[test]
    fn pure_prefill_zero_gen_fires_severe() {
        let s = test_snapshot();
        match eval_r6_default(
            &s,
            Some(5000.0),
            Some(0.0),
            Some(3.2),
            Some(130.0),
            Some(7.85),
        ) {
            Rule6Outcome::Fired(d) => {
                assert_eq!(severity(d.prompt_gen_ratio), Severity::Severe);
            }
            Rule6Outcome::NotFired => panic!("expected fired on pure prefill"),
        }
    }

    #[test]
    fn skewed_mode_fires_when_p99_over_5x_mean() {
        let mut s = test_snapshot();
        s.vllm.prompt_tokens_mean = Some(2000.0);
        s.vllm.prompt_tokens_p99 = Some(50_000.0);
        match eval_r6_default(
            &s,
            Some(5500.0),
            Some(500.0),
            Some(10.0),
            Some(50.0),
            Some(7.85),
        ) {
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
        match eval_r6_default(
            &s,
            Some(5500.0),
            Some(500.0),
            Some(10.0),
            Some(50.0),
            Some(7.85),
        ) {
            Rule6Outcome::Fired(d) => {
                assert!(d.prompt_skew_ratio.unwrap_or(0.0) < 5.0);
            }
            Rule6Outcome::NotFired => panic!("expected fired"),
        }
    }

    #[test]
    fn not_fired_when_ratio_below_threshold() {
        let s = test_snapshot();
        assert!(matches!(
            eval_r6_default(
                &s,
                Some(490.0),
                Some(100.0),
                Some(10.0),
                Some(50.0),
                Some(7.85),
            ),
            Rule6Outcome::NotFired
        ));
    }

    #[test]
    fn fires_at_ratio_boundary() {
        let s = test_snapshot();
        match eval_r6_default(
            &s,
            Some(500.0),
            Some(100.0),
            Some(39.9),
            Some(50.0),
            Some(7.85),
        ) {
            Rule6Outcome::Fired(_) => {}
            Rule6Outcome::NotFired => panic!("should fire at ratio 5.0 boundary"),
        }
    }

    #[test]
    fn not_fired_when_efficiency_above_gate() {
        let s = test_snapshot();
        assert!(matches!(
            eval_r6_default(
                &s,
                Some(600.0),
                Some(100.0),
                Some(40.0),
                Some(50.0),
                Some(7.85),
            ),
            Rule6Outcome::NotFired
        ));
    }

    #[test]
    fn not_fired_when_tpot_below_inflation_gate() {
        let s = test_snapshot();
        assert!(matches!(
            eval_r6_default(
                &s,
                Some(600.0),
                Some(100.0),
                Some(10.0),
                Some(30.0),
                Some(7.85),
            ),
            Rule6Outcome::NotFired
        ));
    }

    #[test]
    fn fires_when_tpot_above_inflation_gate() {
        let s = test_snapshot();
        match eval_r6_default(
            &s,
            Some(600.0),
            Some(100.0),
            Some(10.0),
            Some(32.0),
            Some(7.85),
        ) {
            Rule6Outcome::Fired(_) => {}
            Rule6Outcome::NotFired => panic!("should fire when TPOT exceeds 4x floor"),
        }
    }

    #[test]
    fn not_fired_when_prompt_tps_missing() {
        let s = test_snapshot();
        assert!(matches!(
            eval_r6_default(&s, None, Some(100.0), Some(10.0), Some(50.0), Some(7.85)),
            Rule6Outcome::NotFired
        ));
    }

    #[test]
    fn not_fired_when_gen_tps_missing() {
        let s = test_snapshot();
        assert!(matches!(
            eval_r6_default(&s, Some(600.0), None, Some(10.0), Some(50.0), Some(7.85)),
            Rule6Outcome::NotFired
        ));
    }

    #[test]
    fn not_fired_when_full_cache_zeroes_effective_prompt() {
        let s = test_snapshot();
        let result = evaluate(PrefillBoundEvalInput {
            prompt_tokens_per_sec: Some(5000.0),
            generation_tokens_per_sec: Some(0.0),
            decode_efficiency_pct: Some(5.0),
            tpot_ms: Some(100.0),
            tpot_floor_ms: Some(7.85),
            prefix_cache_hit_rate: Some(1.0),
            snapshot: &s,
            chunked_prefill_enabled: None,
        });
        assert!(matches!(result, Rule6Outcome::NotFired));
    }

    #[test]
    fn verbose_miss_when_efficiency_unavailable() {
        let line = r6_verbose_miss_line(R6GateInput {
            prompt_tokens_per_sec: Some(600.0),
            generation_tokens_per_sec: Some(100.0),
            decode_efficiency_pct: None,
            tpot_ms: Some(50.0),
            tpot_floor_ms: Some(7.85),
            prefix_cache_hit_rate: None,
        });
        assert!(line.is_some());
        assert!(line.expect("line").contains("unavailable"));
    }

    #[test]
    fn severity_tiers_correct() {
        assert_eq!(severity(5.0), Severity::Mild);
        assert_eq!(severity(9.9), Severity::Mild);
        assert_eq!(severity(10.0), Severity::Moderate);
        assert_eq!(severity(19.9), Severity::Moderate);
        assert_eq!(severity(20.0), Severity::Severe);
        assert_eq!(severity(f64::INFINITY), Severity::Severe);
    }

    #[test]
    fn aggregate_filters_infinite_ratio() {
        let base = PrefillBoundDetail {
            prompt_gen_ratio: 6.0,
            decode_efficiency_pct: 10.0,
            tpot_ms: Some(50.0),
            tpot_floor_ms: Some(7.85),
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(2048.0),
            prompt_tokens_p99: Some(4096.0),
            prompt_skew_ratio: Some(2.0),
            running_count: None,
        };
        let inf_window = PrefillBoundDetail {
            prompt_gen_ratio: f64::INFINITY,
            ..base.clone()
        };
        let details = vec![base.clone(), base.clone(), inf_window];
        let agg = aggregate_r6_detail(&details);
        assert!(
            agg.prompt_gen_ratio.is_finite(),
            "INFINITY should not poison the mean"
        );
        assert!((agg.prompt_gen_ratio - 6.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_all_infinite_stays_infinite() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: f64::INFINITY,
            decode_efficiency_pct: 10.0,
            tpot_ms: None,
            tpot_floor_ms: None,
            prefix_caching_enabled: None,
            chunked_prefill_enabled: None,
            prompt_tokens_mean: None,
            prompt_tokens_p99: None,
            prompt_skew_ratio: None,
            running_count: None,
        };
        let agg = aggregate_r6_detail(&[d.clone(), d]);
        assert!(agg.prompt_gen_ratio.is_infinite());
    }

    #[test]
    fn impact_scales_with_severity() {
        assert!(impact(Severity::Severe) > impact(Severity::Moderate));
        assert!(impact(Severity::Moderate) > impact(Severity::Mild));
    }

    #[test]
    fn fix_recommends_routing_when_skewed() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 15.0,
            decode_efficiency_pct: 6.7,
            tpot_ms: Some(130.0),
            tpot_floor_ms: Some(7.85),
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(2048.0),
            prompt_tokens_p99: Some(51_200.0),
            prompt_skew_ratio: Some(25.0),
            running_count: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Route long-context requests"));
        assert!(text.contains("Prefill ratio"));
        assert!(!text.contains("Reduce --max-num-batched-tokens to shrink prefill chunk size"));
    }

    #[test]
    fn fix_recommends_prefix_caching_when_disabled_uniform() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 8.2,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            prefix_caching_enabled: Some(false),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("--enable-prefix-caching"));
    }

    #[test]
    fn fix_recommends_enable_chunked_when_prefix_on_but_chunked_off() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 10.0,
            decode_efficiency_pct: 8.0,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(false),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("--enable-chunked-prefill"));
        assert!(!text.contains("Disaggregate prefill and decode"));
    }

    #[test]
    fn fix_recommends_reduce_chunk_size_when_both_enabled() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 8.0,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Reduce --max-num-batched-tokens to 2048"));
        assert!(!text.contains("Disaggregate prefill and decode"));
    }

    #[test]
    fn fix_recommends_disaggregation_when_severe_and_all_mitigations_on() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 22.0,
            decode_efficiency_pct: 5.0,
            tpot_ms: Some(130.0),
            tpot_floor_ms: Some(7.85),
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Disaggregate prefill and decode"));
    }

    #[test]
    fn decode_eff_shows_prefill_bound_qualifier() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 5.1,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("avg when prefill-bound"));
        assert!(text.contains("5.1%"));
    }

    #[test]
    fn appends_reduce_prompt_length_in_uniform_mode() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 8.0,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Reduce prompt length where possible"));
    }

    #[test]
    fn skewed_mode_omits_reduce_prompt_length() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 15.0,
            decode_efficiency_pct: 6.7,
            tpot_ms: Some(130.0),
            tpot_floor_ms: Some(7.85),
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(2048.0),
            prompt_tokens_p99: Some(51_200.0),
            prompt_skew_ratio: Some(25.0),
            running_count: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(!text.contains("Reduce prompt length where possible"));
        assert!(text.contains("Route long-context requests"));
    }

    #[test]
    fn metric_lines_use_consistent_label_padding() {
        let skewed = PrefillBoundDetail {
            prompt_gen_ratio: 15.0,
            decode_efficiency_pct: 6.7,
            tpot_ms: Some(130.0),
            tpot_floor_ms: Some(7.85),
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(2048.0),
            prompt_tokens_p99: Some(51_200.0),
            prompt_skew_ratio: Some(25.0),
            running_count: None,
        };
        let skewed_lines = format_prefill_bound_window_issue(&skewed, 40);
        assert_eq!(
            skewed_lines[3],
            "    Prefill ratio       15.0x  prompt tok/s vs gen tok/s"
        );
        assert_eq!(
            skewed_lines[4],
            "    Decode eff.         (avg when prefill-bound)   6.7%  of HW ceiling"
        );
        assert_eq!(skewed_lines[5], "    Prompt mean         2048 tok");
        assert_eq!(
            skewed_lines[6],
            "    Prompt p99          51200 tok  (25x mean)"
        );

        let uniform = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 5.1,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
        };
        let uniform_lines = format_prefill_bound_window_issue(&uniform, 100);
        assert_eq!(
            uniform_lines[3],
            "    Prefill ratio       12.0x  prompt tok/s vs gen tok/s"
        );
        assert_eq!(
            uniform_lines[4],
            "    Decode eff.         (avg when prefill-bound)   5.1%  of HW ceiling"
        );
        assert_eq!(uniform_lines[5], "    Avg prompt          4096 tok");
    }

    #[test]
    fn dynamic_batch_budget_rounds_to_128() {
        assert_eq!(recommended_batch_token_budget(1333.0, 161.0), 896);
        assert_eq!(recommended_batch_token_budget(8000.0, 50.0), 4096);
        assert_eq!(recommended_batch_token_budget(200.0, 10.0), 128);
    }

    #[test]
    fn fix_uses_dynamic_batch_budget_when_running_count_available() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 8.0,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(1333.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: Some(161.0),
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Reduce --max-num-batched-tokens to 896"));
    }
}
