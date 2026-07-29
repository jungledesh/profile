use crate::collectors::RawSnapshot;

/// Primary trigger: prompt-to-generation token ratio.
/// Break-even is ridge / decode_batch. For A100-H100 at batch 30+, this falls in
/// the 1.3-5.7 range. 5.0 catches meaningful prefill dominance without
/// false-positiving on agent workloads (~4:1 ratio). Calibrate with production data.
/// Prompt-to-generation token ratio above which prefill dominates decode.
/// Shared by R1 (defer to R6) and R6 (mild fire gate). Calibrate in one place.
pub const PROMPT_GEN_RATIO_MILD: f64 = 5.0;
const PROMPT_GEN_RATIO_MODERATE: f64 = 10.0;
const PROMPT_GEN_RATIO_SEVERE: f64 = 20.0;

/// TPOT must be inflated above this multiple of the physics floor for R6 to fire.
/// Prefill ratio alone isn't a problem if TPOT isn't inflated (server is handling it).
const TPOT_INFLATION_GATE: f64 = 4.0;

/// Confidence cap when TPOT evidence could not be checked (missing tpot or floor).
/// Below Medium threshold (0.6) so the label renders Low.
pub(super) const TPOT_UNVERIFIED_CONFIDENCE_CAP: f64 = 0.5;

/// Decode efficiency below this indicates underperformance that prefill might explain.
const DECODE_EFFICIENCY_GATE: f64 = 40.0;
const PROMPT_SKEW_RATIO: f64 = 5.0;
const SKEWED_EXPECTED: &str = "Eliminates head-of-line blocking from long-tail prompts.";

/// Fixed label column width for R6 metric rows (longest label: "Prefill ratio").
const R6_METRIC_LABEL_W: usize = 20;

fn r6_metric_line(label: &str, value: &str) -> String {
    format!("    {label:<R6_METRIC_LABEL_W$}{value}")
}

/// Fallback when prompt mean or running count unavailable for budget derivation.
const DEFAULT_BATCH_TOKEN_BUDGET: u64 = 2048;

/// Relative band around the recommended `--max-num-batched-tokens` where we
/// treat the knob as already set and name the FLOPs wall instead.
/// Provisional; calibrate on RunPod (same posture as limiter thresholds).
const R6_BUDGET_BAND: f64 = 0.20;

/// Wall-clock policy: how much decode-step stretch we accept when prefill
/// shares the step. Judgment constant, operator-arguable. Not physics coupling.
const DECODE_STRETCH_TARGET: f64 = 0.25;

/// Fallback heuristic: target steps to ingest an average prompt.
/// 1 = no chunking benefit (whole prompt stalls decode once, hard);
/// large = smooth decode but slow TTFT. 2 halves the worst-case decode
/// stall while only modestly delaying prompts. Judgment constant.
/// Used only when `ridge_batch_size` is unavailable.
const PREFILL_TARGET_CHUNKS: f64 = 2.0;

fn round_up_128(n: f64) -> u64 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        ((n / 128.0).ceil() as u64).saturating_mul(128)
    }
}

/// Primary budget from decode ridge. No `running` term: `--max-num-batched-tokens`
/// is the whole step budget; vLLM subtracts live decodes itself.
///
/// "(est)" comes off only after a hardware sweep (512 / 2048 / 8192 / formula
/// value) shows predicted stretch tracks measured TPOT. Until then: estimate,
/// by policy.
fn ridge_batch_token_budget(ridge_batch_size: f64) -> u64 {
    round_up_128(ridge_batch_size * (1.0 + DECODE_STRETCH_TARGET))
}

/// Fallback: chunk average prompt into PREFILL_TARGET_CHUNKS steps after decode overhead.
fn recommended_batch_token_budget(prompt_mean: f64, running: f64) -> u64 {
    let raw = prompt_mean / PREFILL_TARGET_CHUNKS + running;
    round_up_128(raw)
}

/// Which tier produced `batch_token_budget`. Default is a hardcoded fallback;
/// wall claims require a derived (ridge or workload) recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchBudgetTier {
    Ridge,
    Workload,
    Default,
}

fn batch_token_budget(d: &PrefillBoundDetail) -> (u64, BatchBudgetTier) {
    if let Some(ridge) = d.ridge_batch_size.filter(|r| r.is_finite() && *r > 0.0) {
        return (ridge_batch_token_budget(ridge), BatchBudgetTier::Ridge);
    }
    match (d.prompt_tokens_mean, d.running_count) {
        (Some(pm), Some(rc)) if pm.is_finite() && pm > 0.0 && rc.is_finite() && rc >= 0.0 => (
            recommended_batch_token_budget(pm, rc),
            BatchBudgetTier::Workload,
        ),
        _ => (DEFAULT_BATCH_TOKEN_BUDGET, BatchBudgetTier::Default),
    }
}

fn batch_budget_paren(d: &PrefillBoundDetail) -> Option<&'static str> {
    d.ridge_batch_size
        .filter(|r| r.is_finite() && *r > 0.0)
        .map(|_| {
            if d.is_hybrid {
                "(est; optimistic on long prompts)"
            } else {
                "(est)"
            }
        })
}

fn batch_budget_fix_bullet(d: &PrefillBoundDetail, verb: &str) -> String {
    let (budget, _) = batch_token_budget(d);
    match batch_budget_paren(d) {
        Some(paren) => {
            format!(
                "      • {verb} --max-num-batched-tokens to {budget} {paren} to shrink prefill chunk size. Lower for smoother TPOT, raise for lower TTFT."
            )
        }
        None => {
            format!(
                "      • {verb} --max-num-batched-tokens to {budget} to shrink prefill chunk size. Lower for smoother TPOT, raise for lower TTFT."
            )
        }
    }
}

/// Knob already sits within the band of a derived recommendation: no single-GPU
/// retune adds FLOPs. Gauge missing or default-tier budget → never claim the wall.
fn on_compute_wall(d: &PrefillBoundDetail) -> bool {
    let Some(configured) = d.max_num_batched_tokens else {
        return false;
    };
    let (recommended, tier) = batch_token_budget(d);
    if tier == BatchBudgetTier::Default || recommended == 0 {
        return false;
    }
    let rec = recommended as f64;
    let cfg = f64::from(configured);
    (cfg - rec).abs() / rec <= R6_BUDGET_BAND
}

fn compute_wall_fix_lines(d: &PrefillBoundDetail) -> (Vec<String>, String) {
    let mut bullets = vec![
        "      • Prefill is compute-bound for this prompt mix; no single-GPU knob adds FLOPs."
            .to_string(),
        "      • Disaggregate prefill and decode onto separate workers (vLLM disaggregated serving, requires 2+ nodes).".to_string(),
        "      • Add a replica to scale out.".to_string(),
    ];
    if d.prefix_caching_enabled.is_none() {
        bullets.push(
            "      • Enable --enable-prefix-caching if not already on; cached prefixes skip prefill."
                .to_string(),
        );
    }
    (
        bullets,
        "Scale out prefill compute; single-GPU retunes cannot add FLOPs.".to_string(),
    )
}

fn knob_fix_lines(d: &PrefillBoundDetail) -> (Vec<String>, String) {
    (
        vec![batch_budget_fix_bullet(d, "Set")],
        "Lower TTFT variance, steadier decode throughput.".to_string(),
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrefillBoundDetail {
    pub prompt_gen_ratio: f64,
    pub decode_efficiency_pct: f64,
    pub tpot_ms: Option<f64>,
    pub tpot_floor_ms: Option<f64>,
    /// True when the TPOT-inflation mute could not run (tpot or floor missing).
    pub tpot_unverified: bool,
    pub prefix_caching_enabled: Option<bool>,
    pub chunked_prefill_enabled: Option<bool>,
    pub prompt_tokens_mean: Option<f64>,
    pub prompt_tokens_p99: Option<f64>,
    pub prompt_skew_ratio: Option<f64>,
    pub running_count: Option<f64>,
    /// Decode ridge batch size (tokens). When set, drives the primary budget.
    pub ridge_batch_size: Option<f64>,
    /// Configured `--max-num-batched-tokens` (gauge, else config). Unknown → no wall claim.
    pub max_num_batched_tokens: Option<u32>,
    /// Hybrid/linear catalog model: ridge budget is optimistic on long prompts.
    pub is_hybrid: bool,
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
    pub ridge_batch_size: Option<f64>,
    /// Gauge else config; resolved by the caller before evaluate.
    pub max_num_batched_tokens: Option<u32>,
    pub is_hybrid: bool,
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
        ridge_batch_size,
        max_num_batched_tokens,
        is_hybrid,
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
    let tpot_unverified = !(tpot_ms.is_some_and(|v| v.is_finite() && v > 0.0)
        && tpot_floor_ms.is_some_and(|v| v.is_finite() && v > 0.0));

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
        tpot_unverified,
        prefix_caching_enabled: snapshot.vllm.cache_config.enable_prefix_caching,
        chunked_prefill_enabled,
        prompt_tokens_mean: snapshot.vllm.prompt_tokens_mean,
        prompt_tokens_p99: snapshot.vllm.prompt_tokens_p99,
        prompt_skew_ratio,
        running_count: snapshot.vllm.num_requests_running,
        ridge_batch_size,
        max_num_batched_tokens,
        is_hybrid,
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
        tpot_floor_ms: details.first().and_then(|d| d.tpot_floor_ms),
        tpot_unverified: details.iter().any(|d| d.tpot_unverified),
        prefix_caching_enabled: details.first().and_then(|d| d.prefix_caching_enabled),
        chunked_prefill_enabled: details.first().and_then(|d| d.chunked_prefill_enabled),
        prompt_tokens_mean: super::mean_of_present(
            details.iter().filter_map(|d| d.prompt_tokens_mean),
        ),
        prompt_tokens_p99: details
            .iter()
            .filter_map(|d| d.prompt_tokens_p99)
            .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v)))),
        prompt_skew_ratio: details
            .iter()
            .filter_map(|d| d.prompt_skew_ratio)
            .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v)))),
        running_count: super::mean_of_present(details.iter().filter_map(|d| d.running_count)),
        ridge_batch_size: details.first().and_then(|d| d.ridge_batch_size),
        max_num_batched_tokens: details.first().and_then(|d| d.max_num_batched_tokens),
        is_hybrid: details.first().is_some_and(|d| d.is_hybrid),
    }
}

fn skewed_mode(d: &PrefillBoundDetail) -> bool {
    d.prompt_skew_ratio
        .filter(|r| r.is_finite())
        .is_some_and(|r| r >= PROMPT_SKEW_RATIO)
}

fn cause_tpot_line(d: &PrefillBoundDetail) -> Option<String> {
    let tpot = d.tpot_ms.filter(|v| v.is_finite() && *v > 0.0)?;
    let floor = d.tpot_floor_ms.filter(|v| v.is_finite() && *v > 0.0)?;
    let mult = tpot / floor;
    let mult_display = if mult >= 10.0 {
        format!("{:.0}x", mult)
    } else {
        format!("{:.1}x", mult)
    };
    Some(format!(
        "      Decode starves while each step swallows prompt: tpot {tpot:.0}ms vs {floor:.1}ms floor ({mult_display})."
    ))
}

pub(super) fn prefill_fix_lines(d: &PrefillBoundDetail, sev: Severity) -> (Vec<String>, String) {
    let prefix_off = d.prefix_caching_enabled == Some(false);
    let chunked_on = d.chunked_prefill_enabled == Some(true);
    let chunked_not_enabled = d.chunked_prefill_enabled != Some(true);

    if prefix_off {
        (
            vec![
                "      • Enable automatic prefix caching (--enable-prefix-caching).".to_string(),
                "      Repeated prompt prefixes are re-computed every request.".to_string(),
            ],
            "20-40% reduction in prefill time for workloads with shared prefixes.".to_string(),
        )
    } else if chunked_not_enabled {
        let bullet = batch_budget_fix_bullet(d, "Set");
        (
            vec![
                "      • Enable chunked prefill (--enable-chunked-prefill).".to_string(),
                bullet,
            ],
            "Decode batches interleave with prefill, reducing head-of-line blocking.".to_string(),
        )
    } else if sev == Severity::Severe && d.prefix_caching_enabled == Some(true) && chunked_on {
        (
            vec![
                "      • Disaggregate prefill and decode onto separate workers (vLLM disaggregated serving, requires 2+ nodes).".to_string(),
            ],
            "Full separation of prefill and decode compute paths.".to_string(),
        )
    } else if chunked_on {
        if on_compute_wall(d) {
            compute_wall_fix_lines(d)
        } else {
            knob_fix_lines(d)
        }
    } else {
        // Logically unreachable: branch 2 forces chunked_on=true, branch 4 catches it.
        // Safe fallback instead of panic in library code.
        knob_fix_lines(d)
    }
}

pub(super) fn format_prefill_bound_window_issue(
    d: &PrefillBoundDetail,
    seen_pct: u32,
) -> Vec<String> {
    let sev = severity(d.prompt_gen_ratio);
    let conf = if d.tpot_unverified {
        confidence(sev).min(TPOT_UNVERIFIED_CONFIDENCE_CAP)
    } else {
        confidence(sev)
    };
    let (fix_bullets, expected_normal) = prefill_fix_lines(d, sev);
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
        if let Some(line) = cause_tpot_line(d) {
            lines.push(line);
        }
    }
    if d.tpot_unverified {
        let note = match (
            d.tpot_ms.filter(|v| v.is_finite() && *v > 0.0),
            d.tpot_floor_ms.filter(|v| v.is_finite() && *v > 0.0),
        ) {
            (None, _) => "(low confidence, TPOT unavailable)",
            (Some(_), None) => "(low confidence, TPOT floor unavailable)",
            (Some(_), Some(_)) => "(low confidence, TPOT check unavailable)",
        };
        lines.push(format!("      {note}"));
    }

    lines.push(String::new());
    let expected = if let (true, Some(p99), Some(mean)) = (
        skewed,
        d.prompt_tokens_p99.filter(|v| v.is_finite() && *v > 0.0),
        d.prompt_tokens_mean.filter(|v| v.is_finite() && *v > 0.0),
    ) {
        let mut safe = Vec::new();
        if d.chunked_prefill_enabled != Some(true) {
            safe.push(
                "      • Enable --enable-chunked-prefill to interleave short requests with long-prompt chunks."
                    .to_string(),
            );
        }
        super::push_bullet_with_subline(
            &mut safe,
            format!(
                "      • Route long-context requests (p99: {p99:.0} tok) to a dedicated vLLM instance."
            ),
            Some(&format!(
                "Short requests ({mean:.0} tok mean) are blocked by outlier prefills."
            )),
        );
        let rejects = vec![
            "      • Cap --max-model-len at p99 prompt length to reject outlier prompts, or truncate them at app layer.".to_string(),
        ];
        super::push_grouped_fixes(&mut lines, safe, Vec::new(), rejects, false);
        SKEWED_EXPECTED.to_string()
    } else {
        lines.push("    Fix:".to_string());
        lines.extend(fix_bullets);
        if !skewed {
            lines.push("      • Reduce prompt length where possible.".to_string());
        }
        expected_normal
    };
    lines.push(String::new());
    lines.push(format!("    Expected: {expected}"));
    lines.push(format!("    Confidence: {}", super::confidence_label(conf)));
    super::with_seen_pct(lines, seen_pct)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{CacheConfigLabels, RawSnapshot, VllmRawMetrics};

    fn test_snapshot() -> RawSnapshot {
        crate::collectors::snap_vllm(VllmRawMetrics {
            cache_config: CacheConfigLabels::default(),
            ..Default::default()
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
        evaluate(PrefillBoundEvalInput {
            prompt_tokens_per_sec: prompt_tps,
            generation_tokens_per_sec: gen_tps,
            decode_efficiency_pct: eff,
            tpot_ms,
            tpot_floor_ms,
            prefix_cache_hit_rate: None,
            snapshot,
            chunked_prefill_enabled: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
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
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
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
            Rule6Outcome::Fired(d) => {
                assert!(!d.tpot_unverified);
                let conf = confidence(severity(d.prompt_gen_ratio));
                assert!(conf > TPOT_UNVERIFIED_CONFIDENCE_CAP);
            }
            Rule6Outcome::NotFired => panic!("should fire when TPOT exceeds 4x floor"),
        }
    }

    #[test]
    fn fires_low_confidence_when_tpot_missing() {
        let s = test_snapshot();
        match eval_r6_default(&s, Some(600.0), Some(100.0), Some(10.0), None, Some(7.85)) {
            Rule6Outcome::Fired(d) => {
                assert!(d.tpot_unverified);
                let text = format_prefill_bound_window_issue(&d, 100).join("\n");
                assert!(text.contains("(low confidence, TPOT unavailable)"));
                assert!(text.contains("Confidence: Low"));
            }
            Rule6Outcome::NotFired => panic!("should fire with TPOT unverified"),
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
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
        });
        assert!(matches!(result, Rule6Outcome::NotFired));
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
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(2048.0),
            prompt_tokens_p99: Some(4096.0),
            prompt_skew_ratio: Some(2.0),
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
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
            tpot_unverified: false,
            prefix_caching_enabled: None,
            chunked_prefill_enabled: None,
            prompt_tokens_mean: None,
            prompt_tokens_p99: None,
            prompt_skew_ratio: None,
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
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
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(false),
            prompt_tokens_mean: Some(2048.0),
            prompt_tokens_p99: Some(51_200.0),
            prompt_skew_ratio: Some(25.0),
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Route long-context requests"));
        assert!(text.contains("Prefill ratio"));
        assert!(text.contains("    Rejects requests:"));
        assert!(text.contains(&format!("Expected: {SKEWED_EXPECTED}")));
        let fix = text.find("    Fix:").expect("Fix");
        let chunked = text
            .find(
                "Enable --enable-chunked-prefill to interleave short requests with long-prompt chunks.",
            )
            .expect("chunked bullet");
        let rejects = text.find("    Rejects requests:").expect("Rejects");
        let route = text.find("Route long-context requests").expect("route");
        let cap = text
            .find("Cap --max-model-len at p99 prompt length")
            .expect("reject bullet");
        assert!(fix < chunked && chunked < route && route < rejects && rejects < cap);
        assert!(!text.contains("Set --max-num-batched-tokens to shrink prefill chunk size"));
    }

    #[test]
    fn skewed_with_chunked_on_skips_chunked_bullet() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 15.0,
            decode_efficiency_pct: 6.7,
            tpot_ms: Some(130.0),
            tpot_floor_ms: Some(7.85),
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(2048.0),
            prompt_tokens_p99: Some(51_200.0),
            prompt_skew_ratio: Some(25.0),
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(!text.contains("Enable --enable-chunked-prefill to interleave short requests"));
        assert!(text.contains("    Rejects requests:"));
        assert!(text.contains(&format!("Expected: {SKEWED_EXPECTED}")));
        let fix = text.find("    Fix:").expect("Fix");
        let route = text.find("Route long-context requests").expect("route");
        let rejects = text.find("    Rejects requests:").expect("Rejects");
        assert!(fix < route && route < rejects);
    }

    #[test]
    fn fix_recommends_prefix_caching_when_disabled_uniform() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 8.2,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            tpot_unverified: false,
            prefix_caching_enabled: Some(false),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
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
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(false),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
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
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Set --max-num-batched-tokens to 2048"));
        assert!(text.contains("to shrink prefill chunk size"));
        assert!(!text.contains("Disaggregate prefill and decode"));
    }

    #[test]
    fn fix_recommends_disaggregation_when_severe_and_all_mitigations_on() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 22.0,
            decode_efficiency_pct: 5.0,
            tpot_ms: Some(130.0),
            tpot_floor_ms: Some(7.85),
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
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
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
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
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
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
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(2048.0),
            prompt_tokens_p99: Some(51_200.0),
            prompt_skew_ratio: Some(25.0),
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(!text.contains("Reduce prompt length where possible"));
        assert!(text.contains("Route long-context requests"));
    }

    #[test]
    fn skewed_mode_without_p99_falls_back_to_normal_fix() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 15.0,
            decode_efficiency_pct: 6.7,
            tpot_ms: Some(130.0),
            tpot_floor_ms: Some(7.85),
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(2048.0),
            prompt_tokens_p99: None,
            prompt_skew_ratio: Some(25.0),
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
        };
        assert!(skewed_mode(&d));
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(!text.contains("Route long-context requests"));
        assert!(!text.contains("    Rejects requests:"));
        assert!(text.contains("    Fix:"));
        assert!(text.contains("--max-num-batched-tokens"));
        assert!(!text.contains(SKEWED_EXPECTED));
    }

    #[test]
    fn metric_lines_use_consistent_label_padding() {
        let skewed = PrefillBoundDetail {
            prompt_gen_ratio: 15.0,
            decode_efficiency_pct: 6.7,
            tpot_ms: Some(130.0),
            tpot_floor_ms: Some(7.85),
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(2048.0),
            prompt_tokens_p99: Some(51_200.0),
            prompt_skew_ratio: Some(25.0),
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
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
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
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
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(1333.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: Some(161.0),
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Set --max-num-batched-tokens to 896"));
    }

    #[test]
    fn ridge_budget_rounds_up_128_with_stretch_target() {
        // 153 × 1.25 = 191.25 → 256
        assert_eq!(ridge_batch_token_budget(153.0), 256);
        // 128 × 1.25 = 160 → 256
        assert_eq!(ridge_batch_token_budget(128.0), 256);
        // 1024 × 1.25 = 1280 → 1280
        assert_eq!(ridge_batch_token_budget(1024.0), 1280);
    }

    #[test]
    fn ridge_budget_preferred_over_prompt_heuristic() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 8.0,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(1333.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: Some(161.0),
            ridge_batch_size: Some(153.0),
            max_num_batched_tokens: None,
            is_hybrid: false,
        };
        assert_eq!(batch_token_budget(&d).0, 256);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(
            text.contains(
                "Set --max-num-batched-tokens to 256 (est) to shrink prefill chunk size."
            )
        );
        assert!(text.contains("Lower for smoother TPOT, raise for lower TTFT"));
        assert!(!text.contains("decode stretch"));
    }

    #[test]
    fn cause_line_tpot_vs_floor() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 8.0,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
            ridge_batch_size: Some(153.0),
            max_num_batched_tokens: None,
            is_hybrid: false,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains(
            "Decode starves while each step swallows prompt: tpot 80ms vs 7.8ms floor (10x)."
        ));
        assert!(!text.contains("GPU is busy"));
    }

    #[test]
    fn ridge_budget_hybrid_wording() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 8.0,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(1333.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: Some(161.0),
            ridge_batch_size: Some(153.0),
            max_num_batched_tokens: None,
            is_hybrid: true,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains(
            "Set --max-num-batched-tokens to 256 (est; optimistic on long prompts) to shrink prefill chunk size."
        ));
        assert!(!text.contains("decode stretch"));
    }

    #[test]
    fn fallback_budget_when_ridge_absent() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 8.0,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: None,
            prompt_tokens_p99: None,
            prompt_skew_ratio: None,
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            is_hybrid: false,
        };
        assert_eq!(batch_token_budget(&d).0, DEFAULT_BATCH_TOKEN_BUDGET);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Set --max-num-batched-tokens to 2048"));
        assert!(!text.contains("decode stretch"));
    }

    fn wall_path_base(
        configured: Option<u32>,
        ridge: Option<f64>,
        prefix: Option<bool>,
        ratio: f64,
    ) -> PrefillBoundDetail {
        PrefillBoundDetail {
            prompt_gen_ratio: ratio,
            decode_efficiency_pct: 8.0,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            tpot_unverified: false,
            prefix_caching_enabled: prefix,
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
            ridge_batch_size: ridge,
            max_num_batched_tokens: configured,
            is_hybrid: false,
        }
    }

    #[test]
    fn compute_wall_when_configured_within_band_of_ridge_budget() {
        // ridge 1638.4 → budget 2048; configured 2048 within band.
        let d = wall_path_base(Some(2048), Some(1638.4), Some(true), 12.0);
        assert_eq!(batch_token_budget(&d), (2048, BatchBudgetTier::Ridge));
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains(
            "Prefill is compute-bound for this prompt mix; no single-GPU knob adds FLOPs."
        ));
        assert!(text.contains("Disaggregate prefill and decode onto separate workers"));
        assert!(text.contains("Add a replica to scale out."));
        assert!(!text.contains("--max-num-batched-tokens"));
    }

    #[test]
    fn knob_when_configured_below_band() {
        let d = wall_path_base(Some(1024), Some(1638.4), Some(true), 12.0);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Set --max-num-batched-tokens to 2048"));
        assert!(!text.contains("no single-GPU knob adds FLOPs"));
    }

    #[test]
    fn knob_when_configured_above_band() {
        let d = wall_path_base(Some(4096), Some(1638.4), Some(true), 12.0);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Set --max-num-batched-tokens to 2048"));
        assert!(!text.contains("no single-GPU knob adds FLOPs"));
    }

    #[test]
    fn knob_when_configured_unknown() {
        let d = wall_path_base(None, Some(1638.4), Some(true), 12.0);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Set --max-num-batched-tokens to 2048"));
        assert!(!text.contains("no single-GPU knob adds FLOPs"));
    }

    #[test]
    fn knob_when_budget_is_default_tier_even_if_exact_match() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 8.0,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: None,
            prompt_tokens_p99: None,
            prompt_skew_ratio: None,
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: Some(2048),
            is_hybrid: false,
        };
        assert_eq!(batch_token_budget(&d), (2048, BatchBudgetTier::Default));
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Set --max-num-batched-tokens to 2048"));
        assert!(!text.contains("no single-GPU knob adds FLOPs"));
    }

    #[test]
    fn wall_path_prefix_unknown_adds_conditional_bullet() {
        let d = wall_path_base(Some(2048), Some(1638.4), None, 12.0);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains(
            "Enable --enable-prefix-caching if not already on; cached prefixes skip prefill."
        ));
        let d_on = wall_path_base(Some(2048), Some(1638.4), Some(true), 12.0);
        let text_on = format_prefill_bound_window_issue(&d_on, 100).join("\n");
        assert!(!text_on.contains("Enable --enable-prefix-caching if not already on"));
    }

    #[test]
    fn severe_prefix_on_chunked_disagg_unchanged_when_within_band() {
        // Severe branch stays above the wall gate; byte-identical single disagg bullet.
        let d = wall_path_base(Some(2048), Some(1638.4), Some(true), 22.0);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Disaggregate prefill and decode onto separate workers"));
        assert!(!text.contains("no single-GPU knob adds FLOPs"));
        assert!(!text.contains("Add a replica to scale out."));
        assert!(!text.contains("--max-num-batched-tokens"));
    }

    #[test]
    fn compute_wall_at_exact_band_boundary() {
        // ridge 2048 → budget 2560; configured 3072 is exactly +20%.
        let d = wall_path_base(Some(3072), Some(2048.0), Some(true), 12.0);
        assert_eq!(batch_token_budget(&d), (2560, BatchBudgetTier::Ridge));
        let (budget, _) = batch_token_budget(&d);
        let rel = (f64::from(3072) - budget as f64).abs() / budget as f64;
        assert!((rel - R6_BUDGET_BAND).abs() < 1e-12);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("no single-GPU knob adds FLOPs"));
        assert!(!text.contains("--max-num-batched-tokens"));
    }
}
