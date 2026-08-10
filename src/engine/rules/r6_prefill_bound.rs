use crate::collectors::RawSnapshot;

/// Primary trigger: prompt-to-generation token ratio.
/// Break-even is ridge / decode_batch. For A100-H100 at batch 30+, this falls in
/// the 1.3-5.7 range. 5.0 catches meaningful prefill dominance without
/// false-positiving on agent workloads (~4:1 ratio). Calibrate with production data.
/// Prompt-to-generation token ratio above which prefill dominates decode.
/// R6 mild fire gate. Calibrate in one place.
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

/// Model-specific boot floors (e.g. Gemma 4 multimodal 2496) are not scraped.
/// Named Set targets can still sit below that floor; this subline is the backstop.
const BATCH_TOKEN_BOOT_REJECT_SUBLINE: &str =
    "If vLLM rejects this at boot, its error names the model minimum; use that value.";

const BATCH_TOKEN_DIRECTIONS: &str = "Lower for smoother TPOT, raise for lower TTFT.";

const BATCH_TOKEN_SET_EXPECTED: &str = "Lower TTFT variance, steadier decode throughput.";

const BATCH_TOKEN_ALREADY_SET_EXPECTED: &str =
    "No --max-num-batched-tokens change; configured value already matches the target.";

const BATCH_TOKEN_ALREADY_ABOVE_DEFAULT_EXPECTED: &str =
    "No --max-num-batched-tokens change; configured value is already above the default.";

const BATCH_TOKEN_UNREAD_BULLET: &str = "      • --max-num-batched-tokens unread on this server.";

const BATCH_TOKEN_UNREAD_EXPECTED: &str = "Steadier decode once prefill sharing is confirmed.";

/// Launch prescription: when configured is readable, Set 2048 (default) + optional
/// scraped page floor. No ridge/workload Set target (those tiers still feed
/// `on_compute_wall` only).
///
/// Never Set *down* to 2048 when configured is already higher. When configured
/// is unread (common: no gauge, `/info` 404), never blind-Set: name unread,
/// page floor + directions (one subline), boot-reject subline. Operator picks.
///
/// Second return is whether a named `Set … to 2048` was emitted (structural;
/// do not re-read printed text to decide Expected).
fn batch_token_budget_bullets(d: &PrefillBoundDetail, verb: &str) -> (Vec<String>, bool) {
    let mut out = Vec::new();
    let default = DEFAULT_BATCH_TOKEN_BUDGET;
    let floor = d.chunk_floor.filter(|f| *f > 0);

    // Never prescribe a shrink to the launch default.
    if d.max_num_batched_tokens
        .is_some_and(|c| u64::from(c) > default)
    {
        return (out, false);
    }

    // Unread: no absolute Set (cannot know if that would shrink). Floor + direction.
    if d.max_num_batched_tokens.is_none() {
        out.push(BATCH_TOKEN_UNREAD_BULLET.to_string());
        let guide = match floor {
            Some(f) => format!("Page floor is {f} (do not go below). {BATCH_TOKEN_DIRECTIONS}"),
            None => BATCH_TOKEN_DIRECTIONS.to_string(),
        };
        out.push(format!("        {guide}"));
        out.push(format!("        {BATCH_TOKEN_BOOT_REJECT_SUBLINE}"));
        out.push(String::new());
        return (out, false);
    }

    if let Some(f) = floor
        && default < u64::from(f)
    {
        super::push_bullet_with_subline(
            &mut out,
            format!(
                "      • Default --max-num-batched-tokens is {default}; page floor is {f} (do not go below). {BATCH_TOKEN_DIRECTIONS}"
            ),
            Some(BATCH_TOKEN_BOOT_REJECT_SUBLINE),
        );
        return (out, false);
    }

    if super::already_set_u32(d.max_num_batched_tokens, default) {
        return (out, false);
    }

    let main = match floor {
        Some(f) => format!(
            "      • {verb} --max-num-batched-tokens to {default} (default); page floor is {f} (do not go below). {BATCH_TOKEN_DIRECTIONS}"
        ),
        None => format!(
            "      • {verb} --max-num-batched-tokens to {default} (default). {BATCH_TOKEN_DIRECTIONS}"
        ),
    };
    super::push_bullet_with_subline(&mut out, main, Some(BATCH_TOKEN_BOOT_REJECT_SUBLINE));
    (out, true)
}

fn batch_token_prescription_expected(d: &PrefillBoundDetail, named_set: bool) -> &'static str {
    if named_set {
        return BATCH_TOKEN_SET_EXPECTED;
    }
    if d.max_num_batched_tokens.is_none() {
        return BATCH_TOKEN_UNREAD_EXPECTED;
    }
    if d.max_num_batched_tokens
        .is_some_and(|c| u64::from(c) > DEFAULT_BATCH_TOKEN_BUDGET)
    {
        return BATCH_TOKEN_ALREADY_ABOVE_DEFAULT_EXPECTED;
    }
    if let Some(f) = d.chunk_floor.filter(|f| *f > 0)
        && DEFAULT_BATCH_TOKEN_BUDGET < u64::from(f)
    {
        // Readable configured, floor above default: operator still tunes at/above floor.
        return BATCH_TOKEN_SET_EXPECTED;
    }
    BATCH_TOKEN_ALREADY_SET_EXPECTED
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

const R6_TERMINAL_VERIFY: &str =
    "      • Verify prefix caching, chunked prefill and max-num-batched-tokens took effect.";

const SEVERE_FLOPS_WALL_EXPECTED: &str =
    "No large decode recovery on this GPU until prompt work drops or prefill scales out.";

fn compute_wall_fix_lines(_d: &PrefillBoundDetail) -> (Vec<String>, String, bool) {
    // Unknown prefix caching must not get an Enable bullet (unknown is not off).
    // R6_TERMINAL_VERIFY already asks the operator to confirm prefix caching.
    let bullets = vec![
        "      • Prefill is compute-bound for this prompt mix; no single-GPU knob adds FLOPs."
            .to_string(),
        R6_TERMINAL_VERIFY.to_string(),
        "      • Disaggregate prefill and decode onto separate workers (vLLM disaggregated serving, requires 2+ nodes).".to_string(),
        "      • Add a replica to scale out.".to_string(),
    ];
    (
        bullets,
        "Scale out prefill compute; single-GPU retunes cannot add FLOPs.".to_string(),
        true,
    )
}

/// Severe: FLOPs wall first. Chunked/batched knobs do not add prefill compute.
/// Trailing Enable only when chunked is confirmed off (unknown → omit, no guess).
/// "Reduce prompt length" is appended by the uniform format path.
fn severe_flops_wall_fix_lines(d: &PrefillBoundDetail) -> (Vec<String>, String, bool) {
    let mut bullets = vec![
        "      • Prefill FLOPs dominate this mix; single-GPU retunes cannot add prefill compute."
            .to_string(),
        "      • Disaggregate prefill and decode onto separate workers (vLLM disaggregated serving, requires 2+ nodes).".to_string(),
        "      • Add a replica to scale out.".to_string(),
    ];
    let chunked_off = d.chunked_prefill_enabled == Some(false);
    if chunked_off {
        bullets.push("      • Enable chunked prefill (--enable-chunked-prefill).".to_string());
    }
    // Enable is not a FLOPs lever; keep the loop open only while that trailing
    // config action remains (same posture as mild Enable → non-terminal).
    (
        bullets,
        SEVERE_FLOPS_WALL_EXPECTED.to_string(),
        !chunked_off,
    )
}

fn knob_fix_lines(d: &PrefillBoundDetail) -> (Vec<String>, String, bool) {
    let (bullets, named_set) = batch_token_budget_bullets(d, "Set");
    let expected = batch_token_prescription_expected(d, named_set).to_string();
    // Default+floor prescription is always a knob (or already-set / floor-above-default
    // info). Never terminal from this path alone.
    (bullets, expected, false)
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
    /// Decode ridge batch size (tokens). When set, drives `on_compute_wall` only.
    pub ridge_batch_size: Option<f64>,
    /// Configured `--max-num-batched-tokens` (gauge, else config). Unknown → no wall claim.
    pub max_num_batched_tokens: Option<u32>,
    /// Scraped `cache_config.block_size`; minimum bootable chunk size.
    pub chunk_floor: Option<u32>,
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

    let prompt_tokens_mean = snapshot.vllm.prompt_tokens_mean;
    let prompt_tokens_p99 = snapshot.vllm.prompt_tokens_p99;
    let prompt_skew_ratio = skew_ratio_from_lengths(prompt_tokens_mean, prompt_tokens_p99);

    Rule6Outcome::Fired(PrefillBoundDetail {
        prompt_gen_ratio: ratio,
        decode_efficiency_pct: eff,
        tpot_ms,
        tpot_floor_ms,
        tpot_unverified,
        prefix_caching_enabled: snapshot.vllm.cache_config.enable_prefix_caching,
        chunked_prefill_enabled,
        prompt_tokens_mean,
        prompt_tokens_p99,
        prompt_skew_ratio,
        running_count: snapshot.vllm.num_requests_running,
        ridge_batch_size,
        max_num_batched_tokens,
        chunk_floor: super::chunk_batched_tokens_floor(&snapshot.vllm.cache_config),
    })
}

pub(super) fn aggregate_r6_detail(details: &[PrefillBoundDetail]) -> PrefillBoundDetail {
    let n = details.len() as f64;
    // Prompt length triple must stay coherent: never mean-of-means + max-p99
    // across windows (prints e.g. "19900 (11x mean)" next to mean 9337).
    // Prefer the window with the highest finite p99/mean from that window's
    // lengths. If no window has both ends, keep lengths from one window and
    // leave ratio None (no invented ×).
    let (prompt_tokens_mean, prompt_tokens_p99, prompt_skew_ratio) = {
        let skew_winner = details
            .iter()
            .filter_map(|d| {
                let ratio = skew_ratio_from_lengths(d.prompt_tokens_mean, d.prompt_tokens_p99)?;
                Some((ratio, d))
            })
            .max_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((ratio, d)) = skew_winner {
            (d.prompt_tokens_mean, d.prompt_tokens_p99, Some(ratio))
        } else {
            // Newest window that carries any length field; ratio only if both ends.
            let solo = details
                .iter()
                .rev()
                .find(|d| d.prompt_tokens_mean.is_some() || d.prompt_tokens_p99.is_some());
            match solo {
                Some(d) => (
                    d.prompt_tokens_mean,
                    d.prompt_tokens_p99,
                    skew_ratio_from_lengths(d.prompt_tokens_mean, d.prompt_tokens_p99),
                ),
                None => (None, None, None),
            }
        }
    };
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
        // Config fields: last window wins when config drifts mid-run.
        prefix_caching_enabled: details.last().and_then(|d| d.prefix_caching_enabled),
        chunked_prefill_enabled: details.last().and_then(|d| d.chunked_prefill_enabled),
        prompt_tokens_mean,
        prompt_tokens_p99,
        prompt_skew_ratio,
        running_count: super::mean_of_present(details.iter().filter_map(|d| d.running_count)),
        ridge_batch_size: details.first().and_then(|d| d.ridge_batch_size),
        max_num_batched_tokens: details.last().and_then(|d| d.max_num_batched_tokens),
        // Newest known floor: last non-None wins so an unread final window
        // keeps the latest earlier scrape (not last().and_then → None).
        chunk_floor: details.iter().rev().find_map(|d| d.chunk_floor),
    }
}

/// p99/mean when both ends are usable. Display and skew gate must use this (or a
/// stored ratio from the same window), never a max-ratio glued onto other windows' lengths.
fn skew_ratio_from_lengths(mean: Option<f64>, p99: Option<f64>) -> Option<f64> {
    match (mean, p99) {
        (Some(m), Some(p)) if m > 0.0 && m.is_finite() && p.is_finite() && p >= 0.0 => Some(p / m),
        _ => None,
    }
}

fn skewed_mode(d: &PrefillBoundDetail) -> bool {
    skew_ratio_from_lengths(d.prompt_tokens_mean, d.prompt_tokens_p99)
        .or(d.prompt_skew_ratio.filter(|r| r.is_finite()))
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

const CONFIRM_CHUNKED_BULLET: &str =
    "      • Confirm chunked prefill is enabled (--enable-chunked-prefill).";

const UNREAD_WITHIN_BAND_EXPECTED: &str = "No budget change; --max-num-batched-tokens already sits at the derived target. Re-measure after confirming.";

/// Confirmed-off only: Enable + Set. Enable is a knob → never terminal.
fn chunked_off_fix_lines(d: &PrefillBoundDetail) -> (Vec<String>, String, bool) {
    let mut bullets = Vec::new();
    bullets.push("      • Enable chunked prefill (--enable-chunked-prefill).".to_string());
    let (budget, _named_set) = batch_token_budget_bullets(d, "Set");
    bullets.extend(budget);
    (
        bullets,
        "Decode batches interleave with prefill, reducing head-of-line blocking.".to_string(),
        false,
    )
}

/// Unread chunked (mild/moderate): Confirm (never Enable), then budget. Within-band
/// → Confirm only, non-terminal, no FLOPs wall. Confirm keeps the loop open.
fn unread_chunked_fix_lines(d: &PrefillBoundDetail) -> (Vec<String>, String, bool) {
    if on_compute_wall(d) {
        return (
            vec![CONFIRM_CHUNKED_BULLET.to_string()],
            UNREAD_WITHIN_BAND_EXPECTED.to_string(),
            false,
        );
    }
    let (mut bullets, expected, terminal) = knob_fix_lines(d);
    bullets.insert(0, CONFIRM_CHUNKED_BULLET.to_string());
    (bullets, expected, terminal)
}

pub(super) fn prefill_fix_lines(
    d: &PrefillBoundDetail,
    sev: Severity,
) -> (Vec<String>, String, bool) {
    let prefix_off = d.prefix_caching_enabled == Some(false);
    let chunked_on = d.chunked_prefill_enabled == Some(true);
    let chunked_off = d.chunked_prefill_enabled == Some(false);

    // Prefix caching cuts FLOPs; keep ahead of severity / chunked knobs.
    // Wording matches R3 (`Enable prefix caching: --flag`). No invented % Expected.
    if prefix_off {
        let mut bullets = Vec::new();
        super::push_bullet_with_subline(
            &mut bullets,
            super::ENABLE_PREFIX_CACHING_BULLET.to_string(),
            Some("Repeated prompt prefixes are re-computed every request."),
        );
        (
            bullets,
            super::ENABLE_PREFIX_CACHING_EXPECTED.to_string(),
            false,
        )
    } else if sev == Severity::Severe {
        // Severity alone: do not wait for chunked scrape. Unread cannot trap Enable+Set.
        severe_flops_wall_fix_lines(d)
    } else if chunked_off {
        chunked_off_fix_lines(d)
    } else if chunked_on {
        if on_compute_wall(d) {
            compute_wall_fix_lines(d)
        } else {
            knob_fix_lines(d)
        }
    } else {
        // Unread: Confirm + budget rules. Never Enable. Never FLOPs wall.
        unread_chunked_fix_lines(d)
    }
}

#[cfg(test)]
pub(super) fn format_prefill_bound_window_issue(
    d: &PrefillBoundDetail,
    seen_pct: u32,
) -> Vec<String> {
    format_prefill_bound_window_issue_with_terminal(d, seen_pct).0
}

pub(super) fn format_prefill_bound_window_issue_with_terminal(
    d: &PrefillBoundDetail,
    seen_pct: u32,
) -> (Vec<String>, bool) {
    let sev = severity(d.prompt_gen_ratio);
    let conf = if d.tpot_unverified {
        confidence(sev).min(TPOT_UNVERIFIED_CONFIDENCE_CAP)
    } else {
        confidence(sev)
    };
    let (fix_bullets, expected_normal, terminal) = prefill_fix_lines(d, sev);
    let prompt_p99 = d.prompt_tokens_p99.filter(|v| v.is_finite() && *v > 0.0);
    let prompt_mean = d.prompt_tokens_mean.filter(|v| v.is_finite() && *v > 0.0);
    // Skew UI + routing only with both ends of the length distribution.
    // Ratio alone with missing p99/mean is normal path (title, cause, Fix).
    let skewed = skewed_mode(d) && prompt_p99.is_some() && prompt_mean.is_some();

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
        if let Some(pm) = prompt_mean {
            lines.push(r6_metric_line("Prompt mean", &format!("{pm:.0} tok")));
        }
        if let Some(p99) = prompt_p99 {
            // × only from the printed pair; never a stored ratio from elsewhere.
            match skew_ratio_from_lengths(prompt_mean, prompt_p99)
                .filter(|r| *r > 0.0 && r.is_finite())
            {
                Some(ratio) => lines.push(r6_metric_line(
                    "Prompt p99",
                    &format!("{p99:.0} tok  ({ratio:.0}x mean)"),
                )),
                None => lines.push(r6_metric_line("Prompt p99", &format!("{p99:.0} tok"))),
            }
        }
    } else if let Some(pm) = prompt_mean {
        lines.push(r6_metric_line("Avg prompt", &format!("{pm:.0} tok")));
    }

    lines.push(String::new());
    lines.push("    Cause:".to_string());
    if let (true, Some(p99), Some(pm)) = (skewed, prompt_p99, prompt_mean) {
        lines.push(format!(
            "      Outlier prompts (p99: {p99:.0} tok) are monopolizing prefill compute."
        ));
        lines.push(format!(
            "      Short requests ({pm:.0} tok mean) are blocked behind long-tail prefills."
        ));
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
            (None, _) => "(low confidence, TPOT unavailable).",
            (Some(_), None) => "(low confidence, TPOT floor unavailable).",
            (Some(_), Some(_)) => "(low confidence, TPOT check unavailable).",
        };
        lines.push(format!("      {note}"));
    }

    lines.push(String::new());
    let expected = match (skewed, prompt_p99, prompt_mean) {
        (true, Some(p99), Some(mean)) => {
            let mut safe = Vec::new();
            // Enable only on confirmed off. Unread scrape → Confirm (never Enable).
            match d.chunked_prefill_enabled {
                Some(false) => {
                    safe.push(
                        "      • Enable --enable-chunked-prefill to interleave short requests with long-prompt chunks."
                            .to_string(),
                    );
                }
                None => safe.push(CONFIRM_CHUNKED_BULLET.to_string()),
                Some(true) => {}
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
        }
        _ => {
            lines.push("    Fix:".to_string());
            lines.extend(fix_bullets);
            lines.push("      • Reduce prompt length where possible.".to_string());
            expected_normal
        }
    };
    lines.push(String::new());
    lines.push(format!("    Expected: {expected}"));
    lines.push(format!("    Confidence: {}", super::confidence_label(conf)));
    // Wall terminal applies only when the normal (non-routing) Fix branch rendered.
    let terminal = !skewed && terminal;
    (super::with_seen_pct(lines, seen_pct), terminal)
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
            chunk_floor: None,
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
    fn aggregate_keeps_skew_mean_p99_from_same_window() {
        // Mild window + extreme window. Old agg: mean≈(9337+1000)/2, max p99=19900,
        // max ratio≈19.9 → "19900 (20x mean)" while 19900/mean_avg ≉ 20.
        let mild = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 5.0,
            tpot_ms: Some(200.0),
            tpot_floor_ms: Some(10.0),
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(9337.0),
            prompt_tokens_p99: Some(12_000.0),
            prompt_skew_ratio: Some(12_000.0 / 9337.0),
            running_count: Some(100.0),
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            chunk_floor: None,
        };
        let extreme = PrefillBoundDetail {
            prompt_tokens_mean: Some(1000.0),
            prompt_tokens_p99: Some(19_900.0),
            prompt_skew_ratio: Some(19.9),
            ..mild.clone()
        };
        let agg = aggregate_r6_detail(&[mild, extreme]);
        assert_eq!(agg.prompt_tokens_mean, Some(1000.0));
        assert_eq!(agg.prompt_tokens_p99, Some(19_900.0));
        let ratio = agg.prompt_skew_ratio.expect("ratio");
        assert!((ratio - 19.9).abs() < 1e-9);
        assert!(skewed_mode(&agg));
        let text = format_prefill_bound_window_issue(&agg, 100).join("\n");
        assert!(
            text.contains("19900 tok  (20x mean)"),
            "printed × must match p99/mean: {text}"
        );
        assert!(text.contains("Prompt mean         1000 tok"));
    }

    #[test]
    fn aggregate_fallback_same_window_no_frankenstein_ratio() {
        // Split ends across windows: must not invent mean 9337 + p99 19900 + ×.
        let only_mean = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 5.0,
            tpot_ms: Some(200.0),
            tpot_floor_ms: Some(10.0),
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(9337.0),
            prompt_tokens_p99: None,
            prompt_skew_ratio: None,
            running_count: Some(100.0),
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            chunk_floor: None,
        };
        let only_p99 = PrefillBoundDetail {
            prompt_tokens_mean: None,
            prompt_tokens_p99: Some(19_900.0),
            prompt_skew_ratio: None,
            ..only_mean.clone()
        };
        let agg = aggregate_r6_detail(&[only_mean, only_p99]);
        assert_eq!(agg.prompt_tokens_mean, None);
        assert_eq!(agg.prompt_tokens_p99, Some(19_900.0));
        assert!(
            agg.prompt_skew_ratio.is_none(),
            "cross-window lengths must not invent a skew ratio"
        );
        assert!(!skewed_mode(&agg));
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
            chunk_floor: None,
        };
        let agg = aggregate_r6_detail(&[d.clone(), d]);
        assert!(agg.prompt_gen_ratio.is_infinite());
    }

    #[test]
    fn aggregate_config_fields_use_last_window() {
        let first = PrefillBoundDetail {
            prompt_gen_ratio: 6.0,
            decode_efficiency_pct: 10.0,
            tpot_ms: None,
            tpot_floor_ms: None,
            tpot_unverified: false,
            prefix_caching_enabled: Some(false),
            chunked_prefill_enabled: Some(false),
            prompt_tokens_mean: None,
            prompt_tokens_p99: None,
            prompt_skew_ratio: None,
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: Some(1024),
            chunk_floor: Some(16),
        };
        let last = PrefillBoundDetail {
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            max_num_batched_tokens: Some(2048),
            chunk_floor: Some(112),
            ..first.clone()
        };
        let agg = aggregate_r6_detail(&[first, last]);
        assert_eq!(agg.prefix_caching_enabled, Some(true));
        assert_eq!(agg.chunked_prefill_enabled, Some(true));
        assert_eq!(agg.max_num_batched_tokens, Some(2048));
        assert_eq!(agg.chunk_floor, Some(112));
    }

    #[test]
    fn aggregate_chunk_floor_falls_back_when_last_unread() {
        let known = PrefillBoundDetail {
            prompt_gen_ratio: 6.0,
            decode_efficiency_pct: 10.0,
            tpot_ms: None,
            tpot_floor_ms: None,
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: None,
            prompt_tokens_p99: None,
            prompt_skew_ratio: None,
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: Some(2048),
            chunk_floor: Some(16),
        };
        let unread_last = PrefillBoundDetail {
            chunk_floor: None,
            ..known.clone()
        };
        let agg = aggregate_r6_detail(&[known, unread_last]);
        assert_eq!(agg.chunk_floor, Some(16));
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
            chunk_floor: None,
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
    fn fix_recommends_routing_when_skewed_chunked_unknown_confirms() {
        // Unread chunked: Confirm (never Enable). Same humility as uniform unread.
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 15.0,
            decode_efficiency_pct: 6.7,
            tpot_ms: Some(130.0),
            tpot_floor_ms: Some(7.85),
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: None,
            prompt_tokens_mean: Some(2048.0),
            prompt_tokens_p99: Some(51_200.0),
            prompt_skew_ratio: Some(25.0),
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            chunk_floor: None,
        };
        let (lines, terminal) = format_prefill_bound_window_issue_with_terminal(&d, 100);
        let text = lines.join("\n");
        assert!(text.contains("Route long-context requests"));
        assert!(!text.contains("Enable --enable-chunked-prefill"));
        assert!(
            text.contains(CONFIRM_CHUNKED_BULLET.trim_start()),
            "unread chunked must Confirm:\n{text}"
        );
        let confirm = text.find("Confirm chunked prefill").expect("Confirm");
        let route = text.find("Route long-context requests").expect("route");
        assert!(confirm < route, "Confirm before Route:\n{text}");
        assert!(!terminal, "routing path is never terminal:\n{text}");
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
            chunk_floor: None,
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
            chunk_floor: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Enable prefix caching: --enable-prefix-caching"));
        assert!(text.contains("        Repeated prompt prefixes are re-computed every request."));
        assert!(text.contains(&format!(
            "Expected: {}",
            super::super::ENABLE_PREFIX_CACHING_EXPECTED
        )));
        assert!(!text.contains("20-40%"));
        assert!(!text.contains("automatic prefix"));
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
            max_num_batched_tokens: Some(1024),
            chunk_floor: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("--enable-chunked-prefill"));
        assert!(text.contains("Set --max-num-batched-tokens to 2048 (default)"));
        assert!(text.contains(BATCH_TOKEN_BOOT_REJECT_SUBLINE));
        assert!(!text.contains("Takes effect only with chunked prefill on."));
        assert!(!text.contains("could not verify this differs from what's running"));
        assert!(!text.contains("Disaggregate prefill and decode"));
    }

    #[test]
    fn chunked_unknown_confirm_then_unread_budget_guide_no_enable() {
        // Unread chunked + unread budget → Confirm + guide (never Enable, never blind Set).
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 10.0,
            decode_efficiency_pct: 8.0,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: None,
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: Some(6000.0),
            prompt_skew_ratio: Some(1.46),
            running_count: None,
            ridge_batch_size: None,
            max_num_batched_tokens: None,
            chunk_floor: None,
        };
        let (lines, terminal) = format_prefill_bound_window_issue_with_terminal(&d, 100);
        let text = lines.join("\n");
        assert!(
            text.contains(CONFIRM_CHUNKED_BULLET.trim_start()),
            "unread must Confirm: {text}"
        );
        assert!(
            !text.contains("Enable chunked prefill (--enable-chunked-prefill)."),
            "unread must not Enable: {text}"
        );
        assert!(text.contains(BATCH_TOKEN_UNREAD_BULLET.trim_start()));
        assert!(!text.contains("Set --max-num-batched-tokens to 2048"));
        let confirm = text.find("Confirm chunked prefill").expect("Confirm");
        let unread = text
            .find("--max-num-batched-tokens unread")
            .expect("unread");
        assert!(confirm < unread, "Confirm leads budget guide: {text}");
        assert!(!terminal, "Confirm keeps loop open: {text}");
        assert!(!text.contains("no single-GPU knob adds FLOPs"));
        assert!(!text.contains("Takes effect only with chunked prefill on."));
        assert!(!text.contains("could not verify this differs from what's running"));
    }

    #[test]
    fn chunked_on_budget_bullet_has_no_enable() {
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
            max_num_batched_tokens: Some(1024),
            chunk_floor: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(!text.contains("Enable chunked prefill (--enable-chunked-prefill)."));
        assert!(text.contains("Set --max-num-batched-tokens to 2048 (default)"));
        assert!(!text.contains("Takes effect only with chunked prefill on."));
        assert!(!text.contains("could not verify this differs from what's running"));
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
            max_num_batched_tokens: Some(1024),
            chunk_floor: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Set --max-num-batched-tokens to 2048 (default)"));
        assert!(text.contains(BATCH_TOKEN_DIRECTIONS));
        assert!(text.contains(BATCH_TOKEN_BOOT_REJECT_SUBLINE));
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
            chunk_floor: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Prefill FLOPs dominate this mix"));
        assert!(text.contains("Disaggregate prefill and decode"));
        assert!(text.contains(SEVERE_FLOPS_WALL_EXPECTED));
        assert!(!text.contains("Set --max-num-batched-tokens"));
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
            chunk_floor: None,
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
            chunk_floor: None,
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
            chunk_floor: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(!text.contains("Reduce prompt length where possible"));
        assert!(text.contains("Route long-context requests"));
    }

    #[test]
    fn skewed_mode_without_p99_falls_back_to_normal_fix() {
        // Skew signal without p99: full normal path (title/cause/Fix), keep wall terminal.
        let mut d = wall_path_base(Some(2048), Some(1638.4), Some(true), 12.0);
        d.prompt_skew_ratio = Some(25.0);
        d.prompt_tokens_p99 = None;
        d.prompt_tokens_mean = Some(2048.0);
        assert!(skewed_mode(&d));
        assert!(on_compute_wall(&d));
        let (lines, terminal) = format_prefill_bound_window_issue_with_terminal(&d, 100);
        let text = lines.join("\n");
        assert!(
            text.contains("High Prompt Processing Time"),
            "incomplete skew must use normal title: {text}"
        );
        assert!(!text.contains("Skewed Prompt Distribution"));
        assert!(text.contains(
            "Prompt input rate is 12.0x generation output rate, starving decode throughput."
        ));
        assert!(!text.contains("Prompt length outliers are monopolizing"));
        assert!(!text.contains("Route long-context requests"));
        assert!(!text.contains("    Rejects requests:"));
        assert!(text.contains("    Fix:"));
        assert!(text.contains("no single-GPU knob adds FLOPs"));
        assert!(text.contains("Reduce prompt length where possible"));
        assert!(text.contains("Avg prompt"));
        assert!(terminal);
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
            chunk_floor: None,
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
            chunk_floor: None,
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
    fn fix_uses_default_budget_when_running_count_available() {
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
            max_num_batched_tokens: Some(512),
            chunk_floor: None,
        };
        // Workload tier still feeds on_compute_wall; Set line is always default.
        assert_eq!(batch_token_budget(&d).0, 896);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Set --max-num-batched-tokens to 2048 (default)"));
        assert!(!text.contains("to 896"));
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
    fn ridge_budget_still_drives_wall_not_set_target() {
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
            max_num_batched_tokens: Some(1024),
            chunk_floor: None,
        };
        assert_eq!(batch_token_budget(&d).0, 256);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Set --max-num-batched-tokens to 2048 (default)"));
        assert!(text.contains(BATCH_TOKEN_DIRECTIONS));
        assert!(
            text.contains(BATCH_TOKEN_BOOT_REJECT_SUBLINE),
            "named Set must carry boot-reject subline: {text}"
        );
        assert!(!text.contains("to 256"));
        assert!(!text.contains("(est)"));
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
            chunk_floor: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains(
            "Decode starves while each step swallows prompt: tpot 80ms vs 7.8ms floor (10x)."
        ));
        assert!(!text.contains("GPU is busy"));
    }

    #[test]
    fn ridge_budget_hybrid_sets_default_with_floor() {
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
            max_num_batched_tokens: Some(1024),
            chunk_floor: Some(128),
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains(
            "Set --max-num-batched-tokens to 2048 (default); page floor is 128 (do not go below)."
        ));
        assert!(text.contains(BATCH_TOKEN_DIRECTIONS));
        assert!(!text.contains("optimistic on long prompts"));
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
            max_num_batched_tokens: Some(1024),
            chunk_floor: None,
        };
        assert_eq!(batch_token_budget(&d).0, DEFAULT_BATCH_TOKEN_BUDGET);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Set --max-num-batched-tokens to 2048 (default)"));
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
            chunk_floor: None,
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
        assert!(text.contains("Set --max-num-batched-tokens to 2048 (default)"));
        assert!(!text.contains("no single-GPU knob adds FLOPs"));
    }

    #[test]
    fn knob_when_configured_above_default_never_sets_down() {
        // Launch policy: never shrink to 2048 when configured is already higher.
        let d = wall_path_base(Some(4096), Some(1638.4), Some(true), 12.0);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(!text.contains("Set --max-num-batched-tokens to 2048"));
        assert!(text.contains(BATCH_TOKEN_ALREADY_ABOVE_DEFAULT_EXPECTED));
        assert!(!text.contains("no single-GPU knob adds FLOPs"));
        assert!(text.contains("Reduce prompt length where possible"));
    }

    #[test]
    fn knob_when_configured_unknown() {
        let d = wall_path_base(None, Some(1638.4), Some(true), 12.0);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains(BATCH_TOKEN_UNREAD_BULLET.trim_start()));
        assert!(!text.contains("Set --max-num-batched-tokens to 2048"));
        assert!(!text.contains("could not verify this differs from what's running"));
        assert!(!text.contains("no single-GPU knob adds FLOPs"));
    }

    #[test]
    fn already_set_skips_budget_bullet_on_exact_match_including_default_tier() {
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
            chunk_floor: None,
        };
        assert_eq!(batch_token_budget(&d), (2048, BatchBudgetTier::Default));
        assert!(super::super::already_set_u32(
            d.max_num_batched_tokens,
            2048
        ));
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(
            !text.contains("Set --max-num-batched-tokens to 2048 (default)"),
            "equality is a no-op: {text}"
        );
        assert!(!text.contains("no single-GPU knob adds FLOPs"));
    }

    #[test]
    fn already_set_skips_named_target_on_ridge_exact_match_via_wall() {
        // Exact match of a derived (non-default) target is the compute wall, not a Set.
        let d = wall_path_base(Some(2048), Some(1638.4), Some(true), 12.0);
        assert_eq!(batch_token_budget(&d), (2048, BatchBudgetTier::Ridge));
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(!text.contains("Set --max-num-batched-tokens"));
        assert!(text.contains("no single-GPU knob adds FLOPs"));
    }

    #[test]
    fn wall_path_prefix_unknown_does_not_enable() {
        // Unknown is not off: verify covers prefix; no Enable prescription.
        let d = wall_path_base(Some(2048), Some(1638.4), None, 12.0);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains(R6_TERMINAL_VERIFY.trim_start()));
        assert!(!text.contains("Enable prefix caching: --enable-prefix-caching"));
        let d_on = wall_path_base(Some(2048), Some(1638.4), Some(true), 12.0);
        let text_on = format_prefill_bound_window_issue(&d_on, 100).join("\n");
        assert!(!text_on.contains("Enable prefix caching: --enable-prefix-caching"));
    }

    #[test]
    fn unread_configured_guides_without_blind_set() {
        let d = wall_path_base(None, Some(1638.4), Some(true), 12.0);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains(BATCH_TOKEN_UNREAD_BULLET.trim_start()));
        assert!(!text.contains("Set --max-num-batched-tokens to 2048"));
        assert!(text.contains(BATCH_TOKEN_DIRECTIONS));
        assert!(text.contains(BATCH_TOKEN_BOOT_REJECT_SUBLINE));
        assert!(text.contains(BATCH_TOKEN_UNREAD_EXPECTED));
        assert!(!text.contains("could not verify this differs from what's running"));
    }

    #[test]
    fn already_set_expected_does_not_promise_budget_change() {
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
            chunk_floor: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(!text.contains("Set --max-num-batched-tokens to 2048 (default)"));
        assert!(text.contains(
            "No --max-num-batched-tokens change; configured value already matches the target."
        ));
        assert!(!text.contains("Lower TTFT variance"));
    }

    #[test]
    fn severe_flops_wall_when_chunked_on() {
        // Severity alone names the FLOPs wall; no Enable+Set theater.
        let d = wall_path_base(Some(2048), Some(1638.4), Some(true), 22.0);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Prefill FLOPs dominate this mix"));
        assert!(text.contains("Disaggregate prefill and decode onto separate workers"));
        assert!(text.contains("Add a replica to scale out."));
        assert!(text.contains(SEVERE_FLOPS_WALL_EXPECTED));
        assert!(text.contains("Reduce prompt length where possible"));
        assert!(!text.contains("Enable chunked prefill"));
        assert!(!text.contains("Set --max-num-batched-tokens"));
        assert!(!text.contains("Decode batches interleave with prefill"));
    }

    #[test]
    fn severe_unread_chunked_still_names_flops_wall() {
        // Unread must not trap Enable+Set at severe.
        let mut d = wall_path_base(Some(2048), Some(1638.4), Some(true), 22.0);
        d.chunked_prefill_enabled = None;
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains("Prefill FLOPs dominate this mix"));
        assert!(text.contains(SEVERE_FLOPS_WALL_EXPECTED));
        assert!(!text.contains("Enable chunked prefill"));
        assert!(!text.contains("Set --max-num-batched-tokens"));
    }

    #[test]
    fn severe_chunked_off_trails_enable_not_set_budget() {
        let mut d = wall_path_base(Some(2048), Some(1638.4), Some(true), 22.0);
        d.chunked_prefill_enabled = Some(false);
        let (lines, terminal) = format_prefill_bound_window_issue_with_terminal(&d, 100);
        let text = lines.join("\n");
        assert!(text.contains("Prefill FLOPs dominate this mix"));
        let wall = text.find("Prefill FLOPs dominate").expect("wall");
        let enable = text
            .find("Enable chunked prefill")
            .expect("trailing Enable");
        assert!(wall < enable, "Enable trails the wall: {text}");
        assert!(!text.contains("Set --max-num-batched-tokens"));
        assert!(!terminal, "trailing Enable keeps a config action open");
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

    /// Class-example hybrid page floor (14 × 112); not a production constant.
    const HYBRID_ALIGN_FLOOR_EXAMPLE: u32 = 14 * 112;

    fn hybrid_align_floor_detail(max_num_batched_tokens: Option<u32>) -> PrefillBoundDetail {
        PrefillBoundDetail {
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
            running_count: Some(90.0),
            ridge_batch_size: Some(295.2),
            max_num_batched_tokens,
            chunk_floor: Some(HYBRID_ALIGN_FLOOR_EXAMPLE),
        }
    }

    fn prescription_numbers(text: &str) -> Vec<u64> {
        let mut out = Vec::new();
        for line in text.lines() {
            if !line.contains("--max-num-batched-tokens") {
                continue;
            }
            // Named Set target only (`to N`), not "Default … is N" / "page floor is N".
            if let Some(rest) = line.split("to ").nth(1)
                && let Some(tok) = rest.split_whitespace().next()
                && let Ok(n) = tok.parse::<u64>()
            {
                out.push(n);
            }
        }
        out
    }

    /// Every launch prescription shape: chunked on/off/unread × floor none/below/above
    /// default × configured unread / below default / above default.
    #[test]
    fn default_budget_prescription_matrix_all_paths() {
        let floors: [(Option<u32>, &str); 3] = [
            (None, "no_floor"),
            (Some(784), "floor_below_default"),
            (Some(2496), "floor_above_default"),
        ];
        let chunked: [(Option<bool>, &str); 3] = [
            (Some(true), "chunked_on"),
            (Some(false), "chunked_off"),
            (None, "chunked_unread"),
        ];
        let configureds: [(Option<u32>, &str); 3] = [
            (None, "cfg_unread"),
            (Some(1024), "cfg_below_default"),
            (Some(8192), "cfg_above_default"),
        ];
        for (floor, floor_label) in floors {
            for (chunked_prefill_enabled, chunk_label) in chunked {
                for (configured, cfg_label) in configureds {
                    let d = PrefillBoundDetail {
                        prompt_gen_ratio: 12.0,
                        decode_efficiency_pct: 8.0,
                        tpot_ms: Some(80.0),
                        tpot_floor_ms: Some(7.85),
                        tpot_unverified: false,
                        prefix_caching_enabled: Some(true),
                        chunked_prefill_enabled,
                        prompt_tokens_mean: Some(2348.0),
                        prompt_tokens_p99: Some(4000.0),
                        prompt_skew_ratio: Some(1.7),
                        running_count: Some(31.0),
                        ridge_batch_size: Some(246.7),
                        max_num_batched_tokens: configured,
                        chunk_floor: floor,
                    };
                    let text = format_prefill_bound_window_issue(&d, 100).join("\n");
                    let case = format!("{chunk_label}/{floor_label}/{cfg_label}");
                    assert!(
                        !text.contains("floor-limited"),
                        "{case}: retired floor-limited wording: {text}"
                    );
                    assert!(
                        !text.contains("(est)"),
                        "{case}: retired est Set wording: {text}"
                    );
                    assert!(
                        !text.contains("shrink prefill chunk size"),
                        "{case}: retired shrink wording: {text}"
                    );

                    let above_default = configured.is_some_and(|c| u64::from(c) > 2048);
                    if above_default {
                        assert!(
                            !text.contains("Set --max-num-batched-tokens to 2048"),
                            "{case}: must not shrink to default: {text}"
                        );
                        assert!(
                            !text.contains("Default --max-num-batched-tokens is 2048"),
                            "{case}: no default/floor info when already above: {text}"
                        );
                        assert!(
                            !text.contains(BATCH_TOKEN_UNREAD_BULLET.trim_start()),
                            "{case}"
                        );
                        assert!(prescription_numbers(&text).is_empty(), "{case}: {text}");
                        // Enable-chunked path owns Expected; knob path uses already-above.
                        if chunked_prefill_enabled != Some(false) {
                            assert!(
                                text.contains(BATCH_TOKEN_ALREADY_ABOVE_DEFAULT_EXPECTED),
                                "{case}: {text}"
                            );
                        }
                    } else if configured.is_none() {
                        assert!(
                            text.contains(BATCH_TOKEN_UNREAD_BULLET.trim_start()),
                            "{case}: {text}"
                        );
                        assert!(
                            !text.contains("Set --max-num-batched-tokens to 2048"),
                            "{case}: unread must not blind-Set: {text}"
                        );
                        assert!(
                            !text.contains("Default --max-num-batched-tokens is 2048"),
                            "{case}: unread must not claim default Set tier: {text}"
                        );
                        assert!(prescription_numbers(&text).is_empty(), "{case}: {text}");
                        assert!(text.contains(BATCH_TOKEN_DIRECTIONS), "{case}");
                        assert!(text.contains(BATCH_TOKEN_BOOT_REJECT_SUBLINE), "{case}");
                        match floor {
                            Some(2496) => assert!(
                                text.contains("Page floor is 2496 (do not go below)."),
                                "{case}: {text}"
                            ),
                            Some(784) => assert!(
                                text.contains("Page floor is 784 (do not go below)."),
                                "{case}: {text}"
                            ),
                            None => assert!(
                                !text.contains("Page floor is"),
                                "{case}: no floor → no Page floor clause: {text}"
                            ),
                            Some(_) => unreachable!(),
                        }
                        if chunked_prefill_enabled != Some(false) {
                            assert!(text.contains(BATCH_TOKEN_UNREAD_EXPECTED), "{case}: {text}");
                        }
                    } else {
                        match floor {
                            Some(2496) => {
                                assert!(
                                    text.contains(
                                        "Default --max-num-batched-tokens is 2048; page floor is 2496 (do not go below)."
                                    ),
                                    "{case}: {text}"
                                );
                                assert!(
                                    !text.contains("Set --max-num-batched-tokens to 2048"),
                                    "{case}: must not Set below floor: {text}"
                                );
                                assert!(prescription_numbers(&text).is_empty(), "{case}: {text}");
                                assert!(text.contains(BATCH_TOKEN_DIRECTIONS), "{case}");
                                assert!(text.contains(BATCH_TOKEN_BOOT_REJECT_SUBLINE), "{case}");
                            }
                            Some(784) => {
                                assert!(
                                    text.contains(
                                        "Set --max-num-batched-tokens to 2048 (default); page floor is 784 (do not go below)."
                                    ),
                                    "{case}: {text}"
                                );
                                assert_eq!(prescription_numbers(&text), vec![2048], "{case}");
                                assert!(text.contains(BATCH_TOKEN_DIRECTIONS), "{case}");
                                assert!(text.contains(BATCH_TOKEN_BOOT_REJECT_SUBLINE), "{case}");
                            }
                            None => {
                                assert!(
                                    text.contains(
                                        "Set --max-num-batched-tokens to 2048 (default)."
                                    ),
                                    "{case}: {text}"
                                );
                                assert!(
                                    !text.contains("page floor"),
                                    "{case}: no floor → no page-floor clause: {text}"
                                );
                                assert_eq!(prescription_numbers(&text), vec![2048], "{case}");
                                assert!(text.contains(BATCH_TOKEN_DIRECTIONS), "{case}");
                                assert!(text.contains(BATCH_TOKEN_BOOT_REJECT_SUBLINE), "{case}");
                            }
                            Some(_) => unreachable!(),
                        }
                    }

                    match chunked_prefill_enabled {
                        Some(true) => {
                            assert!(!text.contains("Enable chunked prefill"), "{case}");
                            assert!(!text.contains("Confirm chunked prefill"), "{case}");
                        }
                        Some(false) => {
                            assert!(
                                text.contains("Enable chunked prefill (--enable-chunked-prefill)."),
                                "{case}"
                            );
                            assert!(!text.contains("Confirm chunked prefill"), "{case}");
                        }
                        None => {
                            assert!(text.contains(CONFIRM_CHUNKED_BULLET.trim_start()), "{case}");
                            assert!(
                                !text
                                    .contains("Enable chunked prefill (--enable-chunked-prefill)."),
                                "{case}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn default_set_with_floor_when_floor_below_default() {
        // Readable below default → named Set to default above page floor.
        let d = hybrid_align_floor_detail(Some(1024));
        assert_eq!(batch_token_budget(&d).0, 384);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains(&format!(
            "Set --max-num-batched-tokens to 2048 (default); page floor is {HYBRID_ALIGN_FLOOR_EXAMPLE} (do not go below)."
        )));
        assert!(text.contains(BATCH_TOKEN_DIRECTIONS));
        assert!(
            text.contains(BATCH_TOKEN_BOOT_REJECT_SUBLINE),
            "Set must carry boot-reject subline: {text}"
        );
        assert!(!text.contains("floor-limited"));
        assert!(!text.contains("to 384"));
        assert!(!text.contains(&format!("to {HYBRID_ALIGN_FLOOR_EXAMPLE}")));
    }

    #[test]
    fn configured_above_default_skips_set_even_with_floor() {
        let d = hybrid_align_floor_detail(Some(8192));
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(!text.contains("Set --max-num-batched-tokens to 2048"));
        assert!(!text.contains("floor-limited"));
        assert!(text.contains(BATCH_TOKEN_ALREADY_ABOVE_DEFAULT_EXPECTED));
        assert!(!text.contains("Confirm chunked prefill"));
        assert!(text.contains("Reduce prompt length where possible"));
    }

    #[test]
    fn default_set_when_current_at_page_floor() {
        // Was TerminalAtFloor; launch path still offers default above floor.
        let d = hybrid_align_floor_detail(Some(HYBRID_ALIGN_FLOOR_EXAMPLE));
        let (lines, terminal) = format_prefill_bound_window_issue_with_terminal(&d, 100);
        let text = lines.join("\n");
        assert!(!terminal, "default Set is a knob: {text}");
        assert!(text.contains(&format!(
            "Set --max-num-batched-tokens to 2048 (default); page floor is {HYBRID_ALIGN_FLOOR_EXAMPLE} (do not go below)."
        )));
        assert!(text.contains(BATCH_TOKEN_BOOT_REJECT_SUBLINE));
        assert!(!text.contains("no smaller value boots"));
        assert!(!text.contains(R6_TERMINAL_VERIFY.trim_start()));
    }

    #[test]
    fn chunked_off_with_floor_enable_then_default_set() {
        let mut d = hybrid_align_floor_detail(Some(HYBRID_ALIGN_FLOOR_EXAMPLE));
        d.chunked_prefill_enabled = Some(false);
        let (lines, terminal) = format_prefill_bound_window_issue_with_terminal(&d, 100);
        let text = lines.join("\n");
        assert!(!terminal, "Enable is a knob: {text}");
        assert!(text.contains("Enable chunked prefill (--enable-chunked-prefill)."));
        assert!(text.contains(&format!(
            "Set --max-num-batched-tokens to 2048 (default); page floor is {HYBRID_ALIGN_FLOOR_EXAMPLE} (do not go below)."
        )));
        assert!(
            text.contains(
                "Decode batches interleave with prefill, reducing head-of-line blocking."
            )
        );
        assert!(
            !text.contains("Takes effect only with chunked prefill on."),
            "no dependency subline: {text}"
        );
    }

    #[test]
    fn chunked_unknown_with_floor_confirm_then_default_set_non_terminal() {
        let mut d = hybrid_align_floor_detail(Some(HYBRID_ALIGN_FLOOR_EXAMPLE));
        d.chunked_prefill_enabled = None;
        let (lines, terminal) = format_prefill_bound_window_issue_with_terminal(&d, 100);
        let text = lines.join("\n");
        assert!(!terminal, "Confirm + default Set keeps loop open: {text}");
        assert!(text.contains(CONFIRM_CHUNKED_BULLET.trim_start()));
        assert!(!text.contains("Enable chunked prefill (--enable-chunked-prefill)."));
        assert!(text.contains(&format!(
            "Set --max-num-batched-tokens to 2048 (default); page floor is {HYBRID_ALIGN_FLOOR_EXAMPLE} (do not go below)."
        )));
        let confirm = text.find("Confirm chunked prefill").expect("Confirm");
        let set = text.find("Set --max-num-batched-tokens").expect("Set");
        assert!(confirm < set, "order Confirm→Set: {text}");
        assert!(!text.contains("no single-GPU knob adds FLOPs"));
        assert!(!text.contains(R6_TERMINAL_VERIFY.trim_start()));
        assert!(text.contains(BATCH_TOKEN_SET_EXPECTED));
    }

    #[test]
    fn chunked_unknown_within_band_confirm_only_non_terminal() {
        // Within-band unread: Confirm only. No wall, no verify, non-terminal.
        let mut d = wall_path_base(Some(2048), Some(1638.4), Some(true), 12.0);
        d.chunked_prefill_enabled = None;
        assert!(on_compute_wall(&d));
        let (lines, terminal) = format_prefill_bound_window_issue_with_terminal(&d, 100);
        let text = lines.join("\n");
        assert!(!terminal, "Confirm keeps loop open: {text}");
        assert!(text.contains(CONFIRM_CHUNKED_BULLET.trim_start()));
        assert!(!text.contains("Enable chunked prefill"));
        assert!(!text.contains("no single-GPU knob adds FLOPs"));
        assert!(!text.contains("Disaggregate"));
        assert!(!text.contains("Verify prefix caching"));
        assert!(!text.contains("Set --max-num-batched-tokens"));
        assert!(text.contains(UNREAD_WITHIN_BAND_EXPECTED));
    }

    #[test]
    fn unread_with_floor_guides_without_blind_set() {
        let d = hybrid_align_floor_detail(None);
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains(BATCH_TOKEN_UNREAD_BULLET.trim_start()));
        assert!(text.contains(&format!(
            "Page floor is {HYBRID_ALIGN_FLOOR_EXAMPLE} (do not go below). {BATCH_TOKEN_DIRECTIONS}"
        )));
        assert!(text.contains(BATCH_TOKEN_BOOT_REJECT_SUBLINE));
        assert!(!text.contains("Set --max-num-batched-tokens to 2048"));
        assert!(!text.contains("to 384"));
        assert!(!text.contains("floor-limited"));
    }

    #[test]
    fn dense_no_floor_guides_when_unread() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 8.0,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: None,
            prompt_skew_ratio: None,
            running_count: None,
            ridge_batch_size: Some(1638.4),
            max_num_batched_tokens: None,
            chunk_floor: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains(BATCH_TOKEN_UNREAD_BULLET.trim_start()));
        assert!(!text.contains("Set --max-num-batched-tokens to 2048"));
        assert!(!text.contains("(est)"));
        assert!(!text.contains("Page floor is"));
        assert!(text.contains(BATCH_TOKEN_DIRECTIONS));
    }

    #[test]
    fn dense_no_floor_skips_set_when_configured_above_default() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 8.0,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: None,
            prompt_skew_ratio: None,
            running_count: None,
            ridge_batch_size: Some(1638.4),
            max_num_batched_tokens: Some(8192),
            chunk_floor: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(!text.contains("Set --max-num-batched-tokens to 2048"));
        assert!(text.contains(BATCH_TOKEN_ALREADY_ABOVE_DEFAULT_EXPECTED));
    }

    #[test]
    fn hybrid_no_floor_guides_when_unread() {
        let d = PrefillBoundDetail {
            prompt_gen_ratio: 12.0,
            decode_efficiency_pct: 8.0,
            tpot_ms: Some(80.0),
            tpot_floor_ms: Some(7.85),
            tpot_unverified: false,
            prefix_caching_enabled: Some(true),
            chunked_prefill_enabled: Some(true),
            prompt_tokens_mean: Some(4096.0),
            prompt_tokens_p99: None,
            prompt_skew_ratio: None,
            running_count: None,
            ridge_batch_size: Some(295.2),
            max_num_batched_tokens: None,
            chunk_floor: None,
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains(BATCH_TOKEN_UNREAD_BULLET.trim_start()));
        assert!(!text.contains("Set --max-num-batched-tokens to 2048"));
        assert!(!text.contains("Confirm page size in vLLM boot logs"));
        assert!(prescription_numbers(&text).is_empty());
    }

    #[test]
    fn floor_above_default_names_both_when_configured_not_above_default() {
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
            max_num_batched_tokens: Some(1024),
            chunk_floor: Some(2496),
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains(
            "Default --max-num-batched-tokens is 2048; page floor is 2496 (do not go below)."
        ));
        assert!(text.contains(BATCH_TOKEN_DIRECTIONS));
        assert!(text.contains(BATCH_TOKEN_BOOT_REJECT_SUBLINE));
        assert!(text.contains(BATCH_TOKEN_SET_EXPECTED));
        assert!(!text.contains("Set --max-num-batched-tokens to 2048 (default)"));
        assert!(!text.contains("Set --max-num-batched-tokens to 2496"));
        assert!(prescription_numbers(&text).is_empty());
    }

    #[test]
    fn default_set_with_floor_when_ridge_absent() {
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
            max_num_batched_tokens: Some(1024),
            chunk_floor: Some(HYBRID_ALIGN_FLOOR_EXAMPLE),
        };
        let text = format_prefill_bound_window_issue(&d, 100).join("\n");
        assert!(text.contains(&format!(
            "Set --max-num-batched-tokens to 2048 (default); page floor is {HYBRID_ALIGN_FLOOR_EXAMPLE} (do not go below)."
        )));
        assert!(!text.contains("not below"));
        assert!(!text.contains("to 384"));
    }

    #[test]
    fn crash_fixture_never_prescribes_below_scraped_floor() {
        for current in [
            Some(8192),
            Some(HYBRID_ALIGN_FLOOR_EXAMPLE),
            None,
            Some(384),
        ] {
            let d = hybrid_align_floor_detail(current);
            let knob = format_prefill_bound_window_issue(&d, 100).join("\n");
            for n in prescription_numbers(&knob) {
                assert!(
                    n >= u64::from(d.chunk_floor.unwrap()),
                    "prescribed {n} below floor {:?} (current {current:?})",
                    d.chunk_floor
                );
            }
            let mut chunked_off = d.clone();
            chunked_off.chunked_prefill_enabled = Some(false);
            let off = format_prefill_bound_window_issue(&chunked_off, 100).join("\n");
            for n in prescription_numbers(&off) {
                assert!(n >= u64::from(d.chunk_floor.unwrap()));
            }
        }
    }

    #[test]
    fn prescription_numbers_respect_floor_across_fired_fixtures() {
        let fixtures = [
            hybrid_align_floor_detail(Some(8192)),
            hybrid_align_floor_detail(Some(HYBRID_ALIGN_FLOOR_EXAMPLE)),
            hybrid_align_floor_detail(None),
            wall_path_base(Some(2048), Some(1638.4), Some(true), 12.0),
            wall_path_base(Some(1024), Some(1638.4), Some(true), 12.0),
        ];
        for d in fixtures {
            let text = format_prefill_bound_window_issue(&d, 100).join("\n");
            if let Some(floor) = d.chunk_floor {
                for n in prescription_numbers(&text) {
                    assert!(
                        n >= u64::from(floor),
                        "fixture floor {floor}, printed {n}: {text}"
                    );
                }
            }
        }
    }

    #[test]
    fn chunk_floor_reader_uses_block_size_label_only() {
        let cache = CacheConfigLabels {
            block_size: Some(HYBRID_ALIGN_FLOOR_EXAMPLE),
            mamba_block_size: Some(16),
            ..Default::default()
        };
        assert_eq!(
            super::super::chunk_batched_tokens_floor(&cache),
            Some(HYBRID_ALIGN_FLOOR_EXAMPLE)
        );
    }

    #[test]
    fn terminal_flag_true_on_compute_wall_with_verify() {
        let d = wall_path_base(Some(2048), Some(1638.4), Some(true), 12.0);
        let (text, terminal) = format_prefill_bound_window_issue_with_terminal(&d, 100);
        let text = text.join("\n");
        assert!(terminal);
        assert!(text.contains(R6_TERMINAL_VERIFY.trim_start()));
        assert!(text.contains("no single-GPU knob adds FLOPs"));
    }

    #[test]
    fn terminal_flag_true_on_severe_flops_wall() {
        let d = wall_path_base(Some(8192), Some(1638.4), Some(true), 25.0);
        let (text, terminal) = format_prefill_bound_window_issue_with_terminal(&d, 100);
        let text = text.join("\n");
        assert!(terminal);
        assert!(text.contains("Prefill FLOPs dominate this mix"));
        assert!(text.contains("Disaggregate"));
        assert!(text.contains(SEVERE_FLOPS_WALL_EXPECTED));
        let (bullets, expected, term) = prefill_fix_lines(&d, Severity::Severe);
        assert!(term);
        assert_eq!(expected, SEVERE_FLOPS_WALL_EXPECTED);
        assert!(bullets.iter().any(|l| l.contains("Disaggregate")));
        assert!(bullets.iter().any(|l| l.contains("Add a replica")));
        assert!(!bullets.iter().any(|l| l.contains("Verify prefix caching")));
    }

    #[test]
    fn terminal_flag_false_on_enable_chunked_knob() {
        // Mild/moderate + chunked off: Enable path (budget may already match), not severe wall.
        let mut d = wall_path_base(Some(2048), Some(1638.4), Some(true), 12.0);
        d.chunked_prefill_enabled = Some(false);
        let (text, terminal) = format_prefill_bound_window_issue_with_terminal(&d, 100);
        let text = text.join("\n");
        assert!(!terminal);
        assert!(text.contains("Enable chunked prefill"));
        assert!(!text.contains("Prefill FLOPs dominate"));
        assert!(text.contains("Decode batches interleave with prefill"));
    }

    #[test]
    fn terminal_flag_false_on_budget_knob_path() {
        let d = wall_path_base(Some(1024), Some(1638.4), Some(true), 12.0);
        let (text, terminal) = format_prefill_bound_window_issue_with_terminal(&d, 100);
        assert!(!terminal);
        assert!(!text.join("\n").contains("Verify prefix caching"));
    }
}
