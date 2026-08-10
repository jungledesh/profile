mod eval;
mod format;
mod r1_under_batching;
mod r2_kv_cache_pressure;
mod r3_low_prefix_reuse;
mod r4_oom_risk;
mod r5_concurrency_saturation;
mod r6_prefill_bound;
mod r7_config_headroom;

#[cfg(test)]
mod tests;

pub(crate) use eval::aggregate_prefix_hit_rate_for_windows;
pub use eval::build_report_for_windows;
pub use format::{
    LoadHintParams, empty_run_diagnose_lines, format_captured_windows,
    format_diagnose_rules_for_windows,
};
pub(crate) use format::{MuVariant, mu_diagnose_lines};
#[cfg(test)]
pub(crate) use r1_under_batching::{R1EvalInput, Rule1Outcome};
pub(crate) use r2_kv_cache_pressure::KV_CACHE_PRESSURE_MIN_PERC;
pub(crate) use r6_prefill_bound::fix_shows_unread_batched_tokens;
/// KV cache usage below this means the pool has room to absorb new sequences safely.
/// Shared by R5 raise gate and R7 headroom cause line.
pub(super) const KV_CACHE_SAFE_TO_SCALE_PCT: f64 = 80.0;
/// Canonical Fix bullet when prefix caching is confirmed off (`Some(false)`).
/// Shared by R2 / R3 / R6 so Enable wording cannot drift.
pub(super) const ENABLE_PREFIX_CACHING_BULLET: &str =
    "      • Enable prefix caching: --enable-prefix-caching.";
/// Expected line after enabling prefix caching. Direction only; no invented %.
pub(super) const ENABLE_PREFIX_CACHING_EXPECTED: &str =
    "Higher prefix cache hit rate and lower TTFT.";
#[cfg(test)]
pub(crate) use r3_low_prefix_reuse::{LowPrefixReuseDetail, Rule3Outcome, r3_recommendation};
pub use r4_oom_risk::{r4_advisory, r4_recommendation};

/// Single number-to-word mapping for user-facing confidence. Every rule's
/// confidence VALUE stays rule-owned (calibration); the WORD is global.
/// Thresholds match the majority ladder (r1, r7): >= 0.8 High, >= 0.6 Medium, else Low.
/// NaN falls through to Low (comparisons are already false for NaN).
pub(crate) fn confidence_label(c: f64) -> &'static str {
    if c >= CONFIDENCE_HIGH_MIN {
        "High"
    } else if c >= CONFIDENCE_MEDIUM_MIN {
        "Medium"
    } else {
        "Low"
    }
}
pub(crate) const CONFIDENCE_HIGH_MIN: f64 = 0.8;
pub(crate) const CONFIDENCE_MEDIUM_MIN: f64 = 0.6;

/// True when avg or peak KV usage is at/above the R2 pressure bar.
/// Shared by R2's fire gate and MU's memory-wall veto so the threshold cannot drift.
pub(super) fn kv_near_full(snapshot: &crate::collectors::RawSnapshot) -> bool {
    let kv = snapshot.vllm.kv_cache_usage_perc.filter(|v| v.is_finite());
    let peak = snapshot.vllm.kv_cache_peak_perc.filter(|v| v.is_finite());
    kv.is_some_and(|k| k >= KV_CACHE_PRESSURE_MIN_PERC)
        || peak.is_some_and(|p| p >= KV_CACHE_PRESSURE_MIN_PERC)
}

/// Observed `kv_cache_max_concurrency` when it survives flooring, returned raw.
/// Callers floor for their own use (`effective_max_and_binder`,
/// `resolve_r2_kv_capacity`); returning raw keeps the fractional value for
/// labelling and loses nothing, since floor is idempotent.
fn kv_cap_positive_after_floor(snapshot: &crate::collectors::RawSnapshot) -> Option<f64> {
    let raw = snapshot
        .vllm
        .cache_config
        .kv_cache_max_concurrency
        .filter(|c| c.is_finite() && *c > 0.0)?;
    let cap = raw.floor();
    (cap > 0.0).then_some(raw)
}

/// vLLM's `kv_cache_max_concurrency` is a guarantee at full `max_model_len`, not a
/// prediction for observed traffic. When peak running exceeds it, the cap does
/// not describe this workload: decline it rather than treat it as a wall.
///
/// Seat and occupancy use only. Page-model geometry keeps reading the raw label:
/// backing out pages per sequence needs a full-context number precisely because
/// it is one, and peak running says nothing about block arithmetic.
///
/// Uses peak running, not mean: one burst above `floor(cap)` already falsifies the
/// full-context guarantee. Mean would hide that burst. (Cost turnover uses mean
/// running for a different job: covering steady concurrent seats with completions.)
///
/// Absent peak running is not evidence of contradiction: return the cap.
pub(super) fn usable_kv_concurrency(snapshot: &crate::collectors::RawSnapshot) -> Option<f64> {
    let raw = kv_cap_positive_after_floor(snapshot)?;
    let contradicted = snapshot
        .vllm
        .num_requests_running_peak
        .filter(|p| p.is_finite())
        .is_some_and(|peak| peak > raw.floor());
    (!contradicted).then_some(raw)
}

/// True when Observed full-context concurrency is present and peak running has
/// already exceeded `floor(cap)`. R2 still prints a seat bullet; the number is
/// withheld in favor of the direction-only form.
pub(super) fn observed_kv_cap_contradicted(snapshot: &crate::collectors::RawSnapshot) -> bool {
    kv_cap_positive_after_floor(snapshot).is_some() && usable_kv_concurrency(snapshot).is_none()
}

/// Tri-state for the Observed KV concurrency label (seat/headroom paths).
///
/// Peak running above `floor(label)` falsifies that full-context promise.
/// Derived catalog math makes the same class of claim; when the label is
/// [`Contradicted`](ObservedKvResolution::Contradicted), do not substitute
/// derived for R1 headroom (see [`kv_full_context_cap_for_r1`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ObservedKvResolution {
    Observed(u32),
    Absent,
    Contradicted,
}

fn floor_positive_concurrency_u32(c: f64) -> Option<u32> {
    let f = c.floor();
    (f > 0.0 && f <= f64::from(u32::MAX)).then_some({
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            f as u32
        }
    })
}

pub(super) fn resolve_observed_kv(
    snapshot: &crate::collectors::RawSnapshot,
) -> ObservedKvResolution {
    if observed_kv_cap_contradicted(snapshot) {
        return ObservedKvResolution::Contradicted;
    }
    match usable_kv_concurrency(snapshot).and_then(floor_positive_concurrency_u32) {
        Some(n) => ObservedKvResolution::Observed(n),
        None => ObservedKvResolution::Absent,
    }
}

/// Full-context KV cap for R1 Ridge/Config headroom formatting.
///
/// - Observed → use the label
/// - Absent → fall back to derived `kv_max_seqs`
/// - Contradicted → `None` (renders "seats idle; KV fit unknown")
pub(super) fn kv_full_context_cap_for_r1(
    snapshot: &crate::collectors::RawSnapshot,
    derived: Option<u32>,
) -> Option<u32> {
    match resolve_observed_kv(snapshot) {
        ObservedKvResolution::Observed(n) => Some(n),
        ObservedKvResolution::Absent => derived.filter(|&d| d > 0),
        ObservedKvResolution::Contradicted => None,
    }
}

/// Minimum bootable `--max-num-batched-tokens` from scraped page alignment.
/// Boot fact from `cache_config.block_size` only; never from catalog or constants.
pub(super) fn chunk_batched_tokens_floor(
    cache: &crate::collectors::CacheConfigLabels,
) -> Option<u32> {
    cache.block_size.filter(|b| *b > 0)
}

/// Minimum active windows for a trustworthy verdict. Window size scales with run
/// duration (2s for <= 30s runs, else 10s), so this enforces 6s to 30s of sustained
/// traffic. See profiler::logical_window_size.
pub const ENGINE_MIN_PERSISTENT_WINDOWS: usize = 3;
/// Enforces >= 25% density floor across evaluable windows.
pub(super) const ENGINE_MIN_WINDOW_PCT: f64 = 0.25;

/// Inputs for projecting capacity at a hypothetical `max_model_len`.
///
/// Preference order (see [`capacity_at_hypothetical_max_len`]):
/// 1. labels present, page model plausible → observed-geometry projection; a
///    fits-none projection returns no number, catalog math never overrules geometry
/// 2. labels present, gate tripped (state_pages >= transcript pages) →
///    no number, both tiers falsified. Catalog math shares the assumption
///    the gate disproved.
/// 3. labels absent or degenerate → attention-only catalog math
/// 4. otherwise → no number (caller stays directional)
///
/// Assumptions: block geometry constant across `max_model_len` (ladder-proven);
/// NOT proven constant across `gpu-memory-utilization` or vLLM versions.
/// `mamba_cache_mode` changes shift `state_pages` (measured 3→6 none→align).
/// Counterfactuals that change caching mode stay directional, no number.
pub(super) struct HypCapacityCtx<'a> {
    pub cache: &'a crate::collectors::CacheConfigLabels,
    pub kv_headroom_gb: Option<f64>,
    pub model: Option<&'a crate::context::ModelArch>,
    pub kv_cache_dtype: Option<&'a str>,
    pub tp: Option<u32>,
}

/// Capacity at a hypothetical `max_model_len`. Both derived tiers are `(est)`.
///
/// Formatter vocabulary: never bare "capacity"; bounds name their source and
/// condition. Observed counts read as "N concurrent requests"; a derived floor
/// reads as "at least N worst-case requests" (every request priced worst-case).
pub(super) fn capacity_at_hypothetical_max_len(
    target_max_len: u32,
    current_max_len: Option<u32>,
    ctx: &HypCapacityCtx<'_>,
) -> Option<u32> {
    use crate::engine::baseline::{
        attn_pages, counterfactual_concurrency, observed_state_pages, page_model_fits,
    };

    // Hybrid ladder geometry uses mamba_block_size when present; dense uses block_size.
    // Raw kv_cache_max_concurrency: page-model geometry, not a seat wall. Do not route
    // through usable_kv_concurrency; peak running says nothing about block arithmetic.
    let block_size = ctx.cache.mamba_block_size.or(ctx.cache.block_size);
    if let (Some(bs), Some(blocks), Some(obs), Some(cur)) = (
        block_size,
        ctx.cache.num_gpu_blocks,
        ctx.cache.kv_cache_max_concurrency,
        current_max_len,
    ) && let Some(state_pages) = observed_state_pages(bs, blocks, obs, cur)
    {
        let attn_pages_current = attn_pages(cur, bs)?;
        if !page_model_fits(state_pages, attn_pages_current) {
            return None;
        }
        if let Some(c) = counterfactual_concurrency(target_max_len, bs, blocks, obs, cur) {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let n = c.floor() as u32;
            return (n > 0).then_some(n);
        }
    }
    let model = ctx.model?;
    compute_kv_max_seqs_for_cache(
        ctx.kv_headroom_gb,
        Some(target_max_len),
        model,
        ctx.kv_cache_dtype,
        ctx.tp,
        ctx.cache,
    )
    .max_seqs
}

/// True when observed p99 prompt+output is below half of `max_model_len`.
/// Used to lead with model-len shrink: worst-case full-context concurrency is a
/// floor, not a target; leading with it on short-prompt workloads prescribes a
/// large throughput cut the traffic does not require. Fix order follows fit to
/// observed traffic.
pub(super) fn p99_sum_below_half_max_model_len(
    max_model_len: u32,
    prompt_p99: Option<f64>,
    generation_p99: Option<f64>,
) -> bool {
    match (prompt_p99, generation_p99) {
        (Some(pp), Some(gp)) if pp.is_finite() && gp.is_finite() => {
            pp + gp < f64::from(max_model_len) / 2.0
        }
        _ => false,
    }
}

/// Evidence for max_model_len shrink suggestions (p99 target path + sub-floor means).
#[derive(Debug, Clone, Copy)]
pub(super) struct ShrinkEvidence {
    pub prompt_p99: Option<f64>,
    pub generation_p99: Option<f64>,
    pub prompt_mean: Option<f64>,
    pub generation_mean: Option<f64>,
    pub total_count: f64,
}

impl ShrinkEvidence {
    pub(super) fn from_snapshot(snapshot: &crate::collectors::RawSnapshot) -> Self {
        // Every field feeds a printed max-model-len number. NaN and negatives
        // saturate to 0 on the u32 cast, which would prescribe a window that
        // rejects all traffic. Drop them at the boundary.
        let ok = |v: Option<f64>| v.filter(|x| x.is_finite() && *x >= 0.0);
        Self {
            prompt_p99: ok(snapshot.vllm.prompt_tokens_p99),
            generation_p99: ok(snapshot.vllm.generation_tokens_p99),
            prompt_mean: ok(snapshot.vllm.prompt_tokens_mean),
            generation_mean: ok(snapshot.vllm.generation_tokens_mean),
            total_count: ok(snapshot.vllm.generation_tokens_completed).unwrap_or(0.0),
        }
    }
}

/// Result of building shrink suggestion lines. `target` is set only when a
/// concrete max_model_len was prescribed (>= 100 completions, both p99s, >=5% cut).
/// `subline` is decided here and attached at render (rejection warning).
#[derive(Debug, Clone)]
pub(super) struct ShrinkSuggestion {
    pub lines: Vec<String>,
    pub target: Option<u32>,
    pub subline: Option<&'static str>,
}

/// Generic rejection caution when no p99/means evidence shapes the price.
pub(super) const SHRINK_REJECTION_WARNING: &str =
    "Requests above the new limit are rejected with a 400, not truncated.";

/// p99 target path: ~1% is definitional (not computed from the scrape).
pub(super) const SHRINK_P99_REJECTION_WARNING: &str =
    "~1% of observed requests ran longer; those are rejected with a 400, not truncated.";

/// Means path (both-sided and single-sided): avg cannot bound the tail.
pub(super) const SHRINK_MEANS_REJECTION_WARNING: &str = "Some requests are longer than avg; add buffer to it. Requests over the limit are rejected with a 400, not truncated.";

/// Build max_model_len shrink suggestion lines.
/// Hard number only when `total_count >= 100` and both p99s are present.
/// Empty only when a known `max_model_len` would shrink by < 5% (no-op).
/// When `max_model_len` is None, still prescribe lowering it and attach the
/// rejection subline so Fix is never an empty promise.
///
/// `current_shown`: when true, the block already names the current max_model_len
/// above this bullet, so emit `to {suggested}`; otherwise `{current} → {suggested}`.
pub(super) fn model_len_shrink_suggestion_lines(
    max_model_len: Option<u32>,
    evidence: &ShrinkEvidence,
    indent: &str,
    current_shown: bool,
) -> ShrinkSuggestion {
    let mut lines = Vec::new();
    let Some(m) = max_model_len else {
        lines.push(format!(
            "{indent}• Lower --max-model-len to safely raise concurrency."
        ));
        return ShrinkSuggestion {
            lines,
            target: None,
            subline: Some(SHRINK_REJECTION_WARNING),
        };
    };

    if evidence.total_count >= 100.0 {
        let Some(pp) = evidence.prompt_p99 else {
            lines.push(format!(
                "{indent}• Lower --max-model-len (current: {m}) to safely raise concurrency."
            ));
            return ShrinkSuggestion {
                lines,
                target: None,
                subline: Some(SHRINK_REJECTION_WARNING),
            };
        };
        let Some(gp) = evidence.generation_p99 else {
            lines.push(format!(
                "{indent}• Lower --max-model-len (current: {m}) to safely raise concurrency."
            ));
            return ShrinkSuggestion {
                lines,
                target: None,
                subline: Some(SHRINK_REJECTION_WARNING),
            };
        };
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let suggested = (pp as u32).saturating_add(gp as u32);
        // Suppress if reduction is < 5% - not a meaningful change (avoids "5464 → 5465" no-ops)
        if suggested >= m.saturating_sub(m / 20) {
            return ShrinkSuggestion {
                lines,
                target: None,
                subline: None,
            };
        }
        // Projection at the *suggested* length only, not current observed concurrency.
        let len_clause = if current_shown {
            format!("to {suggested}")
        } else {
            format!("{m} → {suggested}")
        };
        let p99_ctx = format_observed_context_tokens(pp + gp);
        lines.push(format!(
            "{indent}• Lower --max-model-len {len_clause}. \
             Observed p99 {p99_ctx} tokens per request."
        ));
        return ShrinkSuggestion {
            lines,
            target: Some(suggested),
            subline: Some(SHRINK_P99_REJECTION_WARNING),
        };
    }
    let (line, has_mean_evidence) =
        sub_floor_shrink_evidence_line(indent, m, evidence.prompt_mean, evidence.generation_mean);
    lines.push(line);
    ShrinkSuggestion {
        lines,
        target: None,
        subline: Some(if has_mean_evidence {
            SHRINK_MEANS_REJECTION_WARNING
        } else {
            SHRINK_REJECTION_WARNING
        }),
    }
}

pub(super) fn format_observed_context_tokens(n: f64) -> String {
    if n >= 1000.0 {
        format!("{:.1}k", n / 1000.0)
    } else {
        format!("{:.0}", n.round())
    }
}

/// Concurrent seats the KV pool can hold at observed mean request size.
///
/// Pool tokens ≈ full-context concurrency × max_model_len (the same full-window
/// pricing that produced `full_context_cap`). Divide by mean prompt+generation.
/// Always estimated: live-traffic means, not a measured allocator label.
pub(super) fn capacity_at_observed_request_sizes(
    full_context_cap: u32,
    max_model_len: u32,
    prompt_mean: Option<f64>,
    generation_mean: Option<f64>,
) -> Option<u32> {
    if full_context_cap == 0 || max_model_len == 0 {
        return None;
    }
    let p = prompt_mean.filter(|v| v.is_finite() && *v >= 0.0)?;
    let g = generation_mean.filter(|v| v.is_finite() && *v >= 0.0)?;
    let mean = p + g;
    if mean <= 0.0 {
        return None;
    }
    let pool_tokens = f64::from(full_context_cap) * f64::from(max_model_len);
    let n = (pool_tokens / mean).floor();
    if !(n.is_finite() && n > 0.0 && n <= f64::from(u32::MAX)) {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n_u = n as u32;
    Some(n_u)
}

/// Below the 100-completion floor: evidence only, no named target.
/// Returns `(line, has_mean_evidence)` so the caller can attach the means tip
/// only when an average was shown.
fn sub_floor_shrink_evidence_line(
    indent: &str,
    max_model_len: u32,
    prompt_mean: Option<f64>,
    gen_mean: Option<f64>,
) -> (String, bool) {
    let prompt = prompt_mean.filter(|v| v.is_finite() && *v >= 0.0);
    let generation = gen_mean.filter(|v| v.is_finite() && *v >= 0.0);
    match (prompt, generation) {
        (Some(p), Some(g)) => {
            let ctx = format_observed_context_tokens(p + g);
            (
                format!(
                    "{indent}• Lower --max-model-len (current: {max_model_len}). \
                 Observed avg {ctx} tokens per request, prompt + generation."
                ),
                true,
            )
        }
        (Some(p), None) => {
            let ctx = format_observed_context_tokens(p);
            (
                format!(
                    "{indent}• Lower --max-model-len (current: {max_model_len}). \
                 Observed avg prompt {ctx} tokens per request."
                ),
                true,
            )
        }
        (None, Some(g)) => {
            let ctx = format_observed_context_tokens(g);
            (
                format!(
                    "{indent}• Lower --max-model-len (current: {max_model_len}). \
                 Observed avg generation {ctx} tokens per request."
                ),
                true,
            )
        }
        (None, None) => (
            format!(
                "{indent}• Lower --max-model-len (current: {max_model_len}) to safely raise concurrency."
            ),
            false,
        ),
    }
}

/// Compare observed geometry `state_pages` to catalog hybrid estimate.
///
/// Runs only when labels (`num_gpu_blocks`, concurrency, page size, max_len)
/// AND catalog hybrid facts both exist. Returns `Some((catalog_pages,
/// observed_pages))` on mismatch; `None` on agreement or incomplete inputs.
///
/// Label uncertainty tracks the printed number's source, not the existence of
/// disagreement between sources. Callers must not change `(est)` labeling from
/// this result; verbose output may surface the note.
pub(super) fn catalog_state_pages_mismatch(
    cache: &crate::collectors::CacheConfigLabels,
    current_max_len: Option<u32>,
    model: &crate::context::ModelArch,
) -> Option<(u64, u64)> {
    use crate::engine::baseline::{
        catalog_hybrid_state_bytes, catalog_state_pages, observed_state_pages, state_dtype_bytes,
    };

    let block_size = cache.mamba_block_size.or(cache.block_size)?;
    let num_gpu_blocks = cache.num_gpu_blocks?;
    // Raw kv_cache_max_concurrency: geometry audit only. Not a seat/occupancy wall.
    let observed_concurrency = cache.kv_cache_max_concurrency?;
    let current_max_len = current_max_len?;
    let page_bytes = cache.mamba_page_size_padded?;

    let observed = observed_state_pages(
        block_size,
        num_gpu_blocks,
        observed_concurrency,
        current_max_len,
    )?;

    let dtype_b = state_dtype_bytes(model.state_dtype.as_deref())?;
    let state_bytes = catalog_hybrid_state_bytes(
        model.linear_num_layers?,
        model.linear_key_heads?,
        model.linear_value_heads?,
        model.linear_key_head_dim?,
        model.linear_value_head_dim?,
        model.linear_conv_kernel_dim?,
        dtype_b,
    )?;
    let catalog = catalog_state_pages(state_bytes, page_bytes)?;
    (catalog != observed).then_some((catalog, observed))
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct DerivedCapacity {
    pub max_seqs: Option<u32>,
}

pub(super) fn compute_kv_max_seqs_for_cache(
    kv_headroom_gb: Option<f64>,
    max_model_len: Option<u32>,
    model: &crate::context::ModelArch,
    kv_cache_dtype: Option<&str>,
    tp: Option<u32>,
    cache: &crate::collectors::CacheConfigLabels,
) -> DerivedCapacity {
    compute_kv_max_seqs_with_mode::<{ crate::engine::MULTI_GPU_TP }>(
        kv_headroom_gb,
        max_model_len,
        model,
        kv_cache_dtype,
        tp,
        Some(cache),
    )
}

fn compute_kv_max_seqs_with_mode<const MULTI_GPU: bool>(
    kv_headroom_gb: Option<f64>,
    max_model_len: Option<u32>,
    model: &crate::context::ModelArch,
    kv_cache_dtype: Option<&str>,
    tp: Option<u32>,
    cache: Option<&crate::collectors::CacheConfigLabels>,
) -> DerivedCapacity {
    use crate::engine::baseline::{bytes_per_seq, kv_bytes_per_element};

    let Some(max_len) = max_model_len.filter(|&v| v > 0) else {
        return DerivedCapacity::default();
    };
    let Some(model_view) = tp_priced_model::<MULTI_GPU>(model, tp.unwrap_or(1)) else {
        return DerivedCapacity::default();
    };

    let kv_bpp = kv_bytes_per_element(kv_cache_dtype);
    let Some(request_bytes) = bytes_per_seq(model_view.as_ref(), max_len, kv_bpp) else {
        return DerivedCapacity::default();
    };
    let max_seqs = kv_pool_budget_bytes(model_view.as_ref(), kv_cache_dtype, kv_headroom_gb, cache)
        .and_then(|bytes| bytes.checked_div(request_bytes))
        .and_then(|n| u32::try_from(n).ok())
        .filter(|&n| n > 0);

    DerivedCapacity { max_seqs }
}

fn derived_budget_bytes(kv_headroom_gb: Option<f64>) -> Option<u64> {
    let gb = kv_headroom_gb.filter(|v| v.is_finite() && *v > 0.0)?;
    let bytes = gb * 1e9;
    (bytes.is_finite() && bytes <= u64::MAX as f64).then_some(bytes as u64)
}

/// TP-priced model view for budget and per-request cost. One view for both;
/// pricing blocks with whole-model heads under TP overstates the budget.
/// `None` when TP is refused (launch flag off, hybrid+TP, missing or
/// non-divisible heads). Never truncates a shard.
fn tp_priced_model<'a, const MULTI_GPU: bool>(
    model: &'a crate::context::ModelArch,
    tp: u32,
) -> Option<std::borrow::Cow<'a, crate::context::ModelArch>> {
    if tp == 0 || (!MULTI_GPU && tp > 1) {
        return None;
    }
    if tp > 1 && r2_kv_cache_pressure::model_is_hybrid(model) {
        return None;
    }
    if tp <= 1 {
        return Some(std::borrow::Cow::Borrowed(model));
    }
    // Resharding heads needs a mutated copy; the tp=1 path borrows above.
    let mut priced = model.clone();
    let heads = priced.num_kv_heads.filter(|&h| h > 0)?;
    if heads % tp != 0 {
        return None;
    }
    priced.num_kv_heads = Some(heads / tp);
    Some(std::borrow::Cow::Owned(priced))
}

/// Observed block budget when labels+geometry resolve, else derived headroom.
/// Call with the TP-priced view from [`tp_priced_model`].
fn kv_pool_budget_bytes(
    model_view: &crate::context::ModelArch,
    kv_cache_dtype: Option<&str>,
    kv_headroom_gb: Option<f64>,
    cache: Option<&crate::collectors::CacheConfigLabels>,
) -> Option<u64> {
    use crate::engine::baseline::kv_bytes_per_element;
    let kv_bpp = kv_bytes_per_element(kv_cache_dtype);
    let observed = cache.and_then(|c| observed_budget_bytes(c, model_view, kv_bpp));
    observed.or(derived_budget_bytes(kv_headroom_gb))
}

/// GPU KV pool bytes: observed block budget when labels+geometry resolve, else
/// derived headroom. `None` for SWA / incomplete inputs (same gates as capacity).
/// On TP refusal, falls back to headroom-only (unlike capacity, which declines).
pub(super) fn resolve_kv_pool_bytes(
    kv_headroom_gb: Option<f64>,
    model: Option<&crate::context::ModelArch>,
    kv_cache_dtype: Option<&str>,
    tp: Option<u32>,
    cache: Option<&crate::collectors::CacheConfigLabels>,
) -> Option<u64> {
    let estimated = derived_budget_bytes(kv_headroom_gb);
    let Some(model) = model else {
        return estimated;
    };
    let Some(model_view) =
        tp_priced_model::<{ crate::engine::MULTI_GPU_TP }>(model, tp.unwrap_or(1))
    else {
        return estimated;
    };
    kv_pool_budget_bytes(model_view.as_ref(), kv_cache_dtype, kv_headroom_gb, cache)
}

/// Takes the same (possibly TP-sharded) view used to price requests. Budget and
/// cost must share one view; per-GPU blocks priced with whole-model heads
/// overstates the budget.
fn observed_budget_bytes(
    cache: &crate::collectors::CacheConfigLabels,
    model: &crate::context::ModelArch,
    kv_dtype_bytes: u8,
) -> Option<u64> {
    if model.swa_window.is_some() || model.num_swa_layers.is_some() {
        return None;
    }
    let blocks = u64::from(cache.num_gpu_blocks?);
    if r2_kv_cache_pressure::model_is_hybrid(model) {
        return blocks.checked_mul(cache.mamba_page_size_padded?);
    }

    let block_size = u64::from(cache.block_size?);
    let layers = u64::from(model.num_kv_layers.or(model.num_layers)?);
    let per_token_per_layer = 2u64
        .checked_mul(u64::from(model.num_kv_heads?))?
        .checked_mul(u64::from(model.head_dim?))?
        .checked_mul(u64::from(kv_dtype_bytes))?;
    blocks
        .checked_mul(block_size)?
        .checked_mul(per_token_per_layer)?
        .checked_mul(layers)
}

/// Safety margin on a recommended `--max-num-seqs`. Shared by R5 and R7 so two
/// rules never print two different recommended values for the same server.
pub(super) const RECOMMENDED_SEQS_SAFETY_MARGIN: f64 = 0.80;

/// Percent buffer implied by [`RECOMMENDED_SEQS_SAFETY_MARGIN`] (0.80 → 20).
/// Never hard-code "20" in operator strings; call this.
pub(super) fn recommended_seqs_safety_buffer_pct() -> u32 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        ((1.0 - RECOMMENDED_SEQS_SAFETY_MARGIN) * 100.0).round() as u32
    }
}

/// Binder reason shared by R5 raise bullets and R7 Recommended lines.
/// Callers wrap in parentheses. Ridge wording for R5 keeps the wall number
/// inline separately; this helper covers R7 ridge + all memory binders.
pub(super) fn recommended_seqs_binder_reason(rec: &RecommendedSeqs) -> String {
    if rec.empirical {
        return "est".to_string();
    }
    match rec.binder {
        BindingWall::Ridge | BindingWall::Config => "bound by compute ridge".to_string(),
        BindingWall::Memory { .. } => {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let wall_n = rec.wall.floor() as u32;
            let pct = recommended_seqs_safety_buffer_pct();
            match rec.source {
                Some(KvBoundSource::Observed) => {
                    format!("fits {wall_n} observed, {pct}% safety buffer")
                }
                Some(KvBoundSource::Derived | KvBoundSource::DerivedHybrid) => {
                    format!("fits {wall_n} (est), {pct}% safety buffer")
                }
                None => format!("fits {wall_n}, {pct}% safety buffer"),
            }
        }
    }
}

/// Empirical KV bounds are structurally optimistic (footprints grow over a
/// request's life; the estimator sees them young). Cap each prescription at a
/// bounded step; the loop re-raises after re-measure.
pub(super) const EMPIRICAL_STEP_CAP_MULT: f64 = 2.0;

/// Action-attached cautions render at 8-space, no bullet, blank line after.
/// Bullets are operator actions only.
pub(super) const KV_SCALE_CAUTION: &str = "        Monitor KV cache when scaling up.";

/// Which wall bound a concurrency estimate. Tie-break ordering: memory > ridge > config.
/// R1 uses all three (occupancy). R5/R7 use only the two physical walls; config is a
/// knob, not a wall (you cannot cap a knob by its own current value).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum BindingWall {
    Config,
    Ridge,
    /// Floored KV concurrency cap.
    Memory {
        cap: u32,
    },
}

/// Source of a resolved KV capacity bound (Observed / Derived only).
/// Live-traffic extrapolation is a separate `kv_floor`, not a source here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum KvBoundSource {
    /// vLLM-reported `kv_cache_max_concurrency`. No "(est)".
    Observed,
    /// `compute_kv_max_seqs_for_cache` on a dense/attention model.
    Derived,
    /// `compute_kv_max_seqs_for_cache` on a hybrid model (linear_* fields set).
    DerivedHybrid,
}

/// Three-wall headroom: `min(max_num_seqs, ridge?, kv_capacity?)`.
/// `kv_capacity` is Observed `kv_cache_max_concurrency.floor()` when present and finite.
/// Absent → two-way min; never claim memory.
/// Ties: memory > ridge > config (memory hurts fastest).
pub(super) fn effective_max_and_binder(
    max_n: u32,
    ridge_batch_size: Option<f64>,
    kv_cache_max_concurrency: Option<f64>,
) -> (f64, BindingWall) {
    let mut value = f64::from(max_n);
    let mut wall = BindingWall::Config;

    // Order + `<=` encodes tie-break: ridge beats config, memory beats both.
    if let Some(ridge) = ridge_batch_size.filter(|r| r.is_finite() && *r > 0.0)
        && ridge <= value
    {
        value = ridge;
        wall = BindingWall::Ridge;
    }

    if let Some(raw) = kv_cache_max_concurrency.filter(|c| c.is_finite() && *c > 0.0) {
        let cap = raw.floor();
        if cap > 0.0 && cap <= f64::from(u32::MAX) && cap <= value {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let cap_u = cap as u32;
            value = cap;
            wall = BindingWall::Memory { cap: cap_u };
        }
    }

    (value, wall)
}

/// Two physical walls only: `min(ridge?, kv_capacity?)`. Config excluded (a knob,
/// not a wall). `None` when neither wall is known. Tie-break: memory > ridge.
pub(super) fn physical_wall_and_binder(
    ridge_batch_size: Option<f64>,
    kv_capacity: Option<f64>,
) -> Option<(f64, BindingWall)> {
    let mut value: Option<f64> = None;
    let mut wall = BindingWall::Ridge;

    if let Some(ridge) = ridge_batch_size.filter(|r| r.is_finite() && *r > 0.0) {
        value = Some(ridge);
        wall = BindingWall::Ridge;
    }

    if let Some(raw) = kv_capacity.filter(|c| c.is_finite() && *c > 0.0) {
        let cap = raw.floor();
        if cap > 0.0 && cap <= f64::from(u32::MAX) && value.is_none_or(|v| cap <= v) {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let cap_u = cap as u32;
            value = Some(cap);
            wall = BindingWall::Memory { cap: cap_u };
        }
    }

    value.map(|v| (v, wall))
}

/// Empirical KV capacity from live metrics: `running / kv_usage_fraction` gives the
/// sequence count that would fill KV to 100%. `None` below 1% KV (extrapolation noise)
/// or when running is not a positive finite number.
pub(super) fn empirical_kv_max(running: f64, kv_cache_usage_perc: Option<f64>) -> Option<f64> {
    let kv_frac = kv_cache_usage_perc.filter(|v| v.is_finite() && *v > 1.0)?;
    let run = running
        .is_finite()
        .then_some(running)
        .filter(|r| *r > 0.0)?;
    Some(run / (kv_frac / 100.0))
}

/// Decline a capacity value when peak running has already exceeded `floor(value)`.
pub(super) fn kv_bound_survives_peak(peak_running: Option<f64>, value: f64) -> bool {
    match peak_running.filter(|p| p.is_finite()) {
        Some(peak) => peak <= value.floor(),
        None => true,
    }
}

/// Resolve the KV concurrency bound once for the run.
///
/// Returns `(capacity, source, floor)`:
/// - `capacity` / `source`: Observed, else derived / DerivedHybrid (peak-gated).
/// - `floor`: empirical `mean(running) / peak(kv_fraction)` when no real capacity
///   survived; never mixed into `capacity`. Callers pass it to
///   [`recommended_seqs`] as `kv_floor` so it caps the target without becoming
///   the binding wall.
///
/// Observed and derived values are declined when `peak_running > floor(value)`;
/// empirical is not peak-gated. It is a conservative floor, not a measured wall:
/// `mean(running) / peak(kv_fraction)` biases low on purpose, so peak running
/// exceeding it is expected by design. Observed and derived are full-context
/// capacity claims, so peak running above those does falsify them.
pub(super) fn resolve_kv_bound(
    observed_concurrency: Option<f64>,
    derived: Option<u32>,
    is_hybrid: bool,
    mean_running: Option<f64>,
    peak_kv_pct: Option<f64>,
    peak_running: Option<f64>,
) -> (Option<f64>, Option<KvBoundSource>, Option<f64>) {
    if let Some(c) = observed_concurrency.filter(|c| c.is_finite() && *c > 0.0)
        && kv_bound_survives_peak(peak_running, c)
    {
        return (Some(c), Some(KvBoundSource::Observed), None);
    }
    if let Some(d) = derived.filter(|&d| d > 0) {
        let value = f64::from(d);
        if kv_bound_survives_peak(peak_running, value) {
            let src = if is_hybrid {
                KvBoundSource::DerivedHybrid
            } else {
                KvBoundSource::Derived
            };
            return (Some(value), Some(src), None);
        }
    }
    if let Some(mean_r) = mean_running
        && let Some(emp) = empirical_kv_max(mean_r, peak_kv_pct)
    {
        return (None, None, Some(emp));
    }
    (None, None, None)
}

/// A margined `--max-num-seqs` recommendation and the wall that produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RecommendedSeqs {
    /// `floor(0.80 x wall)`, optionally floored by `kv_floor`, then step-capped
    /// when empirical-grade.
    pub target: u32,
    /// Binding value used for the margined target: ridge knee, Observed/Derived
    /// KV cap, or (floor-only fallback) the live-traffic floor itself. Only a
    /// real capacity number when [`Self::wall_is_capacity`] is true; the
    /// floor-only fallback must not be divided into as "hardware capacity."
    pub wall: f64,
    /// Ridge or Memory (config is never a wall here).
    pub binder: BindingWall,
    /// Source of the KV bound when memory binds; `None` when ridge binds or when
    /// only the live-traffic floor is known.
    pub source: Option<KvBoundSource>,
    /// True when the live-traffic KV floor lowered the target (or was the only
    /// capacity input). Drives step cap, Low confidence, `(est)`, and the
    /// raise-path "Monitor KV cache" caution line together with derived-unknown
    /// demotion via [`Self::empirical`].
    pub empirical: bool,
    /// True when the live-traffic KV floor lowered the target below
    /// `floor(0.80 x wall)`, or when the floor was the only capacity input.
    pub floor_capped: bool,
    /// True when [`Self::wall`] is a real capacity number (ridge or
    /// Observed/Derived KV). False only for the floor-only fallback, where
    /// `wall` holds the live-traffic estimate so target math still works.
    pub wall_is_capacity: bool,
}

/// One margined recommendation from the physical walls plus an optional
/// live-traffic floor. `None` when neither ridge, real `kv_bound`, nor `kv_floor`
/// is known (never invent a number).
///
/// `kv_bound` / `kv_source` are Observed or Derived only. Empirical arrives as
/// `kv_floor` and caps the target without entering [`physical_wall_and_binder`].
///
/// `kv_dtype_source`: when the binding wall is Derived/DerivedHybrid and the KV
/// element width was priced as [`KvCacheDtypeSource::Unknown`], the bound is
/// demoted to empirical-grade (step cap, Low confidence, monitor caution).
pub(super) fn recommended_seqs(
    ridge: Option<f64>,
    kv_bound: Option<f64>,
    kv_source: Option<KvBoundSource>,
    kv_floor: Option<f64>,
    current_max_num_seqs: Option<u32>,
    kv_dtype_source: Option<crate::engine::baseline::KvCacheDtypeSource>,
) -> Option<RecommendedSeqs> {
    use crate::engine::baseline::KvCacheDtypeSource;

    let floor_positive = kv_floor.filter(|f| f.is_finite() && *f > 0.0);
    let physical = physical_wall_and_binder(ridge, kv_bound);
    let floor_only = physical.is_none() && floor_positive.is_some();

    let (wall, binder) = match physical {
        Some(wb) => wb,
        None => {
            let f = floor_positive?;
            let cap = f.floor();
            if !(cap > 0.0 && cap <= f64::from(u32::MAX)) {
                return None;
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let cap_u = cap as u32;
            (cap, BindingWall::Memory { cap: cap_u })
        }
    };

    let binder_is_memory = matches!(binder, BindingWall::Memory { .. });
    let derived_unknown = binder_is_memory
        && matches!(
            kv_source,
            Some(KvBoundSource::Derived | KvBoundSource::DerivedHybrid)
        )
        && kv_dtype_source == Some(KvCacheDtypeSource::Unknown);

    let mut target = (wall * RECOMMENDED_SEQS_SAFETY_MARGIN).floor();
    let mut floor_capped = floor_only;
    if let Some(f) = floor_positive {
        let floored = (f * RECOMMENDED_SEQS_SAFETY_MARGIN).floor();
        if floored < target {
            target = floored;
            floor_capped = true;
        }
    }

    let empirical = floor_capped || derived_unknown;
    if empirical && let Some(cur) = current_max_num_seqs {
        target = target.min(EMPIRICAL_STEP_CAP_MULT * f64::from(cur));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let target_u = u32::try_from(target as u64).ok().filter(|&n| n > 0)?;

    let source = if floor_only {
        None
    } else {
        binder_is_memory.then_some(kv_source).flatten()
    };
    Some(RecommendedSeqs {
        target: target_u,
        wall,
        binder,
        source,
        empirical,
        floor_capped,
        wall_is_capacity: !floor_only,
    })
}

/// True when a rule fired in enough evaluable windows to be statistically stable.
pub fn rule_is_significant(fired: usize, total_evaluable: usize) -> bool {
    if total_evaluable == 0 {
        return false;
    }
    let pct = fired as f64 / total_evaluable as f64;
    fired >= ENGINE_MIN_PERSISTENT_WINDOWS && pct >= ENGINE_MIN_WINDOW_PCT
}

/// Inserts "Seen in N% of windows" after the rule title line in multi-window display blocks.
pub(super) fn with_seen_pct(mut lines: Vec<String>, seen_pct: u32) -> Vec<String> {
    lines.insert(1, format!("    Seen in {seen_pct}% of windows"));
    lines
}

/// True when a named target equals the configured value (equality only).
/// `None` configured → false: cannot claim a no-op against an unread value.
/// Configured above target → false: lowering is still a prescription.
pub(super) fn already_set_u32(configured: Option<u32>, target: u64) -> bool {
    configured.is_some_and(|c| u64::from(c) == target)
}

pub(super) fn trim_group_trailing_blanks(lines: &mut Vec<String>) {
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
}

/// Bullet plus optional continuation sub-line. After a sub-line, inserts one blank
/// line before the next bullet (D5). Callers rely on group-end trimming so a
/// trailing blank is not left when this bullet is last in its group.
///
/// Punctuation: Cause / Fix / Expected / Last resort body lines are sentences and
/// end with `.`. Section headers end with `:`. Metric scoreboard rows stay bare.
pub(super) fn push_bullet_with_subline(
    out: &mut Vec<String>,
    bullet: String,
    subline: Option<&str>,
) {
    out.push(bullet);
    if let Some(sub) = subline {
        out.push(format!("        {}", sub.trim_start()));
        out.push(String::new());
    }
}

/// One Fix-group bullet with an optional action-attached subline.
/// Subline is decided when the bullet is built, never by re-reading printed text.
pub(super) type CutBullet = (String, Option<&'static str>);

/// Emit `Fix:` then safe / cuts / rejects groups.
///
/// When both safe and cuts are non-empty, both get labels (`Safe to apply:` /
/// `Cuts throughput:`). A labeled group is never followed by unlabeled bullets.
/// When only one of those groups exists, keep unlabeled safe under `Fix:` and
/// labeled cuts only (today's single-group behavior).
///
/// `lead_with_cuts`: emit the cuts group before safe (shrink-led / contradicted-cap
/// paths). Labels and rejection warnings stay attached to their bullets.
pub(super) fn push_grouped_fixes(
    out: &mut Vec<String>,
    mut safe: Vec<String>,
    mut cuts_throughput: Vec<CutBullet>,
    mut rejects: Vec<String>,
    lead_with_cuts: bool,
) {
    trim_group_trailing_blanks(&mut safe);
    while cuts_throughput.last().is_some_and(|(b, _)| b.is_empty()) {
        cuts_throughput.pop();
    }
    trim_group_trailing_blanks(&mut rejects);

    out.push("    Fix:".to_string());
    let both_safe_and_cuts = !safe.is_empty() && !cuts_throughput.is_empty();

    let push_safe = |out: &mut Vec<String>, safe: Vec<String>| {
        if safe.is_empty() {
            return;
        }
        if both_safe_and_cuts {
            out.push("    Safe to apply:".to_string());
        }
        out.extend(safe);
    };
    let push_cuts = |out: &mut Vec<String>, cuts: Vec<CutBullet>, blank_before: bool| {
        if cuts.is_empty() {
            return;
        }
        if blank_before {
            out.push(String::new());
        }
        out.push("    Cuts throughput:".to_string());
        for (bullet, sub) in cuts {
            push_bullet_with_subline(out, bullet, sub);
        }
        // Subline blanks are group-internal; strip trailing so the caller owns
        // the blank before the next section (Expected / Rejects).
        while out.last().is_some_and(|l| l.is_empty()) {
            out.pop();
        }
    };

    if lead_with_cuts {
        push_cuts(out, cuts_throughput, false);
        if both_safe_and_cuts {
            out.push(String::new());
        }
        push_safe(out, safe);
    } else {
        push_safe(out, safe);
        push_cuts(out, cuts_throughput, both_safe_and_cuts);
    }

    if !rejects.is_empty() {
        let only_fix_header = out.last().is_some_and(|l| l == "    Fix:");
        if !only_fix_header && !out.last().is_some_and(|l| l.is_empty()) {
            out.push(String::new());
        }
        out.push("    Rejects requests:".to_string());
        out.extend(rejects);
    }
}

/// Push a shrink suggestion into a cuts group. Subline (rejection warning) is
/// taken from [`ShrinkSuggestion::subline`], decided at build time.
pub(super) fn extend_with_shrink_suggestion(out: &mut Vec<CutBullet>, shrink: ShrinkSuggestion) {
    let mut it = shrink.lines.into_iter();
    let Some(bullet) = it.next() else {
        return;
    };
    out.push((bullet, shrink.subline));
    for extra in it {
        out.push((extra, None));
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Recommendation {
    pub rule_name: &'static str,
    pub layer: u8,
    /// 1-5; 5 = highest impact
    pub impact: u8,
    /// 0.0-1.0
    pub confidence: f64,
    /// Pre-formatted cause + recommendation lines for stdout.
    ///
    /// NOTE (future work): `display_lines` couples presentation to the engine.
    /// Rules build terminal strings here, so wording changes grow rule files.
    /// Deferred decision: migrate formatting to `output/` (rules return structured
    /// facts) if rule-file growth becomes painful. Tracked in `deferred.md`.
    pub display_lines: Vec<String>,
    /// True when the fix branch has no server-local knob (wall / scale-out only).
    /// Set by the rule formatter from the branch it took; never inferred from text.
    pub terminal: bool,
}

/// Mean of present `f64` values. Empty iterator → `None`.
pub(crate) fn mean_of_present(vals: impl Iterator<Item = f64>) -> Option<f64> {
    let mut sum = 0.0_f64;
    let mut n = 0usize;
    for v in vals {
        sum += v;
        n += 1;
    }
    (n > 0).then_some(sum / n as f64)
}

const NO_ISSUES_LINE: &str = "No issues detected.";

/// Canonical `Recommendation.rule_name` values, single source of truth for DAG + output coupling.
pub mod rule_names {
    pub const UNDER_BATCHING: &str = "under_batching";
    pub const KV_CACHE_PRESSURE: &str = "kv_cache_pressure";
    pub const KV_ADMISSION_BACKLOG: &str = "kv_admission_backlog";
    pub const OOM_RISK: &str = "oom_risk";
    pub const CONCURRENCY_SATURATION: &str = "concurrency_saturation";
    pub const LOW_PREFIX_REUSE: &str = "low_prefix_reuse";
    pub const PREFILL_BOUND: &str = "prefill_bound";
    pub const CONFIG_HEADROOM: &str = "config_headroom";
    pub const MASSIVE_UNDERUTILIZATION: &str = "massive_underutilization";

    /// Human-readable label for a rule name, used in journey UI output.
    pub fn display_name(rule_name: &str) -> &str {
        match rule_name {
            UNDER_BATCHING => "Under-batching",
            KV_CACHE_PRESSURE => "KV Cache Pressure",
            KV_ADMISSION_BACKLOG => "KV Admission Backlog",
            OOM_RISK => "OOM Risk",
            CONCURRENCY_SATURATION => "Concurrency Saturation",
            LOW_PREFIX_REUSE => "Low Prefix Reuse",
            PREFILL_BOUND => "Prefill-Bound",
            CONFIG_HEADROOM => "Configured Batch Limit",
            MASSIVE_UNDERUTILIZATION => "Massive Under-utilization",
            _ => rule_name,
        }
    }
}

#[cfg(test)]
mod already_set_tests {
    use super::already_set_u32;

    #[test]
    fn equality_only() {
        assert!(already_set_u32(Some(2048), 2048));
        assert!(!already_set_u32(Some(4096), 2048));
        assert!(!already_set_u32(Some(1024), 2048));
        assert!(!already_set_u32(None, 2048));
    }
}

#[cfg(test)]
mod confidence_tests {
    use super::{CONFIDENCE_HIGH_MIN, CONFIDENCE_MEDIUM_MIN, confidence_label};

    #[test]
    fn confidence_label_boundaries() {
        assert_eq!(confidence_label(CONFIDENCE_HIGH_MIN), "High");
        assert_eq!(confidence_label(CONFIDENCE_HIGH_MIN - 0.01), "Medium");
        assert_eq!(confidence_label(CONFIDENCE_MEDIUM_MIN), "Medium");
        assert_eq!(confidence_label(CONFIDENCE_MEDIUM_MIN - 0.01), "Low");
    }

    #[test]
    fn confidence_label_nan_is_low() {
        assert_eq!(confidence_label(f64::NAN), "Low");
    }

    #[test]
    fn no_rule_defines_own_confidence_ladder() {
        // Single definition site in mod.rs; no rule file may duplicate it.
        let rule_files = [
            include_str!("r1_under_batching.rs"),
            include_str!("r2_kv_cache_pressure.rs"),
            include_str!("r3_low_prefix_reuse.rs"),
            include_str!("r4_oom_risk.rs"),
            include_str!("r5_concurrency_saturation.rs"),
            include_str!("r6_prefill_bound.rs"),
            include_str!("r7_config_headroom.rs"),
        ];
        for src in &rule_files {
            assert!(
                !src.contains("fn confidence_label"),
                "Rule file defines its own confidence_label; use super::confidence_label instead"
            );
        }
    }
}

#[cfg(test)]
/// Cause / Fix / Expected / Last resort / Watch body lines are sentences.
/// They end with `.`. Section headers end with `:`. Scoreboard rows stay bare.
pub(crate) fn assert_issue_prose_periods(lines: &[String]) {
    const SECTION_HEADERS: &[&str] = &[
        "    Cause:",
        "    Fix:",
        "    Safe to apply:",
        "    Cuts throughput:",
        "    Rejects requests:",
        "    Last resort:",
    ];
    let mut in_prose = false;
    for line in lines {
        let t = line.as_str();
        if SECTION_HEADERS.contains(&t) {
            in_prose = true;
            continue;
        }
        if t.starts_with("    Expected:") || t.starts_with("    Watch:") {
            assert!(
                t.ends_with('.'),
                "Expected/Watch line must end with period: {t}"
            );
            in_prose = false;
            continue;
        }
        if t.starts_with("    Confidence:")
            || t.starts_with("[!")
            || t.starts_with("[i]")
            || t.starts_with("ISSUES:")
        {
            in_prose = false;
            continue;
        }
        if !in_prose || t.is_empty() {
            continue;
        }
        // Bullets and indented sublines under the active section.
        if t.starts_with("      ") || t.starts_with("        ") {
            assert!(
                t.ends_with('.'),
                "prose body must end with period: {t}\nfull:\n{}",
                lines.join("\n")
            );
        }
    }
}

#[cfg(test)]
mod prose_punctuation_tests {
    #[test]
    fn enable_prefix_bullet_is_punctuated() {
        assert!(super::ENABLE_PREFIX_CACHING_BULLET.ends_with('.'));
        assert!(super::ENABLE_PREFIX_CACHING_EXPECTED.ends_with('.'));
    }
}
