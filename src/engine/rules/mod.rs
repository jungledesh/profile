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
#[cfg(test)]
pub(crate) use r3_low_prefix_reuse::{LowPrefixReuseDetail, Rule3Outcome, r3_recommendation};
pub use r4_oom_risk::{r4_advisory, r4_recommendation};

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
/// `mamba_cache_mode` changes shift `state_pages` (measured 3→6 none→align) —
/// counterfactuals that change caching mode stay directional, no number.
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

/// Build max_model_len shrink suggestion lines (may be empty).
/// Hard number only when `total_count >= 100` and both p99s are present.
/// Empty when `max_model_len` is None or the shrink is < 5%.
///
/// `current_shown`: when true, the block already names the current max_model_len
/// above this bullet, so emit `to {suggested}`; otherwise `{current} → {suggested}`.
///
/// Projected concurrency at `{suggested}` comes only from
/// [`capacity_at_hypothetical_max_len`](suggested) — never from the current-config
/// R2 ceiling (`r2_kv_max_seqs` / observed concurrency at full `max_model_len`).
pub(super) fn model_len_shrink_suggestion_lines(
    max_model_len: Option<u32>,
    prompt_p99: Option<f64>,
    generation_p99: Option<f64>,
    total_count: f64,
    indent: &str,
    hyp: Option<&HypCapacityCtx<'_>>,
    current_shown: bool,
) -> Vec<String> {
    let mut lines = Vec::new();
    let Some(m) = max_model_len else {
        return lines;
    };

    if total_count >= 100.0 {
        let Some(pp) = prompt_p99 else {
            lines.push(format!(
                "{indent}• Lower --max-model-len (current: {m}) to safely raise concurrency."
            ));
            return lines;
        };
        let Some(gp) = generation_p99 else {
            lines.push(format!(
                "{indent}• Lower --max-model-len (current: {m}) to safely raise concurrency."
            ));
            return lines;
        };
        let suggested = (pp as u32).saturating_add(gp as u32);
        // Suppress if reduction is < 5% - not a meaningful change (avoids "5464 → 5465" no-ops)
        if suggested >= m.saturating_sub(m / 20) {
            return lines;
        }
        // Projection at the *suggested* length only — not current observed concurrency.
        let fits = hyp
            .and_then(|h| capacity_at_hypothetical_max_len(suggested, Some(m), h))
            .map(|n| format!("; fits at least {n} worst-case requests (est)"))
            .unwrap_or_default();
        let len_clause = if current_shown {
            format!("to {suggested}")
        } else {
            format!("{m} → {suggested}")
        };
        lines.push(format!(
            "{indent}• Lower --max-model-len {len_clause} \
             (fits p99 of observed requests){fits}"
        ));
        lines.push(format!(
            "{indent}  Warning: max_model_len is total context (prompt + completion). Truncation risk!"
        ));
    } else {
        lines.push(format!(
            "{indent}• Lower --max-model-len (current: {m}) to safely raise concurrency."
        ));
    }
    lines
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
    /// `(observed_budget_bytes, estimated_budget_bytes)` when both are available.
    pub budget_self_grade: Option<(u64, u64)>,
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
    let tp = tp.unwrap_or(1);
    if tp == 0 || (!MULTI_GPU && tp > 1) {
        return DerivedCapacity::default();
    }
    if tp > 1 && r2_kv_cache_pressure::model_is_hybrid(model) {
        return DerivedCapacity::default();
    }

    // Resharding heads needs a mutated copy; the tp=1 launch path borrows directly.
    // Non-divisible TP is refused (vLLM refuses it too). Never truncate a shard.
    let sharded_model = if tp > 1 {
        let mut priced_model = model.clone();
        let Some(heads) = priced_model.num_kv_heads.filter(|&h| h > 0) else {
            return DerivedCapacity::default();
        };
        if heads % tp != 0 {
            return DerivedCapacity::default();
        }
        priced_model.num_kv_heads = Some(heads / tp);
        Some(priced_model)
    } else {
        None
    };
    let model_view = sharded_model.as_ref().unwrap_or(model);

    let kv_bpp = kv_bytes_per_element(kv_cache_dtype);
    let Some(request_bytes) = bytes_per_seq(model_view, max_len, kv_bpp) else {
        return DerivedCapacity::default();
    };
    let observed = cache.and_then(|c| observed_budget_bytes(c, model_view, kv_bpp));
    let estimated = derived_budget_bytes(kv_headroom_gb);
    let budget = observed.or(estimated);
    let max_seqs = budget
        .and_then(|bytes| bytes.checked_div(request_bytes))
        .and_then(|n| u32::try_from(n).ok())
        .filter(|&n| n > 0);

    DerivedCapacity {
        max_seqs,
        budget_self_grade: observed.zip(estimated),
    }
}

fn derived_budget_bytes(kv_headroom_gb: Option<f64>) -> Option<u64> {
    let gb = kv_headroom_gb.filter(|v| v.is_finite() && *v > 0.0)?;
    let bytes = gb * 1e9;
    (bytes.is_finite() && bytes <= u64::MAX as f64).then_some(bytes as u64)
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

/// Empirical KV bounds are structurally optimistic (footprints grow over a
/// request's life; the estimator sees them young). Cap each prescription at a
/// bounded step; the loop re-raises after re-measure.
pub(super) const EMPIRICAL_STEP_CAP_MULT: f64 = 2.0;

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

/// Source of a resolved `kv_bound`. Observed and derived are trusted in full;
/// empirical is a load extrapolation that takes bounded, verified steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum KvBoundSource {
    /// vLLM-reported `kv_cache_max_concurrency`. No "(est)".
    Observed,
    /// `compute_kv_max_seqs_for_cache` on a dense/attention model.
    Derived,
    /// `compute_kv_max_seqs_for_cache` on a hybrid model (linear_* fields set).
    DerivedHybrid,
    /// `mean(running) / peak(kv_fraction)` extrapolation, last resort.
    Empirical,
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

/// Resolve the KV concurrency bound once for the run: Observed, else derived, else
/// empirical (run-level `mean(running) / peak(kv_fraction)`). Priority mirrors R2's
/// `resolve_r2_kv_capacity`; empirical fills the last gap and says so.
pub(super) fn resolve_kv_bound(
    observed_concurrency: Option<f64>,
    derived: Option<u32>,
    is_hybrid: bool,
    mean_running: Option<f64>,
    peak_kv_pct: Option<f64>,
) -> (Option<f64>, Option<KvBoundSource>) {
    if let Some(c) = observed_concurrency.filter(|c| c.is_finite() && *c > 0.0) {
        return (Some(c), Some(KvBoundSource::Observed));
    }
    if let Some(d) = derived.filter(|&d| d > 0) {
        let src = if is_hybrid {
            KvBoundSource::DerivedHybrid
        } else {
            KvBoundSource::Derived
        };
        return (Some(f64::from(d)), Some(src));
    }
    if let Some(mean_r) = mean_running
        && let Some(emp) = empirical_kv_max(mean_r, peak_kv_pct)
    {
        return (Some(emp), Some(KvBoundSource::Empirical));
    }
    (None, None)
}

/// A margined `--max-num-seqs` recommendation and the wall that produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RecommendedSeqs {
    /// `floor(0.80 x wall)`, step-capped when the binding wall is empirical.
    pub target: u32,
    /// The binding wall value (ridge tok/s knee or floored KV cap).
    pub wall: f64,
    /// Ridge or Memory (config is never a wall here).
    pub binder: BindingWall,
    /// Source of the KV bound when memory binds; `None` when ridge binds.
    pub source: Option<KvBoundSource>,
    /// True when the binding wall's source is empirical (drives step cap, Low
    /// confidence, and the raise-path "Monitor KV cache" caution line).
    pub empirical: bool,
}

/// One margined recommendation from the two physical walls. `None` when neither
/// ridge nor `kv_bound` is known (never invent a number).
pub(super) fn recommended_seqs(
    ridge: Option<f64>,
    kv_bound: Option<f64>,
    kv_source: Option<KvBoundSource>,
    current_max_num_seqs: Option<u32>,
) -> Option<RecommendedSeqs> {
    let (wall, binder) = physical_wall_and_binder(ridge, kv_bound)?;
    let binder_is_memory = matches!(binder, BindingWall::Memory { .. });
    let empirical = binder_is_memory && kv_source == Some(KvBoundSource::Empirical);

    let mut target = (wall * RECOMMENDED_SEQS_SAFETY_MARGIN).floor();
    if empirical && let Some(cur) = current_max_num_seqs {
        target = target.min(EMPIRICAL_STEP_CAP_MULT * f64::from(cur));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let target_u = u32::try_from(target as u64).ok().filter(|&n| n > 0)?;

    let source = binder_is_memory.then_some(kv_source).flatten();
    Some(RecommendedSeqs {
        target: target_u,
        wall,
        binder,
        source,
        empirical,
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

pub(super) fn trim_group_trailing_blanks(lines: &mut Vec<String>) {
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
}

/// Bullet plus optional continuation sub-line. After a sub-line, inserts one blank
/// line before the next bullet (D5). Callers rely on group-end trimming so a
/// trailing blank is not left when this bullet is last in its group.
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

/// Emit `Fix:` then safe bullets, then optional `Cuts throughput:` / `Rejects requests:`
/// groups. Empty safe group: `Cuts throughput:` follows `Fix:` with no blank between.
pub(super) fn push_grouped_fixes(
    out: &mut Vec<String>,
    mut safe: Vec<String>,
    mut cuts_throughput: Vec<String>,
    mut rejects: Vec<String>,
) {
    trim_group_trailing_blanks(&mut safe);
    trim_group_trailing_blanks(&mut cuts_throughput);
    trim_group_trailing_blanks(&mut rejects);

    out.push("    Fix:".to_string());
    let had_safe = !safe.is_empty();
    out.extend(safe);

    if !cuts_throughput.is_empty() {
        if had_safe {
            out.push(String::new());
        }
        out.push("    Cuts throughput:".to_string());
        out.extend(cuts_throughput);
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

/// Push shrink suggestion lines into a group, routing the truncation Warning
/// through [`push_bullet_with_subline`].
pub(super) fn extend_with_shrink_suggestion(out: &mut Vec<String>, shrink_lines: Vec<String>) {
    let mut it = shrink_lines.into_iter();
    let Some(bullet) = it.next() else {
        return;
    };
    let warning = it.next();
    push_bullet_with_subline(out, bullet, warning.as_deref());
    for extra in it {
        out.push(extra);
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
    /// facts) if rule-file growth becomes painful. Tracked in
    /// `architecture_audit_specs.md`.
    pub display_lines: Vec<String>,
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
