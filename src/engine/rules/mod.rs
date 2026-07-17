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
/// 1. observed-geometry page model when labels present
/// 2. attention-only catalog math when labels absent
/// 3. no number (caller stays directional)
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
    pub weight_bytes: u8,
}

/// Capacity at a hypothetical `max_model_len`. Both derived tiers are `(est)`.
pub(super) fn capacity_at_hypothetical_max_len(
    target_max_len: u32,
    current_max_len: Option<u32>,
    ctx: &HypCapacityCtx<'_>,
) -> Option<u32> {
    use crate::engine::baseline::counterfactual_concurrency;

    // Hybrid ladder geometry uses mamba_block_size when present; dense uses block_size.
    let block_size = ctx.cache.mamba_block_size.or(ctx.cache.block_size);
    if let (Some(bs), Some(blocks), Some(obs), Some(cur)) = (
        block_size,
        ctx.cache.num_gpu_blocks,
        ctx.cache.kv_cache_max_concurrency,
        current_max_len,
    ) && let Some(c) = counterfactual_concurrency(target_max_len, bs, blocks, obs, cur)
    {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n = c.floor() as u32;
        if n > 0 {
            return Some(n);
        }
    }
    let model = ctx.model?;
    compute_kv_max_seqs(
        ctx.kv_headroom_gb,
        Some(target_max_len),
        model,
        ctx.kv_cache_dtype,
        ctx.tp,
        ctx.weight_bytes,
    )
}

/// Push a max_model_len shrink suggestion into `lines`.
/// Hard number only when `total_count >= 100` and both p99s are present.
/// No-op when `max_model_len` is None.
/// When `hyp` is set and a target length is suggested, append projected capacity
/// (`≤n (est)`) via the observed-geometry → catalog preference order.
pub(super) fn push_model_len_shrink_suggestion(
    lines: &mut Vec<String>,
    max_model_len: Option<u32>,
    prompt_p99: Option<f64>,
    generation_p99: Option<f64>,
    total_count: f64,
    indent: &str,
    hyp: Option<&HypCapacityCtx<'_>>,
) {
    let Some(m) = max_model_len else { return };

    if total_count >= 100.0 {
        let Some(pp) = prompt_p99 else {
            lines.push(format!(
                "{indent}• Lower --max-model-len (current: {m}) to safely raise concurrency."
            ));
            return;
        };
        let Some(gp) = generation_p99 else {
            lines.push(format!(
                "{indent}• Lower --max-model-len (current: {m}) to safely raise concurrency."
            ));
            return;
        };
        let suggested = (pp as u32).saturating_add(gp as u32);
        // Suppress if reduction is < 5% - not a meaningful change (avoids "5464 → 5465" no-ops)
        if suggested >= m.saturating_sub(m / 20) {
            return;
        }
        let capacity_suffix = hyp
            .and_then(|h| capacity_at_hypothetical_max_len(suggested, Some(m), h))
            .map(|n| format!("; capacity ≤{n} (est)"))
            .unwrap_or_default();
        lines.push(format!(
            "{indent}• Lower --max-model-len (current: {m}) to ~{suggested} \
             (prompt p99 {pp:.0} tok + output p99 {gp:.0} tok), to shrink KV footprint{capacity_suffix}."
        ));
        lines.push(format!(
            "{indent}  Warning: max_model_len is total context (prompt + completion). Truncation risk!"
        ));
    } else {
        lines.push(format!(
            "{indent}• Lower --max-model-len (current: {m}) to safely raise concurrency."
        ));
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

pub(super) fn compute_kv_max_seqs(
    kv_headroom_gb: Option<f64>,
    max_model_len: Option<u32>,
    model: &crate::context::ModelArch,
    kv_cache_dtype: Option<&str>,
    tp: Option<u32>,
    weight_bytes: u8,
) -> Option<u32> {
    use crate::engine::baseline::{kv_bytes_per_element, kv_max_concurrent_seqs};
    let headroom = kv_headroom_gb?;
    let max_len = max_model_len?;
    let num_layers = model.num_kv_layers.or(model.num_layers)?;
    let num_kv_heads = model.num_kv_heads?;
    let head_dim = model.head_dim?;
    let sharded_kv_heads = tp
        .map(|t| num_kv_heads / num_kv_heads.min(t))
        .unwrap_or(num_kv_heads);
    // "auto"/absent KV dtype inherits weight bytes (fp8 weights → 1, bf16 → 2).
    let kv_bpp = kv_bytes_per_element(kv_cache_dtype, weight_bytes.max(1));
    kv_max_concurrent_seqs(
        headroom,
        max_len,
        num_layers,
        sharded_kv_heads,
        head_dim,
        kv_bpp,
    )
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

#[derive(Debug, Clone, PartialEq)]
pub struct Recommendation {
    pub rule_name: &'static str,
    pub layer: u8,
    /// 1–5; 5 = highest impact
    pub impact: u8,
    /// 0.0–1.0
    pub confidence: f64,
    /// Prescriptive: what to change
    pub action: String,
    /// One-liner for closed-loop direction block
    pub short_action: String,
    pub expected_impact: String,
    /// Pre-formatted cause + recommendation lines for stdout
    pub display_lines: Vec<String>,
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

/// Practical achievable efficiency ceiling. No production workload reaches 100% of the
/// roofline due to framework overhead, scheduling, and memory contention. 80% represents
/// a well-optimized production system. Waste is computed against this ceiling, not 100%.
pub const ACHIEVABLE_EFFICIENCY_CEILING: f64 = 0.80;
