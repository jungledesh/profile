use std::time::SystemTime;

use crate::collectors::RawSnapshot;
use crate::engine::baseline::{self, PhysicsBaseline};

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

pub use eval::build_report_for_windows;
pub(crate) use eval::{aggregate_prefix_hit_rate_for_windows, finalize_report_groups};
pub use format::{
    format_diagnose_rules, format_diagnose_rules_for_windows, no_evaluable_diagnose_lines,
};
pub use r1_under_batching::{
    R1EvalInput, R1MissReport, Rule1Outcome, UnderBatchingDetail, r1_recommendation,
    r1_verbose_miss_line,
};
pub use r2_kv_cache_pressure::{
    KvAdmissionBacklogDetail, KvCachePressureDetail, Rule2Outcome, r2_recommendation,
    rule2_kv_admission_backlog, rule2_kv_cache_pressure,
};
pub use r3_low_prefix_reuse::{
    LowPrefixReuseDetail, Rule3Outcome, r3_recommendation, rule3_low_prefix_reuse,
};
pub use r4_oom_risk::{r4_advisory, r4_recommendation};
pub use r5_concurrency_saturation::{
    ConcurrencySaturationDetail, r5_recommendation, rule5_concurrency_saturation,
};
pub use r6_prefill_bound::{
    PrefillBoundDetail, PrefillBoundEvalInput, R6GateInput, Rule6Outcome, r6_recommendation,
    r6_verbose_miss_line,
};
pub use r7_config_headroom::{ConfigHeadroomDetail, rule7_config_headroom};

fn histogram_prefill_fraction_for_confidence(
    baseline: Option<&PhysicsBaseline>,
    snapshot: &RawSnapshot,
) -> Option<f64> {
    baseline
        .and_then(|b| b.prefill_time_fraction)
        .or_else(|| baseline::prefill_time_fraction_from_snapshot(snapshot))
}

pub(super) const MAX_OBSERVATION_SKEW_SECS: f64 = 1.0;
/// Enforces >= 6s temporal substance (3 windows × 2s).
pub(super) const ENGINE_MIN_PERSISTENT_WINDOWS: usize = 3;
/// Enforces >= 25% density floor across evaluable windows.
pub(super) const ENGINE_MIN_WINDOW_PCT: f64 = 0.25;

/// Push a max_model_len shrink suggestion into `lines`.
/// Hard number only when `total_count >= 100` and both p99s are present.
/// No-op when `max_model_len` is None.
pub(super) fn push_model_len_shrink_suggestion(
    lines: &mut Vec<String>,
    max_model_len: Option<u32>,
    prompt_p99: Option<f64>,
    generation_p99: Option<f64>,
    total_count: f64,
    indent: &str,
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
        lines.push(format!(
            "{indent}• Lower --max-model-len (current: {m}) to ~{suggested} \
             (prompt p99 {pp:.0} tok + output p99 {gp:.0} tok), to shrink KV footprint."
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

pub(super) fn compute_kv_max_seqs(
    kv_headroom_gb: Option<f64>,
    max_model_len: Option<u32>,
    model: &crate::context::ModelArch,
    kv_cache_dtype: Option<&str>,
    tp: Option<u32>,
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
    let kv_bpp = kv_bytes_per_element(kv_cache_dtype, 2);
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
    lines.insert(1, format!("  Seen in {seen_pct}% of windows"));
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

#[derive(Debug, Clone)]
pub struct IssueGroup {
    pub primary: Recommendation,
    pub secondary: Vec<Recommendation>,
}

impl IssueGroup {
    pub fn score(&self) -> f64 {
        self.primary.impact as f64 * self.primary.confidence
    }
}

const NO_ISSUES_LINE: &str = "No issues detected in this snapshot.";

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

pub(super) fn skew_secs(a: SystemTime, b: SystemTime) -> f64 {
    match a.duration_since(b) {
        Ok(d) => d.as_secs_f64(),
        Err(e) => -e.duration().as_secs_f64(),
    }
    .abs()
}
