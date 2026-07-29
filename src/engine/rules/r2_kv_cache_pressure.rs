use crate::collectors::RawSnapshot;
use crate::context::ModelArch;
use crate::engine::baseline::kv_bytes_per_element;

#[cfg(test)]
use super::{Recommendation, rule_names};

/// 88% matches observed vLLM production eviction onset; 85% was too conservative.
pub(crate) const KV_CACHE_PRESSURE_MIN_PERC: f64 = 88.0;
/// 0.02/s = ~1 eviction/minute; below this the scheduler is recovering normally,
/// not under sustained KV pressure. Avoids firing on a single-event spike.
const PREEMPTION_RATE_MIN_PER_SEC: f64 = 0.02;
/// Minimum concurrent swapped sequences before treating as active pressure.
/// Avoids firing on a single stale counter reading.
const SWAPPED_REQUESTS_MIN: f64 = 2.0;
/// Low floor avoids firing on transient scheduling jitter; keeps R2 from co-firing
/// with R5 when 1–2 requests queue at the concurrency cap.
const QUEUE_BACKPRESSURE_MIN_WAITING: f64 = 2.0;
/// 30% of active requests waiting signals the scheduler is consistently holding
/// requests for KV capacity, not just transient batching delay.
const KV_ADMISSION_BACKLOG_QUEUE_RATIO_MIN: f64 = 0.30;
/// Minimum headroom (GB) before recommending --gpu-memory-utilization; observed free
/// VRAM and the computed utilization budget must both exceed this value.
const KV_HEADROOM_SAFE_MIN_GB: f64 = 2.0;
const GPU_MEM_UTIL_FIX: &str =
    "      • Raise --gpu-memory-utilization (check vRAM header for avail mem) to expand KV pool";
const FP8_KV_CACHE_FIX: &str =
    "      • Switch --kv-cache-dtype fp8 to halve KV memory footprint (affects output quality)";
/// Suggest prefix caching when mean prompt length exceeds this (tokens).
const PREFIX_CACHING_LONG_PROMPT_MIN_TOKENS: f64 = 200.0;

pub(super) fn fp8_kv_cache_fix_bullet(
    kv_cache_dtype: Option<&str>,
    fp8_compiler_available: bool,
) -> Option<String> {
    // Advising a switch to the dtype already in use costs operator trust;
    // dtype is observable, so observe it.
    if kv_bytes_per_element(kv_cache_dtype) == 1 {
        return None;
    }
    // --kv-cache-dtype fp8 stores KV activations in fp8 via software cast - works on all GPUs
    // including A100. This is distinct from --quantization fp8 (weight quantization) which
    // requires native FP8 hardware and crashes on A100/Qwen3.6.
    Some(if fp8_compiler_available {
        FP8_KV_CACHE_FIX.to_string()
    } else {
        let base = FP8_KV_CACHE_FIX
            .strip_suffix(')')
            .unwrap_or(FP8_KV_CACHE_FIX);
        format!("{base}; FP8 compiler not found)")
    })
}

/// Observed device free VRAM in binary GB (MiB / 1024), matching the vRAM header.
fn observed_free_vram_gb(snapshot: &RawSnapshot) -> Option<f64> {
    let agg = snapshot.aggregate_gpu();
    let total = agg.sum_vram_total_mb?;
    let used = agg.vram_used_mb?;
    Some((total.saturating_sub(used)) as f64 / 1024.0)
}

/// Offer `--gpu-memory-utilization` only when measured free VRAM and the computed
/// utilization budget both clear the safe minimum.
fn gpu_mem_utilization_fix_bullet(
    snapshot: &RawSnapshot,
    kv_headroom_gb: Option<f64>,
) -> Option<String> {
    let free_gb = observed_free_vram_gb(snapshot)?;
    if free_gb <= KV_HEADROOM_SAFE_MIN_GB {
        return None;
    }
    if kv_headroom_gb.is_none_or(|h| h <= KV_HEADROOM_SAFE_MIN_GB) {
        return None;
    }
    Some(GPU_MEM_UTIL_FIX.to_string())
}

fn push_kv_pressure_safe_levers(
    safe: &mut Vec<String>,
    snapshot: &RawSnapshot,
    kv_headroom_gb: Option<f64>,
    kv_cache_dtype: Option<&str>,
    fp8_compiler_available: bool,
) {
    if let Some(bullet) = prefix_caching_fix_bullet(snapshot) {
        safe.push(bullet);
    }
    if let Some(bullet) = gpu_mem_utilization_fix_bullet(snapshot, kv_headroom_gb) {
        safe.push(bullet);
    }
    if let Some(bullet) = fp8_kv_cache_fix_bullet(kv_cache_dtype, fp8_compiler_available) {
        safe.push(bullet);
    }
    push_kv_offload_fix_last(safe, &snapshot.vllm.cache_config);
}

fn prefix_caching_fix_bullet(snapshot: &RawSnapshot) -> Option<String> {
    if snapshot.vllm.cache_config.enable_prefix_caching != Some(true)
        && snapshot
            .vllm
            .prompt_tokens_mean
            .is_some_and(|t| t >= PREFIX_CACHING_LONG_PROMPT_MIN_TOKENS)
    {
        Some(
            "      • Enable --enable-prefix-caching to share KV blocks across identical prompt prefixes"
                .to_string(),
        )
    } else {
        None
    }
}

const KV_OFFLOAD_FIX: &str = "      • Set --kv-offloading-size (GiB) to hold evicted KV in host memory instead of recomputing it";
/// Host-RAM caution; rendered via [`super::push_bullet_with_subline`] (8-space indent).
const KV_OFFLOAD_SUBLINE: &str =
    "Check host RAM and your container memory limit before allocating.";

/// Dead-end verify: config labels read set; effect is what we cannot prove.
const DEAD_END_VERIFY_BULLET: &str = "      • Verify prefix caching, gpu-memory-utilization, kv-cache-dtype and kv-offloading-size took effect.";
const DEAD_END_VERIFY_SUBLINE: &str = "Every lever profile can read is set or unavailable.";
const REPLICA_SCALE_OUT_BULLET: &str = "      • Add a replica to scale out.";
const REPLICA_KV_WALL_SUFFIX: &str = " No config change on this GPU moves the KV wall.";

fn replica_kv_wall_bullet() -> String {
    format!("{REPLICA_SCALE_OUT_BULLET}{REPLICA_KV_WALL_SUFFIX}")
}

fn push_dead_end_fixes(safe: &mut Vec<String>) {
    super::push_bullet_with_subline(
        safe,
        DEAD_END_VERIFY_BULLET.to_string(),
        Some(DEAD_END_VERIFY_SUBLINE),
    );
    safe.push(replica_kv_wall_bullet());
}

/// Suggest KV offload when the build exposes the label and size is off.
fn kv_offload_fix_bullet(cache: &crate::collectors::CacheConfigLabels) -> Option<String> {
    use crate::collectors::KvOffloadState;
    match cache.kv_offloading {
        KvOffloadState::Off => Some(KV_OFFLOAD_FIX.to_string()),
        KvOffloadState::Unsupported | KvOffloadState::Enabled(_) | KvOffloadState::Unreadable => {
            None
        }
    }
}

fn push_kv_offload_fix_last(safe: &mut Vec<String>, cache: &crate::collectors::CacheConfigLabels) {
    if let Some(bullet) = kv_offload_fix_bullet(cache) {
        super::push_bullet_with_subline(safe, bullet, Some(KV_OFFLOAD_SUBLINE));
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvAdmissionBacklogDetail {
    pub kv_cache_usage_perc: f64,
    pub kv_peak_pct: Option<f64>,
    pub admission_ratio: f64,
    pub requests_waiting: f64,
    pub requests_running: f64,
    pub free_kv_tokens: f64,
    pub demand_tokens: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvCachePressureDetail {
    pub kv_cache_usage_perc: Option<f64>,
    pub kv_peak_pct: Option<f64>,
    pub preemptions_active: bool,
    pub queue_backpressure: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Rule2Outcome {
    Fired(KvCachePressureDetail),
    NotFired,
}

pub fn rule2_kv_admission_backlog(snapshot: &RawSnapshot) -> Option<KvAdmissionBacklogDetail> {
    // Spec (CLAUDE.md r2): backlog is queue pressure *with KV near full*, minus
    // the preemption fire. Same bar the pressure path uses.
    if !super::kv_near_full(snapshot) {
        return None;
    }

    let kv = snapshot
        .vllm
        .kv_cache_usage_perc
        .filter(|v| v.is_finite())?;
    let wait = snapshot
        .vllm
        .num_requests_waiting
        .filter(|v| v.is_finite())?;
    let run = snapshot
        .vllm
        .num_requests_running
        .filter(|v| v.is_finite())?;
    let prompt_mean = snapshot.vllm.prompt_tokens_mean.filter(|v| v.is_finite())?;
    let num_gpu_blocks = snapshot.vllm.cache_config.num_gpu_blocks?;
    let block_size = snapshot.vllm.cache_config.block_size?;
    let max_seqs = snapshot.vllm.max_num_seqs?;

    // If running == max_num_seqs the scheduler is stalling on the concurrency cap,
    // not KV exhaustion. Can't rule out that cause without max_num_seqs, so require it.
    if run >= f64::from(max_seqs) {
        return None;
    }

    let total = wait + run;
    if total <= 0.0 {
        return None;
    }
    let ratio = wait / total;
    if ratio < KV_ADMISSION_BACKLOG_QUEUE_RATIO_MIN {
        return None;
    }

    let free_kv_tokens = f64::from(num_gpu_blocks) * f64::from(block_size) * (1.0 - kv / 100.0);
    let demand_tokens = wait * prompt_mean;
    if !(free_kv_tokens.is_finite() && demand_tokens.is_finite()) {
        return None;
    }
    if free_kv_tokens >= demand_tokens {
        return None;
    }

    Some(KvAdmissionBacklogDetail {
        kv_cache_usage_perc: kv,
        kv_peak_pct: snapshot
            .vllm
            .kv_cache_peak_perc
            .filter(|v| v.is_finite())
            .map(|peak| peak.max(kv)),
        admission_ratio: ratio,
        requests_waiting: wait,
        requests_running: run,
        free_kv_tokens,
        demand_tokens,
    })
}

/// Returns true when there is evidence of active KV eviction pressure.
/// Two distinct signals, either sufficient:
///
/// 1. Rate (velocity): preemptions/s > 0.02 - scheduler is actively evicting right now.
/// 2. Debt (static): num_requests_swapped ≥ 2 - sequences parked on CPU. This is a
///    gauge, not a delta. A non-zero count means eviction has already occurred and
///    sequences haven't been rescheduled yet. Risk: stuck alarm if swapped count is
///    stale and GPU has stabilized. A delta guard (swapped growing vs prior window)
///    would eliminate this - deferred until per-rule state is available at eval time.
fn eviction_signal_active(snapshot: &RawSnapshot) -> bool {
    snapshot
        .vllm
        .num_preemptions_per_sec
        .is_some_and(|p| p.is_finite() && p > PREEMPTION_RATE_MIN_PER_SEC)
        || snapshot
            .vllm
            .num_requests_swapped
            .is_some_and(|s| s.is_finite() && s >= SWAPPED_REQUESTS_MIN)
}

fn queue_backpressure(snapshot: &RawSnapshot) -> bool {
    snapshot
        .vllm
        .num_requests_waiting
        .is_some_and(|w| w.is_finite() && w > QUEUE_BACKPRESSURE_MIN_WAITING)
}

pub fn rule2_kv_cache_pressure(snapshot: &RawSnapshot) -> Rule2Outcome {
    if !super::kv_near_full(snapshot) {
        return Rule2Outcome::NotFired;
    }

    let kv = snapshot.vllm.kv_cache_usage_perc.filter(|v| v.is_finite());
    let preemptions_active = eviction_signal_active(snapshot);
    let queue_backpressure = queue_backpressure(snapshot);
    if !preemptions_active && !queue_backpressure {
        return Rule2Outcome::NotFired;
    }

    let kv_p = kv;
    let peak = snapshot
        .vllm
        .kv_cache_peak_perc
        .filter(|v| v.is_finite())
        .map(|peak| match kv_p {
            Some(avg) => peak.max(avg),
            None => peak,
        });

    Rule2Outcome::Fired(KvCachePressureDetail {
        kv_cache_usage_perc: kv_p,
        kv_peak_pct: peak,
        preemptions_active,
        queue_backpressure,
    })
}

#[cfg(test)]
pub struct R2RecommendationInput<'a> {
    pub snapshot: &'a RawSnapshot,
    pub max_model_len: Option<u32>,
    pub kv_headroom_gb: Option<f64>,
    pub kv_max_seqs: Option<u32>,
    pub capacity_label: KvCapacityLabel,
    pub windows_fired: usize,
    pub total_evaluable: usize,
    pub fp8_compiler_available: bool,
}

#[cfg(test)]
pub fn r2_recommendation(input: R2RecommendationInput<'_>) -> Option<Recommendation> {
    let R2RecommendationInput {
        snapshot,
        max_model_len,
        kv_headroom_gb,
        kv_max_seqs,
        capacity_label,
        windows_fired,
        total_evaluable,
        fp8_compiler_available,
    } = input;
    let Rule2Outcome::Fired(d) = rule2_kv_cache_pressure(snapshot) else {
        return None;
    };
    let confidence = if super::rule_is_significant(windows_fired, total_evaluable) {
        kv_pressure_confidence(windows_fired, total_evaluable)
    } else {
        0.5
    };
    Some(Recommendation {
        rule_name: rule_names::KV_CACHE_PRESSURE,
        layer: 2,
        impact: 5,
        confidence,
        display_lines: format_kv_cache_pressure_fired(
            &d,
            &KvFormatCtx {
                snapshot,
                max_model_len,
                kv_headroom_gb,
                kv_max_seqs,
                config_max_num_seqs: snapshot.vllm.max_num_seqs.or(Some(256)),
                capacity_label,
                fp8_compiler_available,
                model: None,
                tp: None,
                kv_cache_dtype: snapshot.vllm.cache_config.cache_dtype.as_deref(),
            },
            windows_fired,
            total_evaluable,
        ),
    })
}

/// How R2 labels a capacity recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCapacityLabel {
    /// From `kv_cache_max_concurrency`. No "(est)".
    Observed,
    /// From `compute_kv_max_seqs_for_cache` on a dense/attention model.
    Derived,
    /// From `compute_kv_max_seqs_for_cache` on a hybrid model (linear_* fields set).
    DerivedHybrid,
}

/// True when any hybrid/linear catalog field is present.
pub(super) fn model_is_hybrid(model: &ModelArch) -> bool {
    model.linear_num_layers.is_some()
        || model.linear_key_heads.is_some()
        || model.linear_value_heads.is_some()
        || model.linear_key_head_dim.is_some()
        || model.linear_value_head_dim.is_some()
        || model.linear_conv_kernel_dim.is_some()
        || model.state_dtype.is_some()
}

/// Prefer vLLM-reported concurrency; else derived math with honesty labels.
pub(super) fn resolve_r2_kv_capacity(
    observed_concurrency: Option<f64>,
    derived: Option<u32>,
    is_hybrid: bool,
) -> (Option<u32>, KvCapacityLabel) {
    if let Some(c) = observed_concurrency.filter(|c| c.is_finite() && *c > 0.0) {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n = c.floor() as u32;
        if n > 0 {
            return (Some(n), KvCapacityLabel::Observed);
        }
    }
    let label = if is_hybrid {
        KvCapacityLabel::DerivedHybrid
    } else {
        KvCapacityLabel::Derived
    };
    (derived, label)
}

fn seat_phrase_shows_number(snapshot: &RawSnapshot, cap: u32) -> bool {
    let Some(current) = snapshot.vllm.max_num_seqs else {
        return false;
    };
    if cap >= current {
        return false;
    }
    // Derived capacity is a full-context worst case, not a limit. If peak running
    // already beat it, printing it as a ceiling tells the operator to cut seats
    // the run proved they do not need to cut. Same test resolve_kv_bound applies
    // to derived (mod.rs kv_bound_survives_peak).
    if super::kv_cap_positive_after_floor(snapshot).is_some() {
        super::usable_kv_concurrency(snapshot).is_some()
    } else {
        super::kv_bound_survives_peak(snapshot.vllm.num_requests_running_peak, f64::from(cap))
    }
}

fn r2_capacity_phrase(n: u32, show_number: bool) -> String {
    if show_number {
        format!("Lower --max-num-seqs to ≤{n} to reduce KV demand")
    } else {
        "Lower --max-num-seqs to reduce KV demand".to_string()
    }
}

/// Follow-on seats after a named shrink target.
const FOLLOW_ON_SEAT_BULLET: &str = "      • Then lower --max-num-seqs to reduce KV demand";

pub(super) fn kv_pressure_confidence(windows_fired: usize, total_evaluable: usize) -> f64 {
    if total_evaluable == 0 {
        return 0.0;
    }
    (windows_fired as f64 / total_evaluable as f64).clamp(0.0, 1.0)
}

pub(super) fn kv_pressure_confidence_label(confidence: f64) -> &'static str {
    if confidence > 0.75 {
        "Confidence: High"
    } else if confidence >= 0.5 {
        "Confidence: Medium-High"
    } else {
        "Confidence: Medium"
    }
}

fn max_num_seqs_bullet(snapshot: &RawSnapshot, kv_max_seqs: Option<u32>) -> String {
    match kv_max_seqs {
        Some(n) => {
            let show = seat_phrase_shows_number(snapshot, n);
            format!("      • {}", r2_capacity_phrase(n, show))
        }
        None => "      • Lower --max-num-seqs to reduce KV demand".to_string(),
    }
}

/// True when `--max-num-seqs` is known and above the floor of 1.
fn seat_lever_available(snapshot: &RawSnapshot, config_max_num_seqs: Option<u32>) -> bool {
    snapshot
        .vllm
        .max_num_seqs
        .or(config_max_num_seqs)
        .is_some_and(|n| n > 1)
}

fn full_window_seat_bullet(
    snapshot: &RawSnapshot,
    kv_max_seqs: Option<u32>,
    config_max_num_seqs: Option<u32>,
) -> Option<String> {
    seat_lever_available(snapshot, config_max_num_seqs)
        .then(|| max_num_seqs_bullet(snapshot, kv_max_seqs))
}

/// Crisis-only risk subline on the full-window seat throttle. Attached when the
/// bullet is built; never inferred from printed text.
const CRISIS_THROTTLE_SUBLINE: &str = "Cuts throughput. Revert after pressure clears.";

pub(super) struct KvFormatCtx<'a> {
    pub snapshot: &'a RawSnapshot,
    pub max_model_len: Option<u32>,
    pub kv_headroom_gb: Option<f64>,
    pub kv_max_seqs: Option<u32>,
    /// Launch/config `--max-num-seqs` when the scrape gauge is absent.
    pub config_max_num_seqs: Option<u32>,
    pub capacity_label: KvCapacityLabel,
    pub fp8_compiler_available: bool,
    pub model: Option<&'a crate::context::ModelArch>,
    pub tp: Option<u32>,
    /// Effective KV dtype (runtime label, else launch config). Single source for
    /// fp8-switch advice and hypothesis capacity pricing; never re-read snapshot.
    pub kv_cache_dtype: Option<&'a str>,
}

impl<'a> KvFormatCtx<'a> {
    fn hyp_capacity(&self) -> super::HypCapacityCtx<'a> {
        super::HypCapacityCtx {
            cache: &self.snapshot.vllm.cache_config,
            kv_headroom_gb: self.kv_headroom_gb,
            model: self.model,
            kv_cache_dtype: self.kv_cache_dtype,
            tp: self.tp,
        }
    }
}

pub(super) fn format_kv_cache_pressure_fired(
    d: &KvCachePressureDetail,
    ctx: &KvFormatCtx<'_>,
    windows_fired: usize,
    total_evaluable: usize,
) -> Vec<String> {
    let hyp = ctx.hyp_capacity();
    let snapshot = ctx.snapshot;
    let max_model_len = ctx.max_model_len;
    let kv_headroom_gb = ctx.kv_headroom_gb;
    let kv_max_seqs = ctx.kv_max_seqs;
    let config_max_num_seqs = ctx.config_max_num_seqs;
    let capacity_label = ctx.capacity_label;
    let fp8_compiler_available = ctx.fp8_compiler_available;
    let kv_cache_dtype = ctx.kv_cache_dtype;
    let kv_avg = d.kv_cache_usage_perc;
    let peak = d.kv_peak_pct;
    let mut out = vec![
        "[!] KV Cache Pressure".to_string(),
        "    Cause:".to_string(),
    ];
    let avg_s = kv_avg
        .filter(|v| v.is_finite())
        .map(|v| format!("{v:.0}%"))
        .unwrap_or_else(|| "-".to_string());
    let peak_s = peak
        .filter(|v| v.is_finite())
        .map(|v| format!("{v:.0}%"))
        .unwrap_or_else(|| "-".to_string());
    let burst = kv_avg.is_none_or(|avg| avg < KV_CACHE_PRESSURE_MIN_PERC);
    if burst {
        out.push(format!(
            "      KV cache {avg_s} avg, {peak_s} peak (burst pressure, threshold: {:.0}%).",
            KV_CACHE_PRESSURE_MIN_PERC
        ));
    } else {
        out.push(format!(
            "      KV cache {avg_s} avg, {peak_s} peak (threshold: {:.0}%).",
            KV_CACHE_PRESSURE_MIN_PERC
        ));
    }
    let wait_count = snapshot.vllm.num_requests_waiting.filter(|v| v.is_finite());
    let evidence = match (
        d.preemptions_active,
        d.queue_backpressure.then_some(wait_count).flatten(),
    ) {
        (true, Some(w)) => {
            format!("      Scheduler evicting; {w:.0} requests queued on KV admission.")
        }
        (true, None) => "      Scheduler evicting sequences to free KV blocks.".to_string(),
        (false, Some(w)) => format!("      {w:.0} requests queued on KV admission."),
        (false, None) => {
            // Aggregate can keep queue_backpressure from earlier windows while the
            // landing snapshot's waiting gauge is gone. Never panic in display.
            "      Scheduler queueing requests; waiting count unavailable this window.".to_string()
        }
    };
    out.push(evidence);
    out.push(String::new());

    // Peak above floor(cap) → numberless seat line; bullet still renders.
    let contradicted = super::observed_kv_cap_contradicted(snapshot);
    // Cap-leads with Observed/Derived + m already name current max_model_len; use "to N".
    // Shrink-leads, DerivedHybrid, and no-capacity keep the arrow form.
    // Crisis follows the same form rules (no preemption force-off).
    let evidence = super::ShrinkEvidence::from_snapshot(snapshot);
    let would_lead_if_shrink = max_model_len.is_some_and(|m| {
        super::p99_sum_below_half_max_model_len(m, evidence.prompt_p99, evidence.generation_p99)
    });
    let seat_for_form = !contradicted && kv_max_seqs.is_some();
    let shrink_current_shown = !would_lead_if_shrink
        && seat_for_form
        && max_model_len.is_some()
        && matches!(
            capacity_label,
            KvCapacityLabel::Observed | KvCapacityLabel::Derived
        );
    let shrink = super::model_len_shrink_suggestion_lines(
        max_model_len,
        &evidence,
        "      ",
        shrink_current_shown,
    );
    // Lead with model-len when observed traffic fits in half the window: the
    // full-context concurrency floor is then a secondary bound, not the primary fix.
    // Composition: when p99s are missing, lead_with_shrink is false and ordering
    // falls back to seat-first under safe.
    let lead_with_shrink = !shrink.lines.is_empty() && would_lead_if_shrink;

    let mut safe = Vec::new();
    push_kv_pressure_safe_levers(
        &mut safe,
        snapshot,
        kv_headroom_gb,
        kv_cache_dtype,
        fp8_compiler_available,
    );

    // Lowering seats always reduces KV demand, down to 1. At 1 there is nothing
    // left to lower and the KV wall is hardware.
    let full_window_seat = full_window_seat_bullet(snapshot, kv_max_seqs, config_max_num_seqs);
    // Gate only: the projection's value is unused, since the bullet carries no
    // number. We offer the follow-on seat cut only when capacity at the new window
    // is computable, from observed page geometry (dense via block_size, hybrid via
    // mamba_block_size) or the catalog fallback. Unknown geometry means we cannot
    // say the shorter window admits more sequences, so we stay quiet.
    let follow_on_seat = if lead_with_shrink {
        shrink.target.and_then(|suggested| {
            super::capacity_at_hypothetical_max_len(suggested, max_model_len, &hyp)
                .map(|_| FOLLOW_ON_SEAT_BULLET.to_string())
        })
    } else {
        None
    };

    let mut cuts: Vec<super::CutBullet> = Vec::new();
    if lead_with_shrink {
        super::extend_with_shrink_suggestion(&mut cuts, shrink);
        if let Some(fo) = follow_on_seat {
            // Permanent seat at the new window, not a temporary throttle.
            cuts.push((fo, None));
        }
    } else {
        super::extend_with_shrink_suggestion(&mut cuts, shrink);
    }

    let has_other_fix = !safe.is_empty() || !cuts.is_empty();
    if !has_other_fix {
        if let Some(seat) = full_window_seat {
            cuts.push((
                seat,
                d.preemptions_active.then_some(CRISIS_THROTTLE_SUBLINE),
            ));
        }
    } else if !lead_with_shrink && let Some(seat) = full_window_seat {
        cuts.insert(
            0,
            (
                seat,
                d.preemptions_active.then_some(CRISIS_THROTTLE_SUBLINE),
            ),
        );
    }

    let has_fix = !safe.is_empty() || !cuts.is_empty();
    if !has_fix {
        push_dead_end_fixes(&mut safe);
    }

    if d.preemptions_active {
        // Crisis: flat Fix list, no group labels. Cuts before safe when shrink leads.
        let lead_with_cuts = lead_with_shrink;
        out.push("    Fix:".to_string());
        let emit_cuts = |out: &mut Vec<String>, cuts: Vec<super::CutBullet>| {
            for (bullet, sub) in cuts {
                super::push_bullet_with_subline(out, bullet, sub);
            }
        };
        super::trim_group_trailing_blanks(&mut safe);
        let safe_nonempty = !safe.is_empty();
        let cuts_nonempty = !cuts.is_empty();
        if lead_with_cuts {
            emit_cuts(&mut out, cuts);
            if cuts_nonempty && safe_nonempty {
                out.push(String::new());
            }
            out.extend(safe);
        } else {
            out.extend(safe);
            if safe_nonempty && cuts_nonempty {
                out.push(String::new());
            }
            emit_cuts(&mut out, cuts);
        }
    } else {
        let lead_with_cuts = lead_with_shrink;
        super::trim_group_trailing_blanks(&mut safe);
        super::push_grouped_fixes(&mut out, safe, cuts, Vec::new(), lead_with_cuts);
    }

    let expected = if d.preemptions_active {
        "    Expected: TTFT and TPOT recover once evictions stop."
    } else {
        "    Expected: Wait queue drains, TTFT recovers once KV pool has capacity."
    };
    super::trim_group_trailing_blanks(&mut out);
    out.push(String::new());
    out.push(expected.to_string());
    if super::rule_is_significant(windows_fired, total_evaluable) {
        let confidence = kv_pressure_confidence(windows_fired, total_evaluable);
        out.push(format!("    {}", kv_pressure_confidence_label(confidence)));
    }
    out
}

pub(super) fn format_kv_admission_backlog_issue(
    d: &KvAdmissionBacklogDetail,
    seen_pct: u32,
    ctx: &KvFormatCtx<'_>,
    windows_fired: usize,
    total_evaluable: usize,
) -> Vec<String> {
    let kv_cache_dtype = ctx.kv_cache_dtype;
    let mut out = vec![
        "[!] KV Cache Pressure: Admission Backlog".to_string(),
        "    Cause:".to_string(),
        format!(
            "      Scheduler holding {:.0} requests in queue ({:.0}% of active requests waiting) to protect KV memory.",
            d.requests_waiting,
            d.admission_ratio * 100.0
        ),
    ];
    let avg_s = format!("{:.0}%", d.kv_cache_usage_perc);
    let peak_s = d
        .kv_peak_pct
        .filter(|v| v.is_finite())
        .map(|v| format!("{v:.0}%"))
        .unwrap_or_else(|| "-".to_string());
    let burst = d.kv_cache_usage_perc < KV_CACHE_PRESSURE_MIN_PERC;
    if burst {
        out.push(format!(
            "      KV cache {avg_s} avg, {peak_s} peak (burst pressure, threshold: {:.0}%).",
            KV_CACHE_PRESSURE_MIN_PERC
        ));
    } else {
        out.push(format!(
            "      KV cache {avg_s} avg, {peak_s} peak (threshold: {:.0}%).",
            KV_CACHE_PRESSURE_MIN_PERC
        ));
    }
    out.push(format!(
        "      Free KV tokens: {:.0} available, {:.0} demanded (est, worst case).",
        d.free_kv_tokens, d.demand_tokens
    ));
    out.push(String::new());

    let evidence = super::ShrinkEvidence::from_snapshot(ctx.snapshot);
    let shrink =
        super::model_len_shrink_suggestion_lines(ctx.max_model_len, &evidence, "      ", false);

    let mut safe = Vec::new();
    push_kv_pressure_safe_levers(
        &mut safe,
        ctx.snapshot,
        ctx.kv_headroom_gb,
        kv_cache_dtype,
        ctx.fp8_compiler_available,
    );

    let mut cuts: Vec<super::CutBullet> = Vec::new();
    if let Some(seat) =
        full_window_seat_bullet(ctx.snapshot, ctx.kv_max_seqs, ctx.config_max_num_seqs)
    {
        cuts.push((seat, None));
    }
    super::extend_with_shrink_suggestion(&mut cuts, shrink);

    if safe.is_empty() && cuts.is_empty() {
        push_dead_end_fixes(&mut safe);
    }

    super::trim_group_trailing_blanks(&mut safe);
    super::push_grouped_fixes(&mut out, safe, cuts, Vec::new(), false);

    out.push(String::new());
    out.push("    Expected: Wait queue drains, TTFT recovers.".to_string());
    if super::rule_is_significant(windows_fired, total_evaluable) {
        let confidence = kv_pressure_confidence(windows_fired, total_evaluable);
        out.push(format!("    {}", kv_pressure_confidence_label(confidence)));
    }
    super::with_seen_pct(out, seen_pct)
}

pub(super) fn aggregate_backlog_detail(
    details: &[KvAdmissionBacklogDetail],
) -> KvAdmissionBacklogDetail {
    debug_assert!(
        !details.is_empty(),
        "aggregate_backlog_detail called with no fired windows - caller should gate on r2_backlog_significant"
    );
    let n = details.len() as f64;
    let kv = details.iter().map(|d| d.kv_cache_usage_perc).sum::<f64>() / n;
    let ratio = details.iter().map(|d| d.admission_ratio).sum::<f64>() / n;
    let wait = details.iter().map(|d| d.requests_waiting).sum::<f64>() / n;
    let run = details.iter().map(|d| d.requests_running).sum::<f64>() / n;
    let free_kv_tokens = details.iter().map(|d| d.free_kv_tokens).sum::<f64>() / n;
    let demand_tokens = details.iter().map(|d| d.demand_tokens).sum::<f64>() / n;
    let peak = details
        .iter()
        .filter_map(|d| d.kv_peak_pct)
        .chain(details.iter().map(|d| d.kv_cache_usage_perc))
        .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v))));
    KvAdmissionBacklogDetail {
        kv_cache_usage_perc: kv,
        kv_peak_pct: peak,
        admission_ratio: ratio,
        requests_waiting: wait,
        requests_running: run,
        free_kv_tokens,
        demand_tokens,
    }
}

pub(super) fn format_kv_cache_window_issue(
    d: &KvCachePressureDetail,
    seen_pct: u32,
    ctx: &KvFormatCtx<'_>,
    windows_fired: usize,
    total_evaluable: usize,
) -> Vec<String> {
    super::with_seen_pct(
        format_kv_cache_pressure_fired(d, ctx, windows_fired, total_evaluable),
        seen_pct,
    )
}

pub(super) fn aggregate_r2_detail(details: &[KvCachePressureDetail]) -> KvCachePressureDetail {
    debug_assert!(
        !details.is_empty(),
        "aggregate_r2_detail called with no fired windows - caller should gate on r2_significant"
    );
    let kv = super::mean_of_present(details.iter().filter_map(|d| d.kv_cache_usage_perc));
    let peak = details
        .iter()
        .filter_map(|d| d.kv_peak_pct)
        .chain(details.iter().filter_map(|d| d.kv_cache_usage_perc))
        .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v))));
    KvCachePressureDetail {
        kv_cache_usage_perc: kv,
        kv_peak_pct: peak,
        preemptions_active: details.iter().any(|d| d.preemptions_active),
        queue_backpressure: details.iter().any(|d| d.queue_backpressure),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{CacheConfigLabels, GpuRawMetrics, RawSnapshotFixture, VllmRawMetrics};

    fn snap(vllm: VllmRawMetrics) -> RawSnapshot {
        crate::collectors::snap_vllm(vllm)
    }

    /// Snapshot with observed VRAM for gpu-memory-utilization gating tests.
    fn snap_vram(vllm: VllmRawMetrics, used_mb: u64, total_mb: u64) -> RawSnapshot {
        RawSnapshotFixture::default()
            .vllm(vllm)
            .gpus(vec![GpuRawMetrics {
                vram_used_mb: Some(used_mb),
                vram_total_mb: Some(total_mb),
                ..Default::default()
            }])
            .build()
    }

    const VRAM_AMPLY_FREE_USED_MB: u64 = 40 * 1024;
    const VRAM_AMPLY_FREE_TOTAL_MB: u64 = 80 * 1024;
    const VRAM_ITER3_USED_MB: u64 = 78 * 1024;
    const VRAM_ITER3_TOTAL_MB: u64 = 80 * 1024;

    fn kv_ctx(
        snapshot: &RawSnapshot,
        max_model_len: Option<u32>,
        kv_headroom_gb: Option<f64>,
        kv_max_seqs: Option<u32>,
    ) -> KvFormatCtx<'_> {
        let config_max_num_seqs = snapshot.vllm.max_num_seqs.or(Some(256));
        kv_ctx_config(
            snapshot,
            max_model_len,
            kv_headroom_gb,
            kv_max_seqs,
            config_max_num_seqs,
        )
    }

    fn kv_ctx_config(
        snapshot: &RawSnapshot,
        max_model_len: Option<u32>,
        kv_headroom_gb: Option<f64>,
        kv_max_seqs: Option<u32>,
        config_max_num_seqs: Option<u32>,
    ) -> KvFormatCtx<'_> {
        KvFormatCtx {
            snapshot,
            max_model_len,
            kv_headroom_gb,
            kv_max_seqs,
            config_max_num_seqs,
            capacity_label: KvCapacityLabel::Derived,
            fp8_compiler_available: false,
            model: None,
            tp: None,
            kv_cache_dtype: snapshot.vllm.cache_config.cache_dtype.as_deref(),
        }
    }

    fn assert_dead_end_pair(text: &str) {
        assert!(text.contains("    Fix:"));
        assert!(text.contains(DEAD_END_VERIFY_BULLET.trim()));
        assert!(text.contains(DEAD_END_VERIFY_SUBLINE));
        assert!(text.contains("Add a replica to scale out."));
        assert!(text.contains("No config change on this GPU moves the KV wall."));
        assert!(text.contains("Expected:"));
    }

    fn assert_no_dead_end_pair(text: &str) {
        assert!(!text.contains("took effect"));
        assert!(!text.contains("No config change on this GPU moves the KV wall."));
    }

    fn format_seat_lever_crisis(
        snap_max_num_seqs: Option<u32>,
        config_max_num_seqs: Option<u32>,
    ) -> String {
        use crate::collectors::KvOffloadState;
        let (snap, m) = dead_end_snap(
            KvOffloadState::Enabled(16.0),
            10000,
            5000.0,
            4600.0,
            snap_max_num_seqs,
        );
        format_kv_cache_pressure_fired(
            &detail(98.0, true),
            &kv_ctx_config(&snap, Some(m), None, None, config_max_num_seqs),
            3,
            4,
        )
        .join("\n")
    }

    fn format_seat_lever_non_crisis(
        snap_max_num_seqs: Option<u32>,
        config_max_num_seqs: Option<u32>,
    ) -> String {
        use crate::collectors::KvOffloadState;
        let (snap, m) = dead_end_snap(
            KvOffloadState::Enabled(16.0),
            10000,
            5000.0,
            4600.0,
            snap_max_num_seqs,
        );
        format_kv_cache_pressure_fired(
            &KvCachePressureDetail {
                kv_cache_usage_perc: Some(98.0),
                kv_peak_pct: Some(98.0),
                preemptions_active: false,
                queue_backpressure: true,
            },
            &kv_ctx_config(&snap, Some(m), None, None, config_max_num_seqs),
            3,
            4,
        )
        .join("\n")
    }

    fn format_seat_lever_backlog(
        snap_max_num_seqs: Option<u32>,
        config_max_num_seqs: Option<u32>,
    ) -> String {
        use crate::collectors::KvOffloadState;
        let (snap, m) = dead_end_snap(
            KvOffloadState::Enabled(16.0),
            10000,
            5000.0,
            4600.0,
            snap_max_num_seqs,
        );
        format_kv_admission_backlog_issue(
            &sample_backlog_detail(),
            50,
            &kv_ctx_config(&snap, Some(m), None, None, config_max_num_seqs),
            3,
            4,
        )
        .join("\n")
    }

    fn backlog_vllm(
        kv: f64,
        wait: f64,
        run: f64,
        prompt_mean: f64,
        num_gpu_blocks: Option<u32>,
        block_size: Option<u32>,
    ) -> VllmRawMetrics {
        // max_num_seqs set well above run so concurrency cap doesn't suppress the rule.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let max_num_seqs = Some((run as u32) + 100);
        VllmRawMetrics {
            kv_cache_usage_perc: Some(kv),
            num_requests_waiting: Some(wait),
            num_requests_running: Some(run),
            prompt_tokens_mean: Some(prompt_mean),
            generation_tokens_per_sec: Some(100.0),
            max_num_seqs,
            cache_config: CacheConfigLabels {
                block_size,
                num_gpu_blocks,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn backlog_declines_when_kv_below_pressure_bar() {
        assert!(
            rule2_kv_admission_backlog(&snap(backlog_vllm(
                40.0,
                10.0,
                5.0,
                20.0,
                Some(100),
                Some(16),
            )))
            .is_none()
        );
    }

    #[test]
    fn backlog_fires_when_free_below_demand_and_ratio_at_least_0_30() {
        // 100 blocks × 16 tok/block × 10% free = 160 free; 10 wait × 20 tok = 200 demand
        let d = rule2_kv_admission_backlog(&snap(backlog_vllm(
            90.0,
            10.0,
            5.0,
            20.0,
            Some(100),
            Some(16),
        )))
        .expect("fired");
        assert!((d.free_kv_tokens - 160.0).abs() < 1e-9);
        assert!((d.demand_tokens - 200.0).abs() < 1e-9);
        assert!((d.admission_ratio - (10.0 / 15.0)).abs() < 1e-9);
    }

    #[test]
    fn backlog_silent_when_free_at_least_demand() {
        // 10% KV used → 90% free pool; demand is small
        assert!(
            rule2_kv_admission_backlog(&snap(backlog_vllm(
                10.0,
                5.0,
                5.0,
                100.0,
                Some(1000),
                Some(16),
            )))
            .is_none()
        );
    }

    #[test]
    fn backlog_silent_when_required_field_missing() {
        assert!(
            rule2_kv_admission_backlog(&snap(backlog_vllm(90.0, 10.0, 5.0, 20.0, None, Some(16))))
                .is_none()
        );
        assert!(
            rule2_kv_admission_backlog(&snap(backlog_vllm(90.0, 10.0, 5.0, 20.0, Some(100), None)))
                .is_none()
        );
        assert!(
            rule2_kv_admission_backlog(&snap(backlog_vllm(
                90.0,
                10.0,
                5.0,
                f64::NAN,
                Some(100),
                Some(16)
            )))
            .is_none()
        );
        let mut v = backlog_vllm(90.0, 10.0, 5.0, 20.0, Some(100), Some(16));
        v.max_num_seqs = None;
        assert!(rule2_kv_admission_backlog(&snap(v)).is_none());
    }

    #[test]
    fn backlog_silent_when_at_concurrency_cap() {
        // run == max_num_seqs → concurrency cap is the cause, not KV. Must stay silent
        // even though physics gate would fire (free=160 < demand=200).
        let mut v = backlog_vllm(90.0, 10.0, 5.0, 20.0, Some(100), Some(16));
        v.max_num_seqs = Some(5);
        assert!(rule2_kv_admission_backlog(&snap(v)).is_none());
    }

    #[test]
    fn backlog_silent_when_ratio_below_0_30() {
        assert!(
            rule2_kv_admission_backlog(&snap(backlog_vllm(
                90.0,
                2.0,
                8.0,
                20.0,
                Some(100),
                Some(16),
            )))
            .is_none()
        );
    }

    fn detail(kv: f64, preemptions: bool) -> KvCachePressureDetail {
        KvCachePressureDetail {
            kv_cache_usage_perc: Some(kv),
            kv_peak_pct: Some(kv),
            preemptions_active: preemptions,
            queue_backpressure: false,
        }
    }

    #[test]
    fn kv_pressure_confidence_is_duration_density() {
        assert!((kv_pressure_confidence(4, 15) - (4.0 / 15.0)).abs() < 1e-9);
        assert!((kv_pressure_confidence(0, 15) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn kv_pressure_confidence_label_maps_density() {
        assert_eq!(kv_pressure_confidence_label(0.4), "Confidence: Medium");
        assert_eq!(kv_pressure_confidence_label(0.5), "Confidence: Medium-High");
        assert_eq!(kv_pressure_confidence_label(0.76), "Confidence: High");
    }

    #[test]
    fn kv_pressure_omits_confidence_until_significant() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            ..Default::default()
        };
        let s = snap(v.clone());
        let single = format_kv_cache_pressure_fired(
            &detail(90.0, true),
            &kv_ctx(&s, None, None, None),
            1,
            1,
        )
        .join("\n");
        assert!(!single.contains("Confidence:"));
        let stable = format_kv_cache_pressure_fired(
            &detail(90.0, true),
            &kv_ctx(&snap(v), None, None, None),
            3,
            4,
        )
        .join("\n");
        assert!(stable.contains("Confidence: Medium-High"));
    }

    #[test]
    fn swapped_requires_at_least_two() {
        let base = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.0),
            ..Default::default()
        };
        let mut one = base.clone();
        one.num_requests_swapped = Some(1.0);
        assert!(matches!(
            rule2_kv_cache_pressure(&snap(one)),
            Rule2Outcome::NotFired
        ));
        let mut two = base;
        two.num_requests_swapped = Some(2.0);
        assert!(matches!(
            rule2_kv_cache_pressure(&snap(two)),
            Rule2Outcome::Fired(_)
        ));
    }

    #[test]
    fn preemption_rate_requires_above_0_02() {
        let mut v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.01),
            ..Default::default()
        };
        assert!(matches!(
            rule2_kv_cache_pressure(&snap(v.clone())),
            Rule2Outcome::NotFired
        ));
        v.num_preemptions_per_sec = Some(0.03);
        assert!(matches!(
            rule2_kv_cache_pressure(&snap(v)),
            Rule2Outcome::Fired(_)
        ));
    }

    #[test]
    fn queue_backpressure_requires_more_than_two_waiting() {
        let v_one_waiting = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_running: Some(10.0),
            num_requests_waiting: Some(1.0),
            ..Default::default()
        };
        assert!(matches!(
            rule2_kv_cache_pressure(&snap(v_one_waiting)),
            Rule2Outcome::NotFired
        ));
        let v_two_waiting = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_running: Some(10.0),
            num_requests_waiting: Some(2.0),
            ..Default::default()
        };
        assert!(matches!(
            rule2_kv_cache_pressure(&snap(v_two_waiting)),
            Rule2Outcome::NotFired
        ));
        let v_three_waiting = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_running: Some(10.0),
            num_requests_waiting: Some(3.0),
            ..Default::default()
        };
        match rule2_kv_cache_pressure(&snap(v_three_waiting)) {
            Rule2Outcome::Fired(d) => assert!(d.queue_backpressure),
            Rule2Outcome::NotFired => panic!("expected fired with queue backpressure"),
        }
    }

    #[test]
    fn high_kv_without_stress_does_not_fire() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(95.0),
            ..Default::default()
        };
        assert!(matches!(
            rule2_kv_cache_pressure(&snap(v)),
            Rule2Outcome::NotFired
        ));
    }

    #[test]
    fn queue_only_fire_shows_gpu_mem_fix() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_running: Some(10.0),
            num_requests_waiting: Some(5.0),
            num_preemptions_per_sec: Some(0.0),
            generation_tokens_per_sec: Some(100.0),
            ..Default::default()
        };
        let snapshot = snap_vram(v.clone(), VRAM_AMPLY_FREE_USED_MB, VRAM_AMPLY_FREE_TOTAL_MB);
        let r = r2_recommendation(R2RecommendationInput {
            snapshot: &snapshot,
            max_model_len: None,
            kv_headroom_gb: Some(30.0),
            kv_max_seqs: None,
            capacity_label: KvCapacityLabel::Derived,
            windows_fired: 1,
            total_evaluable: 1,
            fp8_compiler_available: false,
        })
        .expect("fired");
        let text = r.display_lines.join("\n");
        assert!(!text.contains("evictions stop"));
        assert!(text.contains("Raise --gpu-memory-utilization"));
        assert!(matches!(
            rule2_kv_cache_pressure(&snap(v)),
            Rule2Outcome::Fired(d) if !d.preemptions_active && d.queue_backpressure
        ));
    }

    #[test]
    fn backlog_display_includes_ceiling_and_max_model_len() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_running: Some(10.0),
            num_requests_waiting: Some(5.0),
            num_preemptions_per_sec: Some(0.0),
            generation_tokens_per_sec: Some(100.0),
            ..Default::default()
        };
        let r = r2_recommendation(R2RecommendationInput {
            snapshot: &snap(v),
            max_model_len: Some(8192),
            kv_headroom_gb: Some(10.0),
            kv_max_seqs: Some(14),
            capacity_label: KvCapacityLabel::Derived,
            windows_fired: 1,
            total_evaluable: 1,
            fp8_compiler_available: false,
        })
        .expect("fired");
        let text = r.display_lines.join("\n");
        assert!(text.contains("Lower --max-num-seqs to reduce KV demand"));
        assert!(!text.contains('≤'));
    }

    #[test]
    fn display_includes_max_model_len_when_ceiling_known() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            generation_tokens_per_sec: Some(100.0),
            ..Default::default()
        };
        let r = r2_recommendation(R2RecommendationInput {
            snapshot: &snap(v),
            max_model_len: Some(8192),
            kv_headroom_gb: None,
            kv_max_seqs: Some(15),
            capacity_label: KvCapacityLabel::Derived,
            windows_fired: 1,
            total_evaluable: 4,
            fp8_compiler_available: false,
        })
        .expect("fired");
        let text = r.display_lines.join("\n");
        assert!(text.contains("Lower --max-num-seqs to reduce KV demand"));
        assert!(!text.contains('≤'));
    }

    #[test]
    fn display_includes_ceiling_when_kv_max_seqs_known() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            generation_tokens_per_sec: Some(100.0),
            ..Default::default()
        };
        let r = r2_recommendation(R2RecommendationInput {
            snapshot: &snap(v),
            max_model_len: None,
            kv_headroom_gb: None,
            kv_max_seqs: Some(18),
            capacity_label: KvCapacityLabel::Derived,
            windows_fired: 1,
            total_evaluable: 4,
            fp8_compiler_available: false,
        })
        .expect("fired");
        let text = r.display_lines.join("\n");
        assert!(text.contains("Lower --max-num-seqs to reduce KV demand"));
        assert!(!text.contains('≤'));
    }

    #[test]
    fn kv_pressure_preemption_fix_matches_spec() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            generation_tokens_per_sec: Some(100.0),
            ..Default::default()
        };
        let r = r2_recommendation(R2RecommendationInput {
            snapshot: &snap(v),
            max_model_len: None,
            kv_headroom_gb: None,
            kv_max_seqs: None,
            capacity_label: KvCapacityLabel::Derived,
            windows_fired: 1,
            total_evaluable: 4,
            fp8_compiler_available: false,
        })
        .expect("fired");
        let text = r.display_lines.join("\n");
        assert!(text.contains("Lower --max-num-seqs to reduce KV demand"));
        assert!((r.confidence - 0.5).abs() < 1e-9);
    }

    #[test]
    fn model_len_shown_in_queue_backpressure_path() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            prompt_tokens_p99: Some(6000.0),
            generation_tokens_p99: Some(450.0),
            generation_tokens_completed: Some(150.0),
            ..Default::default()
        };
        let d = KvCachePressureDetail {
            kv_cache_usage_perc: Some(90.0),
            kv_peak_pct: Some(90.0),
            preemptions_active: false,
            queue_backpressure: true,
        };
        let text =
            format_kv_cache_pressure_fired(&d, &kv_ctx(&snap(v), Some(8192), None, None), 3, 4)
                .join("\n");
        assert!(text.contains("Lower --max-model-len 8192 → 6450"));
        assert!(text.contains("rejected with a 400"));
    }

    #[test]
    fn shrink_suggestion_uses_p99_sum_when_count_sufficient() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            prompt_tokens_p99: Some(6000.0),
            generation_tokens_p99: Some(450.0),
            generation_tokens_completed: Some(150.0),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(
            &detail(90.0, true),
            &kv_ctx(&snap(v), Some(8192), None, None),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("Lower --max-model-len 8192 → 6450"));
        assert!(text.contains("rejected with a 400"));
    }

    #[test]
    fn model_len_prescribed_with_rejection_warning_when_ceiling_unknown() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(
            &detail(90.0, true),
            &kv_ctx(&snap(v), None, None, None),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("Lower --max-model-len to safely raise concurrency."));
        assert!(text.contains("rejected with a 400, not truncated."));
        assert!(!text.contains("Verify: check the vLLM start command"));
    }

    #[test]
    fn model_len_in_evictions_path_when_ceiling_known() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(
            &detail(90.0, true),
            &kv_ctx(&snap(v), Some(4096), None, Some(16)),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("Lower --max-num-seqs to reduce KV demand"));
        assert!(!text.contains('≤'));
        assert!(!text.contains("worst-case"));
    }

    fn sample_backlog_detail() -> KvAdmissionBacklogDetail {
        KvAdmissionBacklogDetail {
            kv_cache_usage_perc: 90.0,
            kv_peak_pct: Some(90.0),
            admission_ratio: 0.4,
            requests_waiting: 10.0,
            requests_running: 15.0,
            free_kv_tokens: 160.0,
            demand_tokens: 200.0,
        }
    }

    #[test]
    fn backlog_cause_line_includes_kv_percentage() {
        let text = format_kv_admission_backlog_issue(
            &sample_backlog_detail(),
            27,
            &kv_ctx(&snap(VllmRawMetrics::default()), None, Some(30.0), None),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("KV cache 90% avg, 90% peak (threshold: 88%)."));
        assert!(text.contains("Free KV tokens: 160 available, 200 demanded (est, worst case)."));
        assert!(!text.contains("threshold: 88%). Free KV tokens"));
        assert!(!text.contains("burst pressure"));
    }

    #[test]
    fn backlog_cause_line_names_burst_when_avg_below_bar() {
        let mut d = sample_backlog_detail();
        d.kv_cache_usage_perc = 71.0;
        d.kv_peak_pct = Some(92.0);
        let text = format_kv_admission_backlog_issue(
            &d,
            27,
            &kv_ctx(&snap(VllmRawMetrics::default()), None, Some(30.0), None),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("KV cache 71% avg, 92% peak (burst pressure, threshold: 88%)."));
        assert!(text.contains("Free KV tokens: 160 available, 200 demanded (est, worst case)."));
        assert!(!text.contains("threshold: 88%). Free KV tokens"));
    }

    #[test]
    fn backlog_shows_headroom_when_safe() {
        let text = format_kv_admission_backlog_issue(
            &sample_backlog_detail(),
            27,
            &kv_ctx(
                &snap_vram(
                    VllmRawMetrics::default(),
                    VRAM_AMPLY_FREE_USED_MB,
                    VRAM_AMPLY_FREE_TOTAL_MB,
                ),
                None,
                Some(30.0),
                None,
            ),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("check vRAM header for avail mem"));
    }

    #[test]
    fn backlog_omits_gpu_mem_when_observed_vram_low() {
        let text = format_kv_admission_backlog_issue(
            &sample_backlog_detail(),
            27,
            &kv_ctx(
                &snap_vram(
                    VllmRawMetrics::default(),
                    VRAM_ITER3_USED_MB,
                    VRAM_ITER3_TOTAL_MB,
                ),
                None,
                Some(30.0),
                None,
            ),
            3,
            4,
        )
        .join("\n");
        assert!(!text.contains("Raise --gpu-memory-utilization"));
        assert!(!text.contains("GPU at VRAM capacity"));
    }

    #[test]
    fn backlog_omits_gpu_mem_when_vram_unreadable() {
        let text = format_kv_admission_backlog_issue(
            &sample_backlog_detail(),
            27,
            &kv_ctx(&snap(VllmRawMetrics::default()), None, Some(30.0), None),
            3,
            4,
        )
        .join("\n");
        assert!(!text.contains("Raise --gpu-memory-utilization"));
    }

    #[test]
    fn backlog_omits_confidence_until_significant() {
        let d = sample_backlog_detail();
        let snap = snap(VllmRawMetrics::default());
        let ctx = kv_ctx(&snap, None, Some(30.0), None);
        let single = format_kv_admission_backlog_issue(&d, 27, &ctx, 1, 1).join("\n");
        assert!(!single.contains("Confidence:"));
        let stable = format_kv_admission_backlog_issue(&d, 27, &ctx, 3, 4).join("\n");
        assert!(stable.contains("Confidence: Medium-High"));
    }

    #[test]
    fn admission_backlog_shows_shrink_suggestion_when_p99_known() {
        let v = VllmRawMetrics {
            prompt_tokens_p99: Some(6000.0),
            generation_tokens_p99: Some(450.0),
            generation_tokens_completed: Some(150.0),
            ..Default::default()
        };
        let lines = format_kv_admission_backlog_issue(
            &sample_backlog_detail(),
            27,
            &kv_ctx(&snap(v), Some(8192), Some(30.0), None),
            3,
            4,
        )
        .join("\n");
        assert!(lines.contains("Lower --max-model-len 8192 → 6450"));
        assert!(lines.contains("rejected with a 400"));
    }

    #[test]
    fn queue_backpressure_only_expected_does_not_mention_evictions() {
        let d = KvCachePressureDetail {
            kv_cache_usage_perc: Some(90.0),
            kv_peak_pct: Some(90.0),
            preemptions_active: false,
            queue_backpressure: true,
        };
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(&d, &kv_ctx(&snap(v), None, None, None), 1, 1)
            .join("\n");
        assert!(text.contains("Wait queue drains"));
        assert!(!text.contains("evictions stop"));
    }

    #[test]
    fn queue_backpressure_missing_waiting_on_landing_does_not_panic() {
        // Fired windows had queue pressure; landing snapshot lost the waiting gauge.
        let d = KvCachePressureDetail {
            kv_cache_usage_perc: Some(90.0),
            kv_peak_pct: Some(92.0),
            preemptions_active: false,
            queue_backpressure: true,
        };
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: None,
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(&d, &kv_ctx(&snap(v), None, None, None), 1, 1)
            .join("\n");
        assert!(
            text.contains("Scheduler queueing requests; waiting count unavailable this window.")
        );
    }

    #[test]
    fn queue_backpressure_suggests_max_num_seqs_from_running_count() {
        let d = KvCachePressureDetail {
            kv_cache_usage_perc: Some(90.0),
            kv_peak_pct: Some(90.0),
            preemptions_active: false,
            queue_backpressure: true,
        };
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_running: Some(93.0),
            num_requests_waiting: Some(5.0),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(&d, &kv_ctx(&snap(v), None, None, None), 1, 1)
            .join("\n");
        assert!(text.contains("Lower --max-num-seqs to reduce KV demand"));
    }

    #[test]
    fn queue_backpressure_warns_when_vram_full() {
        let d = KvCachePressureDetail {
            kv_cache_usage_perc: Some(90.0),
            kv_peak_pct: Some(90.0),
            preemptions_active: false,
            queue_backpressure: true,
        };
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(
            &d,
            &kv_ctx(
                &snap_vram(v, VRAM_ITER3_USED_MB, VRAM_ITER3_TOTAL_MB),
                None,
                Some(1.0),
                None,
            ),
            1,
            1,
        )
        .join("\n");
        assert!(!text.contains("Raise --gpu-memory-utilization"));
        assert!(!text.contains("GPU at VRAM capacity"));
        assert!(!text.contains("max context length"));
        assert!(!text.contains("30GB VRAM available"));
    }

    #[test]
    fn evictions_path_shows_raise_gpu_mem_bullet_when_headroom_safe() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(
            &detail(90.0, true),
            &kv_ctx(
                &snap_vram(v, VRAM_AMPLY_FREE_USED_MB, VRAM_AMPLY_FREE_TOTAL_MB),
                None,
                Some(30.0),
                None,
            ),
            1,
            1,
        )
        .join("\n");
        assert!(text.contains(
            "Raise --gpu-memory-utilization (check vRAM header for avail mem) to expand KV pool"
        ));
        assert!(!text.contains("Once stable"));
    }

    #[test]
    fn evictions_path_omits_gpu_mem_when_headroom_below_safe() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(
            &detail(90.0, true),
            &kv_ctx(&snap(v), None, Some(1.0), None),
            1,
            1,
        )
        .join("\n");
        assert!(!text.contains("Raise --gpu-memory-utilization"));
        assert!(!text.contains("GPU at VRAM capacity"));
        assert!(!text.contains("Once stable"));
    }

    #[test]
    fn queue_backpressure_omits_gpu_mem_when_vram_unreadable() {
        let d = KvCachePressureDetail {
            kv_cache_usage_perc: Some(90.0),
            kv_peak_pct: Some(90.0),
            preemptions_active: false,
            queue_backpressure: true,
        };
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(&d, &kv_ctx(&snap(v), None, None, None), 1, 1)
            .join("\n");
        assert!(!text.contains("Raise --gpu-memory-utilization"));
    }

    #[test]
    fn prefix_caching_bullet_when_long_prompts_and_caching_off() {
        let d = detail(90.0, true);
        let mut v_long = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            prompt_tokens_mean: Some(250.0),
            cache_config: CacheConfigLabels {
                enable_prefix_caching: Some(false),
                ..Default::default()
            },
            ..Default::default()
        };
        let with_bullet = format_kv_cache_pressure_fired(
            &d,
            &kv_ctx(&snap(v_long.clone()), None, None, None),
            1,
            1,
        )
        .join("\n");
        assert!(with_bullet.contains("Enable --enable-prefix-caching"));

        v_long.prompt_tokens_mean = Some(150.0);
        let without_bullet =
            format_kv_cache_pressure_fired(&d, &kv_ctx(&snap(v_long), None, None, None), 1, 1)
                .join("\n");
        assert!(!without_bullet.contains("Enable --enable-prefix-caching"));
    }

    #[test]
    fn fp8_kv_cache_bullet_reflects_compiler_availability() {
        let with_compiler =
            fp8_kv_cache_fix_bullet(None, true).expect("bf16/auto should suggest fp8");
        assert!(with_compiler.contains("Switch --kv-cache-dtype fp8"));
        assert!(with_compiler.contains("(affects output quality)"));
        assert!(!with_compiler.contains("FP8 compiler not found"));
        let without_compiler =
            fp8_kv_cache_fix_bullet(None, false).expect("bf16/auto should suggest fp8");
        assert!(without_compiler.contains("(affects output quality; FP8 compiler not found)"));
    }

    #[test]
    fn fp8_kv_cache_bullet_suppressed_when_already_fp8() {
        assert!(fp8_kv_cache_fix_bullet(Some("fp8"), true).is_none());
        assert!(fp8_kv_cache_fix_bullet(Some("FP8"), true).is_none());
        assert!(fp8_kv_cache_fix_bullet(Some("e4m3fnuz"), true).is_none());
        assert!(fp8_kv_cache_fix_bullet(Some("e5m2"), true).is_none());
        assert!(fp8_kv_cache_fix_bullet(Some("auto"), true).is_some());
    }

    #[test]
    fn fp8_kv_cache_bullet_uses_resolved_kv_bytes() {
        assert!(
            fp8_kv_cache_fix_bullet(Some("auto"), true).is_some(),
            "auto uses activation dtype (2 bytes); fp8 KV still helps"
        );
    }

    #[test]
    fn effective_fp8_suppresses_switch_bullet_pressure_and_backlog() {
        use super::super::{HypCapacityCtx, capacity_at_hypothetical_max_len};
        use crate::context::ModelArch;

        let model = ModelArch {
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_layers: Some(32),
            param_count: Some(7_000_000_000),
            default_weight_dtype: Some("bf16".to_string()),
            ..Default::default()
        };
        let pressure = KvCachePressureDetail {
            kv_cache_usage_perc: Some(90.0),
            kv_peak_pct: Some(90.0),
            preemptions_active: false,
            queue_backpressure: true,
        };
        let backlog = sample_backlog_detail();
        let headroom = Some(20.0_f64);
        let max_len = Some(8192_u32);

        // config fp8 + runtime None (eval fills effective dtype onto ctx)
        let snap_config = snap(VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            ..Default::default()
        });
        let mut ctx_config = kv_ctx(&snap_config, max_len, headroom, None);
        ctx_config.kv_cache_dtype = Some("fp8");
        ctx_config.fp8_compiler_available = true;
        ctx_config.model = Some(&model);
        let pressure_config =
            format_kv_cache_pressure_fired(&pressure, &ctx_config, 3, 4).join("\n");
        let backlog_config =
            format_kv_admission_backlog_issue(&backlog, 75, &ctx_config, 3, 4).join("\n");
        assert!(
            !pressure_config.contains("Switch --kv-cache-dtype fp8"),
            "config-only fp8 must suppress pressure bullet"
        );
        assert!(
            !backlog_config.contains("Switch --kv-cache-dtype fp8"),
            "config-only fp8 must suppress backlog bullet"
        );
        let hyp_fp8 = ctx_config.hyp_capacity();
        let cap_fp8 = capacity_at_hypothetical_max_len(4096, max_len, &hyp_fp8);
        let hyp_bf16 = HypCapacityCtx {
            cache: hyp_fp8.cache,
            kv_headroom_gb: hyp_fp8.kv_headroom_gb,
            model: hyp_fp8.model,
            kv_cache_dtype: Some("bf16"),
            tp: hyp_fp8.tp,
        };
        let cap_bf16 = capacity_at_hypothetical_max_len(4096, max_len, &hyp_bf16);
        let cap_fp8 = cap_fp8.expect("fp8 capacity");
        assert_eq!(
            cap_fp8,
            cap_bf16.expect("bf16 capacity") * 2,
            "hypothesis capacity must price config fp8 at 1 byte"
        );

        // runtime fp8 alone (regression)
        let snap_runtime = snap(VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            cache_config: CacheConfigLabels {
                cache_dtype: Some("fp8".to_string()),
                ..Default::default()
            },
            ..Default::default()
        });
        let mut ctx_runtime = kv_ctx(&snap_runtime, max_len, headroom, None);
        ctx_runtime.fp8_compiler_available = true;
        ctx_runtime.model = Some(&model);
        assert_eq!(ctx_runtime.kv_cache_dtype, Some("fp8"));
        let pressure_runtime =
            format_kv_cache_pressure_fired(&pressure, &ctx_runtime, 3, 4).join("\n");
        let backlog_runtime =
            format_kv_admission_backlog_issue(&backlog, 75, &ctx_runtime, 3, 4).join("\n");
        assert!(
            !pressure_runtime.contains("Switch --kv-cache-dtype fp8"),
            "runtime fp8 must suppress pressure bullet"
        );
        assert!(
            !backlog_runtime.contains("Switch --kv-cache-dtype fp8"),
            "runtime fp8 must suppress backlog bullet"
        );
        let hyp_rt = ctx_runtime.hyp_capacity();
        assert_eq!(
            capacity_at_hypothetical_max_len(4096, max_len, &hyp_rt).expect("runtime fp8 cap"),
            cap_fp8
        );

        // runtime bf16 + config fp8 → runtime wins; switch-to-fp8 bullet offered
        let snap_runtime_bf16 = snap(VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            cache_config: CacheConfigLabels {
                cache_dtype: Some("bf16".to_string()),
                ..Default::default()
            },
            ..Default::default()
        });
        let mut ctx_rt_bf16 = kv_ctx(&snap_runtime_bf16, max_len, headroom, None);
        // eval would set effective_kv_cache_dtype(Some("bf16"), Some("fp8")) → bf16
        ctx_rt_bf16.kv_cache_dtype =
            crate::engine::baseline::effective_kv_cache_dtype(Some("bf16"), Some("fp8"));
        ctx_rt_bf16.fp8_compiler_available = true;
        ctx_rt_bf16.model = Some(&model);
        assert_eq!(ctx_rt_bf16.kv_cache_dtype, Some("bf16"));
        let pressure_bf16 =
            format_kv_cache_pressure_fired(&pressure, &ctx_rt_bf16, 3, 4).join("\n");
        let backlog_bf16 =
            format_kv_admission_backlog_issue(&backlog, 75, &ctx_rt_bf16, 3, 4).join("\n");
        assert!(
            pressure_bf16.contains("Switch --kv-cache-dtype fp8"),
            "runtime bf16 must still offer fp8 switch on pressure"
        );
        assert!(
            backlog_bf16.contains("Switch --kv-cache-dtype fp8"),
            "runtime bf16 must still offer fp8 switch on backlog"
        );
    }

    #[test]
    fn peak_fires_when_avg_below_threshold() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(58.0),
            kv_cache_peak_perc: Some(93.0),
            num_preemptions_per_sec: Some(0.05),
            ..Default::default()
        };
        match rule2_kv_cache_pressure(&snap(v)) {
            Rule2Outcome::Fired(d) => {
                assert!((d.kv_cache_usage_perc.unwrap() - 58.0).abs() < 1e-9);
                assert_eq!(d.kv_peak_pct, Some(93.0));
            }
            Rule2Outcome::NotFired => panic!("expected fired on peak >= 88%"),
        }
    }

    #[test]
    fn peak_alone_without_corroboration_does_not_fire() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(58.0),
            kv_cache_peak_perc: Some(95.0),
            ..Default::default()
        };
        assert!(matches!(
            rule2_kv_cache_pressure(&snap(v)),
            Rule2Outcome::NotFired
        ));
    }

    #[test]
    fn display_shows_burst_pressure_when_peak_triggered() {
        let d = KvCachePressureDetail {
            kv_cache_usage_perc: Some(58.0),
            kv_peak_pct: Some(93.0),
            preemptions_active: true,
            queue_backpressure: false,
        };
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(58.0),
            kv_cache_peak_perc: Some(93.0),
            num_preemptions_per_sec: Some(0.05),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(&d, &kv_ctx(&snap(v), None, None, None), 1, 1)
            .join("\n");
        assert!(text.contains("burst pressure"));
        assert!(text.contains("58% avg, 93% peak"));
        assert!(text.contains("Scheduler evicting"));
    }

    #[test]
    fn display_peak_only_renders_dash_avg() {
        let d = KvCachePressureDetail {
            kv_cache_usage_perc: None,
            kv_peak_pct: Some(93.0),
            preemptions_active: true,
            queue_backpressure: false,
        };
        let v = VllmRawMetrics {
            kv_cache_usage_perc: None,
            kv_cache_peak_perc: Some(93.0),
            num_preemptions_per_sec: Some(0.05),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(&d, &kv_ctx(&snap(v), None, None, None), 1, 1)
            .join("\n");
        assert!(text.contains("- avg, 93% peak"));
        assert!(!text.contains("0% avg"));
    }

    #[test]
    fn aggregate_excludes_missing_avg() {
        let details = [
            KvCachePressureDetail {
                kv_cache_usage_perc: None,
                kv_peak_pct: Some(95.0),
                preemptions_active: true,
                queue_backpressure: false,
            },
            KvCachePressureDetail {
                kv_cache_usage_perc: Some(90.0),
                kv_peak_pct: Some(92.0),
                preemptions_active: true,
                queue_backpressure: false,
            },
        ];
        let agg = aggregate_r2_detail(&details);
        assert_eq!(agg.kv_cache_usage_perc, Some(90.0));
        assert_eq!(agg.kv_peak_pct, Some(95.0));
    }

    #[test]
    fn display_shows_normal_when_avg_triggered() {
        let d = KvCachePressureDetail {
            kv_cache_usage_perc: Some(92.0),
            kv_peak_pct: Some(97.0),
            preemptions_active: true,
            queue_backpressure: false,
        };
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(92.0),
            kv_cache_peak_perc: Some(97.0),
            num_preemptions_per_sec: Some(0.05),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(&d, &kv_ctx(&snap(v), None, None, None), 1, 1)
            .join("\n");
        assert!(!text.contains("burst pressure"));
        assert!(text.contains("92% avg, 97% peak"));
        assert!(text.contains("Scheduler evicting"));
    }

    #[test]
    fn resolve_observed_floors_h100_boot_log_ground_truth() {
        // Source: H100 boot log Jul 16 — kv_cache_max_concurrency = 24.64
        let (n, label) = resolve_r2_kv_capacity(Some(24.64), Some(99), false);
        assert_eq!(n, Some(24));
        assert_eq!(label, KvCapacityLabel::Observed);
    }

    #[test]
    fn observed_capacity_fix_shows_number_when_believable() {
        let mut v = VllmRawMetrics {
            max_num_seqs: Some(154),
            num_requests_running_peak: Some(14.0),
            cache_config: CacheConfigLabels {
                kv_cache_max_concurrency: Some(24.64),
                ..Default::default()
            },
            ..Default::default()
        };
        let snap_ok = snap(v.clone());
        assert_eq!(
            r2_capacity_phrase(24, seat_phrase_shows_number(&snap_ok, 24)),
            "Lower --max-num-seqs to ≤24 to reduce KV demand"
        );
        assert!(!r2_capacity_phrase(24, true).contains("vLLM-reported"));
        assert!(!r2_capacity_phrase(24, true).contains("(est)"));
        v.num_requests_running_peak = Some(25.0);
        let snap_bad = snap(v);
        assert_eq!(
            r2_capacity_phrase(24, seat_phrase_shows_number(&snap_bad, 24)),
            "Lower --max-num-seqs to reduce KV demand"
        );
    }

    #[test]
    fn derived_dense_capacity_uses_number_when_reduction() {
        let (n, _label) = resolve_r2_kv_capacity(None, Some(18), false);
        assert_eq!(n, Some(18));
        let v = VllmRawMetrics {
            max_num_seqs: Some(32),
            ..Default::default()
        };
        let snap = snap(v);
        let phrase = r2_capacity_phrase(18, seat_phrase_shows_number(&snap, 18));
        assert_eq!(phrase, "Lower --max-num-seqs to ≤18 to reduce KV demand");
        assert!(!phrase.contains("(est)"));
        assert!(!phrase.contains("hybrid"));
    }

    #[test]
    fn derived_capacity_hides_number_when_peak_running_exceeds_cap() {
        // No Observed label: resolve_r2_kv_capacity returns derived unchecked;
        // seat_phrase must still peak-gate the printed number.
        let (n, label) = resolve_r2_kv_capacity(None, Some(33), false);
        assert_eq!(n, Some(33));
        assert_eq!(label, KvCapacityLabel::Derived);
        let v = VllmRawMetrics {
            max_num_seqs: Some(154),
            num_requests_running_peak: Some(45.0),
            ..Default::default()
        };
        let phrase = r2_capacity_phrase(33, seat_phrase_shows_number(&snap(v), 33));
        assert_eq!(phrase, "Lower --max-num-seqs to reduce KV demand");
        assert!(!phrase.contains('≤'));
    }

    #[test]
    fn derived_capacity_shows_number_when_peak_running_below_cap() {
        let v = VllmRawMetrics {
            max_num_seqs: Some(154),
            num_requests_running_peak: Some(20.0),
            ..Default::default()
        };
        let phrase = r2_capacity_phrase(33, seat_phrase_shows_number(&snap(v), 33));
        assert_eq!(phrase, "Lower --max-num-seqs to ≤33 to reduce KV demand");
    }

    #[test]
    fn derived_capacity_shows_number_when_peak_running_absent() {
        let v = VllmRawMetrics {
            max_num_seqs: Some(154),
            ..Default::default()
        };
        let phrase = r2_capacity_phrase(33, seat_phrase_shows_number(&snap(v), 33));
        assert_eq!(phrase, "Lower --max-num-seqs to ≤33 to reduce KV demand");
    }

    #[test]
    fn derived_capacity_shows_number_when_peak_running_equals_cap() {
        // peak <= floor(value) keeps the number (same boundary as kv_bound_survives_peak).
        let v = VllmRawMetrics {
            max_num_seqs: Some(154),
            num_requests_running_peak: Some(33.0),
            ..Default::default()
        };
        let phrase = r2_capacity_phrase(33, seat_phrase_shows_number(&snap(v), 33));
        assert_eq!(phrase, "Lower --max-num-seqs to ≤33 to reduce KV demand");
    }

    #[test]
    fn derived_hybrid_capacity_direction_when_not_reduction() {
        let (n, label) = resolve_r2_kv_capacity(None, Some(18), true);
        assert_eq!(n, Some(18));
        assert_eq!(label, KvCapacityLabel::DerivedHybrid);
        let v = VllmRawMetrics {
            max_num_seqs: Some(16),
            ..Default::default()
        };
        let snap = snap(v);
        let phrase = r2_capacity_phrase(18, seat_phrase_shows_number(&snap, 18));
        assert_eq!(phrase, "Lower --max-num-seqs to reduce KV demand");
        assert!(!phrase.contains("worst-case"));
        assert!(!phrase.contains("hybrid"));
    }

    #[test]
    fn seat_bullet_numbering_peak_cap_and_reduction() {
        fn pressure_text(v: VllmRawMetrics, kv_max_seqs: u32) -> String {
            format_kv_cache_pressure_fired(
                &detail(98.0, false),
                &kv_ctx(&snap(v), Some(8192), Some(30.0), Some(kv_max_seqs)),
                3,
                4,
            )
            .join("\n")
        }
        let believable = VllmRawMetrics {
            kv_cache_usage_perc: Some(95.0),
            max_num_seqs: Some(154),
            num_requests_running_peak: Some(14.0),
            cache_config: CacheConfigLabels {
                kv_cache_max_concurrency: Some(23.0),
                cache_dtype: Some("bf16".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let t1 = pressure_text(believable, 23);
        assert!(t1.contains("Lower --max-num-seqs to ≤23 to reduce KV demand"));
        assert!(!t1.contains("vLLM-reported"));
        assert!(!t1.contains("worst-case"));

        let contradicted = VllmRawMetrics {
            kv_cache_usage_perc: Some(95.0),
            max_num_seqs: Some(154),
            num_requests_running_peak: Some(35.0),
            cache_config: CacheConfigLabels {
                kv_cache_max_concurrency: Some(13.0),
                cache_dtype: Some("bf16".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let t2 = pressure_text(contradicted, 13);
        assert!(t2.contains("Lower --max-num-seqs to reduce KV demand"));
        assert!(!t2.contains('≤'));

        let not_reduction = VllmRawMetrics {
            kv_cache_usage_perc: Some(95.0),
            max_num_seqs: Some(154),
            cache_config: CacheConfigLabels {
                cache_dtype: Some("bf16".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let t3 = pressure_text(not_reduction, 200);
        assert!(t3.contains("Lower --max-num-seqs to reduce KV demand"));
        assert!(!t3.contains('≤'));
    }

    #[test]
    fn seat_bullet_always_renders_on_admission_backlog() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            max_num_seqs: Some(154),
            num_requests_running_peak: Some(14.0),
            cache_config: CacheConfigLabels {
                kv_cache_max_concurrency: Some(23.0),
                cache_dtype: Some("bf16".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let text = format_kv_admission_backlog_issue(
            &sample_backlog_detail(),
            50,
            &kv_ctx(&snap(v), None, Some(30.0), Some(23)),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("Lower --max-num-seqs to ≤23 to reduce KV demand"));
    }

    #[test]
    fn fix_order_leads_with_model_len_when_p99_below_half() {
        // Source: live run 2026-07-17 — short p99 vs max_model_len; shrink leads.
        // 5465 < 32768/2; projection at 5465 is 39, not observed 8.
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            prompt_tokens_p99: Some(5000.0),
            generation_tokens_p99: Some(465.0),
            generation_tokens_completed: Some(150.0),
            cache_config: CacheConfigLabels {
                block_size: Some(16),
                num_gpu_blocks: Some(390),
                mamba_block_size: Some(784),
                kv_cache_max_concurrency: Some(8.667),
                ..Default::default()
            },
            ..Default::default()
        };
        let snap = snap(v);
        let ctx = KvFormatCtx {
            snapshot: &snap,
            max_model_len: Some(32768),
            kv_headroom_gb: None,
            kv_max_seqs: Some(8),
            config_max_num_seqs: None,
            capacity_label: KvCapacityLabel::Observed,
            fp8_compiler_available: false,
            model: None,
            tp: None,
            kv_cache_dtype: snap.vllm.cache_config.cache_dtype.as_deref(),
        };
        let d = KvCachePressureDetail {
            kv_cache_usage_perc: Some(90.0),
            kv_peak_pct: Some(90.0),
            preemptions_active: false,
            queue_backpressure: true,
        };
        let lines = format_kv_cache_pressure_fired(&d, &ctx, 3, 4);
        let text = lines.join("\n");
        let shrink_idx = lines
            .iter()
            .position(|l| l.contains("Lower --max-model-len 32768 → 5465"))
            .expect("shrink line");
        let follow_idx = lines
            .iter()
            .position(|l| l.contains("Then lower --max-num-seqs to reduce KV demand"))
            .expect("follow-on seats at shrink target");
        let cuts_idx = lines
            .iter()
            .position(|l| l == "    Cuts throughput:")
            .expect("Cuts throughput header");
        let fix_idx = lines
            .iter()
            .position(|l| l == "    Fix:")
            .expect("Fix header");
        assert!(fix_idx < cuts_idx);
        assert!(
            cuts_idx < shrink_idx && shrink_idx < follow_idx,
            "model-len shrink must lead the follow-on seat bullet under Cuts throughput"
        );
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.as_str() == "    Cuts throughput:")
                .count(),
            1
        );
        assert!(!text.contains("fits at least"));
        assert!(!text.contains("worst-case"));
        assert!(text.contains("Then lower --max-num-seqs to reduce KV demand"));
        assert!(!text.contains("Or cap --max-num-seqs"));
        assert!(!text.contains("guaranteed at full"));
        assert!(!text.contains("fits 8 concurrent"));
        // D5: blank after Warning before next bullet
        let warn_idx = lines
            .iter()
            .position(|l| l.contains("rejected with a 400"))
            .expect("warning");
        assert!(lines[warn_idx + 1].is_empty());
        assert!(follow_idx > warn_idx);
    }

    #[test]
    fn fix_order_leads_with_max_num_seqs_when_p99_at_or_above_half() {
        // 6450 >= 8192/2 → full-context bound leads; no "guaranteed at full" reword.
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            prompt_tokens_p99: Some(6000.0),
            generation_tokens_p99: Some(450.0),
            generation_tokens_completed: Some(150.0),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(
            &KvCachePressureDetail {
                kv_cache_usage_perc: Some(90.0),
                kv_peak_pct: Some(90.0),
                preemptions_active: false,
                queue_backpressure: true,
            },
            &kv_ctx(&snap(v), Some(8192), None, Some(16)),
            3,
            4,
        )
        .join("\n");
        let cuts_pos = text.find("    Cuts throughput:").expect("Cuts throughput");
        let seqs_pos = text
            .find("Lower --max-num-seqs")
            .expect("max-num-seqs bullet");
        let shrink_pos = text
            .find("Lower --max-model-len to 6450")
            .expect("shrink line uses to-form when cap names max_model_len");
        assert!(
            cuts_pos < seqs_pos && seqs_pos < shrink_pos,
            "max-num-seqs must lead shrink under Cuts throughput when p99 >= half"
        );
        assert!(!text.contains("guaranteed at full"));
        assert!(
            !text.contains("8192 → 6450"),
            "current already shown on cap bullet"
        );
    }

    #[test]
    fn queue_only_observed_shrink_uses_to_form() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            prompt_tokens_p99: Some(6000.0),
            generation_tokens_p99: Some(800.0),
            generation_tokens_completed: Some(150.0),
            ..Default::default()
        };
        // 6800 >= 8192/2 → cap leads; Observed names max_model_len → "to 6800".
        let snap = snap(v);
        let mut ctx = kv_ctx(&snap, Some(8192), Some(30.0), Some(120));
        ctx.capacity_label = KvCapacityLabel::Observed;
        let text = format_kv_cache_pressure_fired(
            &KvCachePressureDetail {
                kv_cache_usage_perc: Some(90.0),
                kv_peak_pct: Some(90.0),
                preemptions_active: false,
                queue_backpressure: true,
            },
            &ctx,
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("Lower --max-model-len to 6800"));
        assert!(!text.contains('→'));
        assert!(text.contains("    Cuts throughput:"));
    }

    #[test]
    fn queue_only_derived_hybrid_keeps_arrow_form() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            prompt_tokens_p99: Some(6000.0),
            generation_tokens_p99: Some(800.0),
            generation_tokens_completed: Some(150.0),
            ..Default::default()
        };
        let snap = snap(v);
        let mut ctx = kv_ctx(&snap, Some(8192), Some(30.0), Some(120));
        ctx.capacity_label = KvCapacityLabel::DerivedHybrid;
        let text = format_kv_cache_pressure_fired(
            &KvCachePressureDetail {
                kv_cache_usage_perc: Some(90.0),
                kv_peak_pct: Some(90.0),
                preemptions_active: false,
                queue_backpressure: true,
            },
            &ctx,
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("Lower --max-model-len 8192 → 6800"));
        assert!(!text.contains("worst-case"));
        assert!(!text.contains('≤'));
    }

    #[test]
    fn crisis_throttle_with_revert_subline_no_cuts_header_for_it() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            generation_tokens_per_sec: Some(100.0),
            ..Default::default()
        };
        let lines = format_kv_cache_pressure_fired(
            &detail(90.0, true),
            &kv_ctx(
                &snap_vram(v, VRAM_AMPLY_FREE_USED_MB, VRAM_AMPLY_FREE_TOTAL_MB),
                None,
                Some(30.0),
                Some(18),
            ),
            3,
            4,
        );
        let text = lines.join("\n");
        let fix_idx = lines.iter().position(|l| l == "    Fix:").expect("Fix");
        assert!(lines[fix_idx + 1].contains("Raise --gpu-memory-utilization"));
        let throttle_idx = lines
            .iter()
            .position(|l| l.contains("Lower --max-num-seqs"))
            .expect("throttle bullet");
        assert!(throttle_idx > fix_idx);
        assert_eq!(
            lines[throttle_idx + 1].trim(),
            "Cuts throughput. Revert after pressure clears."
        );
        assert!(
            !text.contains("    Cuts throughput:"),
            "no Cuts throughput header when only crisis throttle (no shrink)"
        );
        assert!(!text.contains("Once stable"));
    }

    #[test]
    fn non_crisis_safe_precede_cuts_throughput_header() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            prompt_tokens_mean: Some(250.0),
            cache_config: CacheConfigLabels {
                enable_prefix_caching: Some(false),
                ..Default::default()
            },
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(
            &KvCachePressureDetail {
                kv_cache_usage_perc: Some(90.0),
                kv_peak_pct: Some(90.0),
                preemptions_active: false,
                queue_backpressure: true,
            },
            &kv_ctx(
                &snap_vram(v, VRAM_AMPLY_FREE_USED_MB, VRAM_AMPLY_FREE_TOTAL_MB),
                Some(8192),
                Some(30.0),
                Some(16),
            ),
            3,
            4,
        )
        .join("\n");
        let safe_pos = text.find("    Safe to apply:").expect("Safe to apply");
        let prefix_pos = text.find("Enable --enable-prefix-caching").expect("prefix");
        let gpu_pos = text
            .find("Raise --gpu-memory-utilization")
            .expect("gpu-mem");
        let fp8_pos = text.find("Switch --kv-cache-dtype fp8").expect("fp8");
        let cuts_pos = text.find("    Cuts throughput:").expect("cuts");
        let seqs_pos = text.find("Lower --max-num-seqs").expect("seqs");
        assert!(safe_pos < prefix_pos && prefix_pos < gpu_pos && gpu_pos < fp8_pos);
        assert!(fp8_pos < cuts_pos && cuts_pos < seqs_pos);
        assert_eq!(text.matches("    Cuts throughput:").count(), 1);
        assert_eq!(text.matches("    Safe to apply:").count(), 1);
        assert!(
            !text.contains("Cuts throughput. Revert after pressure clears."),
            "non-crisis seat under Cuts throughput: header needs no throttle subline"
        );
        assert!(text.contains("(affects output quality; FP8 compiler not found)"));
    }

    #[test]
    fn crisis_flat_fix_includes_shrink_without_cuts_header() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            prompt_tokens_p99: Some(6000.0),
            generation_tokens_p99: Some(450.0),
            generation_tokens_completed: Some(150.0),
            cache_config: CacheConfigLabels {
                enable_prefix_caching: Some(true),
                cache_dtype: Some("fp8".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let lines = format_kv_cache_pressure_fired(
            &detail(90.0, true),
            &kv_ctx(&snap(v), Some(8192), Some(1.0), Some(16)),
            3,
            4,
        );
        let text = lines.join("\n");
        let fix_idx = lines.iter().position(|l| l == "    Fix:").expect("Fix");
        assert!(
            !text.contains("    Cuts throughput:"),
            "crisis must not use Cuts throughput header"
        );
        assert!(
            lines[fix_idx + 1].contains("Lower --max-num-seqs"),
            "p99 at/above half: seat leads (safe empty)"
        );
        assert_eq!(
            lines[fix_idx + 2].trim(),
            "Cuts throughput. Revert after pressure clears."
        );
        assert!(lines[fix_idx + 3].is_empty(), "blank after revert subline");
        let shrink = text
            .find("Lower --max-model-len to 6450")
            .expect("to-form shrink when seat leads");
        assert!(text.find("    Fix:").unwrap() < shrink);
        assert!(text.contains("rejected with a 400"));
        let warn = lines
            .iter()
            .position(|l| l.contains("rejected with a 400"))
            .expect("warning subline");
        assert!(lines[warn].contains("rejected with a 400"));
        assert!(
            !lines.windows(2).any(|w| w[0].is_empty() && w[1].is_empty()),
            "no consecutive blank lines in block"
        );
    }

    #[test]
    fn safe_group_empty_crisis_shrink_stays_under_fix() {
        // Crisis, safe empty (caching on, fp8, headroom < 2GB), shrink present.
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            prompt_tokens_p99: Some(6000.0),
            generation_tokens_p99: Some(450.0),
            generation_tokens_completed: Some(150.0),
            cache_config: CacheConfigLabels {
                enable_prefix_caching: Some(true),
                cache_dtype: Some("fp8".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let lines = format_kv_cache_pressure_fired(
            &detail(90.0, true),
            &kv_ctx(&snap(v), Some(8192), Some(1.0), Some(16)),
            3,
            4,
        );
        let fix_idx = lines.iter().position(|l| l == "    Fix:").expect("Fix");
        assert!(!lines.iter().any(|l| l == "    Cuts throughput:"));
        let shrink_idx = lines
            .iter()
            .position(|l| l.contains("Lower --max-model-len to 6450"))
            .expect("shrink under Fix");
        assert!(fix_idx < shrink_idx);
        // Between Fix and shrink: crisis throttle block only.
        let between = &lines[fix_idx + 1..shrink_idx];
        assert!(
            between.iter().all(|l| {
                l.contains("Lower --max-num-seqs")
                    || l.contains("Cuts throughput. Revert")
                    || l.is_empty()
            }),
            "safe empty: only crisis throttle before shrink: {between:?}"
        );
    }

    #[test]
    fn throttle_group_empty_omits_cuts_header() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            // No p99 / low count → no shrink; still has max-num-seqs in cuts though.
            ..Default::default()
        };
        // max-num-seqs always in cuts for non-crisis → header present.
        // Crisis without shrink: no Cuts header.
        let text = format_kv_cache_pressure_fired(
            &detail(90.0, true),
            &kv_ctx(&snap(v), None, Some(30.0), Some(18)),
            1,
            1,
        )
        .join("\n");
        assert!(!text.contains("    Cuts throughput:"));
        assert!(text.contains("Cuts throughput. Revert after pressure clears."));
    }

    #[test]
    fn vram_capacity_bullet_has_no_max_context_fragment() {
        let d = KvCachePressureDetail {
            kv_cache_usage_perc: Some(90.0),
            kv_peak_pct: Some(90.0),
            preemptions_active: false,
            queue_backpressure: true,
        };
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            prompt_tokens_p99: Some(6000.0),
            generation_tokens_p99: Some(450.0),
            generation_tokens_completed: Some(150.0),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(
            &d,
            &kv_ctx(
                &snap_vram(v, VRAM_ITER3_USED_MB, VRAM_ITER3_TOTAL_MB),
                Some(8192),
                Some(1.0),
                None,
            ),
            3,
            4,
        )
        .join("\n");
        assert!(!text.contains("Raise --gpu-memory-utilization"));
        assert!(!text.contains("GPU at VRAM capacity"));
        assert!(!text.contains("max context length"));
        assert_eq!(
            text.matches("Lower --max-model-len").count(),
            1,
            "shrink appears once in throttle group"
        );
        let cuts = text.find("    Cuts throughput:").expect("cuts");
        let shrink = text.find("Lower --max-model-len").expect("shrink");
        assert!(cuts < shrink);
    }

    #[test]
    fn model_is_hybrid_when_linear_field_set() {
        let mut dense = ModelArch::default();
        assert!(!model_is_hybrid(&dense));
        dense.linear_num_layers = Some(48);
        assert!(model_is_hybrid(&dense));
    }

    #[test]
    fn contradicted_cap_renders_direction_only_seat_bullet() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(98.0),
            kv_cache_peak_perc: Some(100.0),
            num_requests_running: Some(16.0),
            num_requests_running_peak: Some(16.0),
            num_requests_waiting: Some(4.0),
            num_preemptions_per_sec: Some(0.05),
            generation_tokens_completed: Some(48.0),
            prompt_tokens_mean: Some(1100.0),
            generation_tokens_mean: Some(4000.0),
            max_num_seqs: Some(256),
            cache_config: CacheConfigLabels {
                kv_cache_max_concurrency: Some(1.06),
                enable_prefix_caching: Some(true),
                cache_dtype: Some("fp8".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let snap = snap(v);
        assert!(super::super::observed_kv_cap_contradicted(&snap));
        let text = format_kv_cache_pressure_fired(
            &detail(98.0, true),
            &kv_ctx(&snap, Some(262144), Some(1.0), Some(1)),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("Lower --max-num-seqs to reduce KV demand"));
        assert!(!text.contains('≤'));
        assert!(!text.contains("Then set --max-num-seqs"));
        assert!(text.contains("Observed 5.1k tokens per request, prompt plus generation."));
        assert!(!text.contains('~'));
    }

    #[test]
    fn usable_kv_concurrency_keeps_cap_when_peak_absent_or_not_above() {
        let mut v = VllmRawMetrics {
            cache_config: CacheConfigLabels {
                kv_cache_max_concurrency: Some(24.4),
                ..Default::default()
            },
            ..Default::default()
        };
        let snap_absent = snap(v.clone());
        assert_eq!(
            super::super::usable_kv_concurrency(&snap_absent),
            Some(24.4)
        );
        v.num_requests_running_peak = Some(1.0);
        let snap_ok = snap(v.clone());
        assert_eq!(super::super::usable_kv_concurrency(&snap_ok), Some(24.4));
        v.num_requests_running_peak = Some(25.0);
        let snap_bad = snap(v);
        assert!(super::super::usable_kv_concurrency(&snap_bad).is_none());
    }

    #[test]
    fn crisis_short_p99_leads_with_shrink_not_throttle() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(98.0),
            num_preemptions_per_sec: Some(0.05),
            prompt_tokens_p99: Some(5000.0),
            generation_tokens_p99: Some(465.0),
            generation_tokens_completed: Some(150.0),
            cache_config: CacheConfigLabels {
                block_size: Some(16),
                num_gpu_blocks: Some(390),
                mamba_block_size: Some(784),
                kv_cache_max_concurrency: Some(8.667),
                enable_prefix_caching: Some(true),
                cache_dtype: Some("fp8".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let snap = snap(v);
        let lines = format_kv_cache_pressure_fired(
            &detail(98.0, true),
            &KvFormatCtx {
                snapshot: &snap,
                max_model_len: Some(32768),
                kv_headroom_gb: Some(1.0),
                kv_max_seqs: Some(8),
                config_max_num_seqs: None,
                capacity_label: KvCapacityLabel::Observed,
                fp8_compiler_available: false,
                model: None,
                tp: None,
                kv_cache_dtype: snap.vllm.cache_config.cache_dtype.as_deref(),
            },
            3,
            4,
        );
        let fix_idx = lines.iter().position(|l| l == "    Fix:").expect("Fix");
        assert!(
            lines[fix_idx + 1].contains("Lower --max-model-len 32768 → 5465"),
            "crisis + short p99: shrink leads: {}",
            lines[fix_idx + 1]
        );
        assert!(!lines[fix_idx + 1].contains("Lower --max-num-seqs"));
        let text = lines.join("\n");
        let then_idx = lines
            .iter()
            .position(|l| l.contains("Then lower --max-num-seqs to reduce KV demand"))
            .expect("follow-on seat at shrink target");
        assert!(
            then_idx + 1 >= lines.len()
                || !lines[then_idx + 1].contains("Cuts throughput. Revert after pressure clears."),
            "follow-on seat is permanent at the new window, not a throttle: {}",
            lines.get(then_idx + 1).map(|s| s.as_str()).unwrap_or("")
        );
        assert!(!text.contains("Or cap --max-num-seqs"));
        assert!(!text.contains("    Cuts throughput:"));
    }

    #[test]
    fn sub_floor_evidence_names_half_when_one_missing() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            generation_tokens_completed: Some(48.0),
            prompt_tokens_mean: Some(1100.0),
            generation_tokens_mean: None,
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(
            &KvCachePressureDetail {
                kv_cache_usage_perc: Some(90.0),
                kv_peak_pct: Some(90.0),
                preemptions_active: false,
                queue_backpressure: true,
            },
            &kv_ctx(&snap(v), Some(262144), Some(30.0), Some(16)),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("Observed prompt 1.1k tokens per request."));
        assert!(!text.contains("prompt plus generation"));
        assert!(!text.contains("unavailable"));
    }

    #[test]
    fn contradicted_cap_still_renders_fits_clause_on_shrink_target() {
        // Peak 16 > floor(8.667)=8 → contradicted; geometry still reads raw 8.667.
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_running_peak: Some(16.0),
            num_requests_waiting: Some(5.0),
            prompt_tokens_p99: Some(5000.0),
            generation_tokens_p99: Some(465.0),
            generation_tokens_completed: Some(150.0),
            cache_config: CacheConfigLabels {
                block_size: Some(16),
                num_gpu_blocks: Some(390),
                mamba_block_size: Some(784),
                kv_cache_max_concurrency: Some(8.667),
                ..Default::default()
            },
            ..Default::default()
        };
        let snap = snap(v);
        assert!(super::super::observed_kv_cap_contradicted(&snap));
        let text = format_kv_cache_pressure_fired(
            &KvCachePressureDetail {
                kv_cache_usage_perc: Some(90.0),
                kv_peak_pct: Some(90.0),
                preemptions_active: false,
                queue_backpressure: true,
            },
            &KvFormatCtx {
                snapshot: &snap,
                max_model_len: Some(32768),
                kv_headroom_gb: None,
                kv_max_seqs: Some(8),
                config_max_num_seqs: None,
                capacity_label: KvCapacityLabel::Observed,
                fp8_compiler_available: false,
                model: None,
                tp: None,
                kv_cache_dtype: None,
            },
            3,
            4,
        )
        .join("\n");
        assert!(!text.contains("fits at least"));
        assert!(!text.contains("worst-case"));
        assert!(text.contains("Then lower --max-num-seqs to reduce KV demand"));
        assert!(!text.contains("Lower --max-num-seqs to ≤8"));
        assert!(!text.contains("Or cap --max-num-seqs"));
    }

    #[test]
    fn unknown_max_model_len_always_prescribes_shrink_with_rejection_warning() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(98.0),
            num_requests_running_peak: Some(16.0),
            num_requests_waiting: Some(4.0),
            num_preemptions_per_sec: Some(0.05),
            generation_tokens_completed: Some(10.0),
            cache_config: CacheConfigLabels {
                kv_cache_max_concurrency: Some(1.06),
                enable_prefix_caching: Some(true),
                cache_dtype: Some("fp8".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        // No max_model_len → still prescribe shrink + rejection warning; contradicted → direction-only seat.
        let lines = format_kv_cache_pressure_fired(
            &detail(98.0, true),
            &kv_ctx(&snap(v), None, Some(1.0), Some(40)),
            3,
            4,
        );
        let text = lines.join("\n");
        assert!(text.contains("Cause:"));
        assert!(text.contains("    Fix:"));
        assert!(text.contains("Lower --max-model-len to safely raise concurrency."));
        let warn_idx = lines
            .iter()
            .position(|l| l.contains("rejected with a 400, not truncated."))
            .expect("rejection subline");
        assert!(
            lines[warn_idx].starts_with("        "),
            "subline at 8 spaces: {:?}",
            lines[warn_idx]
        );
        assert!(lines[warn_idx + 1].is_empty(), "blank after subline");
        assert!(!text.contains("KV cache is the wall"));
        assert!(text.contains("Lower --max-num-seqs to reduce KV demand"));
        assert!(!text.contains('≤'));
        assert!(text.contains("Expected:"));
    }

    fn offload_cache(state: crate::collectors::KvOffloadState) -> CacheConfigLabels {
        CacheConfigLabels {
            kv_offloading: state,
            enable_prefix_caching: Some(true),
            cache_dtype: Some("bf16".into()),
            ..Default::default()
        }
    }

    fn assert_offload_block(lines: &[String]) {
        let idx = lines
            .iter()
            .position(|l| l.contains("Set --kv-offloading-size"))
            .expect("offload bullet");
        let gpu_idx = lines
            .iter()
            .position(|l| l.contains("Raise --gpu-memory-utilization"))
            .expect("gpu-mem bullet");
        let fp8_idx = lines
            .iter()
            .position(|l| l.contains("Switch --kv-cache-dtype fp8"))
            .expect("fp8 bullet");
        assert!(gpu_idx < idx, "offload must follow gpu-memory-utilization");
        assert!(fp8_idx < idx, "offload must follow fp8");
        if let Some(safe_idx) = lines.iter().position(|l| l == "    Safe to apply:") {
            assert!(safe_idx < idx, "offload must sit inside the safe group");
            if let Some(cuts_idx) = lines.iter().position(|l| l == "    Cuts throughput:") {
                assert!(idx < cuts_idx, "offload must precede the cuts group");
            }
        }
        assert!(
            lines[idx].starts_with("      •"),
            "bullet indent 6 spaces: {:?}",
            lines[idx]
        );
        assert_eq!(
            lines[idx + 1].trim_start(),
            "Check host RAM and your container memory limit before allocating."
        );
        assert_eq!(
            lines[idx + 1],
            format!("        {}", KV_OFFLOAD_SUBLINE.trim_start()),
            "offload subline must match push_bullet_with_subline indent"
        );
        if lines.get(idx + 2).is_some_and(|l| l.is_empty()) {
            assert_eq!(lines[idx + 2], String::new(), "blank after subline");
        }
        let fix_idx = lines
            .iter()
            .position(|l| l == "    Fix:")
            .expect("Fix header");
        let section_end = lines
            .iter()
            .position(|l| l.starts_with("    Expected:"))
            .unwrap_or(lines.len());
        if lines.iter().any(|l| l == "    Safe to apply:") {
            let cuts_idx = lines
                .iter()
                .position(|l| l == "    Cuts throughput:")
                .expect("labeled safe requires cuts group in offload test fixture");
            let safe_end = lines[fix_idx + 1..cuts_idx]
                .iter()
                .rfind(|l| l.starts_with("      •") || l.starts_with("        "))
                .expect("safe group content");
            assert!(
                safe_end.contains("Set --kv-offloading-size")
                    || lines[idx + 1].trim_start() == KV_OFFLOAD_SUBLINE,
                "offload must be the last safe-group item"
            );
            for line in &lines[cuts_idx + 1..section_end] {
                assert!(
                    line.is_empty()
                        || line.starts_with("    Cuts throughput:")
                        || line.starts_with("      •")
                        || line.starts_with("        "),
                    "no unlabeled fix line after labeled groups: {line:?}"
                );
            }
        } else {
            // Crisis: flat Fix list (no group labels). Safe bullets precede cut bullets.
            let bullets: Vec<&String> = lines[fix_idx + 1..section_end]
                .iter()
                .filter(|l| l.starts_with("      •"))
                .collect();
            let offload_pos = bullets
                .iter()
                .position(|l| l.contains("Set --kv-offloading-size"))
                .expect("offload bullet in crisis fix list");
            let fp8_pos = bullets
                .iter()
                .position(|l| l.contains("Switch --kv-cache-dtype fp8"))
                .expect("fp8 bullet in crisis fix list");
            assert!(
                fp8_pos < offload_pos,
                "offload must follow fp8 in crisis list"
            );
            if let Some(after) = bullets.get(offload_pos + 1) {
                assert!(
                    after.contains("max-num-seqs") || after.contains("max-model-len"),
                    "only cut bullets may follow offload in crisis list: {after:?}"
                );
            }
        }
    }

    fn format_offload_three_paths(
        cache: CacheConfigLabels,
    ) -> (Vec<String>, Vec<String>, Vec<String>) {
        let mut v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            cache_config: cache.clone(),
            ..Default::default()
        };
        let crisis = format_kv_cache_pressure_fired(
            &detail(90.0, true),
            &kv_ctx(
                &snap_vram(v.clone(), VRAM_AMPLY_FREE_USED_MB, VRAM_AMPLY_FREE_TOTAL_MB),
                None,
                Some(30.0),
                None,
            ),
            3,
            4,
        );
        v.num_preemptions_per_sec = None;
        v.num_requests_waiting = Some(5.0);
        let non_crisis = format_kv_cache_pressure_fired(
            &KvCachePressureDetail {
                kv_cache_usage_perc: Some(90.0),
                kv_peak_pct: Some(90.0),
                preemptions_active: false,
                queue_backpressure: true,
            },
            &kv_ctx(
                &snap_vram(v.clone(), VRAM_AMPLY_FREE_USED_MB, VRAM_AMPLY_FREE_TOTAL_MB),
                None,
                Some(30.0),
                None,
            ),
            3,
            4,
        );
        let backlog = format_kv_admission_backlog_issue(
            &KvAdmissionBacklogDetail {
                kv_cache_usage_perc: 90.0,
                kv_peak_pct: Some(90.0),
                admission_ratio: 0.5,
                requests_waiting: 10.0,
                requests_running: 10.0,
                free_kv_tokens: 100.0,
                demand_tokens: 200.0,
            },
            50,
            &kv_ctx(
                &snap_vram(v, VRAM_AMPLY_FREE_USED_MB, VRAM_AMPLY_FREE_TOTAL_MB),
                None,
                Some(30.0),
                None,
            ),
            3,
            4,
        );
        (crisis, non_crisis, backlog)
    }

    #[test]
    fn kv_offload_subline_matches_dead_end_subline_convention() {
        use crate::collectors::KvOffloadState;
        let (_, non_crisis, _) = format_offload_three_paths(offload_cache(KvOffloadState::Off));
        let offload_idx = non_crisis
            .iter()
            .position(|l| l.contains("Set --kv-offloading-size"))
            .expect("offload bullet");
        let offload_subline = &non_crisis[offload_idx + 1];
        let verify_subline = format!("        {}", DEAD_END_VERIFY_SUBLINE.trim_start());
        assert_eq!(
            offload_subline
                .chars()
                .take_while(|c| *c == ' ')
                .collect::<String>(),
            verify_subline
                .chars()
                .take_while(|c| *c == ' ')
                .collect::<String>(),
            "offload and dead-end sublines must share push_bullet_with_subline indent"
        );
        assert_eq!(
            *offload_subline,
            format!("        {}", KV_OFFLOAD_SUBLINE.trim_start())
        );
        assert_eq!(
            verify_subline,
            format!("        {}", DEAD_END_VERIFY_SUBLINE)
        );
    }

    #[test]
    fn kv_offload_absent_label_no_bullet_all_paths() {
        use crate::collectors::KvOffloadState;
        let (c, n, b) = format_offload_three_paths(offload_cache(KvOffloadState::Unsupported));
        for text in [c.join("\n"), n.join("\n"), b.join("\n")] {
            assert!(
                !text.contains("kv-offloading-size"),
                "absent label must not suggest offload: {text}"
            );
        }
        assert!(kv_offload_fix_bullet(&offload_cache(KvOffloadState::Unsupported)).is_none());
    }

    #[test]
    fn kv_offload_none_literal_bullet_all_paths() {
        use crate::collectors::KvOffloadState;
        let (c, n, b) = format_offload_three_paths(offload_cache(KvOffloadState::Off));
        assert_offload_block(&c);
        assert_offload_block(&n);
        assert_offload_block(&b);
        assert!(kv_offload_fix_bullet(&offload_cache(KvOffloadState::Off)).is_some());
    }

    #[test]
    fn kv_offload_size_16_no_bullet() {
        use crate::collectors::KvOffloadState;
        let (c, n, b) = format_offload_three_paths(offload_cache(KvOffloadState::Enabled(16.0)));
        for text in [c.join("\n"), n.join("\n"), b.join("\n")] {
            assert!(!text.contains("kv-offloading-size"), "already on: {text}");
        }
        assert!(kv_offload_fix_bullet(&offload_cache(KvOffloadState::Enabled(16.0))).is_none());
    }

    #[test]
    fn kv_offload_size_zero_bullet_all_paths() {
        use crate::collectors::KvOffloadState;
        // Zero parses to Off.
        let (c, n, b) = format_offload_three_paths(offload_cache(KvOffloadState::Off));
        assert_offload_block(&c);
        assert_offload_block(&n);
        assert_offload_block(&b);
    }

    #[test]
    fn kv_offload_garbage_no_bullet_no_panic() {
        use crate::collectors::KvOffloadState;
        let cache = offload_cache(KvOffloadState::Unreadable);
        assert!(kv_offload_fix_bullet(&cache).is_none());
        let (c, n, b) = format_offload_three_paths(cache);
        for text in [c.join("\n"), n.join("\n"), b.join("\n")] {
            assert!(!text.contains("kv-offloading-size"), "garbage: {text}");
        }
    }

    /// Crisis dead end: shrink no-op (<5%), cap contradicted, every safe lever already set
    /// or unavailable. Verify then replica bullets under Fix:, with Expected.
    fn dead_end_snap(
        offload: crate::collectors::KvOffloadState,
        max_model_len: u32,
        prompt_p99: f64,
        generation_p99: f64,
        max_num_seqs: Option<u32>,
    ) -> (RawSnapshot, u32) {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(98.0),
            num_requests_running_peak: Some(16.0),
            num_requests_waiting: Some(4.0),
            num_preemptions_per_sec: Some(0.05),
            generation_tokens_completed: Some(150.0),
            prompt_tokens_p99: Some(prompt_p99),
            generation_tokens_p99: Some(generation_p99),
            max_num_seqs,
            cache_config: CacheConfigLabels {
                kv_cache_max_concurrency: Some(1.06),
                enable_prefix_caching: Some(true),
                cache_dtype: Some("fp8".into()),
                kv_offloading: offload,
                ..Default::default()
            },
            ..Default::default()
        };
        (snap(v), max_model_len)
    }

    #[test]
    fn seat_lever_dead_end_at_floor_crisis_and_non_crisis() {
        let crisis = format_seat_lever_crisis(Some(1), None);
        assert_dead_end_pair(&crisis);
        assert!(!crisis.contains("Lower --max-num-seqs to reduce KV demand"));

        let non_crisis = format_seat_lever_non_crisis(Some(1), None);
        assert!(!non_crisis.contains("Lower --max-num-seqs to reduce KV demand"));
        assert_dead_end_pair(&non_crisis);

        let backlog = format_seat_lever_backlog(Some(1), None);
        assert!(!backlog.contains("Lower --max-num-seqs"));
        assert_dead_end_pair(&backlog);
    }

    #[test]
    fn seat_lever_shows_seat_when_above_floor() {
        for text in [
            format_seat_lever_crisis(Some(45), None),
            format_seat_lever_non_crisis(Some(45), None),
            format_seat_lever_backlog(Some(45), None),
        ] {
            assert!(text.contains("Lower --max-num-seqs to reduce KV demand"));
            assert_no_dead_end_pair(&text);
        }
    }

    #[test]
    fn seat_lever_withheld_when_max_num_seqs_unknown() {
        for text in [
            format_seat_lever_crisis(None, None),
            format_seat_lever_non_crisis(None, None),
            format_seat_lever_backlog(None, None),
        ] {
            assert!(!text.contains("Lower --max-num-seqs to reduce KV demand"));
        }
        assert_dead_end_pair(&format_seat_lever_crisis(None, None));
        assert_dead_end_pair(&format_seat_lever_non_crisis(None, None));
        assert_dead_end_pair(&format_seat_lever_backlog(None, None));
    }

    #[test]
    fn seat_lever_available_at_two_not_one() {
        for text in [
            format_seat_lever_crisis(Some(2), None),
            format_seat_lever_non_crisis(Some(2), None),
            format_seat_lever_backlog(Some(2), None),
        ] {
            assert!(text.contains("Lower --max-num-seqs to reduce KV demand"));
            assert_no_dead_end_pair(&text);
        }
    }

    #[test]
    fn seat_lever_reads_config_when_scrape_absent() {
        let text = format_seat_lever_crisis(None, Some(45));
        assert!(text.contains("Lower --max-num-seqs to reduce KV demand"));
        assert_no_dead_end_pair(&text);
        let dead = format_seat_lever_crisis(None, Some(1));
        assert_dead_end_pair(&dead);
    }

    fn headroom_gate_non_crisis(
        used_mb: u64,
        total_mb: u64,
        kv_headroom_gb: Option<f64>,
        max_num_seqs: Option<u32>,
    ) -> String {
        use crate::collectors::KvOffloadState;
        let (base, m) = dead_end_snap(
            KvOffloadState::Enabled(16.0),
            10000,
            5000.0,
            4600.0,
            max_num_seqs,
        );
        let snap = snap_vram(base.vllm.clone(), used_mb, total_mb);
        format_kv_cache_pressure_fired(
            &KvCachePressureDetail {
                kv_cache_usage_perc: Some(98.0),
                kv_peak_pct: Some(98.0),
                preemptions_active: false,
                queue_backpressure: true,
            },
            &kv_ctx_config(&snap, Some(m), kv_headroom_gb, None, None),
            3,
            4,
        )
        .join("\n")
    }

    #[test]
    fn gpu_mem_bullet_omitted_when_observed_vram_low_non_crisis() {
        let text = headroom_gate_non_crisis(
            VRAM_ITER3_USED_MB,
            VRAM_ITER3_TOTAL_MB,
            Some(30.0),
            Some(45),
        );
        assert!(!text.contains("Raise --gpu-memory-utilization"));
    }

    #[test]
    fn gpu_mem_bullet_omitted_when_computed_budget_exhausted() {
        let text = headroom_gate_non_crisis(
            VRAM_AMPLY_FREE_USED_MB,
            VRAM_AMPLY_FREE_TOTAL_MB,
            Some(1.0),
            Some(45),
        );
        assert!(!text.contains("Raise --gpu-memory-utilization"));
    }

    #[test]
    fn gpu_mem_bullet_omitted_when_vram_unreadable() {
        use crate::collectors::KvOffloadState;
        let (snap, m) = dead_end_snap(
            KvOffloadState::Enabled(16.0),
            10000,
            5000.0,
            4600.0,
            Some(45),
        );
        let text = format_kv_cache_pressure_fired(
            &KvCachePressureDetail {
                kv_cache_usage_perc: Some(98.0),
                kv_peak_pct: Some(98.0),
                preemptions_active: false,
                queue_backpressure: true,
            },
            &kv_ctx_config(&snap, Some(m), Some(30.0), None, None),
            3,
            4,
        )
        .join("\n");
        assert!(!text.contains("Raise --gpu-memory-utilization"));
    }

    #[test]
    fn crisis_gpu_mem_bullet_when_headroom_and_vram_ample() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(
            &detail(90.0, true),
            &kv_ctx(
                &snap_vram(v, VRAM_AMPLY_FREE_USED_MB, VRAM_AMPLY_FREE_TOTAL_MB),
                None,
                Some(30.0),
                None,
            ),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("Raise --gpu-memory-utilization"));
    }

    #[test]
    fn iteration3_shape_no_gpu_mem_bullet_at_78_of_80() {
        use crate::collectors::KvOffloadState;
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(97.0),
            num_requests_running_peak: Some(1.0),
            num_preemptions_per_sec: Some(0.05),
            cache_config: CacheConfigLabels {
                kv_cache_max_concurrency: Some(1.06),
                enable_prefix_caching: Some(true),
                cache_dtype: Some("fp8".into()),
                kv_offloading: KvOffloadState::Enabled(16.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(
            &detail(97.0, true),
            &kv_ctx(
                &snap_vram(v, VRAM_ITER3_USED_MB, VRAM_ITER3_TOTAL_MB),
                Some(262144),
                Some(15.0),
                None,
            ),
            3,
            4,
        )
        .join("\n");
        assert!(!text.contains("Raise --gpu-memory-utilization"));
        assert!(!text.contains("GPU at VRAM capacity"));
    }

    #[test]
    fn hardware_wall_dead_end_absent_when_offload_off() {
        use crate::collectors::KvOffloadState;
        let (snap, m) = dead_end_snap(KvOffloadState::Off, 10000, 5000.0, 4600.0, Some(256));
        let text = format_kv_cache_pressure_fired(
            &detail(98.0, true),
            &kv_ctx(&snap, Some(m), None, None),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("    Fix:"));
        assert!(text.contains("Set --kv-offloading-size"));
        assert!(!text.contains("took effect"));
        assert!(!text.contains("No config change on this GPU moves the KV wall."));
        assert!(text.contains("Expected:"));
    }

    #[test]
    fn no_lever_verify_when_only_seat_remains() {
        use crate::collectors::KvOffloadState;
        // 5000+4600=9600; at max_model_len=10000 that is within the 5% no-op band.
        let (snap, m) = dead_end_snap(
            KvOffloadState::Enabled(16.0),
            10000,
            5000.0,
            4600.0,
            Some(256),
        );
        let lines = format_kv_cache_pressure_fired(
            &detail(98.0, true),
            &kv_ctx(&snap, Some(m), None, Some(40)),
            3,
            4,
        );
        let text = lines.join("\n");
        assert!(text.contains("Cause:"));
        assert!(text.contains("    Fix:"));
        assert!(text.contains("Lower --max-num-seqs to reduce KV demand"));
        assert!(!text.contains('≤'));
        assert!(!text.contains("took effect"));
        assert!(!text.contains("No config change on this GPU moves the KV wall."));
        assert!(text.contains("Expected:"));
        assert!(!text.contains("Set --kv-offloading-size"));
        assert!(!text.contains("Lower --max-model-len"));
    }

    #[test]
    fn no_lever_verify_absent_when_offload_off_fills_fix() {
        use crate::collectors::KvOffloadState;
        let (snap, m) = dead_end_snap(KvOffloadState::Off, 10000, 5000.0, 4600.0, Some(256));
        let text = format_kv_cache_pressure_fired(
            &detail(98.0, true),
            &kv_ctx(&snap, Some(m), None, Some(40)),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("    Fix:"));
        assert!(text.contains("Set --kv-offloading-size"));
        assert!(!text.contains("took effect"));
        assert!(!text.contains("No config change on this GPU moves the KV wall."));
        assert!(text.contains("Expected:"));
    }

    #[test]
    fn no_lever_verify_absent_when_shrink_available() {
        use crate::collectors::KvOffloadState;
        // Same p99s; max_model_len far above → named shrink fires.
        let (snap, m) = dead_end_snap(
            KvOffloadState::Enabled(16.0),
            32768,
            5000.0,
            4600.0,
            Some(256),
        );
        let text = format_kv_cache_pressure_fired(
            &detail(98.0, true),
            &kv_ctx(&snap, Some(m), None, Some(40)),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("    Fix:"));
        assert!(text.contains("Lower --max-model-len"));
        assert!(!text.contains("took effect"));
        assert!(!text.contains("No config change on this GPU moves the KV wall."));
        assert!(text.contains("Expected:"));
    }
}
