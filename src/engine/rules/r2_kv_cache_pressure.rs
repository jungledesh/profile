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
/// Minimum KV headroom (GB) before recommending --gpu-memory-utilization; below this,
/// weights and allocator overhead leave no safe room to expand the KV pool.
const KV_HEADROOM_SAFE_MIN_GB: f64 = 2.0;
const FP8_KV_CACHE_FIX: &str =
    "      • Switch --kv-cache-dtype fp8 to halve KV memory footprint (affects output quality)";
/// Suggest prefix caching when mean prompt length exceeds this (tokens).
const PREFIX_CACHING_LONG_PROMPT_MIN_TOKENS: f64 = 200.0;

fn fp8_kv_cache_fix_bullet(
    kv_cache_dtype: Option<&str>,
    weight_bytes_per_param: u8,
    fp8_compiler_available: bool,
) -> Option<String> {
    // Advising a switch to the dtype already in use costs operator trust;
    // dtype is observable, so observe it.
    if kv_bytes_per_element(kv_cache_dtype, weight_bytes_per_param.max(1)) == 1 {
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

fn kv_headroom_gpu_mem_bullet(kv_headroom_gb: Option<f64>) -> String {
    match kv_headroom_gb {
        Some(h) if h >= KV_HEADROOM_SAFE_MIN_GB => {
            "      • Raise --gpu-memory-utilization (check vRAM header for avail mem) to expand KV pool"
                .to_string()
        }
        Some(_) => {
            "      • GPU at VRAM capacity: cannot raise --gpu-memory-utilization. Add a replica to scale out."
                .to_string()
        }
        None => "      • Raise --gpu-memory-utilization (check vRAM header for avail mem) to expand KV pool".to_string(),
    }
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

#[derive(Debug, Clone, PartialEq)]
pub struct KvAdmissionBacklogDetail {
    pub kv_cache_usage_perc: f64,
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
    let kv = snapshot.vllm.kv_cache_usage_perc.filter(|v| v.is_finite());
    let peak = snapshot.vllm.kv_cache_peak_perc.filter(|v| v.is_finite());
    let kv_avg_high = kv.is_some_and(|kv_p| kv_p >= KV_CACHE_PRESSURE_MIN_PERC);
    let kv_peak_high = peak.is_some_and(|p| p >= KV_CACHE_PRESSURE_MIN_PERC);
    if !kv_avg_high && !kv_peak_high {
        return Rule2Outcome::NotFired;
    }

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
                capacity_label,
                weight_bytes_per_param: 2,
                fp8_compiler_available,
                model: None,
                tp: None,
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

/// Self-grade exists to audit derived estimates. Observed prescriptions are the
/// allocator's own numbers; grading them against themselves is noise.
pub(super) fn prescribed_for_self_grade(
    capacity: Option<u32>,
    label: KvCapacityLabel,
) -> Option<u32> {
    match label {
        KvCapacityLabel::Derived | KvCapacityLabel::DerivedHybrid => capacity,
        KvCapacityLabel::Observed => None,
    }
}

fn r2_capacity_phrase(n: u32, max_model_len: Option<u32>, label: KvCapacityLabel) -> String {
    // Vocabulary: simultaneous-work counts are "N concurrent requests", never bare
    // "capacity". Bounds name their source and condition.
    match label {
        KvCapacityLabel::Observed => match max_model_len {
            Some(m) => {
                format!(
                    "Lower --max-num-seqs to ≤{n} concurrent requests \
                     (vLLM-reported at max_model_len={m})"
                )
            }
            None => {
                format!("Lower --max-num-seqs to ≤{n} concurrent requests (vLLM-reported)")
            }
        },
        KvCapacityLabel::Derived => match max_model_len {
            Some(m) => {
                format!(
                    "Lower --max-num-seqs to ≤{n} concurrent requests \
                     (est); at least {n} worst-case requests fit at max_model_len={m}"
                )
            }
            None => format!(
                "Lower --max-num-seqs to ≤{n} concurrent requests \
                 (est); at least {n} worst-case requests fit"
            ),
        },
        KvCapacityLabel::DerivedHybrid => match max_model_len {
            Some(m) => format!(
                "Lower --max-num-seqs to ≤{n} concurrent requests \
                 (est); at least {n} worst-case requests fit at max_model_len={m}, including hybrid state"
            ),
            None => format!(
                "Lower --max-num-seqs to ≤{n} concurrent requests \
                 (est); at least {n} worst-case requests fit, including hybrid state"
            ),
        },
    }
}

/// Cap bullet when model-len shrink leads: full-context concurrency is a floor,
/// not a target. Worst-case capacity is a floor; leading with it on short-prompt
/// workloads prescribes a large throughput cut the traffic does not require.
fn max_num_seqs_cap_after_shrink_bullet(n: u32, max_model_len: u32) -> String {
    format!(
        "      • Or cap --max-num-seqs at {n} \
         (vLLM-reported; guaranteed at full {max_model_len}-token contexts)"
    )
}

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

fn max_num_seqs_bullet(
    kv_max_seqs: Option<u32>,
    max_model_len: Option<u32>,
    label: KvCapacityLabel,
    evictions: bool,
) -> String {
    match kv_max_seqs {
        Some(n) => format!("      • {}", r2_capacity_phrase(n, max_model_len, label)),
        None => {
            if evictions {
                "      • Lower --max-num-seqs to stop evictions".to_string()
            } else {
                "      • Lower --max-num-seqs to free KV blocks".to_string()
            }
        }
    }
}

pub(super) struct KvFormatCtx<'a> {
    pub snapshot: &'a RawSnapshot,
    pub max_model_len: Option<u32>,
    pub kv_headroom_gb: Option<f64>,
    pub kv_max_seqs: Option<u32>,
    pub capacity_label: KvCapacityLabel,
    pub weight_bytes_per_param: u8,
    pub fp8_compiler_available: bool,
    pub model: Option<&'a crate::context::ModelArch>,
    pub tp: Option<u32>,
}

impl<'a> KvFormatCtx<'a> {
    fn hyp_capacity(&self) -> super::HypCapacityCtx<'a> {
        super::HypCapacityCtx {
            cache: &self.snapshot.vllm.cache_config,
            kv_headroom_gb: self.kv_headroom_gb,
            model: self.model,
            kv_cache_dtype: self.snapshot.vllm.cache_config.cache_dtype.as_deref(),
            tp: self.tp,
            weight_bytes: self.weight_bytes_per_param,
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
    let capacity_label = ctx.capacity_label;
    let weight_bytes_per_param = ctx.weight_bytes_per_param;
    let fp8_compiler_available = ctx.fp8_compiler_available;
    let kv_cache_dtype = snapshot.vllm.cache_config.cache_dtype.as_deref();
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
            unreachable!("R2 requires preemptions or queue backpressure to fire")
        }
    };
    out.push(evidence);
    out.push(String::new());

    let total_count = snapshot.vllm.generation_tokens_completed.unwrap_or(0.0);
    let prompt_p99 = snapshot.vllm.prompt_tokens_p99;
    let generation_p99 = snapshot.vllm.generation_tokens_p99;
    // Cap-leads with Observed/Derived + m already name current max_model_len; use "to N".
    // Crisis, shrink-leads, DerivedHybrid, and no-capacity keep the arrow form.
    let would_lead_if_shrink = max_model_len
        .is_some_and(|m| super::p99_sum_below_half_max_model_len(m, prompt_p99, generation_p99));
    let shrink_current_shown = !d.preemptions_active
        && !would_lead_if_shrink
        && kv_max_seqs.is_some()
        && max_model_len.is_some()
        && matches!(
            capacity_label,
            KvCapacityLabel::Observed | KvCapacityLabel::Derived
        );
    let shrink_lines = super::model_len_shrink_suggestion_lines(
        max_model_len,
        prompt_p99,
        generation_p99,
        total_count,
        "      ",
        Some(&hyp),
        shrink_current_shown,
    );
    // Lead with model-len when observed traffic fits in half the window: the
    // full-context concurrency floor is then a secondary bound, not the primary fix.
    let lead_with_shrink = !shrink_lines.is_empty() && would_lead_if_shrink;

    let mut safe = Vec::new();
    if let Some(bullet) = prefix_caching_fix_bullet(snapshot) {
        safe.push(bullet);
    }
    if d.preemptions_active {
        // Same raise string as non-crisis; omit when headroom is missing or < 2GB.
        if kv_headroom_gb.is_some_and(|h| h >= KV_HEADROOM_SAFE_MIN_GB) {
            safe.push(kv_headroom_gpu_mem_bullet(kv_headroom_gb));
        }
    } else {
        safe.push(kv_headroom_gpu_mem_bullet(kv_headroom_gb));
    }
    if let Some(bullet) = fp8_kv_cache_fix_bullet(
        kv_cache_dtype,
        weight_bytes_per_param,
        fp8_compiler_available,
    ) {
        safe.push(bullet);
    }

    if d.preemptions_active {
        // Crisis: one flat Fix list. Throttle subline marks risk; no Cuts header.
        let crisis_throttle = max_num_seqs_bullet(kv_max_seqs, max_model_len, capacity_label, true);
        out.push("    Fix:".to_string());
        super::push_bullet_with_subline(
            &mut out,
            crisis_throttle,
            Some("Cuts throughput. Revert after pressure clears."),
        );
        out.extend(safe);
        // No-op when shrink_lines is empty.
        super::extend_with_shrink_suggestion(&mut out, shrink_lines);
    } else {
        let mut cuts = Vec::new();
        let seqs_bullet = if lead_with_shrink {
            match (kv_max_seqs, max_model_len) {
                (Some(n), Some(m)) => max_num_seqs_cap_after_shrink_bullet(n, m),
                _ => max_num_seqs_bullet(kv_max_seqs, max_model_len, capacity_label, false),
            }
        } else {
            max_num_seqs_bullet(kv_max_seqs, max_model_len, capacity_label, false)
        };
        if lead_with_shrink {
            super::extend_with_shrink_suggestion(&mut cuts, shrink_lines);
            cuts.push(seqs_bullet);
        } else {
            cuts.push(seqs_bullet);
            super::extend_with_shrink_suggestion(&mut cuts, shrink_lines);
        }
        super::push_grouped_fixes(&mut out, safe, cuts, Vec::new());
    }

    let expected = if d.preemptions_active {
        "    Expected: TTFT and TPOT recover once evictions stop."
    } else {
        "    Expected: Wait queue drains, TTFT recovers once KV pool has capacity."
    };
    // Exactly one blank before Expected, regardless of what the last group left.
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
    let kv_cache_dtype = ctx.snapshot.vllm.cache_config.cache_dtype.as_deref();
    let mut out = vec![
        "[!] KV Cache Pressure: Admission Backlog".to_string(),
        "    Cause:".to_string(),
        format!(
            "      Scheduler holding {:.0} requests in queue ({:.0}% of active requests waiting) to protect KV memory.",
            d.requests_waiting,
            d.admission_ratio * 100.0
        ),
        format!(
            "      Free KV tokens: {:.0} available, {:.0} demanded.",
            d.free_kv_tokens, d.demand_tokens
        ),
        String::new(),
    ];

    let total_count = ctx.snapshot.vllm.generation_tokens_completed.unwrap_or(0.0);
    let prompt_p99 = ctx.snapshot.vllm.prompt_tokens_p99;
    let generation_p99 = ctx.snapshot.vllm.generation_tokens_p99;
    let hyp = ctx.hyp_capacity();
    let shrink_lines = super::model_len_shrink_suggestion_lines(
        ctx.max_model_len,
        prompt_p99,
        generation_p99,
        total_count,
        "      ",
        Some(&hyp),
        false,
    );

    let mut safe = Vec::new();
    if let Some(bullet) = prefix_caching_fix_bullet(ctx.snapshot) {
        safe.push(bullet);
    }
    safe.push(kv_headroom_gpu_mem_bullet(ctx.kv_headroom_gb));
    if let Some(bullet) = fp8_kv_cache_fix_bullet(
        kv_cache_dtype,
        ctx.weight_bytes_per_param,
        ctx.fp8_compiler_available,
    ) {
        safe.push(bullet);
    }

    let mut cuts = Vec::new();
    super::extend_with_shrink_suggestion(&mut cuts, shrink_lines);
    super::push_grouped_fixes(&mut out, safe, cuts, Vec::new());

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
    KvAdmissionBacklogDetail {
        kv_cache_usage_perc: kv,
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
    use crate::collectors::{CacheConfigLabels, VllmRawMetrics};

    fn snap(vllm: VllmRawMetrics) -> RawSnapshot {
        crate::collectors::snap_vllm(vllm)
    }

    fn kv_ctx(
        snapshot: &RawSnapshot,
        max_model_len: Option<u32>,
        kv_headroom_gb: Option<f64>,
        kv_max_seqs: Option<u32>,
    ) -> KvFormatCtx<'_> {
        KvFormatCtx {
            snapshot,
            max_model_len,
            kv_headroom_gb,
            kv_max_seqs,
            capacity_label: KvCapacityLabel::Derived,
            weight_bytes_per_param: 2,
            fp8_compiler_available: false,
            model: None,
            tp: None,
        }
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
        let r = r2_recommendation(R2RecommendationInput {
            snapshot: &snap(v.clone()),
            max_model_len: None,
            kv_headroom_gb: None,
            kv_max_seqs: None,
            capacity_label: KvCapacityLabel::Derived,
            windows_fired: 1,
            total_evaluable: 1,
            fp8_compiler_available: false,
        })
        .expect("fired");
        let text = r.display_lines.join("\n");
        assert!(!text.contains("evictions stop"));
        assert!(text.contains("gpu-memory-utilization") || text.contains("GPU memory"));
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
        assert!(text.contains("≤14"));
        assert!(text.contains("max_model_len=8192"));
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
        assert!(text.contains("max_model_len=8192"));
        assert!(text.contains("≤15"));
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
        assert!(text.contains("Lower --max-num-seqs to ≤18 concurrent requests (est)"));
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
        assert!(text.contains("Lower --max-num-seqs to stop evictions"));
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
        assert!(text.contains("Truncation risk"));
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
        assert!(text.contains("Truncation risk"));
    }

    #[test]
    fn model_len_not_in_evictions_path_when_ceiling_unknown() {
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
        assert!(!text.contains("--max-model-len"));
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
        assert!(
            text.contains("max_model_len=4096"),
            "evictions path should include max_model_len when ceiling is known"
        );
        assert!(
            text.contains("≤16"),
            "evictions path should include the ceiling value"
        );
    }

    fn sample_backlog_detail() -> KvAdmissionBacklogDetail {
        KvAdmissionBacklogDetail {
            kv_cache_usage_perc: 90.0,
            admission_ratio: 0.4,
            requests_waiting: 10.0,
            requests_running: 15.0,
            free_kv_tokens: 160.0,
            demand_tokens: 200.0,
        }
    }

    #[test]
    fn backlog_shows_headroom_when_safe() {
        let text = format_kv_admission_backlog_issue(
            &sample_backlog_detail(),
            27,
            &kv_ctx(&snap(VllmRawMetrics::default()), None, Some(30.0), None),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("check vRAM header for avail mem"));
    }

    #[test]
    fn backlog_warns_when_vram_full() {
        let text = format_kv_admission_backlog_issue(
            &sample_backlog_detail(),
            27,
            &kv_ctx(&snap(VllmRawMetrics::default()), None, Some(1.0), None),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("GPU at VRAM capacity"));
    }

    #[test]
    fn backlog_generic_when_headroom_unknown() {
        let text = format_kv_admission_backlog_issue(
            &sample_backlog_detail(),
            27,
            &kv_ctx(&snap(VllmRawMetrics::default()), None, None, None),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("check vRAM header for avail mem"));
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
        assert!(lines.contains("Truncation risk"));
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
        assert!(text.contains("Lower --max-num-seqs to free KV blocks"));
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
        let text =
            format_kv_cache_pressure_fired(&d, &kv_ctx(&snap(v), None, Some(1.0), None), 1, 1)
                .join("\n");
        assert!(text.contains("GPU at VRAM capacity"));
        assert!(text.contains("Add a replica to scale out"));
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
            &kv_ctx(&snap(v), None, Some(30.0), None),
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
    fn queue_backpressure_shows_raise_gpu_mem_bullet_when_headroom_unknown() {
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
        assert!(text.contains(
            "Raise --gpu-memory-utilization (check vRAM header for avail mem) to expand KV pool"
        ));
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
            fp8_kv_cache_fix_bullet(None, 2, true).expect("bf16/auto should suggest fp8");
        assert!(with_compiler.contains("Switch --kv-cache-dtype fp8"));
        assert!(with_compiler.contains("(affects output quality)"));
        assert!(!with_compiler.contains("FP8 compiler not found"));
        let without_compiler =
            fp8_kv_cache_fix_bullet(None, 2, false).expect("bf16/auto should suggest fp8");
        assert!(without_compiler.contains("(affects output quality; FP8 compiler not found)"));
    }

    #[test]
    fn fp8_kv_cache_bullet_suppressed_when_already_fp8() {
        assert!(fp8_kv_cache_fix_bullet(Some("fp8"), 2, true).is_none());
        assert!(fp8_kv_cache_fix_bullet(Some("FP8"), 2, true).is_none());
        assert!(fp8_kv_cache_fix_bullet(Some("e4m3fnuz"), 2, true).is_none());
        assert!(fp8_kv_cache_fix_bullet(Some("e5m2"), 2, true).is_none());
        assert!(fp8_kv_cache_fix_bullet(Some("auto"), 2, true).is_some());
    }

    #[test]
    fn fp8_kv_cache_bullet_uses_resolved_kv_bytes() {
        assert!(
            fp8_kv_cache_fix_bullet(Some("auto"), 2, true).is_some(),
            "auto + bf16 weights should still suggest fp8 KV"
        );
        assert!(
            fp8_kv_cache_fix_bullet(Some("auto"), 1, true).is_none(),
            "auto + fp8 weights resolves to one-byte KV, so no switch is needed"
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
    fn observed_capacity_fix_omits_est_label() {
        let phrase = r2_capacity_phrase(24, Some(8192), KvCapacityLabel::Observed);
        assert!(phrase.contains("vLLM-reported"));
        assert!(phrase.contains("concurrent requests"));
        assert!(phrase.contains("≤24"));
        assert!(!phrase.contains("(est)"));
    }

    #[test]
    fn derived_dense_capacity_uses_est_label() {
        let (n, label) = resolve_r2_kv_capacity(None, Some(18), false);
        assert_eq!(n, Some(18));
        assert_eq!(label, KvCapacityLabel::Derived);
        let phrase = r2_capacity_phrase(18, None, label);
        assert!(phrase.contains("(est)"));
        assert!(!phrase.contains("hybrid"));
    }

    #[test]
    fn derived_hybrid_capacity_uses_hybrid_label() {
        let (n, label) = resolve_r2_kv_capacity(None, Some(18), true);
        assert_eq!(n, Some(18));
        assert_eq!(label, KvCapacityLabel::DerivedHybrid);
        let phrase = r2_capacity_phrase(18, Some(8192), label);
        assert!(phrase.contains("(est)"));
        assert!(phrase.contains("at least 18 worst-case requests fit"));
        assert!(phrase.contains("including hybrid state"));
        assert!(phrase.contains("concurrent requests"));
        assert!(phrase.contains("max_model_len=8192"));
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
            capacity_label: KvCapacityLabel::Observed,
            weight_bytes_per_param: 2,
            fp8_compiler_available: false,
            model: None,
            tp: None,
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
        let cap_idx = lines
            .iter()
            .position(|l| l.contains("vLLM-reported; guaranteed at full"))
            .expect("cap line");
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
            cuts_idx < shrink_idx && shrink_idx < cap_idx,
            "model-len shrink must lead the cap bullet under Cuts throughput"
        );
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.as_str() == "    Cuts throughput:")
                .count(),
            1
        );
        assert!(text.contains("fits at least 39 worst-case requests (est)"));
        assert!(text.contains("Or cap --max-num-seqs at 8"));
        assert!(text.contains("guaranteed at full 32768-token contexts"));
        assert!(!text.contains("fits 8 concurrent"));
        // D5: blank after Warning before next bullet
        let warn_idx = lines
            .iter()
            .position(|l| l.contains("Truncation risk"))
            .expect("warning");
        assert!(lines[warn_idx + 1].is_empty());
        assert!(cap_idx > warn_idx);
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
        assert_eq!(
            text.matches("max_model_len=8192").count(),
            1,
            "current max_model_len once per block"
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
        assert_eq!(text.matches("max_model_len=8192").count(), 1);
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
        assert!(
            text.contains("max_model_len=8192"),
            "DerivedHybrid capacity phrase names max_model_len like Derived"
        );
        assert_eq!(text.matches("max_model_len=8192").count(), 1);
    }

    #[test]
    fn crisis_throttle_first_with_revert_subline_no_cuts_header_for_it() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            generation_tokens_per_sec: Some(100.0),
            ..Default::default()
        };
        let lines = format_kv_cache_pressure_fired(
            &detail(90.0, true),
            &kv_ctx(&snap(v), None, Some(30.0), Some(18)),
            3,
            4,
        );
        let text = lines.join("\n");
        let fix_idx = lines.iter().position(|l| l == "    Fix:").expect("Fix");
        let first_bullet = lines[fix_idx + 1].clone();
        assert!(
            first_bullet.contains("Lower --max-num-seqs"),
            "crisis first bullet must be max-num-seqs throttle: {first_bullet}"
        );
        assert_eq!(
            lines[fix_idx + 2].trim(),
            "Cuts throughput. Revert after pressure clears."
        );
        assert!(lines[fix_idx + 3].is_empty(), "blank after crisis sub-line");
        assert!(
            !text.contains("    Cuts throughput:"),
            "no Cuts throughput header when only crisis throttle (no shrink)"
        );
        assert!(text.contains(
            "Raise --gpu-memory-utilization (check vRAM header for avail mem) to expand KV pool"
        ));
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
            &kv_ctx(&snap(v), Some(8192), Some(30.0), Some(16)),
            3,
            4,
        )
        .join("\n");
        let prefix_pos = text.find("Enable --enable-prefix-caching").expect("prefix");
        let gpu_pos = text
            .find("Raise --gpu-memory-utilization")
            .expect("gpu-mem");
        let fp8_pos = text.find("Switch --kv-cache-dtype fp8").expect("fp8");
        let cuts_pos = text.find("    Cuts throughput:").expect("cuts");
        let seqs_pos = text.find("Lower --max-num-seqs").expect("seqs");
        assert!(prefix_pos < gpu_pos && gpu_pos < fp8_pos && fp8_pos < cuts_pos);
        assert!(cuts_pos < seqs_pos);
        assert_eq!(text.matches("    Cuts throughput:").count(), 1);
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
            "throttle first"
        );
        assert_eq!(
            lines[fix_idx + 2].trim(),
            "Cuts throughput. Revert after pressure clears."
        );
        assert!(lines[fix_idx + 3].is_empty(), "blank after revert subline");
        let shrink = text
            .find("Lower --max-model-len 8192 → 6450")
            .expect("arrow-form shrink in Fix");
        assert!(text.find("    Fix:").unwrap() < shrink);
        assert!(text.contains("Truncation risk"));
        let warn = lines
            .iter()
            .position(|l| l.contains("Truncation risk"))
            .expect("warning subline");
        assert!(lines[warn].contains("Warning") || lines[warn].contains("Truncation"));
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
            .position(|l| l.contains("Lower --max-model-len 8192 → 6450"))
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
            &kv_ctx(&snap(v), Some(8192), Some(1.0), None),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("GPU at VRAM capacity"));
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
    fn self_grade_prescribes_only_derived_labels() {
        assert_eq!(
            prescribed_for_self_grade(Some(24), KvCapacityLabel::Derived),
            Some(24)
        );
        assert_eq!(
            prescribed_for_self_grade(Some(18), KvCapacityLabel::DerivedHybrid),
            Some(18)
        );
        assert_eq!(
            prescribed_for_self_grade(Some(24), KvCapacityLabel::Observed),
            None
        );
        assert_eq!(
            prescribed_for_self_grade(None, KvCapacityLabel::Derived),
            None
        );
    }
}
