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
}

/// Last-resort offload lines (bullet + downside + host RAM), or empty when gated off.
/// Caller owns the `Last resort:` header. Only when eviction is active.
fn kv_offload_last_resort_lines(
    snapshot: &RawSnapshot,
    kv_headroom_gb: Option<f64>,
    kv_cache_dtype: Option<&str>,
    model: Option<&crate::context::ModelArch>,
    tp: Option<u32>,
) -> Vec<String> {
    use crate::collectors::KvOffloadState;

    if !eviction_signal_active(snapshot) {
        return Vec::new();
    }
    if matches!(
        snapshot.vllm.cache_config.kv_offloading,
        KvOffloadState::Unsupported | KvOffloadState::Unreadable
    ) {
        return Vec::new();
    }

    let size_input = KvOffloadSizeInput {
        host_memory: snapshot.host_memory,
        pool_bytes: super::resolve_kv_pool_bytes(
            kv_headroom_gb,
            model,
            kv_cache_dtype,
            tp,
            Some(&snapshot.vllm.cache_config),
        ),
        kv_frac_per_running_peak: peak_kv_frac_per_running(snapshot),
        preempt_per_sec: snapshot.vllm.num_preemptions_per_sec,
        run_duration_secs: snapshot.vllm.window_duration_secs,
        peak_waiting: peak_waiting(snapshot),
    };

    let size = match snapshot.vllm.cache_config.kv_offloading {
        KvOffloadState::Off => resolve_kv_offload_size_gib(size_input),
        KvOffloadState::Enabled(v) => {
            let Some(derived) = resolve_kv_offload_size_gib(size_input) else {
                return Vec::new();
            };
            let Some(set) = ceil_bytes_to_whole_gib(v * GIB_BYTES) else {
                return Vec::new();
            };
            if derived <= set {
                return Vec::new();
            }
            Some(derived)
        }
        KvOffloadState::Unsupported | KvOffloadState::Unreadable => return Vec::new(),
    };

    vec![
        kv_offload_fix_bullet(size),
        format!("        {KV_OFFLOAD_DOWNSIDE}"),
        format!(
            "        {}",
            format_kv_offload_subline(snapshot.host_memory).trim_start()
        ),
        String::new(),
    ]
}

fn emit_last_resort_offload(out: &mut Vec<String>, lines: Vec<String>) {
    if lines.is_empty() {
        return;
    }
    if !out.last().is_some_and(|l| l.is_empty()) && !out.last().is_some_and(|l| l == "    Fix:") {
        out.push(String::new());
    }
    out.push(LAST_RESORT_HEADER.to_string());
    out.extend(lines);
}

fn peak_waiting(snapshot: &RawSnapshot) -> Option<f64> {
    snapshot
        .vllm
        .num_requests_waiting_peak
        .or(snapshot.vllm.num_requests_waiting)
        .filter(|w| w.is_finite())
}

fn peak_kv_frac_per_running(snapshot: &RawSnapshot) -> Option<f64> {
    snapshot
        .vllm
        .kv_frac_per_running_peak
        .filter(|f| f.is_finite() && *f > 0.0)
        .or_else(|| {
            let running = snapshot
                .vllm
                .num_requests_running
                .filter(|r| r.is_finite() && *r >= 1.0)?;
            let kv = snapshot
                .vllm
                .kv_cache_usage_perc
                .filter(|k| k.is_finite())?;
            let frac = (kv / 100.0) / running;
            frac.is_finite().then_some(frac).filter(|f| *f > 0.0)
        })
}

/// Host RAM reserve kept free so the OOM killer stays away. Judgment, provisional.
pub(super) const KV_OFFLOAD_RESERVE_FRACTION: f64 = 0.5;

const GIB_BYTES: f64 = 1024.0 * 1024.0 * 1024.0;

/// Round demand/supply bytes up to a whole GiB for `--kv-offloading-size`.
pub(super) fn ceil_bytes_to_whole_gib(bytes: f64) -> Option<u64> {
    if !bytes.is_finite() || bytes <= 0.0 {
        return None;
    }
    let gib = (bytes / GIB_BYTES).ceil();
    (gib.is_finite() && gib > 0.0 && gib <= u64::MAX as f64).then_some(gib as u64)
}

#[derive(Debug, Clone, Copy)]
pub(super) struct KvOffloadSizeInput {
    pub host_memory: Option<crate::collectors::HostMemoryFacts>,
    pub pool_bytes: Option<u64>,
    pub kv_frac_per_running_peak: Option<f64>,
    pub preempt_per_sec: Option<f64>,
    pub run_duration_secs: Option<f64>,
    pub peak_waiting: Option<f64>,
}

fn usable_host_bytes(facts: crate::collectors::HostMemoryFacts) -> u64 {
    match facts.container_limit_bytes {
        Some(lim) => facts.available_bytes.min(lim),
        None => facts.available_bytes,
    }
}

/// Eviction-spill size in whole GiB, or `None` when any input is missing / zero.
/// Never returns a number from a partial formula.
pub(super) fn resolve_kv_offload_size_gib(input: KvOffloadSizeInput) -> Option<u64> {
    let facts = input.host_memory?;
    let usable = usable_host_bytes(facts);
    if usable == 0 {
        return None;
    }
    let pool = input.pool_bytes.filter(|&b| b > 0)?;
    let frac = input
        .kv_frac_per_running_peak
        .filter(|f| f.is_finite() && *f > 0.0)?;
    let preempt = input
        .preempt_per_sec
        .filter(|p| p.is_finite() && *p > 0.0)?;
    let duration = input
        .run_duration_secs
        .filter(|d| d.is_finite() && *d > 0.0)?;
    // Stock cap required when preempting: flow alone is forbidden.
    let peak_waiting = input.peak_waiting.filter(|w| w.is_finite() && *w > 0.0)?;

    let flow = preempt * duration;
    if !flow.is_finite() || flow <= 0.0 {
        return None;
    }
    let parked = flow.min(peak_waiting);
    let per_seq = (pool as f64) * frac;
    if !per_seq.is_finite() || per_seq <= 0.0 {
        return None;
    }
    let demand = parked * per_seq;
    let supply = KV_OFFLOAD_RESERVE_FRACTION * (usable as f64);
    if !demand.is_finite() || !supply.is_finite() || supply <= 0.0 {
        return None;
    }
    ceil_bytes_to_whole_gib(demand.min(supply))
}

const KV_OFFLOAD_FIX: &str = "      • Set --kv-offloading-size (GiB) to hold evicted KV in host memory instead of recomputing it";
/// Fallback when host memory reads fail; rendered as the second last-resort subline.
const KV_OFFLOAD_SUBLINE_FALLBACK: &str =
    "Check host RAM and your container memory limit before allocating.";
const KV_OFFLOAD_DOWNSIDE: &str =
    "Spills KV to host so more sequences can stay admitted; can starve decode under long prompts.";
const LAST_RESORT_HEADER: &str = "    Last resort:";

pub(super) fn format_kv_offload_subline(
    facts: Option<crate::collectors::HostMemoryFacts>,
) -> String {
    use crate::collectors::host_memory::bytes_to_display_gib;
    match facts {
        Some(f) => {
            let ram_gib = bytes_to_display_gib(f.available_bytes);
            let limit = match f.container_limit_bytes {
                Some(b) => format!("{} GiB", bytes_to_display_gib(b)),
                None => "none".to_string(),
            };
            format!("Host RAM available: {ram_gib} GiB, container limit {limit}.")
        }
        None => KV_OFFLOAD_SUBLINE_FALLBACK.to_string(),
    }
}

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

/// Suggest KV offload size (Off path, or Enabled re-offer when derived > set).
fn kv_offload_fix_bullet(size_gib: Option<u64>) -> String {
    match size_gib {
        Some(n) => format!(
            "      • Set --kv-offloading-size {n} (est) to hold evicted KV in host memory instead of recomputing it"
        ),
        None => KV_OFFLOAD_FIX.to_string(),
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
    let (display_lines, terminal) = format_kv_cache_pressure_fired_with_terminal(
        &d,
        &KvFormatCtx {
            snapshot,
            max_model_len,
            kv_headroom_gb,
            kv_max_seqs,
            config_max_num_seqs: snapshot.vllm.max_num_seqs,
            capacity_label,
            fp8_compiler_available,
            model: None,
            tp: None,
            kv_cache_dtype: snapshot.vllm.cache_config.cache_dtype.as_deref(),
        },
        windows_fired,
        total_evaluable,
    );
    Some(Recommendation {
        rule_name: rule_names::KV_CACHE_PRESSURE,
        layer: 2,
        impact: 5,
        confidence,
        display_lines,
        terminal,
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

/// Follow-on seats after a named shrink target.
const FOLLOW_ON_SEAT_BULLET: &str = "      • Then lower --max-num-seqs to reduce KV demand";

const SEAT_BULLET: &str = "      • Lower --max-num-seqs to reduce KV demand";

pub(super) fn kv_pressure_confidence(windows_fired: usize, total_evaluable: usize) -> f64 {
    if total_evaluable == 0 {
        return 0.0;
    }
    (windows_fired as f64 / total_evaluable as f64).clamp(0.0, 1.0)
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
    config_max_num_seqs: Option<u32>,
) -> Option<String> {
    seat_lever_available(snapshot, config_max_num_seqs).then(|| SEAT_BULLET.to_string())
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

/// Fired-window KV evidence line shared by pressure and admission-backlog Cause.
fn kv_cause_line(avg_s: &str, peak_s: &str, burst: bool) -> String {
    if burst {
        format!(
            "      KV cache {avg_s} avg in fired windows, {peak_s} peak (burst pressure, threshold: {:.0}%).",
            KV_CACHE_PRESSURE_MIN_PERC
        )
    } else {
        format!(
            "      KV cache {avg_s} avg in fired windows, {peak_s} peak (threshold: {:.0}%).",
            KV_CACHE_PRESSURE_MIN_PERC
        )
    }
}

#[cfg(test)]
pub(super) fn format_kv_cache_pressure_fired(
    d: &KvCachePressureDetail,
    ctx: &KvFormatCtx<'_>,
    windows_fired: usize,
    total_evaluable: usize,
) -> Vec<String> {
    format_kv_cache_pressure_fired_with_terminal(d, ctx, windows_fired, total_evaluable).0
}

pub(super) fn format_kv_cache_pressure_fired_with_terminal(
    d: &KvCachePressureDetail,
    ctx: &KvFormatCtx<'_>,
    windows_fired: usize,
    total_evaluable: usize,
) -> (Vec<String>, bool) {
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
    out.push(kv_cause_line(&avg_s, &peak_s, burst));
    // Cause and crisis layout use the same summary snapshot as TRAFFIC preempt/s
    // (run-level mean rate / swapped), not per-window any(). Otherwise a single
    // spike can print "Scheduler evicting" while the header shows preempt/s 0.01.
    // Waiting print uses the measured mean whenever present. The >2 queue bar is a
    // fire gate only; gating display on it falsely claims "unavailable" for 0 < w ≤ 2.
    let preemptions_active = eviction_signal_active(snapshot);
    let wait_count = snapshot.vllm.num_requests_waiting.filter(|v| v.is_finite());
    let evidence = match (preemptions_active, wait_count) {
        (true, Some(w)) => {
            format!("      Scheduler evicting; {w:.0} requests queued on KV admission.")
        }
        (true, None) => "      Scheduler evicting sequences to free KV blocks.".to_string(),
        (false, Some(w)) => format!("      {w:.0} requests queued on KV admission."),
        (false, None) => {
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
    //
    // Invariant: when the seat lever exists, exactly one seat line. Follow-on
    // ("Then…") only when shrink leads with a projectable named target; otherwise
    // the plain form. Crisis subline on the plain form only, gated by summary
    // eviction (same as crisis layout).
    let follow_on_seat = lead_with_shrink
        && shrink.target.is_some_and(|suggested| {
            super::capacity_at_hypothetical_max_len(suggested, max_model_len, &hyp).is_some()
        });
    let mut cuts: Vec<super::CutBullet> = Vec::new();
    super::extend_with_shrink_suggestion(&mut cuts, shrink);
    if seat_lever_available(snapshot, config_max_num_seqs) {
        if follow_on_seat {
            cuts.push((FOLLOW_ON_SEAT_BULLET.to_string(), None));
        } else {
            let seat = SEAT_BULLET.to_string();
            let sub = preemptions_active.then_some(CRISIS_THROTTLE_SUBLINE);
            if lead_with_shrink {
                cuts.push((seat, sub));
            } else {
                cuts.insert(0, (seat, sub));
            }
        }
    }

    let last_resort =
        kv_offload_last_resort_lines(snapshot, kv_headroom_gb, kv_cache_dtype, ctx.model, ctx.tp);

    let terminal = safe.is_empty() && cuts.is_empty() && last_resort.is_empty();
    if terminal {
        push_dead_end_fixes(&mut safe);
    }

    if preemptions_active {
        // Crisis: flat Fix list for cuts/safe, no group labels. Offload is still
        // labeled Last resort so operators never read it as safe.
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
            if cuts_nonempty && safe_nonempty && !out.last().is_some_and(|l| l.is_empty()) {
                out.push(String::new());
            }
            out.extend(safe);
        } else {
            out.extend(safe);
            if safe_nonempty && cuts_nonempty && !out.last().is_some_and(|l| l.is_empty()) {
                out.push(String::new());
            }
            emit_cuts(&mut out, cuts);
        }
        emit_last_resort_offload(&mut out, last_resort);
    } else {
        let lead_with_cuts = lead_with_shrink;
        super::trim_group_trailing_blanks(&mut safe);
        super::push_grouped_fixes(&mut out, safe, cuts, Vec::new(), lead_with_cuts);
        emit_last_resort_offload(&mut out, last_resort);
    }

    let expected = if preemptions_active {
        "    Expected: TTFT and TPOT recover once evictions stop."
    } else {
        "    Expected: Wait queue drains, TTFT recovers once KV pool has capacity."
    };
    super::trim_group_trailing_blanks(&mut out);
    out.push(String::new());
    out.push(expected.to_string());
    if super::rule_is_significant(windows_fired, total_evaluable) {
        let confidence = kv_pressure_confidence(windows_fired, total_evaluable);
        out.push(format!(
            "    Confidence: {}",
            super::confidence_label(confidence)
        ));
    }
    (out, terminal)
}

#[cfg(test)]
pub(super) fn format_kv_admission_backlog_issue(
    d: &KvAdmissionBacklogDetail,
    seen_pct: u32,
    ctx: &KvFormatCtx<'_>,
    windows_fired: usize,
    total_evaluable: usize,
) -> Vec<String> {
    format_kv_admission_backlog_issue_with_terminal(
        d,
        seen_pct,
        ctx,
        windows_fired,
        total_evaluable,
    )
    .0
}

pub(super) fn format_kv_admission_backlog_issue_with_terminal(
    d: &KvAdmissionBacklogDetail,
    seen_pct: u32,
    ctx: &KvFormatCtx<'_>,
    windows_fired: usize,
    total_evaluable: usize,
) -> (Vec<String>, bool) {
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
    out.push(kv_cause_line(&avg_s, &peak_s, burst));
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
    if let Some(seat) = full_window_seat_bullet(ctx.snapshot, ctx.config_max_num_seqs) {
        cuts.push((seat, None));
    }
    super::extend_with_shrink_suggestion(&mut cuts, shrink);

    // Same eviction gate as pressure: Last resort only when eviction is active.
    let last_resort = kv_offload_last_resort_lines(
        ctx.snapshot,
        ctx.kv_headroom_gb,
        kv_cache_dtype,
        ctx.model,
        ctx.tp,
    );

    let terminal = safe.is_empty() && cuts.is_empty() && last_resort.is_empty();
    if terminal {
        push_dead_end_fixes(&mut safe);
    }

    super::trim_group_trailing_blanks(&mut safe);
    super::push_grouped_fixes(&mut out, safe, cuts, Vec::new(), false);
    emit_last_resort_offload(&mut out, last_resort);

    out.push(String::new());
    out.push("    Expected: Wait queue drains, TTFT recovers.".to_string());
    if super::rule_is_significant(windows_fired, total_evaluable) {
        let confidence = kv_pressure_confidence(windows_fired, total_evaluable);
        out.push(format!(
            "    Confidence: {}",
            super::confidence_label(confidence)
        ));
    }
    (super::with_seen_pct(out, seen_pct), terminal)
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
) -> (Vec<String>, bool) {
    let (lines, terminal) =
        format_kv_cache_pressure_fired_with_terminal(d, ctx, windows_fired, total_evaluable);
    (super::with_seen_pct(lines, seen_pct), terminal)
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
        let config_max_num_seqs = snapshot.vllm.max_num_seqs;
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

    #[test]
    fn dead_end_path_is_terminal_pressure_and_backlog() {
        use crate::collectors::KvOffloadState;
        let (snap, m) = dead_end_snap(
            KvOffloadState::Enabled(16.0),
            10000,
            5000.0,
            4600.0,
            Some(1),
        );
        let (_, pressure_term) = format_kv_cache_pressure_fired_with_terminal(
            &detail(98.0, true),
            &kv_ctx_config(&snap, Some(m), None, None, None),
            3,
            4,
        );
        assert!(pressure_term);
        let (_, backlog_term) = format_kv_admission_backlog_issue_with_terminal(
            &sample_backlog_detail(),
            50,
            &kv_ctx_config(&snap, Some(m), None, None, None),
            3,
            4,
        );
        assert!(backlog_term);
    }

    #[test]
    fn lever_path_not_terminal() {
        use crate::collectors::KvOffloadState;
        let (_, terminal) = format_kv_cache_pressure_fired_with_terminal(
            &detail(90.0, true),
            &kv_ctx(
                &snap_vram(
                    VllmRawMetrics {
                        kv_cache_usage_perc: Some(90.0),
                        num_preemptions_per_sec: Some(0.05),
                        cache_config: offload_cache(KvOffloadState::Off),
                        max_num_seqs: Some(256),
                        ..Default::default()
                    },
                    VRAM_AMPLY_FREE_USED_MB,
                    VRAM_AMPLY_FREE_TOTAL_MB,
                ),
                None,
                Some(30.0),
                None,
            ),
            3,
            4,
        );
        assert!(!terminal);
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
        use crate::engine::rules::{CONFIDENCE_HIGH_MIN, CONFIDENCE_MEDIUM_MIN};
        // 4/15 = 0.267: below CONFIDENCE_MEDIUM_MIN (0.6), prints Low.
        let density_4_of_15 = kv_pressure_confidence(4, 15);
        assert!(density_4_of_15 < CONFIDENCE_MEDIUM_MIN);
        // 9/15 = 0.6: at CONFIDENCE_MEDIUM_MIN, prints Medium.
        let density_9_of_15 = kv_pressure_confidence(9, 15);
        assert!(density_9_of_15 >= CONFIDENCE_MEDIUM_MIN);
        assert!(density_9_of_15 < CONFIDENCE_HIGH_MIN);
        // 13/15 = 0.867: above CONFIDENCE_HIGH_MIN (0.8), prints High.
        let density_13_of_15 = kv_pressure_confidence(13, 15);
        assert!(density_13_of_15 >= CONFIDENCE_HIGH_MIN);
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
        assert!(stable.contains("Confidence: Medium"));
        assert!(!stable.contains("Medium-High"));
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
            max_num_seqs: Some(256),
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
    }

    #[test]
    fn display_includes_max_model_len_when_ceiling_known() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            generation_tokens_per_sec: Some(100.0),
            max_num_seqs: Some(256),
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
    }

    #[test]
    fn display_includes_ceiling_when_kv_max_seqs_known() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            generation_tokens_per_sec: Some(100.0),
            max_num_seqs: Some(256),
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
    }

    #[test]
    fn kv_pressure_preemption_fix_matches_spec() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            generation_tokens_per_sec: Some(100.0),
            max_num_seqs: Some(256),
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
            max_num_seqs: Some(256),
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
        assert!(text.contains("KV cache 90% avg in fired windows, 90% peak (threshold: 88%)."));
        assert!(text.contains("Free KV tokens: 160 available, 200 demanded (est, worst case)."));
        assert!(!text.contains("threshold: 88%). Free KV tokens"));
        assert!(!text.contains("burst pressure"));
        assert!(
            !text.contains("peak in fired windows"),
            "peak must not carry the fired-windows label"
        );
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
        assert!(text.contains(
            "KV cache 71% avg in fired windows, 92% peak (burst pressure, threshold: 88%)."
        ));
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
        assert!(stable.contains("Confidence: Medium"));
        assert!(!stable.contains("Medium-High"));
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
            max_num_seqs: Some(256),
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
    fn cause_scheduler_evicting_follows_summary_rate_not_detail_flag() {
        // Detail claims preemptions (as multi-window any() would), but summary
        // rate matches TRAFFIC preempt/s below the 0.02 bar → no "Scheduler evicting".
        let d = KvCachePressureDetail {
            kv_cache_usage_perc: Some(93.0),
            kv_peak_pct: Some(100.0),
            preemptions_active: true,
            queue_backpressure: true,
        };
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(93.0),
            kv_cache_peak_perc: Some(100.0),
            num_preemptions_per_sec: Some(0.01),
            num_requests_waiting: Some(6.0),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(&d, &kv_ctx(&snap(v), None, None, None), 3, 4)
            .join("\n");
        assert!(!text.contains("Scheduler evicting"));
        assert!(text.contains("6 requests queued on KV admission"));
    }

    #[test]
    fn cause_prints_below_bar_waiting_mean_not_unavailable() {
        // Fired on per-window queue spikes; summary waiting 1.4 is below the >2
        // fire bar but still measured. Must not claim "unavailable".
        let d = KvCachePressureDetail {
            kv_cache_usage_perc: Some(93.0),
            kv_peak_pct: Some(100.0),
            preemptions_active: false,
            queue_backpressure: true,
        };
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(93.0),
            kv_cache_peak_perc: Some(100.0),
            num_preemptions_per_sec: Some(0.01),
            num_requests_waiting: Some(1.4),
            max_num_seqs: Some(256),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(&d, &kv_ctx(&snap(v), None, None, None), 3, 4)
            .join("\n");
        assert!(text.contains("1 requests queued on KV admission"));
        assert!(!text.contains("waiting count unavailable"));
        assert!(!text.contains("Scheduler queueing requests"));
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
        assert!(text.contains("58% avg in fired windows, 93% peak"));
        assert!(text.contains("Scheduler evicting"));
        assert!(
            !text.contains("peak in fired windows"),
            "peak must not carry the fired-windows label"
        );
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
        assert!(text.contains("- avg in fired windows, 93% peak"));
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
        assert!(text.contains("92% avg in fired windows, 97% peak"));
        assert!(text.contains("Scheduler evicting"));
        assert!(
            !text.contains("peak in fired windows"),
            "peak must not carry the fired-windows label"
        );
    }

    #[test]
    fn cause_labels_fired_window_avg_final_iteration_fixture() {
        // Journey final iter: header CACHE 85.4% (run-level); Cause 91% (fired only).
        let d = KvCachePressureDetail {
            kv_cache_usage_perc: Some(91.0),
            kv_peak_pct: Some(98.0),
            preemptions_active: false,
            queue_backpressure: true,
        };
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(85.4),
            kv_cache_peak_perc: Some(97.9),
            num_requests_waiting: Some(11.0),
            max_num_seqs: Some(60),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(&d, &kv_ctx(&snap(v), None, None, None), 7, 12)
            .join("\n");
        assert!(text.contains("KV cache 91% avg in fired windows, 98% peak (threshold: 88%)."));
        assert!(!text.contains("peak in fired windows"));
    }

    #[test]
    fn resolve_observed_floors_h100_boot_log_ground_truth() {
        // Source: H100 boot log Jul 16, kv_cache_max_concurrency = 24.64
        let (n, label) = resolve_r2_kv_capacity(Some(24.64), Some(99), false);
        assert_eq!(n, Some(24));
        assert_eq!(label, KvCapacityLabel::Observed);
    }

    #[test]
    fn seat_bullet_is_always_direction_only() {
        // Full-context caps must not cap operator actions (2026-07-30).
        fn pressure_text(v: VllmRawMetrics, kv_max_seqs: Option<u32>) -> String {
            format_kv_cache_pressure_fired(
                &detail(98.0, false),
                &kv_ctx(&snap(v), Some(8192), Some(30.0), kv_max_seqs),
                3,
                4,
            )
            .join("\n")
        }
        let cases = [
            (
                VllmRawMetrics {
                    kv_cache_usage_perc: Some(95.0),
                    max_num_seqs: Some(154),
                    num_requests_running_peak: Some(14.0),
                    cache_config: CacheConfigLabels {
                        kv_cache_max_concurrency: Some(24.64),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                Some(24u32),
            ),
            (
                VllmRawMetrics {
                    kv_cache_usage_perc: Some(95.0),
                    max_num_seqs: Some(154),
                    num_requests_running_peak: Some(45.0),
                    ..Default::default()
                },
                Some(33),
            ),
            (
                VllmRawMetrics {
                    kv_cache_usage_perc: Some(95.0),
                    max_num_seqs: Some(16),
                    ..Default::default()
                },
                Some(18),
            ),
        ];
        for (i, (v, cap)) in cases.into_iter().enumerate() {
            let text = pressure_text(v, cap);
            assert!(
                text.contains("Lower --max-num-seqs to reduce KV demand"),
                "case {i}"
            );
            assert!(!text.contains('≤'), "case {i}: {text}");
        }
        assert_eq!(resolve_r2_kv_capacity(None, Some(18), false).0, Some(18));
        assert_eq!(
            resolve_r2_kv_capacity(None, Some(18), true).1,
            KvCapacityLabel::DerivedHybrid
        );
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
        assert!(t1.contains("Lower --max-num-seqs to reduce KV demand"));
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
        assert!(text.contains("Lower --max-num-seqs to reduce KV demand"));
    }

    #[test]
    fn fix_order_leads_with_model_len_when_p99_below_half() {
        // Source: live run 2026-07-17, short p99 vs max_model_len; shrink leads.
        // 5465 < 32768/2; projection at 5465 is 39, not observed 8.
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            prompt_tokens_p99: Some(5000.0),
            generation_tokens_p99: Some(465.0),
            generation_tokens_completed: Some(150.0),
            max_num_seqs: Some(256),
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
            max_num_seqs: Some(256),
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
            max_num_seqs: Some(256),
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
            max_num_seqs: Some(256),
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
            max_num_seqs: Some(256),
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
            max_num_seqs: Some(256),
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
            num_preemptions_per_sec: Some(0.05),
            // No p99 / low count → no shrink; still has max-num-seqs in cuts though.
            max_num_seqs: Some(256),
            ..Default::default()
        };
        // max-num-seqs always in cuts for non-crisis → header present.
        // Crisis without shrink: no Cuts header. Cause/layout follow snapshot rate.
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
            max_num_seqs: Some(256),
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
        assert!(text.contains("Observed avg 5.1k tokens per request, prompt + generation."));
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
            max_num_seqs: Some(256),
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
        assert!(text.contains("Observed avg prompt 1.1k tokens per request."));
        assert!(!text.contains("prompt + generation"));
        assert!(!text.contains("unavailable"));
    }

    /// Count `--max-num-seqs` mentions in the Fix block only (not Cause).
    fn seat_lines_in_fix(text: &str) -> usize {
        let Some(fix_at) = text.find("    Fix:") else {
            return 0;
        };
        let fix = &text[fix_at..];
        let end = fix.find("\n    Expected:").unwrap_or(fix.len());
        fix[..end].matches("--max-num-seqs").count()
    }

    #[test]
    fn seat_line_plain_when_sub_floor_shrink_leads_no_crisis_subline() {
        // Iter-1 shape: means path (target None) but p99s still short enough that
        // shrink leads. Seat lever must appear as plain form; preempt 0.01 is below bar.
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(93.0),
            kv_cache_peak_perc: Some(100.0),
            num_requests_waiting: Some(6.0),
            num_preemptions_per_sec: Some(0.01),
            generation_tokens_completed: Some(48.0),
            prompt_tokens_mean: Some(11000.0),
            generation_tokens_mean: Some(8200.0),
            prompt_tokens_p99: Some(5000.0),
            generation_tokens_p99: Some(465.0),
            max_num_seqs: Some(256),
            ..Default::default()
        };
        let lines = format_kv_cache_pressure_fired(
            &detail(93.0, true),
            &kv_ctx(&snap(v), Some(262144), Some(30.0), Some(16)),
            3,
            4,
        );
        let text = lines.join("\n");
        let shrink_idx = lines
            .iter()
            .position(|l| l.contains("Lower --max-model-len") && l.contains("Observed avg"))
            .expect("sub-floor shrink");
        let seat_idx = lines
            .iter()
            .position(|l| l.contains("Lower --max-num-seqs") && !l.contains("Then lower"))
            .expect("plain seat");
        assert!(shrink_idx < seat_idx, "shrink leads, seat follows");
        assert!(!text.contains("Then lower --max-num-seqs"));
        assert!(!text.contains("Cuts throughput. Revert after pressure clears."));
        assert_eq!(seat_lines_in_fix(&text), 1);
    }

    #[test]
    fn seat_line_then_form_when_named_target_projects() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            prompt_tokens_p99: Some(5000.0),
            generation_tokens_p99: Some(465.0),
            generation_tokens_completed: Some(150.0),
            max_num_seqs: Some(256),
            cache_config: CacheConfigLabels {
                block_size: Some(16),
                num_gpu_blocks: Some(390),
                mamba_block_size: Some(784),
                kv_cache_max_concurrency: Some(8.667),
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
            &kv_ctx(&snap(v), Some(32768), None, Some(8)),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("Then lower --max-num-seqs to reduce KV demand"));
        assert!(!text.contains("• Lower --max-num-seqs to"));
        assert!(!text.contains("Cuts throughput. Revert after pressure clears."));
        assert_eq!(seat_lines_in_fix(&text), 1);
    }

    #[test]
    fn seat_line_then_form_dense_geometry_without_mamba_block() {
        let dense_model = crate::context::ModelArch {
            num_kv_heads: Some(8),
            head_dim: Some(128),
            num_layers: Some(32),
            ..Default::default()
        };
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            prompt_tokens_p99: Some(5000.0),
            generation_tokens_p99: Some(465.0),
            generation_tokens_completed: Some(150.0),
            max_num_seqs: Some(256),
            cache_config: CacheConfigLabels {
                block_size: Some(16),
                num_gpu_blocks: Some(390),
                kv_cache_max_concurrency: Some(8.667),
                ..Default::default()
            },
            ..Default::default()
        };
        let snap = snap(v);
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
                config_max_num_seqs: snap.vllm.max_num_seqs,
                capacity_label: KvCapacityLabel::Derived,
                fp8_compiler_available: false,
                model: Some(&dense_model),
                tp: None,
                kv_cache_dtype: snap.vllm.cache_config.cache_dtype.as_deref(),
            },
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("Then lower --max-num-seqs to reduce KV demand"));
        assert!(!text.contains("• Lower --max-num-seqs to"));
        assert_eq!(seat_lines_in_fix(&text), 1);
    }

    #[test]
    fn seat_line_plain_when_named_target_projection_fails() {
        // Named p99 target, shrink leads, but no page geometry and no model →
        // capacity_at_hypothetical_max_len is None → plain seat, not Then.
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            prompt_tokens_p99: Some(5000.0),
            generation_tokens_p99: Some(465.0),
            generation_tokens_completed: Some(150.0),
            max_num_seqs: Some(256),
            ..Default::default()
        };
        let text = format_kv_cache_pressure_fired(
            &KvCachePressureDetail {
                kv_cache_usage_perc: Some(90.0),
                kv_peak_pct: Some(90.0),
                preemptions_active: false,
                queue_backpressure: true,
            },
            &kv_ctx(&snap(v), Some(32768), None, Some(8)),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("Observed p99"));
        assert!(text.contains("Lower --max-num-seqs"));
        assert!(!text.contains("Then lower --max-num-seqs"));
        assert_eq!(seat_lines_in_fix(&text), 1);
    }

    #[test]
    fn seat_line_plain_crisis_subline_when_eviction_active() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(93.0),
            num_requests_waiting: Some(6.0),
            num_preemptions_per_sec: Some(0.05),
            generation_tokens_completed: Some(48.0),
            prompt_tokens_mean: Some(11000.0),
            generation_tokens_mean: Some(8200.0),
            prompt_tokens_p99: Some(5000.0),
            generation_tokens_p99: Some(465.0),
            max_num_seqs: Some(256),
            ..Default::default()
        };
        let lines = format_kv_cache_pressure_fired(
            &detail(93.0, true),
            &kv_ctx(&snap(v), Some(262144), Some(30.0), Some(16)),
            3,
            4,
        );
        let seat_idx = lines
            .iter()
            .position(|l| l.contains("Lower --max-num-seqs") && !l.contains("Then lower"))
            .expect("plain seat");
        assert!(
            lines[seat_idx + 1].contains("Cuts throughput. Revert after pressure clears."),
            "crisis subline attaches to plain seat only"
        );
        assert!(!lines.join("\n").contains("Then lower --max-num-seqs"));
        assert_eq!(seat_lines_in_fix(&lines.join("\n")), 1);
    }

    #[test]
    fn seat_line_absent_when_lever_unavailable() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_waiting: Some(5.0),
            prompt_tokens_p99: Some(5000.0),
            generation_tokens_p99: Some(465.0),
            generation_tokens_completed: Some(150.0),
            // No scrape max_num_seqs; config also None via kv_ctx_config.
            cache_config: CacheConfigLabels {
                block_size: Some(16),
                num_gpu_blocks: Some(390),
                mamba_block_size: Some(784),
                kv_cache_max_concurrency: Some(8.667),
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
            &kv_ctx_config(&snap(v), Some(32768), None, Some(8), None),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("Lower --max-model-len"));
        assert_eq!(seat_lines_in_fix(&text), 0);
    }

    #[test]
    fn seat_line_exactly_one_across_fired_shapes_when_lever_exists() {
        let shapes: &[String] = &[
            {
                let v = VllmRawMetrics {
                    kv_cache_usage_perc: Some(90.0),
                    num_requests_waiting: Some(5.0),
                    max_num_seqs: Some(256),
                    ..Default::default()
                };
                format_kv_cache_pressure_fired(
                    &KvCachePressureDetail {
                        kv_cache_usage_perc: Some(90.0),
                        kv_peak_pct: Some(90.0),
                        preemptions_active: false,
                        queue_backpressure: true,
                    },
                    &kv_ctx(&snap(v), None, Some(30.0), Some(18)),
                    3,
                    4,
                )
                .join("\n")
            },
            {
                let v = VllmRawMetrics {
                    kv_cache_usage_perc: Some(93.0),
                    num_requests_waiting: Some(6.0),
                    num_preemptions_per_sec: Some(0.01),
                    generation_tokens_completed: Some(48.0),
                    prompt_tokens_mean: Some(11000.0),
                    generation_tokens_mean: Some(8200.0),
                    prompt_tokens_p99: Some(5000.0),
                    generation_tokens_p99: Some(465.0),
                    max_num_seqs: Some(256),
                    ..Default::default()
                };
                format_kv_cache_pressure_fired(
                    &detail(93.0, true),
                    &kv_ctx(&snap(v), Some(262144), Some(30.0), Some(16)),
                    3,
                    4,
                )
                .join("\n")
            },
            {
                let v = VllmRawMetrics {
                    kv_cache_usage_perc: Some(90.0),
                    num_requests_waiting: Some(5.0),
                    prompt_tokens_p99: Some(5000.0),
                    generation_tokens_p99: Some(465.0),
                    generation_tokens_completed: Some(150.0),
                    max_num_seqs: Some(256),
                    cache_config: CacheConfigLabels {
                        block_size: Some(16),
                        num_gpu_blocks: Some(390),
                        mamba_block_size: Some(784),
                        kv_cache_max_concurrency: Some(8.667),
                        ..Default::default()
                    },
                    ..Default::default()
                };
                format_kv_cache_pressure_fired(
                    &KvCachePressureDetail {
                        kv_cache_usage_perc: Some(90.0),
                        kv_peak_pct: Some(90.0),
                        preemptions_active: false,
                        queue_backpressure: true,
                    },
                    &kv_ctx(&snap(v), Some(32768), None, Some(8)),
                    3,
                    4,
                )
                .join("\n")
            },
        ];
        for text in shapes {
            assert_eq!(
                seat_lines_in_fix(text),
                1,
                "exactly one seat line when lever exists:\n{text}"
            );
        }
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
            max_num_seqs: Some(256),
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
        assert!(!text.contains("Lower --max-num-seqs to reduce KV demand"));
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
            max_num_seqs: Some(256),
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
        let header_idx = lines
            .iter()
            .position(|l| l.as_str() == LAST_RESORT_HEADER)
            .expect("Last resort header");
        let idx = lines
            .iter()
            .position(|l| l.contains("Set --kv-offloading-size"))
            .expect("offload bullet");
        assert_eq!(idx, header_idx + 1, "bullet immediately under Last resort");
        if let Some(gpu_idx) = lines
            .iter()
            .position(|l| l.contains("Raise --gpu-memory-utilization"))
        {
            assert!(gpu_idx < header_idx, "offload after safe gpu-mem lever");
        }
        if let Some(fp8_idx) = lines
            .iter()
            .position(|l| l.contains("Switch --kv-cache-dtype fp8"))
        {
            assert!(fp8_idx < header_idx, "offload after safe fp8 lever");
        }
        if let Some(safe_idx) = lines.iter().position(|l| l == "    Safe to apply:") {
            assert!(
                safe_idx < header_idx,
                "Last resort must follow Safe to apply"
            );
            let safe_end = lines
                .iter()
                .position(|l| l == "    Cuts throughput:")
                .or_else(|| lines.iter().position(|l| l.as_str() == LAST_RESORT_HEADER))
                .unwrap_or(lines.len());
            for line in &lines[safe_idx + 1..safe_end] {
                assert!(
                    !line.contains("kv-offloading-size"),
                    "offload must never sit inside Safe to apply: {line:?}"
                );
            }
            if let Some(cuts_idx) = lines.iter().position(|l| l == "    Cuts throughput:") {
                assert!(cuts_idx < header_idx, "Last resort after Cuts throughput");
            }
        }
        assert!(
            lines[idx].starts_with("      •"),
            "bullet indent 6 spaces: {:?}",
            lines[idx]
        );
        assert_eq!(
            lines[idx + 1],
            format!("        {KV_OFFLOAD_DOWNSIDE}"),
            "downside subline"
        );
        assert_eq!(
            lines[idx + 2],
            format!("        {}", KV_OFFLOAD_SUBLINE_FALLBACK.trim_start()),
            "host RAM subline indent"
        );
        if lines.get(idx + 3).is_some_and(|l| l.is_empty()) {
            assert_eq!(lines[idx + 3], String::new(), "blank after host subline");
        }
        let fix_idx = lines
            .iter()
            .position(|l| l == "    Fix:")
            .expect("Fix header");
        let section_end = lines
            .iter()
            .position(|l| l.starts_with("    Expected:"))
            .unwrap_or(lines.len());
        assert!(fix_idx < header_idx, "Last resort under Fix");
        // Offload is last among fix bullets (after safe/cuts).
        let bullets: Vec<&String> = lines[fix_idx + 1..section_end]
            .iter()
            .filter(|l| l.starts_with("      •"))
            .collect();
        let offload_pos = bullets
            .iter()
            .position(|l| l.contains("Set --kv-offloading-size"))
            .expect("offload bullet in fix list");
        assert_eq!(
            offload_pos,
            bullets.len() - 1,
            "offload must be the last fix bullet"
        );
    }

    fn format_offload_three_paths(
        cache: CacheConfigLabels,
    ) -> (Vec<String>, Vec<String>, Vec<String>) {
        let mut v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            cache_config: cache.clone(),
            max_num_seqs: Some(256),
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
        let (crisis, _, _) = format_offload_three_paths(offload_cache(KvOffloadState::Off));
        let offload_idx = crisis
            .iter()
            .position(|l| l.contains("Set --kv-offloading-size"))
            .expect("offload bullet");
        let host_subline = &crisis[offload_idx + 2];
        let verify_subline = format!("        {}", DEAD_END_VERIFY_SUBLINE.trim_start());
        assert_eq!(
            host_subline
                .chars()
                .take_while(|c| *c == ' ')
                .collect::<String>(),
            verify_subline
                .chars()
                .take_while(|c| *c == ' ')
                .collect::<String>(),
            "offload host and dead-end sublines must share indent"
        );
        assert_eq!(
            crisis[offload_idx + 1],
            format!("        {KV_OFFLOAD_DOWNSIDE}")
        );
        assert_eq!(
            *host_subline,
            format!("        {}", KV_OFFLOAD_SUBLINE_FALLBACK.trim_start())
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
            assert!(
                !text.contains(LAST_RESORT_HEADER),
                "absent label must not emit Last resort: {text}"
            );
        }
    }

    #[test]
    fn kv_offload_none_literal_bullet_all_paths() {
        use crate::collectors::KvOffloadState;
        let (c, n, b) = format_offload_three_paths(offload_cache(KvOffloadState::Off));
        // Eviction only on the crisis fixture; queue-only paths stay quiet.
        assert_offload_block(&c);
        assert_no_offload_bullet(&[n.join("\n"), b.join("\n")]);
    }

    /// Journey-shaped scrape with host memory so derivation can resolve.
    /// `host` sets the supply side (`0.5 * usable`); keep preempt on all paths.
    fn format_offload_three_paths_derived(
        state: crate::collectors::KvOffloadState,
        host: crate::collectors::HostMemoryFacts,
        headroom_si: f64,
    ) -> (Vec<String>, Vec<String>, Vec<String>) {
        use crate::collectors::RawSnapshotFixture;
        let vllm = |preempt: Option<f64>| VllmRawMetrics {
            kv_cache_usage_perc: Some(97.0),
            num_requests_running: Some(34.0),
            num_requests_waiting: Some(36.0),
            num_requests_waiting_peak: Some(36.0),
            kv_frac_per_running_peak: Some(0.97 / 34.0),
            num_preemptions_per_sec: preempt,
            window_duration_secs: Some(120.0),
            max_num_seqs: Some(175),
            prompt_tokens_mean: Some(64.0),
            cache_config: CacheConfigLabels {
                num_gpu_blocks: Some(1000),
                block_size: Some(16),
                ..offload_cache(state)
            },
            ..Default::default()
        };
        let snap = |preempt: Option<f64>| {
            RawSnapshotFixture::default()
                .vllm(vllm(preempt))
                .host_memory(Some(host))
                .build()
        };
        let crisis_snap = snap(Some(0.27));
        let crisis = format_kv_cache_pressure_fired(
            &detail(97.0, true),
            &kv_ctx(&crisis_snap, None, Some(headroom_si), None),
            3,
            4,
        );
        // Keep preempt rate for derivation even when the detail is non-crisis.
        let non_crisis_snap = snap(Some(0.27));
        let non_crisis = format_kv_cache_pressure_fired(
            &KvCachePressureDetail {
                kv_cache_usage_perc: Some(97.0),
                kv_peak_pct: Some(100.0),
                preemptions_active: false,
                queue_backpressure: true,
            },
            &kv_ctx(&non_crisis_snap, None, Some(headroom_si), None),
            3,
            4,
        );
        let backlog_snap = snap(Some(0.27));
        let backlog = format_kv_admission_backlog_issue(
            &KvAdmissionBacklogDetail {
                kv_cache_usage_perc: 97.0,
                kv_peak_pct: Some(100.0),
                admission_ratio: 36.0 / 70.0,
                requests_waiting: 36.0,
                requests_running: 34.0,
                free_kv_tokens: 100.0,
                demand_tokens: 200.0,
            },
            75,
            &kv_ctx(&backlog_snap, None, Some(headroom_si), None),
            3,
            4,
        );
        (crisis, non_crisis, backlog)
    }

    fn assert_no_offload_bullet(texts: &[String]) {
        for text in texts {
            assert!(
                !text.contains("kv-offloading-size"),
                "expected quiet offload arm:\n{text}"
            );
        }
    }

    #[test]
    fn kv_offload_absent_when_eviction_quiet() {
        use crate::collectors::KvOffloadState;
        let (_, n, b) = format_offload_three_paths(offload_cache(KvOffloadState::Off));
        for text in [n.join("\n"), b.join("\n")] {
            assert!(
                !text.contains("kv-offloading-size") && !text.contains(LAST_RESORT_HEADER),
                "no offload without eviction:\n{text}"
            );
        }
    }

    /// Journey-shaped scrape with host memory so derivation can resolve.
    /// `host` sets the supply side (`0.5 * usable`); keep preempt on all paths.
    fn format_offload_three_paths_derived(
        state: crate::collectors::KvOffloadState,
        host: crate::collectors::HostMemoryFacts,
        headroom_si: f64,
    ) -> (Vec<String>, Vec<String>, Vec<String>) {
        use crate::collectors::RawSnapshotFixture;
        let vllm = |preempt: Option<f64>| VllmRawMetrics {
            kv_cache_usage_perc: Some(97.0),
            num_requests_running: Some(34.0),
            num_requests_waiting: Some(36.0),
            num_requests_waiting_peak: Some(36.0),
            kv_frac_per_running_peak: Some(0.97 / 34.0),
            num_preemptions_per_sec: preempt,
            window_duration_secs: Some(120.0),
            max_num_seqs: Some(175),
            prompt_tokens_mean: Some(64.0),
            cache_config: CacheConfigLabels {
                num_gpu_blocks: Some(1000),
                block_size: Some(16),
                ..offload_cache(state)
            },
            ..Default::default()
        };
        let snap = |preempt: Option<f64>| {
            RawSnapshotFixture::default()
                .vllm(vllm(preempt))
                .host_memory(Some(host))
                .build()
        };
        let crisis_snap = snap(Some(0.27));
        let crisis = format_kv_cache_pressure_fired(
            &detail(97.0, true),
            &kv_ctx(&crisis_snap, None, Some(headroom_si), None),
            3,
            4,
        );
        // Keep preempt rate for derivation even when the detail is non-crisis.
        let non_crisis_snap = snap(Some(0.27));
        let non_crisis = format_kv_cache_pressure_fired(
            &KvCachePressureDetail {
                kv_cache_usage_perc: Some(97.0),
                kv_peak_pct: Some(100.0),
                preemptions_active: false,
                queue_backpressure: true,
            },
            &kv_ctx(&non_crisis_snap, None, Some(headroom_si), None),
            3,
            4,
        );
        let backlog_snap = snap(Some(0.27));
        let backlog = format_kv_admission_backlog_issue(
            &KvAdmissionBacklogDetail {
                kv_cache_usage_perc: 97.0,
                kv_peak_pct: Some(100.0),
                admission_ratio: 36.0 / 70.0,
                requests_waiting: 36.0,
                requests_running: 34.0,
                free_kv_tokens: 100.0,
                demand_tokens: 200.0,
            },
            75,
            &kv_ctx(&backlog_snap, None, Some(headroom_si), None),
            3,
            4,
        );
        (crisis, non_crisis, backlog)
    }

    fn assert_no_offload_bullet(texts: &[String]) {
        for text in texts {
            assert!(
                !text.contains("kv-offloading-size"),
                "expected quiet offload arm:\n{text}"
            );
        }
    }

    #[test]
    fn kv_offload_enabled_derived_below_set_quiet_all_paths() {
        use crate::collectors::KvOffloadState;
        // Journey derives 23; set 64 -> never prescribe a shrink.
        let headroom_si = 24.8 * super::GIB_BYTES / 1e9;
        let (c, n, b) = format_offload_three_paths_derived(
            KvOffloadState::Enabled(64.0),
            journey_host_memory(),
            headroom_si,
        );
        assert_no_offload_bullet(&[c.join("\n"), n.join("\n"), b.join("\n")]);
    }

    #[test]
    fn kv_offload_enabled_4_reoffers_12_all_paths() {
        use crate::collectors::{HostMemoryFacts, KvOffloadState};
        // Supply = 0.5 * 24 GiB = 12; journey demand > 12 -> derived 12.
        let host = HostMemoryFacts {
            available_bytes: 24 << 30,
            container_limit_bytes: None,
        };
        let headroom_si = 24.8 * super::GIB_BYTES / 1e9;
        assert_eq!(
            super::resolve_kv_offload_size_gib(super::KvOffloadSizeInput {
                host_memory: Some(host),
                pool_bytes: Some(journey_pool_bytes()),
                kv_frac_per_running_peak: Some(0.97 / 34.0),
                preempt_per_sec: Some(0.27),
                run_duration_secs: Some(120.0),
                peak_waiting: Some(36.0),
            }),
            Some(12)
        );
        let (c, n, b) =
            format_offload_three_paths_derived(KvOffloadState::Enabled(4.0), host, headroom_si);
        for text in [c.join("\n"), n.join("\n"), b.join("\n")] {
            assert!(
                text.contains("Set --kv-offloading-size 12 (est) to hold evicted KV"),
                "re-offer 12:\n{text}"
            );
            assert!(
                text.contains(LAST_RESORT_HEADER),
                "Last resort header:\n{text}"
            );
            assert!(
                text.contains(KV_OFFLOAD_DOWNSIDE),
                "downside subline:\n{text}"
            );
            assert!(text.contains("Host RAM available: 24 GiB, container limit none."));
            assert!(!text.contains("Set --kv-offloading-size (GiB)"));
            let lines: Vec<&str> = text.lines().collect();
            let header_idx = lines
                .iter()
                .position(|l| *l == LAST_RESORT_HEADER)
                .expect("Last resort");
            let idx = lines
                .iter()
                .position(|l| l.contains("Set --kv-offloading-size 12 (est)"))
                .expect("sized bullet");
            assert_eq!(idx, header_idx + 1);
            if let Some(safe_idx) = lines.iter().position(|l| *l == "    Safe to apply:") {
                assert!(safe_idx < header_idx, "never inside Safe");
                for line in &lines[safe_idx + 1..header_idx] {
                    assert!(
                        !line.contains("kv-offloading-size"),
                        "offload leaked into Safe: {line:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn kv_offload_enabled_equal_derived_quiet() {
        use crate::collectors::{HostMemoryFacts, KvOffloadState};
        // Supply clips to 12; set 12 -> equality is a no-op.
        let host = HostMemoryFacts {
            available_bytes: 24 << 30,
            container_limit_bytes: None,
        };
        let headroom_si = 24.8 * super::GIB_BYTES / 1e9;
        let (c, n, b) =
            format_offload_three_paths_derived(KvOffloadState::Enabled(12.0), host, headroom_si);
        assert_no_offload_bullet(&[c.join("\n"), n.join("\n"), b.join("\n")]);
    }

    #[test]
    fn kv_offload_enabled_missing_preempt_quiet_no_numberless() {
        use crate::collectors::{KvOffloadState, RawSnapshotFixture};
        let headroom_si = 24.8 * super::GIB_BYTES / 1e9;
        let snap = RawSnapshotFixture::default()
            .vllm(VllmRawMetrics {
                kv_cache_usage_perc: Some(97.0),
                num_requests_running: Some(34.0),
                num_requests_waiting_peak: Some(36.0),
                kv_frac_per_running_peak: Some(0.97 / 34.0),
                num_preemptions_per_sec: None,
                window_duration_secs: Some(120.0),
                max_num_seqs: Some(175),
                cache_config: offload_cache(KvOffloadState::Enabled(4.0)),
                ..Default::default()
            })
            .host_memory(Some(journey_host_memory()))
            .build();
        let text = format_kv_cache_pressure_fired(
            &detail(97.0, true),
            &kv_ctx(&snap, None, Some(headroom_si), None),
            3,
            4,
        )
        .join("\n");
        assert!(
            !text.contains("kv-offloading-size"),
            "underivable Enabled must not emit numberless bullet:\n{text}"
        );
    }

    #[test]
    fn kv_offload_enabled_4_5_whole_gib_equality_quiet() {
        use crate::collectors::{HostMemoryFacts, KvOffloadState, RawSnapshotFixture};
        // Demand 4.1 GiB -> derived ceil 5; set ceil(4.5)=5 -> quiet.
        let host = HostMemoryFacts {
            available_bytes: 1921 << 30,
            container_limit_bytes: Some(234 << 30),
        };
        let pool = journey_pool_bytes();
        let demand_gib = 4.1;
        let parked = 1.0_f64;
        let frac = (demand_gib * super::GIB_BYTES) / ((pool as f64) * parked);
        assert_eq!(
            super::resolve_kv_offload_size_gib(super::KvOffloadSizeInput {
                host_memory: Some(host),
                pool_bytes: Some(pool),
                kv_frac_per_running_peak: Some(frac),
                preempt_per_sec: Some(1.0),
                run_duration_secs: Some(1.0),
                peak_waiting: Some(1.0),
            }),
            Some(5)
        );
        assert_eq!(
            super::ceil_bytes_to_whole_gib(4.5 * super::GIB_BYTES),
            Some(5)
        );
        let headroom_si = 24.8 * super::GIB_BYTES / 1e9;
        let snap = RawSnapshotFixture::default()
            .vllm(VllmRawMetrics {
                kv_cache_usage_perc: Some(97.0),
                num_requests_running: Some(34.0),
                num_requests_waiting_peak: Some(1.0),
                kv_frac_per_running_peak: Some(frac),
                num_preemptions_per_sec: Some(1.0),
                window_duration_secs: Some(1.0),
                max_num_seqs: Some(175),
                cache_config: offload_cache(KvOffloadState::Enabled(4.5)),
                ..Default::default()
            })
            .host_memory(Some(host))
            .build();
        let text = format_kv_cache_pressure_fired(
            &detail(97.0, true),
            &kv_ctx(&snap, None, Some(headroom_si), None),
            3,
            4,
        )
        .join("\n");
        assert!(
            !text.contains("kv-offloading-size"),
            "whole-GiB equality must stay quiet:\n{text}"
        );
    }

    #[test]
    fn kv_offload_enabled_underivable_quiet_no_numberless() {
        use crate::collectors::KvOffloadState;
        // No host memory -> derivation None; Enabled must not emit numberless bullet.
        let (c, n, b) = format_offload_three_paths(offload_cache(KvOffloadState::Enabled(4.0)));
        assert_no_offload_bullet(&[c.join("\n"), n.join("\n"), b.join("\n")]);
    }

    #[test]
    fn kv_offload_enabled_supply_clip_at_or_below_set_quiet() {
        use crate::collectors::{HostMemoryFacts, KvOffloadState};
        // Supply = 0.5 * 8 GiB = 4; journey demand >> 4 -> derived 4 == set 4.
        let host = HostMemoryFacts {
            available_bytes: 8 << 30,
            container_limit_bytes: None,
        };
        let headroom_si = 24.8 * super::GIB_BYTES / 1e9;
        assert_eq!(
            super::resolve_kv_offload_size_gib(super::KvOffloadSizeInput {
                host_memory: Some(host),
                pool_bytes: Some(journey_pool_bytes()),
                kv_frac_per_running_peak: Some(0.97 / 34.0),
                preempt_per_sec: Some(0.27),
                run_duration_secs: Some(120.0),
                peak_waiting: Some(36.0),
            }),
            Some(4)
        );
        let (c, n, b) =
            format_offload_three_paths_derived(KvOffloadState::Enabled(4.0), host, headroom_si);
        assert_no_offload_bullet(&[c.join("\n"), n.join("\n"), b.join("\n")]);
    }

    #[test]
    fn kv_offload_size_zero_bullet_all_paths() {
        use crate::collectors::KvOffloadState;
        // Zero parses to Off.
        let (c, n, b) = format_offload_three_paths(offload_cache(KvOffloadState::Off));
        assert_offload_block(&c);
        assert_no_offload_bullet(&[n.join("\n"), b.join("\n")]);
    }

    #[test]
    fn kv_offload_garbage_no_bullet_no_panic() {
        use crate::collectors::KvOffloadState;
        let cache = offload_cache(KvOffloadState::Unreadable);
        let (c, n, b) = format_offload_three_paths(cache);
        for text in [c.join("\n"), n.join("\n"), b.join("\n")] {
            assert!(!text.contains("kv-offloading-size"), "garbage: {text}");
        }
    }

    #[test]
    fn kv_offload_subline_shows_measured_host_memory() {
        use crate::collectors::HostMemoryFacts;
        let facts = HostMemoryFacts {
            available_bytes: 1921 << 30,
            container_limit_bytes: Some(234 << 30),
        };
        assert_eq!(
            super::format_kv_offload_subline(Some(facts)),
            "Host RAM available: 1921 GiB, container limit 234 GiB."
        );
    }

    #[test]
    fn kv_offload_subline_unlimited_container_renders_none() {
        use crate::collectors::HostMemoryFacts;
        let facts = HostMemoryFacts {
            available_bytes: 64 << 30,
            container_limit_bytes: None,
        };
        assert_eq!(
            super::format_kv_offload_subline(Some(facts)),
            "Host RAM available: 64 GiB, container limit none."
        );
    }

    #[test]
    fn kv_offload_subline_fallback_when_host_memory_unreadable() {
        assert_eq!(
            super::format_kv_offload_subline(None),
            KV_OFFLOAD_SUBLINE_FALLBACK
        );
    }

    #[test]
    fn kv_offload_rendered_subline_uses_snapshot_host_memory() {
        use crate::collectors::{HostMemoryFacts, KvOffloadState, RawSnapshotFixture};
        let facts = HostMemoryFacts {
            available_bytes: 1921 << 30,
            container_limit_bytes: Some(234 << 30),
        };
        let snap = RawSnapshotFixture::default()
            .vllm(VllmRawMetrics {
                kv_cache_usage_perc: Some(90.0),
                num_requests_waiting: Some(5.0),
                num_preemptions_per_sec: Some(0.05),
                cache_config: offload_cache(KvOffloadState::Off),
                max_num_seqs: Some(256),
                ..Default::default()
            })
            .host_memory(Some(facts))
            .build();
        let text = format_kv_cache_pressure_fired(
            &detail(90.0, true),
            &kv_ctx(&snap, None, Some(30.0), None),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains(LAST_RESORT_HEADER));
        assert!(text.contains(KV_OFFLOAD_DOWNSIDE));
        assert!(text.contains("Host RAM available: 1921 GiB, container limit 234 GiB."));
        assert!(!text.contains(KV_OFFLOAD_SUBLINE_FALLBACK));
    }

    fn journey_host_memory() -> crate::collectors::HostMemoryFacts {
        crate::collectors::HostMemoryFacts {
            available_bytes: 1921 << 30,
            container_limit_bytes: Some(234 << 30),
        }
    }

    fn journey_pool_bytes() -> u64 {
        (24.8 * super::GIB_BYTES).round() as u64
    }

    fn journey_offload_input() -> super::KvOffloadSizeInput {
        super::KvOffloadSizeInput {
            host_memory: Some(journey_host_memory()),
            pool_bytes: Some(journey_pool_bytes()),
            kv_frac_per_running_peak: Some(0.97 / 34.0),
            preempt_per_sec: Some(0.27),
            run_duration_secs: Some(120.0),
            peak_waiting: Some(36.0),
        }
    }

    #[test]
    fn kv_offload_journey_fixture_resolves_23() {
        assert_eq!(
            super::resolve_kv_offload_size_gib(journey_offload_input()),
            Some(23)
        );
    }

    #[test]
    fn kv_offload_kv_percent_units_are_0_to_100() {
        // Window stores 97 (not 0.97); per_seq must divide by 100.
        let mut input = journey_offload_input();
        input.kv_frac_per_running_peak = Some(97.0 / 34.0); // forgot /100
        let wrong = super::resolve_kv_offload_size_gib(input);
        assert_ne!(
            wrong,
            Some(23),
            "forgetting /100 must not yield journey size"
        );
        let mut input = journey_offload_input();
        input.kv_frac_per_running_peak = Some(97.0 / 100.0 / 34.0);
        assert_eq!(super::resolve_kv_offload_size_gib(input), Some(23));
    }

    #[test]
    fn kv_offload_peak_ratio_uses_max_window_not_mean() {
        // Windows: kv50/run50 → 0.01; kv95/run10 → 0.095. Peak is second.
        let peak = (95.0_f64 / 100.0 / 10.0).max(50.0 / 100.0 / 50.0);
        let mean = ((50.0_f64 / 100.0 / 50.0) + (95.0 / 100.0 / 10.0)) / 2.0;
        assert!((peak - 0.095).abs() < 1e-12);
        assert!((mean - 0.0525).abs() < 1e-12);

        let mut peak_input = journey_offload_input();
        peak_input.kv_frac_per_running_peak = Some(peak);
        let mut mean_input = journey_offload_input();
        mean_input.kv_frac_per_running_peak = Some(mean);
        let from_peak = super::resolve_kv_offload_size_gib(peak_input).unwrap();
        let from_mean = super::resolve_kv_offload_size_gib(mean_input).unwrap();
        assert!(from_peak > from_mean, "mean ratio must not pass as peak");
    }

    #[test]
    fn kv_offload_flow_vs_stock_takes_min() {
        let mut flow_wins = journey_offload_input();
        flow_wins.preempt_per_sec = Some(32.0 / 120.0);
        flow_wins.peak_waiting = Some(200.0);
        // parked = 32
        let mut stock_wins = journey_offload_input();
        stock_wins.preempt_per_sec = Some(90.0 / 120.0);
        stock_wins.peak_waiting = Some(36.0);
        // parked = 36
        assert_eq!(
            super::resolve_kv_offload_size_gib(flow_wins),
            super::resolve_kv_offload_size_gib(journey_offload_input())
        );
        let stock = super::resolve_kv_offload_size_gib(stock_wins).unwrap();
        let journey = super::resolve_kv_offload_size_gib(journey_offload_input()).unwrap();
        assert!(stock > journey, "parked 36 must exceed parked 32");
    }

    #[test]
    fn kv_offload_supply_clips_large_demand() {
        let mut input = journey_offload_input();
        // Huge pool → demand >> supply (0.5 × 234 GiB = 117)
        input.pool_bytes = Some(400 << 30);
        input.kv_frac_per_running_peak = Some(1.0);
        input.preempt_per_sec = Some(10.0);
        input.peak_waiting = Some(100.0);
        assert_eq!(super::resolve_kv_offload_size_gib(input), Some(117));
    }

    #[test]
    fn kv_offload_degrades_without_host_memory() {
        let mut input = journey_offload_input();
        input.host_memory = None;
        assert_eq!(super::resolve_kv_offload_size_gib(input), None);
    }

    #[test]
    fn kv_offload_degrades_without_pool() {
        let mut input = journey_offload_input();
        input.pool_bytes = None;
        assert_eq!(super::resolve_kv_offload_size_gib(input), None);
    }

    #[test]
    fn kv_offload_degrades_without_kv_frac() {
        let mut input = journey_offload_input();
        input.kv_frac_per_running_peak = None;
        assert_eq!(super::resolve_kv_offload_size_gib(input), None);
    }

    #[test]
    fn kv_offload_degrades_without_preempt_rate() {
        let mut input = journey_offload_input();
        input.preempt_per_sec = None;
        assert_eq!(super::resolve_kv_offload_size_gib(input), None);
    }

    #[test]
    fn kv_offload_degrades_when_preempt_zero() {
        let mut input = journey_offload_input();
        input.preempt_per_sec = Some(0.0);
        assert_eq!(super::resolve_kv_offload_size_gib(input), None);
    }

    #[test]
    fn kv_offload_degrades_when_peak_waiting_absent_or_zero() {
        let mut absent = journey_offload_input();
        absent.peak_waiting = None;
        assert_eq!(super::resolve_kv_offload_size_gib(absent), None);
        let mut zero = journey_offload_input();
        zero.peak_waiting = Some(0.0);
        assert_eq!(super::resolve_kv_offload_size_gib(zero), None);
    }

    #[test]
    fn kv_offload_journey_renders_23_est_with_ram_subline() {
        use crate::collectors::{KvOffloadState, RawSnapshotFixture};
        // derived_budget_bytes is SI GB; feed SI equal to 24.8 binary GiB so pool
        // bytes match the journey arithmetic that resolves to 23.
        let headroom_si = 24.8 * super::GIB_BYTES / 1e9;
        let snap = RawSnapshotFixture::default()
            .vllm(VllmRawMetrics {
                kv_cache_usage_perc: Some(97.0),
                num_requests_running: Some(34.0),
                num_requests_waiting: Some(36.0),
                num_requests_waiting_peak: Some(36.0),
                kv_frac_per_running_peak: Some(0.97 / 34.0),
                num_preemptions_per_sec: Some(0.27),
                window_duration_secs: Some(120.0),
                max_num_seqs: Some(175),
                cache_config: offload_cache(KvOffloadState::Off),
                ..Default::default()
            })
            .host_memory(Some(journey_host_memory()))
            .build();
        let text = format_kv_cache_pressure_fired(
            &KvCachePressureDetail {
                kv_cache_usage_perc: Some(97.0),
                kv_peak_pct: Some(100.0),
                preemptions_active: true,
                queue_backpressure: true,
            },
            &kv_ctx(&snap, None, Some(headroom_si), None),
            3,
            4,
        )
        .join("\n");
        assert!(
            text.contains("Set --kv-offloading-size 23 (est) to hold evicted KV"),
            "journey render:\n{text}"
        );
        assert!(text.contains(LAST_RESORT_HEADER));
        assert!(text.contains(KV_OFFLOAD_DOWNSIDE));
        assert!(text.contains("Host RAM available: 1921 GiB, container limit 234 GiB."));
        assert!(!text.contains("Set --kv-offloading-size (GiB)"));
        let last = text.find(LAST_RESORT_HEADER).expect("Last resort");
        let off = text
            .find("Set --kv-offloading-size 23")
            .expect("sized offload");
        assert!(off > last, "bullet under Last resort");
        if let Some(safe) = text.find("    Safe to apply:") {
            assert!(safe < last, "Last resort after Safe");
        }
    }

    #[test]
    fn kv_offload_degrade_renders_directionless_bullet() {
        use crate::collectors::{KvOffloadState, RawSnapshotFixture};
        // Eviction active so Last resort emits; no host/pool so size is None.
        let snap = RawSnapshotFixture::default()
            .vllm(VllmRawMetrics {
                kv_cache_usage_perc: Some(97.0),
                num_requests_running: Some(34.0),
                num_requests_waiting_peak: Some(36.0),
                num_preemptions_per_sec: Some(0.05),
                window_duration_secs: Some(120.0),
                max_num_seqs: Some(175),
                cache_config: offload_cache(KvOffloadState::Off),
                ..Default::default()
            })
            .build();
        let text = format_kv_cache_pressure_fired(
            &detail(97.0, true),
            &kv_ctx(&snap, None, None, None),
            1,
            1,
        )
        .join("\n");
        assert!(text.contains(LAST_RESORT_HEADER));
        assert!(text.contains("Set --kv-offloading-size (GiB) to hold evicted KV"));
        assert!(!text.contains("(est)"));
        assert!(text.contains(KV_OFFLOAD_DOWNSIDE));
        assert!(text.contains(KV_OFFLOAD_SUBLINE_FALLBACK));
    }

    #[test]
    fn kv_offload_backlog_path_same_sized_bullet() {
        use crate::collectors::{KvOffloadState, RawSnapshotFixture};
        let headroom_si = 24.8 * super::GIB_BYTES / 1e9;
        let snap = RawSnapshotFixture::default()
            .vllm(VllmRawMetrics {
                kv_cache_usage_perc: Some(97.0),
                num_requests_running: Some(34.0),
                num_requests_waiting: Some(36.0),
                num_requests_waiting_peak: Some(36.0),
                kv_frac_per_running_peak: Some(0.97 / 34.0),
                num_preemptions_per_sec: Some(0.27),
                window_duration_secs: Some(120.0),
                max_num_seqs: Some(175),
                prompt_tokens_mean: Some(64.0),
                cache_config: CacheConfigLabels {
                    num_gpu_blocks: Some(1000),
                    block_size: Some(16),
                    ..offload_cache(KvOffloadState::Off)
                },
                ..Default::default()
            })
            .host_memory(Some(journey_host_memory()))
            .build();
        let ctx = kv_ctx(&snap, None, Some(headroom_si), None);
        let pressure = format_kv_cache_pressure_fired(&detail(97.0, true), &ctx, 3, 4).join("\n");
        let backlog = format_kv_admission_backlog_issue(
            &KvAdmissionBacklogDetail {
                kv_cache_usage_perc: 97.0,
                kv_peak_pct: Some(100.0),
                admission_ratio: 36.0 / 70.0,
                requests_waiting: 36.0,
                requests_running: 34.0,
                free_kv_tokens: 100.0,
                demand_tokens: 200.0,
            },
            75,
            &ctx,
            3,
            4,
        )
        .join("\n");
        let extract = |t: &str| {
            t.lines()
                .find(|l| l.contains("Set --kv-offloading-size"))
                .expect("offload bullet")
                .to_string()
        };
        assert_eq!(extract(&pressure), extract(&backlog));
        assert!(extract(&pressure).contains("23 (est)"));
        assert!(pressure.contains(LAST_RESORT_HEADER));
        assert!(backlog.contains(LAST_RESORT_HEADER));
        assert!(pressure.contains(KV_OFFLOAD_DOWNSIDE));
    }

    #[test]
    fn kv_offload_unlimited_container_supply_from_mem_available() {
        let facts = crate::collectors::HostMemoryFacts {
            available_bytes: 64 << 30,
            container_limit_bytes: None,
        };
        let mut input = journey_offload_input();
        input.host_memory = Some(facts);
        // supply = 0.5 × 64 = 32 GiB; journey demand ~23 → still 23
        assert_eq!(super::resolve_kv_offload_size_gib(input), Some(23));
        assert_eq!(
            super::format_kv_offload_subline(Some(facts)),
            "Host RAM available: 64 GiB, container limit none."
        );
        // Clip when demand exceeds half of MemAvailable alone
        let mut clipped = journey_offload_input();
        clipped.host_memory = Some(facts);
        clipped.pool_bytes = Some(400 << 30);
        clipped.kv_frac_per_running_peak = Some(1.0);
        clipped.preempt_per_sec = Some(10.0);
        clipped.peak_waiting = Some(100.0);
        assert_eq!(super::resolve_kv_offload_size_gib(clipped), Some(32));
    }

    #[test]
    fn kv_offload_printed_value_never_exceeds_reserve_fraction() {
        use crate::collectors::{KvOffloadState, RawSnapshotFixture};
        let usable = (journey_host_memory()
            .available_bytes
            .min(journey_host_memory().container_limit_bytes.unwrap())) as f64
            / super::GIB_BYTES;
        let cap = (super::KV_OFFLOAD_RESERVE_FRACTION * usable).floor() as u64;
        let headroom_si = 24.8 * super::GIB_BYTES / 1e9;
        let snap = RawSnapshotFixture::default()
            .vllm(VllmRawMetrics {
                kv_cache_usage_perc: Some(97.0),
                num_requests_running: Some(34.0),
                num_requests_waiting_peak: Some(36.0),
                kv_frac_per_running_peak: Some(0.97 / 34.0),
                num_preemptions_per_sec: Some(0.27),
                window_duration_secs: Some(120.0),
                max_num_seqs: Some(175),
                cache_config: offload_cache(KvOffloadState::Off),
                ..Default::default()
            })
            .host_memory(Some(journey_host_memory()))
            .build();
        let text = format_kv_cache_pressure_fired(
            &detail(97.0, true),
            &kv_ctx(&snap, None, Some(headroom_si), None),
            1,
            1,
        )
        .join("\n");
        let bullet = text
            .lines()
            .find(|l| l.contains("Set --kv-offloading-size"))
            .expect("bullet");
        if let Some(rest) = bullet.split("Set --kv-offloading-size ").nth(1) {
            if rest.starts_with("(GiB)") {
                return;
            }
            let n: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .expect("sized number");
            assert!(
                bullet.contains("(est)"),
                "sized bullet must carry (est): {bullet}"
            );
            assert!(n <= cap, "value {n} exceeds 0.5×usable ({cap})");
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
    fn r2_recommendation_withholds_seat_when_max_num_seqs_absent() {
        // Test helper must not invent 256; scrape and config both absent → no seat bullet.
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(95.0),
            num_preemptions_per_sec: Some(0.05),
            ..Default::default()
        };
        let snapshot = snap(v);
        assert!(snapshot.vllm.max_num_seqs.is_none());
        let r = r2_recommendation(R2RecommendationInput {
            snapshot: &snapshot,
            max_model_len: None,
            kv_headroom_gb: None,
            kv_max_seqs: None,
            capacity_label: KvCapacityLabel::Derived,
            windows_fired: 3,
            total_evaluable: 4,
            fp8_compiler_available: false,
        })
        .expect("fired");
        let text = r.display_lines.join("\n");
        assert!(!text.contains("Lower --max-num-seqs"));
        assert!(text.contains("Lower --max-model-len"));
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
        assert!(text.contains(LAST_RESORT_HEADER));
        assert!(text.contains("Set --kv-offloading-size"));
        assert!(text.contains(KV_OFFLOAD_DOWNSIDE));
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
        assert!(text.contains(LAST_RESORT_HEADER));
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
