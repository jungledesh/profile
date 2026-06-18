use crate::collectors::RawSnapshot;

use super::{skew_secs, Recommendation, MAX_OBSERVATION_SKEW_SECS};

/// 88% matches observed vLLM production eviction onset; 85% was too conservative.
const KV_CACHE_PRESSURE_MIN_PERC: f64 = 88.0;
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
const NVCC_PATH: &str = "/usr/local/cuda/bin/nvcc";
const FP8_KV_CACHE_FIX: &str = "    • Switch --kv-cache-dtype fp8 to halve KV memory footprint";

static NVCC_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn fp8_kv_cache_fix_bullet() -> String {
    // --kv-cache-dtype fp8 stores KV activations in fp8 via software cast — works on all GPUs
    // including A100. This is distinct from --quantization fp8 (weight quantization) which
    // requires native FP8 hardware and crashes on A100/Qwen3.6.
    let nvcc_present = *NVCC_AVAILABLE.get_or_init(|| std::path::Path::new(NVCC_PATH).exists());
    if nvcc_present {
        FP8_KV_CACHE_FIX.to_string()
    } else {
        format!("{FP8_KV_CACHE_FIX} (requires nvcc)")
    }
}

fn kv_headroom_gpu_mem_bullet(kv_headroom_gb: Option<f64>) -> String {
    match kv_headroom_gb {
        Some(h) if h >= KV_HEADROOM_SAFE_MIN_GB => {
            "    • Raise --gpu-memory-utilization (check vRAM header for avail mem) to expand KV pool"
                .to_string()
        }
        Some(_) => {
            "    • GPU at VRAM capacity: cannot raise --gpu-memory-utilization. Scale out or reduce max context length (--max-model-len)."
                .to_string()
        }
        None => "    • Raise --gpu-memory-utilization (check vRAM header for avail mem) to expand KV pool".to_string(),
    }
}

fn prefix_caching_fix_bullet(snapshot: &RawSnapshot) -> Option<String> {
    if snapshot.vllm.cache_config.enable_prefix_caching != Some(true)
        && snapshot.vllm.prompt_tokens_mean.is_some_and(|t| t >= 200.0)
    {
        Some(
            "    • Enable --enable-prefix-caching to share KV blocks across identical prompt prefixes"
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
    pub kv_cache_usage_perc: f64,
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
/// 1. Rate (velocity): preemptions/s > 0.02 — scheduler is actively evicting right now.
/// 2. Debt (static): num_requests_swapped ≥ 2 — sequences parked on CPU. This is a
///    gauge, not a delta. A non-zero count means eviction has already occurred and
///    sequences haven't been rescheduled yet. Risk: stuck alarm if swapped count is
///    stale and GPU has stabilized. A delta guard (swapped growing vs prior window)
///    would eliminate this — deferred until per-rule state is available at eval time.
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
    let skew = skew_secs(snapshot.gpu_observed_at, snapshot.vllm_observed_at);

    if skew > MAX_OBSERVATION_SKEW_SECS {
        return Rule2Outcome::NotFired;
    }

    let kv = snapshot.vllm.kv_cache_usage_perc.filter(|v| v.is_finite());
    let kv_high = kv.is_some_and(|kv_p| kv_p >= KV_CACHE_PRESSURE_MIN_PERC);
    if !kv_high {
        return Rule2Outcome::NotFired;
    }

    let preemptions_active = eviction_signal_active(snapshot);
    let queue_backpressure = queue_backpressure(snapshot);
    if !preemptions_active && !queue_backpressure {
        return Rule2Outcome::NotFired;
    }

    let kv_p = kv.unwrap_or(0.0);
    let peak = snapshot
        .vllm
        .kv_cache_peak_perc
        .filter(|v| v.is_finite())
        .map(|peak| peak.max(kv_p));

    Rule2Outcome::Fired(KvCachePressureDetail {
        kv_cache_usage_perc: kv_p,
        kv_peak_pct: peak,
        preemptions_active,
        queue_backpressure,
    })
}

pub fn r2_recommendation(
    snapshot: &RawSnapshot,
    max_model_len: Option<u32>,
    kv_headroom_gb: Option<f64>,
    kv_max_seqs: Option<u32>,
    windows_fired: usize,
    total_evaluable: usize,
) -> Option<Recommendation> {
    let Rule2Outcome::Fired(d) = rule2_kv_cache_pressure(snapshot) else {
        return None;
    };
    let confidence = if super::rule_is_significant(windows_fired, total_evaluable) {
        kv_pressure_confidence(windows_fired, total_evaluable)
    } else {
        0.5
    };
    let (action, short_action) = if d.preemptions_active {
        (
            r2_action(true, kv_max_seqs, max_model_len),
            r2_kv_pressure_short_action().to_string(),
        )
    } else {
        (
            r2_action(false, kv_max_seqs, max_model_len),
            r2_backlog_short_action().to_string(),
        )
    };
    Some(Recommendation {
        rule_name: "kv_cache_pressure",
        impact: 5,
        confidence,
        action,
        short_action,
        expected_impact: "Reduced KV evictions and lower latency variance".to_string(),
        display_lines: format_kv_cache_pressure_fired(
            &d,
            &KvFormatCtx {
                snapshot,
                max_model_len,
                kv_headroom_gb,
                kv_max_seqs,
            },
            windows_fired,
            total_evaluable,
        ),
    })
}

pub(super) fn r2_kv_pressure_short_action() -> &'static str {
    "lower --max-num-seqs"
}

pub(super) fn r2_backlog_short_action() -> &'static str {
    "raise --gpu-memory-utilization"
}

fn r2_max_num_seqs_ceiling_phrase(
    kv_max_seqs: Option<u32>,
    max_model_len: Option<u32>,
) -> Option<String> {
    kv_max_seqs.map(|n| match max_model_len {
        Some(m) => format!("Lower --max-num-seqs to ≤{n} (physics ceiling for max_model_len={m})"),
        None => format!("Lower --max-num-seqs to ≤{n}"),
    })
}

pub(super) fn r2_action(
    preemptions_active: bool,
    kv_max_seqs: Option<u32>,
    max_model_len: Option<u32>,
) -> String {
    if preemptions_active {
        r2_max_num_seqs_ceiling_phrase(kv_max_seqs, max_model_len)
            .unwrap_or_else(|| "Lower --max-num-seqs to stop evictions".to_string())
    } else {
        match r2_max_num_seqs_ceiling_phrase(kv_max_seqs, max_model_len) {
            Some(base) => format!("{base} or raise --gpu-memory-utilization"),
            None => "Lower --max-num-seqs or raise --gpu-memory-utilization".to_string(),
        }
    }
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
    evictions: bool,
) -> String {
    match kv_max_seqs {
        Some(n) => {
            let base = match max_model_len {
                Some(m) => {
                    format!("Lower --max-num-seqs to ≤{n} (physics ceiling for max_model_len={m})")
                }
                None => format!("Lower --max-num-seqs to ≤{n}"),
            };
            format!("    • {base}")
        }
        None => {
            if evictions {
                "    • Lower --max-num-seqs to stop evictions".to_string()
            } else {
                "    • Lower --max-num-seqs to free KV blocks".to_string()
            }
        }
    }
}

pub(super) struct KvFormatCtx<'a> {
    pub snapshot: &'a RawSnapshot,
    pub max_model_len: Option<u32>,
    pub kv_headroom_gb: Option<f64>,
    pub kv_max_seqs: Option<u32>,
}

pub(super) fn format_kv_cache_pressure_fired(
    d: &KvCachePressureDetail,
    ctx: &KvFormatCtx<'_>,
    windows_fired: usize,
    total_evaluable: usize,
) -> Vec<String> {
    let KvFormatCtx {
        snapshot,
        max_model_len,
        kv_headroom_gb,
        kv_max_seqs,
    } = *ctx;
    let peak = snapshot
        .vllm
        .kv_cache_peak_perc
        .filter(|v| v.is_finite())
        .unwrap_or(d.kv_cache_usage_perc);
    let mut out = vec!["[!] KV Cache Pressure".to_string(), "  Cause:".to_string()];
    out.push(format!(
        "  - KV cache hit {peak:.1}% peak (threshold: {:.0}%)",
        KV_CACHE_PRESSURE_MIN_PERC
    ));
    if d.preemptions_active {
        out.push(
            "  - Active preemptions: scheduler evicting sequences to free KV blocks".to_string(),
        );
    }
    if d.queue_backpressure {
        if let Some(wait) = snapshot.vllm.num_requests_waiting.filter(|v| v.is_finite()) {
            out.push(format!(
                "  - Queue backpressure: {wait:.0} requests waiting on KV admission"
            ));
        }
    }
    out.push(String::new());
    out.push("  Fix:".to_string());
    if d.preemptions_active {
        out.push(max_num_seqs_bullet(kv_max_seqs, max_model_len, true));
        if let Some(bullet) = prefix_caching_fix_bullet(snapshot) {
            out.push(bullet);
        }
        if kv_headroom_gb.is_some_and(|h| h >= KV_HEADROOM_SAFE_MIN_GB) {
            out.push(
                "    • Once stable, raise --gpu-memory-utilization (check vRAM header) to expand KV pool"
                    .to_string(),
            );
        }
        out.push(fp8_kv_cache_fix_bullet());
        let total_count = snapshot.vllm.generation_tokens_completed.unwrap_or(0.0);
        super::push_model_len_shrink_suggestion(
            &mut out,
            max_model_len,
            snapshot.vllm.prompt_tokens_p99,
            snapshot.vllm.generation_tokens_p99,
            total_count,
            "    ",
        );
    } else {
        out.push(max_num_seqs_bullet(kv_max_seqs, max_model_len, false));
        out.push(kv_headroom_gpu_mem_bullet(kv_headroom_gb));
        if let Some(bullet) = prefix_caching_fix_bullet(snapshot) {
            out.push(bullet);
        }
        out.push(fp8_kv_cache_fix_bullet());
        let total_count = snapshot.vllm.generation_tokens_completed.unwrap_or(0.0);
        super::push_model_len_shrink_suggestion(
            &mut out,
            max_model_len,
            snapshot.vllm.prompt_tokens_p99,
            snapshot.vllm.generation_tokens_p99,
            total_count,
            "    ",
        );
    }
    let expected = if d.preemptions_active {
        "  Expected: TTFT and TPOT recover once evictions stop."
    } else {
        "  Expected: Wait queue drains, TTFT recovers once KV pool has capacity."
    };
    out.push(String::new());
    out.push(expected.to_string());
    if super::rule_is_significant(windows_fired, total_evaluable) {
        let confidence = kv_pressure_confidence(windows_fired, total_evaluable);
        out.push(format!("  {}", kv_pressure_confidence_label(confidence)));
    }
    out
}

pub(super) fn format_kv_admission_backlog_issue(
    d: &KvAdmissionBacklogDetail,
    seen_pct: u32,
    max_model_len: Option<u32>,
    kv_headroom_gb: Option<f64>,
    snapshot: &RawSnapshot,
    windows_fired: usize,
    total_evaluable: usize,
) -> Vec<String> {
    let gpu_mem_bullet = kv_headroom_gpu_mem_bullet(kv_headroom_gb);
    let mut out = vec![
        "[!] KV Cache Pressure: Admission Backlog".to_string(),
        format!("  Seen in {seen_pct}% of windows"),
        "  Cause:".to_string(),
        format!(
            "  - Scheduler holding {:.0} requests in queue ({:.0}% of active requests waiting) to protect KV memory",
            d.requests_waiting,
            d.admission_ratio * 100.0
        ),
        format!(
            "  - Free KV tokens: {:.0} available, {:.0} demanded",
            d.free_kv_tokens, d.demand_tokens
        ),
        String::new(),
        "  Fix:".to_string(),
        gpu_mem_bullet,
    ];
    out.push(fp8_kv_cache_fix_bullet());
    let total_count = snapshot.vllm.generation_tokens_completed.unwrap_or(0.0);
    super::push_model_len_shrink_suggestion(
        &mut out,
        max_model_len,
        snapshot.vllm.prompt_tokens_p99,
        snapshot.vllm.generation_tokens_p99,
        total_count,
        "    ",
    );
    out.push(String::new());
    out.push("  Expected: Wait queue drains, TTFT recovers.".to_string());
    if super::rule_is_significant(windows_fired, total_evaluable) {
        let confidence = kv_pressure_confidence(windows_fired, total_evaluable);
        out.push(format!("  {}", kv_pressure_confidence_label(confidence)));
    }
    out
}

pub(super) fn aggregate_backlog_detail(
    details: &[KvAdmissionBacklogDetail],
) -> KvAdmissionBacklogDetail {
    debug_assert!(
        !details.is_empty(),
        "aggregate_backlog_detail called with no fired windows — caller should gate on r2_backlog_significant"
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
    let mut lines = format_kv_cache_pressure_fired(d, ctx, windows_fired, total_evaluable);
    lines.insert(1, format!("  Seen in {seen_pct}% of windows"));
    lines
}

pub(super) fn aggregate_r2_detail(details: &[KvCachePressureDetail]) -> KvCachePressureDetail {
    debug_assert!(
        !details.is_empty(),
        "aggregate_r2_detail called with no fired windows — caller should gate on r2_significant"
    );
    let kv = details.iter().map(|d| d.kv_cache_usage_perc).sum::<f64>() / details.len() as f64;
    let peak = details
        .iter()
        .filter_map(|d| d.kv_peak_pct)
        .chain(details.iter().map(|d| d.kv_cache_usage_perc))
        .fold(f64::NEG_INFINITY, f64::max);
    debug_assert!(
        peak.is_finite(),
        "kv_cache_usage_perc must be finite when R2 fired"
    );
    KvCachePressureDetail {
        kv_cache_usage_perc: kv,
        kv_peak_pct: Some(peak),
        preemptions_active: details.iter().any(|d| d.preemptions_active),
        queue_backpressure: details.iter().any(|d| d.queue_backpressure),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{CacheConfigLabels, GpuRawMetrics, VllmRawMetrics};
    use std::time::SystemTime;

    fn snap(vllm: VllmRawMetrics) -> RawSnapshot {
        RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm,
            gpu: GpuRawMetrics::default(),
        }
    }

    fn kv_ctx<'a>(
        snapshot: &'a RawSnapshot,
        max_model_len: Option<u32>,
        kv_headroom_gb: Option<f64>,
        kv_max_seqs: Option<u32>,
    ) -> KvFormatCtx<'a> {
        KvFormatCtx {
            snapshot,
            max_model_len,
            kv_headroom_gb,
            kv_max_seqs,
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
        let max_num_seqs = Some((run as u32) + 100);
        VllmRawMetrics {
            kv_cache_usage_perc: Some(kv),
            num_requests_waiting: Some(wait),
            num_requests_running: Some(run),
            prompt_tokens_mean: Some(prompt_mean),
            generation_tokens_per_sec: Some(100.0),
            max_num_seqs,
            cache_config: CacheConfigLabels {
                num_gpu_blocks,
                block_size,
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
        assert!(rule2_kv_admission_backlog(&snap(backlog_vllm(
            10.0,
            5.0,
            5.0,
            100.0,
            Some(1000),
            Some(16),
        )))
        .is_none());
    }

    #[test]
    fn backlog_silent_when_required_field_missing() {
        assert!(rule2_kv_admission_backlog(&snap(backlog_vllm(
            90.0,
            10.0,
            5.0,
            20.0,
            None,
            Some(16)
        )))
        .is_none());
        assert!(rule2_kv_admission_backlog(&snap(backlog_vllm(
            90.0,
            10.0,
            5.0,
            20.0,
            Some(100),
            None
        )))
        .is_none());
        assert!(rule2_kv_admission_backlog(&snap(backlog_vllm(
            90.0,
            10.0,
            5.0,
            f64::NAN,
            Some(100),
            Some(16)
        )))
        .is_none());
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
        assert!(rule2_kv_admission_backlog(&snap(backlog_vllm(
            90.0,
            2.0,
            8.0,
            20.0,
            Some(100),
            Some(16),
        )))
        .is_none());
    }

    fn detail(kv: f64, preemptions: bool) -> KvCachePressureDetail {
        KvCachePressureDetail {
            kv_cache_usage_perc: kv,
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
    fn queue_only_fire_uses_backlog_short_action() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_requests_running: Some(10.0),
            num_requests_waiting: Some(5.0),
            num_preemptions_per_sec: Some(0.0),
            generation_tokens_per_sec: Some(100.0),
            ..Default::default()
        };
        let r = r2_recommendation(&snap(v.clone()), None, None, None, 1, 1).expect("fired");
        assert!(!r.display_lines.join("\n").contains("evictions stop"));
        assert_eq!(r.short_action, "raise --gpu-memory-utilization");
        assert!(r.action.contains("gpu-memory-utilization"));
        assert!(matches!(
            rule2_kv_cache_pressure(&snap(v)),
            Rule2Outcome::Fired(d) if !d.preemptions_active && d.queue_backpressure
        ));
    }

    #[test]
    fn r2_action_backlog_includes_ceiling_and_max_model_len() {
        let action = r2_action(false, Some(14), Some(8192));
        assert!(action.contains("≤14"));
        assert!(action.contains("max_model_len=8192"));
    }

    #[test]
    fn action_string_includes_max_model_len_when_ceiling_known() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            generation_tokens_per_sec: Some(100.0),
            ..Default::default()
        };
        let r = r2_recommendation(&snap(v), Some(8192), None, Some(15), 1, 4).expect("fired");
        assert!(r.action.contains("max_model_len=8192"));
        assert!(r.action.contains("≤15"));
    }

    #[test]
    fn action_string_includes_ceiling_when_kv_max_seqs_known() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            generation_tokens_per_sec: Some(100.0),
            ..Default::default()
        };
        let r = r2_recommendation(&snap(v), None, None, Some(18), 1, 4).expect("fired");
        assert_eq!(r.action, "Lower --max-num-seqs to ≤18");
    }

    #[test]
    fn kv_pressure_short_action_matches_spec() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            num_preemptions_per_sec: Some(0.05),
            generation_tokens_per_sec: Some(100.0),
            ..Default::default()
        };
        let r = r2_recommendation(&snap(v), None, None, None, 1, 4).expect("fired");
        assert_eq!(r.short_action, "lower --max-num-seqs");
        assert_eq!(r.action, "Lower --max-num-seqs to stop evictions");
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
            kv_cache_usage_perc: 90.0,
            kv_peak_pct: Some(90.0),
            preemptions_active: false,
            queue_backpressure: true,
        };
        let text =
            format_kv_cache_pressure_fired(&d, &kv_ctx(&snap(v), Some(4096), None, None), 3, 4)
                .join("\n");
        assert!(text.contains("to ~6450"));
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
        assert!(text.contains("to ~6450"));
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
            None,
            Some(30.0),
            &snap(VllmRawMetrics::default()),
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
            None,
            Some(1.0),
            &snap(VllmRawMetrics::default()),
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
            None,
            None,
            &snap(VllmRawMetrics::default()),
            3,
            4,
        )
        .join("\n");
        assert!(text.contains("check vRAM header for avail mem"));
    }

    #[test]
    fn backlog_omits_confidence_until_significant() {
        let d = sample_backlog_detail();
        let single = format_kv_admission_backlog_issue(
            &d,
            27,
            None,
            Some(30.0),
            &snap(VllmRawMetrics::default()),
            1,
            1,
        )
        .join("\n");
        assert!(!single.contains("Confidence:"));
        let stable = format_kv_admission_backlog_issue(
            &d,
            27,
            None,
            Some(30.0),
            &snap(VllmRawMetrics::default()),
            3,
            4,
        )
        .join("\n");
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
            Some(8192),
            Some(30.0),
            &snap(v),
            3,
            4,
        )
        .join("\n");
        assert!(lines.contains("to ~6450"));
        assert!(lines.contains("Truncation risk"));
    }

    #[test]
    fn queue_backpressure_only_expected_does_not_mention_evictions() {
        let d = KvCachePressureDetail {
            kv_cache_usage_perc: 90.0,
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
            kv_cache_usage_perc: 90.0,
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
            kv_cache_usage_perc: 90.0,
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
            "Once stable, raise --gpu-memory-utilization (check vRAM header) to expand KV pool"
        ));
    }

    #[test]
    fn queue_backpressure_shows_raise_gpu_mem_bullet_when_headroom_unknown() {
        let d = KvCachePressureDetail {
            kv_cache_usage_perc: 90.0,
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
    fn fp8_kv_cache_bullet_reflects_nvcc_availability() {
        let bullet = fp8_kv_cache_fix_bullet();
        assert!(bullet.contains("Switch --kv-cache-dtype fp8"));
        if std::path::Path::new(NVCC_PATH).exists() {
            assert!(!bullet.contains("requires nvcc"));
        } else {
            assert!(bullet.contains("(requires nvcc)"));
        }
    }
}
