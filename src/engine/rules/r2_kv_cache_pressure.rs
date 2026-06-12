use crate::collectors::{GpuRawMetrics, RawSnapshot};

use super::{model_len_suffix, skew_secs, Recommendation, MAX_OBSERVATION_SKEW_SECS};

/// 88% matches observed vLLM production eviction onset; 85% was too conservative.
const KV_CACHE_PRESSURE_MIN_PERC: f64 = 88.0;
pub(super) const KV_CACHE_CRITICAL_THRESHOLD_PCT: f64 = 95.0;
const KV_PRESSURE_VRAM_CORROBORATE_MIN_PERC: f64 = 78.0;
const KV_PRESSURE_CRITICAL_CONFIDENCE: f64 = 0.95;
const KV_PRESSURE_THREAT_CONFIDENCE: f64 = 0.85;
const KV_PRESSURE_WARNING_CONFIDENCE: f64 = 0.7;
const KV_ADMISSION_BACKLOG_QUEUE_RATIO_MIN: f64 = 0.30;
const KV_HEADROOM_SAFE_MIN_GB: f64 = 2.0;
const NVCC_PATH: &str = "/usr/local/cuda/bin/nvcc";
const FP8_KV_CACHE_FIX: &str =
    "    • Switch to fp8 KV cache (--kv-cache-dtype fp8) to halve KV memory footprint";

static NVCC_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn fp8_kv_cache_fix_bullet() -> String {
    let nvcc_present = *NVCC_AVAILABLE.get_or_init(|| std::path::Path::new(NVCC_PATH).exists());
    if nvcc_present {
        FP8_KV_CACHE_FIX.to_string()
    } else {
        format!("{FP8_KV_CACHE_FIX} (requires nvcc — not available on this host)")
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
    pub vram_usage_perc_corroborated: Option<f64>,
    pub preemptions_active: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rule2MissReport {
    pub skew_exceeded: bool,
    pub kv_cache_usage_perc: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Rule2Outcome {
    Fired(KvCachePressureDetail),
    NotFired(Rule2MissReport),
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

pub fn rule2_kv_cache_pressure(snapshot: &RawSnapshot) -> Rule2Outcome {
    let skew = skew_secs(snapshot.gpu_observed_at, snapshot.vllm_observed_at);
    let kv = snapshot.vllm.kv_cache_usage_perc.filter(|v| v.is_finite());

    let miss = |skew_exceeded: bool, kv_cache_usage_perc: Option<f64>| Rule2MissReport {
        skew_exceeded,
        kv_cache_usage_perc,
    };

    if skew > MAX_OBSERVATION_SKEW_SECS {
        return Rule2Outcome::NotFired(miss(true, kv));
    }

    let preemptions_active = snapshot
        .vllm
        .num_preemptions_per_sec
        .is_some_and(|p| p > 0.0)
        || snapshot.vllm.num_requests_swapped.is_some_and(|s| s > 0.0);

    let kv_high = kv.is_some_and(|kv_p| kv_p >= KV_CACHE_PRESSURE_MIN_PERC);

    if !kv_high && !preemptions_active {
        return Rule2Outcome::NotFired(miss(false, kv));
    }

    let kv_p = kv.unwrap_or(0.0);
    let vram = vram_usage_perc(&snapshot.gpu);
    let corroborated = vram.filter(|&p| p >= KV_PRESSURE_VRAM_CORROBORATE_MIN_PERC);

    Rule2Outcome::Fired(KvCachePressureDetail {
        kv_cache_usage_perc: kv_p,
        vram_usage_perc_corroborated: corroborated,
        preemptions_active,
    })
}

pub fn r2_recommendation(
    snapshot: &RawSnapshot,
    max_model_len: Option<u32>,
) -> Option<Recommendation> {
    let Rule2Outcome::Fired(d) = rule2_kv_cache_pressure(snapshot) else {
        return None;
    };
    let confidence = if d.preemptions_active {
        KV_PRESSURE_CRITICAL_CONFIDENCE
    } else {
        KV_PRESSURE_WARNING_CONFIDENCE
    };
    Some(Recommendation {
        rule_name: "kv_cache_pressure",
        impact: 5,
        confidence,
        action: "Reduce max_num_seqs or add tensor parallelism".to_string(),
        short_action: r2_kv_pressure_short_action(),
        expected_impact: "Reduced KV evictions and lower latency variance".to_string(),
        display_lines: format_kv_cache_pressure_fired(&d, snapshot, confidence, max_model_len),
    })
}

pub(super) fn r2_kv_pressure_short_action() -> String {
    "lower --max-num-seqs".to_string()
}

pub(super) fn r2_backlog_short_action() -> String {
    "raise --gpu-memory-utilization".to_string()
}

pub(super) fn kv_pressure_confidence(d: &KvCachePressureDetail) -> f64 {
    if d.preemptions_active {
        KV_PRESSURE_CRITICAL_CONFIDENCE
    } else if d.kv_cache_usage_perc >= KV_CACHE_CRITICAL_THRESHOLD_PCT {
        KV_PRESSURE_THREAT_CONFIDENCE
    } else {
        KV_PRESSURE_WARNING_CONFIDENCE
    }
}

pub(super) fn format_kv_cache_pressure_fired(
    d: &KvCachePressureDetail,
    snapshot: &RawSnapshot,
    confidence: f64,
    max_model_len: Option<u32>,
) -> Vec<String> {
    let kv_p = snapshot.vllm.kv_cache_usage_perc.filter(|v| v.is_finite());
    let conf_label = if (confidence - KV_PRESSURE_CRITICAL_CONFIDENCE).abs() < 1e-9 {
        "Confidence: High"
    } else {
        "Confidence: Medium-High"
    };
    let mut out = vec!["[!] KV Cache Pressure".to_string(), "  Cause:".to_string()];
    if d.preemptions_active {
        out.push(
            "  - Active preemptions: scheduler evicting sequences to free KV blocks".to_string(),
        );
        if let Some(k) = kv_p {
            if let Some(peak) = snapshot.vllm.kv_cache_peak_perc.filter(|v| v.is_finite()) {
                out.push(format!(
                    "  - KV cache {k:.1}% avg ({peak:.1}% peak), peak triggered evictions"
                ));
            } else {
                out.push(format!("  - KV cache {k:.1}%, evictions active"));
            }
        }
    } else {
        out.push(format!(
            "  - KV cache {:.1}% (threshold: {:.0}%), approaching capacity",
            d.kv_cache_usage_perc, KV_CACHE_PRESSURE_MIN_PERC
        ));
    }
    out.push(String::new());
    out.push("  Fix:".to_string());
    let model_len_bullet = format!(
        "    • Lower --max-model-len{} if your workload allows shorter context",
        model_len_suffix(max_model_len)
    );
    if d.preemptions_active {
        out.extend([
            "    • Lower --max-num-seqs now to stop evictions, pick a value below current running count"
                .to_string(),
            fp8_kv_cache_fix_bullet(),
            model_len_bullet,
        ]);
    } else {
        out.extend([
            "    • Raise --gpu-memory-utilization if VRAM headroom exists (check header for available VRAM)"
                .to_string(),
            fp8_kv_cache_fix_bullet(),
            model_len_bullet,
        ]);
    }
    let expected = if d.preemptions_active {
        "  Expected: TTFT and TPOT recover once evictions stop."
    } else {
        "  Expected: Lower TTFT, stable TPOT once evictions stop."
    };
    out.extend([
        String::new(),
        expected.to_string(),
        format!("  {conf_label}"),
    ]);
    out
}

pub(super) fn format_kv_admission_backlog_issue(
    d: &KvAdmissionBacklogDetail,
    seen_pct: u32,
    max_model_len: Option<u32>,
    kv_headroom_gb: Option<f64>,
) -> Vec<String> {
    let model_len_bullet = format!(
        "    • Lower --max-model-len{} if your workload allows shorter context",
        model_len_suffix(max_model_len)
    );
    let gpu_mem_bullet = match kv_headroom_gb {
        Some(h) if h >= KV_HEADROOM_SAFE_MIN_GB => format!(
            "    • Raise --gpu-memory-utilization ({h:.0}GB VRAM available) to expand KV pool"
        ),
        Some(_) => {
            "    • GPU at VRAM capacity: cannot raise --gpu-memory-utilization. Scale out or reduce context footprint."
                .to_string()
        }
        None => "    • Raise --gpu-memory-utilization if VRAM headroom exists".to_string(),
    };
    vec![
        "[!] KV Cache Pressure — Admission Backlog".to_string(),
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
        fp8_kv_cache_fix_bullet(),
        model_len_bullet,
        String::new(),
        "  Expected: Wait queue drains, TTFT recovers.".to_string(),
        "  Confidence: Medium-High".to_string(),
    ]
}

pub(super) fn aggregate_backlog_detail(
    details: &[KvAdmissionBacklogDetail],
) -> KvAdmissionBacklogDetail {
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
    snapshot: &RawSnapshot,
    confidence: f64,
    max_model_len: Option<u32>,
) -> Vec<String> {
    let mut lines = format_kv_cache_pressure_fired(d, snapshot, confidence, max_model_len);
    lines.insert(1, format!("  Seen in {seen_pct}% of windows"));
    lines
}

pub(super) fn aggregate_r2_detail(
    details: &[KvCachePressureDetail],
    summary: &RawSnapshot,
) -> KvCachePressureDetail {
    if details.is_empty() {
        let preemptions_active = summary
            .vllm
            .num_preemptions_per_sec
            .is_some_and(|p| p > 0.0)
            || summary.vllm.num_requests_swapped.is_some_and(|s| s > 0.0);
        return KvCachePressureDetail {
            kv_cache_usage_perc: summary.vllm.kv_cache_usage_perc.unwrap_or(0.0),
            vram_usage_perc_corroborated: None,
            preemptions_active,
        };
    }
    let kv = details.iter().map(|d| d.kv_cache_usage_perc).sum::<f64>() / details.len() as f64;
    let corroborated = details.iter().find_map(|d| d.vram_usage_perc_corroborated);
    let preemptions_active = details.iter().any(|d| d.preemptions_active);
    KvCachePressureDetail {
        kv_cache_usage_perc: kv,
        vram_usage_perc_corroborated: corroborated,
        preemptions_active,
    }
}

fn vram_usage_perc(gpu: &GpuRawMetrics) -> Option<f64> {
    match (gpu.vram_used_mb, gpu.vram_total_mb) {
        (Some(used), Some(total)) if total > 0 => {
            let p = (used as f64 / total as f64) * 100.0;
            p.is_finite().then_some(p)
        }
        _ => None,
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
            vram_usage_perc_corroborated: None,
            preemptions_active: preemptions,
        }
    }

    #[test]
    fn kv_pressure_confidence_critical_when_preemptions_active() {
        assert!((kv_pressure_confidence(&detail(50.0, true)) - 0.95).abs() < 1e-9);
    }

    #[test]
    fn kv_pressure_confidence_threat_when_kv_at_95_no_preemptions() {
        assert!((kv_pressure_confidence(&detail(95.0, false)) - 0.85).abs() < 1e-9);
    }

    #[test]
    fn kv_pressure_confidence_warning_when_kv_below_95_no_preemptions() {
        assert!((kv_pressure_confidence(&detail(90.0, false)) - 0.7).abs() < 1e-9);
    }

    #[test]
    fn model_len_shown_when_some() {
        let text = format_kv_cache_pressure_fired(
            &detail(50.0, true),
            &snap(VllmRawMetrics::default()),
            KV_PRESSURE_CRITICAL_CONFIDENCE,
            Some(4096),
        )
        .join("\n");
        assert!(text.contains("--max-model-len (currently 4096)"));
    }

    #[test]
    fn model_len_generic_when_none() {
        let text = format_kv_cache_pressure_fired(
            &detail(50.0, true),
            &snap(VllmRawMetrics::default()),
            KV_PRESSURE_CRITICAL_CONFIDENCE,
            None,
        )
        .join("\n");
        assert!(text.contains("--max-model-len if"));
        assert!(!text.contains("currently"));
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
        let text =
            format_kv_admission_backlog_issue(&sample_backlog_detail(), 27, None, Some(30.0))
                .join("\n");
        assert!(text.contains("30GB VRAM available"));
    }

    #[test]
    fn backlog_warns_when_vram_full() {
        let text = format_kv_admission_backlog_issue(&sample_backlog_detail(), 27, None, Some(1.0))
            .join("\n");
        assert!(text.contains("GPU at VRAM capacity"));
    }

    #[test]
    fn backlog_generic_when_headroom_unknown() {
        let text =
            format_kv_admission_backlog_issue(&sample_backlog_detail(), 27, None, None).join("\n");
        assert!(text.contains("if VRAM headroom exists"));
    }

    #[test]
    fn fp8_kv_cache_bullet_reflects_nvcc_availability() {
        let bullet = fp8_kv_cache_fix_bullet();
        assert!(bullet.contains("Switch to fp8 KV cache (--kv-cache-dtype fp8)"));
        if std::path::Path::new(NVCC_PATH).exists() {
            assert!(!bullet.contains("not available on this host"));
        } else {
            assert!(bullet.contains("(requires nvcc — not available on this host)"));
        }
    }

    #[test]
    fn kv_pressure_short_action_matches_spec() {
        let v = VllmRawMetrics {
            kv_cache_usage_perc: Some(90.0),
            generation_tokens_per_sec: Some(100.0),
            ..Default::default()
        };
        let r = r2_recommendation(&snap(v), None).expect("fired");
        assert_eq!(r.short_action, "lower --max-num-seqs");
    }

    #[test]
    fn backlog_short_action_matches_spec() {
        assert_eq!(r2_backlog_short_action(), "raise --gpu-memory-utilization");
    }
}
