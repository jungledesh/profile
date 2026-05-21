use crate::collectors::{GpuRawMetrics, RawSnapshot};

use super::{skew_secs, Recommendation, MAX_OBSERVATION_SKEW_SECS};

const KV_CACHE_PRESSURE_MIN_PERC: f64 = 85.0;
pub(super) const KV_CACHE_CRITICAL_THRESHOLD_PCT: f64 = 95.0;
const KV_PRESSURE_VRAM_CORROBORATE_MIN_PERC: f64 = 78.0;
const KV_PRESSURE_CRITICAL_CONFIDENCE: f64 = 0.95;
const KV_PRESSURE_THREAT_CONFIDENCE: f64 = 0.85;
const KV_PRESSURE_WARNING_CONFIDENCE: f64 = 0.7;
const KV_ADMISSION_BACKLOG_KV_MIN_PERC: f64 = 25.0;
const KV_ADMISSION_BACKLOG_QUEUE_RATIO_MIN: f64 = 0.3;

#[derive(Debug, Clone, PartialEq)]
pub struct KvAdmissionBacklogDetail {
    pub kv_cache_usage_perc: f64,
    pub admission_ratio: f64,
    pub requests_waiting: f64,
    pub requests_running: f64,
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
    if kv < KV_ADMISSION_BACKLOG_KV_MIN_PERC {
        return None;
    }
    let wait = snapshot
        .vllm
        .num_requests_waiting
        .filter(|v| v.is_finite())?;
    let run = snapshot
        .vllm
        .num_requests_running
        .filter(|v| v.is_finite())?;
    let total = wait + run;
    if total <= 0.0 {
        return None;
    }
    let ratio = wait / total;
    if ratio < KV_ADMISSION_BACKLOG_QUEUE_RATIO_MIN {
        return None;
    }
    Some(KvAdmissionBacklogDetail {
        kv_cache_usage_perc: kv,
        admission_ratio: ratio,
        requests_waiting: wait,
        requests_running: run,
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

pub fn r2_recommendation(snapshot: &RawSnapshot) -> Option<Recommendation> {
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
        expected_impact: "Reduced KV evictions and lower latency variance".to_string(),
        display_lines: format_kv_cache_pressure_fired(&d, snapshot, confidence),
    })
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
) -> Vec<String> {
    let kv_p = snapshot.vllm.kv_cache_usage_perc.filter(|v| v.is_finite());
    let conf_label = if (confidence - KV_PRESSURE_CRITICAL_CONFIDENCE).abs() < 1e-9 {
        "Confidence: High"
    } else {
        "Confidence: Medium-High"
    };
    let mut out = vec!["[!] KV Cache Pressure".to_string(), "Cause:".to_string()];
    if d.preemptions_active {
        out.push("  - Active evictions — tokens being preempted to free KV cache".to_string());
        if let Some(k) = kv_p {
            out.push(format!("  - KV cache {k:.1}% — evictions active"));
        }
    } else {
        out.push(format!(
            "  - KV cache {:.1}% (threshold: {:.0}%) — approaching capacity",
            d.kv_cache_usage_perc, KV_CACHE_PRESSURE_MIN_PERC
        ));
    }
    out.push(String::new());
    out.push("  Fix:".to_string());
    if d.preemptions_active {
        // Evictions are live — immediate action needed
        out.extend([
            "    • Reduce concurrency now — active evictions are degrading latency".to_string(),
            "    • Lower --max-num-seqs to shed in-flight sequences".to_string(),
            "    • Consider fp8 KV cache (--kv-cache-dtype fp8) to halve KV memory footprint"
                .to_string(),
            "    • Lower max_model_len if workload allows shorter context".to_string(),
        ]);
    } else {
        // Approaching capacity — strategic fixes
        out.extend([
            "    • Raise --gpu-memory-utilization if VRAM headroom exists (check vRAM in header)"
                .to_string(),
            "    • Reduce max_num_seqs to limit peak concurrent KV block consumption".to_string(),
            "    • Consider fp8 KV cache (--kv-cache-dtype fp8) to halve KV memory footprint"
                .to_string(),
            "    • Lower max_model_len only if safe for your workload".to_string(),
        ]);
    }
    out.extend([
        String::new(),
        "  Expected: Lower TTFT, stable TPOT once evictions stop.".to_string(),
        format!("  {conf_label}"),
    ]);
    out
}

pub(super) fn format_kv_admission_backlog_issue(
    d: &KvAdmissionBacklogDetail,
    seen_pct: u32,
) -> Vec<String> {
    vec![
        "[!] KV Cache Pressure — Admission Backlog".to_string(),
        format!("  Seen in {seen_pct}% of windows"),
        "Cause:".to_string(),
        format!(
            "  - Scheduler holding {:.0} requests in queue ({:.0}% of capacity) to protect KV memory",
            d.requests_waiting,
            d.admission_ratio * 100.0
        ),
        format!(
            "  - KV cache {:.1}% — scheduler refusing admission to prevent overflow",
            d.kv_cache_usage_perc
        ),
        String::new(),
        "  Fix:".to_string(),
        "    • Raise --gpu-memory-utilization if VRAM headroom exists".to_string(),
        "    • Reduce max_num_seqs to lower peak KV block demand".to_string(),
        "    • Consider fp8 KV cache (--kv-cache-dtype fp8) to halve KV memory footprint".to_string(),
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
    KvAdmissionBacklogDetail {
        kv_cache_usage_perc: kv,
        admission_ratio: ratio,
        requests_waiting: wait,
        requests_running: run,
    }
}

pub(super) fn format_kv_cache_window_issue(
    d: &KvCachePressureDetail,
    seen_pct: u32,
    snapshot: &RawSnapshot,
    confidence: f64,
) -> Vec<String> {
    let mut lines = format_kv_cache_pressure_fired(d, snapshot, confidence);
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
    use crate::collectors::{GpuRawMetrics, VllmRawMetrics};
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

    fn backlog_vllm(kv: f64, wait: f64, run: f64) -> VllmRawMetrics {
        VllmRawMetrics {
            kv_cache_usage_perc: Some(kv),
            num_requests_waiting: Some(wait),
            num_requests_running: Some(run),
            generation_tokens_per_sec: Some(100.0),
            ..Default::default()
        }
    }

    #[test]
    fn backlog_fires_when_kv_50_and_ratio_0_3() {
        let d = rule2_kv_admission_backlog(&snap(backlog_vllm(50.0, 3.0, 7.0))).expect("fired");
        assert!((d.kv_cache_usage_perc - 50.0).abs() < 1e-9);
        assert!((d.admission_ratio - 0.3).abs() < 1e-9);
    }

    #[test]
    fn backlog_silent_when_kv_below_25() {
        assert!(rule2_kv_admission_backlog(&snap(backlog_vllm(24.9, 5.0, 5.0))).is_none());
    }

    #[test]
    fn backlog_silent_when_ratio_below_0_3() {
        assert!(rule2_kv_admission_backlog(&snap(backlog_vllm(60.0, 2.0, 8.0))).is_none());
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
}
