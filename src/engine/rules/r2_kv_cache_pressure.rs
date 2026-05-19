use crate::collectors::{GpuRawMetrics, RawSnapshot};

use super::{skew_secs, Recommendation, MAX_OBSERVATION_SKEW_SECS};

const KV_CACHE_PRESSURE_MIN_PERC: f64 = 85.0;
const KV_PRESSURE_VRAM_CORROBORATE_MIN_PERC: f64 = 78.0;
const KV_PRESSURE_CRITICAL_CONFIDENCE: f64 = 0.95;
const KV_PRESSURE_WARNING_CONFIDENCE: f64 = 0.7;

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
