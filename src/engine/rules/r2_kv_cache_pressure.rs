use crate::collectors::{GpuRawMetrics, RawSnapshot};

use super::{skew_secs, Issue, MAX_OBSERVATION_SKEW_SECS};

const KV_CACHE_PRESSURE_MIN_PERC: f64 = 85.0;
const KV_PRESSURE_VRAM_CORROBORATE_MIN_PERC: f64 = 78.0;

#[derive(Debug, Clone, PartialEq)]
pub struct KvCachePressureDetail {
    pub kv_cache_usage_perc: f64,
    pub vram_usage_perc_corroborated: Option<f64>,
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

    let Some(kv_p) = kv else {
        return Rule2Outcome::NotFired(miss(false, None));
    };

    if kv_p < KV_CACHE_PRESSURE_MIN_PERC {
        return Rule2Outcome::NotFired(miss(false, Some(kv_p)));
    }

    let vram = vram_usage_perc(&snapshot.gpu);
    let corroborated = vram.filter(|&p| p >= KV_PRESSURE_VRAM_CORROBORATE_MIN_PERC);

    Rule2Outcome::Fired(KvCachePressureDetail {
        kv_cache_usage_perc: kv_p,
        vram_usage_perc_corroborated: corroborated,
    })
}

pub(super) fn issue_from_kv_cache_pressure(d: &KvCachePressureDetail) -> Issue {
    let confidence = if d.vram_usage_perc_corroborated.is_some() {
        0.9
    } else {
        0.82
    };
    let vram_note = d
        .vram_usage_perc_corroborated
        .map(|p| format!(" | device VRAM {:.1}%", p))
        .unwrap_or_default();
    Issue {
        confidence,
        evidence: vec![format!(
            "KV cache pressure: {:.1}% KV usage{}",
            d.kv_cache_usage_perc, vram_note
        )],
    }
}

pub(super) fn format_kv_cache_pressure_fired(
    d: &KvCachePressureDetail,
    snapshot: &RawSnapshot,
) -> Vec<String> {
    let kv = d.kv_cache_usage_perc;
    let conf = if d.vram_usage_perc_corroborated.is_some() {
        "Confidence: High"
    } else {
        "Confidence: Medium-High"
    };
    let mut out = vec![
        "ISSUE: KV Cache Pressure".to_string(),
        "Cause:".to_string(),
        format!("  - KV usage {kv:.1}% — near capacity"),
    ];
    if let Some(r) = snapshot.vllm.num_requests_running.filter(|x| x.is_finite()) {
        out.push(format!("  - High concurrency (~{:.0} running requests)", r));
    }
    if let Some(p) = snapshot.vllm.prompt_tokens_mean.filter(|x| x.is_finite()) {
        out.push(format!("  - Long sequences (~{:.0} token prompts)", p));
    }
    out.extend([
        String::new(),
        "Recommendation:".to_string(),
        "  • Reduce active sequence count (lower concurrency or request rate)".to_string(),
        "  • Shorten prompts/outputs where possible".to_string(),
        "  • Increase KV capacity if needed:".to_string(),
        "      - Raise --gpu-memory-utilization (if VRAM headroom exists)".to_string(),
        "  • Consider fp8 KV cache (kv-cache-dtype=fp8)".to_string(),
        "  • Lower max_model_len only if safe for your workload".to_string(),
        String::new(),
        "Expected: 20–45% better throughput".to_string(),
        conf.to_string(),
    ]);
    out
}

pub(super) fn format_kv_cache_window_issue(
    d: &KvCachePressureDetail,
    seen_pct: u32,
    summary: &RawSnapshot,
) -> Vec<String> {
    let mut out = vec![
        "KV Cache Pressure".to_string(),
        format!("Seen in {seen_pct}% of windows"),
        "Cause:".to_string(),
        format!("  - KV usage {:.1}% — near capacity", d.kv_cache_usage_perc),
    ];
    if let Some(r) = summary.vllm.num_requests_running.filter(|x| x.is_finite()) {
        out.push(format!("  - High concurrency (~{:.0} running requests)", r));
    }
    if let Some(p) = summary.vllm.prompt_tokens_mean.filter(|x| x.is_finite()) {
        out.push(format!("  - Long sequences (~{:.0} token prompts)", p));
    }
    out.extend([
        String::new(),
        "Recommendation:".to_string(),
        "  • Reduce active sequence count (lower concurrency or request rate)".to_string(),
        "  • Shorten prompts/outputs where possible".to_string(),
        "  • Increase KV capacity if needed:".to_string(),
        "      - Raise --gpu-memory-utilization (if VRAM headroom exists)".to_string(),
        "  • Consider fp8 KV cache (kv-cache-dtype=fp8)".to_string(),
        "  • Lower max_model_len only if safe for your workload".to_string(),
    ]);
    out
}

pub(super) fn aggregate_r2_detail(
    details: &[KvCachePressureDetail],
    summary: &RawSnapshot,
) -> KvCachePressureDetail {
    if details.is_empty() {
        return KvCachePressureDetail {
            kv_cache_usage_perc: summary.vllm.kv_cache_usage_perc.unwrap_or(0.0),
            vram_usage_perc_corroborated: None,
        };
    }
    let kv = details.iter().map(|d| d.kv_cache_usage_perc).sum::<f64>() / details.len() as f64;
    let corroborated = details.iter().find_map(|d| d.vram_usage_perc_corroborated);
    KvCachePressureDetail {
        kv_cache_usage_perc: kv,
        vram_usage_perc_corroborated: corroborated,
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
