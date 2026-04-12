//! Rule evaluation on [`crate::collectors::RawSnapshot`] (no network in this module).
//! Rules: under-batching (NVML + vLLM), KV cache pressure (vLLM gauge + optional NVML VRAM).

use std::time::SystemTime;

use crate::collectors::{GpuRawMetrics, RawSnapshot};

/// Correlation gate: GPU vs vLLM observation times must be close.
const MAX_OBSERVATION_SKEW_SECS: f64 = 1.0;
/// Rule 1: fire only when NVML GPU util is strictly below this (percent).
const UNDER_BATCHING_GPU_UTIL_LT: f64 = 62.0;
/// Minimum mean `num_requests_running` (window) so we do not fire on an idle server.
const UNDER_BATCHING_RUNNING_GT: f64 = 0.75;
/// Mean running must stay strictly below this fraction of `max_num_seqs` to fire (8% — primary cap).
const UNDER_BATCHING_OCCUPANCY_FRAC: f64 = 0.08;
/// Fire only when mean waiting is strictly below this (no backlog).
const UNDER_BATCHING_WAITING_LT: f64 = 2.0;

/// Rule 2: KV cache gauge at or above this (percent) indicates pressure.
const KV_CACHE_PRESSURE_MIN_PERC: f64 = 85.0;
/// Rule 2: device VRAM % at or above this corroborates KV pressure when NVML data exists.
const KV_PRESSURE_VRAM_CORROBORATE_MIN_PERC: f64 = 78.0;

#[derive(Debug, Clone, PartialEq)]
pub struct Issue {
    pub confidence: f64,
    pub evidence: Vec<String>,
}

/// Values that triggered rule 1 (under-batching).
#[derive(Debug, Clone, PartialEq)]
pub struct UnderBatchingDetail {
    pub running: f64,
    pub max_num_seqs: u32,
    pub gpu_util: f64,
}

/// Values that triggered rule 2 (KV cache pressure).
#[derive(Debug, Clone, PartialEq)]
pub struct KvCachePressureDetail {
    pub kv_cache_usage_perc: f64,
    /// NVML VRAM % when ≥ [`KV_PRESSURE_VRAM_CORROBORATE_MIN_PERC`]; strengthens confidence/copy.
    pub vram_usage_perc_corroborated: Option<f64>,
}

pub fn evaluate_issues(snapshot: &RawSnapshot) -> Vec<Issue> {
    let mut issues = Vec::new();
    if let Rule1Outcome::Fired(d) = rule1_under_batching(snapshot) {
        issues.push(issue_from_under_batching(&d));
    }
    if let Rule2Outcome::Fired(d) = rule2_kv_cache_pressure(snapshot) {
        issues.push(issue_from_kv_cache_pressure(&d));
    }
    issues
}

fn issue_from_under_batching(d: &UnderBatchingDetail) -> Issue {
    Issue {
        confidence: 0.85,
        evidence: vec![format!(
            "Under-batching: {:.1} running / max_num_seqs {} | GPU {:.1}%",
            d.running, d.max_num_seqs, d.gpu_util
        )],
    }
}

fn issue_from_kv_cache_pressure(d: &KvCachePressureDetail) -> Issue {
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

/// Rule 1 evaluation: either fired detail for diagnose, or a miss report.
#[derive(Debug, Clone, PartialEq)]
pub enum Rule1Outcome {
    Fired(UnderBatchingDetail),
    NotFired(MissReport),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissReport {
    pub running: Option<f64>,
    pub gpu_util: Option<f64>,
    pub max_num_seqs: Option<u32>,
}

/// Rule 2 evaluation: fired detail or miss report for diagnose.
#[derive(Debug, Clone, PartialEq)]
pub enum Rule2Outcome {
    Fired(KvCachePressureDetail),
    NotFired(Rule2MissReport),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rule2MissReport {
    pub skew_exceeded: bool,
    /// Finite KV % when the gauge was read; `None` if missing or non-finite.
    pub kv_cache_usage_perc: Option<f64>,
}

/// Rule 1 lines for the diagnose table (fired or not).
pub fn format_rule1_diagnose(snapshot: &RawSnapshot) -> Vec<String> {
    match rule1_under_batching(snapshot) {
        Rule1Outcome::Fired(d) => format_under_batching_fired(&d),
        Rule1Outcome::NotFired(m) => format_rule1_miss(&m),
    }
}

/// Rule 1 then Rule 2 diagnose blocks (blank line between when both present).
pub fn format_diagnose_rules(snapshot: &RawSnapshot) -> Vec<String> {
    let mut lines = format_rule1_diagnose(snapshot);
    lines.push(String::new());
    lines.extend(format_rule2_diagnose(snapshot));
    lines
}

pub fn rule1_under_batching(snapshot: &RawSnapshot) -> Rule1Outcome {
    let skew = skew_secs(snapshot.gpu_observed_at, snapshot.vllm_observed_at);
    let running = snapshot.vllm.num_requests_running;
    let max_num_seqs = snapshot.vllm.max_num_seqs;
    let gpu_util = snapshot.gpu.gpu_util_pct;
    let waiting = snapshot.vllm.num_requests_waiting;

    let miss = || MissReport {
        running,
        gpu_util,
        max_num_seqs,
    };

    if skew > MAX_OBSERVATION_SKEW_SECS {
        return Rule1Outcome::NotFired(miss());
    }

    let Some(rv) = running.filter(|v| v.is_finite()) else {
        return Rule1Outcome::NotFired(miss());
    };
    let Some(max_n) = max_num_seqs.filter(|&n| n > 0) else {
        return Rule1Outcome::NotFired(miss());
    };
    let Some(gpu) = gpu_util.filter(|v| v.is_finite()) else {
        return Rule1Outcome::NotFired(miss());
    };
    let Some(wv) = waiting.filter(|v| v.is_finite()) else {
        return Rule1Outcome::NotFired(miss());
    };

    let max_f = f64::from(max_n);
    let occupancy_cap = UNDER_BATCHING_OCCUPANCY_FRAC * max_f;

    let fires = rv > UNDER_BATCHING_RUNNING_GT
        && rv < occupancy_cap
        && gpu < UNDER_BATCHING_GPU_UTIL_LT
        && wv < UNDER_BATCHING_WAITING_LT;

    if fires {
        Rule1Outcome::Fired(UnderBatchingDetail {
            running: rv,
            max_num_seqs: max_n,
            gpu_util: gpu,
        })
    } else {
        Rule1Outcome::NotFired(miss())
    }
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

fn vram_usage_perc(gpu: &GpuRawMetrics) -> Option<f64> {
    match (gpu.vram_used_mb, gpu.vram_total_mb) {
        (Some(used), Some(total)) if total > 0 => {
            let p = (used as f64 / total as f64) * 100.0;
            p.is_finite().then_some(p)
        }
        _ => None,
    }
}

pub fn format_rule2_diagnose(snapshot: &RawSnapshot) -> Vec<String> {
    match rule2_kv_cache_pressure(snapshot) {
        Rule2Outcome::Fired(d) => format_kv_cache_pressure_fired(&d),
        Rule2Outcome::NotFired(m) => format_rule2_miss(&m),
    }
}

fn format_rule2_miss(m: &Rule2MissReport) -> Vec<String> {
    let mut lines = vec!["Rule: KV Cache Pressure — Not triggered".to_string()];
    if m.skew_exceeded {
        lines.push(
            "  - GPU/vLLM observation skew > 1.0s — correlated snapshot required".to_string(),
        );
        if let Some(p) = m.kv_cache_usage_perc {
            lines.push(format!(
                "  - KV cache usage {p:.1}% — not evaluated (observation skew)"
            ));
        } else {
            lines.push("  - KV cache metric unavailable".to_string());
        }
    } else if m.kv_cache_usage_perc.is_none() {
        lines.push("  - KV cache metric unavailable".to_string());
    } else if let Some(p) = m.kv_cache_usage_perc {
        lines.push(format!(
            "  - KV cache usage {p:.1}% — within safe operating range"
        ));
    }
    lines
}

fn format_kv_cache_pressure_fired(d: &KvCachePressureDetail) -> Vec<String> {
    let mut lines = vec![
        "ISSUE: KV Cache Pressure Detected".to_string(),
        format!(
            "Cause: High KV cache usage at {:.1}% — approaching capacity with increased eviction risk",
            d.kv_cache_usage_perc
        ),
    ];
    if let Some(vp) = d.vram_usage_perc_corroborated {
        lines.push(format!(
            "       Device VRAM at {vp:.1}% (corroborates memory pressure)."
        ));
    }
    lines.push(String::new());
    lines.push("Recommendation:".to_string());
    lines.push("  • Enable prefix caching (--enable-prefix-caching) if disabled".to_string());
    lines.push("  • Consider fp8 KV cache (kv-cache-dtype=fp8)".to_string());
    lines.push("  • Reduce max_model_len or gpu-memory-utilization if suitable".to_string());
    lines.push(String::new());
    lines.push(
        "Expected Impact: Can be material for throughput and tail latency when KV capacity is the bottleneck"
            .to_string(),
    );
    let conf = if d.vram_usage_perc_corroborated.is_some() {
        "Confidence: High"
    } else {
        "Confidence: Medium-High"
    };
    lines.push(conf.to_string());
    lines
}

fn skew_secs(a: SystemTime, b: SystemTime) -> f64 {
    match a.duration_since(b) {
        Ok(d) => d.as_secs_f64(),
        Err(e) => -e.duration().as_secs_f64(),
    }
    .abs()
}

fn format_rule1_miss(m: &MissReport) -> Vec<String> {
    let mut lines = vec!["Rule: Under-batching — Not triggered".to_string()];
    lines.extend(miss_bullet_lines(m));
    lines
}

fn miss_bullet_lines(m: &MissReport) -> Vec<String> {
    let run = m
        .running
        .filter(|x| x.is_finite())
        .map(|r| format!("{r:.1}"))
        .unwrap_or_else(|| "—".to_string());
    let maxs = m
        .max_num_seqs
        .map(|n| n.to_string())
        .unwrap_or_else(|| "—".to_string());
    let gpu = m
        .gpu_util
        .filter(|x| x.is_finite())
        .map(|g| format!("{g:.1}"))
        .unwrap_or_else(|| "—".to_string());

    vec![
        format!("  - Running {run} / {maxs} max_num_seqs (moderate occupancy)"),
        format!("  - GPU utilization {gpu}% — batching is not the primary bottleneck"),
    ]
}

fn format_under_batching_fired(d: &UnderBatchingDetail) -> Vec<String> {
    let pct = (d.running / f64::from(d.max_num_seqs)) * 100.0;
    vec![
        "ISSUE: Under-batching Detected".to_string(),
        format!(
            "Cause: Very low scheduler occupancy — {:.1} running requests vs max_num_seqs = {} ({:.1}%)",
            d.running, d.max_num_seqs, pct
        ),
        format!(
            "       GPU utilization only {:.1}% with large unused capacity",
            d.gpu_util
        ),
        String::new(),
        "Recommendation:".to_string(),
        "  • Increase client concurrency or request rate to better utilize the GPU".to_string(),
        "  • Consider raising max_num_seqs if it is currently limited".to_string(),
        "  • Verify continuous batching is properly enabled".to_string(),
        String::new(),
        "Expected Impact: Can significantly improve throughput when scheduler occupancy is the bottleneck"
            .to_string(),
        "Confidence: Medium-High".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collectors::{GpuRawMetrics, RawSnapshot, VllmRawMetrics};
    use std::time::{Duration, SystemTime};

    fn snap(
        gpu_at: SystemTime,
        vllm_at: SystemTime,
        vllm: VllmRawMetrics,
        gpu: GpuRawMetrics,
    ) -> RawSnapshot {
        RawSnapshot {
            gpu_observed_at: gpu_at,
            vllm_observed_at: vllm_at,
            timestamp: gpu_at,
            vllm,
            gpu,
        }
    }

    fn vllm_base() -> VllmRawMetrics {
        VllmRawMetrics {
            num_requests_running: Some(3.1),
            num_requests_waiting: Some(0.0),
            max_num_seqs: Some(256),
            ..Default::default()
        }
    }

    fn gpu_low() -> GpuRawMetrics {
        GpuRawMetrics {
            gpu_util_pct: Some(58.0),
            ..Default::default()
        }
    }

    #[test]
    fn under_batching_fires_when_gates_pass() {
        let t = SystemTime::UNIX_EPOCH;
        // 3.1 < 0.08 * 256 = 20.48, gpu 58 < 62, wait 0 < 2, running > 0.75
        let s = snap(t, t, vllm_base(), gpu_low());
        let issues = evaluate_issues(&s);
        assert_eq!(issues.len(), 1);
        assert!((issues[0].confidence - 0.85).abs() < 1e-9);
        match rule1_under_batching(&s) {
            Rule1Outcome::Fired(d) => {
                assert!((d.running - 3.1).abs() < 1e-9);
                assert_eq!(d.max_num_seqs, 256);
                assert!((d.gpu_util - 58.0).abs() < 1e-9);
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired"),
        }
    }

    #[test]
    fn skew_over_one_second_suppresses() {
        let t0 = SystemTime::UNIX_EPOCH;
        let t1 = t0 + Duration::from_secs(2);
        let s = snap(t0, t1, vllm_base(), gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn waiting_none_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_waiting = None;
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn waiting_at_two_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_waiting = Some(2.0);
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn running_at_floor_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(0.75);
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn running_below_activity_floor_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(0.6);
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn high_occupancy_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(40.0); // >= 8% of 256
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn occupancy_at_eight_percent_cap_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        // 8% * 256 = 20.48 — must be strictly below to fire
        v.num_requests_running = Some(21.0);
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn gpu_sixty_two_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut g = gpu_low();
        g.gpu_util_pct = Some(62.0);
        let s = snap(t, t, vllm_base(), g);
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn max_seqs_zero_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.max_num_seqs = Some(0);
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn nan_running_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(f64::NAN);
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn format_rule1_diagnose_fired_matches_template() {
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, vllm_base(), gpu_low());
        let lines = format_rule1_diagnose(&s);
        let text = lines.join("\n");
        assert!(text.contains("ISSUE: Under-batching Detected"));
        assert!(text.contains("Very low scheduler occupancy"));
        assert!(text.contains("3.1 running requests"));
        assert!(text.contains("max_num_seqs = 256"));
        assert!(text.contains("GPU utilization only 58.0% with large unused capacity"));
        assert!(text.contains("Recommendation:"));
        assert!(text.contains("continuous batching is properly enabled"));
        assert!(text.contains("scheduler occupancy is the bottleneck"));
        assert!(text.contains("Confidence: Medium-High"));
    }

    #[test]
    fn format_rule1_diagnose_miss_two_bullets() {
        let t = SystemTime::UNIX_EPOCH;
        let mut g = gpu_low();
        g.gpu_util_pct = Some(75.0);
        let s = snap(t, t, vllm_base(), g);
        let lines = format_rule1_diagnose(&s);
        let text = lines.join("\n");
        assert!(text.contains("Rule: Under-batching — Not triggered"));
        assert_eq!(lines.iter().filter(|l| l.starts_with("  - ")).count(), 2);
        assert!(text.contains("Running 3.1 / 256 max_num_seqs"));
        assert!(text.contains("GPU utilization 75.0%"));
    }

    fn vllm_high_kv() -> VllmRawMetrics {
        VllmRawMetrics {
            kv_cache_usage_perc: Some(86.0),
            ..vllm_base()
        }
    }

    #[test]
    fn kv_cache_pressure_fires_at_85_boundary() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.kv_cache_usage_perc = Some(85.0);
        let s = snap(t, t, v, gpu_low());
        match rule2_kv_cache_pressure(&s) {
            Rule2Outcome::Fired(d) => {
                assert!((d.kv_cache_usage_perc - 85.0).abs() < 1e-9);
                assert!(d.vram_usage_perc_corroborated.is_none());
            }
            Rule2Outcome::NotFired(_) => panic!("expected fired at 85%"),
        }
    }

    #[test]
    fn kv_cache_pressure_suppressed_below_85() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.kv_cache_usage_perc = Some(84.9);
        let s = snap(t, t, v, gpu_low());
        match rule2_kv_cache_pressure(&s) {
            Rule2Outcome::NotFired(m) => {
                assert!(!m.skew_exceeded);
                assert_eq!(m.kv_cache_usage_perc, Some(84.9));
            }
            Rule2Outcome::Fired(_) => panic!("expected not fired"),
        }
    }

    #[test]
    fn kv_cache_pressure_skew_suppresses_with_followup_bullet() {
        let t0 = SystemTime::UNIX_EPOCH;
        let t1 = t0 + Duration::from_secs(2);
        let s = snap(t0, t1, vllm_high_kv(), gpu_low());
        match rule2_kv_cache_pressure(&s) {
            Rule2Outcome::NotFired(m) => {
                assert!(m.skew_exceeded);
                assert_eq!(m.kv_cache_usage_perc, Some(86.0));
            }
            Rule2Outcome::Fired(_) => panic!("expected skew miss"),
        }
        let text = format_rule2_diagnose(&s).join("\n");
        assert!(text.contains("observation skew"));
        assert!(text.contains("correlated snapshot required"));
    }

    #[test]
    fn kv_cache_pressure_vram_corroborates() {
        let t = SystemTime::UNIX_EPOCH;
        let mut g = gpu_low();
        g.vram_used_mb = Some(78 * 1024);
        g.vram_total_mb = Some(100 * 1024);
        let s = snap(t, t, vllm_high_kv(), g);
        match rule2_kv_cache_pressure(&s) {
            Rule2Outcome::Fired(d) => {
                let vp = d.vram_usage_perc_corroborated.expect("corroborated");
                assert!((vp - 78.0).abs() < 0.01);
            }
            Rule2Outcome::NotFired(_) => panic!("expected fired"),
        }
        let lines = format_rule2_diagnose(&s);
        let text = lines.join("\n");
        assert!(text.contains("corroborates memory pressure"));
        assert!(text.contains("Confidence: High"));
    }

    #[test]
    fn kv_cache_pressure_low_vram_not_corroborated() {
        let t = SystemTime::UNIX_EPOCH;
        let mut g = gpu_low();
        g.vram_used_mb = Some(50 * 1024);
        g.vram_total_mb = Some(100 * 1024);
        let s = snap(t, t, vllm_high_kv(), g);
        match rule2_kv_cache_pressure(&s) {
            Rule2Outcome::Fired(d) => assert!(d.vram_usage_perc_corroborated.is_none()),
            Rule2Outcome::NotFired(_) => panic!("expected fired"),
        }
        let text = format_rule2_diagnose(&s).join("\n");
        assert!(text.contains("Confidence: Medium-High"));
        assert!(!text.contains("Device VRAM"));
    }

    #[test]
    fn kv_cache_miss_unavailable_without_gauge() {
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, vllm_base(), gpu_low());
        let text = format_rule2_diagnose(&s).join("\n");
        assert!(text.contains("Rule: KV Cache Pressure — Not triggered"));
        assert!(text.contains("KV cache metric unavailable"));
    }

    #[test]
    fn evaluate_issues_under_batching_then_kv_order() {
        let t = SystemTime::UNIX_EPOCH;
        let v = vllm_high_kv();
        let s = snap(t, t, v, gpu_low());
        let issues = evaluate_issues(&s);
        assert_eq!(issues.len(), 2);
        assert!(issues[0].evidence[0].contains("Under-batching"));
        assert!(issues[1].evidence[0].contains("KV cache pressure"));
    }

    #[test]
    fn format_diagnose_rules_inserts_blank_between_rule_blocks() {
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, vllm_high_kv(), gpu_low());
        let lines = format_diagnose_rules(&s);
        let idx_under = lines
            .iter()
            .position(|l| l.contains("Under-batching"))
            .expect("rule1");
        let idx_kv = lines
            .iter()
            .position(|l| l.contains("KV Cache"))
            .expect("rule2");
        assert!(
            idx_kv > idx_under,
            "under-batching should appear before KV rule"
        );
        let between = &lines[idx_under..idx_kv];
        assert!(
            between.iter().any(|l| l.is_empty()),
            "expected blank line between rule blocks: {between:?}"
        );
    }
}
