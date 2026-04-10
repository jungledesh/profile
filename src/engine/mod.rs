//! Pure rule evaluation on [`crate::collectors::RawSnapshot`] (no network, no NVML).

use std::time::SystemTime;

use crate::collectors::RawSnapshot;

/// Correlation gate: GPU vs vLLM observation times must be close.
const MAX_OBSERVATION_SKEW_SECS: f64 = 1.0;
const GPU_UTIL_BATCH_COLLAPSE_LT: f64 = 50.0;
/// Inclusive lower bound on mean `num_requests_running` over the scrape window.
const ACTIVITY_FLOOR: f64 = 1.0;
/// Treat waiting queue as ~empty when strictly below this.
const WAITING_LT: f64 = 1.0;

#[derive(Debug, Clone, PartialEq)]
pub struct Issue {
    pub confidence: f64,
    pub evidence: Vec<String>,
}

pub fn evaluate_issues(snapshot: &RawSnapshot) -> Vec<Issue> {
    eval_batch_collapse(snapshot).into_iter().collect()
}

fn skew_secs(a: SystemTime, b: SystemTime) -> f64 {
    match a.duration_since(b) {
        Ok(d) => d.as_secs_f64(),
        Err(e) => -e.duration().as_secs_f64(),
    }
    .abs()
}

fn eval_batch_collapse(snapshot: &RawSnapshot) -> Option<Issue> {
    if skew_secs(snapshot.gpu_observed_at, snapshot.vllm_observed_at) > MAX_OBSERVATION_SKEW_SECS {
        return None;
    }

    let avg_running = snapshot.vllm.num_requests_running?;
    let max_seqs = snapshot.vllm.max_num_seqs?;
    let gpu_util = snapshot.gpu.gpu_util_pct?;
    let waiting = snapshot.vllm.num_requests_waiting?;

    if max_seqs == 0 {
        return None;
    }
    if !(avg_running.is_finite() && gpu_util.is_finite() && waiting.is_finite()) {
        return None;
    }

    let max_f = f64::from(max_seqs);
    let half_max = 0.5 * max_f;

    if !(avg_running > ACTIVITY_FLOOR
        && avg_running < half_max
        && gpu_util < GPU_UTIL_BATCH_COLLAPSE_LT
        && waiting < WAITING_LT)
    {
        return None;
    }

    let raw_confidence = 0.95 - avg_running / max_f;
    let confidence = raw_confidence.clamp(0.6, 0.95);

    let queue = snapshot
        .vllm
        .queue_delay_ms
        .filter(|q| q.is_finite())
        .map(|q| format!("{:.0}ms", q))
        .unwrap_or_else(|| "—".to_string());

    let evidence = vec![
        format!(
            "avg batch {:.1} (window) | max seqs {} | GPU util {:.0}%",
            avg_running, max_seqs, gpu_util
        ),
        format!("waiting {:.1} | queue delay {}", waiting, queue),
    ];

    Some(Issue {
        confidence,
        evidence,
    })
}

const FIX_BOX_INNER_W: usize = 71;

/// ISSUES + FIX block for the diagnose table (same width as the static stub).
pub fn format_issue(issue: &Issue) -> Vec<String> {
    let mut lines = vec![
        "ISSUES".to_string(),
        format!("01 BATCH COLLAPSE   confidence {:.2}", issue.confidence),
    ];
    lines.extend(issue.evidence.clone());
    lines.push(String::new());
    lines.push("DIAGNOSIS".to_string());
    lines.push(
        "Decode-heavy load is under-batching vs capacity; GPU stays relatively idle.".to_string(),
    );
    lines.push(String::new());
    lines.push("CAUSE".to_string());
    lines.push("Continuous batching off or batch window too small for decode overlap.".to_string());
    lines.push(String::new());
    lines.push("FIX".to_string());
    let border = format!("+{}+", "-".repeat(FIX_BOX_INNER_W));
    lines.push(border.clone());
    lines.push(padded_box_line(
        "Enable continuous batching + ~15ms batch window",
        FIX_BOX_INNER_W,
    ));
    lines.push(padded_box_line(
        "Typical: +40-60% tokens/s (A100/H100-class, workload-dependent)",
        FIX_BOX_INNER_W,
    ));
    lines.push(border);
    lines
}

fn padded_box_line(s: &str, w: usize) -> String {
    let mut t: String = s.chars().take(w).collect();
    let n = t.chars().count();
    if n < w {
        t.push_str(&" ".repeat(w - n));
    }
    format!("|{}|", t)
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

    fn pass_vllm() -> VllmRawMetrics {
        VllmRawMetrics {
            num_requests_running: Some(2.0),
            num_requests_waiting: Some(0.0),
            max_num_seqs: Some(256),
            queue_delay_ms: Some(0.0),
            ..Default::default()
        }
    }

    fn pass_gpu() -> GpuRawMetrics {
        GpuRawMetrics {
            gpu_util_pct: Some(30.0),
            ..Default::default()
        }
    }

    #[test]
    fn batch_collapse_fires_when_gates_pass() {
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, pass_vllm(), pass_gpu());
        let issues = evaluate_issues(&s);
        assert_eq!(issues.len(), 1);
        assert!((issues[0].confidence - (0.95 - 2.0 / 256.0)).abs() < 1e-6);
    }

    #[test]
    fn skew_over_one_second_suppresses() {
        let t0 = SystemTime::UNIX_EPOCH;
        let t1 = t0 + Duration::from_secs(2);
        let s = snap(t0, t1, pass_vllm(), pass_gpu());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn waiting_none_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = pass_vllm();
        v.num_requests_waiting = None;
        let s = snap(t, t, v, pass_gpu());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn activity_at_floor_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = pass_vllm();
        v.num_requests_running = Some(1.0);
        let s = snap(t, t, v, pass_gpu());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn max_seqs_zero_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = pass_vllm();
        v.max_num_seqs = Some(0);
        let s = snap(t, t, v, pass_gpu());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn confidence_clamps_low() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = pass_vllm();
        v.num_requests_running = Some(100.0);
        let s = snap(t, t, v, pass_gpu());
        let issues = evaluate_issues(&s);
        assert_eq!(issues.len(), 1);
        assert!((issues[0].confidence - 0.6).abs() < 1e-9);
    }

    #[test]
    fn nan_running_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = pass_vllm();
        v.num_requests_running = Some(f64::NAN);
        let s = snap(t, t, v, pass_gpu());
        assert!(evaluate_issues(&s).is_empty());
    }
}
