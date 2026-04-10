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
    match rule1_batch_collapse(snapshot) {
        Rule1Outcome::Fired(issue) => vec![issue],
        Rule1Outcome::NotFired(_) => vec![],
    }
}

/// Rule 1 evaluation: either an [`Issue`] or a report for diagnose when it does not fire.
#[derive(Debug, Clone, PartialEq)]
pub enum Rule1Outcome {
    Fired(Issue),
    NotFired(MissReport),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissReport {
    pub skew_secs: f64,
    pub running: Option<f64>,
    pub gpu_util: Option<f64>,
    pub waiting: Option<f64>,
    pub max_num_seqs: Option<u32>,
    pub reasons: Vec<String>,
}

/// Full ISSUES/DIAGNOSIS/CAUSE/FIX when rule 1 fires; otherwise the three rule metrics + why it did not fire.
pub fn format_rule1_diagnose(snapshot: &RawSnapshot) -> Vec<String> {
    match rule1_batch_collapse(snapshot) {
        Rule1Outcome::Fired(issue) => format_issue(&issue),
        Rule1Outcome::NotFired(m) => format_rule1_miss(&m),
    }
}

pub fn rule1_batch_collapse(snapshot: &RawSnapshot) -> Rule1Outcome {
    let skew = skew_secs(snapshot.gpu_observed_at, snapshot.vllm_observed_at);
    let mut reasons = Vec::new();

    if skew > MAX_OBSERVATION_SKEW_SECS {
        reasons.push(format!(
            "observation skew {:.2}s > {:.0}s (GPU vs vLLM collection end times)",
            skew, MAX_OBSERVATION_SKEW_SECS
        ));
    }

    let running = snapshot.vllm.num_requests_running;
    let max_num_seqs = snapshot.vllm.max_num_seqs;
    let gpu_util = snapshot.gpu.gpu_util_pct;
    let waiting = snapshot.vllm.num_requests_waiting;

    let running_ok = match running {
        None => {
            reasons.push("mean running (window) unavailable".into());
            None
        }
        Some(v) if !v.is_finite() => {
            reasons.push("mean running is not finite".into());
            None
        }
        Some(v) => Some(v),
    };

    let max_ok = match max_num_seqs {
        None => {
            reasons.push("max_num_seqs unavailable".into());
            None
        }
        Some(0) => {
            reasons.push("max_num_seqs is 0".into());
            None
        }
        Some(n) => Some(n),
    };

    let gpu_ok = match gpu_util {
        None => {
            reasons.push("GPU util unavailable".into());
            None
        }
        Some(v) if !v.is_finite() => {
            reasons.push("GPU util is not finite".into());
            None
        }
        Some(v) => Some(v),
    };

    let waiting_ok = match waiting {
        None => {
            reasons.push("waiting metric unavailable (required for this rule)".into());
            None
        }
        Some(v) if !v.is_finite() => {
            reasons.push("waiting is not finite".into());
            None
        }
        Some(v) => Some(v),
    };

    if let (Some(rf), Some(max_seqs_v), Some(gf), Some(wf)) =
        (running_ok, max_ok, gpu_ok, waiting_ok)
    {
        let max_f = f64::from(max_seqs_v);
        let half_max = 0.5 * max_f;
        if rf <= ACTIVITY_FLOOR {
            reasons.push(format!(
                "mean running {:.1} is not > {:.0} (activity floor)",
                rf, ACTIVITY_FLOOR
            ));
        }
        if rf >= half_max {
            reasons.push(format!(
                "mean running {:.1} is not < max_num_seqs/2 ({:.1})",
                rf, half_max
            ));
        }
        if gf >= GPU_UTIL_BATCH_COLLAPSE_LT {
            reasons.push(format!(
                "GPU util {:.0}% is not < {:.0}%",
                gf, GPU_UTIL_BATCH_COLLAPSE_LT
            ));
        }
        if wf >= WAITING_LT {
            reasons.push(format!(
                "waiting {:.1} is not < {:.0} (queue not ~empty)",
                wf, WAITING_LT
            ));
        }
    }

    let miss_report = || MissReport {
        skew_secs: skew,
        running,
        gpu_util,
        waiting,
        max_num_seqs,
        reasons: reasons.clone(),
    };

    if !reasons.is_empty() {
        return Rule1Outcome::NotFired(miss_report());
    }

    let (Some(rf), Some(max_seqs_v), Some(gf), Some(wf)) = (running_ok, max_ok, gpu_ok, waiting_ok)
    else {
        return Rule1Outcome::NotFired(MissReport {
            skew_secs: skew,
            running,
            gpu_util,
            waiting,
            max_num_seqs,
            reasons: vec!["internal: inputs incomplete with no prior reasons".into()],
        });
    };
    let max_f = f64::from(max_seqs_v);

    let raw_confidence = 0.95 - rf / max_f;
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
            rf, max_seqs_v, gf
        ),
        format!("waiting {:.1} | queue delay {}", wf, queue),
    ];

    Rule1Outcome::Fired(Issue {
        confidence,
        evidence,
    })
}

fn skew_secs(a: SystemTime, b: SystemTime) -> f64 {
    match a.duration_since(b) {
        Ok(d) => d.as_secs_f64(),
        Err(e) => -e.duration().as_secs_f64(),
    }
    .abs()
}

fn fmt_opt_running(x: Option<f64>) -> String {
    x.filter(|v| v.is_finite())
        .map(|v| format!("{:.1}", v))
        .unwrap_or_else(|| "—".to_string())
}

fn fmt_opt_gpu_pct(x: Option<f64>) -> String {
    x.filter(|v| v.is_finite())
        .map(|v| format!("{:.0}%", v))
        .unwrap_or_else(|| "—".to_string())
}

fn fmt_opt_waiting(x: Option<f64>) -> String {
    x.filter(|v| v.is_finite())
        .map(|v| format!("{:.1}", v))
        .unwrap_or_else(|| "—".to_string())
}

fn fmt_opt_max_seqs(x: Option<u32>) -> String {
    x.map(|n| n.to_string()).unwrap_or_else(|| "—".to_string())
}

fn format_rule1_miss(m: &MissReport) -> Vec<String> {
    let mut lines = vec![
        "Rule 1 (batch collapse): not triggered".to_string(),
        format!(
            "  mean running (window): {}  |  max_num_seqs: {}",
            fmt_opt_running(m.running),
            fmt_opt_max_seqs(m.max_num_seqs)
        ),
        format!("  GPU util: {}", fmt_opt_gpu_pct(m.gpu_util)),
        format!("  waiting: {}", fmt_opt_waiting(m.waiting)),
    ];
    if !m.reasons.is_empty() {
        lines.push(format!("  Hence: {}", m.reasons.join("; ")));
    } else {
        lines.push("  Hence: rule conditions not satisfied.".to_string());
    }
    lines
}

const FIX_BOX_INNER_W: usize = 71;

/// ISSUES + FIX block for the diagnose table.
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

    #[test]
    fn format_rule1_diagnose_fired_includes_issues_block() {
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, pass_vllm(), pass_gpu());
        let lines = format_rule1_diagnose(&s);
        assert!(lines.iter().any(|l| l == "ISSUES"));
        assert!(lines.iter().any(|l| l.contains("BATCH COLLAPSE")));
        assert!(lines.iter().any(|l| l == "DIAGNOSIS"));
    }

    #[test]
    fn format_rule1_diagnose_miss_shows_three_metrics_and_hence() {
        let t = SystemTime::UNIX_EPOCH;
        let mut g = pass_gpu();
        g.gpu_util_pct = Some(70.0);
        let s = snap(t, t, pass_vllm(), g);
        let lines = format_rule1_diagnose(&s);
        let text = lines.join("\n");
        assert!(text.contains("not triggered"));
        assert!(text.contains("mean running (window):"));
        assert!(text.contains("GPU util:"));
        assert!(text.contains("waiting:"));
        assert!(text.contains("Hence:"));
        assert!(text.contains("GPU util 70%"));
    }
}
