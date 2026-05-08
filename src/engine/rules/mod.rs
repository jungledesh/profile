use std::time::SystemTime;

use crate::collectors::{window_is_evaluable, RawSnapshot};

mod r1_under_batching;
mod r2_kv_cache_pressure;
mod r3_low_prefix_reuse;

pub use r1_under_batching::{rule1_under_batching, MissReport, Rule1Outcome, UnderBatchingDetail};
pub use r2_kv_cache_pressure::{
    rule2_kv_cache_pressure, KvCachePressureDetail, Rule2MissReport, Rule2Outcome,
};
pub use r3_low_prefix_reuse::{rule3_low_prefix_reuse, LowPrefixReuseDetail, Rule3Outcome};

use r1_under_batching::{
    aggregate_r1_detail, format_under_batching_fired, format_under_batching_window_issue,
    issue_from_under_batching,
};
use r2_kv_cache_pressure::{
    aggregate_r2_detail, format_kv_cache_pressure_fired, format_kv_cache_window_issue,
    issue_from_kv_cache_pressure,
};
use r3_low_prefix_reuse::{
    aggregate_r3_detail, format_low_prefix_hit_rate_fired, format_low_prefix_window_issue,
    format_rule3_verbose_miss, issue_from_low_prefix_reuse,
};

pub(super) const MAX_OBSERVATION_SKEW_SECS: f64 = 1.0;

const NO_ISSUES_LINE: &str = "No issues detected in this snapshot.";
const NOTE_NO_EVALUABLE: &str = "Note: No evaluable traffic detected during the window.";

#[derive(Debug, Clone, PartialEq)]
pub struct Issue {
    pub confidence: f64,
    pub evidence: Vec<String>,
}

pub fn evaluate_issues(snapshot: &RawSnapshot) -> Vec<Issue> {
    let mut issues = Vec::new();
    if let Rule1Outcome::Fired(d) = rule1_under_batching(snapshot) {
        issues.push(issue_from_under_batching(&d));
    }
    if let Rule2Outcome::Fired(d) = rule2_kv_cache_pressure(snapshot) {
        issues.push(issue_from_kv_cache_pressure(&d));
    }
    if let Rule3Outcome::Fired(d) = rule3_low_prefix_reuse(snapshot) {
        issues.push(issue_from_low_prefix_reuse(&d));
    }
    issues
}

pub fn format_diagnose_rules(snapshot: &RawSnapshot, verbose_rules: bool) -> Vec<String> {
    if !window_is_evaluable(snapshot) {
        let mut out = vec![NO_ISSUES_LINE.to_string(), NOTE_NO_EVALUABLE.to_string()];
        if verbose_rules {
            out.push("Note: 1 window had insufficient traffic for analysis.".to_string());
        }
        return out;
    }

    let r1 = rule1_under_batching(snapshot);
    let r2 = rule2_kv_cache_pressure(snapshot);
    let r3 = rule3_low_prefix_reuse(snapshot);
    let any_issue = matches!(r1, Rule1Outcome::Fired(_))
        || matches!(r2, Rule2Outcome::Fired(_))
        || matches!(r3, Rule3Outcome::Fired(_));

    let mut out = Vec::new();
    let mut append = |block: Vec<String>| {
        if !out.is_empty() && !block.is_empty() {
            out.push(String::new());
        }
        out.extend(block);
    };

    match &r1 {
        Rule1Outcome::Fired(d) => append(format_under_batching_fired(d)),
        Rule1Outcome::NotFired(_) if verbose_rules => {
            append(vec!["Under-batching: not indicated".to_string()])
        }
        Rule1Outcome::NotFired(_) => {}
    }

    match &r2 {
        Rule2Outcome::Fired(d) => append(format_kv_cache_pressure_fired(d, snapshot)),
        Rule2Outcome::NotFired(_) if verbose_rules => {
            append(vec!["KV cache pressure: not indicated".to_string()])
        }
        Rule2Outcome::NotFired(_) => {}
    }

    match &r3 {
        Rule3Outcome::Fired(d) => append(format_low_prefix_hit_rate_fired(d)),
        Rule3Outcome::NotFired if verbose_rules => append(format_rule3_verbose_miss(snapshot)),
        Rule3Outcome::NotFired => {}
    }

    if !any_issue {
        if !out.is_empty() {
            out.push(String::new());
        }
        out.push(NO_ISSUES_LINE.to_string());
    }

    out
}

pub fn format_diagnose_rules_for_windows(
    windows: &[RawSnapshot],
    summary: &RawSnapshot,
    verbose_rules: bool,
) -> Vec<String> {
    if windows.is_empty() {
        return vec![NO_ISSUES_LINE.to_string()];
    }

    let total = windows.len();
    let skipped = windows.iter().filter(|w| !window_is_evaluable(w)).count();
    let evaluable: Vec<&RawSnapshot> = windows.iter().filter(|w| window_is_evaluable(w)).collect();
    let n_eval = evaluable.len();

    if n_eval == 0 {
        let mut out = vec![NO_ISSUES_LINE.to_string(), NOTE_NO_EVALUABLE.to_string()];
        if verbose_rules {
            out.push(format!(
                "Note: {skipped} of {total} windows had insufficient traffic for analysis."
            ));
        }
        return out;
    }

    let mut r1_fired = 0usize;
    let mut r2_fired = 0usize;
    let mut r3_fired = 0usize;

    let mut r1_details = Vec::new();
    let mut r2_details = Vec::new();
    let mut r3_details = Vec::new();

    for w in &evaluable {
        match rule1_under_batching(w) {
            Rule1Outcome::Fired(d) => {
                r1_fired += 1;
                r1_details.push(d);
            }
            Rule1Outcome::NotFired(_) => {}
        }
        match rule2_kv_cache_pressure(w) {
            Rule2Outcome::Fired(d) => {
                r2_fired += 1;
                r2_details.push(d);
            }
            Rule2Outcome::NotFired(_) => {}
        }
        match rule3_low_prefix_reuse(w) {
            Rule3Outcome::Fired(d) => {
                r3_fired += 1;
                r3_details.push(d);
            }
            Rule3Outcome::NotFired => {}
        }
    }

    if r1_fired + r2_fired + r3_fired == 0 {
        let mut out = Vec::new();
        if verbose_rules {
            out.push("Under-batching: not indicated".to_string());
            out.push(String::new());
            out.push("KV cache pressure: not indicated".to_string());
            out.push(String::new());
            out.extend(format_rule3_verbose_miss(summary));
            out.push(String::new());
        }
        out.push(NO_ISSUES_LINE.to_string());
        if verbose_rules && skipped > 0 {
            out.push(format!(
                "Note: {skipped} of {total} windows had insufficient traffic for analysis."
            ));
        }
        trim_trailing_blank_lines(&mut out);
        return out;
    }

    let mut out = vec!["ISSUES:".to_string(), String::new()];

    if r1_fired > 0 {
        out.extend(format_under_batching_window_issue(
            &aggregate_r1_detail(&r1_details, summary),
            pct(r1_fired, n_eval),
        ));
        out.push(String::new());
    } else if verbose_rules {
        out.push("Under-batching: not indicated".to_string());
        out.push(String::new());
    }

    if r2_fired > 0 {
        out.extend(format_kv_cache_window_issue(
            &aggregate_r2_detail(&r2_details, summary),
            pct(r2_fired, n_eval),
            summary,
        ));
        out.push(String::new());
    } else if verbose_rules {
        out.push("KV cache pressure: not indicated".to_string());
        out.push(String::new());
    }

    if r3_fired > 0 {
        out.extend(format_low_prefix_window_issue(
            &aggregate_r3_detail(&r3_details, summary),
            pct(r3_fired, n_eval),
        ));
        out.push(String::new());
    } else if verbose_rules {
        out.extend(format_rule3_verbose_miss(summary));
        out.push(String::new());
    }

    let mut not_fired = Vec::new();
    if r1_fired == 0 {
        not_fired.push("Under-batching");
    }
    if r2_fired == 0 {
        not_fired.push("KV Cache Pressure");
    }
    if r3_fired == 0 {
        not_fired.push("Low Prefix Cache");
    }
    if !not_fired.is_empty() {
        out.push(format!("No issues for {}", join_rule_names(&not_fired)));
    }
    if verbose_rules && skipped > 0 {
        out.push(String::new());
        out.push(format!(
            "Note: {skipped} of {total} windows had insufficient traffic for analysis."
        ));
    }
    trim_trailing_blank_lines(&mut out);
    out
}

pub(super) fn skew_secs(a: SystemTime, b: SystemTime) -> f64 {
    match a.duration_since(b) {
        Ok(d) => d.as_secs_f64(),
        Err(e) => -e.duration().as_secs_f64(),
    }
    .abs()
}

fn pct(fired: usize, total: usize) -> u32 {
    if total == 0 {
        return 0;
    }
    ((fired as f64 / total as f64) * 100.0).round() as u32
}

fn trim_trailing_blank_lines(lines: &mut Vec<String>) {
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
}

fn join_rule_names(items: &[&str]) -> String {
    match items {
        [] => String::new(),
        [one] => one.to_string(),
        [a, b] => format!("{a} and {b}"),
        _ => {
            let head = &items[..items.len() - 1];
            let last = items[items.len() - 1];
            format!("{}, and {}", head.join(", "), last)
        }
    }
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

    fn gpu_busy() -> GpuRawMetrics {
        GpuRawMetrics {
            gpu_util_pct: Some(75.0),
            ..Default::default()
        }
    }

    #[test]
    fn under_batching_fires_when_gates_pass() {
        let t = SystemTime::UNIX_EPOCH;
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
        v.num_requests_running = Some(40.0);
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn occupancy_at_six_percent_cap_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(16.0);
        let s = snap(t, t, v, gpu_low());
        assert!(evaluate_issues(&s).is_empty());
    }

    #[test]
    fn gpu_sixty_two_percent_suppresses() {
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
    fn format_under_batching_fired_matches_template() {
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, vllm_base(), gpu_low());
        let lines = format_diagnose_rules(&s, false);
        let text = lines.join("\n");
        assert!(text.contains("ISSUE: Under-batching"));
        assert!(text.contains("Very low occupancy"));
        assert!(text.contains("3.1 / 256"));
        assert!(text.contains("GPU util only 58% with headroom"));
        assert!(text.contains("Recommendation:"));
        assert!(text.contains("  • Increase client concurrency or request rate"));
        assert!(text.contains("  • Raise max_num_seqs if VRAM allows"));
        assert!(text.contains("Expected: Better throughput"));
        assert!(text.contains("Confidence: Medium-High"));
    }

    #[test]
    fn format_diagnose_verbose_shows_not_indicated_when_no_issue() {
        let t = SystemTime::UNIX_EPOCH;
        let mut g = gpu_low();
        g.gpu_util_pct = Some(75.0);
        let s = snap(t, t, vllm_base(), g);
        let text = format_diagnose_rules(&s, true).join("\n");
        assert!(text.contains("Under-batching: not indicated"));
        assert!(text.contains("KV cache pressure: not indicated"));
        assert!(text.contains("Prefix cache hit rate: not indicated"));
        assert!(text.contains("No issues detected in this snapshot."));
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
    fn kv_cache_pressure_skew_suppresses() {
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
        let text = format_diagnose_rules(&s, true).join("\n");
        assert!(text.contains("Under-batching: not indicated"));
        assert!(text.contains("KV cache pressure: not indicated"));
        assert!(text.contains("Prefix cache hit rate: not indicated"));
        assert!(text.ends_with("No issues detected in this snapshot."));
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
        let mut gb = gpu_busy();
        gb.vram_used_mb = Some(78 * 1024);
        gb.vram_total_mb = Some(100 * 1024);
        let s_kv_only = snap(t, t, vllm_high_kv(), gb);
        let text = format_diagnose_rules(&s_kv_only, false).join("\n");
        assert!(text.contains("Cause:"));
        assert!(text.contains("  - KV usage 86.0% — near capacity"));
        assert!(text.contains("  - High concurrency (~3 running requests)"));
        assert!(text.contains("Expected: 20–45% better throughput"));
        assert!(
            text.contains("  • Reduce active sequence count (lower concurrency or request rate)")
        );
        assert!(text.contains("  • Consider fp8 KV cache (kv-cache-dtype=fp8)"));
        assert!(text.contains("Confidence: High"));
    }

    #[test]
    fn kv_cache_pressure_low_vram_not_corroborated() {
        let t = SystemTime::UNIX_EPOCH;
        let mut gb = gpu_busy();
        gb.vram_used_mb = Some(50 * 1024);
        gb.vram_total_mb = Some(100 * 1024);
        let s = snap(t, t, vllm_high_kv(), gb);
        match rule2_kv_cache_pressure(&s) {
            Rule2Outcome::Fired(d) => assert!(d.vram_usage_perc_corroborated.is_none()),
            Rule2Outcome::NotFired(_) => panic!("expected fired"),
        }
        let text = format_diagnose_rules(&s, false).join("\n");
        assert!(text.contains("Confidence: Medium-High"));
        assert!(text.contains("  - KV usage 86.0% — near capacity"));
    }

    #[test]
    fn kv_cache_miss_unavailable_without_gauge_verbose() {
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, vllm_base(), gpu_busy());
        let text = format_diagnose_rules(&s, true).join("\n");
        assert!(text.contains("KV cache pressure: not indicated"));
        assert!(text.contains("Prefix cache hit rate: not indicated"));
        assert!(text.contains("No issues detected in this snapshot."));
    }

    #[test]
    fn rule3_fires_when_hit_below_35_and_gates_pass() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.prefix_cache_hit_rate = Some(0.34);
        v.prompt_tokens_mean = Some(25.0);
        v.num_requests_running = Some(1.0);
        let s = snap(t, t, v, gpu_busy());
        match rule3_low_prefix_reuse(&s) {
            Rule3Outcome::Fired(d) => {
                assert!((d.hit_rate - 0.34).abs() < 1e-9);
                assert!((d.prompt_tokens_mean - 25.0).abs() < 1e-9);
            }
            Rule3Outcome::NotFired => panic!("expected fired"),
        }
        let issues = evaluate_issues(&s);
        assert_eq!(issues.len(), 1);
        assert!(issues[0].evidence[0].contains("Low prefix cache hit rate"));
    }

    #[test]
    fn rule3_suppressed_at_or_above_35() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.prefix_cache_hit_rate = Some(0.35);
        v.prompt_tokens_mean = Some(25.0);
        v.num_requests_running = Some(1.0);
        let s = snap(t, t, v, gpu_busy());
        assert!(matches!(rule3_low_prefix_reuse(&s), Rule3Outcome::NotFired));
    }

    #[test]
    fn format_low_prefix_hit_rate_fired_matches_template() {
        let d = LowPrefixReuseDetail {
            hit_rate: 0.24,
            prompt_tokens_mean: 128.0,
        };
        let lines = format_low_prefix_hit_rate_fired(&d);
        let text = lines.join("\n");
        assert!(text.contains("ISSUE: Low Prefix Cache"));
        assert!(text.contains("Cause:"));
        assert!(text.contains("  - Prefix hit rate 24.0%"));
        assert!(text.contains("  - Prompts have no shared leading context"));
        assert!(text.contains("Recommendation:"));
        assert!(
            text.contains("  • Workload shows no prefix reuse — cache is currently ineffective")
        );
        assert!(text.contains("  • Otherwise: no action needed"));
        assert!(text.contains("Expected: Reduced prefill time"));
        assert!(text.contains("Confidence: Medium-High"));
    }

    #[test]
    fn format_diagnose_rule3_verbose_working_effectively_when_rate_healthy() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.prefix_cache_hit_rate = Some(0.50);
        let s = snap(t, t, v, gpu_busy());
        let text = format_diagnose_rules(&s, true).join("\n");
        assert!(text.contains("Rule: Low Prefix Cache — Not triggered"));
        assert!(text.contains("  - Prefix cache hit rate 50.0% — working effectively"));
    }

    #[test]
    fn format_diagnose_rule3_verbose_not_indicated_when_rate_low_but_prompt_below_floor() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.prefix_cache_hit_rate = Some(0.20);
        v.prompt_tokens_mean = Some(10.0);
        let s = snap(t, t, v, gpu_busy());
        let text = format_diagnose_rules(&s, true).join("\n");
        assert!(text.contains("Prefix cache hit rate: not indicated"));
        assert!(!text.contains("working effectively"));
    }

    #[test]
    fn format_diagnose_rules_no_fires_default_is_only_no_issues_line() {
        let t = SystemTime::UNIX_EPOCH;
        let mut g = gpu_low();
        g.gpu_util_pct = Some(75.0);
        let s = snap(t, t, vllm_base(), g);
        let lines = format_diagnose_rules(&s, false);
        assert_eq!(
            lines,
            vec!["No issues detected in this snapshot.".to_string()]
        );
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
        let lines = format_diagnose_rules(&s, false);
        let idx_under = lines
            .iter()
            .position(|l| l.contains("ISSUE: Under-batching"))
            .expect("rule1");
        let idx_kv = lines
            .iter()
            .position(|l| l.contains("ISSUE: KV Cache Pressure"))
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
        assert!(
            !lines.iter().any(|l| l.contains("No issues detected")),
            "should not append no-issues line when at least one rule fired"
        );
    }

    #[test]
    fn format_diagnose_rules_for_windows_matches_requested_style_when_some_rules_fire() {
        let t = SystemTime::UNIX_EPOCH;
        let mut windows = Vec::new();
        for i in 0..10 {
            let mut v = vllm_base();
            v.max_num_seqs = Some(256);
            v.num_requests_waiting = Some(1.0);
            v.kv_cache_usage_perc = Some(71.2);
            v.prefix_cache_hit_rate = Some(0.524);
            v.prompt_tokens_mean = Some(128.0);
            v.generation_tokens_per_sec = Some(1580.0);
            let mut g = gpu_busy();
            g.power_watts = Some(312.0);
            g.vram_used_mb = Some(62 * 1024);
            g.vram_total_mb = Some(80 * 1024);
            if i < 4 {
                v.num_requests_running = Some(3.2);
                g.gpu_util_pct = Some(50.0);
            } else {
                v.num_requests_running = Some(20.0);
                g.gpu_util_pct = Some(74.0);
            }
            windows.push(snap(t, t, v, g));
        }
        let summary = windows.last().expect("summary source").clone();
        let lines = format_diagnose_rules_for_windows(&windows, &summary, false);
        let text = lines.join("\n");
        assert!(text.contains("ISSUES:"));
        assert!(text.contains("Under-batching"));
        assert!(text.contains("Seen in 40% of windows"));
        assert!(text.contains(
            "Cause: Very low occupancy — avg 3.2 / 256 (1.2%), GPU util only 50.0% with headroom"
        ));
        assert!(text.contains("For better efficiency:"));
        assert!(text.contains("No issues for KV Cache Pressure and Low Prefix Cache"));
    }

    #[test]
    fn format_diagnose_rules_for_windows_no_fires_is_single_no_issues_line() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(20.0);
        v.num_requests_waiting = Some(3.0);
        v.kv_cache_usage_perc = Some(71.2);
        v.prefix_cache_hit_rate = Some(0.524);
        v.prompt_tokens_mean = Some(128.0);
        v.generation_tokens_per_sec = Some(100.0);
        let mut g = gpu_busy();
        g.gpu_util_pct = Some(74.0);
        let windows = vec![snap(t, t, v, g)];
        let summary = windows[0].clone();
        let lines = format_diagnose_rules_for_windows(&windows, &summary, false);
        assert_eq!(
            lines,
            vec!["No issues detected in this snapshot.".to_string()]
        );
    }

    #[test]
    fn format_diagnose_rules_non_evaluable_snapshot_shows_note() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(0.0);
        v.generation_tokens_per_sec = Some(0.0);
        let s = snap(t, t, v, gpu_busy());
        let lines = format_diagnose_rules(&s, false);
        assert_eq!(
            lines,
            vec![
                "No issues detected in this snapshot.".to_string(),
                "Note: No evaluable traffic detected during the window.".to_string(),
            ]
        );
        let vlines = format_diagnose_rules(&s, true);
        assert!(vlines
            .iter()
            .any(|l| l.contains("1 window had insufficient traffic")));
    }

    #[test]
    fn format_diagnose_rules_for_windows_all_non_evaluable() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(0.2);
        v.generation_tokens_per_sec = Some(5.0);
        let w1 = snap(t, t, v.clone(), gpu_busy());
        let w2 = snap(t, t, v, gpu_busy());
        let windows = vec![w1, w2];
        let lines = format_diagnose_rules_for_windows(&windows, &windows[0], false);
        assert_eq!(
            lines,
            vec![
                "No issues detected in this snapshot.".to_string(),
                "Note: No evaluable traffic detected during the window.".to_string(),
            ]
        );
        let vlines = format_diagnose_rules_for_windows(&windows, &windows[0], true);
        assert!(vlines.iter().any(|l| l.contains("2 of 2 windows")));
    }
}
