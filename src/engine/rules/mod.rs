use std::time::SystemTime;

use crate::collectors::window_is_evaluable;
use crate::context::{AnalysisInput, RuntimeWindow};

mod r1_under_batching;
mod r2_kv_cache_pressure;
mod r3_low_prefix_reuse;
mod r4_parallelism;

pub use r1_under_batching::{
    r1_recommendation, rule1_under_batching, MissReport, Rule1Outcome, UnderBatchingDetail,
};
pub use r2_kv_cache_pressure::{
    r2_recommendation, rule2_kv_cache_pressure, KvCachePressureDetail, Rule2MissReport,
    Rule2Outcome,
};
pub use r3_low_prefix_reuse::{
    r3_recommendation, rule3_low_prefix_reuse, LowPrefixReuseDetail, Rule3Outcome,
};
pub use r4_parallelism::r4_recommendation;

use r1_under_batching::{aggregate_r1_detail, format_under_batching_window_issue};
use r2_kv_cache_pressure::{aggregate_r2_detail, format_kv_cache_window_issue};
#[cfg(test)]
use r3_low_prefix_reuse::format_low_prefix_hit_rate_fired;
use r3_low_prefix_reuse::{
    aggregate_r3_detail, format_low_prefix_window_issue, format_rule3_verbose_miss,
};

pub(super) const MAX_OBSERVATION_SKEW_SECS: f64 = 1.0;

// TODO(r5): sampling cliff — sampling temperature is not in collected metrics; wire rule when available.

#[derive(Debug, Clone, PartialEq)]
pub struct Recommendation {
    pub rule_name: &'static str,
    /// 1–5; 5 = highest impact
    pub impact: u8,
    /// 0.0–1.0
    pub confidence: f64,
    /// Prescriptive: what to change
    pub action: String,
    pub expected_impact: String,
    /// Pre-formatted cause + recommendation lines for stdout
    pub display_lines: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct IssueGroup {
    pub primary: Recommendation,
    pub secondary: Vec<Recommendation>,
}

impl IssueGroup {
    pub fn score(&self) -> f64 {
        self.primary.impact as f64 * self.primary.confidence
    }
}

const NO_ISSUES_LINE: &str = "No issues detected in this snapshot.";
const R2_SUPPRESSED_BY_R4_VERBOSE_LINE: &str = "  ↳ KV pressure suppressed — symptom of the above";

fn rule_display_block(
    g: &IssueGroup,
    verbose_rules: bool,
    r2_suppressed_by_r4: bool,
) -> Vec<String> {
    let mut block = g.primary.display_lines.clone();
    if verbose_rules && r2_suppressed_by_r4 && g.primary.rule_name == "parallelism_mismatch" {
        block.push(R2_SUPPRESSED_BY_R4_VERBOSE_LINE.to_string());
    }
    block
}

/// User-facing lines when no window met `window_is_evaluable` (shared by stdout and rule formatters).
pub fn no_evaluable_diagnose_lines(verbose: bool, windows: &[RuntimeWindow]) -> Vec<String> {
    let mut out = vec![
        "No qualifying load was detected during this run. Profile only diagnoses behavior under active traffic.".to_string(),
        "Run diagnose again while the server is handling requests (raise concurrency or wait for steady load).".to_string(),
    ];
    if verbose {
        let total = windows.len();
        if total == 0 {
            out.push("Note: No collection windows were recorded.".to_string());
        } else {
            let skipped = windows
                .iter()
                .filter(|w| !window_is_evaluable(&w.snapshot))
                .count();
            out.push(format!(
                "Note: {skipped} of {total} collected windows had insufficient traffic (running ≤ 0.75 and tok/s ≤ 20)."
            ));
        }
    }
    out
}

pub fn format_diagnose_rules(input: AnalysisInput<'_>, verbose_rules: bool) -> Vec<String> {
    let snapshot = &input.window.snapshot;
    if !window_is_evaluable(snapshot) {
        return no_evaluable_diagnose_lines(verbose_rules, std::slice::from_ref(input.window));
    }

    let report = super::build_report(input);
    let any_issue = !report.groups.is_empty();

    let fired_names: std::collections::HashSet<&'static str> =
        report.groups.iter().map(|g| g.primary.rule_name).collect();

    let mut out = Vec::new();
    let mut append = |block: Vec<String>| {
        if !out.is_empty() && !block.is_empty() {
            out.push(String::new());
        }
        out.extend(block);
    };

    for g in &report.groups {
        append(rule_display_block(
            g,
            verbose_rules,
            report.r2_suppressed_by_r4,
        ));
    }

    if verbose_rules {
        if !fired_names.contains("under_batching") {
            append(vec!["Under-batching: not indicated".to_string()]);
        }
        if !fired_names.contains("kv_cache_pressure") && !report.r2_suppressed_by_r4 {
            append(vec!["KV cache pressure: not indicated".to_string()]);
        }
        if !fired_names.contains("low_prefix_reuse") {
            append(format_rule3_verbose_miss(snapshot));
        }
        if !fired_names.contains("parallelism_mismatch") {
            append(vec!["Parallelism mismatch: not indicated".to_string()]);
        }
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
    windows: &[RuntimeWindow],
    summary: AnalysisInput<'_>,
    verbose_rules: bool,
) -> Vec<String> {
    if windows.is_empty() {
        return no_evaluable_diagnose_lines(verbose_rules, &[]);
    }

    let total = windows.len();
    let skipped = windows
        .iter()
        .filter(|w| !window_is_evaluable(&w.snapshot))
        .count();
    let evaluable: Vec<&RuntimeWindow> = windows
        .iter()
        .filter(|w| window_is_evaluable(&w.snapshot))
        .collect();
    let n_eval = evaluable.len();

    if n_eval == 0 {
        return no_evaluable_diagnose_lines(verbose_rules, windows);
    }

    let mut r1_fired = 0usize;
    let mut r2_fired = 0usize;
    let mut r3_fired = 0usize;

    let mut r1_details = Vec::new();
    let mut r2_details = Vec::new();
    let mut r3_details = Vec::new();

    for w in &evaluable {
        match rule1_under_batching(&w.snapshot) {
            Rule1Outcome::Fired(d) => {
                r1_fired += 1;
                r1_details.push(d);
            }
            Rule1Outcome::NotFired(_) => {}
        }
        match rule2_kv_cache_pressure(&w.snapshot) {
            Rule2Outcome::Fired(d) => {
                r2_fired += 1;
                r2_details.push(d);
            }
            Rule2Outcome::NotFired(_) => {}
        }
        match rule3_low_prefix_reuse(&w.snapshot) {
            Rule3Outcome::Fired(d) => {
                r3_fired += 1;
                r3_details.push(d);
            }
            Rule3Outcome::NotFired => {}
        }
    }

    let summary_snap = &summary.window.snapshot;

    if r1_fired + r2_fired + r3_fired == 0 {
        let mut out = Vec::new();
        if verbose_rules {
            out.push("Under-batching: not indicated".to_string());
            out.push(String::new());
            out.push("KV cache pressure: not indicated".to_string());
            out.push(String::new());
            out.extend(format_rule3_verbose_miss(summary_snap));
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

    let mut out = Vec::new();

    if r1_fired > 0 {
        out.extend(format_under_batching_window_issue(
            &aggregate_r1_detail(&r1_details, summary_snap),
            pct(r1_fired, n_eval),
        ));
        out.push(String::new());
    } else if verbose_rules {
        out.push("Under-batching: not indicated".to_string());
        out.push(String::new());
    }

    if r2_fired > 0 {
        out.extend(format_kv_cache_window_issue(
            &aggregate_r2_detail(&r2_details, summary_snap),
            pct(r2_fired, n_eval),
            summary_snap,
        ));
        out.push(String::new());
    } else if verbose_rules {
        out.push("KV cache pressure: not indicated".to_string());
        out.push(String::new());
    }

    if r3_fired > 0 {
        out.extend(format_low_prefix_window_issue(
            &aggregate_r3_detail(&r3_details, summary_snap),
            pct(r3_fired, n_eval),
            summary_snap.vllm.cache_config.enable_prefix_caching,
        ));
        out.push(String::new());
    } else if verbose_rules {
        out.extend(format_rule3_verbose_miss(summary_snap));
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

    let summary_report = super::build_report(summary);
    for g in summary_report
        .groups
        .iter()
        .filter(|g| g.primary.rule_name == "parallelism_mismatch")
    {
        if !out.is_empty() {
            out.push(String::new());
        }
        out.extend(rule_display_block(
            g,
            verbose_rules,
            summary_report.r2_suppressed_by_r4,
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
    use crate::collectors::{GpuRawMetrics, RawSnapshot, VllmConfig, VllmRawMetrics};
    use crate::context::{AnalysisInput, RuntimeWindow, StaticContext};
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

    fn mk_ctx() -> StaticContext {
        StaticContext::default()
    }

    fn mk_win(s: RawSnapshot) -> RuntimeWindow {
        RuntimeWindow::from_snapshot(s)
    }

    fn ai<'a>(ctx: &'a StaticContext, win: &'a RuntimeWindow) -> AnalysisInput<'a> {
        AnalysisInput { ctx, window: win }
    }

    fn input_r4_suppresses_r2() -> (StaticContext, RuntimeWindow) {
        let t = SystemTime::UNIX_EPOCH;
        let snap = RawSnapshot {
            gpu_observed_at: t,
            vllm_observed_at: t,
            timestamp: t,
            vllm: VllmRawMetrics {
                model_name: Some("meta-llama/Llama-3.1-70B-Instruct".to_string()),
                num_requests_running: Some(3.0),
                num_requests_waiting: Some(0.0),
                max_num_seqs: Some(256),
                kv_cache_usage_perc: Some(86.0),
                generation_tokens_per_sec: Some(50.0),
                request_success_per_sec: Some(10.0),
                ..Default::default()
            },
            gpu: GpuRawMetrics {
                gpu_name: Some("NVIDIA H100 80GB HBM3".to_string()),
                vram_total_mb: Some(80 * 1024),
                gpu_util_pct: Some(58.0),
                ..Default::default()
            },
        };
        let cfg = VllmConfig {
            dtype: Some("bf16".to_string()),
            max_model_len: Some(2048),
            ..Default::default()
        };
        let ctx = StaticContext::from_snapshot(&snap, cfg);
        let win = RuntimeWindow::from_snapshot(snap);
        (ctx, win)
    }

    fn vllm_base() -> VllmRawMetrics {
        VllmRawMetrics {
            num_requests_running: Some(3.1),
            num_requests_waiting: Some(0.0),
            max_num_seqs: Some(256),
            request_success_per_sec: Some(10.0),
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
        let mut v = vllm_base();
        v.tpot_ms = Some(120.0);
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        let r = r1_recommendation(&win.snapshot).expect("r1 fired");
        assert_eq!(r.rule_name, "under_batching");
        assert_eq!(r.impact, 4);
        assert!((r.confidence - 0.9).abs() < 1e-9);
        match rule1_under_batching(&win.snapshot) {
            Rule1Outcome::Fired(d) => {
                assert!((d.running - 3.1).abs() < 1e-9);
                assert_eq!(d.max_num_seqs, 256);
                assert!((d.gpu_util - 58.0).abs() < 1e-9);
                assert!((d.tpot_ms - 120.0).abs() < 1e-9);
            }
            Rule1Outcome::NotFired(_) => panic!("expected fired"),
        }
    }

    #[test]
    fn skew_over_one_second_suppresses() {
        let t0 = SystemTime::UNIX_EPOCH;
        let t1 = t0 + Duration::from_secs(2);
        let s = snap(t0, t1, vllm_base(), gpu_low());
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot).is_none());
    }

    #[test]
    fn waiting_none_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_waiting = None;
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot).is_none());
    }

    #[test]
    fn waiting_at_two_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_waiting = Some(2.0);
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot).is_none());
    }

    #[test]
    fn running_at_floor_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(0.75);
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot).is_none());
    }

    #[test]
    fn running_below_activity_floor_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(0.6);
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot).is_none());
    }

    #[test]
    fn gpu_sixty_two_percent_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut g = gpu_low();
        g.gpu_util_pct = Some(62.0);
        let s = snap(t, t, vllm_base(), g);
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot).is_none());
    }

    #[test]
    fn max_seqs_zero_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.max_num_seqs = Some(0);
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot).is_none());
    }

    #[test]
    fn nan_running_suppresses() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.num_requests_running = Some(f64::NAN);
        let s = snap(t, t, v, gpu_low());
        let win = mk_win(s);
        assert!(r1_recommendation(&win.snapshot).is_none());
    }

    #[test]
    fn format_under_batching_fired_matches_template() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.tpot_ms = Some(120.0);
        let s = snap(t, t, v, gpu_low());
        let ctx = mk_ctx();
        let win = mk_win(s);
        let lines = format_diagnose_rules(ai(&ctx, &win), false);
        let text = lines.join("\n");
        assert!(text.contains("[!] Under-batching — Memory-Bandwidth Bottleneck"));
        assert!(text.contains("GPU util   58.0%  (threshold: < 62%)"));
        assert!(text.contains("TPOT       120ms     (threshold: ≥ 100ms)"));
        assert!(text.contains("memory-bandwidth-bound execution"));
        assert!(text.contains("Raise --max-num-seqs (current: 256)"));
        assert!(text.contains("Expected: Lower TPOT, higher throughput at scale."));
        assert!(text.contains("Confidence: High"));
    }

    #[test]
    fn format_diagnose_verbose_shows_r2_suppression_note_on_r4() {
        let (ctx, win) = input_r4_suppresses_r2();
        let text = format_diagnose_rules(ai(&ctx, &win), true).join("\n");
        assert!(text.contains("Parallelism Mismatch"));
        assert!(text.contains(R2_SUPPRESSED_BY_R4_VERBOSE_LINE));
        assert!(!text.contains("KV cache pressure: not indicated"));
        assert!(!text.contains("ISSUE: KV Cache Pressure"));
    }

    #[test]
    fn format_diagnose_non_verbose_omits_r2_suppression_note() {
        let (ctx, win) = input_r4_suppresses_r2();
        let text = format_diagnose_rules(ai(&ctx, &win), false).join("\n");
        assert!(text.contains("Parallelism Mismatch"));
        assert!(!text.contains(R2_SUPPRESSED_BY_R4_VERBOSE_LINE));
        assert!(!text.contains("ISSUE: KV Cache Pressure"));
    }

    #[test]
    fn format_diagnose_verbose_shows_not_indicated_when_no_issue() {
        let t = SystemTime::UNIX_EPOCH;
        let mut g = gpu_low();
        g.gpu_util_pct = Some(75.0);
        let s = snap(t, t, vllm_base(), g);
        let ctx = mk_ctx();
        let win = mk_win(s);
        let text = format_diagnose_rules(ai(&ctx, &win), true).join("\n");
        assert!(text.contains("Under-batching: not indicated"));
        assert!(text.contains("KV cache pressure: not indicated"));
        assert!(text.contains("Prefix cache hit rate: not indicated"));
        assert!(text.contains("Parallelism mismatch: not indicated"));
        assert!(text.contains("No issues detected in this snapshot."));
    }

    fn vllm_high_kv() -> VllmRawMetrics {
        VllmRawMetrics {
            kv_cache_usage_perc: Some(86.0),
            ..vllm_base()
        }
    }

    #[test]
    fn r2_recommendation_critical_confidence_when_preemptions_active() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_high_kv();
        v.kv_cache_usage_perc = Some(50.0);
        v.num_preemptions_per_sec = Some(0.5);
        let s = snap(t, t, v, gpu_low());
        let r = r2_recommendation(&s).expect("fired");
        assert_eq!(r.rule_name, "kv_cache_pressure");
        assert_eq!(r.impact, 5);
        assert!((r.confidence - 0.95).abs() < 1e-9);
    }

    #[test]
    fn r2_recommendation_warning_confidence_when_kv_high_only() {
        let t = SystemTime::UNIX_EPOCH;
        let mut g = gpu_low();
        g.vram_used_mb = Some(78 * 1024);
        g.vram_total_mb = Some(100 * 1024);
        let s = snap(t, t, vllm_high_kv(), g);
        let r = r2_recommendation(&s).expect("fired");
        assert_eq!(r.rule_name, "kv_cache_pressure");
        assert_eq!(r.impact, 5);
        assert!((r.confidence - 0.7).abs() < 1e-9);
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
                assert!(!d.preemptions_active);
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
        let ctx = mk_ctx();
        let win = mk_win(s);
        let text = format_diagnose_rules(ai(&ctx, &win), true).join("\n");
        assert!(text.contains("Under-batching: not indicated"));
        assert!(text.contains("KV cache pressure: not indicated"));
        assert!(text.contains("Prefix cache hit rate: not indicated"));
        assert!(text.contains("Parallelism mismatch: not indicated"));
        assert!(text.contains("No issues detected in this snapshot."));
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
        let ctx2 = mk_ctx();
        let win_kv_only = mk_win(s_kv_only);
        let text = format_diagnose_rules(ai(&ctx2, &win_kv_only), false).join("\n");
        assert!(text.contains("Cause:"));
        assert!(text.contains("  - KV cache 86.0% (threshold: 85%)"));
        assert!(text.contains("  - High concurrency (~3 running requests)"));
        assert!(text.contains("Expected: 20–45% better throughput"));
        assert!(
            text.contains("  • Reduce active sequence count (lower concurrency or request rate)")
        );
        assert!(text.contains("  • Consider fp8 KV cache (kv-cache-dtype=fp8)"));
        assert!(text.contains("Confidence: Medium-High"));
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
        let ctx = mk_ctx();
        let win = mk_win(s);
        let text = format_diagnose_rules(ai(&ctx, &win), false).join("\n");
        assert!(text.contains("Confidence: Medium-High"));
        assert!(text.contains("  - KV cache 86.0% (threshold: 85%)"));
    }

    #[test]
    fn kv_cache_miss_unavailable_without_gauge_verbose() {
        let t = SystemTime::UNIX_EPOCH;
        let s = snap(t, t, vllm_base(), gpu_busy());
        let ctx = mk_ctx();
        let win = mk_win(s);
        let text = format_diagnose_rules(ai(&ctx, &win), true).join("\n");
        assert!(text.contains("KV cache pressure: not indicated"));
        assert!(text.contains("Prefix cache hit rate: not indicated"));
        assert!(text.contains("Parallelism mismatch: not indicated"));
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
        let win = mk_win(s);
        match rule3_low_prefix_reuse(&win.snapshot) {
            Rule3Outcome::Fired(d) => {
                assert!((d.hit_rate - 0.34).abs() < 1e-9);
                assert!((d.prompt_tokens_mean - 25.0).abs() < 1e-9);
            }
            Rule3Outcome::NotFired => panic!("expected fired"),
        }
        let r = r3_recommendation(&win.snapshot).expect("r3 fired");
        assert_eq!(r.rule_name, "low_prefix_reuse");
        assert_eq!(r.impact, 2);
        assert!((r.confidence - 0.9).abs() < 1e-9);
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
        let lines = format_low_prefix_hit_rate_fired(&d, Some(true));
        let text = lines.join("\n");
        assert!(text.contains("ISSUE: Low Prefix Cache"));
        assert!(text.contains("Cause:"));
        assert!(text.contains("  - Prefix hit rate 24.0% (threshold: 35%)"));
        assert!(text.contains("restructure prompts to share common prefixes"));
        assert!(text.contains("Recommendation:"));
        assert!(
            text.contains("  • Workload shows no prefix reuse — cache is currently ineffective")
        );
        assert!(text.contains("  • Otherwise: no action needed"));
        assert!(text.contains("Expected: Reduced prefill time"));
        assert!(text.contains("Confidence: High"));
    }

    #[test]
    fn format_diagnose_rule3_verbose_working_effectively_when_rate_healthy() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_base();
        v.prefix_cache_hit_rate = Some(0.50);
        let s = snap(t, t, v, gpu_busy());
        let ctx = mk_ctx();
        let win = mk_win(s);
        let text = format_diagnose_rules(ai(&ctx, &win), true).join("\n");
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
        let ctx = mk_ctx();
        let win = mk_win(s);
        let text = format_diagnose_rules(ai(&ctx, &win), true).join("\n");
        assert!(text.contains("Prefix cache hit rate: not indicated"));
        assert!(!text.contains("working effectively"));
    }

    #[test]
    fn format_diagnose_rules_no_fires_default_is_only_no_issues_line() {
        let t = SystemTime::UNIX_EPOCH;
        let mut g = gpu_low();
        g.gpu_util_pct = Some(75.0);
        let s = snap(t, t, vllm_base(), g);
        let ctx = mk_ctx();
        let win = mk_win(s);
        let lines = format_diagnose_rules(ai(&ctx, &win), false);
        assert_eq!(
            lines,
            vec!["No issues detected in this snapshot.".to_string()]
        );
    }

    #[test]
    fn format_diagnose_rules_inserts_blank_between_rule_blocks() {
        let t = SystemTime::UNIX_EPOCH;
        let mut v = vllm_high_kv();
        v.tpot_ms = Some(120.0);
        let s = snap(t, t, v, gpu_low());
        let ctx = mk_ctx();
        let win = mk_win(s);
        let lines = format_diagnose_rules(ai(&ctx, &win), false);
        let idx_under = lines
            .iter()
            .position(|l| l.contains("[!] Under-batching"))
            .expect("rule1");
        let idx_kv = lines
            .iter()
            .position(|l| l.contains("ISSUE: KV Cache Pressure"))
            .expect("rule2");
        assert!(
            idx_under < idx_kv,
            "under-batching should rank before KV cache pressure by score"
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
        let ctx = mk_ctx();
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
                v.tpot_ms = Some(120.0);
                g.gpu_util_pct = Some(50.0);
            } else {
                v.num_requests_running = Some(20.0);
                g.gpu_util_pct = Some(74.0);
            }
            windows.push(mk_win(snap(t, t, v, g)));
        }
        let summary = ai(&ctx, windows.last().expect("summary source"));
        let lines = format_diagnose_rules_for_windows(&windows, summary, false);
        let text = lines.join("\n");
        assert!(text.contains("Under-batching — Memory-Bandwidth Bottleneck"));
        assert!(text.contains("Seen in 40% of windows"));
        assert!(text.contains("TPOT       120ms"));
        assert!(text.contains("Raise --max-num-seqs"));
        assert!(text.contains("No issues for KV Cache Pressure and Low Prefix Cache"));
    }

    #[test]
    fn format_diagnose_rules_for_windows_no_fires_is_single_no_issues_line() {
        let t = SystemTime::UNIX_EPOCH;
        let ctx = mk_ctx();
        let mut v = vllm_base();
        v.num_requests_running = Some(20.0);
        v.num_requests_waiting = Some(3.0);
        v.kv_cache_usage_perc = Some(71.2);
        v.prefix_cache_hit_rate = Some(0.524);
        v.prompt_tokens_mean = Some(128.0);
        v.generation_tokens_per_sec = Some(100.0);
        let mut g = gpu_busy();
        g.gpu_util_pct = Some(74.0);
        let windows = vec![mk_win(snap(t, t, v, g))];
        let summary = ai(&ctx, windows.last().unwrap());
        let lines = format_diagnose_rules_for_windows(&windows, summary, false);
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
        let ctx = mk_ctx();
        let win = mk_win(s);
        let lines = format_diagnose_rules(ai(&ctx, &win), false);
        assert_eq!(
            lines,
            no_evaluable_diagnose_lines(false, std::slice::from_ref(&win))
        );
        let vlines = format_diagnose_rules(ai(&ctx, &win), true);
        assert!(vlines
            .iter()
            .any(|l| l.contains("1 of 1 collected windows")));
    }

    #[test]
    fn format_diagnose_rules_for_windows_all_non_evaluable() {
        let t = SystemTime::UNIX_EPOCH;
        let ctx = mk_ctx();
        let mut v = vllm_base();
        v.num_requests_running = Some(0.2);
        v.generation_tokens_per_sec = Some(5.0);
        let w1 = mk_win(snap(t, t, v.clone(), gpu_busy()));
        let w2 = mk_win(snap(t, t, v, gpu_busy()));
        let windows = vec![w1, w2];
        let summary = ai(&ctx, &windows[0]);
        let lines = format_diagnose_rules_for_windows(&windows, summary, false);
        assert_eq!(lines, no_evaluable_diagnose_lines(false, &windows));
        let summary2 = ai(&ctx, &windows[0]);
        let vlines = format_diagnose_rules_for_windows(&windows, summary2, true);
        assert!(vlines
            .iter()
            .any(|l| l.contains("2 of 2 collected windows")));
    }
}
