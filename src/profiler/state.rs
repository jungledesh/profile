use std::collections::VecDeque;

use crate::engine::Report;

use super::DiagnoseResult;

const OSCILLATION_WINDOW: usize = 3;
/// Per-session diagnose iterations stored in `LoopState`; 20 is a generous bound that costs nothing.
pub const MAX_LOOP_ITERATIONS: usize = 20;

pub struct IterationRecord {
    pub result: DiagnoseResult,
    pub report: Report,
    /// `rule_name` of the recommendation shown before this measurement, if any.
    pub recommendation_shown: Option<&'static str>,
}

pub struct LoopState {
    history: VecDeque<IterationRecord>,
    rec_history: VecDeque<&'static str>,
    midpoint_suggested: bool,
}

impl LoopState {
    pub fn new(result: DiagnoseResult, report: Report) -> Self {
        let mut history = VecDeque::with_capacity(MAX_LOOP_ITERATIONS);
        history.push_back(IterationRecord {
            result,
            report,
            recommendation_shown: None,
        });
        Self {
            history,
            rec_history: VecDeque::with_capacity(OSCILLATION_WINDOW + 1),
            midpoint_suggested: false,
        }
    }

    pub fn history(&self) -> &VecDeque<IterationRecord> {
        &self.history
    }

    pub fn midpoint_suggested(&self) -> bool {
        self.midpoint_suggested
    }

    pub fn set_midpoint_suggested(&mut self) {
        self.midpoint_suggested = true;
        self.rec_history.clear();
    }

    pub fn push(
        &mut self,
        result: DiagnoseResult,
        report: Report,
        rec_shown: Option<&'static str>,
    ) {
        self.history.push_back(IterationRecord {
            result,
            report,
            recommendation_shown: rec_shown,
        });
    }

    pub fn last(&self) -> Option<&IterationRecord> {
        self.history.back()
    }

    pub fn record_recommendation(&mut self, rule_name: &'static str) {
        if self.rec_history.len() > OSCILLATION_WINDOW {
            self.rec_history.pop_front();
        }
        self.rec_history.push_back(rule_name);
    }

    /// Ping-pong-ping: last 3 recommendations are `[A, B, A]` (newest first: `A`, `B`, `A`).
    pub fn is_oscillating(&self) -> bool {
        if self.rec_history.len() < OSCILLATION_WINDOW {
            return false;
        }
        let v: Vec<_> = self
            .rec_history
            .iter()
            .rev()
            .take(OSCILLATION_WINDOW)
            .collect();
        v[0] == v[2] && v[0] != v[1]
    }

    /// Returns `(rule_a, rule_b)` when oscillating A→B→A. `rule_a` is the repeated one.
    pub fn oscillating_pair(&self) -> Option<(&'static str, &'static str)> {
        if !self.is_oscillating() {
            return None;
        }
        let v: Vec<_> = self.rec_history.iter().rev().take(3).collect();
        Some((*v[0], *v[1]))
    }

    pub fn iteration_count(&self) -> usize {
        self.history.len().saturating_sub(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Recommendation;
    use crate::{collectors::VllmRawMetrics, context::StaticContext};
    use std::time::{Duration, SystemTime};

    fn empty_report() -> Report {
        Report {
            baseline: None,
            recommendations: Vec::new(),
            suppressed_rules: Vec::new(),
            suppressed_recs: Vec::new(),
            kv_max_seqs: None,
            prescribed_kv_capacity: None,
            catalog_state_mismatch: None,
            memory_budget_self_grade: None,
            n_eval: 0,
            skipped_broken: 0,
            skipped_idle: 0,
            energy_skew_skipped: 0,
            gauge_missing: Default::default(),
            limiter_evidence: None,
        }
    }

    fn recommendation(rule: &'static str) -> Recommendation {
        Recommendation {
            rule_name: rule,
            layer: 4,
            impact: 1,
            confidence: 1.0,
            display_lines: Vec::new(),
        }
    }

    fn minimal_diagnose() -> DiagnoseResult {
        DiagnoseResult {
            snapshot: crate::collectors::RawSnapshot {
                gpu_observed_at: SystemTime::UNIX_EPOCH,
                vllm_observed_at: SystemTime::UNIX_EPOCH,
                timestamp: SystemTime::UNIX_EPOCH,
                vllm: VllmRawMetrics::default(),
                gpus: vec![],
            },
            windows: Vec::new(),
            static_ctx: StaticContext::default(),
            duration: Duration::from_secs(2),
            started_at: SystemTime::UNIX_EPOCH,
            any_evaluable: true,
            all_idle: false,
            metrics_input: String::new(),
        }
    }

    #[test]
    fn oscillation_aba_true() {
        let r = minimal_diagnose();
        let rep = Report {
            baseline: None,
            recommendations: vec![recommendation("under_batching")],
            suppressed_rules: Vec::new(),
            suppressed_recs: Vec::new(),
            kv_max_seqs: None,
            prescribed_kv_capacity: None,
            catalog_state_mismatch: None,
            memory_budget_self_grade: None,
            n_eval: 1,
            skipped_broken: 0,
            skipped_idle: 0,
            energy_skew_skipped: 0,
            gauge_missing: Default::default(),
            limiter_evidence: None,
        };
        let mut s = LoopState::new(r, rep);
        s.record_recommendation("under_batching");
        s.record_recommendation("kv_cache_pressure");
        s.record_recommendation("under_batching");
        assert!(s.is_oscillating());
    }

    #[test]
    fn oscillation_aab_false() {
        let r = minimal_diagnose();
        let rep = Report {
            baseline: None,
            recommendations: vec![recommendation("a")],
            suppressed_rules: Vec::new(),
            suppressed_recs: Vec::new(),
            kv_max_seqs: None,
            prescribed_kv_capacity: None,
            catalog_state_mismatch: None,
            memory_budget_self_grade: None,
            n_eval: 1,
            skipped_broken: 0,
            skipped_idle: 0,
            energy_skew_skipped: 0,
            gauge_missing: Default::default(),
            limiter_evidence: None,
        };
        let mut s = LoopState::new(r, rep);
        s.record_recommendation("under_batching");
        s.record_recommendation("under_batching");
        s.record_recommendation("kv_cache_pressure");
        assert!(!s.is_oscillating());
    }

    #[test]
    fn oscillation_abc_false() {
        let r = minimal_diagnose();
        let mut s = LoopState::new(r, empty_report());
        s.record_recommendation("a");
        s.record_recommendation("b");
        s.record_recommendation("c");
        assert!(!s.is_oscillating());
    }

    #[test]
    fn oscillation_fewer_than_three_false() {
        let r = minimal_diagnose();
        let mut s = LoopState::new(r, empty_report());
        s.record_recommendation("a");
        s.record_recommendation("b");
        assert!(!s.is_oscillating());
    }

    #[test]
    fn iteration_count_zero_on_init() {
        let s = LoopState::new(minimal_diagnose(), empty_report());
        assert_eq!(s.iteration_count(), 0);
    }

    #[test]
    fn iteration_count_increments_on_push() {
        let r = minimal_diagnose();
        let mut s = LoopState::new(r, empty_report());
        for i in 1..=3 {
            s.push(minimal_diagnose(), empty_report(), None);
            assert_eq!(s.iteration_count(), i);
        }
    }

    #[test]
    fn oscillating_pair_returns_ab_on_aba() {
        let r = minimal_diagnose();
        let mut s = LoopState::new(r, empty_report());
        s.record_recommendation("under_batching");
        s.record_recommendation("kv_cache_pressure");
        s.record_recommendation("under_batching");
        assert_eq!(
            s.oscillating_pair(),
            Some(("under_batching", "kv_cache_pressure"))
        );
    }

    #[test]
    fn iteration_cap_reached_after_twenty_pushes() {
        let r = minimal_diagnose();
        let mut s = LoopState::new(r, empty_report());
        for _ in 0..MAX_LOOP_ITERATIONS {
            s.push(minimal_diagnose(), empty_report(), None);
        }
        assert_eq!(s.iteration_count(), MAX_LOOP_ITERATIONS);
        assert!(s.iteration_count() >= MAX_LOOP_ITERATIONS);
    }

    #[test]
    fn oscillating_pair_none_when_not_oscillating() {
        let r = minimal_diagnose();
        let mut s = LoopState::new(r, empty_report());
        s.record_recommendation("a");
        s.record_recommendation("b");
        assert!(s.oscillating_pair().is_none());
    }

    #[test]
    fn set_midpoint_clears_oscillation() {
        let mut s = LoopState::new(minimal_diagnose(), empty_report());
        s.record_recommendation("kv_cache_pressure");
        s.record_recommendation("concurrency_saturation");
        s.record_recommendation("kv_cache_pressure");
        assert!(s.is_oscillating());
        s.set_midpoint_suggested();
        assert!(!s.is_oscillating());
        assert!(s.midpoint_suggested());
    }
}
