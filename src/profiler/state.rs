use std::collections::VecDeque;

use crate::engine::{Recommendation, Report};

use super::DiagnoseResult;

const OSCILLATION_WINDOW: usize = 3;
/// Closed-loop iterations stored in `LoopState`. Each entry is one full diagnose
/// cycle (user applies a change, profile re-measures). In practice a single session
/// rarely exceeds a handful of iterations; 20 is a generous upper bound that costs
/// essentially nothing.
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
        }
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

    pub fn last(&self) -> &IterationRecord {
        let Some(rec) = self.history.back() else {
            unreachable!("history is initialized with one entry and never cleared")
        };
        rec
    }

    pub fn prev(&self) -> Option<&IterationRecord> {
        let len = self.history.len();
        if len >= 2 {
            self.history.get(len - 2)
        } else {
            None
        }
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

    pub fn current_primary_recommendation(&self) -> Option<&Recommendation> {
        self.last().report.groups.first().map(|g| &g.primary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{IssueGroup, Recommendation};
    use crate::{collectors::VllmRawMetrics, context::StaticContext};
    use std::time::{Duration, SystemTime};

    fn empty_report() -> Report {
        Report {
            baseline: None,
            groups: Vec::new(),
            r2_suppressed_by_r4: false,
        }
    }

    fn group(rule: &'static str) -> IssueGroup {
        IssueGroup {
            primary: Recommendation {
                rule_name: rule,
                impact: 1,
                confidence: 1.0,
                action: String::new(),
                short_action: String::new(),
                expected_impact: String::new(),
                display_lines: Vec::new(),
            },
            secondary: Vec::new(),
        }
    }

    fn minimal_diagnose() -> DiagnoseResult {
        DiagnoseResult {
            snapshot: crate::collectors::RawSnapshot {
                gpu_observed_at: SystemTime::UNIX_EPOCH,
                vllm_observed_at: SystemTime::UNIX_EPOCH,
                timestamp: SystemTime::UNIX_EPOCH,
                vllm: VllmRawMetrics::default(),
                gpu: crate::collectors::GpuRawMetrics::default(),
            },
            windows: Vec::new(),
            static_ctx: StaticContext::default(),
            duration: Duration::from_secs(2),
            started_at: SystemTime::UNIX_EPOCH,
            any_evaluable: true,
            metrics_input: String::new(),
        }
    }

    #[test]
    fn oscillation_aba_true() {
        let r = minimal_diagnose();
        let rep = Report {
            baseline: None,
            groups: vec![group("under_batching")],
            r2_suppressed_by_r4: false,
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
            groups: vec![group("a")],
            r2_suppressed_by_r4: false,
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
}
