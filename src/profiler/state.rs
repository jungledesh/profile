use std::collections::VecDeque;

use crate::engine::{Recommendation, Report};

use super::DiagnoseResult;

const MAX_HISTORY: usize = 100;
const OSCILLATION_WINDOW: usize = 3;

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
        let mut history = VecDeque::with_capacity(MAX_HISTORY);
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
        if self.history.len() >= MAX_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(IterationRecord {
            result,
            report,
            recommendation_shown: rec_shown,
        });
    }

    pub fn last(&self) -> &IterationRecord {
        self.history
            .back()
            .expect("history always has at least one entry")
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
}
