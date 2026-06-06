use crate::engine::Report;

use super::DiagnoseResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Better,
    Worse,
    /// &lt; 10% relative change in throughput (when known).
    Plateau,
}

#[derive(Debug, Clone)]
pub struct Delta {
    /// Relative % change in `generation_tokens_per_sec`.
    pub throughput_delta_pct: Option<f64>,
    pub throughput_before: Option<f64>,
    pub throughput_after: Option<f64>,
    /// Absolute percentage-point change in `efficiency_pct`.
    pub efficiency_delta_pp: Option<f64>,
    pub efficiency_pct_before: Option<f64>,
    pub efficiency_pct_after: Option<f64>,
    pub cost_per_million_before: Option<f64>,
    pub cost_per_million_after: Option<f64>,
    pub joules_per_token_before: Option<f64>,
    pub joules_per_token_after: Option<f64>,
    pub cost_source_after: Option<crate::engine::CostSource>,
    pub ttft_before_ms: Option<f64>,
    pub ttft_after_ms: Option<f64>,
    pub tpot_before_ms: Option<f64>,
    pub tpot_after_ms: Option<f64>,
    pub ttft_p99_before_ms: Option<f64>,
    pub ttft_p99_after_ms: Option<f64>,
    pub tpot_p99_before_ms: Option<f64>,
    pub tpot_p99_after_ms: Option<f64>,
    pub direction: Direction,
    pub config_drifted: bool,
}

const PLATEAU_THRESHOLD_PCT: f64 = 10.0;

pub fn compute(
    prev_result: &DiagnoseResult,
    prev_report: &Report,
    curr_result: &DiagnoseResult,
    curr_report: &Report,
    config_drifted: bool,
) -> Delta {
    let throughput_before = prev_result.snapshot.vllm.generation_tokens_per_sec;
    let throughput_after = curr_result.snapshot.vllm.generation_tokens_per_sec;

    let ttft_before_ms = prev_result.snapshot.vllm.ttft_ms;
    let ttft_after_ms = curr_result.snapshot.vllm.ttft_ms;

    let tpot_before_ms = prev_result.snapshot.vllm.tpot_ms;
    let tpot_after_ms = curr_result.snapshot.vllm.tpot_ms;

    let ttft_p99_before_ms = prev_result.snapshot.vllm.ttft_p99_ms;
    let ttft_p99_after_ms = curr_result.snapshot.vllm.ttft_p99_ms;
    let tpot_p99_before_ms = prev_result.snapshot.vllm.tpot_p99_ms;
    let tpot_p99_after_ms = curr_result.snapshot.vllm.tpot_p99_ms;

    let throughput_delta_pct = match (throughput_before, throughput_after) {
        (Some(p), Some(c)) if p > 0.0 && p.is_finite() && c.is_finite() => {
            Some((c - p) / p * 100.0)
        }
        _ => None,
    };

    let efficiency_pct_before = prev_report.baseline.as_ref().and_then(|b| b.efficiency_pct);
    let efficiency_pct_after = curr_report.baseline.as_ref().and_then(|b| b.efficiency_pct);

    let efficiency_delta_pp = match (efficiency_pct_before, efficiency_pct_after) {
        (Some(p), Some(c)) if p.is_finite() && c.is_finite() => Some(c - p),
        _ => None,
    };

    let cost_per_million_before = prev_report
        .baseline
        .as_ref()
        .and_then(|b| b.cost.as_ref())
        .and_then(|c| c.cost_per_million_tokens);
    let cost_per_million_after = curr_report
        .baseline
        .as_ref()
        .and_then(|b| b.cost.as_ref())
        .and_then(|c| c.cost_per_million_tokens);
    let cost_source_after = curr_report
        .baseline
        .as_ref()
        .and_then(|b| b.cost.as_ref())
        .map(|c| c.cost_source);

    let joules_per_token_before = prev_report
        .baseline
        .as_ref()
        .and_then(|b| b.cost.as_ref())
        .and_then(|c| c.joules_per_token);
    let joules_per_token_after = curr_report
        .baseline
        .as_ref()
        .and_then(|b| b.cost.as_ref())
        .and_then(|c| c.joules_per_token);

    let direction = match throughput_delta_pct {
        Some(d) if d > PLATEAU_THRESHOLD_PCT => Direction::Better,
        Some(d) if d < -PLATEAU_THRESHOLD_PCT => Direction::Worse,
        Some(_) => Direction::Plateau,
        None => Direction::Plateau,
    };

    Delta {
        throughput_delta_pct,
        throughput_before,
        throughput_after,
        efficiency_delta_pp,
        efficiency_pct_before,
        efficiency_pct_after,
        cost_per_million_before,
        cost_per_million_after,
        joules_per_token_before,
        joules_per_token_after,
        cost_source_after,
        ttft_before_ms,
        ttft_after_ms,
        tpot_before_ms,
        tpot_after_ms,
        ttft_p99_before_ms,
        ttft_p99_after_ms,
        tpot_p99_before_ms,
        tpot_p99_after_ms,
        direction,
        config_drifted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::baseline::{CostEstimate, CostSource};
    use crate::engine::{CeilingEstimate, PhysicsBaseline, WeightDtypeSource};
    use crate::{
        collectors::{GpuRawMetrics, RawSnapshot, VllmRawMetrics},
        context::{RuntimeWindow, StaticContext},
    };
    use std::time::{Duration, SystemTime};

    fn snap(tps: Option<f64>) -> RawSnapshot {
        RawSnapshot {
            gpu_observed_at: SystemTime::UNIX_EPOCH,
            vllm_observed_at: SystemTime::UNIX_EPOCH,
            timestamp: SystemTime::UNIX_EPOCH,
            vllm: VllmRawMetrics {
                generation_tokens_per_sec: tps,
                ..Default::default()
            },
            gpu: GpuRawMetrics::default(),
        }
    }

    fn diagnose(tps: Option<f64>) -> DiagnoseResult {
        let s = snap(tps);
        DiagnoseResult {
            snapshot: s.clone(),
            windows: vec![RuntimeWindow::from_snapshot(s)],
            static_ctx: StaticContext::default(),
            duration: Duration::from_secs(2),
            started_at: SystemTime::UNIX_EPOCH,
            any_evaluable: true,
            metrics_input: String::new(),
        }
    }

    fn report_eff(
        eff: Option<f64>,
        cost_cpm: Option<f64>,
        joules_per_token: Option<f64>,
    ) -> Report {
        let cost = if cost_cpm.is_some() || joules_per_token.is_some() {
            Some(CostEstimate {
                tok_per_watt: None,
                joules_per_token,
                cost_per_million_tokens: cost_cpm,
                cost_source: CostSource::Catalog,
            })
        } else {
            None
        };
        Report {
            baseline: Some(PhysicsBaseline {
                decode: CeilingEstimate {
                    lower: 1.0,
                    expected: 1.0,
                    upper: 1.0,
                },
                prefill: None,
                efficiency_pct: eff,
                headroom_pct: eff.map(|e| 100.0 - e.min(100.0)),
                weight_dtype_source: WeightDtypeSource::Fallback,
                weight_gb: 1.0,
                kv_headroom_gb: None,
                tpot_floor_ms: 1.0,
                prefill_latency_floor_ms: None,
                ridge_batch_size: 1.0,
                cost,
            }),
            groups: Vec::new(),
            r2_suppressed_by_r4: false,
        }
    }

    #[test]
    fn direction_better_above_threshold() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0), None, None),
            &diagnose(Some(112.0)),
            &report_eff(Some(55.0), None, None),
            false,
        );
        assert_eq!(d.direction, Direction::Better);
        assert!((d.throughput_delta_pct.unwrap() - 12.0).abs() < 1e-9);
        assert_eq!(d.throughput_before, Some(100.0));
        assert_eq!(d.throughput_after, Some(112.0));
    }

    #[test]
    fn direction_worse_below_neg_threshold() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0), None, None),
            &diagnose(Some(80.0)),
            &report_eff(Some(45.0), None, None),
            false,
        );
        assert_eq!(d.direction, Direction::Worse);
        assert!((d.throughput_delta_pct.unwrap() + 20.0).abs() < 1e-9);
    }

    #[test]
    fn direction_plateau_within_band() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0), None, None),
            &diagnose(Some(101.0)),
            &report_eff(Some(50.5), None, None),
            false,
        );
        assert_eq!(d.direction, Direction::Plateau);
    }

    #[test]
    fn direction_plateau_at_nine_pct_drop() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0), None, None),
            &diagnose(Some(91.0)),
            &report_eff(Some(45.0), None, None),
            false,
        );
        assert_eq!(d.direction, Direction::Plateau);
        assert!((d.throughput_delta_pct.unwrap() + 9.0).abs() < 1e-9);
    }

    #[test]
    fn direction_worse_at_eleven_pct_drop() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0), None, None),
            &diagnose(Some(89.0)),
            &report_eff(Some(45.0), None, None),
            false,
        );
        assert_eq!(d.direction, Direction::Worse);
        assert!((d.throughput_delta_pct.unwrap() + 11.0).abs() < 1e-9);
    }

    #[test]
    fn throughput_delta_none_when_prev_zero() {
        let d = compute(
            &diagnose(Some(0.0)),
            &report_eff(None, None, None),
            &diagnose(Some(50.0)),
            &report_eff(Some(10.0), None, None),
            false,
        );
        assert!(d.throughput_delta_pct.is_none());
        assert_eq!(d.direction, Direction::Plateau);
    }

    #[test]
    fn throughput_delta_none_when_missing_gauge() {
        let d = compute(
            &diagnose(None),
            &report_eff(None, None, None),
            &diagnose(Some(50.0)),
            &report_eff(None, None, None),
            false,
        );
        assert!(d.throughput_delta_pct.is_none());
    }

    #[test]
    fn config_drifted_passthrough() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0), None, None),
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0), None, None),
            true,
        );
        assert!(d.config_drifted);
    }

    #[test]
    fn populates_cost_and_efficiency_before_after() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(40.0), Some(2.50), Some(0.31)),
            &diagnose(Some(120.0)),
            &report_eff(Some(55.0), Some(2.00), Some(0.28)),
            false,
        );
        assert_eq!(d.efficiency_pct_before, Some(40.0));
        assert_eq!(d.efficiency_pct_after, Some(55.0));
        assert_eq!(d.cost_per_million_before, Some(2.50));
        assert_eq!(d.cost_per_million_after, Some(2.00));
        assert_eq!(d.cost_source_after, Some(CostSource::Catalog));
        assert!((d.efficiency_delta_pp.unwrap() - 15.0).abs() < 1e-9);
    }

    #[test]
    fn populates_joules_per_token_before_after() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(40.0), None, Some(0.31)),
            &diagnose(Some(110.0)),
            &report_eff(Some(45.0), None, Some(0.28)),
            false,
        );
        assert_eq!(d.joules_per_token_before, Some(0.31));
        assert_eq!(d.joules_per_token_after, Some(0.28));
    }

    #[test]
    fn populates_p99_before_after() {
        let mut prev = diagnose(Some(100.0));
        prev.snapshot.vllm.ttft_p99_ms = Some(500.0);
        prev.snapshot.vllm.tpot_p99_ms = Some(80.0);
        let mut curr = diagnose(Some(110.0));
        curr.snapshot.vllm.ttft_p99_ms = Some(450.0);
        curr.snapshot.vllm.tpot_p99_ms = Some(75.0);
        let d = compute(
            &prev,
            &report_eff(Some(40.0), None, None),
            &curr,
            &report_eff(Some(45.0), None, None),
            false,
        );
        assert_eq!(d.ttft_p99_before_ms, Some(500.0));
        assert_eq!(d.ttft_p99_after_ms, Some(450.0));
        assert_eq!(d.tpot_p99_before_ms, Some(80.0));
        assert_eq!(d.tpot_p99_after_ms, Some(75.0));
    }
}
