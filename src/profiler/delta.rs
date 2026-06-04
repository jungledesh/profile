use crate::engine::Report;

use super::DiagnoseResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Better,
    Worse,
    /// &lt; 2% relative change in throughput (when known).
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
    pub ttft_before_ms: Option<f64>,
    pub ttft_after_ms: Option<f64>,
    pub tpot_before_ms: Option<f64>,
    pub tpot_after_ms: Option<f64>,
    pub direction: Direction,
    pub config_drifted: bool,
}

const PLATEAU_THRESHOLD_PCT: f64 = 2.0;

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

    let throughput_delta_pct = match (throughput_before, throughput_after) {
        (Some(p), Some(c)) if p > 0.0 && p.is_finite() && c.is_finite() => {
            Some((c - p) / p * 100.0)
        }
        _ => None,
    };

    let efficiency_delta_pp = match (
        prev_report.baseline.as_ref().and_then(|b| b.efficiency_pct),
        curr_report.baseline.as_ref().and_then(|b| b.efficiency_pct),
    ) {
        (Some(p), Some(c)) if p.is_finite() && c.is_finite() => Some(c - p),
        _ => None,
    };

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
        ttft_before_ms,
        ttft_after_ms,
        tpot_before_ms,
        tpot_after_ms,
        direction,
        config_drifted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn report_eff(eff: Option<f64>) -> Report {
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
                cost: None,
            }),
            groups: Vec::new(),
            r2_suppressed_by_r4: false,
        }
    }

    #[test]
    fn direction_better_above_threshold() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0)),
            &diagnose(Some(110.0)),
            &report_eff(Some(55.0)),
            false,
        );
        assert_eq!(d.direction, Direction::Better);
        assert!((d.throughput_delta_pct.unwrap() - 10.0).abs() < 1e-9);
        assert_eq!(d.throughput_before, Some(100.0));
        assert_eq!(d.throughput_after, Some(110.0));
    }

    #[test]
    fn direction_worse_below_neg_threshold() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0)),
            &diagnose(Some(80.0)),
            &report_eff(Some(45.0)),
            false,
        );
        assert_eq!(d.direction, Direction::Worse);
        assert!((d.throughput_delta_pct.unwrap() + 20.0).abs() < 1e-9);
    }

    #[test]
    fn direction_plateau_within_band() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0)),
            &diagnose(Some(101.0)),
            &report_eff(Some(50.5)),
            false,
        );
        assert_eq!(d.direction, Direction::Plateau);
    }

    #[test]
    fn throughput_delta_none_when_prev_zero() {
        let d = compute(
            &diagnose(Some(0.0)),
            &report_eff(None),
            &diagnose(Some(50.0)),
            &report_eff(Some(10.0)),
            false,
        );
        assert!(d.throughput_delta_pct.is_none());
        assert_eq!(d.direction, Direction::Plateau);
    }

    #[test]
    fn throughput_delta_none_when_missing_gauge() {
        let d = compute(
            &diagnose(None),
            &report_eff(None),
            &diagnose(Some(50.0)),
            &report_eff(None),
            false,
        );
        assert!(d.throughput_delta_pct.is_none());
    }

    #[test]
    fn config_drifted_passthrough() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0)),
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0)),
            true,
        );
        assert!(d.config_drifted);
    }
}
