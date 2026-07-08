use crate::engine::Report;

use super::DiagnoseResult;

#[derive(Debug, Clone)]
pub struct Delta {
    /// Relative % change in `generation_tokens_per_sec`.
    pub throughput_delta_pct: Option<f64>,
    pub throughput_before: Option<f64>,
    pub throughput_after: Option<f64>,
    /// Absolute percentage-point change in efficiency.
    ///
    /// Efficiency = actual_tps / (decode_ceiling × num_running), where num_running
    /// is the time-weighted average across active windows in the measurement period.
    /// This normalization cancels traffic-induced concurrency changes so the signal
    /// reflects per-request hardware utilization, not load volume.
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
    pub ttft_p95_before_ms: Option<f64>,
    pub ttft_p95_after_ms: Option<f64>,
    pub tpot_p95_before_ms: Option<f64>,
    pub tpot_p95_after_ms: Option<f64>,
    pub ttft_p99_delta_pct: Option<f64>,
    pub ttft_p95_delta_pct: Option<f64>,
    pub tpot_p95_delta_pct: Option<f64>,
    pub config_drifted: bool,
    /// Non-baseline config diffs for display (e.g. max_num_seqs).
    pub config_changes: Vec<String>,
}

fn pct_delta(before: Option<f64>, after: Option<f64>) -> Option<f64> {
    match (before, after) {
        (Some(b), Some(a)) if b > 0.0 && b.is_finite() && a.is_finite() => {
            Some((a - b) / b * 100.0)
        }
        _ => None,
    }
}

pub fn compute(
    prev_result: &DiagnoseResult,
    prev_report: &Report,
    curr_result: &DiagnoseResult,
    curr_report: &Report,
    config_drifted: bool,
    config_changes: Vec<String>,
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
    let ttft_p95_before_ms = prev_result.snapshot.vllm.ttft_p95_ms;
    let ttft_p95_after_ms = curr_result.snapshot.vllm.ttft_p95_ms;
    let tpot_p95_before_ms = prev_result.snapshot.vllm.tpot_p95_ms;
    let tpot_p95_after_ms = curr_result.snapshot.vllm.tpot_p95_ms;

    let throughput_delta_pct = pct_delta(throughput_before, throughput_after);
    let ttft_p99_delta_pct = pct_delta(ttft_p99_before_ms, ttft_p99_after_ms);
    let ttft_p95_delta_pct = pct_delta(ttft_p95_before_ms, ttft_p95_after_ms);
    let tpot_p95_delta_pct = pct_delta(tpot_p95_before_ms, tpot_p95_after_ms);

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
        ttft_p95_before_ms,
        ttft_p95_after_ms,
        tpot_p95_before_ms,
        tpot_p95_after_ms,
        ttft_p99_delta_pct,
        ttft_p95_delta_pct,
        tpot_p95_delta_pct,
        config_drifted,
        config_changes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::baseline::{CostEstimate, CostSource};
    use crate::engine::{CeilingEstimate, PhysicsBaseline, WeightDtypeSource};
    use crate::{
        collectors::{RawSnapshot, VllmRawMetrics},
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
            gpus: vec![],

            nvml_host_gpu_count: None,
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
                config_relative_efficiency_pct: None,
                cost,
            }),
            groups: Vec::new(),
            suppressed_rules: Vec::new(),
            kv_max_seqs: None,
            n_eval: 0,
            skipped: 0,
        }
    }

    #[test]
    fn pct_delta_returns_none_when_before_zero() {
        assert!(pct_delta(Some(0.0), Some(50.0)).is_none());
    }

    #[test]
    fn pct_delta_returns_correct_value() {
        assert!((pct_delta(Some(100.0), Some(110.0)).unwrap() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn compute_populates_efficiency_delta_pp() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0), None, None),
            &diagnose(Some(112.0)),
            &report_eff(Some(55.0), None, None),
            false,
            Vec::new(),
        );
        assert!((d.efficiency_delta_pp.unwrap() - 5.0).abs() < 1e-9);
        assert!((d.throughput_delta_pct.unwrap() - 12.0).abs() < 1e-9);
    }

    #[test]
    fn throughput_delta_none_when_prev_zero() {
        let d = compute(
            &diagnose(Some(0.0)),
            &report_eff(None, None, None),
            &diagnose(Some(50.0)),
            &report_eff(Some(10.0), None, None),
            false,
            Vec::new(),
        );
        assert!(d.throughput_delta_pct.is_none());
    }

    #[test]
    fn throughput_delta_none_when_missing_gauge() {
        let d = compute(
            &diagnose(None),
            &report_eff(None, None, None),
            &diagnose(Some(50.0)),
            &report_eff(None, None, None),
            false,
            Vec::new(),
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
            vec!["  max_num_seqs  32 -> 98".to_string()],
        );
        assert!(d.config_drifted);
        assert_eq!(d.config_changes.len(), 1);
    }

    #[test]
    fn config_changes_passthrough() {
        let changes = vec!["  max_num_seqs  32 -> 98".to_string()];
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0), None, None),
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0), None, None),
            false,
            changes.clone(),
        );
        assert_eq!(d.config_changes, changes);
    }

    #[test]
    fn populates_cost_and_efficiency_before_after() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(40.0), Some(2.50), Some(0.31)),
            &diagnose(Some(120.0)),
            &report_eff(Some(55.0), Some(2.00), Some(0.28)),
            false,
            Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
        );
        assert_eq!(d.ttft_p99_before_ms, Some(500.0));
        assert_eq!(d.ttft_p99_after_ms, Some(450.0));
        assert_eq!(d.tpot_p99_before_ms, Some(80.0));
        assert_eq!(d.tpot_p99_after_ms, Some(75.0));
        assert!((d.ttft_p99_delta_pct.unwrap() - (-10.0)).abs() < 1e-9);
    }

    #[test]
    fn populates_p95_before_after() {
        let mut prev = diagnose(Some(100.0));
        prev.snapshot.vllm.ttft_p95_ms = Some(400.0);
        prev.snapshot.vllm.tpot_p95_ms = Some(70.0);
        let mut curr = diagnose(Some(110.0));
        curr.snapshot.vllm.ttft_p95_ms = Some(340.0);
        curr.snapshot.vllm.tpot_p95_ms = Some(63.0);
        let d = compute(
            &prev,
            &report_eff(Some(40.0), None, None),
            &curr,
            &report_eff(Some(45.0), None, None),
            false,
            Vec::new(),
        );
        assert_eq!(d.ttft_p95_before_ms, Some(400.0));
        assert_eq!(d.ttft_p95_after_ms, Some(340.0));
        assert_eq!(d.tpot_p95_before_ms, Some(70.0));
        assert_eq!(d.tpot_p95_after_ms, Some(63.0));
        assert!((d.ttft_p95_delta_pct.unwrap() - (-15.0)).abs() < 1e-9);
        assert!((d.tpot_p95_delta_pct.unwrap() - (-10.0)).abs() < 1e-9);
    }
}
