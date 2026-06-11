use crate::engine::Report;

use super::DiagnoseResult;

// Direction thresholds. Provisional: calibrate against real workloads before hardening.
const EFFICIENCY_BETTER_PP: f64 = 2.0;
const EFFICIENCY_WORSE_PP: f64 = -5.0;
const DEFAULT_LATENCY_VETO_PCT: f64 = 20.0;
const THROUGHPUT_BETTER_PCT: f64 = 10.0;
const THROUGHPUT_WORSE_PCT: f64 = -10.0;

/// Placeholder until NLP `Goal` is wired from `cli/goal/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Goal;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Better,
    Worse,
    /// Signals contradict each other; reason always set.
    Mixed,
    /// No meaningful change, or no measurable signal.
    NoChange,
}

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
    pub direction: Direction,
    pub direction_reason: Option<&'static str>,
    pub ttft_p99_delta_pct: Option<f64>,
    /// True when the latency veto condition is met (efficiency gain with ttft_p99 regression).
    /// Set in `compute()`; read by `direction_detail` for output.
    pub veto_fired: bool,
    pub config_drifted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Signal {
    Up,
    Flat,
    Down,
}

fn eval_eff(pp: f64) -> Signal {
    if pp >= EFFICIENCY_BETTER_PP {
        Signal::Up
    } else if pp <= EFFICIENCY_WORSE_PP {
        Signal::Down
    } else {
        Signal::Flat
    }
}

fn eval_ttft(pct: f64) -> Signal {
    if pct <= -DEFAULT_LATENCY_VETO_PCT {
        Signal::Up
    } else if pct >= DEFAULT_LATENCY_VETO_PCT {
        Signal::Down
    } else {
        Signal::Flat
    }
}

fn eval_tput(pct: f64) -> Signal {
    if pct >= THROUGHPUT_BETTER_PCT {
        Signal::Up
    } else if pct <= THROUGHPUT_WORSE_PCT {
        Signal::Down
    } else {
        Signal::Flat
    }
}

fn ttft_p99_regression_pct(delta: &Delta) -> Option<f64> {
    delta
        .ttft_p99_after_ms
        .zip(delta.ttft_p99_before_ms)
        .and_then(|(after, before)| {
            if before > 0.0 && after.is_finite() && before.is_finite() {
                Some((after - before) / before * 100.0)
            } else {
                None
            }
        })
}

fn latency_veto_fires(delta: &Delta) -> bool {
    delta
        .efficiency_delta_pp
        .is_some_and(|eff| eff > EFFICIENCY_BETTER_PP)
        && ttft_p99_regression_pct(delta).is_some_and(|v| v > DEFAULT_LATENCY_VETO_PCT)
}

fn evaluate_efficiency_delta(delta: &Delta) -> (Direction, Option<&'static str>) {
    use Signal::*;

    let eff = match delta.efficiency_delta_pp {
        Some(v) if v.is_finite() => eval_eff(v),
        _ => {
            return match delta.throughput_delta_pct {
                Some(d) if d > THROUGHPUT_BETTER_PCT => (Direction::Better, None),
                Some(d) if d < THROUGHPUT_WORSE_PCT => (Direction::Worse, None),
                Some(_) => (Direction::NoChange, None),
                None => (Direction::NoChange, None),
            };
        }
    };

    let ttft = delta
        .ttft_p99_delta_pct
        .filter(|v| v.is_finite())
        .map(eval_ttft)
        .unwrap_or(Flat);

    let tput = delta
        .throughput_delta_pct
        .filter(|v| v.is_finite())
        .map(eval_tput)
        .unwrap_or(Flat);

    match (eff, ttft, tput) {
        (Up, Up, Up) => (Direction::Better, None),
        (Up, Up, Flat) => (Direction::Better, None),
        (Up, Up, Down) => (
            Direction::Better,
            Some("efficiency + latency positive; throughput drop likely reduced concurrency"),
        ),
        (Up, Flat, Up) => (Direction::Better, None),
        (Up, Flat, Flat) => (Direction::Better, None),
        (Up, Flat, Down) => (
            Direction::Better,
            Some("efficiency primary; throughput drop likely load reduction"),
        ),
        (Up, Down, Up) => (
            Direction::Mixed,
            Some("efficiency up, TTFT worse; check latency SLA"),
        ),
        (Up, Down, Flat) => (
            Direction::Mixed,
            Some("efficiency up, TTFT degraded; throughput unchanged"),
        ),
        (Up, Down, Down) => (
            Direction::Worse,
            Some("latency broken, throughput fell; efficiency gain is artifact"),
        ),

        (Flat, Up, Up) => (
            Direction::Better,
            Some("efficiency flat, TTFT + throughput improved; scheduler win"),
        ),
        (Flat, Up, Flat) => (
            Direction::Better,
            Some("efficiency flat, p99 TTFT dropped significantly"),
        ),
        (Flat, Up, Down) => (
            Direction::Mixed,
            Some("latency improved, throughput fell; check load"),
        ),
        (Flat, Flat, Up) => (
            Direction::Better,
            Some("efficiency flat, throughput expanded"),
        ),
        (Flat, Flat, Flat) => (Direction::NoChange, None),
        (Flat, Flat, Down) => (Direction::Worse, Some("efficiency flat, throughput fell")),
        (Flat, Down, Up) => (
            Direction::Mixed,
            Some("throughput up, TTFT worse; check KV pressure"),
        ),
        (Flat, Down, Flat) => (
            Direction::Worse,
            Some("efficiency flat, latency SLA violated"),
        ),
        (Flat, Down, Down) => (Direction::Worse, Some("latency spiked and throughput fell")),

        (Down, Up, Up) => (
            Direction::Better,
            Some("scheduling improved; efficiency drop reflects larger batch"),
        ),
        (Down, Up, Flat) => (
            Direction::Mixed,
            Some("efficiency fell, latency better; verify load didn't drop"),
        ),
        (Down, Up, Down) => (
            Direction::Worse,
            Some("efficiency + throughput fell; latency gain is load artifact"),
        ),
        (Down, Flat, Up) => (
            Direction::Mixed,
            Some("throughput expanded, efficiency fell; monitor fragmentation"),
        ),
        (Down, Flat, Flat) => (Direction::Worse, Some("clear regression")),
        (Down, Flat, Down) => (
            Direction::Worse,
            Some("clear regression, throughput also fell"),
        ),
        (Down, Down, Up) => (
            Direction::Worse,
            Some("efficiency + latency broken; throughput gain doesn't save it"),
        ),
        (Down, Down, Flat) => (Direction::Worse, Some("efficiency + latency degraded")),
        (Down, Down, Down) => (
            Direction::Worse,
            Some("severe regression; everything degraded"),
        ),
    }
}

fn rule_direction_reason(
    prev_rule: Option<&'static str>,
    new_rule: Option<&'static str>,
    direction: Direction,
) -> Option<&'static str> {
    match (prev_rule, new_rule, direction) {
        (Some("concurrency_saturation"), None, Direction::Better) => {
            Some("scheduler bottleneck cleared")
        }
        (Some("concurrency_saturation"), Some("concurrency_saturation"), Direction::NoChange) => {
            Some("scheduler still saturated")
        }
        (Some("kv_cache_pressure"), None, Direction::Better) => Some("KV pressure cleared"),
        (Some("kv_cache_pressure"), Some("kv_cache_pressure"), Direction::NoChange) => {
            Some("KV pressure persists")
        }
        (Some("under_batching"), None, Direction::Better) => Some("batch utilization improved"),
        (Some("under_batching"), Some("under_batching"), Direction::NoChange) => {
            Some("batch still under-utilized")
        }
        (Some("low_prefix_reuse"), None, Direction::Better) => {
            Some("prefix cache hit rate improved")
        }
        (Some("low_prefix_reuse"), Some("low_prefix_reuse"), Direction::NoChange) => {
            Some("prefix cache miss persists")
        }
        _ => None,
    }
}

/// v2: Default/Throughput path only.
///
/// `goal` is reserved; exhaustive match ensures future objectives fail to compile until handled.
pub fn calculate_direction(
    delta: &Delta,
    _goal: Option<&Goal>,
) -> (Direction, Option<&'static str>) {
    evaluate_efficiency_delta(delta)
}

/// Human-readable signal detail for direction output (no label prefix).
pub fn direction_detail(delta: &Delta) -> String {
    match delta.efficiency_delta_pp {
        Some(eff) => {
            if delta.veto_fired && delta.direction == Direction::Mixed {
                if let Some(veto) = ttft_p99_regression_pct(delta) {
                    return format!(
                        "efficiency Δ: {:+.1} pp  |  ttft_p99 Δ: {:+.1}%  latency veto",
                        eff, veto
                    );
                }
            }
            if let Some(reason) = delta.direction_reason {
                return reason.to_string();
            }
            format!("efficiency Δ: {:+.1} pp", eff)
        }
        None => match delta.throughput_delta_pct {
            Some(t) => format!("efficiency unavailable  throughput Δ: {:+.1}%", t),
            None => "no measurable signal".to_string(),
        },
    }
}

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

    let ttft_p99_delta_pct = match (ttft_p99_before_ms, ttft_p99_after_ms) {
        (Some(b), Some(a)) if b > 0.0 && b.is_finite() && a.is_finite() => {
            Some((a - b) / b * 100.0)
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

    let mut delta = Delta {
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
        direction: Direction::NoChange,
        direction_reason: None,
        ttft_p99_delta_pct,
        veto_fired: false,
        config_drifted,
    };
    let (direction, reason) = calculate_direction(&delta, None);
    let prev_rule = prev_report.groups.first().map(|g| g.primary.rule_name);
    let new_rule = curr_report.groups.first().map(|g| g.primary.rule_name);
    delta.direction = direction;
    delta.direction_reason = rule_direction_reason(prev_rule, new_rule, direction).or(reason);
    delta.veto_fired = latency_veto_fires(&delta);
    delta
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

    fn mk_delta(
        eff_pp: Option<f64>,
        throughput_pct: Option<f64>,
        ttft_p99_before: Option<f64>,
        ttft_p99_after: Option<f64>,
    ) -> Delta {
        let ttft_p99_delta_pct = match (ttft_p99_before, ttft_p99_after) {
            (Some(b), Some(a)) if b > 0.0 && b.is_finite() && a.is_finite() => {
                Some((a - b) / b * 100.0)
            }
            _ => None,
        };
        Delta {
            throughput_delta_pct: throughput_pct,
            throughput_before: None,
            throughput_after: None,
            efficiency_delta_pp: eff_pp,
            efficiency_pct_before: None,
            efficiency_pct_after: None,
            cost_per_million_before: None,
            cost_per_million_after: None,
            joules_per_token_before: None,
            joules_per_token_after: None,
            cost_source_after: None,
            ttft_before_ms: None,
            ttft_after_ms: None,
            tpot_before_ms: None,
            tpot_after_ms: None,
            ttft_p99_before_ms: ttft_p99_before,
            ttft_p99_after_ms: ttft_p99_after,
            tpot_p99_before_ms: None,
            tpot_p99_after_ms: None,
            direction: Direction::NoChange,
            direction_reason: None,
            ttft_p99_delta_pct,
            veto_fired: false,
            config_drifted: false,
        }
    }

    fn finalize_delta(mut d: Delta) -> Delta {
        let (direction, reason) = calculate_direction(&d, None);
        d.direction = direction;
        d.direction_reason = reason;
        d.veto_fired = latency_veto_fires(&d);
        d
    }

    fn eval_dir(d: &Delta) -> (Direction, Option<&'static str>) {
        evaluate_efficiency_delta(d)
    }

    #[test]
    fn efficiency_better_above_threshold() {
        let (dir, _) = eval_dir(&mk_delta(Some(3.0), None, Some(500.0), Some(500.0)));
        assert_eq!(dir, Direction::Better);
    }

    #[test]
    fn efficiency_worse_below_threshold() {
        let (dir, reason) = eval_dir(&mk_delta(Some(-6.0), None, None, None));
        assert_eq!(dir, Direction::Worse);
        assert_eq!(reason, Some("clear regression"));
    }

    #[test]
    fn efficiency_no_change_within_band() {
        let (dir, reason) = eval_dir(&mk_delta(Some(1.0), None, None, None));
        assert_eq!(dir, Direction::NoChange);
        assert!(reason.is_none());
    }

    #[test]
    fn latency_veto_fires_on_better() {
        let (dir, reason) = eval_dir(&mk_delta(Some(3.0), None, Some(400.0), Some(500.0)));
        assert_eq!(dir, Direction::Mixed);
        assert_eq!(
            reason,
            Some("efficiency up, TTFT degraded; throughput unchanged")
        );
    }

    #[test]
    fn latency_veto_does_not_fire_on_no_change() {
        let d = finalize_delta(mk_delta(Some(1.0), None, None, None));
        assert!(!d.veto_fired);
        assert_eq!(d.direction, Direction::NoChange);
    }

    #[test]
    fn latency_veto_does_not_fire_on_worse() {
        let (dir, _) = eval_dir(&mk_delta(Some(-6.0), None, Some(400.0), Some(500.0)));
        assert_eq!(dir, Direction::Worse);
    }

    #[test]
    fn latency_veto_skipped_when_ttft_none() {
        let (dir, _) = eval_dir(&mk_delta(Some(3.0), None, None, Some(500.0)));
        assert_eq!(dir, Direction::Better);
    }

    #[test]
    fn efficiency_none_fallback_better() {
        let (dir, _) = eval_dir(&mk_delta(None, Some(12.0), None, None));
        assert_eq!(dir, Direction::Better);
    }

    #[test]
    fn efficiency_none_fallback_worse() {
        let (dir, _) = eval_dir(&mk_delta(None, Some(-15.0), None, None));
        assert_eq!(dir, Direction::Worse);
    }

    #[test]
    fn efficiency_none_fallback_no_change() {
        let (dir, _) = eval_dir(&mk_delta(None, Some(4.0), None, None));
        assert_eq!(dir, Direction::NoChange);
    }

    #[test]
    fn efficiency_and_throughput_both_none() {
        let (dir, _) = eval_dir(&mk_delta(None, None, None, None));
        assert_eq!(dir, Direction::NoChange);
    }

    #[test]
    fn scheduler_win() {
        let d = finalize_delta(mk_delta(Some(1.0), Some(15.0), Some(100.0), Some(75.0)));
        assert_eq!(d.direction, Direction::Better);
        assert!(d
            .direction_reason
            .is_some_and(|r| r.contains("scheduler win")));
    }

    #[test]
    fn latency_only_win() {
        let d = finalize_delta(mk_delta(Some(1.0), Some(5.0), Some(100.0), Some(75.0)));
        assert_eq!(d.direction, Direction::Better);
        assert!(d
            .direction_reason
            .is_some_and(|r| r.contains("TTFT dropped significantly")));
    }

    #[test]
    fn tput_only_win() {
        let d = finalize_delta(mk_delta(Some(1.0), Some(15.0), Some(100.0), Some(95.0)));
        assert_eq!(d.direction, Direction::Better);
        assert!(d
            .direction_reason
            .is_some_and(|r| r.contains("throughput expanded")));
    }

    #[test]
    fn hidden_regression() {
        let d = finalize_delta(mk_delta(Some(1.0), Some(15.0), Some(100.0), Some(125.0)));
        assert_eq!(d.direction, Direction::Mixed);
        assert!(d
            .direction_reason
            .is_some_and(|r| r.contains("throughput up, TTFT worse")));
    }

    #[test]
    fn downward_override() {
        let d = finalize_delta(mk_delta(Some(1.0), Some(-5.0), Some(100.0), Some(125.0)));
        assert_eq!(d.direction, Direction::Worse);
        assert!(d
            .direction_reason
            .is_some_and(|r| r.contains("latency SLA violated")));
    }

    #[test]
    fn eff_down_mixed() {
        let d = finalize_delta(mk_delta(Some(-6.0), Some(15.0), Some(100.0), Some(75.0)));
        assert_eq!(d.direction, Direction::Better);
        assert!(d
            .direction_reason
            .is_some_and(|r| r.contains("scheduling improved")));
    }

    #[test]
    fn severe_regression() {
        let d = finalize_delta(mk_delta(Some(-6.0), Some(-15.0), Some(100.0), Some(125.0)));
        assert_eq!(d.direction, Direction::Worse);
        assert!(d
            .direction_reason
            .is_some_and(|r| r.contains("severe regression")));
    }

    #[test]
    fn latency_veto_intact() {
        let d = finalize_delta(mk_delta(Some(3.0), Some(15.0), Some(100.0), Some(125.0)));
        assert_eq!(d.direction, Direction::Mixed);
        assert!(d.veto_fired);
    }

    #[test]
    fn no_signal() {
        let d = finalize_delta(mk_delta(None, None, None, None));
        assert_eq!(d.direction, Direction::NoChange);
    }

    #[test]
    fn none_goal_uses_efficiency_path() {
        let d = finalize_delta(mk_delta(Some(3.0), None, None, None));
        assert_eq!(d.direction, Direction::Better);
    }

    #[test]
    fn veto_fired_true_when_veto_fires() {
        let d = finalize_delta(mk_delta(Some(3.0), None, Some(400.0), Some(500.0)));
        assert!(d.veto_fired);
        assert_eq!(d.direction, Direction::Mixed);
    }

    #[test]
    fn veto_fired_false_when_no_veto() {
        let d = finalize_delta(mk_delta(Some(3.0), None, Some(400.0), Some(410.0)));
        assert!(!d.veto_fired);
        assert_eq!(d.direction, Direction::Better);
    }

    #[test]
    fn compute_direction_better_on_efficiency_not_raw_throughput() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0), None, None),
            &diagnose(Some(112.0)),
            &report_eff(Some(55.0), None, None),
            false,
        );
        assert_eq!(d.direction, Direction::Better);
        assert!((d.efficiency_delta_pp.unwrap() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn compute_direction_worse_on_efficiency_drop() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0), None, None),
            &diagnose(Some(89.0)),
            &report_eff(Some(43.0), None, None),
            false,
        );
        assert_eq!(d.direction, Direction::Worse);
    }

    #[test]
    fn compute_direction_no_change_when_efficiency_flat_despite_throughput_swing() {
        let d = compute(
            &diagnose(Some(100.0)),
            &report_eff(Some(50.0), None, None),
            &diagnose(Some(91.0)),
            &report_eff(Some(50.0), None, None),
            false,
        );
        assert_eq!(d.direction, Direction::NoChange);
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
        assert_eq!(d.direction, Direction::NoChange);
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
        assert!((d.ttft_p99_delta_pct.unwrap() - (-10.0)).abs() < 1e-9);
    }
}
