use crate::engine::Report;

use super::DiagnoseResult;

// Direction thresholds. Provisional: calibrate against real workloads before hardening.
const THROUGHPUT_IMPROVED_PCT: f64 = 10.0;
const THROUGHPUT_DEGRADED_PCT: f64 = -10.0;
const TTFT_P95_IMPROVED_PCT: f64 = -15.0; // negative = lower latency = better
const TTFT_P95_DEGRADED_PCT: f64 = 20.0;
const TPOT_P95_IMPROVED_PCT: f64 = -10.0; // negative = lower latency = better
const TPOT_P95_DEGRADED_PCT: f64 = 15.0;

// Minimum 2ms regardless of floor — safety net when baseline is unavailable
const TPOT_JITTER_FLOOR_MS: f64 = 2.0;
// Minimum 10ms regardless of floor
const TTFT_JITTER_FLOOR_MS: f64 = 10.0;

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
    pub ttft_p95_before_ms: Option<f64>,
    pub ttft_p95_after_ms: Option<f64>,
    pub tpot_p95_before_ms: Option<f64>,
    pub tpot_p95_after_ms: Option<f64>,
    pub direction: Direction,
    pub direction_reason: Option<&'static str>,
    pub ttft_p99_delta_pct: Option<f64>,
    pub ttft_p95_delta_pct: Option<f64>,
    pub tpot_p95_delta_pct: Option<f64>,
    pub tpot_floor_ms: Option<f64>,
    pub prefill_latency_floor_ms: Option<f64>,
    pub config_drifted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Signal {
    Improved,
    Flat,
    Degraded,
}

fn pct_delta(before: Option<f64>, after: Option<f64>) -> Option<f64> {
    match (before, after) {
        (Some(b), Some(a)) if b > 0.0 && b.is_finite() && a.is_finite() => {
            Some((a - b) / b * 100.0)
        }
        _ => None,
    }
}

fn eval_throughput(pct: f64) -> Signal {
    if pct >= THROUGHPUT_IMPROVED_PCT {
        Signal::Improved
    } else if pct <= THROUGHPUT_DEGRADED_PCT {
        Signal::Degraded
    } else {
        Signal::Flat
    }
}

fn eval_ttft_p95(pct: f64, delta_ms: f64, prefill_floor_ms: Option<f64>) -> Signal {
    // Jitter margin: 1.5× the physics prefill floor.
    // TTFT spikes by full prefill durations from queue timing alone —
    // absorb at least one unlucky queue-wait before declaring regression.
    let jitter = prefill_floor_ms
        .map(|f| (f * 1.5).max(TTFT_JITTER_FLOOR_MS))
        .unwrap_or(TTFT_JITTER_FLOOR_MS);
    if delta_ms > 0.0 && delta_ms < jitter {
        return Signal::Flat;
    }
    if pct <= TTFT_P95_IMPROVED_PCT {
        Signal::Improved
    } else if pct >= TTFT_P95_DEGRADED_PCT {
        Signal::Degraded
    } else {
        Signal::Flat
    }
}

fn eval_tpot_p95(pct: f64, delta_ms: f64, tpot_floor_ms: Option<f64>) -> Signal {
    // Jitter margin: half the physics decode floor.
    // At low latency, percentage swings are noisy — only flag degradation
    // when the absolute increase exceeds hardware-level noise.
    let jitter = tpot_floor_ms
        .map(|f| (f * 0.5).max(TPOT_JITTER_FLOOR_MS))
        .unwrap_or(TPOT_JITTER_FLOOR_MS);
    if delta_ms > 0.0 && delta_ms < jitter {
        return Signal::Flat;
    }
    if pct <= TPOT_P95_IMPROVED_PCT {
        Signal::Improved
    } else if pct >= TPOT_P95_DEGRADED_PCT {
        Signal::Degraded
    } else {
        Signal::Flat
    }
}

fn evaluate_direction(delta: &Delta) -> (Direction, Option<&'static str>) {
    use Signal::*;

    let tput = delta
        .throughput_delta_pct
        .filter(|v| v.is_finite())
        .map(eval_throughput)
        .unwrap_or(Flat);

    let tpot_delta_ms = match (delta.tpot_p95_before_ms, delta.tpot_p95_after_ms) {
        (Some(b), Some(a)) => a - b,
        _ => 0.0,
    };
    let ttft_delta_ms = match (delta.ttft_p95_before_ms, delta.ttft_p95_after_ms) {
        (Some(b), Some(a)) => a - b,
        _ => 0.0,
    };

    let ttft = delta
        .ttft_p95_delta_pct
        .filter(|v| v.is_finite())
        .map(|pct| eval_ttft_p95(pct, ttft_delta_ms, delta.prefill_latency_floor_ms))
        .unwrap_or(Flat);

    let tpot = delta
        .tpot_p95_delta_pct
        .filter(|v| v.is_finite())
        .map(|pct| eval_tpot_p95(pct, tpot_delta_ms, delta.tpot_floor_ms))
        .unwrap_or(Flat);

    match (tput, ttft, tpot) {
        (Improved, Improved, Improved) => (Direction::Better, None),
        (Improved, Improved, Flat) => (Direction::Better, None),
        (Improved, Improved, Degraded) => (
            Direction::Mixed,
            Some("throughput + TTFT improved; TPOT worse — check KV pressure under higher load"),
        ),
        (Improved, Flat, Improved) => (Direction::Better, None),
        (Improved, Flat, Flat) => (Direction::Better, Some("throughput expanded")),
        (Improved, Flat, Degraded) => (
            Direction::Mixed,
            Some("throughput up, TPOT degraded; KV cache pressure likely"),
        ),
        (Improved, Degraded, Improved) => (
            Direction::Mixed,
            Some("throughput + TPOT improved; TTFT spiked — check prefill saturation"),
        ),
        (Improved, Degraded, Flat) => (Direction::Mixed, Some("throughput up, TTFT spiked")),
        (Improved, Degraded, Degraded) => (
            Direction::Worse,
            Some("latency broken; throughput gain at cost of SLA"),
        ),
        (Flat, Improved, Improved) => (Direction::Better, None),
        (Flat, Improved, Flat) => (Direction::Better, Some("TTFT p95 improved")),
        (Flat, Improved, Degraded) => (
            Direction::Mixed,
            Some("TTFT improved, TPOT worse; prefill vs decode trade-off"),
        ),
        (Flat, Flat, Improved) => (Direction::Better, Some("TPOT p95 improved")),
        (Flat, Flat, Flat) => (Direction::NoChange, None),
        (Flat, Flat, Degraded) => (Direction::Worse, Some("TPOT p95 degraded")),
        (Flat, Degraded, Improved) => (
            Direction::Mixed,
            Some("TTFT spiked, TPOT improved; prefill contention"),
        ),
        (Flat, Degraded, Flat) => (Direction::Worse, Some("TTFT p95 spiked")),
        (Flat, Degraded, Degraded) => (Direction::Worse, Some("latency degraded across board")),
        (Degraded, Improved, Improved) => (
            Direction::Mixed,
            Some("throughput fell; latency improvement may be load artifact"),
        ),
        (Degraded, Improved, Flat) => (
            Direction::Worse,
            Some("throughput fell; TTFT gain is load artifact"),
        ),
        (Degraded, Improved, Degraded) => (
            Direction::Worse,
            Some("throughput fell, TPOT worse; TTFT gain is load artifact"),
        ),
        (Degraded, Flat, Improved) => (
            Direction::Mixed,
            Some("throughput fell, TPOT improved; load drop likely"),
        ),
        (Degraded, Flat, Flat) => (Direction::Worse, Some("throughput fell")),
        (Degraded, Flat, Degraded) => (Direction::Worse, Some("throughput fell, TPOT degraded")),
        (Degraded, Degraded, Improved) => (Direction::Worse, Some("throughput + TTFT degraded")),
        (Degraded, Degraded, Flat) => (Direction::Worse, Some("throughput + TTFT degraded")),
        (Degraded, Degraded, Degraded) => (
            Direction::Worse,
            Some("severe regression; all signals degraded"),
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
        (Some("kv_admission_backlog"), None, Direction::Better) => {
            Some("KV admission backlog cleared")
        }
        (Some("kv_admission_backlog"), Some("kv_admission_backlog"), Direction::NoChange) => {
            Some("KV admission backlog persists")
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
    evaluate_direction(delta)
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

    let tpot_floor_ms = curr_report.baseline.as_ref().map(|b| b.tpot_floor_ms);
    let prefill_latency_floor_ms = curr_report
        .baseline
        .as_ref()
        .and_then(|b| b.prefill_latency_floor_ms);

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
        ttft_p95_before_ms,
        ttft_p95_after_ms,
        tpot_p95_before_ms,
        tpot_p95_after_ms,
        direction: Direction::NoChange,
        direction_reason: None,
        ttft_p99_delta_pct,
        ttft_p95_delta_pct,
        tpot_p95_delta_pct,
        tpot_floor_ms,
        prefill_latency_floor_ms,
        config_drifted,
    };
    let (direction, reason) = calculate_direction(&delta, None);
    let prev_rule = prev_report.groups.first().map(|g| g.primary.rule_name);
    let new_rule = curr_report.groups.first().map(|g| g.primary.rule_name);
    delta.direction = direction;
    delta.direction_reason = rule_direction_reason(prev_rule, new_rule, direction).or(reason);
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
        throughput_pct: Option<f64>,
        ttft_p95_before: Option<f64>,
        ttft_p95_after: Option<f64>,
        tpot_p95_before: Option<f64>,
        tpot_p95_after: Option<f64>,
    ) -> Delta {
        mk_delta_with_floors(
            throughput_pct,
            ttft_p95_before,
            ttft_p95_after,
            tpot_p95_before,
            tpot_p95_after,
            None,
            None,
        )
    }

    fn mk_delta_with_floors(
        throughput_pct: Option<f64>,
        ttft_p95_before: Option<f64>,
        ttft_p95_after: Option<f64>,
        tpot_p95_before: Option<f64>,
        tpot_p95_after: Option<f64>,
        tpot_floor_ms: Option<f64>,
        prefill_latency_floor_ms: Option<f64>,
    ) -> Delta {
        Delta {
            throughput_delta_pct: throughput_pct,
            throughput_before: None,
            throughput_after: None,
            efficiency_delta_pp: None,
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
            ttft_p99_before_ms: None,
            ttft_p99_after_ms: None,
            tpot_p99_before_ms: None,
            tpot_p99_after_ms: None,
            ttft_p95_before_ms: ttft_p95_before,
            ttft_p95_after_ms: ttft_p95_after,
            tpot_p95_before_ms: tpot_p95_before,
            tpot_p95_after_ms: tpot_p95_after,
            direction: Direction::NoChange,
            direction_reason: None,
            ttft_p99_delta_pct: None,
            ttft_p95_delta_pct: pct_delta(ttft_p95_before, ttft_p95_after),
            tpot_p95_delta_pct: pct_delta(tpot_p95_before, tpot_p95_after),
            tpot_floor_ms,
            prefill_latency_floor_ms,
            config_drifted: false,
        }
    }

    fn finalize_delta(mut d: Delta) -> Delta {
        let (direction, reason) = calculate_direction(&d, None);
        d.direction = direction;
        d.direction_reason = reason;
        d
    }

    fn eval_dir(d: &Delta) -> (Direction, Option<&'static str>) {
        evaluate_direction(d)
    }

    #[test]
    fn all_improved_is_better() {
        let (dir, reason) = eval_dir(&mk_delta(
            Some(15.0),
            Some(100.0),
            Some(80.0),
            Some(50.0),
            Some(40.0),
        ));
        assert_eq!(dir, Direction::Better);
        assert!(reason.is_none());
    }

    #[test]
    fn throughput_up_tpot_degraded_is_mixed() {
        let d = finalize_delta(mk_delta(
            Some(15.0),
            Some(100.0),
            Some(95.0),
            Some(50.0),
            Some(60.0),
        ));
        assert_eq!(d.direction, Direction::Mixed);
        assert!(d
            .direction_reason
            .is_some_and(|r| r.contains("KV cache pressure")));
    }

    #[test]
    fn throughput_up_ttft_degraded_tpot_degraded_is_worse() {
        let d = finalize_delta(mk_delta(
            Some(15.0),
            Some(100.0),
            Some(125.0),
            Some(50.0),
            Some(60.0),
        ));
        assert_eq!(d.direction, Direction::Worse);
        assert!(d
            .direction_reason
            .is_some_and(|r| r.contains("latency broken")));
    }

    #[test]
    fn throughput_down_latency_improved_is_mixed() {
        let d = finalize_delta(mk_delta(
            Some(-15.0),
            Some(100.0),
            Some(80.0),
            Some(50.0),
            Some(40.0),
        ));
        assert_eq!(d.direction, Direction::Mixed);
        assert!(d
            .direction_reason
            .is_some_and(|r| r.contains("load artifact")));
    }

    #[test]
    fn throughput_down_ttft_improved_tpot_flat_is_worse() {
        let d = finalize_delta(mk_delta(
            Some(-15.0),
            Some(100.0),
            Some(80.0),
            Some(50.0),
            Some(50.0),
        ));
        assert_eq!(d.direction, Direction::Worse);
        assert!(d
            .direction_reason
            .is_some_and(|r| r.contains("load artifact")));
    }

    #[test]
    fn flat_tpot_degraded_is_worse() {
        let d = finalize_delta(mk_delta(
            Some(5.0),
            Some(100.0),
            Some(100.0),
            Some(50.0),
            Some(60.0),
        ));
        assert_eq!(d.direction, Direction::Worse);
        assert_eq!(d.direction_reason, Some("TPOT p95 degraded"));
    }

    #[test]
    fn flat_ttft_degraded_tpot_improved_is_mixed() {
        let d = finalize_delta(mk_delta(
            Some(5.0),
            Some(100.0),
            Some(125.0),
            Some(50.0),
            Some(40.0),
        ));
        assert_eq!(d.direction, Direction::Mixed);
        assert!(d
            .direction_reason
            .is_some_and(|r| r.contains("prefill contention")));
    }

    #[test]
    fn no_signal_is_no_change() {
        let d = finalize_delta(mk_delta(None, None, None, None, None));
        assert_eq!(d.direction, Direction::NoChange);
    }

    #[test]
    fn severe_regression_all_degraded() {
        let d = finalize_delta(mk_delta(
            Some(-15.0),
            Some(100.0),
            Some(125.0),
            Some(50.0),
            Some(60.0),
        ));
        assert_eq!(d.direction, Direction::Worse);
        assert!(d
            .direction_reason
            .is_some_and(|r| r.contains("severe regression")));
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
    fn compute_direction_better_on_throughput_not_efficiency() {
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
    fn compute_direction_worse_on_throughput_drop() {
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
    fn compute_direction_no_change_when_throughput_within_flat_band() {
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
        );
        assert_eq!(d.ttft_p95_before_ms, Some(400.0));
        assert_eq!(d.ttft_p95_after_ms, Some(340.0));
        assert_eq!(d.tpot_p95_before_ms, Some(70.0));
        assert_eq!(d.tpot_p95_after_ms, Some(63.0));
        assert!((d.ttft_p95_delta_pct.unwrap() - (-15.0)).abs() < 1e-9);
        assert!((d.tpot_p95_delta_pct.unwrap() - (-10.0)).abs() < 1e-9);
    }

    #[test]
    fn tpot_jitter_suppresses_small_absolute_increase() {
        let d = mk_delta_with_floors(
            Some(0.0),
            Some(100.0),
            Some(100.0),
            Some(10.0),
            Some(12.0),
            Some(8.0),
            None,
        );
        assert_eq!(
            eval_tpot_p95(d.tpot_p95_delta_pct.unwrap(), 2.0, d.tpot_floor_ms),
            Signal::Flat
        );
    }

    #[test]
    fn tpot_jitter_does_not_suppress_large_absolute_increase() {
        let d = mk_delta_with_floors(
            Some(0.0),
            Some(100.0),
            Some(100.0),
            Some(10.0),
            Some(18.0),
            Some(8.0),
            None,
        );
        assert_eq!(
            eval_tpot_p95(d.tpot_p95_delta_pct.unwrap(), 8.0, d.tpot_floor_ms),
            Signal::Degraded
        );
    }

    #[test]
    fn ttft_jitter_absorbs_one_prefill_spike() {
        let d = mk_delta_with_floors(
            Some(0.0),
            Some(50.0),
            Some(110.0),
            Some(50.0),
            Some(50.0),
            None,
            Some(50.0),
        );
        assert_eq!(
            eval_ttft_p95(
                d.ttft_p95_delta_pct.unwrap(),
                60.0,
                d.prefill_latency_floor_ms
            ),
            Signal::Flat
        );
    }

    #[test]
    fn jitter_uses_minimum_floor_when_baseline_missing() {
        assert_eq!(eval_tpot_p95(50.0, 1.0, None), Signal::Flat);
    }
}
