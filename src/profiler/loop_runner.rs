use std::time::Duration;

use super::{DiagnoseResult, MaxNumSeqsPrompt, delta, drift, poll, run_diagnose, state::LoopState};
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine;
use crate::output;

const CEILING_HEADROOM_THRESHOLD_PCT: f64 = 10.0;
/// Minimum |Δpp| to print Decode eff.; below this the line is omitted (noise).
const EFFICIENCY_DISPLAY_MIN_PP: f64 = 0.05;
/// Absolute efficiency drop (pp) required before appending `worse`.
const EFFICIENCY_WORSE_MIN_PP: f64 = 1.0;
/// Relative throughput drop (%) required before appending `worse`.
const THROUGHPUT_WORSE_MIN_PCT: f64 = 5.0;
/// Absolute USD rise required before Cost/1M appends `worse`.
const COST_WORSE_THRESHOLD_USD: f64 = 0.01;
/// Absolute J/tok rise required before the energy line appends `worse`.
const JTOK_WORSE_THRESHOLD: f64 = 0.02;
/// TTFT avg materiality gate (ms).
const TTFT_WORSE_MIN_MS: f64 = 5.0;
/// TPOT avg materiality gate (ms).
const TPOT_WORSE_MIN_MS: f64 = 0.5;
const EFFICIENCY_PLATEAU_DELTA: f64 = 2.0;
const PLATEAU_CONSECUTIVE_ITERS: u32 = 3;

fn worse_suffix(material_regression: bool) -> &'static str {
    if material_regression { "  worse" } else { "" }
}

/// Inputs for the interactive diagnose closed loop.
pub struct LoopRunnerInput<'a> {
    pub url: &'a str,
    pub max_num_seqs: u32,
    pub cost_per_hour: Option<f64>,
    pub tensor_parallel_size: u32,
    pub gpu_indices: Vec<u32>,
    pub duration: Duration,
    pub initial_result: DiagnoseResult,
    pub initial_report: engine::Report,
    pub verbose_rules: bool,
    pub max_num_seqs_prompt: &'a mut dyn MaxNumSeqsPrompt,
}

pub fn run(input: LoopRunnerInput<'_>) -> anyhow::Result<()> {
    let LoopRunnerInput {
        url,
        max_num_seqs,
        cost_per_hour,
        tensor_parallel_size,
        gpu_indices,
        duration,
        initial_result,
        initial_report,
        verbose_rules,
        max_num_seqs_prompt,
    } = input;

    let mut last_fingerprint = initial_result.snapshot.fingerprint();
    let mut state = LoopState::new(initial_result, initial_report);
    let stdin_rx = poll::spawn_stdin_watcher();
    let mut current_max_num_seqs = max_num_seqs;

    loop {
        let (rule_name, prev_result, prev_report) = {
            let Some(last_state) = state.last() else {
                break;
            };
            let Some(rule_name) = last_state
                .report
                .recommendations
                .first()
                .map(|r| r.rule_name)
            else {
                let baseline = last_state.report.baseline.as_ref();
                let efficiency = baseline.and_then(|b| b.efficiency_pct);
                let enforce_eager = last_state.result.static_ctx.config.enforce_eager;
                let enable_prefix_caching =
                    last_state.result.static_ctx.config.enable_prefix_caching;
                let quantization = last_state.result.static_ctx.config.quantization.clone();
                let msg = healthy_exit_message(HealthyExitInput {
                    efficiency,
                    limiter_evidence: last_state.report.limiter_evidence.unwrap_or_default(),
                    n_eval: last_state.report.n_eval,
                    enforce_eager,
                    enable_prefix_caching,
                    quantization,
                });
                println!("\n{msg}");
                break;
            };
            (
                rule_name,
                last_state.result.clone(),
                last_state.report.clone(),
            )
        };

        if state.is_oscillating() {
            match state.oscillating_pair() {
                Some((a, b))
                    if (a == "kv_cache_pressure" && b == "concurrency_saturation")
                        || (a == "concurrency_saturation" && b == "kv_cache_pressure") =>
                {
                    if state.midpoint_suggested() {
                        println!(
                            "\nNo --max-num-seqs value resolves both KV pressure and queue saturation."
                        );
                        println!("Add a replica to scale out.");
                        break;
                    }

                    let lo = state
                        .history()
                        .iter()
                        .rev()
                        .find(|r| r.recommendation_shown == Some("kv_cache_pressure"))
                        .and_then(|r| r.result.static_ctx.config.max_num_seqs);
                    let hi = state
                        .history()
                        .iter()
                        .rev()
                        .find(|r| r.recommendation_shown == Some("concurrency_saturation"))
                        .and_then(|r| r.result.static_ctx.config.max_num_seqs);

                    match (lo, hi) {
                        (Some(lo), Some(hi)) if hi > lo + 2 => {
                            let mid = (lo + hi) / 2;
                            println!(
                                "\n--max-num-seqs={hi} filled KV cache. --max-num-seqs={lo} saturated the queue."
                            );
                            println!("Try --max-num-seqs={mid}.");
                            state.set_midpoint_suggested();
                        }
                        _ => {
                            println!(
                                "\nNo --max-num-seqs value resolves both KV pressure and queue saturation."
                            );
                            println!("Add a replica to scale out.");
                            break;
                        }
                    }
                }
                _ => {}
            }
        }

        if state.iteration_count() >= super::state::MAX_LOOP_ITERATIONS {
            println!(
                "\nNo further improvement found after {} iterations. Stopping.",
                super::state::MAX_LOOP_ITERATIONS
            );
            break;
        }

        state.record_recommendation(rule_name);

        let _outcome = poll::wait_for_restart_or_skip(url, &stdin_rx);

        println!();
        current_max_num_seqs = max_num_seqs_prompt.ask(current_max_num_seqs, &stdin_rx)?;

        println!("\nMeasuring delta...\n");
        let new_result = run_diagnose(
            url,
            Some(current_max_num_seqs),
            cost_per_hour,
            tensor_parallel_size,
            gpu_indices.clone(),
            duration,
        )?;
        let agg_win = RuntimeWindow::from_snapshot(new_result.snapshot.clone());
        let summary = AnalysisInput::new(&new_result.static_ctx, &agg_win);
        let new_report = engine::build_report_for_diagnose(&new_result.windows, summary);
        if let Some(msg) = mid_loop_abort_message(
            new_result.any_evaluable,
            new_result.all_idle,
            new_report.n_eval,
            new_report.skipped_broken,
            new_report.skipped_idle,
        ) {
            println!("{msg}");
            break;
        }
        last_fingerprint = verified_pass_fingerprint(&last_fingerprint, &new_result.snapshot)?;

        let drifted = drift::config_changed(&prev_result.static_ctx, &new_result.static_ctx);
        let non_baseline =
            drift::non_baseline_drifted(&prev_result.static_ctx, &new_result.static_ctx);

        let d = delta::compute(
            &prev_result,
            &prev_report,
            &new_result,
            &new_report,
            drifted,
            non_baseline,
        );
        let current_eff = new_report.baseline.as_ref().and_then(|b| b.efficiency_pct);
        let plateau_count = state.update_efficiency_plateau(current_eff, EFFICIENCY_PLATEAU_DELTA);
        print_delta(&d, Some(rule_name));
        println!();
        output::stdout::print_diagnose_table_with_report(
            &new_result,
            &new_report,
            &agg_win,
            verbose_rules,
        );

        let headroom = new_report.baseline.as_ref().and_then(|b| b.headroom_pct);
        if at_hardware_ceiling(headroom) {
            println!(
                "\nHardware ceiling reached. Headroom < {CEILING_HEADROOM_THRESHOLD_PCT:.0}%: further gains require scaling hardware."
            );
            break;
        }
        if plateau_count >= PLATEAU_CONSECUTIVE_ITERS {
            let eff_display = current_eff
                .filter(|e| e.is_finite())
                .map(|e| format!("{e:.1}%"))
                .unwrap_or_else(|| "unknown".to_string());
            println!(
                "\nEfficiency plateaued at {eff_display} over {PLATEAU_CONSECUTIVE_ITERS} iterations."
            );
            println!("No further improvement from current config.");
            println!("Either the workload has hit the hardware ceiling, or");
            println!("a bottleneck exists that profile cannot yet identify.");
            break;
        }

        state.push(new_result, new_report, Some(rule_name));
    }

    Ok(())
}

/// Mid-loop remesure abort copy. Pure so the gate is unit-testable.
pub(crate) fn mid_loop_abort_message(
    any_evaluable: bool,
    all_idle: bool,
    n_eval: usize,
    skipped_broken: usize,
    skipped_idle: usize,
) -> Option<String> {
    let captured = crate::engine::format_captured_windows(n_eval, skipped_broken, skipped_idle);
    if !any_evaluable {
        return Some(format!(
            "\nProfile could not extract metrics mid-loop. {captured} Is the server still running?"
        ));
    }
    if all_idle {
        return Some(format!(
            "\nServer went idle mid-loop. {captured} Send continuous traffic."
        ));
    }
    if n_eval < crate::engine::ENGINE_MIN_PERSISTENT_WINDOWS {
        return Some(format!(
            "\nInsufficient data to verify. Required: {} windows. {captured}",
            crate::engine::ENGINE_MIN_PERSISTENT_WINDOWS
        ));
    }
    None
}

fn at_hardware_ceiling(headroom_pct: Option<f64>) -> bool {
    headroom_pct.is_some_and(|h| h < CEILING_HEADROOM_THRESHOLD_PCT)
}

fn capacity_levers(enable_prefix_caching: Option<bool>) -> String {
    let mut levers = Vec::new();
    if enable_prefix_caching != Some(true) {
        levers.push("enable prefix caching");
    }
    levers.push("apply KV quantization (FP8)");
    levers.push("add TP to split KV cache");
    format!("Levers: {}", levers.join(", "))
}

fn physics_levers(quantization: &Option<String>) -> String {
    let mut levers = Vec::new();
    if quantization.is_none() {
        levers.push("quantize further (FP16→FP8/AWQ)");
    }
    levers.push("speculative decoding");
    levers.push("scale out with TP");
    format!("Levers: {}", levers.join(", "))
}

fn framework_overhead_levers(enforce_eager: Option<bool>) -> String {
    let mut levers = Vec::new();
    if enforce_eager != Some(true) {
        levers.push("test --enforce-eager");
    }
    levers.push("verify CPU/PCIe bottlenecks");
    levers.push("evaluate SGLang for this workload");
    format!("Levers: {}", levers.join(", "))
}

struct HealthyExitInput {
    efficiency: Option<f64>,
    limiter_evidence: engine::limiter::LimiterEvidence,
    n_eval: usize,
    enforce_eager: Option<bool>,
    enable_prefix_caching: Option<bool>,
    quantization: Option<String>,
}

fn healthy_exit_message(input: HealthyExitInput) -> String {
    let HealthyExitInput {
        efficiency,
        limiter_evidence,
        n_eval,
        enforce_eager,
        enable_prefix_caching,
        quantization,
    } = input;
    let limiter_line = if n_eval > 0 {
        engine::limiter::limiter_line(&limiter_evidence)
    } else {
        None
    };
    let limiter = engine::limiter::identify(&limiter_evidence);

    let eff_str = efficiency
        .map(|e| format!("Efficiency: {e:.1}%"))
        .unwrap_or_else(|| "Efficiency: unavailable".to_string());

    let limiter_block = match limiter {
        Some(engine::limiter::LimiterVerdict::Known(engine::limiter::PrimaryLimiter::Capacity)) => {
            let levers = capacity_levers(enable_prefix_caching);
            format!(
                "Primary Limiter: KV Cache Capacity\n\
                 {eff_str}\n\
                 {}\n\
                 {levers}",
                limiter_line.as_deref().unwrap_or_default()
            )
        }
        Some(engine::limiter::LimiterVerdict::Known(engine::limiter::PrimaryLimiter::Physics)) => {
            let levers = physics_levers(&quantization);
            format!(
                "Primary Limiter: Physics (Hardware Ceiling)\n\
                 {eff_str}\n\
                 {}\n\
                 {levers}",
                limiter_line.as_deref().unwrap_or_default()
            )
        }
        Some(engine::limiter::LimiterVerdict::Known(
            engine::limiter::PrimaryLimiter::PrefillInterference,
        )) => {
            format!(
                "Primary Limiter: Prefill Interference\n\
                 {eff_str}\n\
                 {}\n\
                 Levers: disaggregate prefill/decode onto separate workers, or tune chunk size.",
                limiter_line.as_deref().unwrap_or_default()
            )
        }
        Some(engine::limiter::LimiterVerdict::Known(
            engine::limiter::PrimaryLimiter::FrameworkOverhead,
        )) => {
            let levers = framework_overhead_levers(enforce_eager);
            format!(
                "Primary Limiter: Framework Overhead\n\
                 {eff_str}\n\
                 {}\n\
                 {levers}",
                limiter_line.as_deref().unwrap_or_default()
            )
        }
        Some(engine::limiter::LimiterVerdict::Known(engine::limiter::PrimaryLimiter::Traffic)) => {
            format!(
                "Primary Limiter: Traffic\n\
                 {eff_str}\n\
                 {}",
                limiter_line.as_deref().unwrap_or_default()
            )
        }
        Some(engine::limiter::LimiterVerdict::CeilingUnknown(_)) => limiter_line
            .unwrap_or_else(|| "Hardware ceiling unknown (GPU not in catalog).".to_string()),
        None => {
            format!("{eff_str} - insufficient data to identify primary limiter.")
        }
    };

    let prefix = match limiter {
        Some(_) => "Rules clear. No actionable config fix identified.",
        None => "Rules clear.",
    };
    format!("{prefix}\n\n{limiter_block}")
}

fn format_efficiency_delta_line(delta_pp: Option<f64>) -> Option<String> {
    match delta_pp {
        Some(v) if v >= EFFICIENCY_DISPLAY_MIN_PP => Some(format!("  Decode eff. +{v:.1}pp")),
        Some(v) if v <= -EFFICIENCY_DISPLAY_MIN_PP => Some(format!(
            "  Decode eff. {v:.1}pp{}",
            worse_suffix(v <= -EFFICIENCY_WORSE_MIN_PP)
        )),
        _ => None,
    }
}

fn format_throughput_delta_line(before: f64, after: f64) -> String {
    let drop_pct = if before > 0.0 {
        (before - after) / before * 100.0
    } else {
        0.0
    };
    let worse = after < before && drop_pct >= THROUGHPUT_WORSE_MIN_PCT;
    format!(
        "  Throughput  {before:.0} → {after:.0} tok/s{}",
        worse_suffix(worse)
    )
}

fn format_ttft_delta_line(
    before: f64,
    after: f64,
    p95_before: Option<f64>,
    p95_after: Option<f64>,
) -> Option<String> {
    let delta = after - before;
    if delta.abs() <= TTFT_WORSE_MIN_MS {
        return None;
    }
    let p95_suffix = match (p95_before, p95_after) {
        (Some(pb), Some(pa)) if pb.is_finite() && pa.is_finite() => {
            format!(" (p95 {pb:.0} → {pa:.0}ms)")
        }
        _ => String::new(),
    };
    Some(format!(
        "  TTFT        {before:.0} → {after:.0}ms{p95_suffix}{}",
        worse_suffix(delta > TTFT_WORSE_MIN_MS)
    ))
}

fn format_tpot_delta_line(
    before: f64,
    after: f64,
    p95_before: Option<f64>,
    p95_after: Option<f64>,
) -> Option<String> {
    let delta = after - before;
    if delta.abs() <= TPOT_WORSE_MIN_MS {
        return None;
    }
    let p95_suffix = match (p95_before, p95_after) {
        (Some(pb), Some(pa)) if pb.is_finite() && pa.is_finite() => {
            format!(" (p95 {pb:.1} → {pa:.1}ms)")
        }
        _ => String::new(),
    };
    Some(format!(
        "  TPOT        {before:.1} → {after:.1}ms{p95_suffix}{}",
        worse_suffix(delta > TPOT_WORSE_MIN_MS)
    ))
}

fn format_jtok_delta_line(before: f64, after: f64) -> String {
    let worse = after - before > JTOK_WORSE_THRESHOLD;
    format!(
        "  J/tok         {before:.2} → {after:.2}{}",
        worse_suffix(worse)
    )
}

fn format_cost_delta_line(before: f64, after: f64, est_suffix: &str) -> String {
    let worse = after - before > COST_WORSE_THRESHOLD_USD;
    format!(
        "  Cost/1M tok   ${before:.2} → ${after:.2}{est_suffix}{}",
        worse_suffix(worse)
    )
}

/// One status line for the remesure delta header.
///
/// Precedence (first match wins): baseline drift → non-baseline drift → load
/// witness (R1 primary only) → prefix-hit witness (R3 primary only) → no change.
///
/// R6 (and any other primary whose fix has no dedicated witness) intentionally
/// falls through to "No change detected." when config did not drift — do not
/// invent attribution from metric moves for those rules.
fn config_status_lines(
    config_drifted: bool,
    non_baseline_drifted: bool,
    load_changed: bool,
    prefix_hit_changed: bool,
    prev_primary: Option<&str>,
) -> Vec<String> {
    let line = if config_drifted {
        "  Config changed. Baseline reset."
    } else if non_baseline_drifted {
        "  Config changed."
    } else if prev_primary == Some(engine::rule_names::UNDER_BATCHING) && load_changed {
        "  Load changed."
    } else if prev_primary == Some(engine::rule_names::LOW_PREFIX_REUSE) && prefix_hit_changed {
        "  Prefix cache hit rate changed."
    } else {
        "  No change detected."
    };
    vec![line.to_string(), String::new()]
}

fn print_delta(d: &delta::Delta, prev_primary: Option<&str>) {
    for line in config_status_lines(
        d.config_drifted,
        d.non_baseline_drifted,
        d.load_changed,
        d.prefix_hit_changed,
        prev_primary,
    ) {
        println!("{line}");
    }
    if let (Some(before), Some(after)) = (d.throughput_before, d.throughput_after)
        && before.is_finite()
        && after.is_finite()
    {
        println!("{}", format_throughput_delta_line(before, after));
    }
    if let (Some(before), Some(after)) = (d.ttft_before_ms, d.ttft_after_ms)
        && before.is_finite()
        && after.is_finite()
        && let Some(line) =
            format_ttft_delta_line(before, after, d.ttft_p95_before_ms, d.ttft_p95_after_ms)
    {
        println!("{line}");
    }
    if let (Some(before), Some(after)) = (d.tpot_before_ms, d.tpot_after_ms)
        && before.is_finite()
        && after.is_finite()
        && let Some(line) =
            format_tpot_delta_line(before, after, d.tpot_p95_before_ms, d.tpot_p95_after_ms)
    {
        println!("{line}");
    }
    if let Some(line) = format_efficiency_delta_line(d.efficiency_delta_pp) {
        println!("{line}");
    }
    if let Some((x, y)) = d.capacity_self_grade {
        let y_disp = if (y - y.round()).abs() < 1e-9 {
            format!("{y:.0}")
        } else {
            format!("{y:.2}")
        };
        println!("  Capacity: prescribed ≤{x}, vLLM now reports {y_disp}.");
    }
    if economics_section_active(d) {
        println!();
        println!("ECONOMICS:");
    }
    match (d.joules_per_token_before, d.joules_per_token_after) {
        (Some(before), Some(after)) if before.is_finite() && after.is_finite() => {
            println!("{}", format_jtok_delta_line(before, after));
        }
        _ => {}
    }
    match (d.cost_per_million_before, d.cost_per_million_after) {
        (Some(before), Some(after)) if before.is_finite() && after.is_finite() => {
            let est = match d.cost_source_after {
                Some(engine::CostSource::Catalog) | None => " (est)",
                _ => "",
            };
            println!("{}", format_cost_delta_line(before, after, est));
        }
        _ => {}
    }
}

fn economics_section_active(d: &delta::Delta) -> bool {
    (d.cost_per_million_before.is_some() && d.cost_per_million_after.is_some())
        || (d.joules_per_token_before.is_some() && d.joules_per_token_after.is_some())
}

/// Closed-loop pass gate: fingerprint must match the prior iteration.
fn verified_pass_fingerprint(
    last: &crate::collectors::GpuFingerprint,
    snapshot: &crate::collectors::RawSnapshot,
) -> anyhow::Result<crate::collectors::GpuFingerprint> {
    let current = snapshot.fingerprint();
    crate::collectors::types::check_topology_drift(last, &current)?;
    Ok(current)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;
    use crate::collectors::test_fixtures::snap_with_gpu_indices;

    #[test]
    fn at_hardware_ceiling_below_threshold() {
        assert!(at_hardware_ceiling(Some(9.0)));
    }

    #[test]
    fn at_hardware_ceiling_at_threshold_not_reached() {
        assert!(!at_hardware_ceiling(Some(10.0)));
    }

    #[test]
    fn at_hardware_ceiling_none_not_reached() {
        assert!(!at_hardware_ceiling(None));
    }

    fn framework_overhead_input(enforce_eager: Option<bool>) -> HealthyExitInput {
        HealthyExitInput {
            efficiency: Some(60.0),
            limiter_evidence: engine::limiter::LimiterEvidence {
                kv_cache_peak_perc: Some(50.0),
                mean_running: Some(50.0),
                ridge_batch_size: Some(40.0),
                mean_tpot_ms: Some(50.0),
                tpot_floor_ms: Some(10.0),
                effective_prompt_decode_ratio: None,
                chunked_prefill_enabled: Some(false),
            },
            n_eval: 1,
            enforce_eager,
            enable_prefix_caching: None,
            quantization: None,
        }
    }

    fn capacity_input(enable_prefix_caching: Option<bool>) -> HealthyExitInput {
        HealthyExitInput {
            efficiency: Some(42.5),
            limiter_evidence: engine::limiter::LimiterEvidence {
                kv_cache_peak_perc: Some(85.0),
                mean_running: Some(50.0),
                ridge_batch_size: Some(40.0),
                mean_tpot_ms: Some(20.0),
                tpot_floor_ms: Some(5.0),
                effective_prompt_decode_ratio: None,
                chunked_prefill_enabled: Some(false),
            },
            n_eval: 1,
            enforce_eager: None,
            enable_prefix_caching,
            quantization: None,
        }
    }

    fn physics_input(quantization: Option<String>) -> HealthyExitInput {
        HealthyExitInput {
            efficiency: Some(91.0),
            limiter_evidence: engine::limiter::LimiterEvidence {
                kv_cache_peak_perc: Some(50.0),
                mean_running: Some(50.0),
                ridge_batch_size: Some(40.0),
                mean_tpot_ms: Some(11.0),
                tpot_floor_ms: Some(10.0),
                effective_prompt_decode_ratio: None,
                chunked_prefill_enabled: Some(false),
            },
            n_eval: 1,
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization,
        }
    }

    #[test]
    fn healthy_exit_capacity_limiter() {
        let msg = healthy_exit_message(capacity_input(None));
        assert!(msg.contains("KV Cache Capacity"));
        assert!(msg.contains("Efficiency: 42.5%"));
        assert!(msg.contains("KV cache at 85%"));
        assert!(msg.contains("No actionable config fix identified."));
    }

    #[test]
    fn healthy_exit_traffic_limiter() {
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: Some(34.0),
            limiter_evidence: engine::limiter::LimiterEvidence {
                kv_cache_peak_perc: Some(50.0),
                mean_running: Some(5.0),
                ridge_batch_size: Some(100.0),
                mean_tpot_ms: Some(20.0),
                tpot_floor_ms: Some(5.0),
                effective_prompt_decode_ratio: None,
                chunked_prefill_enabled: Some(false),
            },
            n_eval: 1,
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(msg.contains("Primary Limiter: Traffic"));
        assert!(msg.contains(
            "Capped by traffic: 5 requests running, hardware has room for ~100. More concurrent requests raises throughput."
        ));
        assert!(!msg.contains("--max-num-seqs"));
    }

    #[test]
    fn healthy_exit_traffic_with_queue_does_not_claim_traffic_gap() {
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: Some(12.4),
            limiter_evidence: engine::limiter::LimiterEvidence {
                kv_cache_peak_perc: Some(10.0),
                mean_running: Some(10.0),
                ridge_batch_size: Some(100.0),
                mean_tpot_ms: Some(20.0),
                tpot_floor_ms: Some(5.0),
                effective_prompt_decode_ratio: None,
                chunked_prefill_enabled: Some(false),
            },
            n_eval: 1,
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(msg.contains("Primary Limiter: Traffic"));
        assert!(msg.contains("Capped by traffic: 10 requests running"));
        assert!(!msg.contains("--max-num-seqs"));
    }

    #[test]
    fn traffic_limiter_does_not_say_optimally_tuned() {
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: Some(34.0),
            limiter_evidence: engine::limiter::LimiterEvidence {
                kv_cache_peak_perc: Some(50.0),
                mean_running: Some(5.0),
                ridge_batch_size: Some(100.0),
                mean_tpot_ms: Some(20.0),
                tpot_floor_ms: Some(5.0),
                effective_prompt_decode_ratio: None,
                chunked_prefill_enabled: Some(false),
            },
            n_eval: 1,
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(msg.contains("Capped by traffic:"));
        assert!(!msg.contains("--max-num-seqs"));
    }

    #[test]
    fn healthy_exit_physics_limiter() {
        let msg = healthy_exit_message(physics_input(None));
        assert!(msg.contains("Physics (Hardware Ceiling)"));
        assert!(msg.contains("Efficiency: 91.0%"));
        assert!(msg.contains("Capped by hardware: TPOT 11.0ms vs ~10.0ms floor."));
    }

    #[test]
    fn healthy_exit_prefill_interference_limiter() {
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: Some(55.0),
            limiter_evidence: engine::limiter::LimiterEvidence {
                kv_cache_peak_perc: Some(50.0),
                mean_running: Some(50.0),
                ridge_batch_size: Some(40.0),
                mean_tpot_ms: Some(50.0),
                tpot_floor_ms: Some(10.0),
                effective_prompt_decode_ratio: Some(0.6),
                chunked_prefill_enabled: Some(true),
            },
            n_eval: 1,
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(msg.contains("Prefill Interference"));
        assert!(msg.contains("Efficiency: 55.0%"));
        assert!(msg.contains("Capped by prefill: prompt work at 0.6x of decode (effective)."));
    }

    #[test]
    fn healthy_exit_framework_overhead_limiter() {
        let msg = healthy_exit_message(framework_overhead_input(None));
        assert!(msg.contains("Framework Overhead"));
        assert!(msg.contains("Efficiency: 60.0%"));
    }

    #[test]
    fn healthy_exit_reuses_same_limiter_line_as_quiet_report() {
        let ev = engine::limiter::LimiterEvidence {
            kv_cache_peak_perc: Some(84.0),
            mean_running: Some(50.0),
            ridge_batch_size: Some(153.0),
            mean_tpot_ms: Some(20.0),
            tpot_floor_ms: Some(10.0),
            effective_prompt_decode_ratio: Some(0.2),
            chunked_prefill_enabled: Some(false),
        };
        let line = engine::limiter::limiter_line(&ev).expect("limiter line");
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: Some(42.0),
            limiter_evidence: ev,
            n_eval: 3,
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(msg.contains(&line));
    }

    #[test]
    fn framework_overhead_levers_hide_enforce_eager_when_on() {
        let msg = healthy_exit_message(framework_overhead_input(Some(true)));
        assert!(msg.contains("Framework Overhead"));
        assert!(!msg.contains("--enforce-eager"));
        assert!(msg.contains("verify CPU/PCIe bottlenecks"));
        assert!(msg.contains("evaluate SGLang for this workload"));
    }

    #[test]
    fn framework_overhead_levers_show_enforce_eager_when_off_or_unknown() {
        for enforce_eager in [Some(false), None] {
            let msg = healthy_exit_message(framework_overhead_input(enforce_eager));
            assert!(
                msg.contains("--enforce-eager"),
                "expected enforce-eager lever for {enforce_eager:?}"
            );
        }
    }

    #[test]
    fn healthy_exit_insufficient_data_when_limiter_unknown() {
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: None,
            limiter_evidence: engine::limiter::LimiterEvidence::default(),
            n_eval: 1,
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(msg.contains("insufficient data to identify primary limiter"));
        assert!(msg.contains("Efficiency: unavailable"));
        assert!(!msg.contains("No actionable config fix identified."));
        assert_eq!(msg.matches("Rules clear.").count(), 1);
    }

    #[test]
    fn capacity_levers_hide_prefix_caching_when_on() {
        let msg = healthy_exit_message(capacity_input(Some(true)));
        assert!(msg.contains("KV Cache Capacity"));
        assert!(!msg.contains("enable prefix caching"));
        assert!(msg.contains("KV quantization"));
        assert!(msg.contains("add TP"));
    }

    #[test]
    fn capacity_levers_show_prefix_caching_when_off() {
        let msg = healthy_exit_message(capacity_input(None));
        assert!(msg.contains("enable prefix caching"));
    }

    #[test]
    fn physics_levers_hide_quantize_when_already_quantized() {
        let msg = healthy_exit_message(physics_input(Some("awq".to_string())));
        assert!(msg.contains("Physics (Hardware Ceiling)"));
        assert!(!msg.contains("quantize further"));
        assert!(msg.contains("speculative decoding"));
        assert!(msg.contains("scale out"));
    }

    #[test]
    fn physics_levers_show_quantize_when_unquantized() {
        let msg = healthy_exit_message(physics_input(None));
        assert!(msg.contains("quantize further"));
    }

    #[test]
    fn non_baseline_max_num_seqs_not_baseline_drift() {
        use crate::collectors::VllmConfig;
        use crate::context::StaticContext;
        let prev_cfg = VllmConfig {
            max_num_seqs: Some(32),
            ..Default::default()
        };
        let curr_cfg = VllmConfig {
            max_num_seqs: Some(98),
            ..Default::default()
        };
        let prev = StaticContext {
            config: prev_cfg,
            ..StaticContext::default()
        };
        let curr = StaticContext {
            config: curr_cfg,
            ..StaticContext::default()
        };
        assert!(!drift::config_changed(&prev, &curr));
        assert!(drift::non_baseline_drifted(&prev, &curr));
    }

    #[test]
    fn config_status_lines_baseline_beats_non_baseline() {
        let lines = config_status_lines(
            true,
            true,
            true,
            true,
            Some(engine::rule_names::UNDER_BATCHING),
        );
        assert_eq!(lines[0], "  Config changed. Baseline reset.");
    }

    #[test]
    fn config_status_lines_non_baseline_beats_witness() {
        let lines = config_status_lines(
            false,
            true,
            true,
            true,
            Some(engine::rule_names::UNDER_BATCHING),
        );
        assert_eq!(lines[0], "  Config changed.");
    }

    #[test]
    fn config_status_lines_r1_load_witness() {
        let lines = config_status_lines(
            false,
            false,
            true,
            false,
            Some(engine::rule_names::UNDER_BATCHING),
        );
        assert_eq!(lines[0], "  Load changed.");
    }

    #[test]
    fn config_status_lines_r1_without_load_is_no_change() {
        let lines = config_status_lines(
            false,
            false,
            false,
            true,
            Some(engine::rule_names::UNDER_BATCHING),
        );
        assert_eq!(lines[0], "  No change detected.");
    }

    #[test]
    fn config_status_lines_r3_prefix_witness() {
        let lines = config_status_lines(
            false,
            false,
            false,
            true,
            Some(engine::rule_names::LOW_PREFIX_REUSE),
        );
        assert_eq!(lines[0], "  Prefix cache hit rate changed.");
    }

    #[test]
    fn config_status_lines_non_r1_r3_load_move_is_no_change() {
        let lines = config_status_lines(
            false,
            false,
            true,
            false,
            Some(engine::rule_names::PREFILL_BOUND),
        );
        assert_eq!(lines[0], "  No change detected.");
    }

    #[test]
    fn config_status_lines_no_changes() {
        let lines = config_status_lines(false, false, false, false, None);
        assert_eq!(lines[0], "  No change detected.");
    }

    #[test]
    fn efficiency_delta_near_zero_suppressed() {
        assert!(format_efficiency_delta_line(Some(-0.04)).is_none());
        assert!(format_efficiency_delta_line(Some(0.03)).is_none());
    }

    #[test]
    fn efficiency_delta_small_drop_prints_without_worse() {
        let line = format_efficiency_delta_line(Some(-0.06)).expect("line");
        assert!(!line.contains("worse"));
        assert!(line.contains("-0.1pp"));
        assert!(!line.contains('↓') && !line.contains('↑'));
    }

    #[test]
    fn worse_appears_on_material_regression() {
        let ttft = format_ttft_delta_line(98.0, 5793.0, Some(228.0), Some(10556.0)).unwrap();
        assert!(ttft.ends_with("  worse"));
        let tpot = format_tpot_delta_line(26.3, 47.8, Some(48.8), Some(85.0)).unwrap();
        assert!(tpot.ends_with("  worse"));
        assert!(format_throughput_delta_line(1000.0, 900.0).ends_with("  worse"));
        assert!(format_jtok_delta_line(0.70, 0.80).ends_with("  worse"));
        assert!(format_cost_delta_line(0.92, 1.00, " (est)").ends_with("  worse"));
        let eff = format_efficiency_delta_line(Some(-4.2)).unwrap();
        assert!(eff.ends_with("  worse"));
    }

    #[test]
    fn worse_absent_on_improvement() {
        assert!(!format_throughput_delta_line(230.0, 902.0).contains("worse"));
        let ttft = format_ttft_delta_line(5793.0, 98.0, Some(10556.0), Some(228.0)).unwrap();
        assert!(!ttft.contains("worse"));
        let tpot = format_tpot_delta_line(47.8, 26.3, Some(85.0), Some(48.8)).unwrap();
        assert!(!tpot.contains("worse"));
        assert!(!format_jtok_delta_line(2.23, 0.70).contains("worse"));
        assert!(!format_cost_delta_line(3.60, 0.92, " (est)").contains("worse"));
        let eff = format_efficiency_delta_line(Some(4.2)).unwrap();
        assert!(!eff.contains("worse"));
        assert_eq!(eff, "  Decode eff. +4.2pp");
    }

    #[test]
    fn worse_absent_below_gate() {
        // Throughput: 4% drop < 5% gate → silent.
        assert!(!format_throughput_delta_line(1000.0, 960.0).contains("worse"));
        // TTFT/TPOT: below materiality → no line.
        assert!(format_ttft_delta_line(100.0, 104.0, None, None).is_none());
        assert!(format_tpot_delta_line(26.0, 26.4, None, None).is_none());
        // Cost/Jtok: rise below absolute gate → silent.
        assert!(!format_cost_delta_line(2.00, 2.005, " (est)").contains("worse"));
        assert!(!format_jtok_delta_line(0.31, 0.32).contains("worse"));
        // Eff below display min → no line.
        assert!(format_efficiency_delta_line(Some(-0.04)).is_none());
        // Eff displayed but below 1.0pp worse gate → silent.
        let small = format_efficiency_delta_line(Some(-0.5)).unwrap();
        assert!(!small.contains("worse"));
        assert!(
            format_efficiency_delta_line(Some(-1.5))
                .unwrap()
                .ends_with("  worse")
        );
    }

    #[test]
    fn p95_has_no_own_label() {
        let line = format_ttft_delta_line(98.0, 5793.0, Some(228.0), Some(10556.0)).unwrap();
        assert!(line.contains("(p95 228 → 10556ms)"));
        assert_eq!(line.matches("worse").count(), 1);
        assert!(line.ends_with("  worse"));
        let start = line.find("(p95").unwrap();
        let end = line.find(')').unwrap();
        let p95 = &line[start..=end];
        assert!(p95.ends_with("ms)"));
        assert!(!p95.contains("worse"));
        assert!(!p95.contains('↑') && !p95.contains('↓'));
        assert!(!p95.contains("improved") && !p95.contains("regressed"));
    }

    #[test]
    fn economics_header_shown_when_only_jtok_available() {
        let d = delta::Delta {
            throughput_before: None,
            throughput_after: None,
            efficiency_delta_pp: None,
            efficiency_pct_before: None,
            efficiency_pct_after: None,
            cost_per_million_before: None,
            cost_per_million_after: None,
            joules_per_token_before: Some(0.31),
            joules_per_token_after: Some(0.28),
            cost_source_after: None,
            ttft_before_ms: None,
            ttft_after_ms: None,
            tpot_before_ms: None,
            tpot_after_ms: None,
            ttft_p95_before_ms: None,
            ttft_p95_after_ms: None,
            tpot_p95_before_ms: None,
            tpot_p95_after_ms: None,
            config_drifted: false,
            non_baseline_drifted: false,
            load_changed: false,
            prefix_hit_changed: false,
            capacity_self_grade: None,
        };
        assert!(economics_section_active(&d));
    }

    #[test]
    fn verified_pass_fingerprint_allows_unchanged_topology() {
        let last = snap_with_gpu_indices(&[0, 1]).fingerprint();
        let next = snap_with_gpu_indices(&[0, 1]);
        assert!(verified_pass_fingerprint(&last, &next).is_ok());
    }

    #[test]
    fn verified_pass_fingerprint_errors_on_topology_drift() {
        let last = snap_with_gpu_indices(&[0, 1]).fingerprint();
        let drifted = snap_with_gpu_indices(&[0, 2]);
        let err = verified_pass_fingerprint(&last, &drifted).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Topology drift detected"));
        assert!(msg.contains("missing [GPU-1]"));
        assert!(msg.contains("new [GPU-2]"));
    }

    #[test]
    fn mid_loop_abort_message_tests() {
        let crash = mid_loop_abort_message(false, false, 0, 15, 0).unwrap();
        assert!(crash.contains("could not extract metrics mid-loop"));
        assert!(crash.contains("Captured: 0 (15 dropped)."));
        let idle_msg = mid_loop_abort_message(true, true, 0, 0, 3).unwrap();
        assert!(idle_msg.contains("Server went idle mid-loop. Captured: 0 (3 idle)."));
        let sparse_msg = mid_loop_abort_message(true, false, 2, 3, 10).unwrap();
        assert!(sparse_msg.contains("Insufficient data to verify. Required: 3 windows."));
        assert!(sparse_msg.contains("Captured: 2 (3 dropped, 10 idle)."));
        assert_eq!(mid_loop_abort_message(true, false, 3, 0, 0), None);
    }
}
