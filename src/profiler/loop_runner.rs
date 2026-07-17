use std::time::Duration;

use super::{DiagnoseResult, delta, drift, poll, run_diagnose, state::LoopState};
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine::{self, ACHIEVABLE_EFFICIENCY_CEILING};
use crate::output;

const CEILING_HEADROOM_THRESHOLD_PCT: f64 = 10.0;
const EFFICIENCY_DISPLAY_MIN_PP: f64 = 0.05;
const EFFICIENCY_PLATEAU_DELTA: f64 = 2.0;
const PLATEAU_CONSECUTIVE_ITERS: u32 = 3;

#[allow(clippy::too_many_arguments)]
pub fn run(
    url: &str,
    max_num_seqs: u32,
    cost_per_hour: Option<f64>,
    tensor_parallel_size: u32,
    gpu_indices: Vec<u32>,
    duration: Duration,
    initial_result: DiagnoseResult,
    initial_report: engine::Report,
    verbose_rules: bool,
) -> anyhow::Result<()> {
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
                let ridge_batch_size = baseline.map(|b| b.ridge_batch_size);
                let tpot_floor_ms = baseline.map(|b| b.tpot_floor_ms);
                let kv_cache_usage_perc = last_state.result.snapshot.vllm.kv_cache_usage_perc;
                let num_running = last_state.result.snapshot.vllm.num_requests_running;
                let num_waiting = last_state.result.snapshot.vllm.num_requests_waiting;
                let tpot_ms = last_state.result.snapshot.vllm.tpot_ms;
                let chunked_prefill_enabled =
                    last_state.result.static_ctx.config.enable_chunked_prefill;
                let enforce_eager = last_state.result.static_ctx.config.enforce_eager;
                let enable_prefix_caching =
                    last_state.result.static_ctx.config.enable_prefix_caching;
                let quantization = last_state.result.static_ctx.config.quantization.clone();
                let msg = healthy_exit_message(HealthyExitInput {
                    efficiency,
                    kv_cache_usage_perc,
                    num_running,
                    num_waiting,
                    ridge_batch_size,
                    tpot_ms,
                    tpot_floor_ms,
                    chunked_prefill_enabled,
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
                Some((a, b)) => {
                    println!(
                        "\nCycling between [{a}] and [{b}]. No further improvement found. Stopping."
                    );
                    break;
                }
                None => {
                    println!(
                        "\nCycling between recommendations. No further improvement found. Stopping."
                    );
                    break;
                }
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
        current_max_num_seqs =
            crate::cli::prompt_for_updated_max_num_seqs(current_max_num_seqs, &stdin_rx)?;

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
        let config_changes =
            drift::non_baseline_changes(&prev_result.static_ctx, &new_result.static_ctx);

        let d = delta::compute(
            &prev_result,
            &prev_report,
            &new_result,
            &new_report,
            drifted,
            config_changes,
        );
        let current_eff = new_report.baseline.as_ref().and_then(|b| b.efficiency_pct);
        let plateau_count = state.update_efficiency_plateau(current_eff, EFFICIENCY_PLATEAU_DELTA);
        print_delta(&d);
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
    kv_cache_usage_perc: Option<f64>,
    num_running: Option<f64>,
    num_waiting: Option<f64>,
    ridge_batch_size: Option<f64>,
    tpot_ms: Option<f64>,
    tpot_floor_ms: Option<f64>,
    chunked_prefill_enabled: Option<bool>,
    enforce_eager: Option<bool>,
    enable_prefix_caching: Option<bool>,
    quantization: Option<String>,
}

fn healthy_exit_message(input: HealthyExitInput) -> String {
    let HealthyExitInput {
        efficiency,
        kv_cache_usage_perc,
        num_running,
        num_waiting,
        ridge_batch_size,
        tpot_ms,
        tpot_floor_ms,
        chunked_prefill_enabled,
        enforce_eager,
        enable_prefix_caching,
        quantization,
    } = input;
    let limiter = engine::limiter::identify(
        kv_cache_usage_perc,
        num_running,
        ridge_batch_size,
        tpot_ms,
        tpot_floor_ms,
        chunked_prefill_enabled,
    );

    if limiter == Some(engine::limiter::PrimaryLimiter::Traffic) {
        let eff = efficiency
            .filter(|e| e.is_finite())
            .map(|e| format!("Efficiency {e:.1}%"))
            .unwrap_or_else(|| "Efficiency unavailable".to_string());
        let waiting = num_waiting.filter(|w| w.is_finite() && *w >= 0.0);
        if waiting.is_some_and(|w| w >= 1.0) {
            let wait_n = waiting
                .map(|w| format!("{:.0}", w.trunc()))
                .unwrap_or_else(|| "-".to_string());
            return format!(
                "No issues sustained. {eff} with {wait_n} requests waiting - queue present but no rule explains it. Re-run with longer duration; report this pattern if it persists."
            );
        }
        return format!(
            "No issues. {eff} - gap is traffic, not config. Increase load to find real limits."
        );
    }

    let eff_str = efficiency
        .map(|e| format!("Efficiency: {e:.1}%"))
        .unwrap_or_else(|| "Efficiency: unavailable".to_string());

    let limiter_block = match limiter {
        Some(engine::limiter::PrimaryLimiter::Capacity) => {
            let kv = kv_cache_usage_perc
                .map(|k| format!(" KV cache at {k:.0}%."))
                .unwrap_or_default();
            let levers = capacity_levers(enable_prefix_caching);
            format!(
                "Primary Limiter: KV Cache Capacity\n\
                 {eff_str} -{kv} concurrency is capped before bandwidth saturates.\n\
                 {levers}"
            )
        }
        Some(engine::limiter::PrimaryLimiter::Physics) => {
            let floor_str = tpot_floor_ms
                .zip(tpot_ms)
                .map(|(floor, actual)| format!(" TPOT {actual:.1}ms vs floor {floor:.1}ms."))
                .unwrap_or_default();
            let levers = physics_levers(&quantization);
            format!(
                "Primary Limiter: Physics (Hardware Ceiling)\n\
                 {eff_str} -{floor_str} Hardware is at the limits of this model and dtype.\n\
                 {levers}"
            )
        }
        Some(engine::limiter::PrimaryLimiter::PrefillInterference) => {
            format!(
                "Primary Limiter: Prefill Interference\n\
                 {eff_str} - chunked prefill is sharing decode memory bandwidth with prefill GEMMs.\n\
                 Levers: disaggregate prefill/decode onto separate workers, or tune chunk size."
            )
        }
        Some(engine::limiter::PrimaryLimiter::FrameworkOverhead) => {
            let levers = framework_overhead_levers(enforce_eager);
            format!(
                "Primary Limiter: Framework Overhead\n\
                 {eff_str} - batch healthy, VRAM available, but GPU is waiting on the system.\n\
                 {levers}"
            )
        }
        Some(engine::limiter::PrimaryLimiter::Traffic) => {
            unreachable!("traffic limiter returns before limiter_block")
        }
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
        Some(v) if v >= EFFICIENCY_DISPLAY_MIN_PP => Some(format!("  Decode eff. +{v:.1}pp ↑")),
        Some(v) if v <= -EFFICIENCY_DISPLAY_MIN_PP => Some(format!("  Decode eff. {v:.1}pp ↓")),
        _ => None,
    }
}

fn config_status_lines(config_drifted: bool, config_changes: &[String]) -> Vec<String> {
    if config_drifted {
        vec![
            "  Config changed, baseline reset.".to_string(),
            String::new(),
        ]
    } else if !config_changes.is_empty() {
        let mut lines = vec!["  Config changed:".to_string()];
        lines.extend(config_changes.iter().cloned());
        lines.push(String::new());
        lines
    } else {
        vec!["  No config changed.".to_string(), String::new()]
    }
}

fn print_delta(d: &delta::Delta) {
    for line in config_status_lines(d.config_drifted, &d.config_changes) {
        println!("{line}");
    }
    if let (Some(before), Some(after)) = (d.throughput_before, d.throughput_after)
        && before.is_finite()
        && after.is_finite()
    {
        let arrow = throughput_arrow(before, after);
        println!("  Throughput  {before:.0} → {after:.0} tok/s {arrow}");
    }
    if let (Some(before), Some(after)) = (d.ttft_before_ms, d.ttft_after_ms)
        && before.is_finite()
        && after.is_finite()
    {
        let delta = after - before;
        if delta.abs() > 5.0 {
            let arrow = latency_quality(delta);
            let p95_suffix = match (d.ttft_p95_before_ms, d.ttft_p95_after_ms) {
                (Some(pb), Some(pa)) if pb.is_finite() && pa.is_finite() => {
                    format!("  (p95 {pb:.0} → {pa:.0}ms {})", latency_quality(pa - pb))
                }
                _ => String::new(),
            };
            println!("  TTFT        {before:.0} → {after:.0}ms {arrow}{p95_suffix}");
        }
    }
    if let (Some(before), Some(after)) = (d.tpot_before_ms, d.tpot_after_ms)
        && before.is_finite()
        && after.is_finite()
    {
        let delta = after - before;
        if delta.abs() > 0.5 {
            let arrow = latency_quality(delta);
            let p95_suffix = match (d.tpot_p95_before_ms, d.tpot_p95_after_ms) {
                (Some(pb), Some(pa)) if pb.is_finite() && pa.is_finite() => {
                    format!("  (p95 {pb:.1} → {pa:.1}ms {})", latency_quality(pa - pb))
                }
                _ => String::new(),
            };
            println!("  TPOT        {before:.1} → {after:.1}ms {arrow}{p95_suffix}");
        }
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
    let has_cost = economics_section_active(d);
    if has_cost {
        println!();
        println!("ECONOMICS:");
    }
    match (d.joules_per_token_before, d.joules_per_token_after) {
        (Some(before), Some(after)) if before.is_finite() && after.is_finite() => {
            let arrow = jtok_change_arrow(before, after);
            println!("  J/tok         {before:.2} → {after:.2} {arrow}");
        }
        _ => {}
    }
    match (d.cost_per_million_before, d.cost_per_million_after) {
        (Some(before), Some(after)) if before.is_finite() && after.is_finite() => {
            let arrow = cost_change_arrow(before, after);
            let est = match d.cost_source_after {
                Some(engine::CostSource::Catalog) | None => " (est)",
                _ => "",
            };
            println!("  Cost/1M tok   ${before:.2} → ${after:.2} {arrow}{est}");
        }
        _ => {}
    }
    if let (Some(cpm_b), Some(cpm_a), Some(tps_b), Some(tps_a), Some(eff_b), Some(eff_a)) = (
        d.cost_per_million_before,
        d.cost_per_million_after,
        d.throughput_before,
        d.throughput_after,
        d.efficiency_pct_before,
        d.efficiency_pct_after,
    ) && cpm_b.is_finite()
        && cpm_a.is_finite()
        && tps_b > 0.0
        && tps_a > 0.0
    {
        let waste_b = recoverable_waste_per_hr(cpm_b, tps_b, eff_b);
        let waste_a = recoverable_waste_per_hr(cpm_a, tps_a, eff_a);
        if waste_b.is_finite() && waste_a.is_finite() {
            let arrow = recoverable_waste_arrow(waste_b, waste_a);
            println!("  Waste         ${waste_b:.2} → ${waste_a:.2}/hr {arrow}");
        }
    }
}

const COST_ARROW_THRESHOLD_USD: f64 = 0.01;
const RECOVERABLE_ARROW_THRESHOLD_USD_PER_HR: f64 = 0.05;
const JTOK_ARROW_THRESHOLD: f64 = 0.02;

fn recoverable_waste_per_hr(cpm: f64, tps: f64, efficiency_pct: f64) -> f64 {
    let cost_per_hr = cpm * tps * 3600.0 / 1_000_000.0;
    cost_per_hr * (ACHIEVABLE_EFFICIENCY_CEILING - efficiency_pct / 100.0).max(0.0)
}

fn recoverable_waste_available(d: &delta::Delta) -> bool {
    matches!(
        (
            d.cost_per_million_before,
            d.cost_per_million_after,
            d.throughput_before,
            d.throughput_after,
            d.efficiency_pct_before,
            d.efficiency_pct_after,
        ),
        (
            Some(cpm_b),
            Some(cpm_a),
            Some(tps_b),
            Some(tps_a),
            Some(eff_b),
            Some(eff_a),
        ) if cpm_b.is_finite()
            && cpm_a.is_finite()
            && tps_b > 0.0
            && tps_a > 0.0
            && eff_b.is_finite()
            && eff_a.is_finite()
    )
}

fn economics_section_active(d: &delta::Delta) -> bool {
    (d.cost_per_million_before.is_some() && d.cost_per_million_after.is_some())
        || (d.joules_per_token_before.is_some() && d.joules_per_token_after.is_some())
        || recoverable_waste_available(d)
}

fn jtok_change_arrow(before: f64, after: f64) -> &'static str {
    let diff = after - before;
    if diff < -JTOK_ARROW_THRESHOLD {
        "↓ (improved)"
    } else if diff > JTOK_ARROW_THRESHOLD {
        "↑ (regressed)"
    } else {
        ""
    }
}

fn cost_change_arrow(before: f64, after: f64) -> &'static str {
    let diff = after - before;
    if diff < -COST_ARROW_THRESHOLD_USD {
        "↓ (improved)"
    } else if diff > COST_ARROW_THRESHOLD_USD {
        "↑ (regressed)"
    } else {
        ""
    }
}

fn recoverable_waste_arrow(waste_before: f64, waste_after: f64) -> &'static str {
    let waste_diff = waste_after - waste_before;
    if waste_diff < -RECOVERABLE_ARROW_THRESHOLD_USD_PER_HR {
        "↓ (improved)"
    } else if waste_diff > RECOVERABLE_ARROW_THRESHOLD_USD_PER_HR {
        "↑ (regressed)"
    } else {
        ""
    }
}

fn throughput_arrow(before: f64, after: f64) -> &'static str {
    if after > before {
        "↑"
    } else if after < before {
        "↓"
    } else {
        ""
    }
}

fn latency_quality(delta_ms: f64) -> &'static str {
    if delta_ms < 0.0 {
        "↓ (improved)"
    } else if delta_ms > 0.0 {
        "↑ (regressed)"
    } else {
        ""
    }
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
            kv_cache_usage_perc: Some(50.0),
            num_running: Some(50.0),
            num_waiting: None,
            ridge_batch_size: Some(40.0),
            tpot_ms: Some(50.0),
            tpot_floor_ms: Some(10.0),
            chunked_prefill_enabled: Some(false),
            enforce_eager,
            enable_prefix_caching: None,
            quantization: None,
        }
    }

    fn capacity_input(enable_prefix_caching: Option<bool>) -> HealthyExitInput {
        HealthyExitInput {
            efficiency: Some(42.5),
            kv_cache_usage_perc: Some(85.0),
            num_running: Some(50.0),
            num_waiting: None,
            ridge_batch_size: Some(40.0),
            tpot_ms: Some(20.0),
            tpot_floor_ms: Some(5.0),
            chunked_prefill_enabled: Some(false),
            enforce_eager: None,
            enable_prefix_caching,
            quantization: None,
        }
    }

    fn physics_input(quantization: Option<String>) -> HealthyExitInput {
        HealthyExitInput {
            efficiency: Some(91.0),
            kv_cache_usage_perc: Some(50.0),
            num_running: Some(50.0),
            num_waiting: None,
            ridge_batch_size: Some(40.0),
            tpot_ms: Some(11.0),
            tpot_floor_ms: Some(10.0),
            chunked_prefill_enabled: Some(false),
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
            kv_cache_usage_perc: Some(50.0),
            num_running: Some(5.0),
            num_waiting: Some(0.0),
            ridge_batch_size: Some(100.0),
            tpot_ms: Some(20.0),
            tpot_floor_ms: Some(5.0),
            chunked_prefill_enabled: Some(false),
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert_eq!(
            msg,
            "No issues. Efficiency 34.0% - gap is traffic, not config. Increase load to find real limits."
        );
        assert!(!msg.contains('\n'));
    }

    #[test]
    fn healthy_exit_traffic_with_queue_does_not_claim_traffic_gap() {
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: Some(12.4),
            kv_cache_usage_perc: Some(10.0),
            num_running: Some(10.0),
            num_waiting: Some(5.0),
            ridge_batch_size: Some(100.0),
            tpot_ms: Some(20.0),
            tpot_floor_ms: Some(5.0),
            chunked_prefill_enabled: Some(false),
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(msg.contains("queue present but no rule explains it"));
        assert!(msg.contains("5 requests waiting"));
        assert!(!msg.contains("gap is traffic, not config"));
    }

    #[test]
    fn traffic_limiter_does_not_say_optimally_tuned() {
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: Some(34.0),
            kv_cache_usage_perc: Some(50.0),
            num_running: Some(5.0),
            num_waiting: None,
            ridge_batch_size: Some(100.0),
            tpot_ms: Some(20.0),
            tpot_floor_ms: Some(5.0),
            chunked_prefill_enabled: Some(false),
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(msg.contains("No issues."));
        assert!(!msg.contains("No actionable config fix identified."));
        assert!(!msg.contains("Rules clear"));
    }

    #[test]
    fn healthy_exit_physics_limiter() {
        let msg = healthy_exit_message(physics_input(None));
        assert!(msg.contains("Physics (Hardware Ceiling)"));
        assert!(msg.contains("Efficiency: 91.0%"));
        assert!(msg.contains("TPOT 11.0ms vs floor 10.0ms"));
    }

    #[test]
    fn healthy_exit_prefill_interference_limiter() {
        let msg = healthy_exit_message(HealthyExitInput {
            efficiency: Some(55.0),
            kv_cache_usage_perc: Some(50.0),
            num_running: Some(50.0),
            num_waiting: None,
            ridge_batch_size: Some(40.0),
            tpot_ms: Some(50.0),
            tpot_floor_ms: Some(10.0),
            chunked_prefill_enabled: Some(true),
            enforce_eager: None,
            enable_prefix_caching: None,
            quantization: None,
        });
        assert!(msg.contains("Prefill Interference"));
        assert!(msg.contains("Efficiency: 55.0%"));
    }

    #[test]
    fn healthy_exit_framework_overhead_limiter() {
        let msg = healthy_exit_message(framework_overhead_input(None));
        assert!(msg.contains("Framework Overhead"));
        assert!(msg.contains("Efficiency: 60.0%"));
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
            kv_cache_usage_perc: None,
            num_running: None,
            num_waiting: None,
            ridge_batch_size: None,
            tpot_ms: None,
            tpot_floor_ms: None,
            chunked_prefill_enabled: None,
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
        assert_eq!(
            drift::non_baseline_changes(&prev, &curr),
            vec!["  max_num_seqs  32 -> 98".to_string()]
        );
    }

    #[test]
    fn config_status_lines_drifted() {
        let lines = config_status_lines(true, &[]);
        assert!(
            lines
                .iter()
                .any(|l| l.contains("Config changed, baseline reset."))
        );
    }

    #[test]
    fn config_status_lines_non_baseline_changes() {
        let changes = vec!["  max_num_seqs  32 -> 98".to_string()];
        let lines = config_status_lines(false, &changes);
        assert!(lines.iter().any(|l| l.contains("Config changed:")));
        assert!(lines.iter().any(|l| l.contains("max_num_seqs  32 -> 98")));
    }

    #[test]
    fn config_status_lines_no_changes() {
        let lines = config_status_lines(false, &[]);
        assert!(lines.iter().any(|l| l == "  No config changed."));
    }

    #[test]
    fn efficiency_delta_near_zero_suppressed() {
        assert!(format_efficiency_delta_line(Some(-0.04)).is_none());
        assert!(format_efficiency_delta_line(Some(0.03)).is_none());
    }

    #[test]
    fn efficiency_delta_below_threshold_prints_down() {
        let line = format_efficiency_delta_line(Some(-0.06)).expect("line");
        assert!(line.contains('↓'));
        assert!(line.contains("-0.1pp"));
    }

    #[test]
    fn cost_arrow_suppressed_when_delta_below_threshold() {
        assert_eq!(cost_change_arrow(2.00, 2.005), "");
    }

    #[test]
    fn cost_arrow_down_when_improvement_above_threshold() {
        assert_eq!(cost_change_arrow(2.00, 1.98), "↓ (improved)");
    }

    #[test]
    fn recoverable_arrow_suppressed_when_delta_below_threshold() {
        assert_eq!(recoverable_waste_arrow(10.0, 9.97), "");
    }

    #[test]
    fn recoverable_arrow_down_when_waste_reduced_above_threshold() {
        assert_eq!(recoverable_waste_arrow(10.0, 9.94), "↓ (improved)");
    }

    #[test]
    fn jtok_arrow_down_when_energy_improves() {
        assert_eq!(jtok_change_arrow(0.31, 0.28), "↓ (improved)");
    }

    #[test]
    fn jtok_arrow_suppressed_below_threshold() {
        assert_eq!(jtok_change_arrow(0.31, 0.30), "");
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
            config_changes: Vec::new(),
            capacity_self_grade: None,
        };
        assert!(economics_section_active(&d));
    }

    #[test]
    fn economics_header_shown_for_recoverable_waste() {
        let d = delta::Delta {
            throughput_before: Some(1580.0),
            throughput_after: Some(4120.0),
            efficiency_delta_pp: Some(18.4),
            efficiency_pct_before: Some(36.0),
            efficiency_pct_after: Some(54.4),
            cost_per_million_before: Some(0.16),
            cost_per_million_after: Some(0.16),
            joules_per_token_before: None,
            joules_per_token_after: None,
            cost_source_after: Some(engine::CostSource::Catalog),
            ttft_before_ms: None,
            ttft_after_ms: None,
            tpot_before_ms: None,
            tpot_after_ms: None,
            ttft_p95_before_ms: None,
            ttft_p95_after_ms: None,
            tpot_p95_before_ms: None,
            tpot_p95_after_ms: None,
            config_drifted: false,
            config_changes: Vec::new(),
            capacity_self_grade: None,
        };
        assert!(recoverable_waste_available(&d));
        assert!(economics_section_active(&d));
    }

    #[test]
    fn delta_waste_uses_80_pct_ceiling() {
        let cpm = 0.34;
        let tps = 1216.0;
        let eff = 50.0;
        let d = delta::Delta {
            throughput_before: Some(tps),
            throughput_after: Some(tps),
            efficiency_delta_pp: None,
            efficiency_pct_before: Some(eff),
            efficiency_pct_after: Some(eff),
            cost_per_million_before: Some(cpm),
            cost_per_million_after: Some(cpm),
            joules_per_token_before: None,
            joules_per_token_after: None,
            cost_source_after: Some(engine::CostSource::Catalog),
            ttft_before_ms: None,
            ttft_after_ms: None,
            tpot_before_ms: None,
            tpot_after_ms: None,
            ttft_p95_before_ms: None,
            ttft_p95_after_ms: None,
            tpot_p95_before_ms: None,
            tpot_p95_after_ms: None,
            config_drifted: false,
            config_changes: Vec::new(),
            capacity_self_grade: None,
        };
        assert!(recoverable_waste_available(&d));
        let waste = recoverable_waste_per_hr(
            d.cost_per_million_before.unwrap(),
            d.throughput_before.unwrap(),
            d.efficiency_pct_before.unwrap(),
        );
        let cost_per_hr = cpm * tps * 3600.0 / 1_000_000.0;
        let expected = cost_per_hr * 0.30;
        let full_roofline_gap = cost_per_hr * 0.50;
        assert!(
            (waste - expected).abs() < 1e-9,
            "expected {expected}, got {waste}"
        );
        assert!(
            (waste - full_roofline_gap).abs() > 1e-9,
            "must not use 100% roofline gap"
        );
        let waste_at_85 = recoverable_waste_per_hr(cpm, tps, 85.0);
        assert_eq!(waste_at_85, 0.0);
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
