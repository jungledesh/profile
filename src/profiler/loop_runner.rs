use std::time::Duration;

use super::{DiagnoseResult, delta, drift, poll, run_diagnose, state::LoopState};
use crate::context::{AnalysisInput, RuntimeWindow};
use crate::engine;
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
                .groups
                .first()
                .map(|g| g.primary.rule_name)
            else {
                let baseline = last_state.report.baseline.as_ref();
                let efficiency = baseline.and_then(|b| b.efficiency_pct);
                let ridge_batch_size = baseline.map(|b| b.ridge_batch_size);
                let tpot_floor_ms = baseline.map(|b| b.tpot_floor_ms);
                let kv_cache_usage_perc = last_state.result.snapshot.vllm.kv_cache_usage_perc;
                let num_running = last_state.result.snapshot.vllm.num_requests_running;
                let tpot_ms = last_state.result.snapshot.vllm.tpot_ms;
                let chunked_prefill_enabled =
                    last_state.result.static_ctx.config.enable_chunked_prefill;
                let msg = healthy_exit_message(
                    efficiency,
                    kv_cache_usage_perc,
                    num_running,
                    ridge_batch_size,
                    tpot_ms,
                    tpot_floor_ms,
                    chunked_prefill_enabled,
                );
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
        last_fingerprint = verified_pass_fingerprint(&last_fingerprint, &new_result.snapshot)?;
        let agg_win = RuntimeWindow::from_snapshot(new_result.snapshot.clone());
        let summary = AnalysisInput::new(&new_result.static_ctx, &agg_win);
        let new_report = engine::build_report_for_diagnose(&new_result.windows, summary);

        let drifted = drift::config_changed(&prev_result.static_ctx, &new_result.static_ctx);

        let d = delta::compute(
            &prev_result,
            &prev_report,
            &new_result,
            &new_report,
            drifted,
        );
        let current_eff = new_report.baseline.as_ref().and_then(|b| b.efficiency_pct);
        let plateau_count = state.update_efficiency_plateau(current_eff, EFFICIENCY_PLATEAU_DELTA);
        print_delta(&d);
        println!();
        output::stdout::print_diagnose_table(&new_result, false);

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

fn at_hardware_ceiling(headroom_pct: Option<f64>) -> bool {
    headroom_pct.is_some_and(|h| h < CEILING_HEADROOM_THRESHOLD_PCT)
}

fn healthy_exit_message(
    efficiency: Option<f64>,
    kv_cache_usage_perc: Option<f64>,
    num_running: Option<f64>,
    ridge_batch_size: Option<f64>,
    tpot_ms: Option<f64>,
    tpot_floor_ms: Option<f64>,
    chunked_prefill_enabled: Option<bool>,
) -> String {
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
            format!(
                "Primary Limiter: KV Cache Capacity\n\
                 {eff_str} -{kv} concurrency is capped before bandwidth saturates.\n\
                 Levers: enable prefix caching, apply KV quantization (FP8), or add TP to split KV cache."
            )
        }
        Some(engine::limiter::PrimaryLimiter::Physics) => {
            let floor_str = tpot_floor_ms
                .zip(tpot_ms)
                .map(|(floor, actual)| format!(" TPOT {actual:.1}ms vs floor {floor:.1}ms."))
                .unwrap_or_default();
            format!(
                "Primary Limiter: Physics (Hardware Ceiling)\n\
                 {eff_str} -{floor_str} Hardware is at the limits of this model and dtype.\n\
                 Levers: quantize further (FP16→FP8/AWQ), speculative decoding, or scale out with TP."
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
            format!(
                "Primary Limiter: Framework Overhead\n\
                 {eff_str} - batch healthy, VRAM available, but GPU is waiting on the system.\n\
                 Levers: test --enforce-eager, verify CPU/PCIe bottlenecks, or evaluate SGLang for this workload."
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
        Some(_) => "Rules clear. Server is optimally tuned for current constraints.",
        None => "Rules clear.",
    };
    format!("{prefix}\n\n{limiter_block}")
}

fn format_efficiency_delta_line(delta_pp: Option<f64>) -> Option<String> {
    match delta_pp {
        Some(v) if v >= EFFICIENCY_DISPLAY_MIN_PP => Some(format!("  Efficiency  +{v:.1}pp ↑")),
        Some(v) if v <= -EFFICIENCY_DISPLAY_MIN_PP => Some(format!("  Efficiency  {v:.1}pp ↓")),
        _ => None,
    }
}

fn config_change_status_lines(config_drifted: bool) -> [&'static str; 2] {
    if config_drifted {
        ["  Config changed, baseline reset.", ""]
    } else {
        ["  No config changed.", ""]
    }
}

fn print_delta(d: &delta::Delta) {
    for line in config_change_status_lines(d.config_drifted) {
        if line.is_empty() {
            println!();
        } else {
            println!("{line}");
        }
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
            let arrow = latency_arrow(delta);
            let p95_suffix = match (d.ttft_p95_before_ms, d.ttft_p95_after_ms) {
                (Some(pb), Some(pa)) if pb.is_finite() && pa.is_finite() => {
                    format!("  (p95 {pb:.0} → {pa:.0}ms {})", latency_arrow(pa - pb))
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
            let arrow = latency_arrow(delta);
            let p95_suffix = match (d.tpot_p95_before_ms, d.tpot_p95_after_ms) {
                (Some(pb), Some(pa)) if pb.is_finite() && pa.is_finite() => {
                    format!("  (p95 {pb:.1} → {pa:.1}ms {})", latency_arrow(pa - pb))
                }
                _ => String::new(),
            };
            println!("  TPOT        {before:.1} → {after:.1}ms {arrow}{p95_suffix}");
        }
    }
    if let Some(line) = format_efficiency_delta_line(d.efficiency_delta_pp) {
        println!("{line}");
    }
    if let (Some(before), Some(after)) = (
        d.prefill_time_fraction_before,
        d.prefill_time_fraction_after,
    ) && before.is_finite()
        && after.is_finite()
    {
        let delta_pp = (after - before) * 100.0;
        if delta_pp.abs() > 1.0 {
            let arrow = if delta_pp < 0.0 { "↓" } else { "↑" };
            println!(
                "  Prefill time    {:.0}% → {:.0}%    {:+.0} pp {arrow}",
                before * 100.0,
                after * 100.0,
                delta_pp
            );
        }
    }
    if let (Some(before), Some(after)) = (d.prefill_efficiency_before, d.prefill_efficiency_after)
        && before.is_finite()
        && after.is_finite()
    {
        let delta_pp = after - before;
        if delta_pp.abs() > 0.5 {
            let arrow = if delta_pp < 0.0 { "↓" } else { "↑" };
            println!(
                "  Prefill eff.   {before:.1}% → {after:.1}%  {:+.1} pp {arrow}",
                delta_pp
            );
        }
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
        let waste_b = (cpm_b * tps_b * 3600.0 / 1_000_000.0) * (1.0 - eff_b / 100.0).max(0.0);
        let waste_a = (cpm_a * tps_a * 3600.0 / 1_000_000.0) * (1.0 - eff_a / 100.0).max(0.0);
        if waste_b.is_finite() && waste_a.is_finite() {
            let arrow = recoverable_waste_arrow(waste_b, waste_a);
            println!("  Waste         ${waste_b:.2} → ${waste_a:.2}/hr {arrow}");
        }
    }
}

const COST_ARROW_THRESHOLD_USD: f64 = 0.01;
const RECOVERABLE_ARROW_THRESHOLD_USD_PER_HR: f64 = 0.05;
const JTOK_ARROW_THRESHOLD: f64 = 0.02;

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
        "↓"
    } else if diff > JTOK_ARROW_THRESHOLD {
        "↑"
    } else {
        ""
    }
}

fn cost_change_arrow(before: f64, after: f64) -> &'static str {
    let diff = after - before;
    if diff < -COST_ARROW_THRESHOLD_USD {
        "↓"
    } else if diff > COST_ARROW_THRESHOLD_USD {
        "↑"
    } else {
        ""
    }
}

fn recoverable_waste_arrow(waste_before: f64, waste_after: f64) -> &'static str {
    let waste_diff = waste_after - waste_before;
    if waste_diff < -RECOVERABLE_ARROW_THRESHOLD_USD_PER_HR {
        "↓"
    } else if waste_diff > RECOVERABLE_ARROW_THRESHOLD_USD_PER_HR {
        "↑"
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

fn latency_arrow(delta_ms: f64) -> &'static str {
    if delta_ms < 0.0 {
        "↓"
    } else if delta_ms > 0.0 {
        "↑"
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

    #[test]
    fn healthy_exit_capacity_limiter() {
        let msg = healthy_exit_message(
            Some(42.5),
            Some(85.0),
            Some(50.0),
            Some(40.0),
            Some(20.0),
            Some(5.0),
            Some(false),
        );
        assert!(msg.contains("KV Cache Capacity"));
        assert!(msg.contains("Efficiency: 42.5%"));
        assert!(msg.contains("KV cache at 85%"));
    }

    #[test]
    fn healthy_exit_traffic_limiter() {
        let msg = healthy_exit_message(
            Some(34.0),
            Some(50.0),
            Some(5.0),
            Some(100.0),
            Some(20.0),
            Some(5.0),
            Some(false),
        );
        assert_eq!(
            msg,
            "No issues. Efficiency 34.0% - gap is traffic, not config. Increase load to find real limits."
        );
        assert!(!msg.contains('\n'));
    }

    #[test]
    fn traffic_limiter_does_not_say_optimally_tuned() {
        let msg = healthy_exit_message(
            Some(34.0),
            Some(50.0),
            Some(5.0),
            Some(100.0),
            Some(20.0),
            Some(5.0),
            Some(false),
        );
        assert!(msg.contains("No issues."));
        assert!(!msg.contains("optimally tuned"));
        assert!(!msg.contains("Rules clear"));
    }

    #[test]
    fn healthy_exit_physics_limiter() {
        let msg = healthy_exit_message(
            Some(91.0),
            Some(50.0),
            Some(50.0),
            Some(40.0),
            Some(11.0),
            Some(10.0),
            Some(false),
        );
        assert!(msg.contains("Physics (Hardware Ceiling)"));
        assert!(msg.contains("Efficiency: 91.0%"));
        assert!(msg.contains("TPOT 11.0ms vs floor 10.0ms"));
    }

    #[test]
    fn healthy_exit_prefill_interference_limiter() {
        let msg = healthy_exit_message(
            Some(55.0),
            Some(50.0),
            Some(50.0),
            Some(40.0),
            Some(50.0),
            Some(10.0),
            Some(true),
        );
        assert!(msg.contains("Prefill Interference"));
        assert!(msg.contains("Efficiency: 55.0%"));
    }

    #[test]
    fn healthy_exit_framework_overhead_limiter() {
        let msg = healthy_exit_message(
            Some(60.0),
            Some(50.0),
            Some(50.0),
            Some(40.0),
            Some(50.0),
            Some(10.0),
            Some(false),
        );
        assert!(msg.contains("Framework Overhead"));
        assert!(msg.contains("Efficiency: 60.0%"));
    }

    #[test]
    fn healthy_exit_insufficient_data_when_limiter_unknown() {
        let msg = healthy_exit_message(None, None, None, None, None, None, None);
        assert!(msg.contains("insufficient data to identify primary limiter"));
        assert!(msg.contains("Efficiency: unavailable"));
        assert!(!msg.contains("optimally tuned"));
        assert_eq!(msg.matches("Rules clear.").count(), 1);
    }

    #[test]
    fn config_change_status_lines_for_drifted_and_not_drifted() {
        let drifted = config_change_status_lines(true);
        assert_eq!(drifted[0], "  Config changed, baseline reset.");
        assert_eq!(drifted[1], "");
        let stable = config_change_status_lines(false);
        assert_eq!(stable[0], "  No config changed.");
        assert_eq!(stable[1], "");
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
        assert_eq!(cost_change_arrow(2.00, 1.98), "↓");
    }

    #[test]
    fn recoverable_arrow_suppressed_when_delta_below_threshold() {
        assert_eq!(recoverable_waste_arrow(10.0, 9.97), "");
    }

    #[test]
    fn recoverable_arrow_down_when_waste_reduced_above_threshold() {
        assert_eq!(recoverable_waste_arrow(10.0, 9.94), "↓");
    }

    #[test]
    fn jtok_arrow_down_when_energy_improves() {
        assert_eq!(jtok_change_arrow(0.31, 0.28), "↓");
    }

    #[test]
    fn jtok_arrow_suppressed_below_threshold() {
        assert_eq!(jtok_change_arrow(0.31, 0.30), "");
    }

    #[test]
    fn economics_header_shown_when_only_jtok_available() {
        let d = delta::Delta {
            throughput_delta_pct: None,
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
            ttft_p99_before_ms: None,
            ttft_p99_after_ms: None,
            tpot_p99_before_ms: None,
            tpot_p99_after_ms: None,
            ttft_p95_before_ms: None,
            ttft_p95_after_ms: None,
            tpot_p95_before_ms: None,
            tpot_p95_after_ms: None,
            ttft_p99_delta_pct: None,
            ttft_p95_delta_pct: None,
            tpot_p95_delta_pct: None,
            config_drifted: false,
            prefill_time_fraction_before: None,
            prefill_time_fraction_after: None,
            prefill_efficiency_before: None,
            prefill_efficiency_after: None,
        };
        assert!(economics_section_active(&d));
    }

    #[test]
    fn economics_header_shown_for_recoverable_waste() {
        let d = delta::Delta {
            throughput_delta_pct: Some(160.8),
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
            ttft_p99_before_ms: None,
            ttft_p99_after_ms: None,
            tpot_p99_before_ms: None,
            tpot_p99_after_ms: None,
            ttft_p95_before_ms: None,
            ttft_p95_after_ms: None,
            tpot_p95_before_ms: None,
            tpot_p95_after_ms: None,
            ttft_p99_delta_pct: None,
            ttft_p95_delta_pct: None,
            tpot_p95_delta_pct: None,
            config_drifted: false,
            prefill_time_fraction_before: None,
            prefill_time_fraction_after: None,
            prefill_efficiency_before: None,
            prefill_efficiency_after: None,
        };
        assert!(recoverable_waste_available(&d));
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
}
