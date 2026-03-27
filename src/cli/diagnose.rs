//! `profile diagnose` — collect snapshot, print v1-style summary.

use super::DiagnoseArgs;
use crate::profiler;

pub fn execute(args: &DiagnoseArgs) -> anyhow::Result<()> {
    let result = profiler::run_diagnose(&args.url)?;

    println!("=== PROFILE DIAGNOSE ===");
    println!();

    match &result.snapshot.gpu.gpu_name {
        Some(name) => println!("GPU             : {}", name),
        None => println!("GPU             : (no GPU / NVML not ready)"),
    }

    match result.snapshot.gpu.gpu_util_pct {
        Some(util) => println!("GPU Utilization : {:.1}%", util),
        None => println!("GPU Utilization : (no GPU / NVML not ready)"),
    }

    match result.snapshot.gpu.power_watts {
        Some(power) => println!("Power Draw      : {:.1} W", power),
        None => println!("Power Draw      : (no GPU / NVML not ready)"),
    }

    match result.snapshot.vllm.ttft_ms {
        Some(ms) => println!("TTFT (est. ms)  : {:.1}", ms),
        None => println!("TTFT (est. ms)  : (not available)"),
    }

    match result.snapshot.vllm.generation_tokens_total {
        Some(n) => println!("Gen tokens      : {:.0} (counter)", n),
        None => println!("Gen tokens      : (not parsed)"),
    }

    if result.snapshot.gpu.gpu_util_pct.is_none() && !result.snapshot.vllm.has_scrape_data() {
        println!(
            "\n(No metrics in snapshot — NVML unavailable or vLLM scrape not implemented yet.)"
        );
    } else {
        println!("\nSnapshot collected; rule engine and richer /metrics fields still TODO.");
    }

    Ok(())
}
