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

    match result.snapshot.vllm.tps {
        Some(tps) => println!("Tokens/sec      : {:.1}", tps),
        None => println!("Tokens/sec      : (not parsed yet)"),
    }

    if result.snapshot.gpu.gpu_util_pct.is_none() && result.snapshot.vllm.tps.is_none() {
        println!(
            "\n(No metrics in snapshot — NVML unavailable or vLLM scrape not implemented yet.)"
        );
    } else {
        println!("\nSnapshot collected; rule engine + vLLM /metrics parse still TODO.");
    }

    Ok(())
}
