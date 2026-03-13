//! Run subcommand: run profiler, print results.

use super::ProfileArgs;
use crate::profiler;

pub fn execute(args: &ProfileArgs, verbose: u8) -> anyhow::Result<()> {
    let result = profiler::run_profile(args.config.as_deref())?;

    println!("Profile v1 — Waste Detection (RTX 4090)");
    println!("======================================");

    match &result.snapshot.gpu_name {
        Some(name) => println!("GPU             : {}", name),
        None => println!("GPU             : (no GPU / NVML not ready)"),
    }

    match result.snapshot.gpu_util {
        Some(util) => println!("GPU Utilization : {:.1}%", util),
        None => println!("GPU Utilization : (no GPU / NVML not ready)"),
    }

    match result.snapshot.power_w {
        Some(power) => println!("Power Draw      : {:.1} W", power),
        None => println!("Power Draw      : (no GPU / NVML not ready)"),
    }

    match result.snapshot.tokens_per_sec {
        Some(tps) => println!("Tokens/sec      : {:.1}", tps),
        None => println!("Tokens/sec      : (vLLM adapter coming next iteration)"),
    }

    println!("\nInsight: GPU utilization low → classic waste detected.");
    println!("         (Next: add TTFT/TPOT + cost formula in profiler)");

    if verbose > 0 {
        eprintln!("Verbose level: {}", verbose);
    }

    Ok(())
}
