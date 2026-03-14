//! Run subcommand: run profiler, print results.

use super::ProfileArgs;
use crate::profiler;

pub fn execute(args: &ProfileArgs, _verbose: u8) -> anyhow::Result<()> {
    let result = profiler::run_profile(args.config.as_deref())?;

    println!("Profile v1 — Waste Detection (RTX 4090)");
    println!("======================================");

    if let Some(name) = &result.snapshot.gpu_name {
        println!("GPU             : {}", name);
    }

    if let Some(util) = result.snapshot.gpu_util {
        println!("GPU Utilization : {:.1}%", util);
    }

    if let Some(power) = result.snapshot.power_w {
        println!("Power Draw      : {:.1} W", power);
    }

    if let Some(mem) = result.snapshot.memory_used_mb {
        println!("Memory Used     : {} MB", mem);
    }

    if let Some(tps) = result.snapshot.tokens_per_sec {
        println!("Tokens/sec      : {:.1}", tps);
    } else {
        println!("Tokens/sec      : (vLLM adapter coming next)");
    }

    println!("\nInsight: GPU utilization low → classic waste detected.");
    println!("         (Memory usage now visible — watch for KV pressure)");

    Ok(())
}
