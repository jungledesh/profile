//! CLI: parse commands, print results.

mod diagnose;
mod info;

use clap::{Parser, Subcommand};

/// Matches vLLM engine default when `--max-num-seqs` is omitted.
pub const DEFAULT_MAX_NUM_SEQS: u32 = 256;

const DIAGNOSE_LONG_HELP: &str = r#"Print GPU (NVML) and vLLM /metrics in one view.

Example:
  GPU name                 : NVIDIA GeForce RTX 4090
  GPU index                : 0
  GPU ID (UUID)            : GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
  GPU util %               : 45.0
  Mem ctrl util %          : 12.0
  VRAM % used              : 12000 / 24564 MiB (48.8)
  Power draw               : 220 / 450 W
  SM clock                 : 2100 MHz
  In-batch reqs            : 4.0 (avg 2s)
  Waiting reqs             : 0.0 (avg 2s)
  Max seqs                 : 256
  TTFT (est. ms)           : 120.0 (window)
  Prefill ms               : 80.0 (window)
  Queue ms                 : 2.0 (window)
  TPOT ms                  : 12.0 (window)
  Prompt mean              : 128.0 tok (window)
  Gen tok/s                : 42.0 (window)
  Prefix cache hit rate (last scrape) : 82.0%
  Gen tokens               : 1000 (total)
"#;

#[derive(Debug, Parser)]
#[command(name = "profile")]
#[command(about = "Diagnose vLLM GPU and inference efficiency")]
pub struct Cli {
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    #[command(about = "Print GPU (NVML) and vLLM /metrics in one view.", long_about = DIAGNOSE_LONG_HELP)]
    Diagnose(DiagnoseArgs),

    /// Print tool information.
    Info,
}

#[derive(Debug, clap::Args)]
pub struct DiagnoseArgs {
    /// vLLM server base URL
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    pub url: String,

    /// Engine `max_num_seqs` if absent on `/metrics` (Prometheus gauge still wins when present)
    #[arg(long, default_value_t = DEFAULT_MAX_NUM_SEQS)]
    pub max_num_seqs: u32,
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    match &cli.command {
        Commands::Diagnose(args) => diagnose::execute(args)?,
        Commands::Info => info::execute(cli.verbose)?,
    }

    if cli.verbose > 0 {
        eprintln!("Verbose level: {}", cli.verbose);
    }

    Ok(())
}
