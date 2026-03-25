//! CLI: parse commands, print results.

mod diagnose;
mod info;

use clap::{Parser, Subcommand};

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
    /// Collect metrics and surface issues (v1).
    Diagnose(DiagnoseArgs),

    /// Print tool information.
    Info,
}

#[derive(Debug, clap::Args)]
pub struct DiagnoseArgs {
    /// vLLM server base URL
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    pub url: String,
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
