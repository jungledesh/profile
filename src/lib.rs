//! infer-pro — CLI for profiling vLLM GPU and system metrics.

mod cmd;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "infer-pro")]
#[command(about = "CLI tool for profiling vLLM GPU and system metrics", long_about = None)]
pub struct Cli {
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Perform a dry-run profile (no real GPU work yet).
    Profile(ProfileArgs),

    /// Print basic information about the tool.
    Info,
}

#[derive(Debug, clap::Args)]
pub struct ProfileArgs {
    #[arg(short, long)]
    pub config: Option<String>,
}

/// Entry point: parse CLI and run the chosen subcommand.
pub fn run(cli: Cli) -> anyhow::Result<()> {
    match &cli.command {
        Commands::Profile(args) => cmd::profile::execute(args, cli.verbose)?,
        Commands::Info => cmd::info::execute(cli.verbose)?,
    }

    if cli.verbose > 0 {
        eprintln!("Verbose level: {}", cli.verbose);
    }

    Ok(())
}
