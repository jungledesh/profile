//! CLI: parse commands, print results.

mod info;
mod run;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "profile")]
#[command(about = "Profile vLLM GPU and system metrics")]
pub struct Cli {
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Run a profile (dry-run for now).
    Run(ProfileArgs),

    /// Print tool information.
    Info,
}

#[derive(Debug, clap::Args)]
pub struct ProfileArgs {
    #[arg(short, long)]
    pub config: Option<String>,
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    match &cli.command {
        Commands::Run(args) => run::execute(args, cli.verbose)?,
        Commands::Info => info::execute(cli.verbose)?,
    }

    if cli.verbose > 0 {
        eprintln!("Verbose level: {}", cli.verbose);
    }

    Ok(())
}
