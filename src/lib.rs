use clap::{Parser, Subcommand};

/// Command-line interface definition for the `infer-pro` CLI.
#[derive(Debug, Parser)]
#[command(name = "infer-pro")]
#[command(about = "CLI tool for profiling vLLM GPU and system metrics", long_about = None)]
pub struct Cli {
    /// Increase output verbosity (can be used multiple times)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Subcommand to run
    #[command(subcommand)]
    pub command: Commands,
}

/// Top-level subcommands.
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Perform a dry-run profile (no real GPU work yet).
    Profile {
        /// Path to a configuration file for vLLM (not used yet).
        #[arg(short, long)]
        config: Option<String>,
    },

    /// Print basic information about the tool.
    Info,
}

/// Entry point for running the CLI, separated from `main` for testability.
pub fn run(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Commands::Profile { config } => {
            if let Some(path) = config {
                println!("(dry-run) would profile vLLM using config at: {path}");
            } else {
                println!("(dry-run) would profile vLLM with default configuration");
            }
        }
        Commands::Info => {
            println!("infer-pro: Rust CLI for profiling vLLM GPU and system metrics (scaffold)");
        }
    }

    if cli.verbose > 0 {
        eprintln!("Verbose level: {}", cli.verbose);
    }

    Ok(())
}

