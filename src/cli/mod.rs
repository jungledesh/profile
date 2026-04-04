//! CLI: parse commands, print results.

mod diagnose;

use clap::{CommandFactory, Parser, Subcommand};

/// Matches vLLM engine default when `--max-num-seqs` is omitted.
pub const DEFAULT_MAX_NUM_SEQS: u32 = 256;

/// Default `-u` / `--url` value (full `/metrics` URL; base URLs are also accepted).
pub const DEFAULT_METRICS_URL: &str = "http://localhost:8000/metrics";

const ABOUT: &str = "Detects inefficiencies. Suggests fixes.";

/// Shown for `profile diagnose --help` only (root help omits options via template).
const DIAGNOSE_ABOUT: &str = "Collects metrics. Detects inefficiencies. Suggests fixes.";

#[derive(Debug, Parser)]
#[command(
    name = "profile",
    about = ABOUT,
    arg_required_else_help = true,
    disable_help_subcommand = true,
    override_usage = "profile <COMMAND> [OPTIONS]",
    help_template = "\n\n{about}\n\n{usage-heading} {usage}\n\nCommands:\n{subcommands}\n",
    disable_help_flag = true,
)]
pub struct Cli {
    #[arg(
        short = 'h',
        long = "help",
        global = true,
        action = clap::ArgAction::Help,
        help = "Display this message",
        display_order = 2
    )]
    pub help_flag: Option<bool>,

    #[arg(
        short = 'm',
        long = "max-num-seqs",
        global = true,
        default_value_t = DEFAULT_MAX_NUM_SEQS,
        help = "Engine max_num_seqs if absent on /metrics",
        display_order = 1
    )]
    pub max_num_seqs: u32,

    #[arg(
        short = 'u',
        long,
        global = true,
        default_value = DEFAULT_METRICS_URL,
        help = "vLLM metrics endpoint",
        display_order = 0
    )]
    pub url: String,

    #[arg(
        short,
        long,
        action = clap::ArgAction::Count,
        global = true,
        hide = true
    )]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    #[command(
        about = "Run diagnostics",
        long_about = DIAGNOSE_ABOUT,
        override_usage = "profile diagnose [OPTIONS]",
        help_template = "\n\n{about}\n\n{usage-heading} {usage}\n\n{all-args}\n",
        display_order = 0
    )]
    Diagnose,

    #[command(about = "Display this message", display_order = 1)]
    Help,
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    match &cli.command {
        Commands::Diagnose => diagnose::execute(&cli.url, cli.max_num_seqs)?,
        Commands::Help => {
            Cli::command().print_long_help()?;
            println!();
        }
    }

    if cli.verbose > 0 {
        eprintln!("Verbose level: {}", cli.verbose);
    }

    Ok(())
}
