//! CLI: parse commands, print results.

mod diagnose;

use clap::{CommandFactory, Parser, Subcommand};
use std::time::Duration;

/// Matches vLLM engine default when `--max-num-seqs` is omitted.
pub const DEFAULT_MAX_NUM_SEQS: u32 = 256;

/// Default `-u` / `--url` value (full `/metrics` URL; base URLs are also accepted).
pub const DEFAULT_METRICS_URL: &str = "http://localhost:8000/metrics";
pub const DEFAULT_DURATION: &str = "2s";

const ABOUT: &str = "Detects inefficiencies. Suggests fixes.";

/// Shown for `profile diagnose --help` only (root help omits options via template).
const DIAGNOSE_ABOUT: &str = "Collects metrics. Detects inefficiencies. Suggests fixes.\nPass -v to show per-rule status when no issue is detected.";

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
    Diagnose {
        #[arg(
            long = "duration",
            default_value = DEFAULT_DURATION,
            value_parser = parse_duration_arg,
            help = "Observation duration (e.g. 30s, 1m, 5m, 30m)"
        )]
        duration: Duration,
    },

    #[command(about = "Display this message", display_order = 1)]
    Help,
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    match &cli.command {
        Commands::Diagnose { duration } => {
            diagnose::execute(&cli.url, cli.max_num_seqs, cli.verbose > 0, *duration)?
        }
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

fn parse_duration_arg(input: &str) -> Result<Duration, String> {
    let s = input.trim();
    if s.len() < 2 {
        return Err("duration must be like 30s, 1m, 5m".to_string());
    }
    let (num, unit) = s.split_at(s.len() - 1);
    let value: u64 = num
        .parse()
        .map_err(|_| format!("invalid duration value: {input}"))?;
    if value == 0 {
        return Err("duration must be greater than zero".to_string());
    }
    match unit {
        "s" => Ok(Duration::from_secs(value)),
        "m" => Ok(Duration::from_secs(value.saturating_mul(60))),
        _ => Err(format!("invalid duration unit in {input}; use s or m")),
    }
}
