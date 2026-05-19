//! CLI: parse commands, print results.

mod diagnose;

use clap::{CommandFactory, Parser, Subcommand};
use std::time::Duration;

const DEFAULT_MAX_NUM_SEQS: u32 = 256;
const DEFAULT_METRICS_URL: &str = "http://localhost:8000/metrics";
const DEFAULT_DURATION: &str = "30s";

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
            help = "Observation window (default: 30s). s=seconds, m=minutes (not ms/mins). Examples: 30s, 2m, 5m, 10m, 30m",
            long_help = "How long to collect metrics before analyzing each iteration (default: 30s).\n\n\
                Units:\n  \
                  s  seconds\n  \
                  m  minutes — not ms (milliseconds), not \"mins\", not bare m\n\n\
                Examples:\n  \
                  30s   half-minute snapshot\n  \
                  2m    short run\n  \
                  5m    typical load test\n  \
                  10m   sustained observation\n  \
                  30m   long soak"
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
        return Err(
            "duration needs a number and unit: s (seconds) or m (minutes). Examples: 30s, 2m, 5m, 10m, 30m"
                .to_string(),
        );
    }
    let (num, unit) = s.split_at(s.len() - 1);
    let value: u64 = num
        .parse()
        .map_err(|_| format!("invalid duration value in \"{input}\" (examples: 30s, 5m, 10m)"))?;
    if value == 0 {
        return Err("duration must be greater than zero".to_string());
    }
    match unit {
        "s" => Ok(Duration::from_secs(value)),
        "m" => Ok(Duration::from_secs(value.saturating_mul(60))),
        _ => Err(format!(
            "invalid unit in \"{input}\": use s (seconds) or m (minutes), not ms/mins — e.g. 5m, 10m, 30m"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_duration_seconds_and_minutes() {
        assert_eq!(parse_duration_arg("30s").unwrap(), Duration::from_secs(30));
        assert_eq!(parse_duration_arg("2m").unwrap(), Duration::from_secs(120));
        assert_eq!(parse_duration_arg("10m").unwrap(), Duration::from_secs(600));
        assert_eq!(
            parse_duration_arg("30m").unwrap(),
            Duration::from_secs(1800)
        );
    }

    #[test]
    fn parse_duration_rejects_ms_and_bare_number() {
        assert!(parse_duration_arg("5ms").is_err());
        assert!(parse_duration_arg("5").is_err());
    }
}
