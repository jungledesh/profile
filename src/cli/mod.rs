//! CLI: parse commands, print results.

mod diagnose;
mod gpu_assignment;

use clap::{CommandFactory, Parser, Subcommand};
use std::io::{self, Write};
use std::time::Duration;

const DEFAULT_METRICS_URL: &str = "http://localhost:8000/metrics";
const DEFAULT_DURATION: &str = "30s";

const ABOUT: &str = "Detects inefficiencies. Suggests fixes.";
const MAX_DURATION: Duration = Duration::from_secs(30 * 60);
const MIN_DURATION: Duration = Duration::from_secs(30);

/// Shown for `profile diagnose --help` only (root help omits the loop line).
const DIAGNOSE_ABOUT: &str = "Collects metrics. Detects inefficiencies. Suggests fixes.\nA loop: collect, you apply a fix, remeasure. Not a one-shot dump.\nPass -v to show rules that did not fire, physics limits, and extra GPU, latency, cache, and config detail.";

#[derive(Debug, Parser)]
#[command(
    name = "profile",
    version = env!("CARGO_PKG_VERSION"),
    about = ABOUT,
    arg_required_else_help = true,
    disable_help_subcommand = true,
    override_usage = "profile <COMMAND> [OPTIONS]",
    help_template = "\n\n{name} {version}\n\n{about}\n\n{usage-heading} {usage}\n\nCommands:\n{subcommands}\n\nOptions:\n{options}\n",
    disable_help_flag = true,
    next_help_heading = "Options",
)]
pub struct Cli {
    #[arg(
        short = 'u',
        long,
        global = true,
        default_value = DEFAULT_METRICS_URL,
        env = "PROFILE_URL",
        hide_env_values = true,
        help = "vLLM metrics endpoint",
        display_order = 0
    )]
    pub url: String,

    #[arg(
        long = "duration",
        global = true,
        default_value = DEFAULT_DURATION,
        value_parser = parse_duration_arg,
        env = "PROFILE_DURATION",
        hide_env_values = true,
        help = "Measurement period (default: 30s, minimum: 30s, maximum: 30m). s=seconds, m=minutes (not ms/mins). Examples: 30s, 2m, 10m, 30m",
        long_help = "How long to collect metrics before analyzing each iteration (default: 30s).\n\n\
            Units:\n  \
              s  seconds\n  \
              m  minutes (not ms, not \"mins\", not bare m)\n\n\
            Examples:\n  \
              30s   minimum (default)\n  \
              2m    short run\n  \
              10m   longer run\n  \
              30m   maximum",
        display_order = 1
    )]
    pub duration: Duration,

    #[arg(
        short = 'm',
        long = "max-num-seqs",
        global = true,
        env = "PROFILE_MAX_NUM_SEQS",
        hide_env_values = true,
        help = "Engine max_num_seqs (auto-detected from /metrics when available; prompted if absent)",
        display_order = 2
    )]
    pub max_num_seqs: Option<u32>,

    #[arg(
        long = "tensor-parallel-size",
        global = true,
        env = "PROFILE_TENSOR_PARALLEL_SIZE",
        hide_env_values = true,
        help = "Tensor parallel degree for this vLLM instance",
        display_order = 3
    )]
    pub tensor_parallel_size: Option<u32>,

    #[arg(
        long = "cost-per-hour",
        global = true,
        value_parser = parse_cost_per_hour_arg,
        env = "PROFILE_COST_PER_HOUR",
        hide_env_values = true,
        help = "GPU cost in USD/hr (overrides catalog estimate)",
        display_order = 4
    )]
    pub cost_per_hour: Option<f64>,

    #[arg(
        short,
        long,
        action = clap::ArgAction::Count,
        global = true,
        env = "PROFILE_VERBOSE",
        hide_env_values = true,
        help = "Show rules that did not fire, physics limits, and extra GPU, latency, cache, and config detail",
        display_order = 5
    )]
    pub verbose: u8,

    #[arg(
        short = 'h',
        long = "help",
        global = true,
        action = clap::ArgAction::Help,
        help = "Display this message",
        display_order = 6
    )]
    pub help_flag: Option<bool>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    #[command(
        about = "Run diagnostics",
        long_about = DIAGNOSE_ABOUT,
        version = env!("CARGO_PKG_VERSION"),
        disable_version_flag = true,
        override_usage = "profile diagnose [OPTIONS]",
        help_template = "\n\nprofile {version}\n\n{about}\n\n{usage-heading} {usage}\n\n{all-args}\n",
        display_order = 0
    )]
    Diagnose,

    #[command(about = "Display this message", display_order = 1)]
    Help,

    #[command(
        about = "Print shell completion script",
        override_usage = "profile completions <SHELL>",
        display_order = 2
    )]
    Completions {
        #[arg(value_enum, help = "Shell: bash, elvish, fish, powershell, zsh")]
        shell: clap_complete::Shell,
    },

    #[command(
        about = "Print man page",
        override_usage = "profile man",
        display_order = 3
    )]
    Man,
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    match &cli.command {
        Commands::Diagnose => diagnose::execute(
            &cli.url,
            cli.max_num_seqs,
            cli.cost_per_hour,
            cli.tensor_parallel_size,
            cli.verbose > 0,
            cli.duration,
        )?,
        Commands::Help => {
            Cli::command().print_long_help()?;
            println!();
        }
        Commands::Completions { shell } => {
            let mut cmd = Cli::command();
            clap_complete::generate(*shell, &mut cmd, "profile", &mut io::stdout());
        }
        Commands::Man => {
            let cmd = Cli::command();
            let man = clap_mangen::Man::new(cmd);
            let mut buf = Vec::new();
            man.render(&mut buf)?;
            io::stdout().write_all(&buf)?;
        }
    }

    Ok(())
}

fn parse_duration_arg(input: &str) -> Result<Duration, String> {
    let s = input.trim();
    if s.len() < 2 {
        return Err(
            "duration needs a number and unit: s (seconds) or m (minutes). Examples: 30s, 2m, 10m, 30m"
                .to_string(),
        );
    }
    let (num, unit) = s.split_at(s.len() - 1);
    let value: u64 = num.parse().map_err(|_| {
        format!("invalid duration value in \"{input}\" (examples: 30s, 2m, 10m, 30m)")
    })?;
    if value == 0 {
        return Err("duration must be greater than zero".to_string());
    }
    let duration = match unit {
        "s" => Duration::from_secs(value),
        "m" => Duration::from_secs(value.saturating_mul(60)),
        _ => {
            return Err(format!(
                "invalid unit in \"{input}\": use s (seconds) or m (minutes), not ms/mins. Examples: 30s, 2m, 10m, 30m"
            ));
        }
    };
    if duration < MIN_DURATION {
        return Err("minimum duration is 30s".to_string());
    }
    if duration > MAX_DURATION {
        return Err("maximum duration is 30m".to_string());
    }
    Ok(duration)
}

fn parse_cost_per_hour_arg(input: &str) -> Result<f64, String> {
    let v: f64 = input
        .parse()
        .map_err(|_| format!("invalid --cost-per-hour value \"{input}\""))?;
    if !(v.is_finite() && v > 0.0) {
        return Err("--cost-per-hour must be a positive number".to_string());
    }
    Ok(v)
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
            Duration::from_secs(30 * 60)
        );
    }

    #[test]
    fn parse_duration_rejects_ms_and_bare_number() {
        assert!(parse_duration_arg("5ms").is_err());
        assert!(parse_duration_arg("5").is_err());
    }

    #[test]
    fn parse_duration_rejects_below_30s() {
        assert_eq!(
            parse_duration_arg("29s").unwrap_err(),
            "minimum duration is 30s"
        );
        assert_eq!(
            parse_duration_arg("1s").unwrap_err(),
            "minimum duration is 30s"
        );
        assert!(parse_duration_arg("30s").is_ok());
    }

    #[test]
    fn parse_duration_rejects_above_30m() {
        assert_eq!(
            parse_duration_arg("31m").unwrap_err(),
            "maximum duration is 30m"
        );
        assert_eq!(
            parse_duration_arg("1801s").unwrap_err(),
            "maximum duration is 30m"
        );
        assert!(parse_duration_arg("30m").is_ok());
        assert!(parse_duration_arg("3m").is_ok());
    }

    #[test]
    fn parse_cost_per_hour_accepts_positive() {
        assert!((parse_cost_per_hour_arg("3.5").unwrap() - 3.5).abs() < 1e-9);
    }

    #[test]
    fn parse_cost_per_hour_rejects_non_positive() {
        assert!(parse_cost_per_hour_arg("0").is_err());
        assert!(parse_cost_per_hour_arg("-1").is_err());
    }
}
