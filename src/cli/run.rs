//! Run subcommand: run profiler, print results.

use crate::profiler;
use super::ProfileArgs;

pub fn execute(args: &ProfileArgs, _verbose: u8) -> anyhow::Result<()> {
    let result = profiler::run_profile(args.config.as_deref())?;
    match &result.config_path {
        Some(path) => println!("(dry-run) would profile vLLM using config at: {path}"),
        None => println!("(dry-run) would profile vLLM with default configuration"),
    }
    Ok(())
}
