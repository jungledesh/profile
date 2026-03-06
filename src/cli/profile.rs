//! Profile subcommand: run profiler and print results.

use crate::profiler;
use super::ProfileArgs;

pub fn execute(args: &ProfileArgs, _verbose: u8) -> anyhow::Result<()> {
    let result = profiler::run_profile(args.config.as_deref())?;
    print_result(&result);
    Ok(())
}

fn print_result(result: &profiler::ProfileResult) {
    if let Some(ref path) = result.config_path {
        println!("(dry-run) would profile vLLM using config at: {path}");
    } else {
        println!("(dry-run) would profile vLLM with default configuration");
    }
}
