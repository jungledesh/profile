use anyhow::Result;
use clap::Parser;
use infer_profiler::{run, Cli};

fn main() -> Result<()> {
    let cli = Cli::parse();
    run(cli)
}