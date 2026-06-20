use anyhow::Result;
use clap::Parser;
use profile::{Cli, run};

fn main() -> Result<()> {
    let cli = Cli::parse();
    run(cli)
}
