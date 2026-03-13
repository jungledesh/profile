use anyhow::Result;
use clap::Parser;
use profile::{run, Cli};

fn main() -> Result<()> {
    let cli = Cli::parse();
    run(cli)
}
