//! Info subcommand: print tool information.

pub fn execute(_verbose: u8) -> anyhow::Result<()> {
    println!("profile: Rust CLI for profiling vLLM GPU and system metrics (scaffold)");
    Ok(())
}