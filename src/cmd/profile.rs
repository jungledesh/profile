use crate::ProfileArgs;

pub fn execute(args: &ProfileArgs, _verbose: u8) -> anyhow::Result<()> {
    if let Some(ref path) = args.config {
        println!("(dry-run) would profile vLLM using config at: {path}");
    } else {
        println!("(dry-run) would profile vLLM with default configuration");
    }
    Ok(())
}
