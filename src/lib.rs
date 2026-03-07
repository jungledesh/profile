//! CLI for profiling vLLM GPU and system metrics. Layout: cli, profiler, collectors.

pub mod cli;
pub mod collectors;
pub mod profiler;

pub use cli::{run, Cli};
