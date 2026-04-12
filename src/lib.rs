//! CLI for profiling vLLM GPU and system metrics. Layout: cli, profiler, collectors, engine.

pub mod cli;
pub mod collectors;
pub mod engine;
pub mod profiler;

pub use cli::{run, Cli};
