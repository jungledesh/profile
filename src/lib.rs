//! profile — CLI for profiling vLLM GPU and system metrics.
//!
//! Layout:
//! - **cli**: command parsing, print results
//! - **profiler**: run requests, measure latency, compute metrics
//! - **collectors**: GPU util, power draw, token stats

pub mod cli;
pub mod collectors;
pub mod profiler;

pub use cli::{run, Cli};
