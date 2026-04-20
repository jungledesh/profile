//! Live bottleneck diagnosis for vLLM inference. Layout: cli, profiler, collectors, engine.

#![warn(dead_code)]

pub mod cli;
pub mod collectors;
pub mod engine;
pub mod profiler;

pub use cli::{run, Cli};
