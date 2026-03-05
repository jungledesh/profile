## infer-profiler

**infer-profiler** is a Rust-based CLI tool intended to profile vLLM inference runs, focusing on GPU and related system metrics.

This branch contains only the basic Rust scaffold with a short `profile` CLI.

### Prerequisites

- **Rust toolchain**: Install via [`https://rustup.rs`](https://rustup.rs).

### Build

From the project root:

```bash
cargo build
```

### Run

From the project root, use the short `profile` binary name:

```bash
cargo run --bin profile -- --help
```

or after building:

```bash
./target/debug/profile --help
```

This will show the Clap-based CLI help for the `profile` command.
