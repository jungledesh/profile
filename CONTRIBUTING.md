# Contributing

Accepted: a new rule, an engine port, a GPU or model catalog entry, docs, or a bug report from a real server. For large changes, open an issue first. PRs target `main`.

Read [ARCHITECTURE.md](ARCHITECTURE.md) first. It tells you where code lives, which module owns what, and where your change belongs.

## Build

```bash
git clone https://github.com/jungledesh/profile
cd profile
cargo build
```

The toolchain is pinned in `rust-toolchain.toml`. `Cargo.lock` is committed and stays committed; it is required for reproducible builds and OSV scanning.

## Test

```bash
cargo test
```

Unit tests cover rule logic, physics math, and mock payload validation. Rule logic is the core engine: smoke test every rule with mock payloads, and confirm ranked output, suppression-table dedup, and confidence values.

End-to-end validation is manual, on a live vLLM server with a GPU (e.g. RunPod). No GPU-dependent tests in CI.

## The gate

Required before merge:

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo audit
cargo deny check
cargo test
```

Fix clippy findings, do not silence them. Never mask CI security scanners; `|| true` after a scanner is a rejection. Allowlist real false positives in `deny.toml`.

## Code rules

- **No dead code.** If it is not called outside tests, delete it.
- **No duplication.** Two near-identical functions become one. Near-duplicates count as duplicates.
- **No speculative abstraction.** No trait, enum, or generic until the second caller exists.
- **`Option<T>` over sentinels.** `Some(0.0) != None`.
- **No `unwrap()` / `expect()` in library code.** Use `?`, `let-else`, or `unwrap_or`. Panics only in `main` or provably unreachable branches, with the reason documented.
- **Errors:** `thiserror` for typed errors in engine and collectors, `anyhow` for top-level propagation. Never swallow errors silently.
- **Prefer immutability.** `let` by default, `let mut` only when mutation is required.
- **Deps are a tax.** Do not add a crate for something a 20-line helper solves.
- **Allocate intentionally.** `&str`/`&[T]` over `String`/`Vec`, borrow over clone, reuse buffers in hot loops.
- **HTTP uses rustls, not OpenSSL.** `default-features = false` on `reqwest`.
- **No secrets, tokens, or personal helper scripts in the repo.**

All user-facing strings, not only rule output, follow the output rules: prescriptive, units on every number, no colors, no emojis, `-` for a metric that cannot be read, `(est)` for values from the physics model.

## Adding a new rule

1. Create `src/engine/rules/rN_name.rs`. Match the API shape of the nearest existing rule: a detection function over the snapshot or window evidence, plus a `*_recommendation(...) -> Option<Recommendation>` builder. APIs vary by rule; the neighbors are the reference.
2. Threshold constants at the top of the new file.
3. Assign the rule to a DAG layer (L2-L6) and define any mutual exclusivity suppressions in the suppression table.
4. Wire it into `rules::build_report_for_windows` in `src/engine/rules/eval.rs`.
5. Smoke test with a mock payload in unit tests. Then validate against a live vLLM server with a GPU.
6. Output is prescriptive: what to change and how, one line, with units. Rules fire on evidence of harm (evictions, queue growth, latency inflation), never on utilization alone.

## Where to start

[Open an issue](https://github.com/jungledesh/profile/issues) with what you run and what is missing, or email [jungledesh@gmail.com](mailto:jungledesh@gmail.com).
