# Contributing

Accepted: a new rule, an engine port, a GPU or model catalog entry, docs, or a bug report from a real server. For large changes, open an issue first. PRs target `main`.

Read [ARCHITECTURE.md](ARCHITECTURE.md) first. It tells you where code lives, which module owns what, and where your change belongs.

## Build

```bash
git clone https://github.com/jungledesh/profile
cd profile
cargo build --locked
```

The toolchain is pinned in `rust-toolchain.toml`. `Cargo.lock` is committed and stays committed; it is required for reproducible builds and OSV scanning. Use `--locked` on build, test, and Clippy so the committed lockfile is what runs.

## Test

```bash
cargo test --locked
```

Unit tests cover rule logic, physics math, and mock payload validation. Rule logic is the core engine: smoke test every rule with mock payloads, and confirm ranked output, suppression-table dedup, and confidence values.

End-to-end validation is manual, on a live vLLM server with a GPU (e.g. RunPod). No GPU-dependent tests in CI.

## The gate

Required before merge (matches CI in `.github/workflows/build.yml`):

```bash
cargo fmt -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo audit
cargo deny check --all-features
cargo test --locked
```

CI also runs OSV-Scanner (fail on HIGH/CRITICAL in `Cargo.lock`), Semgrep SAST, and [Socket](https://socket.dev/) supply-chain scanning on every pull request and merge to `main`. Socket runs as the Socket for GitHub App and reports as the `Socket Sec: Project Report` check.

Fix clippy findings, do not silence them. Never mask CI security scanners; `|| true` after a scanner is a rejection. Allowlist real false positives in `deny.toml`.

## Code rules

- **No dead production code.** If production code is unused outside tests, delete it. Test fixtures, mock payload helpers, and other `#[cfg(test)]` utilities are exempt.
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

All user-facing strings, not only rule output, follow the output rules: prescriptive, units on every number, no colors, no emojis, `-` for a metric that cannot be read, `(est)` for values from the physics model, `~` for values derived from estimated ceilings.

## Adding a new rule

Follow this order. Skipping a step creates a rule the DAG cannot see.

1. Create `src/engine/rules/rN_name.rs` with a production evaluator over the snapshot or window evidence. Match the nearest production neighbor. Add a `*_recommendation(...)` helper only when that neighbor exposes one in production code; do not treat test-only helpers (for example `r1_recommendation`) as the API.
2. Threshold constants at the top of that file. Named, not magic numbers in the evaluator.
3. Assign a DAG layer on the `Recommendation` you push (today: L2 OOM/KV, L3 seats, L4 under-batching, L5 prefill/prefix, L6 config headroom). Add suppression-table rows in `SUPPRESSION_TABLE` in `src/engine/rules/eval.rs` when another rule already explains the same root cause. Cross-layer rows (Prefill → Under-batching) must sit in that table; min-layer alone will drop the higher layer and make the row a no-op.
4. Wire per-window detection in `eval_window_rules` and report assembly (significance, layer, display) in `build_report_from_eval` in `src/engine/rules/eval.rs`. `build_report_for_windows` is the entry; it is not enough by itself. Significance is at least 3 evaluable non-idle windows and ≥ 25% of those windows (`ENGINE_MIN_PERSISTENT_WINDOWS`, `ENGINE_MIN_WINDOW_PCT`).
5. Smoke test with a mock payload in the rule file. Confirm ranked output, suppression-table dedup, and the confidence label (High / Medium / Low). Then validate against a live vLLM server with a GPU.
6. Output is prescriptive: what to change and how, one line, with units. Fire on evidence of harm (evictions, queue growth, latency inflation) or under-use (low occupancy or efficiency). Never on utilization alone. Enable/Set a feature only on `Some(false)`; unread is not off. A named target that already equals the configured value is a no-op (`already_set_u32`).

Do not add a ninth DAG rule by injecting into `maybe_add_massive_underutilization` (`src/engine/mod.rs`). That path is diagnose-only, after windows evaluation, when the recommendation list is empty. Soft-field under-fed inject lives in `eval.rs` and is also not a new layer.

## Where to start

[Open an issue](https://github.com/jungledesh/profile/issues) with what you run and what is missing, or email [jungledesh@gmail.com](mailto:jungledesh@gmail.com).
