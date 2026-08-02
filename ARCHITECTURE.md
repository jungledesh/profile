# Architecture

Structure of the `profile` binary: data flow, module boundaries, key types. How to change the code, and the merge gate (`fmt`, `clippy`, `audit`, `deny`, `test`, plus OSV and Semgrep in CI), is in [CONTRIBUTING.md](CONTRIBUTING.md). What the tool does and why is in the [README](README.md).

Single Rust binary. `profile diagnose` runs an interactive closed loop: collect, analyze, recommend, wait for the operator to apply the fix, re-collect, compute delta, repeat.

## Data flow

```text
CLI flags
  └─ cli (clap) → cli::diagnose::execute
       │
       ├─ profiler::run_diagnose
       │    collectors::          all I/O; 250ms cadence inside each collection window (sampling.rs)
       │      ├─ gpu/             NVML (nvidia.rs) or amdgpu (amd.rs), polled via polling.rs
       │      ├─ vllm.rs          HTTP /metrics scrape
       │      └─ host_memory.rs   host RAM + container cgroup cap, for kv-offload sizing
       │           → one RawSnapshot per collection window (2s or 10s; many 250ms samples inside)
       │    aggregate_windows → run-level RawSnapshot (DiagnoseResult.snapshot)
       │    config::build_config(snapshot + CLI; best-effort GET /v1/models and /info)
       │    StaticContext from catalogs + config
       │
       ├─ engine::build_report_for_diagnose(windows, aggregate AnalysisInput)
       │    wraps rules::build_report_for_windows (eval.rs); may add post-DAG recommendations (e.g. MU)
       │    baseline/ + limiter.rs feed the Report
       │
       ├─ output::stdout  (initial report)
       │
       └─ profiler::loop_runner  only if any_evaluable, not all_idle, and
            n_eval >= ENGINE_MIN_PERSISTENT_WINDOWS
            wait → re-run_diagnose → delta → drift → loop
```

## Module responsibilities

| Module            | Owns                                                                    | Never touches          |
| ----------------- | ----------------------------------------------------------------------- | ---------------------- |
| `src/collectors/` | All I/O: GPU, vLLM metrics, config, host memory; the scrape cadence     | No reasoning, no rules |
| `src/context/`    | StaticContext + RuntimeWindow from raw collector output; the catalogs   | No I/O, no rules       |
| `src/engine/`     | All reasoning: baseline, rules, limiter                                 | No I/O, no interaction |
| `src/profiler/`   | Orchestration: loop, state, delta, drift detection; drives collection   | No rule or collector logic |
| `src/cli/`        | Arg parsing, GPU assignment, dispatch to profiler                       | No reasoning, no rules |
| `src/output/`     | Emit bus: stdout                                                        | No reasoning           |

`engine/` never imports from `cli/`. Engine is deterministic; CLI is interactive.

## Key types

- `StaticContext` (`context/types.rs`): `model: ModelArch`, `gpu: GPUModel`, `config: VllmConfig`, `fp8_compiler_available`. Built once per `run_diagnose` from the catalogs; re-baselined on config drift.
- `RuntimeWindow` (`context/types.rs`): wraps one per-window `RawSnapshot`. The snapshot carries its own timestamp.
- `AnalysisInput<'a>` (`context/types.rs`): `{ ctx: &StaticContext, window: &RuntimeWindow }`. The engine's input; no copies in the hot loop.
- `RawSnapshot` (`collectors/types.rs`): result of one collection window. Collectors sample at 250ms inside the window; `run_diagnose` keeps one `RawSnapshot` per window and aggregates them into the run-level reporting snapshot.
- `PhysicsBaseline` (`engine/baseline/roofline.rs`): decode ceiling as `CeilingEstimate { lower, expected, upper }`; prefill ceiling in prompts/s (`Option`); efficiency and headroom percentages; weight footprint with dtype provenance; KV element width. Physics only, no causality. The struct's doc comments are canonical for the full field list.
- `Recommendation` (`engine/rules/`): rule name, DAG layer (2-6), impact (1-5), confidence, pre-formatted display lines, `terminal` (no server-local knob left). Ranked by impact x confidence within the winning layer; the layer filter and suppression table enforce one signal per root cause.
- `LimiterVerdict` (`engine/limiter.rs`): the no-issue path. When no rule fires, names the boundary capping a healthy server (capacity, traffic, physics, prefill interference, framework overhead) from run-level aggregates, or reports the ceiling unknown rather than guessing.
- `Report` (`engine/mod.rs`): recommendations plus baseline and skip counts; produced by `engine::build_report_for_diagnose`, which wraps `rules::build_report_for_windows` and may append post-DAG recommendations; input to stdout formatting.
- `window_is_evaluable` / `window_is_idle` (`collectors/types.rs`): shared gates. Evaluable means the window has a positive duration and the metrics endpoint answered. Idle means evaluable with no meaningful traffic. Rules skip idle windows; idle is valid telemetry, not a collection failure.

## Where things go

- **A new rule:** `src/engine/rules/rN_name.rs`, wired into `rules::build_report_for_windows` in `src/engine/rules/eval.rs`. Follow the checklist in [CONTRIBUTING.md](CONTRIBUTING.md).
- **A new metric source:** `src/collectors/`. I/O only; no reasoning.
- **A new GPU or model:** `src/context/gpu_catalog.rs`, `src/context/model_catalog.rs`, `src/context/gpu_prices.json`.
- **Closed-loop behavior:** `src/profiler/` (loop_runner, delta, drift, state). Orchestrates collection; implements no collector logic and no rules.
- **Config enrichment:** `src/collectors/config.rs` (`build_config`). Runs after collection inside `run_diagnose`; snapshot fields + CLI, then best-effort `/v1/models`, `/info`, and `/server_info`. Scheduler knobs (`enable_chunked_prefill`, `max_num_batched_tokens`) are filled from those JSON/text endpoints when Prometheus omits them (modern vLLM keeps them on `SchedulerConfig`, not `cache_config_info`).
- **Output formatting:** `src/output/stdout.rs`. Convention: `~` marks values derived from estimated ceilings; measured values carry no tilde.
- **Rule thresholds:** named constants at the top of each rule file in `src/engine/rules/`. Semantics and edge cases are in the [rules documentation](https://jungledesh.github.io/profile/docs.html#rules).
