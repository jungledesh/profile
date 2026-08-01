# Architecture

Structure of the `profile` binary: data flow, module boundaries, key types. How to change the code is in [CONTRIBUTING.md](CONTRIBUTING.md). What the tool does and why is in the [README](README.md).

Single Rust binary. `profile diagnose` runs an interactive closed loop: collect, analyze, recommend, wait for the operator to apply the fix, re-collect, compute delta, repeat.

## Data flow

```
CLI flags
  └─ cli (clap) → cli::diagnose::execute
       └─ closed loop (profiler::loop_runner)
            ↓
  collectors::             all I/O; shared 250ms scrape cadence in sampling.rs
    ├─ gpu/           NVML (nvidia.rs) or amdgpu (amd.rs), polled via polling.rs
    ├─ vllm.rs        HTTP /metrics scrape
    ├─ config.rs      vLLM config once at startup (GET /v1/models + env vars)
    └─ host_memory.rs host RAM + container cgroup cap, for kv-offload sizing
         → RawSnapshot (one scrape window, timestamped)
            ↓
  context::
    StaticContext  (once):         ModelArch + GPUModel + VllmConfig + fp8_compiler_available,
                                   resolved from gpu_catalog, model_catalog, gpu_prices
    RuntimeWindow  (per snapshot): wraps one RawSnapshot
    AnalysisInput<'a> = { ctx: &StaticContext, window: &RuntimeWindow }
            ↓
  engine::
    baseline/   roofline.rs + math.rs → PhysicsBaseline
    rules::build_report_for_windows (eval.rs) → Report
                ranked by impact x confidence, deduped by the suppression table
    limiter.rs  → LimiterVerdict ("Capped by ..." when no rule fires)
            ↓
  output::stdout
            ↓
  profiler::  wait for operator → re-collect → delta (delta.rs) → drift check (drift.rs) → loop
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

- `StaticContext` (`context/types.rs`): `model: ModelArch`, `gpu: GPUModel`, `config: VllmConfig`, `fp8_compiler_available`. Built once at startup from the catalogs; re-baselined on config drift.
- `RuntimeWindow` (`context/types.rs`): wraps one `RawSnapshot`. The snapshot carries its own timestamp.
- `AnalysisInput<'a>` (`context/types.rs`): `{ ctx: &StaticContext, window: &RuntimeWindow }`. The engine's input; no copies in the hot loop.
- `RawSnapshot` (`collectors/types.rs`): one scrape window at 250ms cadence, across all collectors, timestamped.
- `PhysicsBaseline` (`engine/baseline/roofline.rs`): decode ceiling as `CeilingEstimate { lower, expected, upper }`; prefill ceiling in prompts/s (`Option`); efficiency and headroom percentages; weight footprint with dtype provenance; KV element width. Physics only, no causality. The struct's doc comments are canonical for the full field list.
- `Recommendation` (`engine/rules/`): rule name, DAG layer (2-6), impact (1-5), confidence, pre-formatted display lines. Ranked by impact x confidence within the winning layer; the layer filter and suppression table enforce one signal per root cause.
- `LimiterVerdict` (`engine/limiter.rs`): the no-issue path. When no rule fires, names the boundary capping a healthy server (capacity, traffic, physics, prefill interference, framework overhead) from run-level aggregates, or reports the ceiling unknown rather than guessing.
- `Report` (`engine/mod.rs`): recommendations plus baseline and skip counts; built by `rules::build_report_for_windows`; input to stdout formatting.
- `window_is_evaluable` / `window_is_idle` (`collectors/types.rs`): shared gates. Evaluable means the window has a positive duration and the metrics endpoint answered. Idle means evaluable with no meaningful traffic. Rules skip idle windows; idle is valid telemetry, not a collection failure.

## Where things go

- **A new rule:** `src/engine/rules/rN_name.rs`, wired into `rules::build_report_for_windows` in `src/engine/rules/eval.rs`. Follow the checklist in [CONTRIBUTING.md](CONTRIBUTING.md).
- **A new metric source:** `src/collectors/`. I/O only; no reasoning.
- **A new GPU or model:** `src/context/gpu_catalog.rs`, `src/context/model_catalog.rs`, `src/context/gpu_prices.json`.
- **Closed-loop behavior:** `src/profiler/` (loop_runner, delta, drift, state). Orchestrates collection; implements no collector logic and no rules.
- **Output formatting:** `src/output/stdout.rs`. Convention: `~` marks values derived from estimated ceilings; measured values carry no tilde.
- **Rule thresholds:** named constants at the top of each rule file in `src/engine/rules/`. Semantics and edge cases are in the [rules documentation](https://jungledesh.github.io/profile/docs.html#rules).
