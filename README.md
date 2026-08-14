<div align="center">

# Profile

**Inference diagnostics for production vLLM servers.**

*Less words. Less noise. More signal. More value.*

[Install](#install) · [What is Profile](#what-is-profile) · [The engine](#the-engine) · [Proof](#proof) · [Docs](#documentation) · [Website](https://jungledesh.github.io/profile/index.html)

</div>

---

```text
  Throughput          31 → 470 tok/s          15x
  Cost/1M output tok  $13.26 → $0.89 (est)    93% lower
```

- One A100. Same model, same hardware, different flags.
- Profile named each bottleneck, gave the flag, measured the delta after every change.
- Regressions included. [Both runs below.](#proof)

```text
✓ Single binary      ✓ No agent to deploy
✓ No config file     ✓ Nothing leaves the machine
```

---

## What is Profile

A new kind of tool. Not monitoring, not autotuning: a **diagnostic loop**.

```text
dashboards:  metrics ---------------------------> you -> guess
profile:     metrics -> physics ceiling -> cause -> fix -> re-measure
```

- **The question it answers:** is your GPU serving as fast as physics allows, and if not, what exactly do you change?
- **Ceiling.** Computes the fastest your GPU can serve your model, from memory bandwidth and FLOPs. Physics, not vibes.
- **Cause.** Compares the live server against that ceiling and names the one cause holding it back.
- **Fix.** Gives you the exact vLLM flag. You apply it. Profile never touches your server.
- **Proof.** Re-measures, prints the delta, labels regressions `worse`.

```text
✗ Not a dashboard      it reasons, not just reports
✗ Not an autotuner     no restarts, no synthetic load
✗ Not a simulator      reads the server you actually run
```

First of its kind. [Where it sits among every neighbouring tool.](docs/positioning.md)

---

## Install

```bash
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/jungledesh/profile/releases/latest/download/profile-installer.sh | sh
```

```bash
profile diagnose --url http://localhost:8000/metrics --duration 2m
```

That is the whole setup. No calibration run, no restart of your server.

- **Needs:** one GPU (NVIDIA via NVML, or AMD via amdgpu), vLLM with `/metrics` reachable, live traffic.
- **Idle server?** No waste to find; Profile says so instead of inventing a number. Drive load with `vllm bench serve`.
- **No curl-pipe?** [Releases page](https://github.com/jungledesh/profile/releases/latest), or `cargo install --git https://github.com/jungledesh/profile`.
- **Scope:** single GPU at launch; TP > 1 refused ([roadmap](docs/roadmap.md)). Flags: [docs/workflow.md](docs/workflow.md).

---

## The engine

The center of Profile. Deterministic. Precise.

<p align="center">
  <img src="docs/assets/rule-engine.svg" width="880" alt="Profile's rule engine: eight rules on DAG priority layers. Mutual exclusivity removes explained symptoms; highest surviving layer wins; one primary shown, losers held.">
</p>

- **Eight rules, eight failure modes.** On a struggling server, several fire at once.
- **Mutual exclusivity** removes symptoms another cause already explains. Weights overflowing VRAM? Then KV pressure is a symptom, and treating it would be malpractice.
- **Priority DAG:** fix what is broken before tuning what is healthy. A tuning tip can never outrank an active bottleneck.
- **One primary survives.** A wall of alerts carries the same information as no alert.
- **Nothing is discarded.** Losing rules are held, and surface exactly when they become the next hypothesis.
- **Evidence of harm, never heat.** 95% KV with no evictions and no queue is a healthy, busy server. Profile stays quiet.

The full machinery: [docs/engine.md](docs/engine.md).

---

## The loop

1. **Diagnose.** One cause, its evidence, its threshold, the flags to change.
2. **Apply.** You restart vLLM with the flag. Profile waits, reconnects on its own.
3. **Delta.** Before → after on every metric that matters. Regressions labelled, not buried.

Real output, shortened (H100 run; full blocks in [docs/workflow.md](docs/workflow.md)):

```text
|PROFILE v2.1.4 [Qwen3.6-27B] [NVIDIA H100 80GB HBM3]                  |
|GPU =>    decode_eff ~0.9% | $5.09/1M output tok (est) | vRAM 74/80GB |
|REQUESTS  run 14 (4.0%) | wait 4 | max 345                            |
|CACHE     kv_cache 93.1% avg (100.0% peak)                            |
|                                                                      |
|[!] KV Cache Pressure          Seen in 92% of windows                 |
|    Cause: KV cache 94% avg, 100% peak (threshold: 88%).              |
|           4 requests queued on KV admission.                         |
|    Fix:   • Lower --max-model-len (current: 262144). Observed avg    |
|             13.9k tokens per request.                                |
|    Expected: Wait queue drains, TTFT recovers.                       |
|    Confidence: High                                                  |
```

```text
Measuring delta...

  Throughput          163 → 328 tok/s
  TTFT                8720 → 495ms (p95 19185 → 950ms)
  Cost/1M output tok  $5.09 → $2.53 (est)
```

- A dash means Profile could not read it.
- `(est)` means the physics model, not the server.
- A tilde marks a value derived from an estimated ceiling.
- Gaps are never filled with guesses.

---

## Proof

```text
                                                                 tok/s
A100   before  |██ 31
       after   |███████████████████████████████ 470                 15x
H100   before  |███████████ 163
       after   |██████████████████████████████████████████ 631     3.9x
```

- **A100-SXM4-80GB:** 15x throughput, 93% lower cost, $13.26 to $0.89 per 1M tokens. [Watch the run.](https://www.youtube.com/watch?v=XuPPKBteWH0)
- **H100 80GB HBM3:** seven iterations, 3.9x, 74% lower cost.
- The H100 path: 163, 328, 545, 543, 482, 610, 545, 631 tok/s. Two steps regressed. Profile labelled both.
- That honesty is why the wins are believable.
- A server already near its ceiling has nothing to recover, and Profile tells you that instead of manufacturing a recommendation.

---

## Documentation

We did the hard part: a core engine that is precise and deterministic. These pages show the work.

- **[Workflow](docs/workflow.md)**
  Every CLI flag, every line of output decoded, the loop step by step.
  Read this and you can run a full optimization session in one sitting.

- **[Engine](docs/engine.md)**
  The ceiling math from first principles, all eight rules with their exact thresholds, suppression and ranking.
  No black box: every threshold that can silence or fire a rule is on this page.

- **[Research](docs/research.md)**
  Profile is not heuristics. The ceiling is a roofline, the field's standard. The engine descends from Intel's Top-down analysis, a decade in `perf` and VTune. The loop is Coz-style causal perturbation, run as a side effect of normal use.
  We also list every known weakness of each method ourselves, with citations, before you find them.

- **[Positioning](docs/positioning.md)**
  The full serving stack, and a straight comparison against dashboards, kernel profilers, autotuners, and simulators.
  Ends with the only honest column that matters: who measures whether the fix worked.

- **[Limitations](docs/limitations.md)**
  Every boundary, stated plainly, with the reason it exists.
  Shorter than you fear, and nothing hidden in it.

- **[Roadmap](docs/roadmap.md)**
  Multi-GPU, calibrated ceilings, more engines, and the end state: a server that heals itself.
  Demand reorders it: [tell us what you run](https://github.com/jungledesh/profile/issues).

Deep reference on the [website](https://jungledesh.github.io/profile/docs.html): rule thresholds and edge cases, metric sources, the math, the GPU catalog, and engine design.

---

## Principles

The tool follows these rules.

- Every number is measured or marked `(est)`. A missing metric prints `-`. Nothing is invented.
- One cause at a time. Eight rules, one primary.
- Regressions are named, never buried. A tool that only reports wins cannot be trusted when it reports one.
- Every character earns its place. If it does not help you act, it is not there.
- No jargon. We write to help, not to confuse.
- Crafted, not just engineered.

---

## Contributing, license, contact

- **Contributing:** new rules, engine ports, catalog entries, bug reports from real servers. Code map: [ARCHITECTURE.md](ARCHITECTURE.md). Build, merge gate, rule checklist: [CONTRIBUTING.md](CONTRIBUTING.md).
- **License:** [Apache 2.0](LICENSE). Copyright 2026 Gagandeep Singh.
- **Contact:** need cluster aggregation, multi-engine support, or a custom hardware catalog? [Open an issue](https://github.com/jungledesh/profile/issues) or email [jungledesh@gmail.com](mailto:jungledesh@gmail.com).

---

<div align="center">

**We only show truth.**

*The end state: servers that heal themselves, bottlenecks surfaced by physics, no human in the loop.*

*Until then: close the gap between what you pay for and what your hardware delivers.*

</div>
