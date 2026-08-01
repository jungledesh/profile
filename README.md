

# Profile

**Inference diagnostics for production vLLM servers.**

*Less words. Less noise. More signal. More value.*

[Website](https://jungledesh.github.io/profile/index.html) · [Docs](https://jungledesh.github.io/profile/docs.html) · [Install](#install-and-run) · [Proof](#proof) · [Roadmap](#roadmap) · [Contributing](#contributing)



---

**Are you getting what your hardware is capable of?** Profile finds what is keeping your GPU below the fastest it could serve your model, its physics ceiling, gives you the flag to change, and measures whether the fix worked.

On one A100, that loop took Qwen3.6-27B from **31 to 470 tok/s** and cut cost from **$13.26 to $0.89 per million tokens**. On an H100, **163 to 631 tok/s** at 74% lower cost. [Both runs, regressions included.](#proof)

```text
✓ Single binary      ✓ No agent to deploy
✓ No config file     ✓ Nothing leaves the machine
```

One GPU, NVIDIA or AMD. Reads live `/metrics` and NVML or amdgpu.

---



## Install and run

```bash
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/jungledesh/profile/releases/latest/download/profile-installer.sh | sh
```

```bash
profile diagnose --url http://localhost:8000/metrics --duration 2m
```

That is the whole setup. No calibration run, no restart of your server.

Prefer not to pipe curl into sh? Download the binary from the [releases page](https://github.com/jungledesh/profile/releases/latest), or build from source: `cargo install --git https://github.com/jungledesh/profile`

**Requirements**

- **GPU.** NVIDIA through NVML, or AMD through the amdgpu driver. Profile probes NVIDIA first and falls back to AMD.
- **vLLM** with `/metrics` reachable. Default `http://localhost:8000/metrics`.
- **Live traffic** during the window. An idle server has no waste to find, and Profile says so rather than invent a number. Trying Profile without production traffic? Drive load with vLLM's own `vllm bench serve`, or any load generator.

Launch scope is a single GPU. `--tensor-parallel-size` greater than 1 is refused today. Multi-GPU is on the [roadmap](#roadmap). Profile still tells you when a model needs tensor parallelism to fit at all.

**Profile CLI flags** (these configure Profile, not vLLM)  



| Flag                     | Default                         | Description                                                                         |
| ------------------------ | ------------------------------- | ----------------------------------------------------------------------------------- |
| `-u, --url`              | `http://localhost:8000/metrics` | vLLM metrics endpoint                                                               |
| `--duration`             | `30s`                           | Sampling window. Minimum `30s`, maximum `30m`. Units are `s` or `m`.                |
| `-m, --max-num-seqs`     | Prompted if absent              | Pass to skip the prompt. Read from `/metrics` when available.                       |
| `--tensor-parallel-size` | Unset                           | Must be 1 today. Values above 1 are refused until multi-GPU ships                   |
| `--cost-per-hour`        | Catalog estimate                | GPU cost in USD/hr                                                                  |
| `-v`                     | Off                             | Show rules that did not fire, physics limits, and expanded GPU/latency/CACHE detail |




---



## What is Profile

> **TL;DR.** Profile computes the physics ceiling for your GPU and model, compares the live server against it, names the single highest-priority cause, and gives you the exact vLLM flag to change. Then it re-measures. Not a dashboard, not an autotuner, not a simulator.

Your server is slower than your hardware allows. Every monitoring tool will show you that it is slow. None will tell you why.

```text
dashboards:  metrics -------------------> you -> guess
profile:     metrics -> physics ceiling -> cause -> fix -> re-measure
```

**For:** engineers running vLLM who want to know whether their GPU is earning its price, and what to change when it is not.

**Not for you if:** you shard across GPUs today (on the [roadmap](#roadmap)), run an engine other than vLLM, or have no traffic to measure. [Limitations](#limitations) lists every boundary.

---



## What you get

The state of the server, the one thing wrong with it, and the fix. Four things to find in the block below:

```text
GPU / vLLM header    where the server stands: efficiency vs ceiling, latency, cache, cost
ISSUES               the one cause that survived ranking, with its evidence and threshold
Fix                  the flags to change, split by whether applying them costs throughput
Expected/Confidence  what should happen next, and how sure Profile is
```

```text
+----------------------------------------------------------------------------------------------------------------------------+
|PROFILE v2.1.4 [Qwen3.6-27B] [NVIDIA H100 80GB HBM3] (2m from 2026-07-31 07:13:29 UTC)                                      |
|                                                                                                                            |
|GPU =>               decode_eff ~0.9% | power 653W | 3.57 J/tok | $5.09/1M output tok (est) | vRAM 74/80GB (peak 75GB)      |
|                     mem_util 56%                                                                                           |
|                                                                                                                            |
|vLLM =>                                                                                                                     |
|REQUESTS             run 14 (4.0%) | wait 4 | max 345                                                                       |
|LATENCY              ttft 8.7s (p95 19.2s) | tpot 97ms (p95 159ms)                                                          |
|CACHE                kv_cache 93.1% avg (100.0% peak) | pfix_cache -                                                        |
|THROUGHPUT           163 tok/s                                                                                              |
|TRAFFIC              qps 0.5 | req_total 112 | gen_total 32368 | preempt/s 0.01 | preempt_total 2                           |
|                                                                                                                            |
|ISSUES:                                                                                                                     |
|                                                                                                                            |
|[!] KV Cache Pressure                                                                                                       |
|    Seen in 92% of windows                                                                                                  |
|    Cause:                                                                                                                  |
|      KV cache 94% avg in fired windows, 100% peak (threshold: 88%).                                                        |
|      4 requests queued on KV admission.                                                                                    |
|                                                                                                                            |
|    Fix:                                                                                                                    |
|    Cuts throughput:                                                                                                        |
|      • Lower --max-model-len (current: 262144). Observed avg 13.9k tokens per request, prompt + generation.                |
|        Some requests are longer than avg; add buffer to it. Requests over the limit are rejected with a 400, not truncated.|
|                                                                                                                            |
|      • Lower --max-num-seqs to reduce KV demand                                                                            |
|                                                                                                                            |
|    Safe to apply:                                                                                                          |
|      • Enable --enable-prefix-caching to share KV blocks across identical prompt prefixes                                  |
|      • Raise --gpu-memory-utilization (check vRAM header for avail mem) to expand KV pool                                  |
|      • Switch --kv-cache-dtype fp8 to halve KV memory footprint (affects output quality)                                   |
|      • Set --kv-offloading-size 4 (est) to hold evicted KV in host memory instead of recomputing it                        |
|        Host RAM available: 1953 GiB, container limit 234 GiB.                                                              |
|                                                                                                                            |
|    Expected: Wait queue drains, TTFT recovers once KV pool has capacity.                                                   |
|    Confidence: High                                                                                                        |
+----------------------------------------------------------------------------------------------------------------------------+
```

Every value is measured or marked. A dash means Profile could not read it. An `(est)` means the number came from the physics model, not the server. A tilde marks a value derived from an estimated ceiling. Gaps are never filled with guesses.

---



## Proof

Two runs, same model, different hardware and different starting configs.

```text
                                                            tok/s
A100   before  |██ 31
       after   |███████████████████████████████ 470               15x

H100   before  |███████████ 163
       after   |██████████████████████████████████████████ 631    3.9x
```

**Qwen3.6-27B on an A100-SXM4-80GB.** 15x throughput, 93% lower cost: $13.26 to $0.89 per 1M tokens.

**Qwen3.6-27B on an H100 80GB HBM3.** Seven iterations, 3.9x throughput, 74% lower cost: $5.09 to $1.32 per 1M output tokens. Every output sample on this page is from this run.

[Watch the A100 run](https://www.youtube.com/watch?v=XuPPKBteWH0)

The H100 run is the more honest picture of a working session. Throughput went 163, 328, 545, 543, 482, 610, 545, 631. Two of those steps were regressions, and Profile labelled both.

Your starting point sets your gain. A server already near its ceiling has nothing to recover, and Profile will tell you that rather than manufacture a recommendation.

---



## The loop

**Apply the fix.** Apply everything in the block together in one restart. One flag per restart spends an hour learning what one restart would have told you.

```text
▶  Apply the fix above. Profile re-measures after your change.

Press Enter when done.
```

**Profile measures.** It reconnects when vLLM returns and computes the delta.

```text
Connection restored. Resuming in 5s...

New --max-num-seqs [current: 345]: 170

Measuring delta...

  Config changed.

  Throughput          163 → 328 tok/s
  TTFT                8720 → 495ms (p95 19185 → 950ms)
  TPOT                96.9 → 50.9ms (p95 158.7 → 73.0ms)

ECONOMICS:
  Cost/1M output tok  $5.09 → $2.53 (est)
```

**Iterate.** Fixing one bottleneck usually exposes the next. The path is not always upward, and Profile does not pretend otherwise. Regressions are labelled, not buried:

```text
  Throughput          610 → 545 tok/s  worse
  TTFT                1024 → 6067ms (p95 2407 → 16687ms)  worse
  TPOT                118.4 → 147.0ms (p95 147.7 → 195.9ms)  worse
  Decode eff.         -0.4pp

ECONOMICS:
  Cost/1M output tok  $1.36 → $1.52 (est)  worse
```

A tool that only reports improvements cannot be trusted when it reports one.

Four things the loop will not do to you:

- **Repeat itself.** When the same cause returns without a material improvement, Profile releases the issues it was holding behind it. The next hypothesis is already on screen.
- **Dead-end you.** When no config lever remains, Profile names the wall and points at scale-out instead of inventing another flag.
- **Tell you to do what you already did.** Where Profile can read your running config, it skips levers you already set (prefix caching on, fp8 KV already active, seats already at the target, and the like).
- **Ping-pong you.** When KV pressure and concurrency saturation alternate on `--max-num-seqs`, Profile detects the cycle, names the bracket it has tried, and suggests the midpoint. It offers this at most three times, then names the wall instead of guessing again.

**Scale out.** Eventually no flag helps. Profile says the scheduler is at its cap with the pool full, that no config change helps, and that the next move is a replica. That is the answer, not a failure.

---



## How it works

Profile does three things:

1. **Computes your hardware ceiling.** The fastest your GPU could serve this model, derived from physics.
2. **Finds the bottleneck.** Eight rules cut everything suppressing the server down to one primary cause.
3. **Measures whether your fix worked.** The re-measure is what makes the diagnosis causal.



### The ceiling

Physics gives two hard limits. Profile derives both from your GPU and model.

**Prefill** is compute-bound (prompts/s):

```text
prefill_ceiling = (peak_flops x 1e12)
                / (2 x params x seq_len + attn_coeff x layers x seq_len^2)
```

The first term is the weight math, linear in sequence length. The second is attention, quadratic in it. That is why the prefill ceiling falls as context grows, and why a linear-only roofline overstates long-context capability.

**Decode** is memory-bandwidth-bound:

```text
decode_ceiling_tps = (peak_bandwidth_gbps x 1e9) / (params x bytes_per_param)
```

Efficiency is measured throughput against the decode ceiling times the ridge batch size, the batch size where decode stops being limited by memory and starts being limited by compute.

Ceilings are coarse upper bounds from published specifications, not a hardware simulation. They are marked `(est)`, and values derived from them carry a tilde. An uncatalogued GPU or model gets `Hardware ceiling unknown` with the reason, rather than a wrong number.

Cost works differently. Dollars per million output tokens is `cost_per_hr × 1e6 / (tok/s × 3600)`, so it holds even where the ceiling is uncertain.

### The rule engine

One cause at a time. Eight rules watch eight failure modes, and on a struggling server several fire at once. A wall of alerts carries the same information as no alert at all, so two filters cut them to one: a mutual exclusivity table removes symptoms another cause already explains, and a priority DAG keeps a tuning suggestion from ever outranking an active bottleneck.



**Mutual exclusivity** removes symptoms another cause already explains. If the model weights alone overflow VRAM, your KV cache is under pressure because of that. Telling you to shrink the KV pool would be treating a symptom. A buffer squeeze without weights overflow does not silence R2.

**The DAG layers** encode one rule: fix what is broken before tuning what is healthy. A tuning suggestion sits structurally below an active bottleneck and can never outrank it.

**Nothing is discarded.** Losing rules are held and released when the same cause returns without a material improvement, so you get the next hypothesis without ever seeing five at once.

Rules fire on evidence of harm, never on heat. A server at 95% KV cache with no evictions and no queue is healthy and busy, and Profile stays quiet.

**The eight rules**


| Rule                          | Fires when                                                                                                                                          | Prescribes                                                                                                                                                                |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **R1 Under-batching**         | Known GPU: config-relative efficiency under 60% with occupancy under 75% and no backlog. Uncatalogued GPU: occupancy under 25%.                     | Batch more requests or raise client concurrency, naming the binding wall: config, compute ridge, or memory                                                                |
| **R2 KV cache pressure**      | KV at or above 88%, and either an eviction signal (`num_preemptions_per_sec` > 0.02, or swapped ≥ 2) or queue backpressure (waiting > 2)            | Split by cost. Cuts throughput: lower `--max-model-len` or `--max-num-seqs`. Safe: prefix caching, raise `--gpu-memory-utilization`, fp8 KV cache, `--kv-offloading-size` |
| **R2b KV admission backlog**  | Queue ratio at or above 0.30 with KV near full, scheduler not at the seat cap, and estimated free KV below demand                                   | Same family, without the eviction signal                                                                                                                                  |
| **R3 Low prefix reuse**       | Active traffic (running > 0.75), mean prompt ≥ 20, and qps × mean prompt ≥ 1000. Caching off: fires on that volume. Caching on: hit rate under 35%. | `--enable-prefix-caching`, or restructure prompts if already on                                                                                                           |
| **R4 OOM risk**               | Weights overflow VRAM, or weights fit but free VRAM cannot hold one worst-case request                                                              | `--tensor-parallel-size` at the computed minimum, raise `--gpu-memory-utilization`, a smaller model, or the model does not fit on this hardware                           |
| **R5 Concurrency saturation** | Scheduler at `--max-num-seqs`, at least 2 waiting, queue ratio at or above 0.30                                                                     | Below 80% KV, raise `--max-num-seqs` to a bounded target. Above it, name the wall or add a replica                                                                        |
| **R6 Prefill-bound**          | Prompt/gen ratio ≥ 5 with decode efficiency under 40%. Muted when TPOT is measured and under 4x its floor.                                          | Chunked prefill, `--max-num-batched-tokens`, shorter prompts; at the compute wall, disaggregate or add a replica                                                          |
| **R7 Config headroom**        | `--max-num-seqs` below 90% of the recommended target, with occupancy at or above 50% and at most 1 waiting                                          | Raise it, and name what binds the target                                                                                                                                  |


When nothing fires but efficiency is still low, a fallback names the shape of the underuse. When nothing fires at all, Profile names the boundary capping the server: capacity, traffic, physics, prefill interference, or framework overhead. Where the ceiling itself is unknown, it says that instead of naming a boundary it cannot prove. Full thresholds, confidence scoring, and edge cases are in the [rules documentation](https://jungledesh.github.io/profile/docs.html#rules).

---



## Where Profile sits

```text
Orchestration        NVIDIA Dynamo, Ray Serve            schedules across nodes
Monitoring           Grafana, Datadog, vLLM /metrics     reports what happened
>>> PROFILE          is any of this working on your hardware, config, and traffic
Inference engine     vLLM, SGLang, TensorRT-LLM          serves the requests
Kernels and runtime  CUDA, ROCm, custom kernels          executes the math
Silicon              NVIDIA, AMD, Cerebras, Groq         sets the ceiling
```

Every layer above and below optimises something. None measures whether the result is any good on your machine.


|                                    | Profile | Dashboards | Kernel profilers | Autotuners  | Simulators  |
| ---------------------------------- | ------- | ---------- | ---------------- | ----------- | ----------- |
| Hardware ceiling from physics      | yes     | no         | no               | no          | predicted   |
| Live server, real traffic          | yes     | yes        | yes              | no          | no          |
| Names one root cause               | yes     | no         | no               | no          | no          |
| Prescribes the change              | yes     | no         | no               | config only | config only |
| Measures the delta after the fix   | yes     | no         | no               | partial     | no          |
| Cost per million tokens            | yes     | no         | no               | no          | no          |
| No auto-restart, no synthetic load | yes     | yes        | yes              | no          | n/a         |


Dashboards: Grafana, Datadog, vLLM `/metrics`. Kernel profilers: Nsight Systems, Nsight Compute. Autotuners: vLLM `auto_tune`, SCOOT. Simulators: Vidur, LLMCompass, GenZ.

**What Profile is not.** Not a dashboard: it reasons rather than reports. Not an autotuner: it does not restart your server or search a config space. Not a kernel profiler: that is Nsight's layer and a different question. Not multi-engine: vLLM only. Not autonomous: you apply the fix, Profile owns the measurement and the memory.

---



## Where this came from

Two launches, and the users who tried them told us what was missing. Not more metrics. An answer.

The shape came from [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch): propose a change, run it, measure, keep or revert. We applied it to inference serving and changed who sits in the proposer's seat. Hypotheses come from roofline physics and a deterministic rule engine, which are checkable, not from a model forming guesses. And you apply the change, not the tool. Autonomy is the roadmap, not the pitch.

The research came last. We found it after the design had settled, and it agreed with us. That is a better outcome than finding it first.

---



## Limitations

- **One GPU.** `--tensor-parallel-size` greater than 1 is refused. KV and weight sharding math is single-GPU only.
- **vLLM only.** The engine boundary is clean, but SGLang is not built.
- **Ceilings are uncalibrated.** Published specifications overestimate. Every ceiling-derived number is marked.
- **The overhead-bound regime is named, not measured.** Profile can say the GPU is idling on CPU work but cannot quantify it.
- **Unknown GPU or model gets no ceiling.** Profile reports `Hardware ceiling unknown` with the reason, rather than guessing.
- **No load, no answer.** Idle windows are skipped. There is nothing to diagnose on a server at rest.
- **You apply the fix.** Profile never changes your server.

---



## Roadmap

Tentative, not exhaustive. Demand reorders this list and adds to it: [tell us](https://github.com/jungledesh/profile/issues) what you run and what is missing.

- [ ] Multi-GPU and tensor parallelism
- [ ] Calibrated ceilings, using a fitted overhead constant
- [ ] More engines: SGLang first, then llama.cpp and other local runtimes
- [ ] Cluster aggregation across nodes
- [ ] OTLP export, so findings land in the observability stack you already run (Grafana, Datadog)

Beyond that: a server that heals itself. Bottlenecks surfaced by physics, fixes applied without a human waiting to press Enter, traffic moved off a node under KV pressure before latency spikes. That is where this goes. None of it is built, and none of it is safe to build until the physics and the engine are right on one node. That is the work happening now.

---



## Research foundation

Profile is not heuristics. Each part is the field's validated answer to a question it has already studied, and each answer has a known weakness Profile is built to survive. The ceiling is a roofline model, the standard for LLM inference analysis; its weakness is that raw spec sheets overestimate, so every ceiling is marked `(est)` and calibration is on the roadmap. The rule engine uses mutual exclusivity under a priority DAG, the structure Intel's Top-down analysis has shipped in `perf` and VTune for a decade; its weakness is non-causal misattribution, and the loop is the remedy: your applied fix is the perturbation, the re-measure is the check. Rules beat learned models here because a single vLLM server is the bounded case where rules hold up, and you can read why one fired. Autotuners and simulators answer adjacent questions, at the price of restarts, synthetic load, or lagging real serving features; Profile reads the live server you already run.

**The full argument, with citations**  


**The ceiling: roofline.** The standard model for LLM inference analysis. The 2024 survey *LLM Inference Unveiled* ([arXiv 2402.16363](https://arxiv.org/abs/2402.16363)) organises the field around it, and RooflineBench ([arXiv 2602.11506](https://arxiv.org/abs/2602.11506)) still builds on it. Physics offers compute and bandwidth, and time is the maximum of the two. There is no tighter limit to derive.

*Its weakness:* a raw spec-sheet roofline overestimates. Real servers have a third regime, overhead-bound, where the GPU idles on CPU work. The proven fix is calibration: a fitted overhead constant cut error up to 80% on a vLLM server ([NeurIPS 2024 MLforSystems](https://mlforsystems.org/assets/papers/neurips2024/paper28.pdf)), and GenZ ([arXiv 2406.01698](https://arxiv.org/abs/2406.01698)) reaches 5.82% geomean error with calibrated efficiency factors. Nobody accurate uses peak specs raw.

*Where Profile stands:* uncalibrated, and it says so. Ceilings are marked `(est)`, derived values carry a tilde, and an uncatalogued GPU or model gets no ceiling at all. Cost per million output tokens is `cost_per_hr × 1e6 / (tok/s × 3600)`, so the dollar figure never inherits the ceiling's error. Calibration is on the roadmap.

**The engine: DAG and mutual exclusivity.** Intel's Top-down Microarchitecture Analysis ([Yasin, ISPASS 2014](https://ieeexplore.ieee.org/document/6844459)) has shipped in VTune and Linux `perf` for a decade: mutually exclusive failure categories under a hierarchical-safety rule, which is a suppression table. TMA exists because printing every issue at once breaks down when stalls overlap.

*Its weakness:* non-causal misattribution. Counters correlate, they do not establish cause. In one documented case ([arXiv 2412.13207](https://arxiv.org/abs/2412.13207)) TMA reported a region as 44.1% memory-bound and 43.4% core-bound when the real bottleneck was a dependence chain. TMA never catches this, because it reports once and never checks itself.

*Profile's answer:* the loop. The literature's remedy for this failure is perturbation: change one resource and measure the response ([Coz, SOSP 2015](https://arxiv.org/abs/1608.03676)). Profile runs that as a side effect of normal operation. Your applied fix is the perturbation, the re-measure is the response. The loop is what makes the rule engine causal.

**Rules rather than learning.** Rule-based root cause analysis holds up in bounded, rule-defined systems and degrades in sprawling dynamic ones ([arXiv 2408.00803](https://arxiv.org/abs/2408.00803)). A single vLLM server is the bounded case, and rules have the property learning does not: you can read why.

*The cycle objection:* [Murphy (SIGCOMM 2023)](https://dl.acm.org/doi/10.1145/3603269.3604877) left DAGs for Markov Random Fields because a DAG cannot represent cyclic dependencies. That critique targets graph-inference RCA, which propagates blame along edges. Profile's DAG is a priority and suppression ordering and never infers along an edge. Cycles are handled in time by the loop, and the visible symptom, two rules alternating, has an explicit detector with a midpoint escape.

**Not an autotuner.** [SCOOT](https://arxiv.org/abs/2408.04323) (WWW 2025) and SLO-Guard ([arXiv 2604.17627](https://arxiv.org/abs/2604.17627)) search config space with Bayesian optimisation, and [vLLM](https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md) `auto_tune` grid-searches under a latency cap. All need dozens of restarts, with crashes as training data. Not something to run against production, and they emit a config, not a cause.

*Profile:* reads the live server under its own traffic. It does not restart your process or inject load, and it does not use crashes as training data. Autotuners explore configs nobody tried; you explore at your own pace, with a cause and a measured delta attached to each step.

**Not a simulator.** [Vidur](https://arxiv.org/abs/2405.05465) (MLSys 2024) found the best LLaMA2-70B config in one CPU-hour against an estimated 42,000 GPU-hours of sweep, and [LLMCompass](https://arxiv.org/abs/2312.03134) (ISCA 2024) reports 4.1% error. But those figures are author-reported on narrow validation sets, and simulators lag serving features by generations: the Frontier critique ([arXiv 2605.21312](https://arxiv.org/abs/2605.21312)) finds Vidur missing chunked prefill, CUDA graphs, speculative decoding, disaggregation and MoE, with attention-predictor error up to 376% at p95. Each also needs per-hardware profiling upfront.

*Profile:* measures the server you have. A new vLLM feature is covered the moment its metrics exist.



---



## Principles

A diagnostic tool has nothing but its credibility. Five rules protect it.

- **Correct.** Every diagnosis rests on physics and measured evidence, never assumption.
- **Transparent.** The evidence and the derivation are shown. You can check every claim.
- **Humble.** Where the evidence is insufficient, Profile declines. A dash for a missing metric, `(est)` on a derived value, silence over a guess.
- **Useful, and no harm.** Every step is re-measured. A regression is named plainly, never buried.
- **Legible.** Plain language, every number with its unit, the fix in one line. Every character on screen earns its place. If it does not help you act, it is not there.

---



## Documentation

Start with the Workflow, then Rules. The rest is reference.

- **[Workflow](https://jungledesh.github.io/profile/docs.html#home)**: usage, output walkthrough, flag mapping.
- **[Rules](https://jungledesh.github.io/profile/docs.html#rules)**: thresholds, confidence, edge cases.
- **[Data](https://jungledesh.github.io/profile/docs.html#data)**: metric sources.
- **[Math](https://jungledesh.github.io/profile/docs.html#math)**: the physics behind efficiency.
- **[Catalog](https://jungledesh.github.io/profile/docs.html#catalog)**: GPU bandwidth, FLOPs, and prices.
- **[Limitations](https://jungledesh.github.io/profile/docs.html#limitations)**: where the math is approximate.
- **[Design](https://jungledesh.github.io/profile/docs.html#design)**: engine design philosophy.

---



## Contributing

Accepted: a new rule, an engine port, a GPU catalog entry, or a bug report from a real server.

Start with [ARCHITECTURE.md](ARCHITECTURE.md) for where code lives and what owns what. [CONTRIBUTING.md](CONTRIBUTING.md) has the build, the merge gate, and the checklist for adding a rule. If the entry point is unclear, [open an issue](https://github.com/jungledesh/profile/issues) with what you run and what is missing.

---



## License

Apache License 2.0. Copyright 2026 Gagandeep Singh.

Need cluster aggregation, multi-engine support, or a custom hardware catalog? [Open an issue](https://github.com/jungledesh/profile/issues) or email **[jungledesh@gmail.com](mailto:jungledesh@gmail.com)**.