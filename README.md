# Profile

Profile finds what is limiting your vLLM server, names the cause, and prescribes the flag to change. On one A100 it took Qwen3.6-27B from [31 to 470 tok/s and cut cost from $13.26 to $0.89 per million tokens](#proof).

Single GPU. NVIDIA and AMD.

[Release](https://github.com/jungledesh/profile/releases/latest)
[License](LICENSE)
[GPU](#requirements)

**[Website](https://jungledesh.github.io/profile/index.html)** | **[Docs](https://jungledesh.github.io/profile/docs.html)** | **[Demo](#proof)**



---

## What is Profile

Your server is slower than your hardware allows. Every monitoring tool will show you that it is slow. None will tell you why.

Profile computes the physics ceiling for your GPU and model, measures the live server against it, names the one thing holding it back, and tells you the flag to change. Then you change it, and Profile measures whether it worked.

```text
dashboards:  metrics -------------------> you -> guess
profile:     metrics -> physics ceiling -> cause -> fix -> re-measure
```

---

## Install and run

```bash
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/jungledesh/profile/releases/latest/download/profile-installer.sh | sh
```

```bash
profile diagnose --url http://localhost:8000/metrics --duration 2m
```

That is the whole setup. No agent to deploy, no config file, no calibration run. Nothing leaves the machine.

From source: `cargo install --git https://github.com/jungledesh/profile`

---

## Requirements

- **GPU.** NVIDIA through NVML, or AMD through the amdgpu driver. Profile probes NVIDIA first and falls back to AMD.
- **vLLM** with `/metrics` reachable. Default `http://localhost:8000/metrics`.
- **Live traffic** during the window. An idle server has no waste to find, and Profile says so rather than invent a number.

Launch scope is a single GPU. `--tensor-parallel-size` is accepted and scales the physics ceiling, but KV and weight sharding math stays single-GPU. Multi-GPU is on the [roadmap](#roadmap). Profile still tells you when a model needs tensor parallelism to fit at all.

**Profile CLI flags**

These configure Profile, not vLLM.


| Flag                     | Default                         | Description                                                          |
| ------------------------ | ------------------------------- | -------------------------------------------------------------------- |
| `-u, --url`              | `http://localhost:8000/metrics` | vLLM metrics endpoint                                                |
| `--duration`             | `30s`                           | Sampling window. Minimum `30s`, maximum `30m`. Units are `s` or `m`. |
| `-m, --max-num-seqs`     | Prompted if absent              | Pass to skip the prompt. Read from `/metrics` when available.        |
| `--tensor-parallel-size` | Unset                           | TP degree used in the ceiling math                                   |
| `--cost-per-hour`        | Catalog estimate                | GPU cost in USD/hr                                                   |
| `-v`                     | Off                             | Show rules that did not fire, and the physics limits                 |




---

## What you get

The state of the server, the one thing wrong with it, and the fix.

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

`decode_eff ~0.9%` is the gap against the hardware ceiling. `Cause:` is the evidence and the threshold crossed. `Fix:` is what to change, split by whether it costs you throughput to apply.

Every value is measured or marked. A dash means Profile could not read it. An `(est)` means the number came from the physics model, not the server. A tilde marks a value derived from an estimated ceiling, and measured values carry none. Gaps are never filled with guesses.



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

  Config changed. Baseline reset.

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
  Decode eff. -0.4pp

ECONOMICS:
  Cost/1M output tok  $1.36 → $1.52 (est)  worse
```

A tool that only reports improvements cannot be trusted when it reports one.

Four things the loop will not do to you:

- **Repeat itself.** When the same cause returns, Profile releases the issues it was holding behind it. The next hypothesis is already on screen.
- **Dead-end you.** A rule with no fix left releases what it suppressed immediately, without waiting for another round.
- **Tell you to do what you already did.** Fixes are checked against your running config. No suggestion to enable a flag that is on, or to set a value you already exceed.
- **Ping-pong you.** When two rules alternate and pull the same knob in opposite directions, Profile detects the cycle, names the bracket it has tried, and suggests the midpoint. It offers this at most three times, then names the wall instead of guessing again.

**Scale out.** Eventually no flag helps. Profile says the scheduler is at its cap with the pool full, that no config change helps, and that the next move is a replica. That is the answer, not a failure.

---

## How it works

### The ceiling

Physics gives two hard limits. Profile derives both from your GPU and model.

**Prefill** is compute-bound:

```text
prefill_ceiling_tps = (peak_flops x 1e12) / (2 x params x seq_len + attn_coeff x layers x seq_len^2)
```

The first term is the weight math, linear in sequence length. The second is attention, quadratic in it. That is why the prefill ceiling falls as context grows, and why a linear-only roofline overstates long-context capability.

**Decode** is memory-bandwidth-bound:

```text
decode_ceiling_tps = (peak_bandwidth_gbps x 1e9) / (params x bytes_per_param)
```

Efficiency is measured throughput against the decode ceiling times the ridge batch size, the batch size where decode stops being limited by memory and starts being limited by compute.

Ceilings are coarse upper bounds from published specifications, not a hardware simulation. They are marked `(est)`, and values derived from them carry a tilde. An uncatalogued GPU gets `Hardware ceiling unknown` with the reason, rather than a wrong number.

Cost works differently. Dollars per million output tokens is GPU price divided by measured throughput, so it holds even where the ceiling is uncertain.

### The rule engine

Eight rules watch eight failure modes. On a struggling server, several fire at once. A wall of alerts carries the same information as no alert at all.

Two filters cut them to one. The rules sit in a DAG, a fixed priority ordering where a rule can only be outranked by one above it, never by one below.

```text
                          rules evaluate
                                |
        R2 KV pressure    R5 saturation    R1 under-batching
        R6 prefill-bound  R7 headroom      R4 OOM risk
                                |
                                v
                    mutual exclusivity table
        R4 OOM risk      silences   R2 KV cache pressure
        R4 OOM risk      silences   R2b KV admission backlog
        R6 prefill-bound silences   R1 under-batching
                                |
                                v
                      DAG priority layers
        L2  OOM risk, KV pressure, KV admission backlog     broken
        L3  concurrency saturation                            |
        L4  under-batching                                    |
        L5  low prefix reuse, prefill-bound                   |
        L6  config headroom                                 healthy
                    keep only the highest surviving layer
                                |
                                v
                     rank by impact x confidence
                                |
                    +-----------+------------+
                    |                        |
                 primary                   held
             one cause, shown       released when the same
                                     cause fires again
```

**Mutual exclusivity** removes symptoms another cause already explains. If the model does not fit in VRAM, your KV cache is under pressure because of that. Telling you to shrink the KV pool would be treating a symptom.

**The DAG layers** encode one rule: fix what is broken before tuning what is healthy. A tuning suggestion can never outrank an active bottleneck, because it sits structurally below one.

**Nothing is discarded.** Losing rules are held and released when the same cause fires again, so you get the next hypothesis without ever seeing five at once.

Rules fire on evidence of harm, never on heat. A server at 95% KV cache with no evictions and no queue is healthy and busy, and Profile stays quiet.



**The eight rules**


| Rule                          | Fires when                                                                                                                               | Prescribes                                                                                                                                                                |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **R1 Under-batching**         | Known GPU: config-relative efficiency under 60% with occupancy under 75% and no backlog. Uncatalogued GPU: occupancy under 25%.          | Batch more requests or raise client concurrency, naming the binding wall: config, compute ridge, or memory                                                                |
| **R2 KV cache pressure**      | KV at or above 88%, **and** either preemptions active or queue backpressure                                                              | Split by cost. Cuts throughput: lower `--max-model-len` or `--max-num-seqs`. Safe: prefix caching, raise `--gpu-memory-utilization`, fp8 KV cache, `--kv-offloading-size` |
| **R2b KV admission backlog**  | Queue ratio at or above 0.30 with KV near full, scheduler not at the seat cap, and estimated free KV below demand                        | Same family, without the eviction signal                                                                                                                                  |
| **R3 Low prefix reuse**       | Prefix caching off: on prompt-token and traffic volume alone. Prefix caching on: hit rate under 35% with at least 20 mean prompt tokens. | `--enable-prefix-caching`, or restructure prompts if already on                                                                                                           |
| **R4 OOM risk**               | Weights overflow VRAM, or weights fit but free VRAM cannot hold one worst-case request                                                   | `--tensor-parallel-size` at the computed minimum, raise `--gpu-memory-utilization`, a smaller model, or the model does not fit on this hardware                           |
| **R5 Concurrency saturation** | Scheduler at `--max-num-seqs`, at least 2 waiting, queue ratio at or above 0.30                                                          | Below 80% KV, raise `--max-num-seqs` to a bounded target. Above it, name the wall or add a replica                                                                        |
| **R6 Prefill-bound**          | Prompt to generation ratio elevated with decode efficiency low. Muted only when TPOT is measured and under 4x its floor.                 | Chunked prefill, `--max-num-batched-tokens`, shorter prompts; at the compute wall, disaggregate or add a replica                                                          |
| **R7 Config headroom**        | `--max-num-seqs` below 90% of the recommended target, with occupancy at or above 50% and at most 1 waiting                               | Raise it, and name what binds the target                                                                                                                                  |


When nothing fires but efficiency is still low, a fallback names the shape of the underuse. When nothing fires at all, Profile names the boundary capping the server: capacity, traffic, physics, prefill interference, or framework overhead. Where the ceiling itself is unknown, it says that instead of naming a boundary it cannot prove. Thresholds and edge cases are in the [rules documentation](https://jungledesh.github.io/profile/docs.html#rules).



---

## Proof

Two runs, same model, different hardware and different starting configs.

**Qwen3.6-27B on an A100-SXM4-80GB.** 15x throughput, 93% lower cost.


|            | Before             | After             |
| ---------- | ------------------ | ----------------- |
| Throughput | 31 tok/s           | 470 tok/s         |
| Cost       | $13.26 / 1M tokens | $0.89 / 1M tokens |


**[Watch the run](https://www.youtube.com/watch?v=XuPPKBteWH0)**



**Qwen3.6-27B on an H100 80GB HBM3.** Seven iterations, 3.9x throughput, 74% lower cost. Every output sample on this page is from this run.


|            | Before                   | After                    |
| ---------- | ------------------------ | ------------------------ |
| Throughput | 163 tok/s                | 631 tok/s                |
| Cost       | $5.09 / 1M output tokens | $1.32 / 1M output tokens |


The H100 run is the more honest picture of a working session. Throughput went 163, 328, 545, 543, 482, 610, 545, 631. Two of those steps were regressions, and Profile labelled both.

Your starting point sets your gain. A server already near its ceiling has nothing to recover, and Profile will tell you that rather than manufacture a recommendation.

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


|                                  | Profile | Dashboards | Kernel profilers | Autotuners  | Simulators  |
| -------------------------------- | ------- | ---------- | ---------------- | ----------- | ----------- |
| Hardware ceiling from physics    | yes     | no         | no               | no          | predicted   |
| Live server, real traffic        | yes     | yes        | yes              | no          | no          |
| Names one root cause             | yes     | no         | no               | no          | no          |
| Prescribes the change            | yes     | no         | no               | config only | config only |
| Measures the delta after the fix | yes     | no         | no               | partial     | no          |
| Cost per million tokens          | yes     | no         | no               | no          | no          |
| No restarts, no synthetic load   | yes     | yes        | yes              | no          | n/a         |


Dashboards: Grafana, Datadog, vLLM `/metrics`. Kernel profilers: Nsight Systems, Nsight Compute. Autotuners: vLLM `auto_tune`, SCOOT. Simulators: Vidur, LLMCompass, GenZ.

**What Profile is not.** Not a dashboard: it reasons rather than reports. Not an autotuner: it does not restart your server or search a config space. Not a kernel profiler: that is Nsight's layer and a different question. Not multi-engine: vLLM only. Not autonomous: you apply the fix, Profile owns the measurement and the memory.

---

## Where this came from

Two launches, and the users who tried them told us what was missing. Not more metrics. An answer.

The shape came from [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch): propose a change, run it, measure, keep or revert. A loop that improves by measuring rather than predicting. We applied it to inference serving and changed who sits in the proposer's seat. Hypotheses come from roofline physics and a rule engine, which are checkable and deterministic, not from a model forming guesses. And you apply the change, not the tool. Autonomy is the roadmap, not the pitch.

The research came last. We found it after the design had settled and it agreed with us, which is a better outcome than finding it first.

---

## Research foundation

Profile is not heuristics. Each part is the field's validated answer to a question it has already studied, and each of those answers has a known weakness Profile is built to survive.

**The ceiling: roofline.** The standard model for LLM inference analysis. The 2024 survey *LLM Inference Unveiled* ([arXiv 2402.16363](https://arxiv.org/abs/2402.16363)) organises the field around it and RooflineBench ([arXiv 2602.11506](https://arxiv.org/abs/2602.11506)) still builds on it. Physics offers compute and bandwidth, and time is the maximum of the two. There is no tighter limit to derive.

*Its weakness:* a raw spec-sheet roofline overestimates. Real servers have a third regime, overhead-bound, where the GPU idles on CPU work. The proven fix is calibration. A fitted overhead constant improved R-squared by roughly 12% and cut error up to 80% on a vLLM server ([NeurIPS 2024 MLforSystems](https://mlforsystems.org/assets/papers/neurips2024/paper28.pdf)), and GenZ ([arXiv 2406.01698](https://arxiv.org/abs/2406.01698)) reaches 5.82% geomean error by multiplying roofline by calibrated efficiency factors. Nobody accurate uses peak specs raw.

*Where Profile stands:* uncalibrated, and it says so. Ceilings are marked `(est)` and derived values carry a tilde. An uncatalogued GPU gets no ceiling at all. Cost per million tokens is measured throughput against GPU price, so the dollar figure never inherits the ceiling's error. Calibration is on the roadmap.

**The engine: DAG and mutual exclusivity.** Intel's Top-down Microarchitecture Analysis ([Yasin, ISPASS 2014](https://ieeexplore.ieee.org/document/6844459)) has shipped in VTune and Linux `perf` for a decade. It sorts failure modes into mutually exclusive categories under a hierarchical-safety rule: disregard an inner node unless every node on the path to it is flagged. That is a suppression table. TMA exists because the naive alternative, printing every issue with additive penalties, breaks down when stalls overlap.

*Its weakness:* non-causal misattribution. Counters correlate, they do not establish cause. In one documented case ([arXiv 2412.13207](https://arxiv.org/abs/2412.13207)) TMA reported a region as 44.1% memory-bound and 43.4% core-bound when the real bottleneck was a dependence chain. TMA never catches this, because it reports once and never checks itself.

*Profile's answer:* the loop. The literature's remedy for exactly this failure is perturbation, change one resource and measure the response ([Coz, SOSP 2015](https://arxiv.org/abs/1608.03676)). Profile runs that as a side effect of normal operation. Your applied fix is the perturbation and the re-measure is the response. The loop is not a workflow wrapped around a rule engine, it is what makes the rule engine causal.

**Rules rather than learning.** Rule-based root cause analysis holds up in bounded, rule-defined systems and degrades in sprawling dynamic ones ([arXiv 2408.00803](https://arxiv.org/abs/2408.00803)), which is what pushed the field toward causal graphs and GNNs for microservice meshes. A single vLLM server is the bounded case, and rules have the property learning does not: you can read why.

*The cycle objection:* [Murphy (SIGCOMM 2023)](https://dl.acm.org/doi/10.1145/3603269.3604877) moved from DAGs to Markov Random Fields because a DAG cannot represent cyclic dependencies. That critique targets graph-inference RCA, which propagates blame along edges. Profile's DAG is a priority and suppression ordering and never infers along an edge. Cycles are handled in time by the loop, and the visible symptom, two rules alternating, has an explicit detector with a midpoint escape.

**Not an autotuner.** [SCOOT](https://arxiv.org/abs/2408.04323) (Ant Group, WWW 2025) and SLO-Guard ([arXiv 2604.17627](https://arxiv.org/abs/2604.17627)) search config space with Bayesian optimisation, and [vLLM `auto_tune](https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md)` grid-searches under a latency cap.

*Their weakness:* dozens of server restarts, with crashes used as training data. Not something to run against production. They also emit a config, not a cause.

*Profile:* reads the live server under its own traffic. No restarts, no synthetic load, no crash trials. The trade, that autotuners explore configs nobody tried, is covered by you exploring at your own pace with a cause and a measured delta attached to each step.

**Not a simulator.** [Vidur](https://arxiv.org/abs/2405.05465) (MLSys 2024) found the best LLaMA2-70B config in one CPU-hour against an estimated 42,000 GPU-hours of sweep. [LLMCompass](https://arxiv.org/abs/2312.03134) (ISCA 2024) reports 4.1% error.

*Their weakness:* those figures are author-reported on narrow validation sets, and simulators lag serving features by generations. The Frontier critique ([arXiv 2605.21312](https://arxiv.org/abs/2605.21312)) finds Vidur missing chunked prefill, CUDA graphs, speculative decoding, disaggregation and MoE, with attention-predictor error up to 376% at p95 on modern dynamic workloads. Each also needs per-hardware profiling upfront.

*Profile:* measures the server you have. A new vLLM feature is covered the moment its metrics exist.

---

## Limitations

- **One GPU.** KV and weight sharding math is single-GPU only.
- **vLLM only.** The engine boundary is clean, but SGLang is not built.
- **Ceilings are uncalibrated.** Published specifications overestimate. Every ceiling-derived number is marked.
- **The overhead-bound regime is named, not measured.** Profile can say the GPU is idling on CPU work but cannot quantify it.
- **Unknown GPUs get no ceiling.** Profile reports `Hardware ceiling unknown` with the reason, rather than guessing.
- **No load, no answer.** Idle windows are skipped. There is nothing to diagnose on a server at rest.
- **You apply the fix.** Profile never changes your server.

---

## Roadmap

- [ ] Multi-GPU and tensor parallelism
- [ ] Calibrated ceilings, using a fitted overhead constant
- [ ] A second engine (SGLang)
- [ ] Cluster aggregation across nodes

Beyond that: a server that heals itself. Bottlenecks surfaced by physics, fixes applied without a human waiting to press Enter, traffic moved off a node under KV pressure before latency spikes. That is where this goes. None of it is built, and none of it is safe to build until the physics and the engine are right on one node. That is the work happening now.

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

## License

Apache License 2.0. Copyright 2026 Gagandeep Singh.

Need cluster aggregation, multi-engine support, or a custom hardware catalog? [Open an issue](https://github.com/jungledesh/profile/issues) or email **[jungledesh@gmail.com](mailto:jungledesh@gmail.com)**.