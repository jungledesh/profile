# Profile: research and position

Merged from two independent research passes (2026-07-21): a deep-research run
(5 search angles, ~15 primary sources, each surviving claim verified by 3
adversarial votes against the primary source) and a separate prior-art scan.
Both reached the same verdict independently. Section 1 is the research.
Section 2 is profile's position and defensible wins. References at the end.
This document is the source for the README and docs. Open work is tracked in
deferred.md only.

---

## 1. Research

### 1.1 The problem has three layers; the world treats them separately

1. **Ceiling (physics).** How far from the hardware limit. Roofline territory.
2. **Diagnosis (root cause).** Why, and what to fix. Rule engines, RCA.
3. **Search (find the fix).** Which config closes the gap. Autotuners,
   simulators.

No prior tool ships all three, live, for LLM serving.

### 1.2 Ceiling: roofline is the standard, calibration is the frontier

- The 2024 survey "LLM Inference Unveiled" (arXiv 2402.16363) organizes all of
  LLM inference analysis around the roofline model. RooflineBench (2026, arXiv
  2602.11506) still builds directly on it. Roofline is the first-principles
  model: physics gives exactly two hard limits (FLOPS, bandwidth); time =
  max(compute time, memory time). There is no tighter limit to derive.
- A raw spec-sheet roofline overestimates. Production servers have a third
  regime beyond compute-bound and memory-bandwidth-bound: **overhead-bound**,
  the GPU idle waiting on CPU work (scheduler, kernel launches, tokenization).
- The proven fix is calibration. IBM/Northeastern (NeurIPS 2024 MLforSystems):
  adding a fitted overhead constant to roofline improved R-squared ~12% and cut
  MSE up to 80% on a vLLM server. GenZ (arXiv 2406.01698), the most accurate
  analytical model (max 5.82% geomean error across five GPU platforms), is a
  roofline multiplied by empirically calibrated per-operator efficiency
  factors. Nobody accurate uses peak specs raw.

### 1.3 Diagnosis: DAG + mutual exclusivity has a decade of prior art

- **Intel TMA** (Yasin, ISPASS 2014; ships in VTune and Linux perf today) is
  the direct precedent: all pipeline slots partitioned into four mutually
  exclusive categories, recursively subdivided, with a hierarchical-safety
  property: "a value of an inner node should be disregarded unless nodes on
  the path from the root to that particular node are all flagged." That is a
  suppression table. TMA exists because the naive alternative (print every
  issue with additive penalty sums) fails: stalls overlap, penalties are
  workload-dependent. Scope of the analogy: TMA partitions 100% of pipeline
  slots; profile does not partition. The parallel is hierarchical safety +
  mutual exclusivity. Profile's completeness backstop is different machinery:
  when no rule fires, the limiter names the binding boundary (one "Capped
  by ..." verdict from run aggregates), skipped windows are counted, and an
  unknown GPU degrades to "ceiling unknown" rather than a wrong name.
- **Checklist methods** (Brendan Gregg's USE/RED): profile's rules are
  effectively a USE checklist specialized to GPU + vLLM resources.
- **Rule-based expert systems** (Xu 2008, "Performance Booster"): causes +
  diagnostic rules + recommendation rules. Same shape as profile's engine.
- **Known failure mode**: non-causal misattribution. Counters correlate; they
  do not establish cause. Documented case (Inria, arXiv 2412.13207): TMA
  reported a region 44.1% memory-bound / 43.4% core-bound when the true
  bottleneck was a dependence chain. TMA never discovers such errors because
  it never re-measures. The literature's remedy is **sensitivity analysis**
  (differential profiling): perturb a resource, measure the response, identify
  the bottleneck causally (Gus 2024; Coz, SOSP 2015).
- **RCA literature** (arXiv 2408.00803): rule-based RCA works well in clear,
  bounded, rule-defined scenarios; the maintenance burden appears in sprawling
  dynamic environments (microservice meshes), which motivated learned RCA
  (causal graphs, GNNs, LLM agents) and cyclic models (Murphy, SIGCOMM 2023,
  moved DAG to Markov Random Fields; RADICE 2025 learns causal graphs from
  telemetry). A single vLLM server is the bounded regime where rules win.

### 1.4 Search: autotuners and simulators

- **vLLM auto_tune**: grid search over max-num-seqs x max-num-batched-tokens
  under a latency cap.
- **SCOOT** (Ant Group, in production): Bayesian optimization + random forest;
  works on vLLM and TensorRT-LLM. Requires dozens of trial restarts.
- **SLO-Guard** (2026, arXiv 2604.17627): crash-aware two-phase BO; treats
  OOM/CUDA crashes as training data; optimizes goodput under hard latency and
  memory constraints.
- **Simulators**: Vidur (MLSys 2024, under 9% latency error; Vidur-Search
  found the best LLaMA2-70B config in 1 CPU-hour vs an estimated 42,000
  GPU-hours of sweep), LLMCompass (ISCA 2024, 4.1% error), GenZ. All accuracy
  figures are author self-reported on narrow validation sets. The Frontier
  critique (2026, arXiv 2605.21312): Vidur lacks chunked prefill, CUDA graphs,
  speculative decoding, disaggregated serving, MoE; attention-predictor errors
  up to 376% (p95) on modern dynamic workloads. Simulators also require
  per-hardware profiling upfront.

---

## 2. Profile's position

### 2.1 One product, all three layers, live

Profile combines the validated answer to each layer: a roofline ceiling
(layer 1, the field's standard model), a TMA-style DAG + mutual-exclusivity
rule engine (layer 2, the decade-proven diagnosis structure), and a closed
measure-fix-remeasure loop with the human applying fixes (layer 3, exploration
at operator pace with a cause attached). Roofline calculators predict offline
and diagnose nothing. Autotuners emit a config, not a cause. Expert systems
diagnosed at design time, not on a live GPU. The live combination is the open
field.

### 2.2 The loop is a causal instrument, not just a workflow

The literature's gold standard for causal bottleneck identification is
perturbation: change one resource, measure the response (Gus; Coz). Profile
performs this as a side effect of normal operation: the operator's applied fix
IS the perturbation, and the re-measure step captures the response. TMA cannot
do this (it reports once and never checks itself). Simulators cannot (they
predict instead of measure). This is the structural answer to the rule engine's
known misattribution failure mode.

The loop also acts on a flat response: when the same primary re-fires with no
material improvement after a fix, profile reveals the full bodies of the
issues it had suppressed, under one warning line, so the operator has the next
hypothesis without losing the one-primary discipline on healthy iterations.

### 2.3 Passive diagnosis is a feature

Autotuners are active: SCOOT and SLO-Guard restart the server dozens of times
and use crashes as data points. Disruptive on production. Profile observes the
live server's own metrics under real traffic: no restarts, no synthetic load,
no crash trials. The trade (autotuners explore configs nobody tried) is covered
by the human-in-loop fix step, which explores at operator pace with a cause and
a measured delta attached.

### 2.4 Defensible wins, checked against the code

- **Where simulators break, profile holds.** Simulators lag serving features
  by generations and need upfront profiling. Profile measures the live server;
  new vLLM features are covered the moment their metrics exist.
- **Disaggregation is already a first-class fix.** The prior-art scan
  suggested R6 "should eventually" recommend prefill/decode disaggregation.
  Built: r6_prefill_bound.rs emits "Disaggregate prefill and decode onto
  separate workers" plus "Add a replica to scale out" at the compute-wall
  terminal case, with tests.
- **The cycle critique does not apply as stated.** The DAG-cannot-represent-
  cycles problem (Murphy's MRF motivation) afflicts graph-inference RCA, which
  propagates blame along causal edges. Profile's DAG is a priority +
  suppression ordering; it never infers along edges. Cyclic dependencies are
  handled temporally by the loop, and the cycle symptom (A-B-A rule ping-pong)
  has an explicit detector with a midpoint escape (profiler/state.rs,
  is_oscillating / set_midpoint_suggested).
- **Where dashboards report, profile reasons.** Idle windows are gated out
  (collectors/types.rs, window_is_evaluable / window_is_idle); only active
  traffic is diagnosed; one primary cause per iteration; every recommendation
  is held to its measured delta, with regression labels and drift attribution
  (profiler/loop_runner.rs).
- **Cost is measured, not modeled.** $/1M tokens = GPU price / actual
  measured throughput (engine/baseline/roofline.rs, build_cost_estimate), so
  the cost line is independent of the ceiling and unaffected by ceiling
  calibration status. Only a "cost of the gap" projection (gap size x price)
  would inherit the uncalibrated ceiling; state that when adding one.
- **Honest-uncertainty machinery matches the charter.** (est) labels,
  lower/expected/upper ceiling bands, Observed-over-derived capacity
  preference, "-" for missing metrics, unknown-GPU degradation to "Hardware
  ceiling unknown" rather than a wrong name. Provenance is tracked end to end:
  estimated bounds and unrecognized dtypes are named inline with confidence
  capped, never worded as measurements.

---

## Open work

Tracked exclusively in deferred.md (repo root). This document intentionally
carries no task list.

---

## References

Primary papers (verified against source in the deep-research pass):

- LLM Inference Unveiled: Survey and Roofline Model Insights. arXiv 2402.16363.
- GenZ: analytical LLM inference model, roofline x calibrated efficiency
  factors. arXiv 2406.01698.
- Predicting LLM Inference Latency (hybrid roofline + regression;
  overhead-bound regime). NeurIPS 2024 MLforSystems.
  https://mlforsystems.org/assets/papers/neurips2024/paper28.pdf
- RooflineBench. arXiv 2602.11506.
- Vidur: simulation framework for LLM inference. MLSys 2024, arXiv 2405.05465.
- LLMCompass. ISCA 2024.
- Yasin, A Top-Down Method for Performance Analysis and Counters Architecture.
  ISPASS 2014. (TMA; DAG + mutual exclusivity prior art.)
- Performance Debugging through Microarchitectural Sensitivity and Causality
  Analysis. arXiv 2412.13207. (TMA failure mode; sensitivity analysis / Gus.)
- Coz: causal profiling. SOSP 2015.
- A Comprehensive Survey on Root Cause Analysis in (Micro) Services.
  arXiv 2408.00803.
- Frontier critique of simulator fidelity. arXiv 2605.21312.
- SLO-Guard: Crash-Aware Autotuning for SLO-Constrained LLM Serving. 2026,
  arXiv 2604.17627.
- SCOOT: SLO-Oriented Performance Tuning for LLM Inference Engines. WWW 2025,
  Ant Group.
- Murphy: Performance Diagnosis of Distributed Cloud Applications.
  SIGCOMM 2023. (DAG to MRF for cyclic dependencies.)
- RADICE: Causal Graph Based Root Cause Analysis. 2025.
- Xu: Rule-based automatic software performance diagnosis. WOSP 2008.

Secondary (methods and landscape; blog-grade, do not cite for numbers):

- Brendan Gregg, The USE Method. https://www.brendangregg.com/usemethod.html
- vLLM auto_tune.
  https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md
- Inferbase: roofline estimates vs real vLLM benchmarks.
  https://inferbase.ai/blog/inferbase-vs-vllm-benchmarks
- vLLM RFC #42484: production-boundary measurement.
  https://github.com/vllm-project/vllm/issues/42484
