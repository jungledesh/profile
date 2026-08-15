# Research foundation

What each design choice stands on, with citations, and every known weakness named before you find it. Back to the [README](../README.md).

---

## Where this came from

Two launches, and the users who tried them told us what was missing. Not more metrics. An answer.

The shape came from Andrej Karpathy's autoresearch [1]: propose a change, run it, measure, keep or revert. We applied it to inference serving and changed who sits in the proposer's seat. Hypotheses come from roofline physics and a deterministic rule engine, which are checkable, not from a model forming guesses. And you apply the change, not the tool. Autonomy is the roadmap, not the pitch.

The research came last. We found it after the design had settled, and it agreed with us. That is a better outcome than finding it first.

---

## The argument, in short

Profile is not heuristics. Each part is the field's validated answer to a question it has already studied, and each answer has a known weakness Profile is built to survive.

- **The ceiling** is a roofline model, the standard for LLM inference analysis. Its weakness: raw spec sheets overestimate. So every ceiling is marked `(est)`, and calibration is on the roadmap.
- **The rule engine** uses mutual exclusivity under a priority DAG, the structure Intel's Top-down analysis has shipped in `perf` and VTune for a decade. Its weakness: non-causal misattribution. The loop is the remedy: your applied fix is the perturbation, the re-measure is the check.
- **Rules beat learned models** here because a single vLLM server is the bounded case where rules hold up, and you can read why one fired.
- **Autotuners and simulators** answer adjacent questions, at the price of restarts, synthetic load, or lagging real serving features. Profile reads the live server you already run.

---

## The full argument

**The ceiling: roofline.** The standard model for LLM inference analysis. The 2024 survey *LLM Inference Unveiled* [2] organises the field around it, and RooflineBench [3] still builds on it. On a single accelerator, the roofline models two limits, compute and bandwidth; time is the maximum of the two, and Profile takes the binding one. Distributed serving also faces interconnect bandwidth and latency [4].

*Its weakness:* a raw spec-sheet roofline overestimates. Real servers have a third regime, overhead-bound, where the GPU idles on CPU work. The proven fix is calibration: [5] combines a Roofline Model with regression models trained on historical data to capture runtime overhead, reporting MSE reductions of up to 80% on vLLM and 61% on Triton, and GenZ reaches 5.82% geomean error among the platforms it evaluated with calibrated efficiency factors [4]. Raw peak specifications can overestimate production performance.

*Where Profile stands:* uncalibrated, and it says so. Ceilings are marked `(est)`, derived values carry a tilde, and an uncatalogued GPU or model gets no ceiling at all. Cost per million output tokens is `cost_per_hr × 1e6 / (tok/s × 3600)`, so the dollar figure never inherits the ceiling's error. Calibration is on the roadmap.

**The engine: DAG and mutual exclusivity.** Intel's Top-down Microarchitecture Analysis [6] has shipped in VTune and Linux `perf` for a decade: mutually exclusive failure categories under a hierarchical-safety rule, which is a suppression table. TMA exists because printing every issue at once breaks down when stalls overlap.

*Its weakness:* non-causal misattribution. Counters correlate, they do not establish cause. In one documented case [7] TMA reported a region as 44.1% memory-bound and 43.4% core-bound when the real bottleneck was a dependence chain. TMA never catches this, because it reports once and never checks itself.

*Profile's answer:* the loop. The literature's remedy for this failure is perturbation: change one resource and measure the response (Coz [8]). Profile runs that as a side effect of normal operation. Your applied fix is the perturbation, the re-measure is the response. The loop is what makes the rule engine causal. Causal within limits: the delta attributes to everything changed in that restart, under the traffic that ran. Profile detects config drift and never credits a traffic shift as a fix.

**Rules rather than learning.** Rule-based root cause analysis holds up in bounded, rule-defined systems and degrades in sprawling dynamic ones [9]. A single vLLM server is the bounded case, and rules have the property learning does not: you can read why.

*The cycle objection:* Murphy [10] left DAGs for Markov Random Fields because a DAG cannot represent cyclic dependencies. That critique targets graph-inference RCA, which propagates blame along edges. Profile's DAG is a priority and suppression ordering and never infers along an edge. Cycles are handled in time by the loop, and the visible symptom, two rules alternating, has an explicit detector with a midpoint escape.

**Not an autotuner.** SCOOT [11] and SLO-Guard [12] search config space with Bayesian optimisation: SCOOT restarts the server across many trials and learns hidden crash constraints; SLO-Guard treats crashes as first-class training observations. vLLM `auto_tune` [13] grid-searches under a latency cap and also needs repeated restarts. Not something to run against production, and they emit a config, not a cause.

*Profile:* reads the live server under its own traffic. It does not restart your process or inject load, and it does not use crashes as training data. Autotuners explore configs nobody tried; you explore at your own pace, with a cause and a measured delta attached to each step.

**Not a simulator.** Vidur [14] found the best LLaMA2-70B config in one CPU-hour against an estimated 42,000 GPU-hours of sweep, and LLMCompass [15] reports 4.1% error. But those figures are author-reported on narrow validation sets, and simulators lag serving features by generations: the Frontier critique [16] finds Vidur missing chunked prefill, CUDA graphs, speculative decoding, disaggregation and MoE, with attention-predictor error up to 376% at p95. Each also needs per-hardware profiling upfront.

*Profile:* measures the server you have. A new vLLM feature is covered the moment its metrics exist.

---

## Sources

1. Karpathy, *autoresearch*. [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)
2. *LLM Inference Unveiled: Survey and Roofline Model Insights*, 2024. [arXiv:2402.16363](https://arxiv.org/abs/2402.16363)
3. *RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis*, 2026. [arXiv:2602.11506](https://arxiv.org/abs/2602.11506)
4. *Demystifying AI Platform Design for Distributed Inference of Next-Generation LLM models*, 2024. [arXiv:2406.01698](https://arxiv.org/abs/2406.01698)
5. *Predicting LLM Inference Latency: A Roofline-Driven ML Method*, NeurIPS 2024 ML for Systems. [paper28](https://mlforsystems.org/assets/papers/neurips2024/paper28.pdf)
6. Yasin, *A Top-Down Method for Performance Analysis and Counters Architecture*, ISPASS 2014. [IEEE 6844459](https://ieeexplore.ieee.org/document/6844459)
7. *Performance Debugging through Microarchitectural Sensitivity and Causality Analysis*, 2024. [arXiv:2412.13207](https://arxiv.org/abs/2412.13207)
8. Curtsinger and Berger, *Coz: Finding Code that Counts with Causal Profiling*, SOSP 2015. [arXiv:1608.03676](https://arxiv.org/abs/1608.03676)
9. *A Comprehensive Survey on Root Cause Analysis in (Micro) Services: Methodologies, Challenges, and Trends*, 2024. [arXiv:2408.00803](https://arxiv.org/abs/2408.00803)
10. Harsh et al., *Murphy: Performance Diagnosis of Distributed Cloud Applications*, SIGCOMM 2023. [doi:10.1145/3603269.3604877](https://dl.acm.org/doi/10.1145/3603269.3604877)
11. *SCOOT: SLO-Oriented Performance Tuning for LLM Inference Engines*, WWW 2025. [arXiv:2408.04323](https://arxiv.org/abs/2408.04323)
12. *SLO-Guard: Crash-Aware, Budget-Consistent Autotuning for SLO-Constrained LLM Serving*, 2026. [arXiv:2604.17627](https://arxiv.org/abs/2604.17627)
13. vLLM *auto_tune*. [benchmarks/auto_tune](https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md)
14. *Vidur: A Large-Scale Simulation Framework for LLM Inference*, MLSys 2024. [arXiv:2405.05465](https://arxiv.org/abs/2405.05465)
15. *A Hardware Evaluation Framework for Large Language Model Inference*, ISCA 2024. [arXiv:2312.03134](https://arxiv.org/abs/2312.03134)
16. *Frontier: Towards Comprehensive and Accurate LLM Inference Simulation*, 2026. [arXiv:2605.21312](https://arxiv.org/abs/2605.21312)
