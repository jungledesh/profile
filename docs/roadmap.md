# Roadmap

Tentative, not exhaustive. Demand reorders this list and adds to it: [tell us](https://github.com/jungledesh/profile/issues) what you run and what is missing. Back to the [README](../README.md).

- [ ] Multi-GPU and tensor parallelism
- [ ] Calibrated ceilings, using a fitted overhead constant
- [ ] Speculative decoding awareness: today Profile withdraws `% of ceiling` when a drafter beats the one-token-per-step roof, rather than print a false number. Next: read vLLM's draft/accept counters and restore the ceiling from measured acceptance, no guessing
- [ ] SLO targets: the throughput vs latency curve, its knee, and your latency budget as the yardstick, not only the physics ceiling
- [ ] More engines: SGLang first, then llama.cpp and other local runtimes
- [ ] Cluster aggregation across nodes
- [ ] OTLP export, so findings land in the observability stack you already run (Grafana, Datadog)

## The north star

Inference servers that tune themselves. No engineer babysitting configs, no guessing why p99 spiked: the server knows its ceiling, measures the gap, and iterates until nothing is left to fix. Bottlenecks surfaced by physics, fixes applied with no one waiting to press Enter, traffic moved off a node under KV pressure before latency spikes.

Profile today keeps you in the seat: you apply the change, it measures, tracks causality, recommends the next step. Profile v3: the human steps out.

None of it is safe to build until the physics and the engine are right on one node. That is the work happening now.
