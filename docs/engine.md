# The engine

The ceiling math, the eight rules, and what decides which one you see. Back to the [README](../README.md).

Profile does three things:

1. **Computes your hardware ceiling.** The fastest your GPU could serve this model, derived from physics.
2. **Finds the bottleneck.** Eight rules cut everything suppressing the server down to one primary cause.
3. **Measures whether your fix worked.** The re-measure is what makes the diagnosis causal.

---

## The ceiling

Physics gives two hard limits. Profile derives both from your GPU and model.

**Prefill** is compute-bound (prompts/s):

```text
prefill_ceiling = (peak_flops x tp x 1e12)
                / (2 x params x seq_len + attn_coeff x layers x seq_len^2)
```

- The first term is the weight math, linear in sequence length.
- The second is attention, quadratic in it.
- That is why the prefill ceiling falls as context grows, and why a linear-only roofline overstates long-context capability.
- TP scales FLOPs the same way decode scales bandwidth; Profile refuses TP > 1 at launch today.

**Decode** is memory-bandwidth-bound (TP scales bandwidth; runtime TP is 1 at launch):

```text
decode_ceiling_tps = (peak_bandwidth_gbps x tp x 1e9) / (params x bytes_per_param)
```

- Efficiency is measured throughput against the decode ceiling times the ridge batch size: the batch size where decode stops being limited by memory and starts being limited by compute.
- Ceilings are coarse upper bounds from published specifications, not a hardware simulation. They are marked `(est)`, and values derived from them carry a tilde.
- An uncatalogued GPU or model gets `Hardware ceiling unknown` with the reason, rather than a wrong number.

**Cost works differently.** Dollars per million output tokens is `cost_per_hr × 1e6 / (tok/s × 3600)` when enough completions cover mean running (turnover gate); otherwise the cost line is omitted. That keeps the dollar figure independent of the ceiling.

---

## The rule engine

One cause at a time. Eight rules watch eight failure modes, and on a struggling server several fire at once. A wall of alerts carries the same information as no alert at all, so two filters cut them to one.

<p align="center">
  <img src="assets/rule-engine.svg" width="880" alt="Profile's rule engine: eight rules on DAG priority layers L2 to L6. Mutual exclusivity: R4 weights-alone overflow silences R2 and R2b; R6 silences R1 when the box is pressed (not under soft field). Highest surviving layer wins, ranked by impact x confidence, one primary shown, losers held.">
</p>

- **Mutual exclusivity** removes symptoms another cause already explains. If the model weights alone overflow VRAM, your KV cache is under pressure because of that. Telling you to shrink the KV pool would be treating a symptom. A buffer squeeze without weights overflow does not silence R2. Prefill-bound silences Under-batching when the server is pressed (near decode ridge, KV binding, or a queue). Under soft field (no wait, cool KV, running well below ridge), that row is skipped so Under-batching owns first fire and Prefill/Prefix are held for remeasure reveal. Light-load Prefill is not sold as the setup wall.
- **The DAG layers** encode one rule: fix what is broken before tuning what is healthy. A tuning suggestion sits structurally below an active bottleneck and can never outrank it.
- **Nothing is discarded.** Losing rules are held. They surface when the same primary re-fires, or immediately when the primary has no fix left to offer, so you get the next hypothesis without ever seeing five at once.
- **Rules fire on evidence of harm, never on heat.** A server at 95% KV cache with no evictions and no queue is healthy and busy, and Profile stays quiet.

## The eight rules

| Rule                          | Fires when                                                                                                                                          | Prescribes                                                                                                                                                                |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **R1 Under-batching**         | Known GPU: config-relative efficiency under 60% with occupancy under 75% and no backlog. Uncatalogued GPU: occupancy under 25%. Soft field (no wait, cool KV, running ≪ ridge): ranks above Prefill/Prefix on first fire. | Batch more requests or raise client concurrency, naming the binding wall: config, compute ridge, or memory                                                                |
| **R2 KV cache pressure**      | KV at or above 88%, and either an eviction signal (`num_preemptions_per_sec` > 0.02, or swapped ≥ 2) or queue backpressure (waiting > 2)            | Split by cost. Cuts throughput: lower `--max-model-len` or `--max-num-seqs`. Safe: prefix caching, raise `--gpu-memory-utilization`, fp8 KV cache, `--kv-offloading-size` |
| **R2b KV admission backlog**  | Queue ratio at or above 0.30 with KV near full, scheduler not at the seat cap, and estimated free KV below demand                                   | Same family, without the eviction signal                                                                                                                                  |
| **R3 Low prefix reuse**       | Active traffic (running > 0.75), mean prompt ≥ 20, and qps × mean prompt ≥ 1000. Caching off: fires on that volume. Caching on: hit rate under 35%. | `--enable-prefix-caching`, or restructure prompts if already on                                                                                                           |
| **R4 OOM risk**               | Weights overflow VRAM, or weights fit but free VRAM cannot hold one worst-case request                                                              | `--tensor-parallel-size` at the computed minimum, raise `--gpu-memory-utilization`, a smaller model, or the model does not fit on this hardware                           |
| **R5 Concurrency saturation** | Scheduler at `--max-num-seqs`, at least 2 waiting, queue ratio at or above 0.30                                                                     | Below 80% KV, raise `--max-num-seqs` to a bounded target. Above it, name the wall or add a replica                                                                        |
| **R6 Prefill-bound**          | Prompt/gen ratio ≥ 5 with decode efficiency under 40%. Muted when TPOT is measured and under 4x its floor. Bound path silences R1; soft field does not. | Chunked prefill, `--max-num-batched-tokens` (no blind Set when unread; never Set down when already above default), shorter prompts; at the compute wall, disaggregate or add a replica |
| **R7 Config headroom**        | `--max-num-seqs` below 90% of the recommended target, with occupancy at or above 50% and at most 1 waiting                                          | Raise it, and name what binds the target                                                                                                                                  |

When nothing fires but efficiency is still low, a fallback names the shape of the underuse. When nothing fires at all, Profile names the boundary capping the server: capacity, traffic, physics, prefill interference, or framework overhead. Where the ceiling itself is unknown, it says that instead of naming a boundary it cannot prove.

---

Deep reference on the website: [Rules (thresholds, confidence, edge cases)](https://jungledesh.github.io/profile/docs.html#rules) · [Math](https://jungledesh.github.io/profile/docs.html#math) · [Catalog (GPU bandwidth, FLOPs, prices)](https://jungledesh.github.io/profile/docs.html#catalog) · [Design](https://jungledesh.github.io/profile/docs.html#design)
