# The engine

The ceiling math, the eight rules, and what decides which one you see. This is where opaque inference becomes deterministic engineering. Back to the [README](../README.md).

Profile does three things:

1. **Computes your hardware ceiling.** The fastest your GPU could serve this model, derived from physics.
2. **Finds the bottleneck.** Eight rules cut everything suppressing the server down to one primary cause.
3. **Measures whether your fix worked.** The re-measure is what makes the diagnosis causal.

---

## The ceiling

Physics gives two hard limits. Profile derives both from your GPU and model.

**Prefill** is compute-bound (prompts/s, not tok/s):

```text
prefill_ceiling = (peak_flops x tp x 1e12)
                / (2 x params x seq_len + attn_coeff x layers x seq_len^2)
```

- The first term is the weight math, linear in sequence length.
- The second is attention, quadratic in it. `attn_coeff` is the catalog value, else `2 x hidden_dim` for MHA/GQA, else 0 (linear-only fallback).
- `seq_len` is mean prompt tokens when that histogram is present, else `--max-model-len`. No length, no prefill ceiling.
- That is why the prefill ceiling falls as context grows, and why a linear-only roofline overstates long-context capability.
- TP scales FLOPs the same way decode scales bandwidth. Profile refuses TP > 1 at launch today; the `tp` factor is 1 in production.

**Decode** is memory-bandwidth-bound:

```text
decode_ceiling_tps = (peak_bandwidth_gbps x tp x 1e9 x 8) / (params x bits_per_param)
```

Same as dividing by `params x bytes_per_param`. `bits_per_param` comes from `/info` quantization, then `QUANTIZATION`, then reported dtype, then `DTYPE`, then the catalog default, then bf16 (16-bit) fallback. Fallback is labelled; it is not silent.

- Efficiency is measured generation tok/s against decode ceiling times ridge batch size: the batch where decode stops being memory-bound and starts being compute-bound. Config-relative efficiency uses `min(max_num_seqs, ridge)` instead of ridge.
- If measured decode beats the one-token-per-read roof (speculation suspected), efficiency, config-relative efficiency, and headroom are cleared. The scoreboard shows `-`, not a false %.
- Ceilings are coarse upper bounds from published specifications, not a hardware simulation. They are marked `(est)`, and values derived from them carry a tilde.
- An uncatalogued GPU or model gets `Hardware ceiling unknown` with the reason, rather than a wrong number.
- `params` in both ceilings is the active parameter count when the catalog defines one (MoE experts on path, or multimodal models where decode streams the text stack only). Weight footprint and OOM sizing use the total. One model, two numbers, on purpose.

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

| Rule                          | Fires when                                                                                                                                                                                                 | Prescribes                                                                                                                                                                                                 |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **R1 Under-batching**         | Waiting under 2. Occupancy is running over `min(max_num_seqs, ridge, observed kv_cache_max_concurrency)`. Known GPU: config-relative efficiency under 60% and occupancy under 75%. Uncatalogued GPU: occupancy under 25%. Soft field (no wait, KV mean under 80%, running well below ridge): ranks above Prefill/Prefix on first fire. | Batch more requests or raise client concurrency, naming the binding wall: config, compute ridge, or memory                                                                                                 |
| **R2 KV cache pressure**      | KV avg or peak at or above 88%, and either eviction (`num_preemptions_per_sec` > 0.02, or swapped ≥ 2) or waiting > 2                                                                                      | Split by cost. Cuts throughput: lower `--max-model-len` or `--max-num-seqs`. Safe: prefix caching, raise `--gpu-memory-utilization`, fp8 KV, `--kv-offloading-size`. Dead-end: verify, then replica.     |
| **R2b KV admission backlog**  | Same 88% near-full bar, queue ratio at or above 0.30, running below `--max-num-seqs`, free KV tokens below waiting × mean prompt. Mutually exclusive with R2 via eval `else if`, not the suppression table. | Same family, without the eviction signal                                                                                                                                                                   |
| **R3 Low prefix reuse**       | Running > 0.75, mean prompt ≥ 20, qps × mean prompt ≥ 1000. Caching off: fires on that volume. Caching on: hit rate under 35%.                                                                             | `--enable-prefix-caching` when confirmed off, or restructure prompts if already on. No Enable when the flag is unread.                                                                                     |
| **R4 OOM risk**               | Weights overflow VRAM (`kv_headroom_gb` < 0), or weights fit but free VRAM cannot hold one worst-case request. Silent on dtype fallback.                                                                   | `--tensor-parallel-size` at the computed minimum, raise `--gpu-memory-utilization`, a smaller model, or the model does not fit on this hardware                                                            |
| **R5 Concurrency saturation** | Collector `seat_wall_cooccurred` (a scrape with running in the churn band at `--max-num-seqs` and waiting ≥ 2), plus window waiting ≥ 2 and queue ratio ≥ 0.30. No peak/3% fallback: flag unread or false → silent. | KV under 80%: raise `--max-num-seqs` to 0.8 × min(ridge, KV bound), capped by a live-traffic floor when one exists. At a wall, or KV ≥ 80%: name the wall or replica.                                      |
| **R6 Prefill-bound**          | Effective prompt/gen ratio ≥ 5 (prefix hits removed from prompt tok/s) and decode efficiency under 40%. Muted when TPOT is measured and under 4× its floor. Bound path silences R1; soft field does not.   | Prefix caching first when confirmed off. Then chunked prefill / `--max-num-batched-tokens` (no Enable on unread, no blind Set on unread, never Set down when already above 2048). Severe (ratio ≥ 20) or compute wall: disaggregate or replica. |
| **R7 Config headroom**        | `--max-num-seqs` under 90% of the shared R5 target, occupancy (running / max_num_seqs) at or above 50%, waiting at most 1. Waiting ≥ 2 is R5's court.                                                      | Raise it, and name what binds the target (ridge, Observed/derived memory, or empirical floor).                                                                                                             |

When R1's gates miss but the field is soft and Prefill would own the page, a soft under-fed inject reuses R1's fix without bending R1 thresholds. When nothing fires but efficiency is still low, a diagnose-only fallback names the shape of the underuse. When nothing fires at all, Profile names the boundary capping the server: capacity, traffic, physics, prefill interference, or framework overhead. Speculation suspected, waiting unread, or an unknown GPU ceiling: decline a healthy cap rather than name one.

---

Deep reference on the website: [Rules (thresholds, confidence, edge cases)](https://jungledesh.github.io/profile/docs.html#rules) · [Math](https://jungledesh.github.io/profile/docs.html#math) · [Catalog (GPU bandwidth, FLOPs, prices)](https://jungledesh.github.io/profile/docs.html#catalog) · [Design](https://jungledesh.github.io/profile/docs.html#design)
