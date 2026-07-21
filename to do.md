# To do

## Roadmap

1. **Core product fix + v2.2** (single-GPU-only catalog + work + ridge-based metrics / efficiency journey)
2. Test fixes with Gemma, Llama, Qwen on NVIDIA + AMD
3. Set agent-swarm traffic
4. Test with Gemma, Llama, Qwen (for demo) on NVIDIA + AMD
5. README + docs + clean GitHub
6. Webpage
7. GTM

## # (one coherent bucket, not four patches)

1. **Wall detection.** Name the hardware wall instead of offering spinning knobs (Humble + Useful). Status: R1 three-wall `min(max_num_seqs, ridge, observed kv_capacity)` with binding-wall naming SHIPPED. Remaining: limiter/no-issue path wall naming where applicable.
2. **Rank + label throttling fixes.** Order fixes by leverage where computable; mark throttles (levers that cut throughput to save latency) as last-resort before they're applied (Useful, charter #4 "risky levers marked"). Status: OPEN.
3. **Waste honesty.** Waste line DELETED (shipped). Remaining: the ceiling-cost line, "Cost/1M: $3.60 now, ~$0.19 best possible on this hardware (est)" derived from `cost_per_hr / (decode_ceiling x ridge x 3600 / 1e6)`. States unused compute ceiling, never "wasted." Never label latency-collapse "improved" (shipped via silent-good/`worse` delta). Status: ceiling-cost line OPEN.



## v2.2 (confirmed, do not lose)

- **Sliding-window counterfactual gate** (spec below). When deduced `state_pages` is implausibly large (Gemma's 1862: a signal the two-term model doesn't fit), suppress the projected number, stay directional.
- **Three-currency memory budget:** SHIPPED (see `memory_budget_spec.md`).
- **R6 magnitude re-sweep.** The ridge x1.25 formula's direction is proven, magnitude isn't (moderate-load sweep was non-binding). Re-run under saturating prefill to earn removing the `(est)`.

---



## 1. Sliding-window counterfactual gate

**Problem.** `observed_state_pages` (`src/engine/baseline/math.rs:205-222`) deduces per-sequence page cost from allocator geometry. For models the two-currency page model does not describe, the deduction absorbs the structural mismatch and the projected "fits N concurrent requests" line from `capacity_at_hypothetical_max_len` (`src/engine/rules/mod.rs:60-91`) becomes fiction.

**Fix.** After `observed_state_pages` returns, suppress both projection tiers when deduced fixed state is at least as large as the current transcript page count. That comparison falsifies the shared two-currency assumption.

Downstream is already correct: `model_len_shrink_suggestion_lines` (`src/engine/rules/mod.rs:118`) handles `None` by emitting the direction-only line ("Lower --max-model-len...") with no number.

**Test.** Gemma-shaped geometry suppresses both tiers; ladder and dense geometry still project; all-labels-absent and partial-labels (no `num_gpu_blocks`) both retain catalog fallback. Gate-trip output keeps direction (`Lower --max-model-len`) and drops the projected concurrency clause (`; fits N concurrent requests (est)`), not the legitimate `(fits p99 of observed requests)` phrase.

---



## Engineering debt (small, not blocking)

- **R7 Observed alignment.** R7 still mins ridge against the empirical KV estimator (`running / kv%`) instead of Observed `kv_cache_max_concurrency`. Align with R1's three-wall Observed path.
- **R1 aggregate binder mismatch.** `aggregate_r1_detail` averages `effective_max` across windows but keeps `details[0]`'s binding wall. If the binder flips mid-run (e.g. Observed capacity appears partway), the label can mismatch the averaged number. Pick the binder from the same window as the min, or the most frequent binder.

