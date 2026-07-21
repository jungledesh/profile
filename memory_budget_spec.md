# Spec: three-currency memory budget

v2.2 item 2. Final, decisions folded in: byte-denominated budget (no token-unit mixing),
sliding window included, R4 fires at B1 (weights + one request), single GPU behind a
compile-time const. Supersedes `to do.md` section 2; delete that section on merge.

Charter: Correct (price what the memory actually costs), Humble (decline any currency
we cannot price; labels degrade with the source), Transparent (every number names its
source).

---

## Part 1: one pricing function (the choke point)

New in `math.rs`:

```rust
/// Bytes one worst-case request costs in GPU memory. Three currencies:
/// full-attention transcript, window-capped transcript, fixed whiteboard.
/// None = a currency this model uses cannot be priced. Callers decline, never guess.
pub fn bytes_per_seq(arch: &ModelArch, max_model_len: u32, kv_dtype_bytes: u8) -> Option<u64>
```

```
full_layers  = kv_layers - swa_layers            (kv_layers = num_kv_layers.or(num_layers))
per_tok      = 2 x num_kv_heads x head_dim x kv_dtype_bytes   (per layer)

transcript   = full_layers x per_tok x max_model_len
window_part  = swa_layers  x per_tok x min(max_model_len, swa_window)
whiteboard   = catalog_hybrid_state_bytes(..)                  (0 when no linear_* fields)

bytes_per_seq = transcript + window_part + whiteboard
```

Decline rules (return `None`):
- `linear_*` fields present but `state_dtype` missing → cannot price the whiteboard.
- `num_swa_layers` present but `swa_window` missing (or reverse) → cannot price the window.
- Missing `num_kv_heads`/`head_dim` → cannot price the transcript (as today).

Models not in the catalog get attention-only pricing, as today. We cannot know what we
have not cataloged; item 1's gate already protects the observed-geometry path.

`checked_*` arithmetic throughout. No floats until the final division.

### Catalog additions (two fields)

```rust
/// Sliding-window size in tokens (Gemma 3: 1B=512; 4B/12B/27B=1024).
/// None = no windowed layers.
pub swa_window: Option<u32>,
/// Number of windowed layers. Full-attention layers = kv_layers - this.
pub num_swa_layers: Option<u32>,
```

Populate for the Gemma 3 family from the same HF configs the entries already cite
(5:1 local:global pattern). All other entries: `None` via the `catalog_dense!` /
`catalog_hybrid!` macros, zero rows touched.

## Part 2: budget in bytes, best source first

Bytes only. `kv_cache_size_tokens` is not used anywhere: a token-denominated budget
cannot sit over a byte-denominated per-request cost without re-answering the same
multi-currency question in the conversion. Settled.

- **Rung A, observed bytes.** Hybrid allocator: `num_gpu_blocks x mamba_page_size_padded`
  (bytes directly, no unit ambiguity). Dense: `num_gpu_blocks x block_size x per_tok x kv_layers`
  (token blocks priced at the model's own rate, units safe).
  Windowed models: rung A is NOT trusted, block accounting is exactly what vLLM's own
  docstring warns can mislead for hybrid layouts → rung B.
- **Rung B, derived estimate.** `(vram x gpu_util) - ACTIVATION_KV_BUFFER_GB - weights`.
  The 3 GB stays, stays labeled `(est)`, and gets self-graded: whenever rung A is also
  available, compute both, surface the gap in verbose. RunPod matrix run before launch
  decides whether 3 GB is safe, fat, or thin. Data, not opinion.
- Neither → derived tier declines entirely; empirical (rung 3) takes over.

## Part 3: capacity = budget / price

`compute_kv_max_seqs` (`rules/mod.rs`) becomes: resolve budget bytes (Part 2), price one
request (Part 1), divide, floor, `n > 0`. Existing signature kept for callers.

`kv_headroom_gb` (`roofline.rs:161-167`) is NOT modified. R4's `h < 0` keeps meaning
exactly "weights don't fit."

### Labels as sources degrade

| Rung | Source | Shown |
|---|---|---|
| 1 | vLLM `kv_cache_max_concurrency` | plain number (existing) |
| 2 | this derivation | `(est)`; source stays `Derived` / `DerivedHybrid` (windowed prices correctly under `Derived`, no new enum variant) |
| 3 | empirical `running / kv%` | `(est)` + Low confidence + 2x step cap (existing, keep) |
| out | nothing | no number, direction-only line |

The derived number is a floor by construction (every request worst-case). Anywhere it
prints alongside prose, the wording is "at least N" / "fits N worst-case requests", never
a prediction of mixed-traffic behavior.

## Part 4: R4 state floor, B1

R4 gains one branch, distinct from weights-overflow so attribution holds:

- Existing: `h < 0` → weights don't fit. Unchanged, lines unchanged.
- New: `h >= 0` but `h_bytes < bytes_per_seq(1)` → the GPU holds the weights and cannot
  hold one request. Cause line names it plainly ("free VRAM after weights cannot hold a
  single request's KV + state"); fix line: raise `--gpu-memory-utilization` if headroom
  exists, else smaller model / larger GPU. No TP suggestion at launch (flag off).
- Rare by design. Confidence follows the existing weight-dtype source table.

No `MIN_HYBRID_CONCURRENCY_TARGET`. "Can't reach useful concurrency" is R5/R7's story,
told through the corrected denominator.

## Part 5: single GPU, compile-time

```rust
/// Launch scope: single GPU, no tensor parallelism. TP machinery stays behind this.
pub const MULTI_GPU_TP: bool = false;
```

One const in `src/engine/mod.rs`. Behavior when off:
- GPU assignment scans and uses the first GPU; everything downstream runs as single-GPU.
- If the server's config still reports `tensor_parallel_size > 1`, the derived capacity
  tier declines (we will not price sharded state we have not measured); empirical takes
  over with its Low label. R4's shipped weights/TP behavior is left as-is.
- No state-sharding math executes anywhere in the launch path. The TP divisor
  (vLLM-source-confirmed, own-data-unvalidated) waits behind the flag for a TP2 RunPod run.

Promote to a CLI flag only when TP ships. Dead code warnings: the flagged TP paths keep
their tests, so they are not dead.

## Part 6: catalog for launch

Rule: a model enters only if weights + one worst-case request fit at least one launch GPU
at default dtype. Launch GPUs: A100-80GB, H100-80GB, L40S-48GB, A10G-24GB, RTX 4090-24GB,
RTX 3090-24GB. Prices: one reference provider per GPU, on-demand, not spot.

Fit-check gemini's model list against that rule before any row lands (32B dense at bf16
is 64 GB → 80 GB GPUs only; quantized-32B-on-24GB rows stress weight-dtype detection,
include only with a tested dtype source). Prune what fails. Add after launch from what
users actually run.

## Tests

1. **Dense regression:** `bytes_per_seq` with no swa/linear fields reproduces today's
   arithmetic byte-for-byte; full suite green with no dense fixture edited.
2. **Whiteboard:** Qwen3.6 27B entry: hand-computed `transcript + state`, capacity drops
   vs attention-only by the expected count on a fixed budget.
3. **Window:** Gemma 3 27B entry at max_model_len 8192: window term prices 1024-capped
   layers; assert capacity is a hand-computed multiple of the all-full-context result.
   Boundary: `max_model_len < swa_window` collapses to dense pricing exactly.
4. **Decline:** linear_* without state_dtype → `bytes_per_seq` None → derived tier None →
   empirical label Low. Same for swa fields half-present.
5. **Budget rungs:** hybrid labels → rung A bytes via page size; dense → rung A via token
   blocks; windowed model with labels present → rung B (rung A refused); no labels → rung B;
   no vram → decline.
6. **R4 B1:** geometry where weights fit and one request does not → new branch fires with
   the state-floor line; identical geometry minus whiteboard → no fire. `h < 0` path
   byte-identical to today.
7. **Flag:** config tp=2 with `MULTI_GPU_TP = false` → derived tier None, empirical used.
8. `tests/physics_tests.rs` updated for any signature change.

Hygiene block. Then done.

## RunPod checklist (pre-launch, same session as R6 re-sweep)

- 3 GB self-grade across the launch matrix (small GPU, big GPU, dense, hybrid, windowed).
- Derived-vs-Observed residual: state-aware and window-aware pricing should shrink the
  gap vs Observed for hybrid and Gemma respectively. If it does not, the pricing is wrong.
- Item 1 e2e: the Gemma config that produced 1862 prints direction-only.

## Out of scope

TP>1 math (flagged), R6 magnitude (item 3), R7 Observed alignment (item 4), output
formatting beyond the labels named here.
