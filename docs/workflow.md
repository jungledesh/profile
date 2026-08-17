# Workflow

How to run a session, then every flag and output line. Back to the [README](../README.md).

---

## Get started

1. vLLM on one GPU, `/metrics` reachable, live traffic on the box.
2. Install from the [README](../README.md#get-started), or build with `cargo install --git https://github.com/jungledesh/profile`.
3. `profile diagnose --url http://localhost:8000/metrics --duration 30s`
4. Apply the Fix. Press Enter. Read the delta. Repeat until the loop names a wall or goes quiet.

Idle server: drive load with `vllm bench serve`. Match `--duration` to the cycle (table below).

## Requirements, in full

- **GPU.** NVIDIA through NVML, or AMD through the amdgpu driver. Profile probes NVIDIA first and falls back to AMD.
- **vLLM** with `/metrics` reachable. Default `http://localhost:8000/metrics`.
- **Live traffic** during the window. An idle server has no waste to find, and Profile says so rather than invent a number. Trying Profile without production traffic? Drive load with vLLM's own `vllm bench serve`, or any load generator.
- Launch scope is a single GPU. `--tensor-parallel-size` greater than 1 is refused today. Profile still tells you when a model needs tensor parallelism to fit at all.

## CLI flags

These configure Profile, not vLLM. Each flag also has an env var (`PROFILE_URL`, `PROFILE_DURATION`, `PROFILE_MAX_NUM_SEQS`, `PROFILE_TENSOR_PARALLEL_SIZE`, `PROFILE_COST_PER_HOUR`, `PROFILE_VERBOSE`).

| Flag                     | Default                         | Description                                                                                          |
| ------------------------ | ------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `-u, --url`              | `http://localhost:8000/metrics` | vLLM metrics endpoint                                                                                |
| `--duration`             | `30s`                           | Collection window per iteration. Minimum `30s`, maximum `30m`. Units `s` or `m` only, not `ms`/`mins`. Match to traffic shape (below). |
| `-m, --max-num-seqs`     | Prompted if unread              | Skip the prompt. Preflight reads `/metrics` when the gauge is present; otherwise you are asked.      |
| `--tensor-parallel-size` | Unset                           | Must be 1. Values above 1 are refused. Pass `1` to skip the GPU-assignment prompt.                   |
| `--cost-per-hour`        | Catalog estimate                | GPU cost in USD/hr. Must be a positive number.                                                       |
| `-v, --verbose`          | Off                             | Rules that did not fire, physics limits, and extra GPU, latency, cache, and config detail            |

`profile help`, `profile completions <SHELL>`, and `profile man` exist. They are not diagnose flags.

**Duration and traffic shape.** Match `--duration` to the cycle. Default for steady load. Raise it when traffic repeats inside the window. Do not raise it when load changes between iterations.

| Shape | What it looks like | `--duration` |
| ----- | ------------------ | ------------ |
| Steady | Load holds still | Default `30s` is enough |
| Fast bounce | Wobble shorter than a slice (2s at `30s`, 10s above) | Already averaged. Leave it |
| Cycle near the window | Agents start, time out, and restart together | Raise so several cycles fit, up to `30m`. A ~10 min cycle needs about `30m` |
| Step between iterations | Flat during a run, different on the next | Not a duration problem |

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
|PROFILE v2.2.2 [Qwen3.6-27B] [NVIDIA H100 80GB HBM3] (2m from 2026-07-31 07:13:29 UTC)                                      |
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

## What the loop will not do to you

- **Repeat itself.** If the same primary fires again after a re-measure, Profile reveals the alternative causes it was holding back. It does not wait for proof that your fix changed nothing; the repeat itself is the signal.
- **Dead-end you.** When no config lever remains (empty Fix, or a terminal wall such as replica / FLOPs), Profile names the wall, points at scale-out, and surfaces the suppressed alternatives under that block on first fire instead of inventing another flag.
- **Tell you to do what you already did.** Where Profile can read your running config, it skips levers you already set (prefix caching on, fp8 KV already active, chunked prefill on, `--max-num-batched-tokens` already at or above the suggestion, seats already at the target). Exception: `--kv-offloading-size` is re-derived on each R2 fire; if the new size differs from what is set, the flag is offered again.
- **Ping-pong you.** When KV pressure and concurrency saturation alternate on `--max-num-seqs`, Profile detects the cycle, names the bracket it has tried, and suggests the midpoint. It offers this at most three times, then names the wall instead of guessing again.
- **Spin on unread Prefill.** When Prefill is primary twice in a row and both Fix blocks include the unread `--max-num-batched-tokens` guide (common when vLLM never emits the gauge), the second table still prints, then the loop exits: no new server lever to apply. First unread show stays open (Confirm + guide).

**Scale out.** Eventually no flag helps. Profile says the scheduler is at its cap with the pool full, that no config change helps, and that the next move is a replica. That is the answer, not a failure.

---

Deep reference on the website: [Get started](https://jungledesh.github.io/profile/docs.html#home) · [Data (metric sources)](https://jungledesh.github.io/profile/docs.html#data)
