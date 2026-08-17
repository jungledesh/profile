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
+----------------------------------------------------------------------------------------------------------------------+
|PROFILE v2.2.1 [muse-glimmer-30b] [NVIDIA GeForce RTX 5090] (5m from 2026-08-14 10:30:26 UTC)                         |
|                                                                                                                      |
|GPU =>               decode_eff ~3.5% | power 531W | 4.05 J/tok | $2.10/1M output tok (est) | vRAM 29/32GB            |
|                     mem_util 37%                                                                                     |
|                                                                                                                      |
|vLLM =>                                                                                                               |
|REQUESTS             run 9 (27.4%) | wait 15 | max 32                                                                 |
|LATENCY              ttft 32.8s (p95 66.9s) | tpot 58ms (p95 89ms)                                                    |
|CACHE                kv_cache 88.6% avg (99.9% peak) | pfix_cache 22.7%                                               |
|THROUGHPUT           131 tok/s                                                                                        |
|TRAFFIC              qps 0.5 | req_total 642 | gen_total 175857 | preempt/s 0.16 | preempt_total 59                   |
|                                                                                                                      |
|ISSUES:                                                                                                               |
|                                                                                                                      |
|[!] KV Cache Pressure                                                                                                 |
|    Seen in 100% of windows                                                                                           |
|    Cause:                                                                                                            |
|      KV cache 89% avg in fired windows, 100% peak (threshold: 88%).                                                  |
|      Scheduler evicting; 15 requests queued on KV admission.                                                        |
|                                                                                                                      |
|    Fix:                                                                                                              |
|      • Raise --gpu-memory-utilization (check vRAM header for avail mem) to expand KV pool.                           |
|      • Switch --kv-cache-dtype fp8 to halve KV memory footprint (affects output quality).                            |
|      • Lower --max-num-seqs to reduce KV demand.                                                                     |
|        Cuts throughput. Revert after pressure clears.                                                                |
|      • Reduce client concurrency toward sustained running (9 in-flight).                                            |
|        Cuts queue wait, not throughput. Demand exceeds admitted capacity.                                            |
|      • Lower --max-model-len 32768 → 21933. Observed p99 21.9k tokens per request.                                   |
|        ~1% of observed requests ran longer; those are rejected with a 400, not truncated.                            |
|                                                                                                                      |
|    Expected: TTFT and TPOT recover once evictions stop.                                                              |
|    Confidence: High                                                                                                  |
+----------------------------------------------------------------------------------------------------------------------+
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

New --max-num-seqs [current: 32]: 12

Measuring delta...

  Config changed. Baseline reset.

  Throughput          174 → 401 tok/s
  TTFT                44157 → 239ms (p95 77146 → 604ms)
  TPOT                53.7 → 24.9ms (p95 87.0 → 46.3ms)

ECONOMICS:
  J/tok               3.01 → 1.02
  Cost/1M output tok  $1.58 → $0.69 (est)
```

**Iterate.** Fixing one bottleneck usually exposes the next. The path is not always upward, and Profile does not pretend otherwise. Regressions are labelled, not buried:

```text
  Throughput          183 → 131 tok/s  worse
  TTFT                430 → 32797ms (p95 2108 → 66870ms)  worse
  TPOT                28.1 → 58.5ms (p95 49.5 → 89.4ms)  worse
  Decode eff.         -1.4pp

ECONOMICS:
  J/tok               2.35 → 4.05  worse
  Cost/1M output tok  $1.50 → $2.10 (est)  worse
```

A tool that only reports improvements cannot be trusted when it reports one.

## What the loop will not do to you

- **Repeat itself.** If the same primary fires again after a re-measure, Profile reveals the alternative causes it was holding back. It does not wait for proof that your fix changed nothing; the repeat itself is the signal.
- **Dead-end you.** When no config lever remains (empty Fix, or a terminal wall such as replica / FLOPs), Profile names the wall, points at scale-out, and surfaces the suppressed alternatives under that block on first fire instead of inventing another flag.
- **Tell you to do what you already did.** Where Profile can read your running config, it skips levers you already set (prefix caching on, fp8 KV already active, chunked prefill on, `--max-num-batched-tokens` already at or above the suggestion, seats already at the target). Exception: `--kv-offloading-size` is re-derived on each R2 fire; if the new size differs from what is set, the flag is offered again.
- **Ping-pong you.** When KV pressure and concurrency saturation alternate on `--max-num-seqs`, Profile detects the cycle, names the bracket it has tried, and suggests the midpoint. It offers this at most three times, then names the wall instead of guessing again.
- **Spin on unread Prefill.** When Prefill is primary twice in a row and both Fix blocks include the unread `--max-num-batched-tokens` guide (common when vLLM never emits the gauge), the second table still prints, then the loop exits: no new server lever to apply. First unread show stays open (Confirm + guide).

**Scale out.** Eventually no flag helps. Profile says the scheduler is at its cap with the pool full, that no config change helps, and that the next move is a replica. That is the answer, not a failure. When the box is healthy and quiet, it names the cap instead: the 5090 Muse run ended `Capped by vLLM overhead` at 421 tok/s; the H100 Qwen3.8 run ended `Capped by traffic` at 490 tok/s.

---

Deep reference on the website: [Get started](https://jungledesh.github.io/profile/docs.html#home) · [Data (metric sources)](https://jungledesh.github.io/profile/docs.html#data)
