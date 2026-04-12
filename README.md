# Profile

Cut GPU waste. Find issues and fix them in a minute.

---

**Bottlenecks <> Profile <> Tuning**

Follow **Pain → Metric → Diagnosis → Cause → Evidence → Fix**: simply, surface waste, rank issues, and return diagnosis, cause, evidence, confidence, and a recommended plan (impact + basis)—or a clean **do nothing** when there’s nothing major to fix.

---

![X ms, Y joules, Z dollars](assets/inference-metrics.jpeg)


**per request.**

We focus on less words, more signal: a tool that tells the truth, and gives actions you can use to save money.

---

## Docs

- [Roadmap](docs/roadmap.md)
- [GPU setup](docs/gpu-setup.md)

---

## Technical design

Diagnose derives some vLLM numbers from a **short multi-scrape window** (several successive `/metrics` polls, on the order of a couple of seconds)—for example **prefix cache hit rate** as Δhits/Δqueries between the first and last sample in that window. Those values describe **what happened while Profile was observing**, not lifetime or steady-state server behavior. Runtime output stays user-facing; this note is the place for that caveat.
