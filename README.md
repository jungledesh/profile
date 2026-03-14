# Profile

The truth engine for LLM inference performance.

---

**Bottlenecks <> Metrics <> Tuning**

Profile reveals where GPU capacity, power, and inference pipelines are wasting money, and gives actionable insights to improve batching, scheduling, and model efficiency.

---

![X ms, Y joules, Z dollars](assets/inference-metrics.jpeg)


**per request.**

For automated tuning and configuration changes, see NVIDIA’s AIConfigurator. Profile is the **truth layer** for AI serving economics.

We focus on less words, more signal: a tool that tells the truth, and gives actions you can use to save money.

---

## What Profile Measures

- Latency: TTFT, TPOT, prefill phase, decode phase  
- Batching & scheduling: active batch size, queue delay, collapse vs saturation  
- KV cache: usage, hit rate, fragmentation, prefix reuse  
- Model & quantization: model choice, dtype, memory wall indicators  
- Cost & energy: tokens/sec, joules per request, cost per token  
- And much more.

Profile provides low-level GPU and inference metrics that help AI teams identify wasted compute, improve GPU utilization, reduce energy usage, and lower cost per token.

---

## Architecture

- **Collectors**: capture raw GPU and inference metrics from your serving stack.  
- **Processors**: compute higher-level diagnostics and cost/economics signals.  
- **Exporters**: ship metrics and insights to OTLP, StatsD, a TUI, and agent-facing APIs.

---

## Project Layout

At a high level, the repository is organized as follows:

```text
/profile (Root)
├── Cargo.toml          # Workspace definition
├── crates/
│   ├── profile-collectors   # Lib: Pulls raw data from NVML/vLLM and other backends
│   ├── profile-core         # Lib: The "brain" / processing and state management logic
│   └── profile-exporters    # Lib: Formatting and export for OTLP, StatsD, TUI, REST/JSON
└── bin/
    ├── profile-cli          # Binary: Simple CLI tool
    └── profile-agent        # Binary: Production sidecar/daemon
```

---

## Docs

- [Roadmap](docs/roadmap.md)
- [GPU setup](docs/gpu-setup.md)
