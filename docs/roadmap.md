# Roadmap

### V1 — Core Profiling

- [ ] Request efficiency: prompt tokens, output tokens, latency
- [ ] GPU metrics: utilization, power draw
- [ ] Derived metrics: tokens/sec, tokens/watt, estimated cost
- [ ] `profile request`
- [ ] `profile batch`

### V2 — Cache Efficiency

- [ ] KV cache hit rate
- [ ] KV cache miss rate
- [ ] Prefix reuse metrics

### V3 — Transformer Pipeline Metrics

- [ ] Prefill vs decode latency
- [ ] Prefill tokens/sec, decode tokens/sec

### V4 — Scheduler Diagnostics

- [ ] Queue delay
- [ ] Batch wait time
- [ ] Scheduler delay

### V5 — Live Dashboard

- [ ] `profile watch`
- [ ] Real-time throughput, queue time, batch size, GPU efficiency
