# Profile Roadmap

## Vision

Become the de facto advisor for efficient, physics-grounded AI inference serving, starting with individuals/power users, then teams, enterprises, and large fleets.

## Core Philosophy

Ship small, strong foundational blocks frequently. Prioritize quality and user trust over feature bloat. Communicate progress aggressively.

---

## v2.2: Engine + Rules

The foundation. Harden the physics engine, deepen the rule set, prove it on real hardware.

- Multi-GPU / TP support (done)
- Physics engine hardening (ceiling math, edge cases, needle precision)
- New rules in DAG with mutual exclusivity
- Test and validate on RunPod with real workloads
- Update README, docs, and interactive web page
- Demo on xAI's popular model with real numbers
- Launch and announce

## v3.0: The Curve

The differentiator. Derive the optimal operating point from physics. No other tool does this.

- Throughput-latency curve derived from hardware physics at startup
- Knee detection algorithm (point of diminishing returns)
- SLA input: operator sets a target, or profile defaults to the knee
- Rule recommendations framed against the target operating point
- Delta shows progress toward target each iteration
- Terminal sparkline visualization
- Test and validate on RunPod
- Update README, docs, and web page
- Demo on popular and trending model + setup
- Launch and announce

## v3.1: Observability

Let operators plug profile's brain into their existing monitoring stack.

- Prometheus exporter
- Grafana dashboards (optional, profile stays the brain)
- Test, update docs + README + web page, launch

## v4.0: Fleet

New product surface. From single node to fleet-wide optimization.

- k8s integration
- Multi-node orchestration
- Fleet-wide efficiency reporting
- Enterprise features
- Test, update docs + README + web page, demo, launch

## v4.1: Hardware + Engine Expansion

Same physics, new inputs. Each expansion is its own announcement.

- AMD GPU support (new physics constants, same math)
- SGLang support (new collector, same engine)
- TensorRT-LLM support
- Each gets its own launch cycle

---

## Principles across all releases

- Engine is the product. Everything else is surface.
- Every ceiling, recommendation, and operating point is derived from physics. No heuristics. No vibes.
- Each version ships independently and is useful on its own.
- Each version gets its own announcement across all channels: X, LinkedIn, HN, Reddit, YouTube.
- The moat deepens with every release. The physics engine is what no other tool does.
