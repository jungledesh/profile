# Where Profile sits

The stack, and a straight comparison against every neighbouring tool. Back to the [README](../README.md).

---

## The stack

```text
Orchestration        NVIDIA Dynamo, Ray Serve            schedules across nodes
Monitoring           Grafana, Datadog, vLLM /metrics     reports what happened
>>> PROFILE          is any of this working on your hardware, config, and traffic
Inference engine     vLLM, SGLang, TensorRT-LLM          serves the requests
Kernels and runtime  CUDA, ROCm, custom kernels          executes the math
Silicon              NVIDIA, AMD, Cerebras, Groq         sets the ceiling
```

Every layer above and below optimises something. None measures whether the result is any good on your machine.

## The comparison

|                                    | Profile | Dashboards | Kernel profilers | Autotuners  | Simulators  |
| ---------------------------------- | ------- | ---------- | ---------------- | ----------- | ----------- |
| Hardware ceiling from physics      | yes     | no         | no               | no          | predicted   |
| Live server, real traffic          | yes     | yes        | yes              | no          | no          |
| Names one root cause               | yes     | no         | no               | no          | no          |
| Prescribes the change              | yes     | no         | no               | config only | config only |
| Measures the delta after the fix   | yes     | no         | no               | partial     | no          |
| Cost per million tokens            | yes     | no         | no               | no          | no          |
| No auto-restart, no synthetic load | yes     | yes        | yes              | no          | n/a         |

- Dashboards: Grafana, Datadog, vLLM `/metrics`.
- Kernel profilers: Nsight Systems, Nsight Compute.
- Autotuners: vLLM `auto_tune`, SCOOT.
- Simulators: Vidur, LLMCompass, GenZ.

## What Profile is not

- **Not a dashboard.** It reasons rather than reports.
- **Not an autotuner.** It does not restart your server or search a config space.
- **Not a kernel profiler.** That is Nsight's layer and a different question.
- **Not multi-engine.** vLLM only, today.
- **Not autonomous.** You apply the fix; Profile owns the measurement and the memory.

## Who it is for

- **For:** engineers running vLLM who want to know whether their GPU is earning its price, and what to change when it is not.
- **Not for you if:** you shard across GPUs today (on the [roadmap](roadmap.md)), run an engine other than vLLM, or have no traffic to measure. [Limitations](limitations.md) lists every boundary.
