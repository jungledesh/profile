# Where Profile sits

The inference serving stack, and a straight comparison against every neighbouring tool. Back to the [README](../README.md).

---

## The inference serving stack

```text
Orchestration        NVIDIA Dynamo, Ray Serve            schedules across nodes

Monitoring           Grafana, Datadog, vLLM /metrics     reports what happened

>>> PROFILE          is any of this working on your hardware, config, and traffic

Inference engine     vLLM, SGLang, TensorRT-LLM          serves the requests

Kernels and runtime  CUDA, ROCm, custom kernels          executes the math

Silicon              NVIDIA, AMD, Cerebras, Groq         sets the ceiling
```

Every layer above and below optimises something. None measures whether the result is any good on your machine. Profile does that in a few iterations, so you are not guessing for days.

Profile plugs into this stack; it replaces nothing. It reads what your server already emits, and export into the observability tools you already run is on the [roadmap](roadmap.md).

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
- **Not autonomous.** You apply the fix; Profile owns the measurement and the memory.

## Who it is for

- **For:** everyone running vLLM on one GPU. Tuned or not, Profile tells you what your hardware can still give, in a few measured iterations instead of days of guessing.
- **Not yet:** multi-GPU sharding, and engines beyond vLLM. Both on the [roadmap](roadmap.md). Every boundary, with its reason: [limitations](limitations.md).
