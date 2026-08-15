# Limitations

Every boundary, and why it exists. Back to the [README](../README.md).

- **One GPU.** `--tensor-parallel-size` greater than 1 is refused. KV and weight sharding math is single-GPU only. Profile still tells you when a model needs tensor parallelism to fit at all. Multi-GPU is on the [roadmap](roadmap.md).
- **vLLM only.** The engine boundary is clean, but SGLang is not built. It is first in line on the [roadmap](roadmap.md).
- **Ceilings are uncalibrated.** Published specifications overestimate. Every ceiling-derived number is marked.
- **The overhead-bound regime is named, not measured.** Profile can say the GPU is idling on CPU work but cannot quantify it.
- **Unknown GPU or model gets no ceiling.** Profile reports `Hardware ceiling unknown` with the reason, rather than guessing. Missing yours? [Open an issue](https://github.com/jungledesh/profile/issues) with the name; a catalog entry is a quick add.
- **No load, no answer.** Idle windows are skipped. There is nothing to diagnose on a server at rest. No production traffic yet? Drive load with `vllm bench serve`.
- **You apply the fix.** Profile never changes your server.

Where the math is approximate, in detail: [website limitations page](https://jungledesh.github.io/profile/docs.html#limitations).
