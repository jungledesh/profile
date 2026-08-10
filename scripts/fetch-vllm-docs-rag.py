#!/usr/bin/env python3
"""Build a frozen vLLM-docs RAG task pack for the Profile AMD demo.

Snapshots official vLLM markdown docs (pinned git SHA), chunks them, attaches
real excerpts to real-ish operator questions, and writes rag-tasks.json next to
this script (or --out).

Wording: this is enterprise-style RAG over real public engine docs, not a
scored RAG benchmark. No random filler corpus.

Usage:
  python3 fetch-vllm-docs-rag.py
  python3 fetch-vllm-docs-rag.py --docs-dir /path/to/vllm/docs --git-sha <sha>
  python3 fetch-vllm-docs-rag.py --clone-dir /tmp/vllm-docs-snap

Requires: git, network (unless --docs-dir points at an existing tree).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Pin a known-good docs tree. Bump deliberately when regenerating the pack.
DEFAULT_GIT_SHA = "7581c56c86ce8dd881a84739c0a1eb83f9f2fdd6"
VLLM_REPO = "https://github.com/vllm-project/vllm.git"

# Shared across every request: prefix-cache lever for R3.
SYSTEM_PROMPT = """You are the internal documentation assistant for a team that runs vLLM in production.
Answer using only the retrieved vLLM documentation excerpts provided in the user message.
Be precise and operator-facing: name flags, defaults, and failure modes when the excerpts do.
If the excerpts are insufficient, say what is missing instead of inventing behavior.
Prefer short, actionable answers unless the question asks for detail.
Do not invent metrics, version numbers, or flags that are not supported by the excerpts."""

# Identical template wrapper; only {excerpts} and {question} change per request.
RETRIEVAL_TEMPLATE = """Retrieved vLLM documentation excerpts:

{excerpts}

Question: {question}

Answer from the excerpts above."""

# ~700-900 tokens of prose at ~4 chars/token. Keep chunks retrieval-sized.
CHUNK_CHARS = 3200
CHUNK_OVERLAP = 400

# Question bank: operator-shaped, grounded in real doc areas.
# tier: lookup (short) | multi (several docs) | hard (long tail).
QUESTION_BANK: list[dict] = [
    {
        "id": "prefix-caching-what",
        "tier": "lookup",
        "question": "What is automatic prefix caching in vLLM, and what problem does it solve under repeated prompts?",
        "keywords": ["prefix", "caching", "kv", "reuse", "hash"],
        "paths": ["docs/design/prefix_caching.md", "docs/features/automatic_prefix_caching.md"],
        "max_tokens": 280,
    },
    {
        "id": "prefix-caching-enable",
        "tier": "lookup",
        "question": "How do I enable prefix caching when serving with vLLM, and which engine argument controls it?",
        "keywords": ["enable_prefix_caching", "prefix", "caching", "engine"],
        "paths": [
            "docs/configuration/engine_args.md",
            "docs/configuration/serve_args.md",
            "docs/design/prefix_caching.md",
        ],
        "max_tokens": 220,
    },
    {
        "id": "chunked-prefill",
        "tier": "lookup",
        "question": "What is chunked prefill, and why would an operator enable it when decode latency suffers under long prompts?",
        "keywords": ["chunked", "prefill", "decode", "batch"],
        "paths": [
            "docs/configuration/optimization.md",
            "docs/features/chunked.md",
            "docs/configuration/engine_args.md",
        ],
        "max_tokens": 280,
    },
    {
        "id": "max-num-seqs",
        "tier": "lookup",
        "question": "What does --max-num-seqs control, and how does it interact with scheduling concurrency?",
        "keywords": ["max_num_seqs", "max-num-seqs", "scheduler", "concurrency"],
        "paths": ["docs/configuration/engine_args.md", "docs/configuration/serve_args.md"],
        "max_tokens": 240,
    },
    {
        "id": "max-num-batched-tokens",
        "tier": "lookup",
        "question": "What is --max-num-batched-tokens used for, and how does it bound prefill work in a step?",
        "keywords": ["max_num_batched_tokens", "batched", "tokens", "prefill"],
        "paths": ["docs/configuration/engine_args.md", "docs/configuration/optimization.md"],
        "max_tokens": 240,
    },
    {
        "id": "gpu-memory-utilization",
        "tier": "lookup",
        "question": "What does --gpu-memory-utilization control, and what happens if it is set too high?",
        "keywords": ["gpu_memory_utilization", "gpu-memory-utilization", "kv", "memory"],
        "paths": [
            "docs/configuration/engine_args.md",
            "docs/configuration/conserving_memory.md",
        ],
        "max_tokens": 240,
    },
    {
        "id": "max-model-len",
        "tier": "lookup",
        "question": "How should I choose --max-model-len, and what is the cost of setting it far above real traffic context?",
        "keywords": ["max_model_len", "max-model-len", "context", "kv"],
        "paths": [
            "docs/configuration/engine_args.md",
            "docs/configuration/conserving_memory.md",
        ],
        "max_tokens": 260,
    },
    {
        "id": "paged-attention",
        "tier": "lookup",
        "question": "In one paragraph, explain paged attention and why it improves KV cache memory packing.",
        "keywords": ["paged", "attention", "block", "kv"],
        "paths": ["docs/design/paged_attention.md"],
        "max_tokens": 300,
    },
    {
        "id": "kv-cache-dtype",
        "tier": "lookup",
        "question": "Which options exist for KV cache dtype / quantization, and when would an operator use FP8 KV?",
        "keywords": ["kv_cache_dtype", "fp8", "kv", "quant"],
        "paths": [
            "docs/configuration/engine_args.md",
            "docs/features/quantization/README.md",
            "docs/configuration/conserving_memory.md",
        ],
        "max_tokens": 260,
    },
    {
        "id": "tensor-parallel",
        "tier": "lookup",
        "question": "When do I need tensor parallel size > 1, and what does --tensor-parallel-size change?",
        "keywords": ["tensor_parallel", "tensor-parallel", "tp", "shard"],
        "paths": [
            "docs/configuration/engine_args.md",
            "docs/serving/distributed_serving.md",
            "docs/usage/distributed.md",
        ],
        "max_tokens": 260,
    },
    {
        "id": "metrics-endpoint",
        "tier": "lookup",
        "question": "Where does vLLM expose Prometheus metrics, and which request/queue gauges should an operator watch under load?",
        "keywords": ["metrics", "prometheus", "num_requests", "gauge"],
        "paths": ["docs/design/metrics.md", "docs/usage/metrics.md", "docs/serving/online_serving/README.md"],
        "max_tokens": 280,
    },
    {
        "id": "openai-server",
        "tier": "lookup",
        "question": "How do I start the OpenAI-compatible API server, and which host/port flags matter for a single-node demo?",
        "keywords": ["api_server", "openai", "host", "port", "serve"],
        "paths": [
            "docs/getting_started/quickstart.md",
            "docs/serving/online_serving/README.md",
            "docs/usage/openai_compatible_server.md",
        ],
        "max_tokens": 240,
    },
    {
        "id": "engine-args-overview",
        "tier": "multi",
        "question": "Summarize the engine arguments that most affect memory and concurrency: gpu-memory-utilization, max-model-len, max-num-seqs, and max-num-batched-tokens. Give one line each.",
        "keywords": [
            "gpu_memory_utilization",
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
        ],
        "paths": [
            "docs/configuration/engine_args.md",
            "docs/configuration/optimization.md",
            "docs/configuration/conserving_memory.md",
        ],
        "max_tokens": 420,
        "n_chunks": 6,
    },
    {
        "id": "prefill-decode-tradeoff",
        "tier": "multi",
        "question": "Using the docs, explain how chunked prefill and max-num-batched-tokens interact with decode latency when prompts are long.",
        "keywords": ["chunked", "prefill", "batched", "decode", "latency"],
        "paths": [
            "docs/configuration/optimization.md",
            "docs/design/prefix_caching.md",
            "docs/configuration/engine_args.md",
        ],
        "max_tokens": 400,
        "n_chunks": 5,
    },
    {
        "id": "memory-pressure",
        "tier": "multi",
        "question": "An operator sees KV cache near full and rising preemptions. Which configuration knobs and design docs should they read first, and what do those knobs change?",
        "keywords": ["preempt", "kv", "memory", "swap", "gpu_memory"],
        "paths": [
            "docs/configuration/conserving_memory.md",
            "docs/design/paged_attention.md",
            "docs/design/hybrid_kv_cache_manager.md",
            "docs/configuration/engine_args.md",
        ],
        "max_tokens": 450,
        "n_chunks": 6,
    },
    {
        "id": "prefix-vs-chunked",
        "tier": "multi",
        "question": "Compare prefix caching and chunked prefill: what problem each solves, and can both be enabled together according to the docs?",
        "keywords": ["prefix", "caching", "chunked", "prefill"],
        "paths": [
            "docs/design/prefix_caching.md",
            "docs/configuration/optimization.md",
            "docs/features/automatic_prefix_caching.md",
        ],
        "max_tokens": 400,
        "n_chunks": 5,
    },
    {
        "id": "distributed-serving",
        "tier": "multi",
        "question": "What are the main distributed serving options in vLLM (TP/PP/EP as documented), and when is single-GPU enough?",
        "keywords": ["tensor", "pipeline", "parallel", "distributed", "expert"],
        "paths": [
            "docs/serving/distributed_serving.md",
            "docs/serving/expert_parallel_deployment.md",
            "docs/usage/distributed.md",
        ],
        "max_tokens": 420,
        "n_chunks": 5,
    },
    {
        "id": "optimization-guide",
        "tier": "multi",
        "question": "From the optimization and configuration docs, list five practical steps to improve throughput before buying another GPU.",
        "keywords": ["optimization", "throughput", "batch", "prefix", "memory"],
        "paths": [
            "docs/configuration/optimization.md",
            "docs/configuration/conserving_memory.md",
            "docs/design/prefix_caching.md",
        ],
        "max_tokens": 450,
        "n_chunks": 6,
    },
    {
        "id": "hard-hybrid-kv",
        "tier": "hard",
        "question": "Walk through how the hybrid KV cache manager relates to paged attention and what an operator should verify when serving hybrid (attention + SSM/Mamba) models.",
        "keywords": ["hybrid", "kv", "mamba", "ssm", "paged", "block"],
        "paths": [
            "docs/design/hybrid_kv_cache_manager.md",
            "docs/design/paged_attention.md",
            "docs/configuration/conserving_memory.md",
        ],
        "max_tokens": 550,
        "n_chunks": 8,
    },
    {
        "id": "hard-cuda-graphs",
        "tier": "hard",
        "question": "Explain CUDA graphs in vLLM from the design docs: when they help, what they capture, and what can prevent graph capture from succeeding.",
        "keywords": ["cuda", "graph", "capture", "decode"],
        "paths": ["docs/design/cuda_graphs.md", "docs/configuration/optimization.md"],
        "max_tokens": 550,
        "n_chunks": 7,
    },
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_docs(clone_dir: Path, git_sha: str) -> tuple[Path, str]:
    docs = clone_dir / "docs"
    if docs.is_dir() and any(docs.rglob("*.md")):
        sha = git_sha
        try:
            sha = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=clone_dir, text=True
                ).strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return docs, sha

    clone_dir.mkdir(parents=True, exist_ok=True)
    if not (clone_dir / ".git").exists():
        run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--sparse",
                VLLM_REPO,
                str(clone_dir),
            ]
        )
        run(["git", "sparse-checkout", "set", "docs"], cwd=clone_dir)
    run(["git", "fetch", "--depth", "1", "origin", git_sha], cwd=clone_dir)
    run(["git", "checkout", git_sha], cwd=clone_dir)
    run(["git", "sparse-checkout", "set", "docs"], cwd=clone_dir)
    if not docs.is_dir():
        sys.exit(f"docs/ missing after checkout at {clone_dir}")
    return docs, git_sha


def strip_md(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_file(rel_path: str, text: str) -> list[dict]:
    clean = strip_md(text)
    if len(clean) < 80:
        return []
    chunks = []
    i = 0
    n = 0
    while i < len(clean):
        piece = clean[i : i + CHUNK_CHARS].strip()
        if len(piece) >= 80:
            n += 1
            chunks.append(
                {
                    "path": rel_path,
                    "chunk_id": f"{rel_path}#{n}",
                    "text": piece,
                }
            )
        if i + CHUNK_CHARS >= len(clean):
            break
        i += CHUNK_CHARS - CHUNK_OVERLAP
    return chunks


def load_corpus(docs_dir: Path) -> list[dict]:
    """Chunk a docs tree. Paths always look like docs/... for stable grounding."""
    docs_dir = docs_dir.resolve()
    # Bare --docs-dir pointing at the docs root: prefix with that directory name.
    # Clone layout (.../vllm/docs): also docs/... via name.
    path_prefix = docs_dir.name
    chunks: list[dict] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".mdx"}:
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        rel = f"{path_prefix}/{path.relative_to(docs_dir).as_posix()}"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunks.extend(chunk_file(rel, text))
    return chunks


def score_chunk(chunk: dict, keywords: list[str], preferred_paths: list[str]) -> float:
    text = chunk["text"].lower()
    path = chunk["path"].lower()
    score = 0.0
    for kw in keywords:
        k = kw.lower()
        score += 3.0 * text.count(k)
        if k.replace("_", "-") in path or k.replace("-", "_") in path:
            score += 5.0
    for pref in preferred_paths:
        p = pref.lower()
        if path.endswith(p) or path == p or path.endswith(p.split("/")[-1]):
            score += 25.0
        elif p in path:
            score += 10.0
    return score


def _path_preferred(path: str, preferred_paths: list[str]) -> bool:
    pl = path.lower()
    for pref in preferred_paths:
        p = pref.lower()
        if pl == p or pl.endswith(p) or pl.endswith(p.split("/")[-1]):
            return True
    return False


def pick_chunks(
    corpus: list[dict],
    keywords: list[str],
    preferred_paths: list[str],
    n: int,
) -> list[dict]:
    ranked = sorted(
        ((score_chunk(c, keywords, preferred_paths), c) for c in corpus),
        key=lambda x: x[0],
        reverse=True,
    )
    picked: list[dict] = []
    seen_ids: set[str] = set()

    def take(c: dict) -> None:
        if c["chunk_id"] in seen_ids:
            return
        picked.append({"path": c["path"], "chunk_id": c["chunk_id"], "text": c["text"]})
        seen_ids.add(c["chunk_id"])

    # Seed from preferred paths so lookup questions stay on-doc.
    pref_ranked = [
        (score_chunk(c, keywords, preferred_paths), c)
        for c in corpus
        if _path_preferred(c["path"], preferred_paths)
    ]
    pref_ranked.sort(key=lambda x: x[0], reverse=True)
    seed_n = min(n, max(1, n // 2) if pref_ranked else 0)
    for _, c in pref_ranked:
        take(c)
        if len(picked) >= seed_n:
            break

    for score, c in ranked:
        if len(picked) >= n:
            break
        if score <= 0:
            break
        take(c)
    return picked[:n]


def format_excerpts(excerpts: list[dict]) -> str:
    parts = []
    for i, ex in enumerate(excerpts, 1):
        parts.append(f"[Excerpt {i} | {ex['path']}]\n{ex['text']}")
    return "\n\n".join(parts)


def tier_defaults(tier: str) -> int:
    return {"lookup": 3, "multi": 5, "hard": 8}.get(tier, 3)


# Follow-ups keep the same topic; runner may re-subsample excerpts (re-retrieve).
DEFAULT_FOLLOWUPS = [
    "Name the exact CLI flag(s) and any default the excerpts state.",
    "What breaks or gets slow if I set this wrong in production?",
    "Give me three on-call bullets: check, change, verify.",
    "Is there a related knob I should set at the same time?",
]


def make_task(
    corpus: list[dict],
    *,
    eid: str,
    tier: str,
    question: str,
    keywords: list[str],
    paths: list[str],
    max_tokens: int,
    n: int,
    follow_ups: list[str] | None = None,
    vague: bool = False,
) -> dict | None:
    # Pool is larger than n so the runner can jitter retrieval per request.
    pool_n = min(len(corpus), max(n * 2, n + 2))
    pool = pick_chunks(corpus, keywords, paths, pool_n)
    if not pool and not vague:
        return None
    if vague and len(pool) < 1:
        # Off-doc / weak retrieve: take a few arbitrary mid-corpus chunks.
        mid = len(corpus) // 2
        pool = [
            {
                "path": corpus[i]["path"],
                "chunk_id": corpus[i]["chunk_id"],
                "text": corpus[i]["text"],
            }
            for i in range(mid, min(mid + 2, len(corpus)))
        ]
    if not pool:
        return None
    excerpts = pool[:n] if len(pool) >= n else pool
    fus = follow_ups if follow_ups is not None else list(DEFAULT_FOLLOWUPS)
    return {
        "id": eid,
        "tier": tier,
        "question": question,
        "excerpts": excerpts,
        "excerpt_pool": pool,
        "follow_ups": fus,
        "max_tokens": max_tokens,
        "vague": vague,
    }


def build_tasks(corpus: list[dict]) -> list[dict]:
    tasks = []
    by_tier = {"lookup": 0, "multi": 0, "hard": 0}
    for q in QUESTION_BANK:
        n = int(q.get("n_chunks", tier_defaults(q["tier"])))
        task = make_task(
            corpus,
            eid=q["id"],
            tier=q["tier"],
            question=q["question"],
            keywords=q["keywords"],
            paths=q.get("paths", []),
            max_tokens=q.get("max_tokens", 256),
            n=n,
            follow_ups=q.get("follow_ups"),
        )
        if not task:
            print(f"warning: no excerpts for {q['id']}", file=sys.stderr)
            continue
        tasks.append(task)
        by_tier[q["tier"]] = by_tier.get(q["tier"], 0) + 1

    # Expand lookup/multi by rotating extra keyword variants from engine_args
    # so the pack has enough distinct prompts for sustained load without
    # inventing fake docs.
    extras = [
        (
            "enforce-eager",
            "lookup",
            "When should I use --enforce-eager, and what performance tradeoff does it make?",
            ["enforce_eager", "eager", "cuda", "graph"],
            ["docs/configuration/engine_args.md", "docs/design/cuda_graphs.md"],
            220,
            3,
        ),
        (
            "dtype-flag",
            "lookup",
            "What does the --dtype flag control for model weights, and how is it different from KV cache dtype?",
            ["dtype", "kv_cache_dtype", "bfloat16", "float16"],
            ["docs/configuration/engine_args.md"],
            220,
            3,
        ),
        (
            "swap-space",
            "lookup",
            "What is --swap-space used for, and when does CPU swap help versus hurt under KV pressure?",
            ["swap", "cpu", "kv", "preempt"],
            ["docs/configuration/engine_args.md", "docs/configuration/conserving_memory.md"],
            240,
            3,
        ),
        (
            "block-size",
            "lookup",
            "What is KV block size in vLLM, and how does it relate to paged attention?",
            ["block_size", "block", "paged", "kv"],
            ["docs/design/paged_attention.md", "docs/configuration/engine_args.md"],
            240,
            3,
        ),
        (
            "tokenizer-mode",
            "lookup",
            "What tokenizer modes does vLLM document, and when would auto vs slow matter?",
            ["tokenizer", "mode", "auto"],
            ["docs/configuration/engine_args.md", "docs/models/README.md"],
            200,
            2,
        ),
        (
            "guided-decoding",
            "lookup",
            "What is guided decoding / structured output support as described in the docs?",
            ["guided", "structured", "decoding", "grammar"],
            ["docs/features/structured_outputs.md", "docs/serving/openai_compatible_server.md"],
            260,
            3,
        ),
        (
            "lora-serving",
            "lookup",
            "How does vLLM serve LoRA adapters according to the docs, and which flags enable it?",
            ["lora", "adapter", "enable_lora"],
            ["docs/features/lora.md", "docs/configuration/engine_args.md"],
            260,
            3,
        ),
        (
            "spec-decode",
            "multi",
            "Summarize speculative decoding in vLLM: when to use it and what components are involved.",
            ["speculat", "draft", "decode"],
            ["docs/features/speculative_decoding.md", "docs/configuration/engine_args.md"],
            400,
            5,
        ),
        (
            "disagg-prefill",
            "hard",
            "What do the docs say about disaggregated / separated prefill and decode serving, and what operational pieces are required?",
            ["disaggregat", "prefill", "decode", "kv", "connector"],
            [
                "docs/features/disagg_prefill.md",
                "docs/serving/distributed_serving.md",
                "docs/design/nixl_kv_push_connector.md",
            ],
            550,
            8,
        ),
        (
            "quantization-overview",
            "multi",
            "Which weight quantization approaches does vLLM document for serving, and what should an operator verify before enabling one?",
            ["quant", "awq", "gptq", "fp8", "bitsandbytes"],
            ["docs/features/quantization/README.md", "docs/configuration/engine_args.md"],
            400,
            5,
        ),
        (
            "sleep-mode",
            "lookup",
            "What is sleep mode / engine sleep in vLLM, and when would an operator use it?",
            ["sleep", "wake", "idle"],
            ["docs/features/sleep_mode.md", "docs/configuration/engine_args.md"],
            220,
            3,
        ),
        (
            "tool-calling",
            "lookup",
            "How does tool calling work on the OpenAI-compatible server, and which parser flags matter?",
            ["tool", "parser", "function", "auto"],
            [
                "docs/features/tool_calling.md",
                "docs/serving/openai_compatible_server.md",
                "docs/configuration/serve_args.md",
            ],
            260,
            3,
        ),
        (
            "multimodal-serve",
            "lookup",
            "What do the docs say about serving multimodal models, and which limits should I watch?",
            ["multimodal", "image", "mm", "limit"],
            ["docs/models/multimodal.md", "docs/configuration/engine_args.md"],
            260,
            3,
        ),
        (
            "ray-serve",
            "multi",
            "How does vLLM integrate with Ray Serve / distributed deployment according to the docs?",
            ["ray", "serve", "distributed", "deployment"],
            ["docs/deployment/frameworks/ray_serve.md", "docs/serving/distributed_serving.md"],
            380,
            5,
        ),
        (
            "env-vars",
            "lookup",
            "Which environment variables does vLLM document for debugging or performance, and name three that matter under load?",
            ["VLLM_", "environment", "env"],
            ["docs/configuration/env_vars.md"],
            280,
            3,
        ),
        (
            "batch-invariant",
            "lookup",
            "What is batch invariance / deterministic behavior in vLLM as far as the docs describe it?",
            ["invariant", "determinist", "batch"],
            ["docs/features/batch_invariance.md", "docs/design/vllm_ir.md"],
            240,
            3,
        ),
        (
            "logprobs",
            "lookup",
            "How do I request logprobs from the OpenAI-compatible API according to the docs?",
            ["logprob", "prompt_logprobs", "openai"],
            ["docs/serving/openai_compatible_server.md", "docs/usage/openai_compatible_server.md"],
            220,
            3,
        ),
        (
            "preemption-mode",
            "multi",
            "What preemption modes exist, and how should an operator choose when KV is tight?",
            ["preempt", "recompute", "swap"],
            ["docs/configuration/engine_args.md", "docs/configuration/conserving_memory.md"],
            360,
            5,
        ),
        (
            "compile-cache",
            "lookup",
            "What do the torch.compile / compilation docs say about warmup cost versus steady-state decode?",
            ["compile", "torch", "warmup", "graph"],
            ["docs/design/torch_compile.md", "docs/configuration/optimization.md"],
            260,
            3,
        ),
    ]
    for eid, tier, question, keywords, paths, max_tokens, n in extras:
        task = make_task(
            corpus,
            eid=eid,
            tier=tier,
            question=question,
            keywords=keywords,
            paths=paths,
            max_tokens=max_tokens,
            n=n,
        )
        if not task:
            continue
        tasks.append(task)
        by_tier[tier] = by_tier.get(tier, 0) + 1

    # Vague / poorly specified asks (small fraction). Weak or off-topic retrieve
    # on purpose — real internal search includes these.
    vague_qs = [
        (
            "vague-slow",
            "Things are slow and the GPU looks busy. What should I tweak first?",
            ["latency", "throughput", "optimization", "batch"],
            ["docs/configuration/optimization.md"],
        ),
        (
            "vague-oom",
            "We keep OOMing somehow. Is it the model or the cache? Help.",
            ["memory", "oom", "kv", "gpu"],
            ["docs/configuration/conserving_memory.md"],
        ),
        (
            "vague-whats-prefix",
            "Someone said turn on prefix something. What is that and do I want it?",
            ["prefix", "caching"],
            ["docs/design/prefix_caching.md"],
        ),
        (
            "vague-offdoc",
            "How do I connect vLLM to our private Okta group sync for doc ACLs?",
            ["auth", "acl", "token"],
            ["docs/serving/openai_compatible_server.md"],
        ),
    ]
    for eid, question, keywords, paths in vague_qs:
        task = make_task(
            corpus,
            eid=eid,
            tier="lookup",
            question=question,
            keywords=keywords,
            paths=paths,
            max_tokens=220,
            n=2,
            vague=True,
        )
        if not task:
            continue
        tasks.append(task)
        by_tier["lookup"] = by_tier.get("lookup", 0) + 1

    print(
        f"tasks: {len(tasks)} "
        f"(lookup={by_tier.get('lookup', 0)} "
        f"multi={by_tier.get('multi', 0)} "
        f"hard={by_tier.get('hard', 0)})"
    )
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "rag-tasks.json",
    )
    ap.add_argument("--docs-dir", type=Path, default=None, help="Existing vllm/docs tree")
    ap.add_argument("--clone-dir", type=Path, default=None)
    ap.add_argument("--git-sha", default=DEFAULT_GIT_SHA)
    ap.add_argument(
        "--write-corpus",
        type=Path,
        default=None,
        help="Optional path to write chunk corpus JSONL (large).",
    )
    args = ap.parse_args()

    if args.docs_dir:
        docs_dir = args.docs_dir
        git_sha = args.git_sha
        if not docs_dir.is_dir():
            sys.exit(f"--docs-dir not found: {docs_dir}")
    else:
        clone_dir = args.clone_dir or Path(tempfile.mkdtemp(prefix="vllm-docs-"))
        print(f"Ensuring docs at {clone_dir} @ {args.git_sha}...")
        docs_dir, git_sha = ensure_docs(clone_dir, args.git_sha)

    print(f"Chunking {docs_dir}...")
    corpus = load_corpus(docs_dir)
    print(f"corpus chunks: {len(corpus)}")
    if len(corpus) < 50:
        sys.exit("corpus too small; check docs path")

    if args.write_corpus:
        with args.write_corpus.open("w", encoding="utf-8") as f:
            for c in corpus:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"wrote corpus {args.write_corpus}")

    tasks = build_tasks(corpus)
    if len(tasks) < 10:
        sys.exit("too few tasks grounded; aborting")

    payload = {
        "version": 2,
        "source": {
            "repo": VLLM_REPO,
            "git_sha": git_sha,
            "docs_path": "docs",
            "chunk_chars": CHUNK_CHARS,
            "chunk_overlap": CHUNK_OVERLAP,
            "corpus_chunks": len(corpus),
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "system_prompt": SYSTEM_PROMPT,
        "retrieval_template": RETRIEVAL_TEMPLATE,
        # Runner defaults (overridable via env on rag-load.py).
        "load_defaults": {
            "session_multi_turn_frac": 0.40,
            "turns_min": 2,
            "turns_max": 4,
            "burst_frac": 0.10,
        },
        "tasks": tasks,
        "checksum": hashlib.sha256(
            json.dumps(
                [{"id": t["id"], "question": t["question"]} for t in tasks],
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()[:16],
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
