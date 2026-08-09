#!/usr/bin/env python3
"""vLLM-docs RAG load runner for Profile (AMD demo).

Reads frozen rag-tasks.json (real docs excerpts + shared system/template).
Closer to production-shaped traffic than v1:

  - Multi-turn sessions (history grows; real assistant text carried forward)
  - Excerpt jitter from excerpt_pool (re-retrieve feel, still real chunks)
  - Per-worker think-time heterogeneity + occasional bursts
  - HTTP status counted separately from transport errors

Usage:
  ./rag-load.py
  WORKERS=32 LAMBDA=0.5 ./rag-load.py
  MULTI_TURN_FRAC=0.4 TURNS_MAX=4 ./rag-load.py

Requires: aiohttp, rag-tasks.json (or RAG_TASKS=path).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

try:
    import aiohttp
except ImportError:  # pragma: no cover - present in vLLM images
    aiohttp = None  # type: ignore

FAMILY_DEFAULTS = {
    "gemma": "gemma-4-26b-a4b",
    "qwen": "Qwen3.6-27B",
    "llama": "llama3",
}


def resolve_model() -> str:
    profile = os.environ.get("PROFILE_MODEL", "qwen")
    family = FAMILY_DEFAULTS.get(profile)
    if family is None:
        sys.exit(f"PROFILE_MODEL must be gemma|qwen|llama (got: {profile})")
    return os.environ.get("MODEL") or os.environ.get("SERVED_NAME") or family


def format_excerpts(excerpts: list[dict]) -> str:
    parts = []
    for i, ex in enumerate(excerpts, 1):
        parts.append(f"[Excerpt {i} | {ex['path']}]\n{ex['text']}")
    return "\n\n".join(parts)


def sample_excerpts(task: dict, rng: random.Random) -> list[dict]:
    """Subsample/shuffle from excerpt_pool when present (frozen offline retrieve)."""
    pool = task.get("excerpt_pool") or task.get("excerpts") or []
    if not pool:
        return []
    base_n = len(task.get("excerpts") or pool)
    # Keep at least 1; jitter count ±1 around the frozen size.
    n = max(1, min(len(pool), base_n + rng.choice([-1, 0, 0, 0, 1])))
    if len(pool) <= n:
        picked = list(pool)
    else:
        picked = rng.sample(pool, n)
    rng.shuffle(picked)
    return picked


def build_user_message(template: str, question: str, excerpts: list[dict]) -> str:
    return template.format(
        excerpts=format_excerpts(excerpts),
        question=question,
    )


async def chat_completion(
    session,  # aiohttp.ClientSession
    url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    stats: dict,
) -> str | None:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }
    try:
        async with session.post(url, json=body) as resp:
            raw = await resp.read()
            if resp.status >= 400:
                stats["http_err"] += 1
                return None
            stats["http_ok"] += 1
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                stats["parse_err"] += 1
                return None
            choices = data.get("choices") or []
            if not choices:
                return None
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if content is None:
                return None
            return str(content)
    except (aiohttp.ClientError, asyncio.TimeoutError):
        stats["transport_err"] += 1
        return None


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


async def worker(
    wid: int,
    session,
    url: str,
    model: str,
    system_prompt: str,
    template: str,
    tasks: list[dict],
    mean_think: float,
    multi_turn_frac: float,
    turns_min: int,
    turns_max: int,
    burst_frac: float,
    stop_at: float | None,
    stats: dict,
) -> None:
    rng = random.Random(wid * 9973 + 13)
    # Heterogeneous mean think time per worker (production user mix).
    worker_lambda = mean_think * rng.uniform(0.45, 1.85) if mean_think > 0 else 0.0
    order = list(range(len(tasks)))
    rng.shuffle(order)
    i = 0

    async def think(bursting: bool) -> None:
        if worker_lambda <= 0:
            return
        if bursting:
            await asyncio.sleep(rng.uniform(0.01, 0.08))
            return
        delay = rng.expovariate(1.0 / worker_lambda)
        await asyncio.sleep(min(delay, worker_lambda * 5))

    while True:
        if stop_at is not None and time.time() >= stop_at:
            return

        task = tasks[order[i % len(order)]]
        i += 1
        multi = rng.random() < multi_turn_frac and bool(task.get("follow_ups"))
        n_turns = 1
        if multi:
            n_turns = rng.randint(max(1, turns_min), max(turns_min, turns_max))

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        follow_ups = list(task.get("follow_ups") or [])
        rng.shuffle(follow_ups)

        for turn in range(n_turns):
            if stop_at is not None and time.time() >= stop_at:
                return
            if turn == 0:
                question = task["question"]
            else:
                question = follow_ups[(turn - 1) % len(follow_ups)]
            excerpts = sample_excerpts(task, rng)
            user = build_user_message(template, question, excerpts)
            messages.append({"role": "user", "content": user})
            # Later turns: shorter decode; first turn uses task budget.
            max_tokens = int(task.get("max_tokens", 256))
            if turn > 0:
                max_tokens = min(max_tokens, 180)

            content = await chat_completion(
                session, url, model, messages, max_tokens, stats
            )
            stats["requests"] += 1
            if content:
                messages.append({"role": "assistant", "content": content})
                stats["sessions_turns"] += 1
            else:
                # Broken turn: drop session, avoid poisoning history.
                break

        stats["sessions"] += 1
        bursting = rng.random() < burst_frac
        # Small burst: start next session almost immediately (same worker).
        await think(bursting)
        if bursting and rng.random() < 0.5:
            await think(True)


async def async_main(args: argparse.Namespace) -> int:
    if aiohttp is None:
        sys.exit("aiohttp required (pip install aiohttp); present in vLLM envs.")

    tasks_path = Path(
        os.environ.get("RAG_TASKS", Path(__file__).resolve().parent / "rag-tasks.json")
    )
    if not tasks_path.is_file():
        print(
            f"missing {tasks_path}; run: python3 fetch-vllm-docs-rag.py",
            file=sys.stderr,
        )
        return 1
    pack = json.loads(tasks_path.read_text(encoding="utf-8"))
    system_prompt = pack["system_prompt"]
    template = pack["retrieval_template"]
    tasks = pack["tasks"]
    if not tasks:
        print("rag-tasks.json has no tasks", file=sys.stderr)
        return 1

    defaults = pack.get("load_defaults") or {}
    model = resolve_model()
    base = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
    url = f"{base}/v1/chat/completions"
    workers = env_int("WORKERS", args.workers)
    mean_think = env_float("LAMBDA", args.lambda_s)
    duration = env_float("DURATION", args.duration)
    multi_turn_frac = env_float(
        "MULTI_TURN_FRAC", float(defaults.get("session_multi_turn_frac", 0.40))
    )
    turns_min = env_int("TURNS_MIN", int(defaults.get("turns_min", 2)))
    turns_max = env_int("TURNS_MAX", int(defaults.get("turns_max", 4)))
    burst_frac = env_float("BURST_FRAC", float(defaults.get("burst_frac", 0.10)))
    stop_at = time.time() + duration if duration > 0 else None

    print(
        f"RAG load: model={model} workers={workers} lambda={mean_think}s "
        f"multi_turn={multi_turn_frac:.0%} turns={turns_min}-{turns_max} "
        f"burst={burst_frac:.0%} tasks={len(tasks)} pack={tasks_path.name} "
        f"v{pack.get('version', '?')} sha={pack.get('source', {}).get('git_sha', '?')[:12]}"
    )
    print(f"endpoint: {url}")
    print("Ctrl-C to stop." if stop_at is None else f"duration: {duration:.0f}s")

    stats = {
        "requests": 0,
        "http_ok": 0,
        "http_err": 0,
        "transport_err": 0,
        "parse_err": 0,
        "sessions": 0,
        "sessions_turns": 0,
    }
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        try:
            async with session.get(f"{base}/v1/models") as resp:
                if resp.status >= 400:
                    print(f"vLLM /v1/models status {resp.status}", file=sys.stderr)
                    return 1
        except aiohttp.ClientError as e:
            print(f"vLLM not reachable at {base}: {e}", file=sys.stderr)
            return 1

        jobs = [
            asyncio.create_task(
                worker(
                    wid,
                    session,
                    url,
                    model,
                    system_prompt,
                    template,
                    tasks,
                    mean_think,
                    multi_turn_frac,
                    turns_min,
                    turns_max,
                    burst_frac,
                    stop_at,
                    stats,
                )
            )
            for wid in range(workers)
        ]
        try:
            await asyncio.gather(*jobs)
        except asyncio.CancelledError:
            pass

    print(
        "stats: "
        f"sessions={stats['sessions']} "
        f"requests={stats['requests']} "
        f"http_ok={stats['http_ok']} "
        f"http_err={stats['http_err']} "
        f"transport_err={stats['transport_err']} "
        f"parse_err={stats['parse_err']}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--lambda",
        dest="lambda_s",
        type=float,
        default=1.0,
        help="Base mean think time seconds between sessions per worker",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to run (0 = until Ctrl-C)",
    )
    args = ap.parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nstopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
