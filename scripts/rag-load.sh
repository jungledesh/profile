#!/usr/bin/env bash
# vLLM-docs RAG load for the Profile AMD demo.
#
# Real public vLLM documentation excerpts + shared system prompt / retrieval
# template (prefix-cache lever). Multi-turn sessions, excerpt jitter, and
# heterogeneous think times. Not a scored RAG benchmark. Not random filler.
#
# Usage:
#   ./rag-load.sh              # setup check, then run
#   ./rag-load.sh setup        # verify/regenerate rag-tasks.json if missing
#   ./rag-load.sh run          # fire load (setup already done)
#
# Knobs (env):
#   WORKERS=16              concurrent clients (default 16)
#   LAMBDA=1.0              base mean think time (s) between sessions; per-worker jitter applied
#   DURATION=0              seconds to run (0 = until Ctrl-C; use 0 during profile)
#   MULTI_TURN_FRAC=0.4     fraction of sessions that continue with follow-ups
#   TURNS_MIN=2 TURNS_MAX=4 follow-up session length when multi-turn
#   BURST_FRAC=0.1          chance of near-back-to-back sessions (bursty arrivals)
#   VLLM_URL=http://localhost:8000
#   PROFILE_MODEL=qwen|gemma|llama
#   SERVED_NAME=...         wins over PROFILE_MODEL family default
#   MODEL=...               wins over SERVED_NAME
#   RAG_TASKS=path          override task pack (default: ./rag-tasks.json)
#   REFETCH=1               setup regenerates the pack (needs git + network)
#
# Profile demo (steady RAG traffic):
#   WORKERS=32 LAMBDA=0.5 DURATION=0 ./rag-load.sh run
#   Wait ~30s for in-flight to fill, then run profile diagnose.
#
# Requires: vLLM on localhost:8000; python3 + aiohttp; rag-tasks.json
# (committed, or produced by fetch-vllm-docs-rag.py).

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_JSON="${RAG_TASKS:-$SCRIPT_DIR/rag-tasks.json}"
FETCHER="$SCRIPT_DIR/fetch-vllm-docs-rag.py"
RUNNER="$SCRIPT_DIR/rag-load.py"
CMD="${1:-all}"

do_setup() {
    if [[ ! -f "$TASKS_JSON" ]] || [[ "${REFETCH:-0}" == "1" ]]; then
        if [[ ! -f "$FETCHER" ]]; then
            echo "missing $FETCHER and no task pack at $TASKS_JSON" >&2
            exit 1
        fi
        echo "Building rag-tasks.json from pinned vLLM docs (git + network)..."
        python3 "$FETCHER" --out "$TASKS_JSON"
    else
        echo "Using existing task pack: $TASKS_JSON"
    fi
    RAG_TASKS="$TASKS_JSON" python3 - <<'PY'
import json, os
p = os.environ["RAG_TASKS"]
d = json.load(open(p, encoding="utf-8"))
assert d.get("system_prompt"), "system_prompt missing"
assert d.get("retrieval_template"), "retrieval_template missing"
assert d.get("tasks"), "tasks missing"
n = len(d["tasks"])
pools = sum(1 for t in d["tasks"] if t.get("excerpt_pool"))
follows = sum(1 for t in d["tasks"] if t.get("follow_ups"))
print(
    f"ok: {n} tasks (v{d.get('version', '?')}), "
    f"excerpt_pool={pools}, follow_ups={follows}, "
    f"sha={d.get('source', {}).get('git_sha', '?')[:12]}"
)
PY
}

do_run() {
    [[ -f "$TASKS_JSON" ]] || {
        echo "missing $TASKS_JSON; run: ./rag-load.sh setup" >&2
        exit 1
    }
    [[ -f "$RUNNER" ]] || {
        echo "missing $RUNNER" >&2
        exit 1
    }
    export RAG_TASKS="$TASKS_JSON"
    export PROFILE_MODEL="${PROFILE_MODEL:-qwen}"
    exec python3 "$RUNNER" \
        --workers "${WORKERS:-16}" \
        --lambda "${LAMBDA:-1.0}" \
        --duration "${DURATION:-0}"
}

case "$CMD" in
    setup) do_setup ;;
    run)   do_run ;;
    all)   do_setup; do_run ;;
    *)
        echo "usage: $0 [setup|run|all]" >&2
        exit 1
        ;;
esac
