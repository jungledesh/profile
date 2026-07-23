#!/usr/bin/env bash
# Agent swarm for the Profile demo: real Grok Build agents working real
# SWE-bench Verified issues (psf/requests + pytest-dev/pytest) against local
# vLLM. No choreography: each agent gets a pinned pre-fix checkout and the
# original issue text, then does whatever the fix needs. Profile diagnoses
# whatever traffic emerges.
#
# Wording rule (charter): we say "agents working SWE-bench Verified issues",
# never "we run SWE-bench". No harness, no scoring, no pass@k.
#
# Usage:
#   ./agent-swarm.sh            # setup, then launch the swarm
#   ./agent-swarm.sh setup      # install grok + clone repos + checkouts + venvs
#   ./agent-swarm.sh run        # launch the swarm (setup already done)
#
# Requires next to this script: swarm-tasks.json
#   (generate once with: python3 fetch-swarm-tasks.py)
#
# Knobs (env):
#   AGENTS=16          concurrent agent workers
#   STAGGER=5          seconds between worker launches
#   DURATION=0         total seconds to run (0 = until Ctrl-C)
#   TASK_TIMEOUT=600   max seconds per task before the worker moves on
#   SWARM_HOME=/workspace/swarm   scratch area (clones, checkouts, venvs, log)
#   MODEL_ALIAS=qwen-local        grok model alias (see config below)
#
# Requires: vLLM serving Qwen3.6-27B on localhost:8000 (see start.sh); jq; git.
#
# Smoke gate (non-negotiable, from the plan): on the pod, run ONE agent on ONE
# instance end to end before AGENTS=16:
#   AGENTS=1 DURATION=900 ./agent-swarm.sh run

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO"' ERR

AGENTS="${AGENTS:-16}"
STAGGER="${STAGGER:-5}"
DURATION="${DURATION:-0}"
TASK_TIMEOUT="${TASK_TIMEOUT:-600}"
SWARM_HOME="${SWARM_HOME:-/workspace/swarm}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen-local}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_JSON="${TASKS_JSON:-$SCRIPT_DIR/swarm-tasks.json}"

CLONES_DIR="$SWARM_HOME/clones"
CHECKOUTS_DIR="$SWARM_HOME/checkouts"
AGENTS_DIR="$SWARM_HOME/agents"
SWARM_LOG="$SWARM_HOME/swarm.log"

export PATH="$HOME/.local/bin:$PATH"

# ── Grok Build binary ───────────────────────────────────────────────────────
install_grok() {
    if ! command -v grok >/dev/null 2>&1; then
        echo "Installing Grok Build..."
        curl -fsSL https://x.ai/cli/install.sh | bash
        export PATH="$HOME/.local/bin:$PATH"
    fi
    command -v grok >/dev/null 2>&1 || { echo "grok not on PATH after install" >&2; exit 1; }
}

# ── Config (validated on H100, Jul 16 2026; see demo_setup_validated.md §5) ─
write_grok_config() {
    mkdir -p "$HOME/.grok"
    cat > "$HOME/.grok/config.toml" <<'EOF'
[models]
default = "qwen-local"

[model.qwen-local]
model = "Qwen3.6-27B"
base_url = "http://localhost:8000/v1"
api_key = "none"
name = "Qwen3.6-27B (vLLM)"
context_window = 32768

# Internal harness calls request the model named "grok-build" explicitly.
# Redirect it locally. api_backend must be chat_completions: the built-in
# defaults to the Responses API, which crashes on vLLM reasoning stream events.
[model.grok-build]
model = "Qwen3.6-27B"
base_url = "http://localhost:8000/v1"
api_key = "none"
api_backend = "chat_completions"
context_window = 32768

[features]
telemetry = false
EOF
    echo "Grok Build ready. Config written to ~/.grok/config.toml"
}

# ── Repos, checkouts, venvs ─────────────────────────────────────────────────
# Full clones (old base commits need full history), then one detached worktree
# per instance at its pinned base_commit: the commit where the bug exists.
#
# Shared venv per repo carries third-party deps ONLY. The code under test is
# NOT installed; it reaches Python via PYTHONPATH (requests: checkout root,
# pytest: checkout/src). Python 3.9 per SWE-bench's own env specs for both
# repos. Dep pins below follow those specs; expect to adjust on the pod, and
# DELETE instances from swarm-tasks.json that still refuse to import
# (load generation, not preservation of the 27).
clone_url() {
    case "$1" in
        psf/requests)      echo "https://github.com/psf/requests.git" ;;
        pytest-dev/pytest) echo "https://github.com/pytest-dev/pytest.git" ;;
        *) echo "unknown repo: $1" >&2; return 1 ;;
    esac
}

repo_slug() {  # psf/requests -> requests
    basename "$1"
}

setup_repos() {
    command -v jq >/dev/null 2>&1 || { echo "jq required" >&2; exit 1; }
    [[ -f "$TASKS_JSON" ]] || { echo "missing $TASKS_JSON; run: python3 $SCRIPT_DIR/fetch-swarm-tasks.py $TASKS_JSON" >&2; exit 1; }

    mkdir -p "$CLONES_DIR" "$CHECKOUTS_DIR"

    local repos
    repos=$(jq -r '[.[].repo] | unique | .[]' "$TASKS_JSON")
    local repo
    for repo in $repos; do
        local slug clone
        slug=$(repo_slug "$repo")
        clone="$CLONES_DIR/$slug"
        if [[ ! -d "$clone/.git" ]]; then
            echo "Cloning $repo..."
            git clone --quiet "$(clone_url "$repo")" "$clone"
        fi
    done

    local n i id base slug clone dst ver
    n=$(jq length "$TASKS_JSON")
    for (( i = 0; i < n; i++ )); do
        id=$(jq -r ".[$i].instance_id" "$TASKS_JSON")
        base=$(jq -r ".[$i].base_commit" "$TASKS_JSON")
        slug=$(repo_slug "$(jq -r ".[$i].repo" "$TASKS_JSON")")
        clone="$CLONES_DIR/$slug"
        dst="$CHECKOUTS_DIR/$id"
        if [[ ! -d "$dst" ]]; then
            git -C "$clone" worktree add --quiet --detach "$dst" "$base" \
                || { echo "worktree failed for $id at $base" >&2; exit 1; }
        fi
        # pytest generates _pytest/_version.py at install time; our
        # PYTHONPATH approach skips install, so shim it from the instance's
        # version field (imports need it to exist, not to be exact).
        if [[ "$slug" == "pytest" && ! -f "$dst/src/_pytest/_version.py" ]]; then
            ver=$(jq -r ".[$i].version" "$TASKS_JSON")
            printf 'version = "%s.0"\nversion_tuple = (%s, 0)\n' \
                "$ver" "${ver/./, }" > "$dst/src/_pytest/_version.py"
        fi
    done
    echo "Checkouts ready: $n instances under $CHECKOUTS_DIR"
}

setup_venvs() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    uv python install 3.9

    # requests: old pins (2013-2015) vendor their deps; new pins (2.22/2.26)
    # need the urllib3 family. pytest runs the tests here, so it IS a dep.
    if [[ ! -d "$SWARM_HOME/venv-requests" ]]; then
        uv venv --python 3.9 "$SWARM_HOME/venv-requests"
        uv pip install --python "$SWARM_HOME/venv-requests/bin/python" \
            pytest pytest-httpbin pytest-mock \
            "urllib3<2" "chardet==4.0.0" "charset_normalizer<3" "idna<3" certifi
    fi

    # pytest repo: deps per SWE-bench specs (4.6-7.x band). Deliberately NO
    # pytest package installed: the code under test provides it via
    # PYTHONPATH, and an installed one would shadow the entry point.
    # pluggy 0.13.1 is in-spec for 4.6-7.x (7.x declares >=0.12,<2); if 7.x
    # runs misbehave on the pod, split venvs by version there, not here.
    if [[ ! -d "$SWARM_HOME/venv-pytest" ]]; then
        uv venv --python 3.9 "$SWARM_HOME/venv-pytest"
        uv pip install --python "$SWARM_HOME/venv-pytest/bin/python" \
            "attrs" "iniconfig" "packaging" "pluggy==0.13.1" "py==1.11.0" \
            "more-itertools" "atomicwrites" "six" "wcwidth" "tomli" \
            "exceptiongroup" "hypothesis" "setuptools<69"
        # Agents type `pytest`; there is no installed entry point. Shim it
        # to the checkout's code via python -m (PYTHONPATH already set).
        cat > "$SWARM_HOME/venv-pytest/bin/pytest" <<'SHIM'
#!/usr/bin/env bash
exec "$(dirname "$0")/python" -m pytest "$@"
SHIM
        chmod +x "$SWARM_HOME/venv-pytest/bin/pytest"
    fi
}

# repo -> PYTHONPATH root inside a checkout copy
pythonpath_for() {
    case "$1" in
        psf/requests)      echo "" ;;      # package at checkout root
        pytest-dev/pytest) echo "src" ;;   # src layout
        *) echo "unsupported repo in pythonpath_for: $1" >&2; return 1 ;;
    esac
}

venv_for() {
    case "$1" in
        psf/requests)      echo "$SWARM_HOME/venv-requests" ;;
        pytest-dev/pytest) echo "$SWARM_HOME/venv-pytest" ;;
        *) echo "unsupported repo in venv_for: $1" >&2; return 1 ;;
    esac
}

# Import smoke: every checkout must import under its venv + PYTHONPATH.
# Any failure fails setup (exit 1): delete those instances from the JSON or
# fix deps, then rerun. Override with ALLOW_IMPORT_FAIL=1 to proceed anyway.
smoke_imports() {
    local n i id repo sub venv mod pp bad=0
    n=$(jq length "$TASKS_JSON")
    for (( i = 0; i < n; i++ )); do
        id=$(jq -r ".[$i].instance_id" "$TASKS_JSON")
        repo=$(jq -r ".[$i].repo" "$TASKS_JSON")
        sub=$(pythonpath_for "$repo")
        venv=$(venv_for "$repo")
        mod=$(repo_slug "$repo")
        pp="$CHECKOUTS_DIR/$id${sub:+/$sub}"
        if ! PYTHONPATH="$pp" "$venv/bin/python" -c "import $mod" 2>/dev/null; then
            echo "IMPORT FAIL: $id (drop it from swarm-tasks.json or fix deps)"
            bad=$(( bad + 1 ))
        fi
    done
    echo "Import smoke: $(( n - bad ))/$n OK"
    if (( bad > 0 )) && [[ "${ALLOW_IMPORT_FAIL:-0}" != "1" ]]; then
        echo "Setup FAILED: $bad instance(s) do not import. Fix or drop them (ALLOW_IMPORT_FAIL=1 to override)." >&2
        exit 1
    fi
}

# ── Swarm ───────────────────────────────────────────────────────────────────
END=0

worker() {
    local id_w="$1"
    local run=0
    local n order pos
    n=$(jq length "$TASKS_JSON")
    order=()
    pos=$n
    while true; do
        local now t
        now=$(date +%s)
        if (( END > 0 && now >= END - 30 )); then break; fi
        t="$TASK_TIMEOUT"
        if (( END > 0 && END - now < t )); then t=$(( END - now )); fi

        if (( pos >= n )); then
            mapfile -t order < <(shuf -i 0-$(( n - 1 )))
            pos=0
        fi
        local idx="${order[$pos]}"
        pos=$(( pos + 1 ))

        local iid repo statement sub venv pp dir
        iid=$(jq -r ".[$idx].instance_id" "$TASKS_JSON")
        repo=$(jq -r ".[$idx].repo" "$TASKS_JSON")
        statement=$(jq -r ".[$idx].problem_statement" "$TASKS_JSON")
        sub=$(pythonpath_for "$repo")
        venv=$(venv_for "$repo")

        dir="$AGENTS_DIR/agent${id_w}-run${run}"
        rm -rf "$dir"
        cp -r "$CHECKOUTS_DIR/$iid" "$dir"
        rm -f "$dir/.git"   # plain files for the agent; no shared-worktree git

        pp="$dir${sub:+/$sub}"
        local prompt="$statement

Fix this issue in this repository. Verify with the relevant failing tests only, not the full test suite. Run tests with: python -m pytest <test file>::<test>"

        local start rc=0
        start=$(date +%s)
        PYTHONPATH="$pp" PATH="$venv/bin:$PATH" \
            timeout --signal=TERM --kill-after=15 "$t" \
            grok -p "$prompt" -m "$MODEL_ALIAS" --yolo --disable-web-search \
                 --cwd "$dir" --output-format json \
                 > "$dir/grok.log" 2>&1 || rc=$?
        printf '%s worker=%s run=%d instance=%s rc=%d secs=%d\n' \
            "$(date -u +%FT%TZ)" "$id_w" "$run" "$iid" "$rc" \
            "$(( $(date +%s) - start ))" >> "$SWARM_LOG"
        run=$(( run + 1 ))
    done
}

WORKER_PIDS=()

cleanup() {
    trap - INT TERM
    echo
    echo "Stopping swarm..."
    for pid in "${WORKER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    # -x: match the grok process name exactly; a -f pattern would kill any
    # process whose command line merely contains the string.
    pkill -TERM -x grok 2>/dev/null || true
    wait 2>/dev/null || true
    summary
    exit 0
}

summary() {
    local total=0
    [[ -f "$SWARM_LOG" ]] && total=$(wc -l < "$SWARM_LOG")
    echo "Swarm done: $total task runs. Log: $SWARM_LOG  Agent dirs: $AGENTS_DIR"
}

run_swarm() {
    command -v grok >/dev/null 2>&1 || { echo "grok not installed; run: $0 setup" >&2; exit 1; }
    command -v jq >/dev/null 2>&1 || { echo "jq required" >&2; exit 1; }
    command -v timeout >/dev/null 2>&1 || { echo "GNU timeout required" >&2; exit 1; }
    [[ -f "$TASKS_JSON" ]] || { echo "missing $TASKS_JSON" >&2; exit 1; }
    [[ -d "$CHECKOUTS_DIR" ]] || { echo "no checkouts; run: $0 setup" >&2; exit 1; }

    mkdir -p "$AGENTS_DIR"
    : > "$SWARM_LOG"

    local start_ts
    start_ts=$(date +%s)
    (( DURATION > 0 )) && END=$(( start_ts + DURATION ))

    trap cleanup INT TERM
    echo "Launching $AGENTS agents (stagger ${STAGGER}s, task timeout ${TASK_TIMEOUT}s, duration $( ((DURATION > 0)) && echo "${DURATION}s" || echo "until Ctrl-C" ))"
    local i
    for (( i = 0; i < AGENTS; i++ )); do
        worker "$i" &
        WORKER_PIDS+=($!)
        (( i < AGENTS - 1 )) && sleep "$STAGGER"
    done
    echo "All $AGENTS agents running. Live log: tail -f $SWARM_LOG"
    wait
    summary
}

do_setup() {
    mkdir -p "$SWARM_HOME"
    install_grok
    write_grok_config
    setup_repos
    setup_venvs
    smoke_imports
    echo "Setup complete."
}

case "${1:-all}" in
    setup) do_setup ;;
    run)   run_swarm ;;
    all)   do_setup; run_swarm ;;
    *)     echo "usage: $0 [setup|run]" >&2; exit 2 ;;
esac
