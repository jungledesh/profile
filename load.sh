#!/usr/bin/env bash
# Profile v2 — rule + flag test suite
#
# Usage:
#   ./load.sh r1       # under-batching
#   ./load.sh r2       # kv cache pressure
#   ./load.sh r3       # low prefix reuse
#   ./load.sh r4       # oom risk  (no load — config fact)
#   ./load.sh r5       # concurrency saturation
#   ./load.sh flags    # tp, cost-per-hour, -m, -v
#   ./load.sh all      # all sequentially

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────
BASE_URL="${BASE_URL:-http://localhost:8000}"
METRICS_URL="$BASE_URL/metrics"
PROFILE="${PROFILE:-./target/release/profile}"

MODEL=$(curl -sf "$BASE_URL/v1/models" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" \
  2>/dev/null || echo "")

if [[ -z "$MODEL" ]]; then
  echo "error: could not reach $BASE_URL/v1/models — set BASE_URL and ensure vLLM is running"
  exit 1
fi

LOAD_PID=""

cleanup() {
  if [[ -n "$LOAD_PID" ]]; then
    kill "$LOAD_PID" 2>/dev/null || true
    LOAD_PID=""
  fi
}
trap cleanup EXIT

# ── load generator ────────────────────────────────────────────────────
# concurrency  — in-flight requests at all times
# prompt_tokens — approx input length
# max_tokens    — max output tokens
# unique        — true = random prompt each req (no prefix reuse)
# duration_secs — how long to sustain load

start_load() {
  local concurrency=$1
  local prompt_tokens=$2
  local max_tokens=$3
  local unique=${4:-false}
  local duration_secs=${5:-300}

  python3 -u - \
    "$BASE_URL" "$MODEL" \
    "$concurrency" "$prompt_tokens" "$max_tokens" \
    "$unique" "$duration_secs" <<'PYEOF' &
import sys, asyncio, aiohttp, random, string, time

base, model        = sys.argv[1], sys.argv[2]
concurrency        = int(sys.argv[3])
prompt_tokens      = int(sys.argv[4])
max_tokens         = int(sys.argv[5])
unique             = sys.argv[6].lower() == "true"
stop_at            = time.time() + float(sys.argv[7])

SHARED_PREFIX = (
    "You are a helpful assistant. Provide a detailed technical explanation. "
)

def make_prompt():
    words = [
        "distributed", "inference", "latency", "throughput", "memory",
        "bandwidth", "compute", "token", "batch", "cache", "scheduler",
        "kernel", "tensor", "attention", "layer", "weight", "activation",
    ]
    body = " ".join(random.choices(words, k=prompt_tokens // 2))
    if unique:
        body += " " + "".join(random.choices(string.ascii_lowercase, k=24))
        return body
    return SHARED_PREFIX + body

async def send(session):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": make_prompt()}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        async with session.post(
            f"{base}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as r:
            await r.read()
    except Exception:
        pass

async def main():
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def worker():
            while time.time() < stop_at:
                await send(session)
        await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(concurrency)])

asyncio.run(main())
PYEOF
  LOAD_PID=$!
  echo "  load running  PID=$LOAD_PID  concurrency=$concurrency  prompt_tokens≈$prompt_tokens  max_tokens=$max_tokens  unique=$unique"
}

# ── rule tests ────────────────────────────────────────────────────────

test_r1() {
  echo ""
  echo "══ R1: Under-batching ══════════════════════════════════════════"
  echo "  3 concurrent requests. Server has 256 slots. Occupancy ~1%."
  echo "  Expect: [!] Under-batching — increase client concurrency"
  echo ""
  start_load 3 50 50 false 180
  sleep 3
  $PROFILE diagnose --url "$METRICS_URL" -m 256 --duration 30s
}

test_r2() {
  echo ""
  echo "══ R2: KV cache pressure ════════════════════════════════════════"
  echo "  400 concurrent, 2000-token prompts + 4000 output tokens."
  echo "  8B on H200 has ~124GB KV headroom — need long seqs to fill it."
  echo "  vLLM queues excess; running sequences saturate KV blocks."
  echo "  Expect: [!] KV Cache Pressure — reduce max_num_seqs or max_model_len"
  echo "  Tune: drop to 200 if vLLM returns too many 503s."
  echo ""
  start_load 400 2000 4000 false 300
  sleep 10
  $PROFILE diagnose --url "$METRICS_URL" --duration 30s
}

test_r3() {
  echo ""
  echo "══ R3: Low prefix reuse ═════════════════════════════════════════"
  echo "  20 concurrent, fully unique prompts, no shared prefix."
  echo "  Expect: [!] Low Prefix Reuse — enable prefix caching, restructure prompts"
  echo "  Requires: vLLM started with --enable-prefix-caching"
  echo ""
  start_load 20 120 100 true 180
  sleep 3
  $PROFILE diagnose --url "$METRICS_URL" --duration 30s
}

test_r4() {
  echo ""
  echo "══ R4: OOM risk ═════════════════════════════════════════════════"
  echo "  No load needed — R4 is a config fact, not a runtime observation."
  echo "  Passing --tensor-parallel-size 1 forces kv_headroom_gb < 0 on large models."
  echo "  Expect: [!] OOM Risk — set --tensor-parallel-size N"
  echo "  Note: only fires if model weight_gb > per-GPU VRAM (e.g. 70B bf16 on 80GB GPU)."
  echo ""
  $PROFILE diagnose --url "$METRICS_URL" --tensor-parallel-size 1 --duration 30s
}

test_r5() {
  echo ""
  echo "══ R5: Concurrency saturation ═══════════════════════════════════"
  echo "  60 concurrent against -m 32 cap. Fills all slots and builds queue."
  echo "  Short outputs keep KV usage low (≪80%)."
  echo "  Expect: [!] Concurrency Saturation — raise --max-num-seqs"
  echo ""
  start_load 60 100 50 false 300
  sleep 5
  $PROFILE diagnose --url "$METRICS_URL" -m 32 --duration 30s
}

# ── flag tests ────────────────────────────────────────────────────────

test_flags() {
  echo ""
  echo "══ FLAGS ════════════════════════════════════════════════════════"

  echo ""
  echo "── --tensor-parallel-size 4 ──"
  echo "  Ceilings should be 4× single-GPU values. Efficiency % unchanged (TP implicit)."
  start_load 10 100 100 false 120
  sleep 2
  $PROFILE diagnose --url "$METRICS_URL" --tensor-parallel-size 4 --duration 30s
  cleanup

  echo ""
  echo "── --tensor-parallel-size 1 (OOM check) ──"
  echo "  On a large model: R4 fires. On a small model: kv_headroom_gb positive, no R4."
  $PROFILE diagnose --url "$METRICS_URL" --tensor-parallel-size 1 --duration 30s

  echo ""
  echo "── --cost-per-hour 3.45 ──"
  echo "  Economics block: $/1M tok exact (not labeled est). Recoverable waste in dollars."
  start_load 10 100 100 false 120
  sleep 2
  $PROFILE diagnose --url "$METRICS_URL" --cost-per-hour 3.45 --duration 30s
  cleanup

  echo ""
  echo "── -m 32 (max-num-seqs override) ──"
  echo "  R1 threshold = 8 running. R5 fires at 32 running."
  start_load 5 50 50 false 120
  sleep 2
  $PROFILE diagnose --url "$METRICS_URL" -m 32 --duration 30s
  cleanup

  echo ""
  echo "── -v (verbose) ──"
  echo "  When R1 doesn't fire: shows 'not triggered (prefill saturated at X%)'."
  echo "  Load: high concurrency to saturate prefill so R1 is suppressed."
  start_load 30 500 200 false 120
  sleep 2
  $PROFILE diagnose --url "$METRICS_URL" -v --duration 30s
  cleanup
}

# ── dispatch ──────────────────────────────────────────────────────────

python3 -c "import aiohttp" 2>/dev/null || pip install aiohttp -q --break-system-packages

echo ""
echo "Profile v2 — test suite"
echo "  BASE_URL : $BASE_URL"
echo "  MODEL    : $MODEL"
echo "  PROFILE  : $PROFILE"
echo ""

case "${1:-all}" in
  r1)    test_r1 ;;
  r2)    test_r2 ;;
  r3)    test_r3 ;;
  r4)    test_r4 ;;
  r5)    test_r5 ;;
  flags) test_flags ;;
  all)
    test_r1;  cleanup
    test_r2;  cleanup
    test_r3;  cleanup
    test_r4
    test_r5;  cleanup
    test_flags
    ;;
  *)
    echo "Usage: $0 [r1|r2|r3|r4|r5|flags|all]"
    exit 1
    ;;
esac

echo ""
echo "done."
