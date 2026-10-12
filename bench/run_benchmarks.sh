#!/usr/bin/env bash
# Runs the standard benchmark suite against ./build/main and appends each
# result as one JSON line to bench/results.jsonl.
#
#   1. Batching: 16 closed-loop clients, batch size 1 (baseline) vs 4 vs 8.
#   2. Overload: open-loop traffic at 3x the measured batched capacity, with a
#      small bounded queue, to show rejection keeps accepted latency stable.
#
# Usage: bench/run_benchmarks.sh [duration_seconds]
set -euo pipefail

cd "$(dirname "$0")/.."
DURATION="${1:-60}"
CLIENTS=16
N_PREDICT=64
OUT=bench/results.jsonl
LOG=bench/server.log

start_server() {
    ./build/main "$@" > "$LOG" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 60); do
        curl -sf localhost:8080/healthz > /dev/null && return 0
        sleep 1
    done
    echo "server did not start; see $LOG" >&2
    exit 1
}

stop_server() {
    kill "$SERVER_PID" 2> /dev/null || true
    wait "$SERVER_PID" 2> /dev/null || true
}
trap stop_server EXIT

echo "== batching: $CLIENTS clients, ${DURATION}s each"
for bs in 1 4 8; do
    start_server --batch-size "$bs" --max-queue-depth 256
    python3 bench/loadgen.py --clients "$CLIENTS" --duration "$DURATION" \
        --n-predict "$N_PREDICT" --label "batch=$bs clients=$CLIENTS" --json "$OUT"
    stop_server
    echo
done

# Capacity = best batched throughput from the runs above.
CAPACITY=$(python3 -c "
import json
rows = [json.loads(l) for l in open('$OUT') if l.strip()]
rows = [r for r in rows if r['label'].startswith('batch=') and 'batch=1 ' not in r['label']]
print(max(r['throughput_req_per_s'] for r in rows[-2:]))
")
RATE=$(python3 -c "print(round(3 * $CAPACITY, 2))")

echo "== overload: capacity ~${CAPACITY} req/s, offering ${RATE} req/s (3x)"
start_server --batch-size 8 --max-queue-depth 16
python3 bench/loadgen.py --rate "$RATE" --duration "$DURATION" \
    --n-predict "$N_PREDICT" --label "overload 3x, queue=16" --json "$OUT"
stop_server

echo
echo "results appended to $OUT"
