#!/bin/bash
# Start one A/B arm of the Strata engine on :8000 and wait until the model is loaded.
#
#   arm_start.sh <serve-dir> <config.json> <stdout-log>
#
# Why a helper: the two arms differ ONLY in which server.py and which engine
# binaries they use.  Everything else (the 39 engine args, the tokenizer, the
# pack, the MTP folder) comes from the config, so the comparison stays
# single-variable.  Both arms run on the same interpreter (the production venv):
# the v0.1.45 git worktree has no .venv of its own, and a different Python build
# would be a second variable.
#
# NOTE: the process lands in the cgroup of whoever runs this.  That is fine for a
# temporary measurement arm (it dies with the caller); production must not be
# started this way - systemd kills the whole scope (see docs/OPERATIONS.md).
set -uo pipefail

SERVE_DIR=${1:?serve dir}
CFG=${2:?config json}
OUT=${3:-/tmp/arm.log}
PORT=${PORT:-8000}
PY=${PY:-python3}          # use the installed tree's .venv/bin/python, so both arms share one interpreter

[ -f "$CFG" ] || { echo "no config: $CFG"; exit 1; }
[ -f "$SERVE_DIR/server.py" ] || { echo "no server.py in: $SERVE_DIR"; exit 1; }

# nothing may hold the port or the card before an arm starts
if ss -tln 2>/dev/null | grep -q ":$PORT "; then
    echo "port $PORT already in use"; exit 1
fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ "${used:-99999}" -gt 1500 ]; then
    echo "VRAM still ${used} MiB: wait for the previous arm to release it"; exit 1
fi

echo "starting arm: serve=$SERVE_DIR"
echo "              config=$CFG"
echo "              log=$OUT  python=$PY"
nohup "$PY" "$SERVE_DIR/server.py" --engine strata --config "$CFG" --port "$PORT" >"$OUT" 2>&1 &
pid=$!
echo "  pid $pid"

for i in $(seq 1 300); do
    code=$(curl -s -o /tmp/arm_models.json -w '%{http_code}' --max-time 3 \
                "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
    if [ "$code" = "200" ] && grep -q '"value": "loaded"' /tmp/arm_models.json 2>/dev/null; then
        echo "  ready after $((i * 2))s"
        exit 0
    fi
    if [ "$i" -gt 5 ] && ! kill -0 "$pid" 2>/dev/null; then
        echo "  server exited early; last lines of $OUT:"
        tail -25 "$OUT"
        exit 1
    fi
    sleep 2
done
echo "  not ready after 600s; last lines of $OUT:"
tail -25 "$OUT"
exit 1
