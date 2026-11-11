#!/bin/bash
# A/B queue: runs arms from a queue file, one line at a time ("A,B" = a pair side by side, "A" = alone), so the
# file can be edited while the queue runs. A line starts only if it can end by DEADLINE (its measured or
# assumed minutes + MARGIN_MIN); a line that cannot is skipped and logged. Each arm trains with
# scripts/ab/run_ab.py from <CFG>/<arm>.toml into <RUNS>/<arm>; <CFG>/<arm>.code (one line: a checkout path)
# overrides the code that trains it. With LOCK set, the whole queue holds that flock (one GPU owner).
#
#   CFG=... RUNS=... QUEUE=... [REPO=.] [DEADLINE=16:30] [PAIR_MIN=29] [SINGLE_MIN=18] [MARGIN_MIN=3]
#   [LOCK=/path/gpu.lock] [LOCK_WAIT=3600] [MEMORY_MAX=12G] [PY=.venv/bin/python] scripts/ab/queue.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(cd "$HERE/../.." && pwd)}
PY=${PY:-$REPO/.venv/bin/python}
: "${CFG:?CFG: folder of <arm>.toml}" "${RUNS:?RUNS: folder for the run folders}" "${QUEUE:?QUEUE: queue file}"
DEADLINE=${DEADLINE:-23:59}
PAIR_MIN=${PAIR_MIN:-29}
SINGLE_MIN=${SINGLE_MIN:-18}
MARGIN_MIN=${MARGIN_MIN:-3}

if [ -n "${LOCK:-}" ] && [ -z "${AB_LOCKED:-}" ]; then
  AB_LOCKED=1 exec flock -w "${LOCK_WAIT:-3600}" "$LOCK" "$0" "$@"
fi

run_arm() {  # <arm>: train it; prints the end line (done = done.json written)
  local arm=$1 out="$RUNS/$1" code="$REPO" cap=()
  [ -f "$CFG/$arm.code" ] && code=$(cat "$CFG/$arm.code")
  [ -n "${MEMORY_MAX:-}" ] && cap=(systemd-run --user --scope --quiet -p "MemoryMax=$MEMORY_MAX" -p MemorySwapMax=0 --)
  mkdir -p "$out"
  ( cd "$out" && "${cap[@]}" "$PY" "$HERE/run_ab.py" "$code" "$CFG/$arm.toml" "$out" > "$out/stdout.log" 2>&1 )
  echo "$(date +%T) end $arm rc=$? done=$([ -f "$out/done.json" ] && echo yes || echo NO)"
}

fits() {  # <minutes>: true if now + minutes + margin <= DEADLINE (today)
  [ $(( $(date +%s) + ($1 + MARGIN_MIN) * 60 )) -le "$(date -d "$DEADLINE" +%s)" ]
}

while :; do
  line=$(head -1 "$QUEUE" 2>/dev/null)
  [ -z "$line" ] && break
  sed -i 1d "$QUEUE"
  IFS=, read -r a b <<< "$line"
  need=$([ -n "$b" ] && echo "$PAIR_MIN" || echo "$SINGLE_MIN")
  if ! fits "$need"; then
    echo "$(date +%T) skip '$line': needs $need + $MARGIN_MIN min, deadline $DEADLINE"
    continue
  fi
  echo "$(date +%T) start $a${b:+ $b}"
  run_arm "$a" & pa=$!
  [ -n "$b" ] && { run_arm "$b" & pb=$!; }
  wait "$pa"; [ -n "$b" ] && wait "$pb"
done
echo "$(date +%T) queue finished"
