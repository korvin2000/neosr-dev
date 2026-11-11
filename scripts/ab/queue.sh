#!/bin/bash
# A/B queue: runs arms from a queue file, one line at a time ("A,B" = a pair side by side, "A" = alone), so the
# file can be edited while the queue runs; blank and '#' lines are skipped, the queue ends when the file is
# empty. A line starts only if it can end by DEADLINE (PAIR_MIN or SINGLE_MIN, estimates, + MARGIN_MIN); a line
# that cannot is skipped and logged. DEADLINE is HH:MM (its next occurrence) or any `date -d` time. Each arm
# trains with scripts/ab/run_ab.py from <CFG>/<arm>.toml into <RUNS>/<arm>; <CFG>/<arm>.code (one line: a
# checkout path, absolute or relative to the starting folder) overrides the code that trains it. With LOCK
# set, the whole queue holds that flock (one GPU owner).
#
#   CFG=... RUNS=... QUEUE=... [REPO=.] [DEADLINE=16:30] [PAIR_MIN=29] [SINGLE_MIN=18] [MARGIN_MIN=3]
#   [LOCK=/path/gpu.lock] [LOCK_WAIT=3600] [MEMORY_MAX=12G] [PY=.venv/bin/python] scripts/ab/queue.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
: "${CFG:?CFG: folder of <arm>.toml}" "${RUNS:?RUNS: folder for the run folders}" "${QUEUE:?QUEUE: queue file}"
CFG=$(realpath "$CFG") RUNS=$(realpath -m "$RUNS") QUEUE=$(realpath "$QUEUE")
REPO=$(realpath "${REPO:-$HERE/../..}")
PY=${PY:-$REPO/.venv/bin/python}
[[ $PY == */* ]] && PY=$(realpath "$PY")
PAIR_MIN=${PAIR_MIN:-29}
SINGLE_MIN=${SINGLE_MIN:-18}
MARGIN_MIN=${MARGIN_MIN:-3}
END=$(date -d "${DEADLINE:-23:59}" +%s) || exit 2
if [[ ${DEADLINE:-23:59} =~ ^[0-9]{1,2}:[0-9]{2}$ ]] && [ "$END" -le "$(date +%s)" ]; then
  END=$((END + 86400))  # HH:MM already past today: tomorrow's
fi

if [ -n "${LOCK:-}" ] && [ -z "${AB_LOCKED:-}" ]; then
  export CFG RUNS QUEUE REPO PY DEADLINE="@$END" PAIR_MIN SINGLE_MIN MARGIN_MIN
  AB_LOCKED=1 exec flock -w "${LOCK_WAIT:-3600}" "$LOCK" "$0" "$@"
fi

run_arm() {  # <arm>: train it; prints the end line (rc of the trainer; done = done.json written)
  local arm=$1 out="$RUNS/$1" code="$REPO" cap=() rc
  [ -f "$CFG/$arm.code" ] && code=$(realpath "$(head -1 "$CFG/$arm.code" | tr -d '\r')")
  [ -n "${MEMORY_MAX:-}" ] && cap=(systemd-run --user --scope --quiet -p "MemoryMax=$MEMORY_MAX" -p MemorySwapMax=0 --)
  mkdir -p "$out"
  ( cd "$out" && "${cap[@]}" "$PY" "$HERE/run_ab.py" "$code" "$CFG/$arm.toml" "$out" > "$out/stdout.log" 2>&1 )
  rc=$?
  echo "$(date +%T) end $arm rc=$rc done=$([ -f "$out/done.json" ] && echo yes || echo NO)"
}

while [ -s "$QUEUE" ]; do
  line=$(head -1 "$QUEUE")
  sed -i 1d "$QUEUE"
  line=$(tr -d '\r' <<< "$line" | xargs)  # CRLF files, surrounding blanks
  [[ -z $line || $line == \#* ]] && continue
  IFS=, read -r a b <<< "$line"
  a=$(xargs <<< "$a") b=$(xargs <<< "${b:-}")
  need=$([ -n "$b" ] && echo "$PAIR_MIN" || echo "$SINGLE_MIN")
  if [ $(( $(date +%s) + (need + MARGIN_MIN) * 60 )) -gt "$END" ]; then
    echo "$(date +%T) skip '$line': needs $need + $MARGIN_MIN min, deadline $(date -d "@$END" '+%F %R')"
    continue
  fi
  echo "$(date +%T) start $a${b:+ $b}"
  run_arm "$a" & pa=$!
  [ -n "$b" ] && { run_arm "$b" & pb=$!; }
  wait "$pa"; [ -n "$b" ] && wait "$pb"
done
echo "$(date +%T) queue finished"
