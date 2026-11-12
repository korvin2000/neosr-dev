#!/bin/bash
# A/B queue: runs arms from a queue file, one line at a time ("A,B" = a pair side by side, "A" = alone), so the
# file can be edited while the queue runs; blank and '#' lines are skipped, the queue ends when the file is
# empty. A line starts only if it can end by DEADLINE (PAIR_MIN or SINGLE_MIN, estimates, + MARGIN_MIN); a line
# that cannot is skipped and logged. DEADLINE is HH:MM (its next occurrence) or any `date -d` time. Each arm
# trains with scripts/ab/run_ab.py from <CFG>/<arm>.toml into <RUNS>/<arm>; <CFG>/<arm>.code (one line: a
# checkout path, absolute or relative to the starting folder) overrides the code that trains it. A line is
# checked before it leaves the file: one or two arm names ("A" or "A,B", distinct, no blanks inside), each with
# a <CFG>/<arm>.toml; any other line stops the queue with an error and stays in the file. The queue holds an
# flock on QUEUE itself, so a second queue on the same file refuses to start (lines are removed in place,
# keeping the file and its lock). With LOCK set, the whole queue also holds that flock (one GPU owner).
#
#   CFG=... RUNS=... QUEUE=... [REPO=.] [DEADLINE=16:30] [PAIR_MIN=29] [SINGLE_MIN=18] [MARGIN_MIN=3]
#   [LOCK=/path/gpu.lock] [LOCK_WAIT=3600] [MEMORY_MAX=12G] [PY=.venv/bin/python] scripts/ab/queue.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
: "${CFG:?CFG: folder of <arm>.toml}" "${RUNS:?RUNS: folder for the run folders}" "${QUEUE:?QUEUE: queue file}"
CFG=$(realpath "$CFG") RUNS=$(realpath -m "$RUNS") QUEUE=$(realpath "$QUEUE")
REPO=$(realpath "${REPO:-$HERE/../..}")
PY=${PY:-$REPO/.venv/bin/python}
[[ $PY == */* ]] && PY=$(realpath -s "$PY")  # -s: a venv python is a symlink; resolving it leaves the venv
PAIR_MIN=${PAIR_MIN:-29}
SINGLE_MIN=${SINGLE_MIN:-18}
MARGIN_MIN=${MARGIN_MIN:-3}
END=$(date -d "${DEADLINE:-23:59}" +%s) || exit 2
if [[ ${DEADLINE:-23:59} =~ ^[0-9]{1,2}:[0-9]{2}$ ]] && [ "$END" -le "$(date +%s)" ]; then
  END=$((END + 86400))  # HH:MM already past today: tomorrow's
fi

if [ -z "${AB_LOCKED:-}" ]; then  # the LOCK re-exec below inherits fd 9 and its lock
  exec 9< "$QUEUE" || exit 2
  flock -n 9 || { echo "queue.sh: $QUEUE is run by another queue (its flock is held)" >&2; exit 2; }
fi
if [ -n "${LOCK:-}" ] && [ -z "${AB_LOCKED:-}" ]; then
  export CFG RUNS QUEUE REPO PY DEADLINE="@$END" PAIR_MIN SINGLE_MIN MARGIN_MIN
  AB_LOCKED=1 exec flock -w "${LOCK_WAIT:-3600}" "$LOCK" "$0" "$@"
fi
unset AB_LOCKED  # only for the re-exec: a queue started by an arm takes both locks itself

run_arm() {  # <arm>: train it; prints the end line (rc of the trainer; done = done.json written)
  local arm=$1 out="$RUNS/$1" code="$REPO" cap=() rc
  [ -f "$CFG/$arm.code" ] && code=$(realpath "$(head -1 "$CFG/$arm.code" | tr -d '\r')")
  [ -n "${MEMORY_MAX:-}" ] && cap=(systemd-run --user --scope --quiet -p "MemoryMax=$MEMORY_MAX" -p MemorySwapMax=0 --)
  mkdir -p "$out"
  ( cd "$out" && "${cap[@]}" "$PY" "$HERE/run_ab.py" "$code" "$CFG/$arm.toml" "$out" > "$out/stdout.log" 2>&1 )
  rc=$?
  echo "$(date +%T) end $arm rc=$rc done=$([ -f "$out/done.json" ] && echo yes || echo NO)"
}

check_line() {  # <line>: sets a and b (empty: a alone); a line the queue cannot run sets why and fails
  local arm re='^([^[:space:],/]+)([[:space:]]*,[[:space:]]*([^[:space:],/]+))?$'
  [[ $1 =~ $re ]] || { why="a line is 'A' or 'A,B' (arm names without blanks, ',' or '/')"; return 1; }
  a=${BASH_REMATCH[1]} b=${BASH_REMATCH[3]}
  for arm in "$a" ${b:+"$b"}; do
    [ -f "$CFG/$arm.toml" ] || { why="no $CFG/$arm.toml"; return 1; }
  done
  [ "$a" != "$b" ] || { why="a pair of one arm"; return 1; }
}

drop_first() {  # in place: the file (and the flock on it) stays the same
  local rest
  rest=$(tail -n +2 "$QUEUE")
  printf '%s' "${rest:+$rest$'\n'}" > "$QUEUE"
}

while [ -s "$QUEUE" ]; do
  line=$(head -1 "$QUEUE" | tr -d '\r')  # CRLF files
  read -r line <<< "$line"  # surrounding blanks
  if [[ -n $line && $line != \#* ]] && ! check_line "$line"; then
    echo "$(date +%T) STOP at '$line': $why; the line stays in $QUEUE" >&2
    exit 1
  fi
  drop_first
  [[ -z $line || $line == \#* ]] && continue
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
