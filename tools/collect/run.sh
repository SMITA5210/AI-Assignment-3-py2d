#!/bin/bash
# Training-data collection: tools/collect/agent.py (as ME) vs full, 3 matches
# at a time, stopping at the first invalid match, forfeit, cut-off, traceback
# or policy violation.
#
# usage (repo root):  PYTHON=/path/to/python tools/collect/run.sh FIRST LAST
#   e.g. tools/collect/run.sh 1 20   -> opponents named KC1 .. KC20
# Opponent names KC<n> mark training matches; evaluation matches use other
# names and are never passed to the fitting scripts.
cd "$(dirname "$0")/../.." || exit 1
PY=${PYTHON:-python3}
AGENT=tools/collect/agent.py
PREFIX=${PREFIX:-KC}      # opponent-name prefix; use another one for throwaway smoke runs
first=$1; last=$2
mkdir -p runlogs/batch runlogs/keep
manifest=runlogs/keep/manifest_${PREFIX}${first}-${last}.txt

header() {
  echo "date        : $(date -Is)"
  echo "collector   : $AGENT md5 $(md5sum $AGENT | cut -d' ' -f1)"
  echo "derived from: tools/versions/v8b/agent.py md5 $(md5sum tools/versions/v8b/agent.py | cut -d' ' -f1)"
  echo "config      : base.conf md5 $(md5sum base.conf | cut -d' ' -f1) (server random seed not set: matches are not exactly reproducible)"
  echo "git HEAD    : $(git rev-parse --short HEAD)"
  echo "opponent    : full"
  echo "reproducible: no - match identity is the log name below; the per-decision"
  echo "              exploration seed (unum*100003 + cycle) is in each keep_ctx note"
}

check() {   # $1 = batch output file; prints problems, returns 1 if any
  local f=$1 bad=0
  local res stamp
  res=$(sed 's/\x1b\[[0-9;]*m//g' "$f" | grep "result     :")
  stamp=$(sed 's/\x1b\[[0-9;]*m//g' "$f" | grep -o "replay     : [^ ]*" | awk '{print $3}' | sed 's/\.rcg$//')
  if [[ "$res" != *"[valid]"* ]]; then echo "INVALID: $f $res"; bad=1; fi
  if grep -q "FORFEIT\|cut short" "$f"; then echo "FORFEIT/CUT: $f"; bad=1; fi
  if [[ -n "$stamp" ]]; then
    if grep -q "Traceback" "runlogs/${stamp}_ME.out" 2>/dev/null; then echo "TRACEBACK: $stamp"; bad=1; fi
    if grep -rq "POLICY VIOLATION" "runlogs/${stamp}_ME/" 2>/dev/null; then echo "POLICY VIOLATION: $stamp"; bad=1; fi
    echo "match       : ${stamp} $(echo $res | tr -s ' ' | cut -d: -f2)" >> "$manifest"
  else
    echo "NO REPLAY: $f"; bad=1
  fi
  return $bad
}

if [[ "$1" == "--check" ]]; then    # self-test of the stop rules on one output file
  manifest=/dev/null; check "$2"; exit $?
fi
header > "$manifest"

n=$first
while (( n <= last )); do
  group=()
  for k in 0 1 2; do
    (( n + k > last )) && break
    i=$((n + k)); port=$((6000 + 10 * k))
    out=runlogs/batch/collect_${PREFIX}$i.txt
    $PY a3.py match ME "${PREFIX}$i" --left "$AGENT" --right full -p $port > "$out" 2>&1 &
    group+=("$out")
    sleep 3
  done
  wait
  for f in "${group[@]}"; do
    if ! check "$f"; then
      echo "stopping collection (see $manifest)"; exit 1
    fi
  done
  n=$((n + 3))
done
echo "collection done: $manifest"
cat "$manifest"
