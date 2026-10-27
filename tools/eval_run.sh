#!/bin/bash
# Interleaved evaluation vs full: each argument is VERSION_FILE:OPP_NAME; matches
# run 3 at a time in the order given (interleave versions in that order), and
# the run stops at the first invalid match / forfeit / cut-off / traceback /
# policy violation (checks from tools/collect/run.sh --check).
#
# usage (repo root):  PYTHON=... tools/eval_run.sh TAG tools/versions/v3/agent.py:E3P1 tools/versions/v10/agent.py:ECP1 ...
# Output: runlogs/batch/<TAG>_<OPP_NAME>.txt and runlogs/eval/<TAG>.tsv (version, name, replay, result)
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python3}
tag=$1; shift
mkdir -p runlogs/batch runlogs/eval
tsv=runlogs/eval/$tag.tsv
: > "$tsv"
specs=("$@")
n=0
while (( n < ${#specs[@]} )); do
  group=()
  for k in 0 1 2; do
    (( n + k >= ${#specs[@]} )) && break
    spec=${specs[$((n + k))]}
    file=${spec%%:*}; name=${spec#*:}
    out=runlogs/batch/${tag}_${name}.txt
    $PY a3.py match ME "$name" --left "$file" --right full -p $((6000 + 10 * k)) > "$out" 2>&1 &
    group+=("$file:$name:$out")
    sleep 3
  done
  wait
  for g in "${group[@]}"; do
    file=${g%%:*}; rest=${g#*:}; name=${rest%%:*}; out=${rest#*:}
    if ! tools/collect/run.sh --check "$out"; then
      echo "STOP: $out failed a check"; exit 1
    fi
    clean=$(sed 's/\x1b\[[0-9;]*m//g' "$out")
    rep=$(echo "$clean" | grep -o "replay     : [^ ]*" | awk '{print $3}')
    res=$(echo "$clean" | grep "result     :" | sed 's/.*result     : //' | tr -s ' ')
    plan=$(echo "$clean" | grep "planning   : ME" | grep -o "[0-9]*%")
    printf "%s\t%s\t%s\t%s\t%s\n" "$file" "$name" "$rep" "$res" "$plan" >> "$tsv"
  done
  n=$((n + 3))
done
echo "done: $tsv"; cat "$tsv"
