#!/bin/bash
# Run several matches of src/student/agent.py (team ME) in parallel, 3 at a time.
#
# usage (repo root, ai-a3 env active):
#   tools/batch.sh TAG NAME:SPEC [NAME:SPEC ...]
#   tools/batch.sh run1 F1:full F2:full BASE:tools/baseline/agent.py DUMMY:src/student/agent.py:Agent2
#
# NAME is the opponent's team name (must differ per match), SPEC is what a3.py
# accepts after --right. Output goes to runlogs/batch/<TAG>_<n>.txt.
# LEFT=tools/versions/v1/agent.py tools/batch.sh ...  plays an older version as ME.
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python3}
LEFT=${LEFT:-src/student/agent.py}
tag=$1; shift
mkdir -p runlogs/batch
i=0
for spec in "$@"; do
  name=${spec%%:*}; opp=${spec#*:}
  port=$((6000 + 10 * (i % 3)))
  $PY a3.py match ME "$name" --left "$LEFT" --right "$opp" -p $port \
      > runlogs/batch/${tag}_$i.txt 2>&1 &
  i=$((i + 1))
  if (( i % 3 == 0 )); then wait; fi
  sleep 3
done
wait
for f in runlogs/batch/${tag}_*.txt; do
  grep -hE "^== |result|planning   : ME|FORFEIT|BENCH|cut short" "$f" | tr -s ' '
done
