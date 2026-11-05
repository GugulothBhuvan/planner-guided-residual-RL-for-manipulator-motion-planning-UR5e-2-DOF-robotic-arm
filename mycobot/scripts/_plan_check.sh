#!/bin/bash
cd "$(dirname "$0")/.."
R=results/runs/m5/residual_b99f130c_s1
for a in "s4_chal 0" "s4_chal 25" "s4_chal 50" "s4_chal 75" "s4_shape 0" "s4_shape 25" "s4_shape 50"; do
  set -- $a
  for m in b2 hybrid; do
    extra=""; [ $m = hybrid ] && extra="--run $R --checkpoint final"
    echo -n "$1 ep$2 $m: "
    ../.venv/Scripts/python.exe scripts/run_segments.py --set $1 --episode $2 --method $m $extra --dry-run --plan-only --tol 6 2>&1 | grep -E "^simulation|nominal|Error|ABORT|Traceback|B2 found" | tr '\n' ' ' | cut -c1-330
    echo
  done
done
echo PLAN_CHECK_DONE
