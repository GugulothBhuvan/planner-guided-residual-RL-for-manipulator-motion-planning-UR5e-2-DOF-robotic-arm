#!/bin/bash
# 30 real-arm runs (episodes 0-9 x b2 / base / hybrid) with the segment executor (SPEC M7g). Stops at the first abort.
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
R=results/runs/m5/residual_b99f130c_s1
SPEED=${SPEED:-40}
for ep in 0 1 2 3 4 5 6 7 8 9; do
  for m in b2 base hybrid; do
    extra=""; [ "$m" = hybrid ] && extra="--run $R --checkpoint final"
    f="results/seg/real_s4_test_ep${ep}_${m}"; [ "$m" != b2 ] && f="${f}@3000"; f="${f}_seg.json"
    if [ -f "$f" ] && grep -q '"status": "ok"' "$f"; then echo "skip ep $ep $m (done)"; continue; fi
    echo "$(date +%H:%M:%S) ep $ep $m"
    out=$($PY scripts/run_segments.py --episode $ep --method $m $extra --execute --speed $SPEED --tol 6 --tag-suffix _seg 2>&1 | tail -1)
    echo "$out" | cut -c1-330
    case "$out" in *'"status": "ok"'*) ;; *) echo "STOPPING: run did not finish ok"; exit 1;; esac
  done
done
echo ALL_DONE
