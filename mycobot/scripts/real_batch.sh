#!/bin/bash
# Real-arm batch: episodes 0-9, b2 / base@3000 / hybrid@3000 (seed-1 final), interleaved so partial results stay comparable.
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
RUN=results/runs/m5/residual_b99f130c_s1
for ep in 0 1 2 3 4 5 6 7 8 9; do
  for m in b2 base hybrid; do
    tag="real_s4_test_ep${ep}_${m}"; [ "$m" != b2 ] && tag="real_s4_test_ep${ep}_${m}@3000"
    [ -f "results/real/${tag}.json" ] && continue
    extra=""; [ "$m" = hybrid ] && extra="--run $RUN --checkpoint final"
    echo "$(date +%T) ep $ep $m"
    out=$($PY scripts/run_real.py --episode $ep --method $m --budget 3000 $extra --execute 2>&1 | tail -3)
    echo "$out" | tail -1
    echo "$out" | grep -q "ABORT\|Traceback" && { echo "STOPPED at ep $ep $m"; exit 1; }
  done
done
echo "BATCH DONE"
