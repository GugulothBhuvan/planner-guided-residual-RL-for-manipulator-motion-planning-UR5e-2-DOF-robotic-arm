#!/bin/bash
# Vibration experiment: episode 0, B2 (same plan every run), conditions x 2 reps, logs -> results/vib/
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
# label speed send_every extra
CONDS=("s20e1 20 1" "s10e1 10 1" "s30e1 30 1" "s20e2 20 2" "s20e1notrim 20 1 --no-trim" "s10e2 10 2")
run() { # rep label speed every extra...
  rep=$1; label=$2; speed=$3; every=$4; shift 4
  echo "$(date +%T) rep $rep $label"
  out=$($PY scripts/run_real.py --episode 0 --method b2 --execute --speed $speed --send-every $every --out-dir results/vib --tag-suffix _${label}_r$rep "$@" 2>&1 | tail -2)
  echo "$out" | tail -1 | cut -c1-230
  echo "$out" | grep -q "ABORT\|Traceback" && echo "ABORTED $label rep $rep"
}
for c in "${CONDS[@]}"; do set -- $c; run 1 "$@"; done
for ((i=${#CONDS[@]}-1; i>=0; i--)); do set -- ${CONDS[$i]}; run 2 "$@"; done
echo "VIB DONE"
