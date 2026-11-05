#!/bin/bash
# after chal1 finishes: B2 and mixed-shape videos of the shape set
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
while tasklist //FI "PID eq 29624" 2>/dev/null | grep -q python; do sleep 10; done
$PY scripts/film_batch.py make --name shape1 --layout grid --only s4_shape:0:b2 s4_shape:25:b2 s4_shape:50:b2 s4_shape:50:hybrid 2>> results/grid_rest.err
echo REST2_DONE
