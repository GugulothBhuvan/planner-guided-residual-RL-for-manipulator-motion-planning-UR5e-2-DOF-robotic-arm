#!/bin/bash
# five-panel videos for every remaining filmed case (B2 and mixed-shape included); existing files are kept
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
for n in chal1 chal2 shape1 seg1; do
  $PY scripts/film_batch.py make --name $n --layout grid 2>> results/grid_rest.err
done
echo REST_DONE
