#!/bin/bash
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
$PY scripts/film_batch.py make --layout grid --name seg1 --only s4_test:9:hybrid s4_test:8:hybrid s4_test:7:b2 s4_test:7:hybrid
$PY scripts/film_batch.py make --layout grid --name chal1 --only s4_chal:2:hybrid s4_chal:25:hybrid s4_chal:50:hybrid
$PY scripts/film_batch.py make --layout grid --name chal2 --only s4_chal:75:hybrid
$PY scripts/film_batch.py make --layout grid --name shape1 --only s4_shape:0:hybrid s4_shape:25:hybrid
echo GRID_MAIN_DONE
