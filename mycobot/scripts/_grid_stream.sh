#!/bin/bash
# five-panel re-render of the streaming-run videos (report videos 6.2-6.6)
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
for spec in "9 hybrid" "8 hybrid" "7 hybrid" "7 b2" "0 hybrid"; do
  set -- $spec
  out=results/media/grid_stream_s4_test_ep$1_$2.mp4
  [ -f "$out" ] && continue
  echo "stream ep$1 $2"
  $PY scripts/render_grid_compare.py --set s4_test --episode $1 --method $2 --real-dir results/video \
     --tag-suffix _webcam_ep$1_$2 --real-video results/video/webcam_ep$1_$2_webcam.mp4 --no-sim-fit --out "$out" > /dev/null 2>> results/grid_stream.err
  echo "  -> $out"
done
echo STREAM_DONE
