#!/bin/bash
# layered_day first runs (2026-11-01): baseline (no large-scale forcing, 3 h) and trough (ascent 0 → 4 cm/s at 1–4 h, 6 h),
# 128² at 150 m + 64² precursor, side by side on 2 threads each; then GIF + mp4 from 1-min snapshots and thinning.
cd "$(dirname "$0")/../../.."
export MPLBACKEND=Agg NUMBA_NUM_THREADS=2

run() {   # variant T_END
  local TAG=first
  ./venv/bin/python3 run_case.py layered_day:$1 $2 SNAPSHOT_EVERY=60.0 TAG=$TAG > outputs/logs/layered_day_$1_$TAG.runner.log 2>&1
  local RUN=$(ls -d runs/*_layered_day_$1_$(( $2 / 60 ))min_$TAG | tail -n 1)
  NUMBA_NUM_THREADS=4 VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow ./venv/bin/python3 scripts/finish_run.py $RUN \
    --out cloud_gifs/20261101_layered_day_$1_0-$(( $2 / 60 ))min.gif > outputs/logs/finish_layered_day_$1.txt 2>&1
}

run baseline 10800 &
run trough 21600 &
wait
