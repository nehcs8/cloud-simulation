#!/bin/bash
# layered_day with a full-size (128²) precursor — no tiled inflow (docs/PLAN_layered_day2.md, caveat 2026-11-07).
# Repeats of M3 (trough) and M1 (turb) with LW + falling ice, 4.5 h, side by side on 2 threads each (~143 wall-min per
# simulated hour → ~10.7 h), then mp4 from the 1-min snapshots and thinning.
# Start: systemd-run --user --unit=ld3-fullpre --working-directory="$PWD" /bin/bash outputs/logs/launch/ld3_fullpre.sh
cd "$(dirname "$0")/../../.."
export MPLBACKEND=Agg NUMBA_NUM_THREADS=2
run() {   # id variant
  local TAG=ld3$1
  ./venv/bin/python3 run_case.py layered_day:$2 16200 SNAPSHOT_EVERY=60.0 "RAD_SCHEME='cloud_lw'" ICE_SEDIMENT=True \
      PRECURSOR_NX=128 PRECURSOR_NY=128 TAG=$TAG > outputs/logs/campaign_ld2/$TAG.runner.log 2>&1
  local RUN=$(ls -d runs/*_layered_day_$2_270min_$TAG | tail -n 1)
  cp outputs/logs/campaign_ld2/$TAG.runner.log $RUN/runner.log
  VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow ./venv/bin/python3 scripts/finish_run.py $RUN --jobs 2 \
      --out cloud_gifs/$(basename $RUN | cut -c1-8)_ld3_$1_$2128_fullpre_lw_icefall_0-270min.mp4 \
      > outputs/logs/campaign_ld2/$TAG.finish.log 2>&1
}
run P1 trough &
run P2 turb &
wait
