#!/bin/bash
# 6-h Styria graupel + two-moment run with precursor inflow on all open faces, then GIF/mp4 and comparison with styria6h_g2m.
set -e
TAG=styria6h_g2m_pre
./venv/bin/python3 run_case.py single_cu:congestus 21600 NX=128 NY=128 SHF_SPOT_X0=3500.0 "SHF_SPOT_SHAPE='gaussian'" \
  SNAPSHOT_EVERY=60.0 BUBBLE_DTHETA=0.0 SCU_THETA_ML=302.0 SCU_Z_ML=1200.0 Z_BL=1200.0 SCU_RH_ML_TOP=0.8 \
  "SCU_RH=((1200.0,0.8),(3000.0,0.75),(5500.0,0.55),(9500.0,0.4))" \
  "SCU_LAPSE=((3000.0,0.003),(5500.0,0.0045),(6000.0,0.0145),(9500.0,0.0045))" \
  "ICE_SCHEME='graupel'" RAIN_MOMENTS=2 "OPEN_INFLOW_REF='precursor'" PRECURSOR_NX=64 PRECURSOR_NY=64 \
  TAG=$TAG > outputs/logs/$TAG.runner.log 2>&1
RUN=$(ls -d runs/*_single_cu_congestus_360min_$TAG | tail -n 1)
VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow ./venv/bin/python3 scripts/finish_run.py $RUN \
  --out cloud_gifs/20261028_single_cu_styria6h_g2m_precursor_0-360min.gif > outputs/logs/finish_$TAG.txt 2>&1
./venv/bin/python3 scripts/compare_microphysics.py runs/20261026_1234_single_cu_congestus_360min_styria6h_g2m $RUN \
  --budget 0 360 > outputs/logs/compare_styria6h_pre.txt 2>&1
