#!/bin/bash
# usage: diag.sh TAG [extra overrides...]
tag=$1; shift
exec ./venv/bin/python3 run_case.py single_cu:congestus 7200 BUBBLE_DTHETA=0.0 FLUX_RAMP_TIME=2700.0 LHF=280.0 NX=64 NY=64 \
  "SCU_LAPSE=((3000.0,0.003),(5500.0,0.0045),(6000.0,0.0145),(9500.0,0.0045))" \
  "SCU_RH=((1200.0,0.8),(3000.0,0.75),(5500.0,0.55),(9500.0,0.4))" SCU_RH_ML_TOP=0.8 SCU_THETA_ML=302.0 SCU_Z_ML=1200.0 \
  SHF=180.0 SHF_SPOT_AMP=0.0 SHF_SPOT_LHF_AMP=0.0 SNAPSHOT_EVERY=60.0 U_GEO=3.0 V_GEO=3.0 U_GEO_PROFILE=None Z_BL=1200.0 \
  "$@" TAG=$tag > outputs/logs/$tag.runner.log 2>&1
