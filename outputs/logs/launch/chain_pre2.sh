#!/bin/bash
# diagonal precursor test with the perturbation-only face velocity, then the 6-h Styria rerun
cd "$(dirname "$0")/../../.."
outputs/logs/launch/diag.sh diag_precursor2 "OPEN_INFLOW_REF='precursor'" PRECURSOR_NX=32 PRECURSOR_NY=32
RUN=$(ls -d runs/*_diag_precursor2 | tail -n 1)
VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow ./venv/bin/python3 scripts/finish_run.py $RUN \
  --out cloud_gifs/20261028_single_cu_real_flat_diag_precursor2_0-120min.gif > outputs/logs/finish_diag_precursor2.txt 2>&1
./venv/bin/python3 scripts/analyze_inflow.py $RUN --t0 60 --t1 120 > outputs/logs/analyze_inflow_diag2.txt 2>&1
outputs/logs/launch/styria6h_g2m_pre2.sh
