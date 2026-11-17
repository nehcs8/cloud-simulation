#!/bin/bash
# finish + comparison for styria6h_g2m_pre2 (the chain passed the wrong run dir to finish_run)
cd "$(dirname "$0")/../../.."
RUN=runs/20261028_1920_single_cu_congestus_360min_styria6h_g2m_pre2
./venv/bin/python3 scripts/compare_microphysics.py runs/20261026_1234_single_cu_congestus_360min_styria6h_g2m runs/20261028_1017_single_cu_congestus_360min_styria6h_g2m_pre $RUN \
  --budget 0 360 > outputs/logs/compare_styria6h_pre2.txt 2>&1
VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow ./venv/bin/python3 scripts/finish_run.py $RUN \
  --out cloud_gifs/20261028_single_cu_styria6h_g2m_precursor2_0-360min.gif > outputs/logs/finish_styria6h_g2m_pre2.txt 2>&1
