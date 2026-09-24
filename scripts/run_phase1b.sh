#!/bin/bash
# Run Phase 1b: Extended QV_SURF sweep to find 10-20% range and 100% threshold

echo "======================================================"
echo "Phase 1b: Extended QV_SURF Sweep (5 experiments)"
echo "Target: Find 10-20% cloud cover range"
echo "======================================================"
echo

# Experiment 6: QV_SURF = 0.014 (historical 100% value with TAU_NUDGE_Q=1800)
echo "[1/5] Running exp06_qv0140 (QV_SURF=0.014)..."
./venv/bin/python3 run_experiment.py \
  --exp-id exp06_qv0140 \
  --phase phase1_qvsurf \
  --qv-surf 0.014 \
  --qv-scale 4000 \
  --tau-nudge-q 3600 \
  --lhf 100 \
  --w-subs 0.005 \
  --duration 1800 \
  --restart snap_0001200.npz

# Experiment 7: QV_SURF = 0.015
echo "[2/5] Running exp07_qv0150 (QV_SURF=0.015)..."
./venv/bin/python3 run_experiment.py \
  --exp-id exp07_qv0150 \
  --phase phase1_qvsurf \
  --qv-surf 0.015 \
  --qv-scale 4000 \
  --tau-nudge-q 3600 \
  --lhf 100 \
  --w-subs 0.005 \
  --duration 1800 \
  --restart snap_0001200.npz

# Experiment 8: QV_SURF = 0.016
echo "[3/5] Running exp08_qv0160 (QV_SURF=0.016)..."
./venv/bin/python3 run_experiment.py \
  --exp-id exp08_qv0160 \
  --phase phase1_qvsurf \
  --qv-surf 0.016 \
  --qv-scale 4000 \
  --tau-nudge-q 3600 \
  --lhf 100 \
  --w-subs 0.005 \
  --duration 1800 \
  --restart snap_0001200.npz

# Experiment 9: QV_SURF = 0.017
echo "[4/5] Running exp09_qv0170 (QV_SURF=0.017)..."
./venv/bin/python3 run_experiment.py \
  --exp-id exp09_qv0170 \
  --phase phase1_qvsurf \
  --qv-surf 0.017 \
  --qv-scale 4000 \
  --tau-nudge-q 3600 \
  --lhf 100 \
  --w-subs 0.005 \
  --duration 1800 \
  --restart snap_0001200.npz

# Experiment 10: QV_SURF = 0.018
echo "[5/5] Running exp10_qv0180 (QV_SURF=0.018)..."
./venv/bin/python3 run_experiment.py \
  --exp-id exp10_qv0180 \
  --phase phase1_qvsurf \
  --qv-surf 0.018 \
  --qv-scale 4000 \
  --tau-nudge-q 3600 \
  --lhf 100 \
  --w-subs 0.005 \
  --duration 1800 \
  --restart snap_0001200.npz

echo
echo "======================================================"
echo "Phase 1b Complete!"
echo "======================================================"
echo "Check experiments/results_summary.csv for results"
echo "Snapshots saved for comparison GIF generation"
