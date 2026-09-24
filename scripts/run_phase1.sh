#!/bin/bash
# Run Phase 1: QV_SURF sweep experiments

echo "=================================================="
echo "Starting Phase 1: QV_SURF Sweep (5 experiments)"
echo "=================================================="
echo

# Experiment 1: QV_SURF = 0.011
echo "[1/5] Running exp01_qv0110 (QV_SURF=0.011)..."
./venv/bin/python3 run_experiment.py \
  --exp-id exp01_qv0110 \
  --phase phase1_qvsurf \
  --qv-surf 0.011 \
  --qv-scale 4000 \
  --tau-nudge-q 3600 \
  --lhf 100 \
  --w-subs 0.005 \
  --duration 1800 \
  --restart snap_0001200.npz

# Experiment 2: QV_SURF = 0.012 (baseline)
echo "[2/5] Running exp02_qv0120 (QV_SURF=0.012)..."
./venv/bin/python3 run_experiment.py \
  --exp-id exp02_qv0120 \
  --phase phase1_qvsurf \
  --qv-surf 0.012 \
  --qv-scale 4000 \
  --tau-nudge-q 3600 \
  --lhf 100 \
  --w-subs 0.005 \
  --duration 1800 \
  --restart snap_0001200.npz

# Experiment 3: QV_SURF = 0.0125
echo "[3/5] Running exp03_qv0125 (QV_SURF=0.0125)..."
./venv/bin/python3 run_experiment.py \
  --exp-id exp03_qv0125 \
  --phase phase1_qvsurf \
  --qv-surf 0.0125 \
  --qv-scale 4000 \
  --tau-nudge-q 3600 \
  --lhf 100 \
  --w-subs 0.005 \
  --duration 1800 \
  --restart snap_0001200.npz

# Experiment 4: QV_SURF = 0.013
echo "[4/5] Running exp04_qv0130 (QV_SURF=0.013)..."
./venv/bin/python3 run_experiment.py \
  --exp-id exp04_qv0130 \
  --phase phase1_qvsurf \
  --qv-surf 0.013 \
  --qv-scale 4000 \
  --tau-nudge-q 3600 \
  --lhf 100 \
  --w-subs 0.005 \
  --duration 1800 \
  --restart snap_0001200.npz

# Experiment 5: QV_SURF = 0.0135
echo "[5/5] Running exp05_qv0135 (QV_SURF=0.0135)..."
./venv/bin/python3 run_experiment.py \
  --exp-id exp05_qv0135 \
  --phase phase1_qvsurf \
  --qv-surf 0.0135 \
  --qv-scale 4000 \
  --tau-nudge-q 3600 \
  --lhf 100 \
  --w-subs 0.005 \
  --duration 1800 \
  --restart snap_0001200.npz

echo
echo "=================================================="
echo "Phase 1 Complete!"
echo "=================================================="
echo "Check experiments/results_summary.csv for results"
