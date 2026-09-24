#!/bin/bash
# Run Phase 2: TAU_NUDGE_Q sweep to optimize moisture nudging
# Using QV_SURF=0.016 from Phase 1b (gave 13.8% cloud cover)

echo "======================================================"
echo "Phase 2: TAU_NUDGE_Q Sweep (4 experiments)"
echo "Goal: Minimize qv_bl drift while maintaining clouds"
echo "======================================================"
echo

# Experiment 11: TAU_NUDGE_Q = 1800s (strong nudging)
echo "[1/4] Running exp11_tau1800 (TAU_NUDGE_Q=1800s)..."
./venv/bin/python3 run_experiment.py \
  --exp-id exp11_tau1800 \
  --phase phase2_nudging \
  --qv-surf 0.016 \
  --qv-scale 4000 \
  --tau-nudge-q 1800 \
  --lhf 100 \
  --w-subs 0.005 \
  --duration 1800 \
  --restart spinup_t1200_qv0.016.npz

# Experiment 12: TAU_NUDGE_Q = 2400s
echo "[2/4] Running exp12_tau2400 (TAU_NUDGE_Q=2400s)..."
./venv/bin/python3 run_experiment.py \
  --exp-id exp12_tau2400 \
  --phase phase2_nudging \
  --qv-surf 0.016 \
  --qv-scale 4000 \
  --tau-nudge-q 2400 \
  --lhf 100 \
  --w-subs 0.005 \
  --duration 1800 \
  --restart spinup_t1200_qv0.016.npz

# Experiment 13: TAU_NUDGE_Q = 3600s (baseline from Phase 1b)
echo "[3/4] Running exp13_tau3600 (TAU_NUDGE_Q=3600s - baseline)..."
./venv/bin/python3 run_experiment.py \
  --exp-id exp13_tau3600 \
  --phase phase2_nudging \
  --qv-surf 0.016 \
  --qv-scale 4000 \
  --tau-nudge-q 3600 \
  --lhf 100 \
  --w-subs 0.005 \
  --duration 1800 \
  --restart spinup_t1200_qv0.016.npz

# Experiment 14: TAU_NUDGE_Q = 5400s (weak nudging)
echo "[4/4] Running exp14_tau5400 (TAU_NUDGE_Q=5400s)..."
./venv/bin/python3 run_experiment.py \
  --exp-id exp14_tau5400 \
  --phase phase2_nudging \
  --qv-surf 0.016 \
  --qv-scale 4000 \
  --tau-nudge-q 5400 \
  --lhf 100 \
  --w-subs 0.005 \
  --duration 1800 \
  --restart spinup_t1200_qv0.016.npz

echo
echo "======================================================"
echo "Phase 2 Complete!"
echo "======================================================"
echo "Check experiments/results_summary.csv for results"
