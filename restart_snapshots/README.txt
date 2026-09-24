RESTART FILE REPOSITORY - PROTECTED SNAPSHOTS
==============================================

This directory contains validated restart files from clean spinup runs.
These files are IMMUTABLE - never modify or overwrite them!

For new experiments, COPY from here to snapshots/ directory before running.

---

Inventory of Restart Files
---------------------------

File: spinup_t1200_qv0.014.npz
  Time: t = 1200s (20 minutes)
  Parameters: QV_SURF=0.014, QV_SCALE=4000, TAU_NUDGE_Q=3600
  State: First clouds forming, scattered cumulus
  Reference qv₀ surface: 12.689 g/kg
  Cloud water max: 1.124 g/kg
  Checksums: qc_sum=1.944, qv_sum=9278.804
  Source: experiments/phase1_qvsurf/exp06_qv0140 (Phase 1b)
  Use case: Low moisture baseline

File: spinup_t1200_qv0.015.npz
  Time: t = 1200s (20 minutes)
  Parameters: QV_SURF=0.015, QV_SCALE=4000, TAU_NUDGE_Q=3600
  State: Developing cumulus field
  Reference qv₀ surface: 13.125 g/kg
  Cloud water max: 2.497 g/kg
  Checksums: qc_sum=6.085, qv_sum=9654.070
  Source: experiments/phase1_qvsurf/exp07_qv0150 (Phase 1b)
  Use case: Medium moisture baseline

File: spinup_t1200_qv0.016.npz
  Time: t = 1200s (20 minutes)
  Parameters: QV_SURF=0.016, QV_SCALE=4000, TAU_NUDGE_Q=3600
  State: Active cumulus convection (~14% cloud cover expected)
  Reference qv₀ surface: 13.689 g/kg
  Cloud water max: 3.643 g/kg
  Checksums: qc_sum=15.982, qv_sum=10099.828
  Source: experiments/phase1_qvsurf/exp08_qv0160 (Phase 1b)
  Use case: Standard baseline for parameter sweeps (RECOMMENDED)

---

Usage Instructions
------------------

CORRECT way to use restart files:

  # 1. Clear working snapshots directory
  rm -f snapshots/snap_*.npz

  # 2. Copy desired restart to working directory
  cp restart_snapshots/spinup_t1200_qv0.016.npz snapshots/snap_0001200.npz

  # 3. Run experiment with restart
  ./venv/bin/python3 run_experiment.py --restart snap_0001200.npz

WRONG - Never do this:

  # ❌ Using snapshots/ directly as restart source (gets contaminated!)
  ./venv/bin/python3 run_experiment.py --restart snap_0001200.npz

  # ❌ Overwriting files in restart_snapshots/ directory
  cp snapshots/snap_0001200.npz restart_snapshots/  # DON'T!

---

Validation Procedure
--------------------

To verify a restart file before use:

  ./venv/bin/python3 << 'EOF'
  import numpy as np

  data = np.load('restart_snapshots/spinup_t1200_qv0.016.npz')

  print(f"Time: {data['t']:.1f}s")
  print(f"qv surface: {data['qv'][:,:,0].mean()*1000:.3f} g/kg")
  print(f"qv₀ surface: {data['prof_qv'][0]*1000:.3f} g/kg")
  print(f"qc max: {data['qc'].max()*1000:.3f} g/kg")
  print(f"Checksums: qc={data['qc'].sum():.3f}, qv={data['qv'].sum():.3f}")
  EOF

Compare checksums against expected values in this README.

---

History
-------

2026-09-22: Initial repository created
  - Added 3 clean restart files from Phase 1b (QV_SURF sweep)
  - exp06, exp07, exp08 validated and preserved
  - Documented contamination bug (Phase 1b vs Phase 2 discrepancy)

Critical bug discovered 2026-09-21:
  - Shared snapshots/ directory caused restart file corruption
  - Phase 2 experiments unknowingly used high-moisture restart (15.059 g/kg)
  - This caused 2× higher cloud cover than expected (27% vs 14%)
  - Root cause: experiments saved to snapshots/ and overwrote snap_0001200.npz
