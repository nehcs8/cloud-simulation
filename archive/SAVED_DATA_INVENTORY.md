# Saved Simulation Data Inventory

## Parent Simulations (dx=100m, 12×12 km domain, 2-hour runs)

### Phase 1: QV_SURF sweep (moisture sensitivity)
Testing different surface moisture values to achieve realistic cloud cover.

**Completed experiments with full data (252 snapshots each, ~3 GB):**
- `exp06_qv0140/` - QV_SURF = 0.014 kg/kg
- `exp07_qv0150/` - QV_SURF = 0.015 kg/kg
- `exp08_qv0160/` - QV_SURF = 0.016 kg/kg ⭐ **Used for nested simulations**
- `exp09_qv0170/` - QV_SURF = 0.017 kg/kg
- `exp10_qv0180/` - QV_SURF = 0.018 kg/kg

**Experiments without snapshots (incomplete/early tests):**
- exp01_qv0110, exp02_qv0120, exp03_qv0125, exp04_qv0130, exp05_qv0135

### Phase 2: TAU_NUDGE_Q sweep (profile nudging timescale)
Testing different moisture nudging timescales.

**Completed experiments with full data (252 snapshots each, ~3 GB):**
- `exp11_tau1800/` - TAU_NUDGE_Q = 1800s (30 min)
- `exp12_tau2400/` - TAU_NUDGE_Q = 2400s (40 min)
- `exp13_tau3600/` - TAU_NUDGE_Q = 3600s (60 min)
- `exp14_tau5400/` - TAU_NUDGE_Q = 5400s (90 min)

**Other phases (empty directories for future work):**
- phase3_lhf, phase4_qvscale, phase5_subsidence, validation

---

## Nested Simulations (high-resolution cloud studies)

### Current data in `snapshots/` directory:
**Cloud #11 - Ultra-high-resolution test (dx=10m)**
- Parent source: `exp08_qv0160/snapshots/snap_0001200.npz`
- Cloud center: (108, 61, 50) in parent grid
- Domain: 1×1 km at dx=10m (100×100×60 cells = 600k cells)
- Timestep: dt=0.2s
- Duration: t=0-240s (4 minutes, stopped due to numerical instability)
- Snapshots: 9 files, ~72 MB total
- Status: ❌ Numerically unstable (qc_max grew to 8.3 g/kg)

### Previous nested attempts (data overwritten):
1. **Cloud #8** (first test):
   - Domain: 2×2 km at dx=25m (80×80×60)
   - Duration: 5 minutes (stable)
   - Used in `comparison.gif`

2. **Cloud #9** (large cloud):
   - Domain: 2×2 km at dx=25m (80×80×60)
   - Duration: 12 minutes (went unstable)
   - Used in `comparison_cloud9.gif`

---

## Visualizations Created

### Parent simulations:
- Various GIFs in `cloud_gifs/` from parent runs

### Nested comparisons (side-by-side parent vs nested):
1. **`cloud_gifs/comparison.gif`** (431 KB)
   - Cloud #8, dx=25m, 5 min, 11 frames
   - Status: Stable, small cloud

2. **`cloud_gifs/comparison_cloud9.gif`** (2.7 MB)
   - Cloud #9, dx=25m, 12 min, 13 frames
   - Status: Goes unstable (qc_max → 28 g/kg)

3. **`cloud_gifs/comparison_10m.gif`** (1.5 MB) ⭐ **Latest**
   - Cloud #11, dx=10m, 4 min, 9 frames
   - Status: Ultra-high-res, goes unstable (qc_max → 8.3 g/kg)

---

## Total Disk Usage

**Parent simulations:** ~27 GB (9 complete experiments × 3 GB each)
**Nested simulations:** ~72 MB (current data only)
**Total:** ~27 GB

---

## Key Files for Nested Simulation Setup

- `nested_ic.npz` - Current initial conditions (Cloud #11, dx=10m)
- `config_nested.py` - Nested configuration
- `find_cloud.py` - Cloud detection utility
- `setup_nested.py` - IC interpolation tool
- `main_nested.py` - Nested simulation runner
- `boundary_conditions.py` - Relaxation BC implementation
- `forcing_nested.py` - Modified forcing
- `visualize_comparison.py` - Side-by-side visualization tool

---

## Notes

### Numerical Instability Issue
All nested simulations with strong clouds develop unrealistic cloud water accumulation:
- Physical limit: qc_max should be < 5-6 g/kg
- Observed: qc_max grows exponentially to 8-28 g/kg
- Likely causes:
  1. Saturation adjustment positive feedback
  2. Autoconversion threshold too high
  3. Timestep still too large despite reduction

### Recommendations for Future Work
1. Fix microphysics instability before running longer simulations
2. Consider using weaker clouds (qc_max < 2 g/kg initially)
3. Test even smaller timesteps (dt < 0.2s)
4. Implement better rain formation scheme
5. Add diagnostic output to track condensation rate
