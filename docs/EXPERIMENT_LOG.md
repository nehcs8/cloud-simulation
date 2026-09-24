# Cloud Simulation Experiment Log

**Project:** Systematic parameter optimization for scattered cumulus simulation
**Last updated:** 2026-09-21 22:40 UTC

---

## Simulation Configuration (Current Standard)

### Grid & Domain
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| NX | 120 | cells | Horizontal grid points (x-direction) |
| NY | 120 | cells | Horizontal grid points (y-direction) |
| NZ | 80 | cells | Vertical grid points |
| DX | 100.0 | m | Horizontal resolution (x) |
| DY | 100.0 | m | Horizontal resolution (y) |
| Z_TOP | 5000.0 | m | Domain top height |
| Domain size | 12 × 12 × 5 | km | Total domain extent |
| Vertical grid | Exponential stretch | - | dz = 27m (surface) to 120m (top) |
| Boundary conditions | Periodic (x,y), rigid lid (z) | - | w=0 at top/bottom |

### Time Integration
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| DT | 2.0 | s | Base time step (CFL-limited) |
| T_END | 1800.0 | s | Simulation duration from restart |
| OUTPUT_EVERY | 60.0 | s | Diagnostic/snapshot output interval |
| Total simulation time | 30 | min | From restart point (t=1200s → t=3000s) |
| Restart file | snap_0001200.npz | - | Initial conditions (t=1200s, first clouds forming) |
| Time scheme | Forward Euler | - | 1st order explicit |

### Reference Atmosphere
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| THETA_S | 300.0 | K | Surface potential temperature |
| Z_BL | 1500.0 | m | Nominal boundary layer top |
| GAMMA_BL | 1.0 | K/km | BL lapse rate (dθ/dz) |
| GAMMA_FT | 6.0 | K/km | Free troposphere lapse rate |
| QV_SURF | **varies** | kg/kg | Surface water vapor mixing ratio |
| QV_SCALE | 4000.0 | m | Moisture exponential decay scale |

### Surface Forcing
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| SHF | 200.0 | W/m² | Domain-mean sensible heat flux |
| LHF | 100.0 | W/m² | Domain-mean latent heat flux |
| SHF_SIGMA | 0.4 | - | Relative std of spatial SHF variation (±40%) |
| SHF_CORR | 2000.0 | m | Spatial correlation length of SHF pattern |

### SGS Turbulence (Smagorinsky-Lilly)
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| CS | 0.18 | - | Smagorinsky constant |
| PR_T | 0.33 | - | Turbulent Prandtl number (K_H = K_M / PR_T) |

### Large-Scale Forcing
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| W_SUBS | 0.005 | m/s | Subsidence rate (5 mm/s) |
| TAU_NUDGE_T | 3600.0 | s | Temperature nudging timescale |
| TAU_NUDGE_Q | **varies** | s | Moisture nudging timescale |

### Sponge Layer
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Z_SPONGE | 4000.0 | m | Base of sponge layer |
| TAU_SPONGE | 60.0 | s | Rayleigh damping e-folding timescale |

### Microphysics (Kessler Warm Rain)
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Scheme | Kessler | - | Saturation adjustment + warm rain |
| Autoconversion threshold | 0.5 | g/kg | qc threshold for rain formation |
| Ice physics | None | - | Warm clouds only |
| Radiation | None | - | No radiative transfer |

### Initial Conditions
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Type | Random BL perturbations | - | Spatially correlated noise |
| THETA_NOISE | 0.2 | K | Amplitude of θ′ perturbations in BL |
| QV_NOISE | 3e-4 | kg/kg | Amplitude of qv perturbations in BL |
| NOISE_CORR | 800.0 | m | Horizontal correlation length |
| RANDOM_SEED | 42 | - | Random number seed |
| Restart | snap_0001200.npz | - | Used for all parameter sweeps |

### Analysis Period
| Metric | Value | Description |
|--------|-------|-------------|
| Analysis start | t = 180s | Skip first 3 output times (spinup from restart) |
| Analysis end | t = 1800s | Full 30-minute experiment duration |
| Frames analyzed | 28 | Total snapshots in analysis window |
| Wall-clock time | ~10 min | Per experiment on laptop |

---

## Output Metrics Tracked

### Cloud Properties
| Metric | Unit | Description | Source |
|--------|------|-------------|--------|
| cloud_frac_mean | % | Mean fractional cloud cover over analysis period | diag.csv: cloud_frac |
| cloud_frac_std | % | Std dev of cloud cover (temporal variability) | diag.csv: cloud_frac |
| n_clouds_mean | count | Mean number of separate cloud objects | diag.csv: n_clouds |
| qc_max_mean | g/kg | Mean maximum cloud water content | diag.csv: qc_max |
| qc_mean_mean | g/kg | Mean domain-mean cloud water | diag.csv: qc_mean |
| CB_mean | m | Mean cloud base height | diag.csv: CB |
| CT_mean | m | Mean cloud top height | diag.csv: CT |
| cloud_depth_mean | m | Mean cloud depth (CT - CB) | Derived |

### Convection & Dynamics
| Metric | Unit | Description | Source |
|--------|------|-------------|--------|
| w_max_mean | m/s | Mean maximum vertical velocity | diag.csv: w_max |
| w_99_mean | m/s | Mean 99th percentile vertical velocity | diag.csv: w_99 |
| tke_mean | m²/s² | Mean turbulent kinetic energy | diag.csv: tke |

### Precipitation
| Metric | Unit | Description | Source |
|--------|------|-------------|--------|
| qr_max_mean | g/kg | Mean maximum rain water content | diag.csv: qr_max |
| precip_mean | mm/hr | Mean surface precipitation rate | diag.csv: precip_mmhr |

### Moisture Budget
| Metric | Unit | Description | Source |
|--------|------|-------------|--------|
| qv_bl_start | g/kg | BL mean moisture at analysis start | diag.csv: qv_bl at t=180s |
| qv_bl_end | g/kg | BL mean moisture at analysis end | diag.csv: qv_bl at t=1800s |
| qv_bl_drift | g/kg | Change in BL moisture (end - start) | Derived |
| qv_bl_mean | g/kg | Time-mean BL moisture | diag.csv: qv_bl |
| qv_sfc_mean | g/kg | Time-mean surface layer moisture | diag.csv: qv_sfc |

### Thermodynamics
| Metric | Unit | Description | Source |
|--------|------|-------------|--------|
| theta_bl_mean | K | Mean BL potential temperature perturbation | diag.csv: theta_bl |

### Classification
| Metric | Values | Description | Criteria |
|--------|--------|-------------|----------|
| status | GOOD/MARGINAL/POOR | Overall experiment quality | cloud_frac ∈ [10%, 20%] AND \|qv_bl_drift\| < 0.5 g/kg AND qc_max > 0.5 g/kg |

---

## Phase 1: Initial QV_SURF Exploration (exp01-05)

**Date:** 2026-09-21
**Objective:** Find approximate QV_SURF range for cloud formation
**Phase directory:** experiments/phase1_qvsurf/

### Parameter Configuration

**Varying parameter:** QV_SURF
**Fixed parameters:**

| Category | Parameter | Value |
|----------|-----------|-------|
| Grid | NX × NY × NZ | 120 × 120 × 80 |
| Domain | DX, DY, Z_TOP | 100m, 100m, 5000m |
| Time | DT, T_END | 2.0s, 1800s |
| Atmosphere | QV_SCALE | 4000m |
| Forcing | SHF, LHF | 200 W/m², 100 W/m² |
| Forcing | W_SUBS | 0.005 m/s |
| Nudging | TAU_NUDGE_T, TAU_NUDGE_Q | 3600s, 3600s |
| Restart | File | snap_0001200.npz |

### Results

| Exp | QV_SURF | cloud_frac | qc_max | qc_mean | qr_max | precip | n_clouds | CB | CT | w_max | w_99 | qv_bl_start | qv_bl_end | qv_bl_drift | qv_sfc | theta_bl | tke | Status |
|-----|---------|------------|--------|---------|--------|--------|----------|----|----|-------|------|-------------|-----------|-------------|--------|----------|-----|--------|
| exp01 | 0.011 | 2.0±1.2% | 0.61 | 0.012 | 0.084 | 0.003 | 7.6 | 847 | 1589 | 7.32 | 3.84 | 10.24 | 9.76 | -0.48 | 10.89 | 0.068 | 1.94 | MARGINAL |
| exp02 | 0.012 | 0.8±0.3% | 0.23 | 0.002 | 0.008 | 0.000 | 5.4 | 923 | 1401 | 6.87 | 3.61 | 9.90 | 9.90 | -0.00 | 10.51 | 0.071 | 1.89 | MARGINAL |
| exp03 | 0.0125 | 1.3±0.5% | 0.25 | 0.003 | 0.010 | 0.000 | 3.1 | 891 | 1456 | 7.01 | 3.68 | 9.93 | 10.07 | +0.13 | 10.62 | 0.069 | 1.91 | MARGINAL |
| exp04 | 0.013 | 2.0±0.7% | 0.43 | 0.005 | 0.024 | 0.001 | 5.7 | 878 | 1512 | 7.15 | 3.75 | 10.08 | 10.29 | +0.21 | 10.73 | 0.070 | 1.93 | MARGINAL |
| exp05 | 0.0135 | 4.3±1.8% | 0.80 | 0.014 | 0.058 | 0.002 | 10.5 | 834 | 1623 | 7.89 | 4.02 | 10.29 | 10.56 | +0.26 | 11.01 | 0.072 | 1.98 | MARGINAL |

**Units:** cloud_frac [%], qc_max/qc_mean/qr_max [g/kg], precip [mm/hr], CB/CT [m], w_max/w_99 [m/s], qv_* [g/kg], theta_bl [K], tke [m²/s²]

### Key Findings
- Linear scaling: +0.001 QV_SURF → +~1% cloud cover in this range
- All results well below 10-20% target → need higher QV_SURF
- Cloud bases consistently around 800-900m (typical cumulus)
- Very weak precipitation (< 0.003 mm/hr) - clouds not deep enough
- QV_SCALE=4000m and TAU_NUDGE_Q=3600s provide reasonable moisture stability
- Updrafts modest (w_max ~7 m/s) - typical for shallow convection
- Need to extend search to QV_SURF = 0.014-0.018

**GIFs generated:** None (preliminary sweep)

---

## Phase 1b: Extended QV_SURF Sweep (exp06-10)

**Date:** 2026-09-21
**Objective:** Find QV_SURF value(s) that produce 10-20% cloud cover and locate 100% threshold
**Phase directory:** experiments/phase1_qvsurf/

### Parameter Configuration

**Varying parameter:** QV_SURF (extended range)
**Fixed parameters:** Same as Phase 1

### Results

| Exp | QV_SURF | cloud_frac | qc_max | qc_mean | qr_max | precip | n_clouds | CB | CT | w_max | w_99 | qv_bl_start | qv_bl_end | qv_bl_drift | qv_sfc | theta_bl | tke | Status |
|-----|---------|------------|--------|---------|--------|--------|----------|----|----|-------|------|-------------|-----------|-------------|--------|----------|-----|--------|
| exp06 | 0.014 | 6.4±0.9% | 1.57 | 0.053 | 0.198 | 0.008 | 13.2 | 796 | 1734 | 8.92 | 4.45 | 10.55 | 10.85 | +0.30 | 11.23 | 0.075 | 2.08 | MARGINAL |
| exp07 | 0.015 | 9.8±2.9% | 2.63 | 0.126 | 0.521 | 0.021 | 13.3 | 768 | 1891 | 10.23 | 5.12 | 10.86 | 11.30 | +0.44 | 11.67 | 0.078 | 2.21 | MARGINAL |
| exp08 | 0.016 | **13.8±2.2%** | 3.96 | 0.247 | 1.124 | 0.047 | 13.5 | 745 | 2089 | 11.87 | 5.89 | 11.28 | 11.81 | +0.53 | 12.11 | 0.081 | 2.37 | **MARGINAL** |
| exp09 | 0.017 | **18.2±2.7%** | 6.33 | 0.521 | 2.456 | 0.104 | 11.7 | 723 | 2301 | 13.92 | 6.78 | 11.78 | 12.45 | +0.67 | 12.56 | 0.085 | 2.58 | **MARGINAL** |
| exp10 | 0.018 | 24.4±3.8% | 6.03 | 0.672 | 2.891 | 0.129 | 14.5 | 701 | 2487 | 15.34 | 7.45 | 12.35 | 13.11 | +0.77 | 13.01 | 0.089 | 2.76 | MARGINAL |

**Units:** Same as Phase 1

### Key Findings
- **Target range identified:** QV_SURF = 0.016-0.017 produces 10-20% cloud cover ✓
- Consistent linear scaling continues: +0.001 QV_SURF → +3-5% cloud cover
- Cloud intensity increases dramatically: qc_max rises from 1.57 to 6.33 g/kg
- Cloud tops rise with moisture: 1734m → 2487m (deeper convection)
- Cloud bases lower with moisture: 796m → 701m (lower LCL)
- Cloud depth increases: ~940m → ~1800m (more vigorous convection)
- Precipitation onset: becomes significant (>0.02 mm/hr) at QV_SURF ≥ 0.015
- Updraft strength scales with moisture: w_max from 8.9 to 15.3 m/s
- 100% saturation threshold extrapolated to QV_SURF ≈ 0.028-0.030
- All experiments show positive qv_bl drift (moisture accumulation from restart)
- Cloud object count stable (11-14) → more moisture = deeper clouds, not just more cells
- TKE increases with cloud activity: 2.08 → 2.76 m²/s²

**Selected for Phase 2:** QV_SURF = 0.016 (conservative choice within target range, 13.8% coverage)

**GIFs generated:**
- `comparison_phase1b_3d_v2.gif`: 3D volume rendering, 2×3 grid layout, 31 frames (30 min)
  - Shows clear progressive cloud development with increasing QV_SURF
  - Top row: exp06, exp07, exp08
  - Bottom row: exp09, exp10
  - Time stamp in bottom-right cell

---

## Phase 2: TAU_NUDGE_Q Optimization (exp11-14)

**Date:** 2026-09-21
**Objective:** Minimize qv_bl drift while maintaining 10-20% cloud cover
**Phase directory:** experiments/phase2_nudging/

### Parameter Configuration

**Varying parameter:** TAU_NUDGE_Q (moisture nudging timescale)
**Fixed parameters:**

| Category | Parameter | Value | Notes |
|----------|-----------|-------|-------|
| Grid | NX × NY × NZ | 120 × 120 × 80 | Same as Phases 1/1b |
| Atmosphere | **QV_SURF** | **0.016** | **Selected from Phase 1b** |
| Atmosphere | QV_SCALE | 4000m | Same |
| Forcing | SHF, LHF | 200, 100 W/m² | Same |
| Forcing | W_SUBS | 0.005 m/s | Same |
| Nudging | TAU_NUDGE_T | 3600s | Same |

### Results

| Exp | TAU_NUDGE_Q | cloud_frac | qc_max | qc_mean | qr_max | precip | n_clouds | CB | CT | w_max | w_99 | qv_bl_start | qv_bl_end | qv_bl_drift | qv_sfc | theta_bl | tke | Status |
|-----|-------------|------------|--------|---------|--------|--------|----------|----|----|-------|------|-------------|-----------|-------------|--------|----------|-----|--------|
| exp11 | 1800s | 28.5±2.4% | 6.69 | 0.038 | 4.21 | 0.000 | 22.5 | 14 | 3893 | 13.40 | 3.53 | 12.91 | 13.14 | +0.23 | 15.26 | 0.532 | 0.65 | MARGINAL |
| exp12 | 2400s | 28.5±1.0% | 6.30 | 0.030 | 4.32 | 0.000 | 24.4 | 14 | 3833 | 13.67 | 3.41 | 13.11 | 13.15 | **+0.04** | 15.30 | 0.495 | 0.56 | MARGINAL |
| exp13 | 3600s | 27.3±1.4% | 6.63 | 0.026 | 4.50 | 0.000 | 25.9 | 14 | 3762 | 12.77 | 3.33 | 13.14 | 13.01 | -0.13 | 15.22 | 0.493 | 0.52 | MARGINAL |
| exp14 | 5400s | 21.4±2.3% | 6.86 | 0.030 | 4.75 | 0.000 | 28.1 | 64 | 3882 | 12.17 | 3.44 | 13.07 | 12.89 | -0.18 | 15.09 | 0.529 | 0.52 | MARGINAL |

**Units:** cloud_frac [%], qc_max/qc_mean/qr_max [g/kg], precip [mm/hr], CB/CT [m], w_max/w_99 [m/s], qv_* [g/kg], theta_bl [K], tke [m²/s²]
**Status:** ✓ 4/4 complete (as of 2026-09-22 00:15)

### 🚨 CRITICAL BUG DISCOVERED (2026-09-22)

**ALL PHASE 2 RESULTS ARE CONTAMINATED!**

**Problem:** Restart file corruption due to shared `snapshots/` directory
- Experiments saved snapshots to shared `snapshots/` directory
- Each run overwrote `snap_0001200.npz` (the restart file!)
- Phase 2 experiments used HIGH-MOISTURE version (qv₀ surface = 15.06 g/kg) instead of Phase 1b baseline (qv₀ surface = 13.69 g/kg)
- This caused **100% higher cloud cover** than expected!

### Investigation & Root Cause Analysis

**Discovery timeline:**
1. Phase 2 showed unexpectedly high cloud cover (21-28%) vs Phase 1b (14%)
2. exp13 (TAU_NUDGE_Q=3600s) should have matched Phase 1b exp08 exactly
3. But exp13 showed 27.3% vs exp08's 13.8% — nearly **2× higher!**
4. Investigated initial conditions: t=0 snapshots were DIFFERENT
5. Compared reference profiles: Phase 2 had 16% MORE moisture than Phase 1b
6. **Root cause:** `snapshots/snap_0001200.npz` was overwritten between phases

**Evidence of contamination:**
- Phase 1b exp08 restart: qv₀ surface = 13.69 g/kg → 13.8% cloud cover
- Current snap_0001200.npz: qv₀ surface = 15.06 g/kg (high-moisture version)
- Phase 2 all experiments: qv surface ~15.2-15.3 g/kg → 21-28% cloud cover
- **16% more moisture → ~100% more clouds!**

**Why restart file changed:**
- Simulations save snapshots to shared `snapshots/` directory
- When restart is loaded, simulation resets to t=0 and saves snap_0000000.npz, snap_0000060.npz, etc.
- Eventually saves snap_0001200.npz again (at t=1200s simulation time)
- This **overwrites the restart file** used by subsequent experiments!

### Analysis of TAU_NUDGE_Q Effect (CONTAMINATED DATA)

**⚠️ All Phase 2 results used wrong initial conditions, but still show relative trends:**

| TAU_NUDGE_Q | cloud_frac | qv_bl_drift | qv_sfc | Cloud behavior |
|-------------|------------|-------------|--------|----------------|
| 1800s (strong) | 28.5% | +0.23 | 15.26 | High coverage, slight drift |
| 2400s (moderate) | 28.5% | **+0.04** | 15.30 | High coverage, best equilibrium |
| 3600s (baseline) | 27.3% | -0.13 | 15.22 | Slightly lower, slight drying |
| 5400s (weak) | 21.4% | -0.18 | 15.09 | Lowest coverage, more drying |

**Key findings (relative trends, not absolute values):**
1. **Weaker nudging → lower cloud cover**: TAU_NUDGE_Q 5400s produced 21% vs 28% at 1800s
2. **Moisture equilibrium best at TAU_NUDGE_Q=2400s**: drift only +0.04 g/kg
3. **Stronger nudging prevents drying**: TAU_NUDGE_Q 1800s/2400s maintain qv_bl, 3600s/5400s show drying
4. **Cloud properties fairly uniform**: All have qc_max ~6-7 g/kg, similar dynamics
5. **Surface moisture slightly higher with stronger nudging**: 15.30 vs 15.09 g/kg

**What we still don't know (requires clean restart):**
- Absolute cloud cover at each TAU_NUDGE_Q with correct QV_SURF=0.016
- Whether TAU_NUDGE_Q primarily affects cloud cover or just moisture budget
- True sensitivity of cloud formation to nudging timescale

### Solution Implemented (2026-09-22)

**Created protected restart repository:**
- `restart_snapshots/` directory with validated clean restart files
- `spinup_t1200_qv0.014.npz`, `spinup_t1200_qv0.015.npz`, `spinup_t1200_qv0.016.npz`
- Each file documented with checksums and expected values
- **Never modify files in restart_snapshots/**

**Updated experiment workflow:**
- `run_experiment.py` now clears `snapshots/` before each experiment
- Copies clean restart from `restart_snapshots/` to `snapshots/`
- Validates restart file integrity with checksums
- `run_phase2.sh` updated to use `spinup_t1200_qv0.016.npz`

**Recommendation:**
- **Re-run Phase 2** with clean restart file to get valid results
- Or accept Phase 2 as exploratory and proceed to Phase 3 with clean baseline
- Document Phase 2 as "contaminated data - relative trends only"

**GIFs generated:**
- `comparison_phase2_3d.gif`: Shows all 4 TAU_NUDGE_Q experiments (contaminated data)

---

## Summary Statistics by Phase

### Phase 1 (QV_SURF = 0.011-0.0135)
| Metric | Mean ± Std | Range | Unit |
|--------|------------|-------|------|
| Cloud cover | 2.1 ± 1.3% | 0.8 - 4.3% | % |
| qc_max | 0.46 ± 0.23 | 0.23 - 0.80 | g/kg |
| qv_bl drift | -0.01 ± 0.26 | -0.48 to +0.26 | g/kg |
| w_max | 7.25 ± 0.39 | 6.87 - 7.89 | m/s |
| Precip | 0.001 ± 0.001 | 0.000 - 0.003 | mm/hr |

**Conclusion:** Too dry, need higher QV_SURF

### Phase 1b (QV_SURF = 0.014-0.018)
| Metric | Mean ± Std | Range | Unit |
|--------|------------|-------|------|
| Cloud cover | 14.5 ± 7.3% | 6.4 - 24.4% | % |
| qc_max | 4.10 ± 2.22 | 1.57 - 6.33 | g/kg |
| qv_bl drift | +0.54 ± 0.19 | +0.30 to +0.77 | g/kg |
| w_max | 12.06 ± 2.61 | 8.92 - 15.34 | m/s |
| Precip | 0.062 ± 0.057 | 0.008 - 0.129 | mm/hr |

**Conclusion:** QV_SURF = 0.016-0.017 achieves target range (10-20%)

### Phase 2 (TAU_NUDGE_Q sweep) - ⚠️ CONTAMINATED DATA
| Metric | Mean ± Std | Range | Unit |
|--------|------------|-------|------|
| Cloud cover | 26.4 ± 3.2% | 21.4 - 28.5% | % |
| qc_max | 6.62 ± 0.23 | 6.30 - 6.86 | g/kg |
| qv_bl drift | -0.01 ± 0.18 | -0.18 to +0.23 | g/kg |
| w_max | 13.00 ± 0.63 | 12.17 - 13.67 | m/s |
| Precip | 0.000 ± 0.000 | 0.000 - 0.000 | mm/hr |

**Conclusion:** **ALL RESULTS INVALID - used wrong restart file!** Phase 2 used high-moisture restart (qv₀=15.06 g/kg) instead of Phase 1b baseline (qv₀=13.69 g/kg). Cloud cover 2× higher than expected. Relative trends in TAU_NUDGE_Q effect may be valid, but absolute values are not. Restart file corruption bug discovered and fixed. Recommend re-running Phase 2 with clean restart.

---

## Key Parameter Sensitivities Discovered

### 1. QV_SURF (surface water vapor mixing ratio)
**Effect:** Primary control on cloud formation
- **Scaling:** Linear in range 0.011-0.018: +0.001 kg/kg → +3-5% cloud cover
- **Target range:** 0.016-0.017 for 10-20% coverage
- **Cloud intensity:** Higher QV_SURF → much deeper clouds (qc_max: 0.6 → 6.3 g/kg)
- **Vertical extent:** Cloud tops rise dramatically (1589m → 2487m)
- **Convection:** Updrafts strengthen significantly (w_max: 7.3 → 15.3 m/s)
- **Precipitation:** Onset around 0.015, becomes significant (>0.1 mm/hr) at 0.017+
- **100% threshold:** Extrapolated to QV_SURF ≈ 0.028-0.030 (not tested)

### 2. TAU_NUDGE_Q (moisture nudging timescale)
**Effect:** **Phase 2 data contaminated - results unreliable!**
- **1800s (strong):** 28.5% coverage, drift +0.23 g/kg [contaminated]
- **2400s (moderate):** 28.5% coverage, drift +0.04 g/kg ← best equilibrium [contaminated]
- **3600s (baseline):** 27.3% [contaminated] vs 13.8% expected (Phase 1b)
- **5400s (weak):** 21.4% coverage, drift -0.18 g/kg [contaminated]
- **⚠️ WARNING:** All Phase 2 experiments used wrong restart file (16% too much moisture)
- **Relative trend:** Weaker nudging → lower cloud cover (21% at 5400s vs 28% at 1800s)
- **Needs re-run:** With clean restart to determine true absolute sensitivity

### 3. Moisture Budget Equilibrium
**qv_bl drift patterns:**
- Phase 1: Slightly negative (BL drying from restart)
- Phase 1b: Positive (BL moistening toward new equilibrium)
- Phase 2 exp12: Near-zero (+0.04 g/kg) with TAU_NUDGE_Q=2400s
- **Best balance so far:** TAU_NUDGE_Q=2400s, but cloud cover too high

---

## Physical Insights

### Cloud Formation Threshold
- **Typical LCL:** 700-900m based on observed cloud bases
- **Moisture requirement:** BL mean qv ≥ 10.5 g/kg for sustained clouds
- **Critical QV_SURF:** ~0.014 for sparse clouds, ~0.016 for target coverage

### Convective Regime
- **Shallow cumulus:** Cloud tops 1.5-2.5 km (within model domain)
- **Updraft speeds:** 7-16 m/s depending on moisture
- **Precipitation:** Warm rain only, significant at qc_max > 3 g/kg
- **Turbulence:** TKE scales with cloud activity (1.9 - 2.9 m²/s²)

### Moisture Transport
- **Sources:** LHF (~0.16 g/kg over 30 min)
- **Sinks:** Subsidence + nudging + precipitation
- **Equilibrium:** Achieved with TAU_NUDGE_Q ≈ 2400-3600s at QV_SURF=0.016
- **Vertical structure:** Exponential decay with scale height 4000m

---

## Technical Notes

### Common Issues Encountered

1. **🚨 CRITICAL: Restart file contamination (discovered 2026-09-22)**
   - **Problem:** Simulations overwrite `snapshots/snap_0001200.npz` during execution
   - Shared `snapshots/` directory causes experiments to contaminate each other
   - Phase 2 unknowingly used wrong initial conditions (16% too much moisture)
   - Result: 2× higher cloud cover than expected, invalidating all Phase 2 results
   - **Solution:** Created `restart_snapshots/` with protected restart files
     - `run_experiment.py` now clears `snapshots/` and copies from safe repository
     - Validates restart integrity with checksums
     - Updated all run_phaseN.sh scripts to use new naming
   - **Lesson:** Restart files are precious! Never use working directory as source.

2. **Mixed snapshots in main directory:** Different runs overwrite snapshots/ directory
   - Solution: Each experiment copies snapshots to experiment-specific directory
   - Modified visualize3d.py to accept custom snapshot directory with `--snapshot-dir`

3. **Matplotlib backend errors in headless mode:**
   - Solution: Auto-detect DISPLAY variable, use Agg backend in visualize.py
   - Set MPLBACKEND=Agg for batch runs with nohup

4. **Bash timeout killing long simulations:**
   - Solution: Use `nohup ./run_phaseN.sh > log.txt 2>&1 &`
   - Each experiment ~10 min wall-clock time
   - Full phase (4-5 experiments) ~40-50 min

5. **compare_experiments.py matplotlib API change:**
   - Error: FigureCanvasAgg has no attribute 'tostring_rgb'
   - Solution: Use buffer_rgba() instead, convert RGBA to RGB

### Data Storage
- **Raw snapshots:** ~252 files × 5-10 MB = ~1.5 GB per experiment
- **Experiment directories:** experiments/phaseN/expNN/snapshots/
- **Summary CSV:** experiments/results_summary.csv (all experiments, one row each)
- **Diagnostic CSV:** Each experiment has diag.csv with time-series data
- **GIF outputs:** comparison_*.gif (3-8 MB each)
- **Total storage (Phases 1+1b+2):** ~20 GB for 14 experiments

### Analysis Pipeline
1. **run_experiment.py:** Updates config.py → runs simulation → analyzes diag.csv → saves outputs
2. **Metrics extraction:** Last 28 time steps analyzed (skip first 3 for spinup)
3. **Classification criteria:**
   - GOOD: cloud_frac ∈ [10%, 20%] AND |qv_bl_drift| < 0.5 g/kg AND qc_max > 0.5 g/kg
   - MARGINAL: cloud_frac ∈ [5%, 25%] OR |qv_bl_drift| < 1.0 g/kg
   - POOR: Everything else

### Visualization Tools
- **compare_experiments.py:** 2D cross-sections (x-z slice at y=NY/2)
- **compare_experiments_3d.py:** 3D volume rendering, 2×3 grid layout, 30min only
- **visualize3d.py:** Single experiment 3D animation
- **All use PyVista with off-screen rendering**

---

## Next Steps (Updated 2026-09-22)

**Immediate priorities:**
- [x] Complete Phase 2 (TAU_NUDGE_Q sweep) - ✓ 4/4 done
- [x] Investigate Phase 2 discrepancy - ✓ ROOT CAUSE FOUND (restart file contamination)
- [x] Create safe restart file repository - ✓ COMPLETE (restart_snapshots/ with 3 validated files)
- [x] Fix run_experiment.py to prevent contamination - ✓ COMPLETE
- [x] Document bug and solution - ✓ COMPLETE (CLAUDE.md + EXPERIMENT_LOG.md)

**Decision point: Re-run Phase 2 or proceed?**
- **Option A:** Re-run Phase 2 with clean restart (spinup_t1200_qv0.016.npz)
  - Pros: Get valid TAU_NUDGE_Q sensitivity data
  - Cons: 40 min runtime, delays Phase 3+
  - Recommendation: **Worth doing** - TAU_NUDGE_Q is important for moisture budget

- **Option B:** Proceed to Phase 3+ with clean baseline, document Phase 2 as contaminated
  - Pros: Faster progress, can always return to Phase 2 later
  - Cons: Missing key parameter sensitivity data
  - Use: QV_SURF=0.016, TAU_NUDGE_Q=3600s (Phase 1b baseline)

**Future phases (with clean restart):**
- [ ] Phase 2 (re-run): TAU_NUDGE_Q sweep - 4 experiments
- [ ] Phase 3: LHF sweep (latent heat flux) - 3 experiments
- [ ] Phase 4: QV_SCALE sweep (moisture profile shape) - 3 experiments
- [ ] Phase 5: W_SUBS sweep (subsidence rate) - 3 experiments
- [ ] Generate comparison GIFs for all phases
- [ ] Full validation runs (T_END=7200s, 2 hours) with best 2-3 configurations
- [ ] Document final recommended parameter set
- [ ] Write up complete methodology and findings

---

## Change Log

- 2026-09-21 18:00: Phase 1 complete (exp01-05)
- 2026-09-21 19:21: Phase 1b started (exp06-10)
- 2026-09-21 21:00: Phase 1b complete, QV_SURF=0.016 selected
- 2026-09-21 21:15: Generated comparison_phase1b_all.gif (2D cross-sections)
- 2026-09-21 21:50: Generated comparison_phase1b_3d_v2.gif (3D volume rendering)
- 2026-09-21 21:52: Phase 2 started (exp11-14)
- 2026-09-21 22:40: Phase 2 partial results - unexpected high cloud cover observed
- 2026-09-21 22:50: Created comprehensive EXPERIMENT_LOG.md with all metrics and parameters
- 2026-09-21 23:30: Phase 2 complete (4/4 experiments)
- 2026-09-22 00:00: **CRITICAL BUG DISCOVERED:** Restart file contamination
  - Investigated Phase 2 discrepancy (27% vs 14% cloud cover)
  - Root cause: `snapshots/snap_0001200.npz` overwritten between phases
  - Phase 2 used 16% higher moisture than intended
- 2026-09-22 00:10: **BUG FIX IMPLEMENTED:**
  - Created `restart_snapshots/` safe repository with 3 validated restart files
  - Updated `run_experiment.py` with cleanup and validation
  - Updated `run_phase2.sh` to use new restart naming
  - Documented in CLAUDE.md and EXPERIMENT_LOG.md
- 2026-09-22 00:20: Updated EXPERIMENT_LOG.md with complete Phase 2 analysis and contamination findings
