# LES Core Rebuild Plan

**Date:** 2026-09-28
**Status:** Planning Phase
**Goal:** Build a proper shallow-cumulus LES with validated BOMEX case

---

## Executive Summary

The current LES has fundamental numerical and physical problems that prevent realistic cumulus cloud simulation. This document outlines a complete rebuild of the solver core, following established best practices from FastEddy, UWLCM, and the BOMEX intercomparison (Siebesma et al. 2003).

**Key insight:** We've been adding parametric fixes (entrainment, Coriolis, WENO-qc) to mask deeper solver problems. The rebuild will fix the foundation first, then validate against standard cases before adding any optional physics.

---

## Critical Problems Identified

### 1. **Pressure Projection Doesn't Remove Divergence** ⚠️ CRITICAL
**Problem:**
- u, v are cell-centered with 2Δx differences
- w is face-centered (staggered in z)
- Poisson solver uses eigenvalue 2(cos kΔx - 1)/Δx²
- Actual operator formed: -sin²(kΔx)/Δx²
- **These don't match** → ~50% of divergence remains after projection
- Checkerboard (2Δx) modes can't be corrected at all

**Consequences:**
- Spurious convergence/divergence patterns
- Artificial vertical velocities
- Grid-scale noise in horizontal flow
- Likely source of "artificial rotation" we tried to fix with Coriolis

**Fix:**
- Move u, v to x/y faces (full Arakawa C-grid)
- Match divergence/gradient stencils to Poisson eigenvalues
- Verify max|∇·u| < machine precision after projection

**References:**
- Arakawa & Lamb (1977) - C-grid design
- Durran (1999) - Numerical Methods for Fluid Dynamics

---

### 2. **Initial Sounding Can't Produce Cumulus**
**Problem:**
```
z        θ       qv      RH
0 m      300.0   15.9    70%
1000 m   301.0   12.4    84%
1500 m   301.6   10.9    92%   ← should be inversion!
2000 m   304.5   9.7     88%
3000 m   310.7   7.5     81%   ← free troposphere too humid
```

**Issues:**
1. No sharp inversion at cloud layer top
2. Free troposphere nearly saturated (RH 81-92%)
3. Subcloud layer not well-mixed (1 K/km instead of ~0)
4. Exponential qv(z) with H = 4 km is unrealistic

**Consequences:**
- Entrainment doesn't cause evaporation (no dry air!)
- Clouds merge into smooth slabs instead of sharp turrets
- No "cauliflower" structure (requires dry air mixing)

**Fix:**
- Use BOMEX sounding (see Phase 2 below)
- Sharp inversion: θ jumps from 302.4 K (1480 m) to 308.2 K (2000 m)
- Dry free troposphere: qt drops from 10.7 g/kg to 4.2 g/kg across inversion
- Well-mixed BL: constant θ_l and qt below cloud base

---

### 3. **Inconsistent, Over-Diffusive Numerics**
**Problems:**
1. First-order upwind for θ, qv, u, v, w
   - Implicit diffusion: ~|u|Δx/2 ≈ 50-100 m²/s
   - Comparable to or larger than SGS diffusion!
2. WENO-5 only for qc (different from qv, θ)
   - Inconsistent transport → spurious condensation/evaporation at edges
3. Smagorinsky with Pr_t = 0.33 (K_H = 3 K_M)
   - Excessive scalar diffusion
4. Forward Euler time stepping
   - Requires excessive damping for stability
   - Makes high-order advection "unstable"

**Consequences:**
- Thermals smeared before they can organize
- Cloud edges incorrectly resolved
- Can't use consistent high-order schemes

**Fix:**
- RK3 time integration (Wicker-Skamarock 2002)
- WENO-5 for all fields (or consistent 3rd/5th order)
- Prognose θ_l, qt; diagnose qc (standard LES practice)
- Reduce Pr_t to 1/3 (typical for atmospheric LES)

**References:**
- Wicker & Skamarock (2002) - RK3 schemes for atmospheric models
- Stevens et al. (2005) - DALES LES (θ_l/qt framework)

---

### 4. **Forcings Fight Convection**
**Problems:**
1. Profile nudging with TAU_NUDGE = 3600 s
   - Relaxes domain-mean θ′ → 0 at all levels
   - **Removes the 200 W/m² surface heating we're applying!**
   - Prevents realistic BL deepening
2. Sponge layer issues:
   - Multiplies qv by damping → drives to zero (not qv0)
   - Same for u, v (drives to 0, not geostrophic wind)
   - τ = 60 s is too strong (typical: 300-600 s)
3. Subsidence advects fixed gradient dθ0/dz
   - Should advect evolving mean profile
4. Coriolis without geostrophic wind
   - Implemented as +fv, -fu
   - Correct form: f(v - v_g), -f(u - u_g)
   - Current version just creates inertial oscillation

**Fix:**
- Don't nudge the boundary layer (only above ~2 km)
- Fix sponge to relax toward reference profiles, not zero
- Subsidence should use mean state, not reference state
- Add geostrophic wind to Coriolis terms

---

### 5. **Surface Boundary Condition**
**Problem:**
- Sets u = v = 0 in lowest cell each timestep
- This is an artificial wall at cell center
- Has no surface-layer physics
- Re-introduces divergence after projection

**Fix:**
- Implement Monin-Obukhov surface layer
- Apply surface stress u*² as momentum flux
- Compute sensible/latent heat fluxes from bulk formulae

**References:**
- Businger et al. (1971) - M-O similarity
- Sullivan et al. (1994) - LES surface layer

---

### 6. **Microphysics Issues**
**Problems:**
1. Ad-hoc entrainment parameterization
   - LES should resolve entrainment, not parameterize it!
   - Not conservative: delta_qc = -ε·qc deletes water
2. Autoconversion disabled (_K1 = 0)
   - Rain processes are all dead code
3. Sedimentation subtracts w from fall speed
   - But qr is already advected by w in dynamics!
   - Double-counts vertical transport

**Fix:**
- Remove entrainment parameterization entirely
- For non-precipitating cumulus: keep _K1 = 0 (fine)
- For precipitating cumulus: enable Kessler scheme properly
- Remove w from sedimentation (handled by advection)

---

### 7. **Secondary Issues**
- Smagorinsky has no stability correction
  - Add Richardson-number damping near inversion
- Boussinesq over 5 km is marginal
  - Consider anelastic (already have ρ0, just unused)
- 2-hour runs are mostly spin-up
  - BOMEX runs 6 hours, statistics from hours 3-6

---

## Implementation Plan

### **PHASE 1: Core Solver Rebuild** (Week 1-2)

**Priority:** Fix the foundation before anything else.

#### 1.1 C-Grid Staggering
**File:** `grid.py`, `state.py`, `dynamics.py`

**Current state:**
```python
# Cell-centered
u: (nx, ny, nz)
v: (nx, ny, nz)
# Face-centered (z only)
w: (nx, ny, nz+1)
```

**Target state (Arakawa C-grid):**
```python
# Face-centered
u: (nx+1, ny, nz)    # on x-faces
v: (nx, ny+1, nz)    # on y-faces
w: (nx, ny, nz+1)    # on z-faces (already correct)

# Cell-centered
theta_l: (nx, ny, nz)
qt: (nx, ny, nz)
p: (nx, ny, nz)
```

**Implementation steps:**
1. Modify `State.__init__()` to allocate face-staggered u, v
2. Update all interpolation operations (u→cell, v→cell, etc.)
3. Rewrite `divergence()` to use face velocities
4. Rewrite `gradient()` to match divergence stencil
5. Update boundary conditions for u, v on domain edges
6. Verify: run divergence test, check max|∇·u| < 1e-12

**Test:** Feed random velocity → project() → measure residual divergence

---

#### 1.2 RK3 Time Integration
**File:** `main.py`

**Current:** Forward Euler (1st order, conditionally unstable)
```python
s.u += dt * rhs_u
```

**Target:** Wicker-Skamarock RK3 (3rd order, more stable)
```python
# RK3 coefficients
α = [1/3, 1/2, 1]
β = [1/3, 1/2, 1]

for stage in range(3):
    rhs = compute_tendencies(s)
    s_stage = s0 + α[stage] * dt * rhs
    s = s0 + β[stage] * dt * rhs
```

**Implementation:**
1. Create `timestepping.py` with RK3 integrator
2. Move all tendency computations into single function
3. Save initial state s0 before stages
4. Update CFL condition (can use larger dt with RK3)

**Test:** Run conservation tests (mass, energy should be conserved)

---

#### 1.3 Consistent High-Order Advection
**File:** `advection.py`

**Current:** Mix of upwind-1 (momentum, scalars) and WENO-5 (qc only)

**Target:** WENO-5 for all fields

**Implementation:**
1. Extend WENO-5 to θ_l, qt, u, v, w
2. Use same flux limiters throughout
3. Handle staggered grid correctly (fluxes at faces)

**Alternative (faster):** Use 3rd or 5th order upwind-biased for all fields

**Test:** Advect Gaussian blob, measure numerical diffusion

---

#### 1.4 Moist Conserved Variables
**File:** `state.py`, `microphysics.py`, `advection.py`

**Current:** Transport θ, qv, qc separately

**Target:** Transport θ_l, qt; diagnose qc

**Physical definitions:**
```python
# Liquid water potential temperature
theta_l = theta - (Lv/cp) * qc / pi0

# Total water mixing ratio
qt = qv + qc

# Diagnose qc from saturation:
if qt > qs(theta_l):
    qc = qt - qs(theta_l)
else:
    qc = 0
```

**Benefits:**
- Conserves total water exactly (no numerical condensation/evaporation)
- Standard practice in shallow-cumulus LES (DALES, UWLCM, etc.)
- Eliminates θ/qv/qc consistency problems

**Implementation:**
1. Add `theta_l` and `qt` to `State`
2. Modify advection to transport θ_l, qt
3. Rewrite saturation adjustment to diagnose qc from (θ_l, qt)
4. Update output to save θ_l, qt (keep θ, qv, qc for compatibility)

---

#### 1.5 Monin-Obukhov Surface Layer
**File:** `forcing.py` (new: `surface_layer.py`)

**Replace:** `u[..., 0] = 0; v[..., 0] = 0`

**With:** Bulk flux formulation
```python
# Surface friction velocity
u* = sqrt(τ_surface / ρ)

# Monin-Obukhov length
L = -u*³ / (κ * g/θ0 * w'θ')

# Stability functions ψ_m(z/L), ψ_h(z/L)

# Surface stress
τ = ρ * u*² * (u_lowest / |u_lowest|)

# Apply as momentum flux (not velocity BC)
du/dt|_sfc = -τ / (ρ Δz)
```

**Implementation:**
1. Create `surface_layer.py` with M-O functions
2. Compute u* iteratively from lowest-level wind
3. Apply stress as flux, not velocity BC
4. Update heat/moisture fluxes similarly

**References:**
- Businger et al. (1971)
- Paulson (1970) - ψ_m, ψ_h functions

---

### **PHASE 2: BOMEX Validation** (Week 3)

**Goal:** Run the standard BOMEX case exactly, with all band-aids removed.

#### 2.1 BOMEX Sounding
**File:** `config.py`, `state.py`

**Domain:**
- Horizontal: 6.4 km × 6.4 km
- Vertical: 3 km
- Resolution: Δx = 100 m, Δz = 40 m
- 64 × 64 × 75 grid points

**Initial conditions:**
```python
# Liquid water potential temperature [K]
z [m]     θ_l [K]
0         298.7
520       298.7
1480      302.4    # Cloud layer top
2000      308.2    # Inversion!
3000      311.85

# Total water mixing ratio [g/kg]
z [m]     qt [g/kg]
0         17.0
520       16.3
1480      10.7     # Cloud layer top
2000      4.2      # Dry free troposphere!
3000      3.0
```

**Surface fluxes:**
```python
w'θ'_surface = 8e-3  # K m/s (constant)
w'qt'_surface = 5.2e-5  # m/s (constant)
```

**Large-scale forcing:**

1. **Subsidence:** w_ls(z) reaches -0.65 cm/s at 1500 m
```python
if z < 1500:
    w_ls = -0.0065 * z / 1500  # m/s
else:
    w_ls = -0.0065  # m/s (constant above)
```

2. **Radiative cooling:** -2 K/day (constant with height)

3. **Advective tendencies:**
```python
# Horizontal advection drying
if z < 1500:
    dqt/dt|_adv = -1.2e-8  # s^-1 (drying)
else:
    dqt/dt|_adv = 0

# No thermal advection (dθ/dt|_adv = 0)
```

**Geostrophic wind:**
```python
u_g = -10 m/s  # westerly
v_g = 0 m/s
f = 0.376e-4 s^-1  # Coriolis parameter (15°N)
```

**Implementation:**
1. Create `bomex_case.py` with all setup functions
2. Implement subsidence operator properly (use mean profiles)
3. Add radiative cooling tendency
4. Add large-scale advection tendencies
5. Set up geostrophic wind and Coriolis

---

#### 2.2 Strip All Optional Physics
**Files:** All

**Remove/Disable:**
- ❌ Entrainment parameterization
- ❌ Cloud shading
- ❌ Heterogeneous surface forcing
- ❌ Cold pool scheme
- ❌ WENO cauliflower fix (use consistent advection)
- ❌ Profile nudging (except sponge layer)
- ❌ Pulsed forcing

**Keep only:**
- ✅ Saturation adjustment (θ_l/qt → qc)
- ✅ Smagorinsky SGS (with stability correction)
- ✅ Sponge layer (fixed to relax properly)
- ✅ Coriolis (with geostrophic wind)

---

#### 2.3 Run BOMEX for 6 Hours
**File:** `config.py`

```python
T_END = 21600  # 6 hours
OUTPUT_EVERY = 300  # 5 minutes
STATISTICS_START = 10800  # Start stats at hour 3
```

**Expected results** (from Siebesma et al. 2003):

| Metric | BOMEX Intercomparison | Our Target |
|--------|----------------------|------------|
| Cloud base | 500-600 m | 500-600 m |
| Cloud top | 1500-2000 m | 1500-2000 m |
| Cloud fraction | 2-4% | 2-4% |
| LWP | ~20 g/m² | 15-25 g/m² |
| Surface precip | ~0.1 mm/day | ~0.1 mm/day |
| w* (convective) | ~2 m/s | 1.5-2.5 m/s |

**Diagnostics to compute:**
```python
# Time-averaged (hours 3-6)
cloud_base = min(z where qc > 0.01 g/kg)
cloud_top = max(z where qc > 0.01 g/kg)
cloud_fraction = fraction(qc > 0.01 g/kg)
LWP = ∫ qc dz  # Liquid water path

# Vertical profiles (domain-mean)
<θ_l>(z), <qt>(z), <qc>(z)
<u>(z), <v>(z), <w'²>(z)
<w'θ'>(z), <w'qt'>(z)

# Cloud size distribution
chord_length_PDF
cloud_area_PDF
```

---

#### 2.4 Validate Against Intercomparison
**References:**
- Siebesma et al. (2003) - BOMEX intercomparison results
- van Laar et al. (2019) - Cloud size statistics
- Griewank et al. (2020) - Chord length distributions

**Validation criteria:**

1. ✅ Cloud base, top within 10% of intercomparison mean
2. ✅ Cloud fraction within factor of 2
3. ✅ LWP within factor of 2
4. ✅ Vertical profiles show well-mixed BL
5. ✅ Sharp inversion maintained
6. ✅ Realistic cloud size distribution (power law?)

**If validation fails:** Debug systematically
- Check divergence after projection (should be ~0)
- Check mass conservation (∫ρ dV should be constant)
- Check energy conservation
- Check that forcings are applied correctly
- Compare to FastEddy BOMEX results

---

### **PHASE 3: Add Physics Systematically** (Week 4+)

**Only after BOMEX validates!**

#### 3.1 Precipitation (RICO Case)
- Enable Kessler warm-rain microphysics
- Fix sedimentation (don't double-count advection)
- Run RICO case (precipitating shallow cumulus)
- Validate against Rauber et al. (2007)

#### 3.2 Diurnal Cycle (ARM Case)
- Add time-varying surface fluxes
- Add solar radiation (diurnal heating)
- Run ARM SGP case
- Check morning/afternoon cloud evolution

#### 3.3 Heterogeneous Forcing
- Add surface flux heterogeneity (±20%, not ±40%)
- Larger correlation length (5 km, not 2 km)
- Study impact on cloud organization

#### 3.4 Cloud-Radiative Feedback
- Add longwave/shortwave radiation scheme
- Couple to cloud water (qc)
- Study cloud self-organization

#### 3.5 Terrain Effects
- Add topography
- Orographic forcing
- Valley/mountain flows

**Rule:** Add ONE new feature at a time, validate against baseline.

---

## Testing Strategy

### Unit Tests
```python
tests/
  test_grid.py           # Grid geometry, staggering
  test_projection.py     # Divergence removal
  test_advection.py      # Numerical schemes
  test_microphysics.py   # Saturation adjustment
  test_surface_layer.py  # M-O bulk formulae
```

**Key test: Divergence-free projection**
```python
def test_projection_divergence_free():
    # Random velocity field
    u = np.random.randn(nx+1, ny, nz)
    v = np.random.randn(nx, ny+1, nz)
    w = np.random.randn(nx, ny, nz+1)

    # Project to divergence-free
    u, v, w, p = project(u, v, w)

    # Compute divergence
    div = divergence(u, v, w)

    # Should be machine precision
    assert np.max(np.abs(div)) < 1e-12
```

### Integration Tests
- BOMEX validation (hours 3-6 statistics)
- Conservation tests (mass, energy)
- Grid convergence (50 m, 100 m, 200 m)
- Domain size sensitivity (3.2 km, 6.4 km, 12.8 km)

### Regression Tests
- Save BOMEX baseline results
- Check new commits don't break validation
- CI/CD with automated BOMEX runs

---

## Development Workflow

### Branch Strategy
```
main                     # Stable, validated code
├── dev/c-grid          # Phase 1.1
├── dev/rk3             # Phase 1.2
├── dev/weno5           # Phase 1.3
├── dev/theta-l-qt      # Phase 1.4
├── dev/surface-layer   # Phase 1.5
└── dev/bomex           # Phase 2
```

**Process:**
1. Create feature branch from `main`
2. Implement + test feature
3. Merge to `dev/integration` for combined testing
4. Only merge to `main` after full validation

### Documentation
For each phase, document:
- **What changed:** Code diffs, new functions
- **Why:** Physical/numerical motivation
- **Validation:** Test results, plots
- **Parameters:** New config options

---

## Timeline Estimate

| Phase | Task | Effort | Deliverable |
|-------|------|--------|-------------|
| **1.1** | C-grid staggering | 3-4 days | max\|∇·u\| < 1e-12 |
| **1.2** | RK3 time stepping | 1-2 days | Conservation tests pass |
| **1.3** | WENO-5 advection | 2-3 days | Reduced diffusion |
| **1.4** | θ_l/qt framework | 2-3 days | Exact water conservation |
| **1.5** | Surface layer | 2-3 days | Realistic surface stress |
| **2.1** | BOMEX setup | 1-2 days | Correct sounding/forcing |
| **2.2** | Strip physics | 1 day | Clean baseline |
| **2.3** | 6-hour run | 1 day | Output data |
| **2.4** | Validation | 2-3 days | Match intercomparison |
| **Total** | | **~3-4 weeks** | Validated LES core |

**Note:** This is aggressive. Real-world debugging may extend Phase 1 to 3-4 weeks.

---

## Success Criteria

### Phase 1 Complete When:
- [x] max|∇·u| < 1e-12 after projection
- [x] RK3 runs stably with CFL = 1
- [x] WENO-5 applied consistently to all fields
- [x] θ_l, qt conserved to machine precision
- [x] Surface stress from M-O theory (no wall BC)
- [x] All unit tests pass

### Phase 2 Complete When:
- [x] BOMEX cloud base: 500-600 m (±10%)
- [x] BOMEX cloud top: 1500-2000 m (±10%)
- [x] Cloud fraction: 2-4% (within factor 2)
- [x] LWP: 15-25 g/m² (within factor 2)
- [x] Vertical profiles match intercomparison
- [x] No artificial rotation/patchiness
- [x] Realistic cloud size distribution

### Phase 3 Complete When:
- [x] Each new physics validated separately
- [x] Combined physics shows expected behavior
- [x] Published results replicable

---

## References

**Core numerics:**
- Arakawa & Lamb (1977) - Computational design of the basic dynamical processes
- Durran (1999) - Numerical Methods for Fluid Dynamics
- Wicker & Skamarock (2002) - Time-splitting methods for elastic models

**LES methodology:**
- Deardorff (1980) - Stratocumulus-capped mixed layers
- Sullivan et al. (1994) - Large-eddy simulation of the convective boundary layer
- Stevens et al. (2005) - Evaluation of LES for shallow cumulus

**BOMEX case:**
- Siebesma et al. (2003) - A large eddy simulation intercomparison study of shallow cumulus convection
- Siebesma & Cuijpers (1995) - Evaluation of parametric assumptions for shallow cumulus convection

**Cloud microphysics:**
- Kessler (1969) - On the distribution and continuity of water substance
- Khairoutdinov & Randall (2003) - Cloud resolving modeling of the ARM SGP

**FastEddy:**
- Sauer & Muñoz-Esparza (2020) - The FastEddy GPU-Accelerated LES model
- FastEddy User Guide (NCAR/RAL)

**Cloud statistics:**
- van Laar et al. (2019) - Towards understanding the physics of cumulus cloud size
- Griewank et al. (2020) - Interpreting observations of cloud organization
- Lamaakel et al. (2022) - Importance of domain size for shallow cumulus LES

---

## Open Questions

1. **Anelastic vs Boussinesq?**
   - Boussinesq works for BOMEX (3 km)
   - Anelastic better for deeper domains (>5 km)
   - Decision: Stick with Boussinesq for now, revisit for deep convection

2. **SGS model?**
   - Smagorinsky is standard, well-tested
   - Could try 1.5-order TKE (DALES uses this)
   - Decision: Keep Smagorinsky, add Richardson correction

3. **Radiation?**
   - BOMEX uses prescribed -2 K/day
   - RICO and ARM need full radiation
   - Decision: Start with prescribed, add RRTM later (Phase 3)

4. **Microphysics?**
   - Kessler sufficient for warm-rain
   - Morrison 2-moment for ice?
   - Decision: Kessler for Phase 2-3, Morrison as future work

---

## Notes for Continuation

When resuming work:

1. **Start with Phase 1.1** (C-grid)
   - This is the most critical fix
   - Everything else builds on this
   - Test divergence removal obsessively

2. **Don't skip validation**
   - Unit tests for every component
   - Integration test after each phase
   - BOMEX is the benchmark

3. **Resist adding physics early**
   - It's tempting to add heterogeneity, cloud shading, etc.
   - But a clean baseline is essential
   - Physics comes in Phase 3

4. **Document everything**
   - Git commits with detailed messages
   - Markdown docs for each phase
   - Plots showing before/after

5. **Expect setbacks**
   - Debugging C-grid takes time
   - BOMEX may not validate first try
   - That's normal for LES development

---

## Contact / Support

**FastEddy resources:**
- GitHub: https://github.com/NCAR/FastEddy-model
- User guide: https://ral.ucar.edu/projects/fasteddymodel
- BOMEX case setup in examples/

**LES community:**
- DALES code: https://github.com/dalesteam/dales
- UWLCM code: https://github.com/igfuw/UWLCM
- MicroHH code: https://github.com/microhh/microhh

**Papers with open data:**
- Siebesma et al. (2003) BOMEX results: https://doi.org/10.1175/1520-0477
- ARM SGP case: https://www.arm.gov/capabilities/vaps

---

**End of Plan**

This is a major undertaking, but it's the right path forward. The foundation must be solid before we can study interesting cloud physics. Good luck! 🌥️
