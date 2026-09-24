# Nested Simulation Design Document

## Objective
Run a high-resolution nested simulation of a single cloud from the coarse-resolution domain (dx=100m) at much finer resolution (dx=10-25m), using the parent simulation to provide boundary conditions.

---

## Physical Considerations

### Scale Separation & Resolution Changes

**Parent (coarse) domain:**
- dx = dy = 100 m
- Domain: 12×12 km
- Resolves: Cloud-field scale organization, larger eddies
- SGS model: Smagorinsky (CS=0.18) handles ~100-300m eddies

**Nested (fine) domain:**
- dx = dy = 10-25 m (4-10× refinement)
- Domain: 2×2 km (subset around one cloud)
- Resolves: Individual thermals, cloud-top entrainment, fine-scale turbulence
- **SGS model needs adjustment!**

### What Changes at Finer Resolution?

#### 1. **Smagorinsky Constant (CRITICAL)**
**Current:** CS = 0.18 tuned for dx=100m

**Problem:**
- At dx=10m, filter scale Δ = (dx·dy·dz)^(1/3) is ~10× smaller
- K_M ~ (CS·Δ)² ~ 100× smaller
- Effective Reynolds number changes dramatically
- Too much CS → over-diffusion, kills fine-scale structures
- Too little CS → numerical instability

**Solution:**
- **Do NOT change CS** - it's a fundamental constant (~0.18-0.23 for atmospheric LES)
- The resolution change is correct: finer grid → less SGS diffusion → more explicit resolution
- At dx=10m, we're entering "eddy-resolving" regime where most turbulence is explicit
- May need to reduce CS slightly (0.15-0.17) if seeing numerical noise, or increase (0.20-0.22) if turbulence is too energetic

**Recommendation:** Start with CS=0.18 (same as parent), adjust only if needed based on TKE spectra or visual inspection.

#### 2. **Time Step (CFL Condition)**
**Current:** dt = 2.0s for dx=100m

**Problem:** CFL stability requires dt ∝ dx/u_max
- If dx → dx/10, need dt → dt/10 for same CFL number
- Assuming w_max ~ 15 m/s: CFL = w·dt/dz
- At dx=100m: dz~30-120m → CFL ~ 0.25-1.0 (stable)
- At dx=10m: dz~3-12m → need dt ~ 0.2-0.5s

**Solution:** Implement adaptive CFL limiter (already exists in simulation.py)
```python
dt = min(0.8 * dx / u_max, 0.8 * dy / v_max, 0.8 * min(dz) / w_max)
```

#### 3. **Advection Scheme**
**Current:**
- WENO-3 for qc, qr (horizontal)
- Upwind for u, v, theta, qv
- Upwind in vertical for all

**At finer resolution:**
- WENO-3 is appropriate, may want higher-order (WENO-5) for better accuracy
- Upwind for dynamical fields is still stabilizing
- **Consider:** Switch theta, qv to WENO-3 as well at fine scale (less implicit diffusion needed)

**Recommendation:** Keep current scheme initially, but monitor for excessive numerical diffusion.

#### 4. **Microphysics**
**Current:** Kessler warm rain (threshold-based)

**At finer resolution:**
- Autoconversion threshold (0.5 g/kg) is still valid
- But individual cloud droplets/parcels are better resolved
- Sedimentation velocity unchanged (physical, not resolution-dependent)

**No changes needed** - Kessler is scale-aware.

---

## Boundary Conditions from Parent Domain

### Current System (Periodic)
- x, y: Periodic (np.roll)
- z: Rigid lid (w=0 at top/bottom)

### Nested System (Open/Relaxation)

**Spatial structure:**
```
Parent (coarse):          Nested (fine):
┌─────────────────┐       ┌──────┐
│  12 km × 12 km  │       │ 2 km │ embedded inside parent
│  120×120 cells  │       │200cells (dx=10m)
│                 │       └──────┘
│     Cloud →  ◉  │
│                 │
└─────────────────┘
```

**Key issue:** Nested domain is NOT periodic - needs lateral boundary conditions!

### Option 1: Relaxation Zone (RECOMMENDED)

**Concept:** Blend nested solution → parent solution in outer rim

**Implementation:**
- Inner domain: free evolution (dx_inner = 10m)
- Outer relaxation zone: 5-10 cells wide (~50-100m)
- Nudging: φ_nest += (dt/τ_relax) · (φ_parent - φ_nest)
- τ_relax: short near boundary (~5-10s), infinite in interior

**Advantages:**
- Smooth, non-reflecting
- Allows waves/eddies to exit cleanly
- Standard in nested LES

**Relaxation profile:**
```python
def relaxation_timescale(i, j, nx, ny, n_relax=10):
    """
    Returns τ_relax(i,j) in seconds.
    n_relax: number of cells in relaxation zone
    """
    # Distance from boundary (in cells)
    dist_x = min(i, nx-1-i)
    dist_y = min(j, ny-1-j)
    dist = min(dist_x, dist_y)

    if dist >= n_relax:
        return np.inf  # No relaxation in interior
    else:
        # Exponential decay toward boundary
        # τ = 5s at boundary, → ∞ at n_relax
        tau_min = 5.0
        return tau_min / (1 - dist/n_relax)**2
```

**Apply to all fields:**
- u, v, w, theta, qv (dynamic fields from parent)
- qc, qr (cloud fields - allow to evolve freely initially, then relax if exiting domain)

### Option 2: One-Way Nesting (Simpler, Less Accurate)

**Concept:** Parent → Nested (one-way)
- Interpolate parent fields to fine grid at boundaries
- Simply prescribe boundary values, no feedback

**Problem:**
- Reflections at boundaries
- Not mass-conserving
- Works OK for short simulations (< 10 min)

**When to use:** Quick tests, proof of concept

### Option 3: Two-Way Nesting (Complex, Not Recommended for First Attempt)

**Concept:** Nested ↔ Parent (bidirectional feedback)
- Parent provides BC to nested
- Nested averages back to parent grid to update parent solution

**Problem:**
- Requires modifying parent simulation during nested run
- Complex synchronization
- Overkill for single-cloud study

---

## Large-Scale Forcing Adjustments

### Current Forcing (Domain-Mean Operations)

**1. Subsidence** (forcing.py:76-90)
- Current: `w_ls = -5 mm/s` above BL, advects reference profiles
- **Nested:** Still apply, but use parent's w_ls vertical profile
- If nested domain is small (2 km), subsidence is approximately uniform

**2. Profile Nudging** (forcing.py:93-106)
- Current: Nudges **domain-mean** θ, qv toward reference profiles
- **Problem in nested:** Domain-mean of 2×2 km is cloud-biased (we selected a cloudy region!)
- **Solution:**
  - **Option A:** Disable nudging (parent already maintains large-scale balance)
  - **Option B:** Nudge to parent's horizontally-averaged profiles (not nest's own mean)
  - **Recommendation:** Option B - use parent domain-mean as "reference environment"

```python
def apply_profile_nudging_nested(s_nest, parent_mean_theta, parent_mean_qv, dt):
    """
    Nudge nested domain toward parent's horizontal mean profiles.
    parent_mean_theta, parent_mean_qv: (nz_nest,) arrays from parent at same z-levels
    """
    nest_mean_theta = np.mean(s_nest.theta, axis=(0,1))
    nest_mean_qv = np.mean(s_nest.qv, axis=(0,1))

    # Relax nest's mean toward parent's mean (NOT toward zero!)
    s_nest.theta -= (dt / TAU_NUDGE_T) * (nest_mean_theta - parent_mean_theta)[None,None,:]
    s_nest.qv -= (dt / TAU_NUDGE_Q) * (nest_mean_qv - parent_mean_qv)[None,None,:]
```

**3. Surface Fluxes** (forcing.py:36-74)
- Current: Spatially heterogeneous SHF/LHF from correlation length ~2 km
- **Nested:**
  - Interpolate parent's SHF/LHF map to fine grid
  - Or regenerate at fine scale (but then it's different physics!)
  - **Recommendation:** Interpolate parent map (maintains consistency)

**4. Sponge Layer** (forcing.py:139-177)
- Current: Damps near top of domain
- **Nested:** Keep same z_sponge, same formulation (physical, not resolution-dependent)

**5. Cold Pool Parameterization** (forcing.py:109-137)
- Current: Instant evaporation of qr in lowest 300m
- **Nested:** Same physics, unchanged

---

## Implementation Strategy

### Step 1: Extract Parent Domain Snapshot

**Goal:** Get coarse-resolution fields at one timestep around target cloud

```python
def extract_parent_snapshot(snapshot_file, cloud_center, nest_size_km=2.0):
    """
    Extract subregion from parent snapshot for nesting.

    Parameters:
    -----------
    snapshot_file : Path
        Parent simulation snapshot (e.g., snap_0001200.npz)
    cloud_center : tuple (i, j, k)
        Grid indices of cloud center in parent domain
    nest_size_km : float
        Size of nested domain in km (default 2 km)

    Returns:
    --------
    parent_data : dict
        Subregion of all fields on parent grid
        Keys: u, v, w, theta, qv, qc, qr, z, x, y
    """
    # Load parent snapshot
    data = np.load(snapshot_file)

    # Extract region
    i_c, j_c = cloud_center[:2]
    n_cells = int(nest_size_km * 1000 / cfg.DX)  # cells in parent resolution

    i_start = max(0, i_c - n_cells//2)
    i_end = min(cfg.NX, i_c + n_cells//2)
    j_start = max(0, j_c - n_cells//2)
    j_end = min(cfg.NY, j_c + n_cells//2)

    return {
        'u': data['u'][i_start:i_end, j_start:j_end, :],
        'v': data['v'][i_start:i_end, j_start:j_end, :],
        'w': data['w'][i_start:i_end, j_start:j_end, :],
        'theta': data['theta'][i_start:i_end, j_start:j_end, :],
        'qv': data['qv'][i_start:i_end, j_start:j_end, :],
        'qc': data['qc'][i_start:i_end, j_start:j_end, :],
        'qr': data['qr'][i_start:i_end, j_start:j_end, :],
        'prof_theta': data['prof_theta'],
        'prof_qv': data['prof_qv'],
        't': data['t'],
    }
```

### Step 2: Interpolate to Fine Grid

**Method:** Trilinear interpolation in 3D

```python
from scipy.interpolate import RegularGridInterpolator

def interpolate_to_fine_grid(parent_data, dx_fine=10.0):
    """
    Interpolate parent fields to fine grid.

    Current parent: dx = 100m
    Target fine: dx = 10m (10× refinement)
    """
    # Parent grid coordinates
    nx_p, ny_p, nz_p = parent_data['u'].shape
    x_p = np.arange(nx_p) * 100.0  # parent dx = 100m
    y_p = np.arange(ny_p) * 100.0
    z_p = Grid().z  # same vertical grid initially

    # Fine grid coordinates
    nx_f = int(nx_p * 100.0 / dx_fine)
    ny_f = int(ny_p * 100.0 / dx_fine)
    x_f = np.arange(nx_f) * dx_fine
    y_f = np.arange(ny_f) * dx_fine
    z_f = z_p  # initially same vertical, can refine later

    # Interpolate each field
    fine_data = {}
    for field in ['u', 'v', 'w', 'theta', 'qv', 'qc', 'qr']:
        interp = RegularGridInterpolator(
            (x_p, y_p, z_p),
            parent_data[field],
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        # Create fine mesh
        xf_mesh, yf_mesh, zf_mesh = np.meshgrid(x_f, y_f, z_f, indexing='ij')
        points = np.stack([xf_mesh.ravel(), yf_mesh.ravel(), zf_mesh.ravel()], axis=1)

        # Interpolate
        fine_data[field] = interp(points).reshape(nx_f, ny_f, len(z_f))

    return fine_data
```

### Step 3: Boundary Condition Specification

**Create BC handler class:**

```python
class NestedBoundaryConditions:
    def __init__(self, parent_snapshot, n_relax=10, tau_min=5.0):
        """
        parent_snapshot: dict with coarse-resolution fields at boundaries
        n_relax: number of cells in relaxation zone
        tau_min: minimum relaxation timescale (at boundary) in seconds
        """
        self.parent = parent_snapshot
        self.n_relax = n_relax
        self.tau_min = tau_min

        # Precompute relaxation mask
        nx, ny = parent_snapshot['u'].shape[:2]
        self.tau_relax = self._compute_relaxation_mask(nx, ny)

    def _compute_relaxation_mask(self, nx, ny):
        """2D array of relaxation timescales."""
        tau = np.full((nx, ny), np.inf)

        for i in range(nx):
            for j in range(ny):
                dist_x = min(i, nx-1-i)
                dist_y = min(j, ny-1-j)
                dist = min(dist_x, dist_y)

                if dist < self.n_relax:
                    # Quadratic ramping from boundary
                    tau[i,j] = self.tau_min / max(0.01, (1 - dist/self.n_relax)**2)

        return tau

    def apply_relaxation(self, state_nest, dt):
        """
        Apply relaxation BC: nudge nest toward parent in relaxation zone.
        """
        # Extend tau to 3D
        tau_3d = self.tau_relax[:, :, np.newaxis]

        # Relaxation factor
        alpha = dt / tau_3d
        alpha = np.minimum(alpha, 0.5)  # limit for stability

        # Apply to all fields
        for field in ['u', 'v', 'theta', 'qv']:
            phi_nest = getattr(state_nest, field)
            phi_parent = self.parent[field]  # already interpolated to fine grid

            # Only apply where tau < inf (i.e., in relaxation zone)
            mask = np.isfinite(tau_3d)
            phi_nest[mask] += alpha[mask] * (phi_parent[mask] - phi_nest[mask])

        # For w (face-staggered), handle separately
        # For qc, qr: only relax if cloud material is leaving domain
        # (otherwise let clouds evolve freely in interior)
```

### Step 4: Modified Forcing Functions

**New file: `forcing_nested.py`**

```python
def apply_profile_nudging_nested(s_nest, parent_mean_profiles, g, dt):
    """
    Nudge toward parent domain-mean profiles (not nest's own mean).
    """
    parent_theta_mean = parent_mean_profiles['theta']  # (nz,)
    parent_qv_mean = parent_mean_profiles['qv']

    nest_theta_mean = np.mean(s_nest.theta, axis=(0,1))
    nest_qv_mean = np.mean(s_nest.qv, axis=(0,1))

    # Relax nest mean toward parent mean
    s_nest.theta -= (dt / cfg.TAU_NUDGE_T) * (nest_theta_mean - parent_theta_mean)[None,None,:]
    s_nest.qv -= (dt / cfg.TAU_NUDGE_Q) * (nest_qv_mean - parent_qv_mean)[None,None,:]


def apply_subsidence_nested(s_nest, parent_ref_profiles, g, dt):
    """
    Apply subsidence using parent's reference profiles (same as parent).
    """
    # Same implementation as parent, but using parent's reference
    z = g.z
    w_ls = -cfg.W_SUBS * np.minimum(z / cfg.Z_BL, 1.0)

    dtheta0_dz = np.gradient(parent_ref_profiles['theta'], z)
    dqv0_dz = np.gradient(parent_ref_profiles['qv'], z)

    s_nest.theta -= dt * w_ls[None,None,:] * dtheta0_dz[None,None,:]
    s_nest.qv -= dt * w_ls[None,None,:] * dqv0_dz[None,None,:]
```

### Step 5: New Config for Nested Run

**config_nested.py:**

```python
# Nested domain configuration
NX_NEST = 200          # 2 km / 10 m = 200 cells
NY_NEST = 200
NZ_NEST = 80           # same vertical levels as parent initially
DX_NEST = 10.0         # m - 10× finer than parent
DY_NEST = 10.0
Z_TOP = 5000.0         # same as parent

# Time
DT_NEST = 0.2          # s - smaller timestep for finer grid
T_END_NEST = 600.0     # s - run for 10 minutes
OUTPUT_EVERY = 10.0    # s - more frequent output

# SGS (START WITH SAME, ADJUST IF NEEDED)
CS = 0.18              # Same as parent initially
PR_T = 0.33

# Forcing - reference parent domain!
PARENT_SNAPSHOT = "experiments/phase1_qvsurf/exp08_qv0160/snapshots/snap_0001200.npz"
CLOUD_CENTER = (60, 60, 15)  # Identify cloud in parent snapshot first
USE_RELAXATION_BC = True
N_RELAX = 10           # cells in relaxation zone
TAU_RELAX_MIN = 5.0    # s

# Parent forcing parameters (inherit from parent)
W_SUBS = 0.005         # same as parent
TAU_NUDGE_T = 3600.0
TAU_NUDGE_Q = 3600.0

# Disable surface fluxes? (already in parent forcing)
# Option: turn off if you want pure cloud evolution
# Option: interpolate parent SHF/LHF map
USE_SURFACE_FLUXES = False  # clouds already triggered in parent
```

---

## Workflow

### 1. Identify Target Cloud

```bash
# Visualize parent snapshot to find interesting cloud
python3 visualize3d.py --snapshot experiments/.../snap_0001200.npz

# Identify cloud center (i, j, k) visually or via automated detection:
python3 find_cloud_centers.py --snapshot snap_0001200.npz --min-qc 1.0
```

### 2. Setup Nested Simulation

```bash
# Create nested initial conditions from parent
python3 setup_nested.py \
  --parent-snapshot experiments/phase1_qvsurf/exp08_qv0160/snapshots/snap_0001200.npz \
  --cloud-center 60,60,15 \
  --nest-size 2.0 \
  --dx-fine 10.0 \
  --output nested_ic.npz
```

### 3. Run Nested Simulation

```bash
# Run with nested BC module
python3 main_nested.py --config config_nested.py
```

### 4. Analyze & Compare

```python
# Compare parent vs nested resolution
python3 compare_parent_nested.py \
  --parent experiments/.../snap_0001200.npz \
  --nested nested_output/snap_0000600.npz
```

---

## Expected Differences at Fine Resolution

### What Should Be Similar:
- Overall cloud structure and lifecycle
- Condensation/evaporation rates (bulk)
- Domain-mean precipitation

### What Should Be Different:
- **Sharper cloud edges** (less numerical diffusion)
- **Finer turbulent eddies** explicitly resolved
- **Cloud-top entrainment** better captured (100m eddies → 10m resolution)
- **Updraft cores** more concentrated, higher peak w
- **Downdrafts** more organized
- **Cloud breakup** may differ (if parent cloud was marginal, fine-scale mixing might evaporate it faster)

---

## Potential Issues & Solutions

### Issue 1: Nested Cloud Dissipates Quickly
**Cause:** Fine-scale turbulent mixing evaporates cloud faster than parent
**Solution:**
- Check that parent mean profiles are being used for nudging
- Verify moisture BC from parent is adequate
- May indicate parent resolution was under-resolving entrainment (good finding!)

### Issue 2: Numerical Instability
**Cause:** dt too large, or CS too small
**Solution:**
- Reduce dt (CFL < 0.8)
- Check w_max isn't exploding
- Slightly increase CS if seeing grid-scale noise (0.18 → 0.20)

### Issue 3: Reflections at Boundaries
**Cause:** Relaxation zone too narrow or τ_relax too large
**Solution:**
- Increase n_relax (10 → 15 cells)
- Decrease τ_min (5s → 2s)

### Issue 4: Cloud Drifts Out of Domain
**Cause:** Mean wind in parent advects cloud
**Solution:**
- Track cloud center in parent simulation
- Either: move nested domain with cloud (Lagrangian nesting)
- Or: run shorter nested period (5-10 min max)

---

## Summary of Changes Required

### New Files:
1. `config_nested.py` - nested domain configuration
2. `forcing_nested.py` - modified forcing for nested domain
3. `boundary_nested.py` - relaxation BC implementation
4. `setup_nested.py` - extract parent data and interpolate to fine grid
5. `main_nested.py` - nested simulation driver
6. `find_cloud_centers.py` - automated cloud detection utility

### Modified Files:
- `grid.py` - accept different dx, dy from config
- `state.py` - initialize from interpolated parent data
- `simulation.py` - call nested BC update each timestep

### No Changes Needed:
- `microphysics.py` - scale-aware
- `pressure.py` - works at any resolution (FFT is scale-independent)
- `dynamics.py` - works at any resolution (just change dx, dy in config)

---

## Recommended First Test

**Conservative approach:**

1. **dx_nest = 25m** (4× refinement, not 10×)
   - Less extreme, easier to debug
   - dt ~ 0.5s (more manageable)

2. **Small domain: 1 km × 1 km**
   - Faster to run
   - Easier to visualize

3. **Short duration: 5 minutes**
   - See if cloud evolves reasonably
   - Check for BC issues

4. **Relaxation BC with wide zone**
   - n_relax = 15 cells
   - tau_min = 3s

5. **Same CS = 0.18**
   - Don't change two things at once

**Success criteria:**
- Cloud maintains structure for 5 min
- No obvious reflections at boundaries
- TKE spectrum shows expected -5/3 slope at resolved scales
- Visual inspection: finer turbulent eddies visible

**Then iterate:**
- If successful → refine to dx=10m
- Extend duration to 10-15 min
- Try different clouds (young vs mature)
- Quantify entrainment rates vs parent

---

## Scientific Questions This Enables

1. **Entrainment parameterization:** How much mixing occurs at cloud edge?
2. **Updraft structure:** Are parent updrafts really coherent or just numerical?
3. **Turbulence statistics:** Does parent resolution under-predict TKE?
4. **Microphysics:** Are we missing fine-scale rain processes?
5. **Cloud lifetime:** Does better-resolved turbulence change cloud evolution?

This is a classic "LES within LES" approach used in atmospheric science to bridge scales!
