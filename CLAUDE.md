# Cloud Simulation — Project Notes

## Goal
Simulate cumulus clouds on a laptop for a small domain, eventually driven by
real-world initial conditions (ERA5) and terrain (Alpine DEM), with surface
heterogeneity from the WegenerNet observational network.

---

## Original Plan (phases)

### Phase 1 — Flat domain, idealised forcing (current)
- [x] Grid + stretched vertical levels
- [x] Boussinesq dynamics (forward-Euler + RK upgrade later)
- [x] FFT pressure solver (2-D FFT in x-y, Thomas tridiagonal in z)
- [x] Kessler warm-rain microphysics
- [x] Surface heat/moisture flux forcing
- [x] SGS diffusion (constant eddy diffusivity)
- [x] Rayleigh sponge layer
- [x] 2-D cross-section visualisation (matplotlib)
- [x] 3-D volume rendering animation (PyVista, off-screen)
- [ ] Fix scattered cumulus (see open issue below)

### Phase 2 — Terrain
- [ ] Fetch Alpine DEM (Copernicus GLO-30 recommended)
- [ ] Crop/regrid to model domain
- [ ] Implement sigma (terrain-following) coordinates or immersed boundary
- [ ] Switch lateral BCs from periodic to open/nudging inflow

### Phase 3 — Real-world initial conditions
- [ ] Download ERA5 sounding for chosen date/location via cdsapi
- [ ] Replace warm-bubble IC with ERA5 T(z), q(z), u(z), v(z) profiles
- [ ] Wire in spatially-varying surface forcing from WegenerNet
- [ ] Time-varying lateral nudging from ERA5/COSMO

### Phase 4 — Refinements
- [ ] Diurnal cycle and radiation scheme
- [ ] Smagorinsky SGS turbulence (replace constant K)
- [ ] Two-moment microphysics
- [ ] 3-D PyVista viewer with interactive controls

---

## Code Structure

| File | Role |
|---|---|
| `config.py` | All parameters in one place |
| `grid.py` | Grid class: NX×NY×NZ, dx=100m, exponential z-stretching |
| `state.py` | State class: fields + reference atmosphere + initial conditions |
| `dynamics.py` | 1st-order upwind advection + constant-K SGS diffusion |
| `pressure.py` | FFT projection (2-D FFT in x-y, Thomas tridiagonal in z) |
| `microphysics.py` | Kessler: saturation adjustment, autoconversion, accretion, sedimentation |
| `forcing.py` | Spatially heterogeneous SHF/LHF, large-scale subsidence, profile nudging, sponge |
| `simulation.py` | Forward-Euler time loop with CFL limiter, snapshot saving |
| `visualize.py` | Live 4-panel matplotlib (cross-sections + mean profiles) |
| `visualize3d.py` | Off-screen PyVista volume rendering → GIF or MP4 |
| `main.py` | Entry point |

---

## Key Design Decisions

### Dynamics
- **Boussinesq equations** — valid for shallow convection (< 5 km)
- **w face-staggered in z** (shape nx×ny×(nz+1)); u, v, θ, qv, qc, qr at cell centres
- **Periodic BCs in x and y**; w = 0 at top and bottom (rigid lid)
- **Advection**: 1st-order upwind (stable with forward-Euler; centred/FTCS is unconditionally unstable)
- **Pressure solve**: Neumann BCs in z; (0,0) Fourier mode set to zero (pressure reference)

### Grid (current)
- Domain: 8 × 8 × 5 km
- Horizontal: 80 × 80 cells, dx = dy = 100 m
- Vertical: 50 levels, exponential stretching (beta = 1.5), dz = 44 m (surface) → 190 m (top)

### Microphysics
- Saturation adjustment: instant condensation with Newton correction for latent heat feedback
- Kessler warm-rain: autoconversion (threshold 0.5 g/kg), accretion, rain evaporation, sedimentation
- No ice, no radiation

### Surface forcing
- SHF and LHF maps generated once at init from a spatially correlated random field
  (Gaussian-filtered white noise, correlation length 2 km, ±40% variation around the mean)
- Represents fixed land-use heterogeneity driving individual thermals

### Large-scale forcing (periodic domain corrections)
- **Subsidence**: w_ls linear from 0 at surface to -5 mm/s at Z_BL, constant above
  Adds drying/warming tendencies: ∂φ/∂t -= w_ls · dφ₀/dz
- **Profile nudging**: removes domain-mean drift of θ′ and qv toward reference profiles
  τ_theta = 3600 s, τ_qv = 1800 s — prevents unlimited accumulation in periodic box

---

## Environment
- Python: `/home/simo/Documents/Python/Thesis/venv/bin/python3`
- Packages: numpy 2.3.4, scipy 1.16.3, matplotlib 3.10.7, pyvista 0.49.0, imageio 2.37.4, cdsapi 0.7.7
- CDS API key configured in `~/.cdsapirc` (for ERA5 downloads)
- Run: `cd /home/simo/Documents/Python/Clouds && <venv>/python3 main.py`
- Render animation: `<venv>/python3 visualize3d.py --out clouds.gif`

---

## Open Issue: Scattered Cumulus

**Problem**: the simulation produces 100% cloud cover (uniform slab) instead of
scattered individual cumulus towers (~20-40% cover).

**Root cause**: with periodic BCs and uniform-ish forcing, even moderate surface
heating eventually saturates the entire domain. The initial bubble (or noise)
triggers one dominant convective cell that fills the domain within ~10 minutes.

**What has been tried**:
- Single warm bubble → single cell fills domain
- Random BL perturbations (THETA_NOISE = 0.2 K) → still fills domain
- Spatially heterogeneous SHF (±40%) → still fills domain
- Large-scale subsidence (5 mm/s) → too weak to counteract LHF
- Profile nudging (τ = 1800-3600 s) → helps but not enough at QV_SURF = 0.014

**Fix applied (works)**:
Raised the LCL by drying the initial atmosphere:
- `QV_SURF = 0.011` kg/kg — LCL at ~1470 m, reachable only by hottest SHF patches
- `LHF = 5` W/m² (nearly no latent flux)
- `SHF = 150` W/m² with ±40% spatial variation (unchanged)

Result: cloud cover ~15–20% (range 5–36%), CB ~1470–1650 m, CT up to 3 km.
Scattered towers forming and dissipating. Light rain (qr_max ~0.1 g/kg). No slab.

Note: `QV_SURF = 0.008` was too dry (LCL ~2500 m → 0% cloud cover).

**Remaining issues after fix**:
- `w_max` grows to 20+ m/s by end of 2-hour run (convection not fully stabilised)
- Occasional spurious near-surface CB reports (66–300 m) late in run

---

## Real-World Data Sources

| Data | Source | Resolution | Notes |
|---|---|---|---|
| Atmospheric profiles | ERA5 (Copernicus CDS) | 0.25°, 37 levels, hourly | Free; needs cdsapi |
| Terrain DEM | Copernicus GLO-30 | 30 m | Best quality for Alps |
| Surface obs | WegenerNet (Feldbach, SE Styria) | ~150 stations, 5 min | For SHF heterogeneity + validation |

**Suggested first real case**: summer convective afternoon over SE Styria
(e.g. 2019-07-25 12:00 UTC, lat 46.9°N lon 15.9°E).
