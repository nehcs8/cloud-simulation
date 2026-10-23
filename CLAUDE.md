# Cloud Simulation — Reference

Anelastic LES of shallow cumulus (C-grid, RK3, WENO-3, θ_l/q_t, M-O surface layer), plus
nested high-resolution runs. Current focus: `single_cu`, one cumulus congestus (tops 4–7 km)
from a hot spot or a warm bubble. The earlier `land_cu` shallow-Cu field runs and how to
reproduce them: `docs/RUNNING_LAND_CU.md`.

---

## Where files go — read before creating anything

The repo root holds ONLY core model modules, entry points and config. Everything else
has a home. Do not leave logs, text output, images, data or scratch scripts in the root.

| You are creating…                                  | Put it in                                   |
|----------------------------------------------------|---------------------------------------------|
| Core model code (dynamics, physics, grid, …)       | root (`*.py` next to `simulation.py`)        |
| A new case (sounding, forcing, settings)           | `cases/<name>.py` (see `cases/land_cu.py`)   |
| A run entry point for a case                       | `run_case.py <case>` (cases list in `cases/__init__.py`) |
| A test or sanity/diagnostic check of the model     | `tests/test_<what>.py` (script-style, `__main__` block, `sys.path` header) |
| Analysis / post-processing / one-off tool worth keeping | `scripts/` (visualisation: `scripts/visualize/`) |
| Small generic helpers                              | `utils/`                                     |
| Simulation output (diag, snapshots, logs, budgets) | `runs/<timestamp>_<CASE_NAME>/` — automatic via `Simulation` |
| stdout of a background run                         | that run's folder, e.g. `> runs/<run>/runner.log`, or `outputs/logs/` |
| Output of a test/diagnostic script                 | `outputs/logs/<name>.txt`                    |
| Plots / figures                                    | `outputs/plots/`                             |
| Animations                                         | `cloud_gifs/YYYYMMDD_<description>.gif` (date = run date) |
| Nested-run initial conditions                      | `nested_runs/ics/`                           |
| Protected restart files                            | `restart_snapshots/` (immutable)             |
| Documentation, plans, progress notes               | `docs/`                                      |
| Superseded code / backups                          | `archive/` (never `*.backup` in the root)    |
| Throwaway experiments                              | inline `python - <<'EOF'` or the session scratchpad — never a new root file |

Rules:
- Before writing a new script, check `scripts/`, `tests/` and `utils/` for an existing one to extend.
- Never redirect output into the root (`> foo.log`, `> bar.txt`); use the destinations above.
- Scripts in `tests/` or `scripts/` import the model via
  `sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))`.
- Run everything from the repo root (`runs/`, `cloud_gifs/` etc. are relative to the cwd).
- Large data (`runs/`, `snapshots/`, `experiments/`, `*.gif`, `*.png`) is not for git.

---

## Environment

- Python: `./venv/bin/python3` (local venv); run from the repo root.
- Packages: numpy, scipy, numba, matplotlib, pyvista (VTK 9.7), imageio, pyyaml, cdsapi.
- i7-6700: 4 physical cores / 8 threads, NVIDIA GTX 1080 (unused by the model).
- Hot loops run as numba kernels (`kernels.py`, `cfg.USE_NUMBA = True`) on 4 threads — one per
  physical core; hyperthreads make them slower. Override with `NUMBA_NUM_THREADS`. The first run
  after editing `kernels.py` spends ~1 min compiling (cached in `__pycache__` afterwards).
- If VTK/pyvista fails with `X_GLXCreateContext BadValue` (e.g. NVIDIA driver/library version
  mismatch after an update, fixed by a reboot), render headless on the CPU with
  `VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow`.

---

## Code structure

### Core (root)
- `main.py`, `main_nested.py` — entry points (default case / nested runs)
- `run_case.py CASE[:VARIANT] [T_END_s] [KEY=VALUE ...] [TAG=name]` — runner for any case in
  `cases/`: variant presets, config overrides, process-by-process water budget, dt stats
- `run_land_cu.py …` — wrapper, identical to `run_case.py land_cu …` (kept for old commands)
- `run_smoke_test.py`, `run_bomex.py` — smoke test, BOMEX runner
- `config.py` — all parameters (defaults = "default" case); `config_nested.py` — nested runs
- `simulation.py` — time loop, adaptive dt (CFL), diagnostics, snapshots, run folders
- `state.py` — fields, reference atmosphere, u_g(z), case dispatch on `cfg.CASE` (`cases.get`)
- `grid.py` — horizontal uniform; vertical `GRID_TYPE` = `"exp"` or `"uniform_stretched"`
- `dynamics.py` — anelastic WENO-3 scalar advection, flux-form momentum, Smagorinsky, Coriolis, buoyancy
- `pressure.py` — anelastic spectral projection ∇·(ρ₀u)=0 (pass `rho0=state.rho0`); per direction
  `BC_X`/`BC_Y` = `"periodic"` (FFT) or `"open"` (DCT-II, boundary-face normal velocity kept, net boundary
  mass flux balanced to zero first)
- Terrain in `pressure.py` (`_project_terrain`, stage 1 of `docs/PLAN_terrain.md`): when `g.terrain` is set,
  preconditioned CG on the fluid cells (`TERRAIN_PCG_TOL` 1e-8), zero flow through solid faces; ~2.2× the
  step cost on 160². Stencils are terrain-aware (stage 2: ground fill + masked diffusion in `dynamics.py` /
  `kernels.py`); surface layer, sedimentation, initial noise/bubble (`Grid.z_agl()`) and the water budget
  follow the terrain surface (stage 3). `TERRAIN_PARTIAL_CELLS=True`: partial bottom cells (fluid fractions,
  exact ground height; off by default). Smagorinsky K is 0 in solid cells with one-sided
  du/dz at `k_sfc` (fixed 2026-10-19; before, a spurious K spike under the ground cut dt 3–6×). Mountain-wave
  drag: 25 m levels pass (1.05–1.19× theory); 12.5 m gives the same, so finer levels are not the fix; a 100 m
  ridge is still 1.2–1.6× (see docs/PLAN_terrain.md, stage 4). Run with `"TERRAIN_TYPE='gaussian_hill'" TERRAIN_H_MAX=500.0
  TERRAIN_SIGMA=1500.0` (hill at the domain centre)
- `halo.py` — halo (ghost-cell, H = 2) padding for all horizontal stencils: `pad_c`, `pad_u`, `pad_v`
  gather with one index map per direction (periodic: wrap; open: zero-gradient placeholder until stage 3
  of `docs/PLAN_open_boundaries.md`). `dynamics.py` / `kernels.py` read neighbours only from padded
  copies — no `np.roll` / `% n` left in the stencils. Periodic results are bitwise identical to before
  (checked 2026-10-17 on tendencies and a 10-min congestus run). `"open"` runs (stable, 30-min test)
  with the conditions of `open_bc.py`
- `open_bc.py` — open lateral BCs (stage 3): inflow faces put the sounding (θ_l′ = 0, q_t = qv0, q_r = 0) in the
  scalar ghosts, outflow faces keep zero-gradient ghosts; boundary-face normal velocity by Klemp–Wilhelmson
  radiation (u_n ± c*, `OPEN_C_STAR` = 10 m/s) + weak relaxation to u_g (`OPEN_TAU_NORMAL`); Davies inflow
  relaxation zone (`OPEN_RELAX_WIDTH` 4 cells, `OPEN_RELAX_TAU` 60 s); dt also limited by c*; water crossing
  the boundaries → `s.boundary_water_out` and the `boundary_outflow_pct` column of water_budget.csv
  (budget residual exactly 0 in an open/open congestus run). Run with `"BC_X='open'" "BC_Y='open'"`
- `kernels.py` — numba versions of WENO-3, momentum advection, SGS diffusion/K_M, saturation
  adjustment, Kessler microphysics, tridiagonal pressure solve. The numpy originals stay in
  `dynamics.py` / `thermodynamics.py` / `microphysics.py` / `pressure.py` as the reference
  (`USE_NUMBA = False`); keep both paths in sync and rerun `tests/test_kernels.py` after changes
- `timestepping.py` — RK3 + physics chain
- `thermodynamics.py` — θ_l/q_t conversions, q_sat (Buck). `ICE_SCHEME="mixed"` (2026-10-22, off by default):
  cloud condensate q_c is liquid with fraction α(T) = (T − `T_ICE_00`)/(`T_ICE_0` − `T_ICE_00`) clipped to [0, 1]
  (0 °C / −20 °C) and ice otherwise; saturation over the mix (Buck e_w, e_i), latent heat L_v + (1 − α)L_f, so
  θ_l is the ice-liquid potential temperature (`diagnose_cloud_water_mixed`: safeguarded Newton on T; numba twins
  `kernels.diagnose_cloud_water_mixed` / `theta_l_mixed`). q_c stays one field (ice = (1 − α(T))·q_c, recover it
  from θ′ in the snapshots). Rain stays liquid: cloud → rain conversion of the ice part takes L_f back
  (`warm_rain`), so there is no precipitating ice yet (step 2: graupel/snow + melting). Rain also evaporates
  inside icy cloud (q_v = q_s,mix < q_sw), the vapour is redeposited as cloud ice — a crude Bergeron path.
  Reversible parcel (`tests/test_ice.py`): +0.6 K at −7 °C, +1.4 K at −12 °C, +2.5 K at −17 °C vs warm
- `microphysics.py` — warm rain: `RAIN_SCHEME` "sb2001" (Seifert–Beheng 2001 autoconversion/accretion,
  droplet number `CLOUD_NC`, default 300 cm⁻³) or "kessler" (before 2026-10-19); rain evaporation `RAIN_EVAP`
  "kw78" (Klemp–Wilhelmson 1978) or "legacy"; implicit Kessler-fall-speed sedimentation.
  `ICE_SCHEME="graupel"` (2026-10-22, step 2): the "mixed" thermodynamics + prognostic graupel/snow `s.qg` (advected,
  diffused, in q_t, buoyancy loading, open-BC ghosts = 0, sponge, fringe, water budget, snapshots `qg`, diag `qg_max`;
  only transported when on, so warm runs are bitwise unchanged — checked on a 25-min bubble run). `ice_graupel`
  (numba twin `kernels.ice_graupel`): cloud-ice autoconversion (Lin 1983, `ICE_AUTO_QI0`), collection of cloud ice
  (`GR_E_ICE`), cloud liquid (riming; shed as rain above 0 °C) and rain by graupel, Bigg rain freezing, melting (never
  below 0 °C in a step), sublimation. Marshall–Palmer graupel `GR_N0` 4e6 m⁻⁴, `GR_RHO` 400 kg/m³, V = 19.3 D^0.37 →
  1.7–2.4 m/s mass-weighted; it sediments with the rain scheme's implicit sweep (`_sediment`), and surface precip =
  rain + graupel. Heating Δθ = [L_v(−Δq_v) + L_f Δ(ice)]/(c_p π), ice = (1 − α)q_c + q_g, so the T-diagnosed
  liquid/ice split stays energy-consistent (the step-1 "melt" term is the same relabelling). Warm rain uses α q_c only
- `surface_layer.py` — M-O surface layer, optional cloud shading
- `lateral_bc.py` — `LATERAL_BC = "fringe"`: inflow/outflow in x on the periodic grid. The last
  FRINGE_WIDTH is relaxed toward a copy of the strip upstream of RECYCLE_X ([0, RECYCLE_X) = homogeneous
  fetch acting as its own precursor); flux patterns are confined to the test region in between
  (`hetero_window`). Needs mean wind along +x. Default `"periodic"` is unchanged
- `forcing.py` — flux maps (+ warm stripe `SHF_STRIPE_*`, hot spot `SHF_SPOT_*`) / evolving flux
  pattern, subsidence, nudging, radiative cooling, sponge
- `forcing_nested.py`, `boundary_conditions.py`, `terrain.py`

### Cases (`cases/`)
A case module defines `configure()`, `reference_profiles(z)`, `exner_profile(θ0, z)`,
`init_perturbations(s, g)`, `print_rh_profile(s, g)` and optionally `VARIANTS` (dicts of config
presets). Register it in `cases.CASES`; `Simulation`/`State` dispatch via `cases.get(cfg.CASE)`.
`configure()` runs once (guard flag); the runner applies variant, then overrides.
- `single_cu.py` — congestus: no inversion, θ 300 K mixed layer to 600 m, then `SCU_LAPSE` dθ/dz
  3.5 K/km to 1.2 km, 2.0 to 3 km, 4.5 to the tropopause at 9.5 km (15 K/km above), RH 0.75 → 0.50 at
  5 km; parcel CAPE ≈ 1220 J/kg, LCL 1175 m, LFC 1575 m, EL 9.3 km, freezing level 3.55 km
  (until 2026-10-15: uniform 3.5 K/km, LFC 1.9 km — `"SCU_LAPSE=((9500.0, 3.5e-3),)"`). 128²/100 m, 50 m levels to 7 km stretched to
  12 km (160 levels), sponge 9.5 km, U 1.5 m/s uniform, no subsidence/nudging/radiation, dt ≤ 3 s,
  background SHF 50 / LHF 100 W/m². Variants: `hotspot` (default; +300 SHF +300 LHF W/m² disc, 3 km, centre)
  and `bubble` (+2 K, RH 0.85, centre 500 m, 3 km × 1 km, no hot spot).
- `land_cu.py` — continental Cu sounding + `configure()` (64² at 100 m, fluxes, sheared u_g,
  forcing, 40 m grid to 3 km, sponge 3.2 km, dt cap 5 s).
- `bomex.py` — BOMEX profiles and forcings (used by `run_bomex.py`)

### Tests (`tests/`)
Unit tests (projection, advection, RK3, θ_l/q_t, surface layer, land_cu / single_cu sounding) and
physics sanity checks (dry warm bubble, water budget/conservation, sedimentation, stratification).
Run as scripts: `./venv/bin/python3 tests/test_projection.py`.
`tests/test_fringe.py` checks the fringe copy/weights and windowed flux maps.
`tests/test_halo.py` checks the halo stencils with open BCs (numba = numpy, uniform scalar unchanged,
flux-form conservation = boundary flux, interior identical to periodic away from the edges).
`tests/test_open_bc.py` checks inflow/outflow ghosts, radiation, relaxation and the boundary water flux.
`tests/test_terrain_physics.py` checks surface fluxes, stress and sedimentation on the terrain.
`tests/test_terrain_stencils.py` checks the terrain-aware stencils (conservation, walls, numba = numpy).
`tests/test_pressure_terrain.py` checks the terrain projection (divergence, no flow through solids, CG count).
`tests/test_pressure_open.py` checks the open-boundary projection (residual, flux balance, boundary
velocities kept, idempotence, uniform through-flow) on numba and numpy paths.
`tests/test_single_cu.py` checks the single_cu state and prints parcel LCL/LFC/EL/CAPE (target
500–1500 J/kg); run it after any sounding change.
`tests/test_ice.py` checks the mixed-phase thermodynamics (warm limit, numba = numpy, round trip, q_v = q_s over the
mix, L_f of the ice → rain conversion, parcel warming vs warm).
`tests/test_kernels.py [snap.npz]` checks numba vs numpy (agreement to round-off) and times both.
`test_advection.py` and `test_cgrid_end_to_end.py` are stale (they expect a `theta` tendency key
from before the θ_l/qt switch).

### Scripts (`scripts/`)
- `analyze_land_cu.py RUN_DIR [t0 t1]` — Task-10 metrics (cloud frac, base/top, w_max, precip, persistence)
- `analyze_fringe.py RUN_DIR [t0 t1] [--shf-map]` — stats per x-region (fetch/test/fringe) and along x;
  `--shf-map` correlates cloud frequency with the SHF patches (uses the snapshots in the window)
- `analyze_fringe_paired.py RUN_DIR [t0 t1]` — fringe runs with patches: patch effect = test region minus
  its upstream twin (x − RECYCLE_X), along x and as lagged correlation with the SHF map (+ null from
  unrelated maps). Caches 2-D fields in `RUN_DIR/reduced_2d.npz` (survives snapshot thinning)
- `compare_microphysics.py RUN_DIR [RUN_DIR …] [--t0 60] [--every 15]` — warm/ice runs side by side: rain timing, peaks,
  surface total; per snapshot cloud liquid/ice (α(T) split), rain, graupel masses [kt], graupel height, export of rain + graupel
  through the edges (estimate), cover above 5 km, cold pool, local surface rain
- `visualize/render_clouds.py` — fast lit iso-surface renderer → GIF (see Visualize)
- `visualize/visualize3d.py` — older volume renderer (slow without a GPU)
- `find_cloud.py`, `track_cloud.py`, `setup_nested.py`, `archive_current_nested.py`,
  `run_experiment.py`, `run_manager.py`, `debug_theta_l.py`, parameter-sweep scripts

### Data folders
- `runs/<YYYYMMDD_HHMM>_<CASE_NAME>/` — one folder per run: `diag.csv`, `console.log`,
  `config.py` copy, `effective_config.txt` (values after overrides), `profiles.npz`,
  `snapshots/snap_*.npz`, plus `water_budget.csv` / `runner.log` from `run_case.py`
  **Thinned 2026-10-21:** every run up to `real_flat`/`real_hill` keeps only snapshots at multiples of 15 min
  (t % 900 = 0); diag.csv (60 s) is complete. GIFs re-rendered from these runs have 15-min frames.
- `experiments/` — older parent-run experiment series (Sept 2026); thinned 2026-10-21 to snapshots at multiples of 15 min plus each
  run's last snapshot (`final_snapshot.npz`, diag.csv and logs kept)
- `nested_runs/` — nested runs (YAML metadata); `nested_runs/ics/` — nested initial conditions
- `restart_snapshots/` — protected restart files (copy, never modify)
- `snapshots/` — legacy ephemeral snapshot dir from older `main.py` runs (current runs write into `runs/`)
- `outputs/logs/`, `outputs/plots/` — archived logs and figures
- `cloud_gifs/` — animations; `docs/` — plans, progress, experiment logs; `archive/` — old material

---

## Common workflows

### single_cu (congestus) run
```bash
./venv/bin/python3 tests/test_single_cu.py                       # sounding + CAPE check
MPLBACKEND=Agg ./venv/bin/python3 run_case.py single_cu:hotspot 3600 NX=64 NY=64 TAG=tune  # quick test
MPLBACKEND=Agg ./venv/bin/python3 run_case.py single_cu:bubble 14400 TAG=main   # main run, 128², 4 h
```
Folders: `runs/<ts>_single_cu_<variant>_<min>min[_<tag>]/`. Always name the variant (the bare case
is the hot spot, but its folder name then lacks `_hotspot`). Tuning knobs: `SHF_SPOT_AMP/_DIAM`,
`BUBBLE_DTHETA/_RH/_Z0/_RX/_RZ`, `SHF`, `LHF`, `U_GEO`; sounding constants at the top of
`cases/single_cu.py` (`SCU_LAPSE`, `SCU_RH_CL`). Warm rain only unless `"ICE_SCHEME='mixed'"` (cloud ice, no
precipitating ice yet; all runs before 2026-10-22 are warm). Measured cost at 128² (160 levels): 4 threads alone 2.2 s/step, 43 wall-min
per simulated hour; two runs side by side on 2 threads each 3.1–3.2 s/step, 66–69 wall-min/h. Mean dt
2.7–2.9 s. Snapshots ~45 MB each (60 s) → ~11 GB per 4 h run.
64² tuning 2026-10-15 (2 threads each, ~18 wall-min per simulated hour, mean dt 2.7–3 s):
bubble RH 0.95 → w_max 24 m/s, tops 8.4 km (too deep); RH 0.85 → w_max 18 m/s, top 6.8 km at 21 min,
rain ≤ 0.5 mm/h, cloud gone by 35 min. Hot spot (+300 W/m², U 1.5 m/s): only small Cu at 1.3–1.4 km
in the first hour (spot air just +0.3 K warmer: advected away in ~30 min, plus 30 min ramp).
Hot spot never reached congestus on 64² (tops ≤ 2.6 km, no rain): +300 for 3 h (tops 1.6–2.3 km by
2–3 h), +500 W/m² (2.6 km at 2 h), U 0.5 m/s (2.3 km), +300 SHF +300 LHF `SHF_SPOT_LHF_AMP` (2.1 km).
Likely cause: narrow thermals (few cells at 100 m) entrain RH-0.7 air above the LCL and stall below
the LFC (~1.9 km). The wide bubble survives.
128² main runs 2026-10-15: bubble (2 threads, shared CPU, 62 wall-min/h) top 7.0 km at 20 min, w 18 m/s,
gone by 30 min, clear 1–3 h, 8.0 GB; moist hot spot (4 threads, 43 wall-min/h) tops ≤ 2.7 km, no rain,
7.7 GB, so domain size was not the cause. `SCU_RH_CL` 0.80/0.85 (RH held to 3 km, CAPE unchanged)
did not help either on 64² (tops ≤ 2.3 km): the barrier is the stable layer between the LCL
(1.2 km) and the LFC (~2 km), not the dryness aloft.
New sounding 2026-10-16 (`SCU_LAPSE`, 2.0 K/km at 1.2–3 km): 64² hot spot reaches 4.3 km at 1 h 45 min,
5.5 km by 2.5 h, w 16–20 m/s, light rain; clouds sit 1.5–2 km downwind of the spot centre, background
clear for 3 h. Bubble: top 7.3 km at ~20 min, w 28 m/s, rain 1.7 mm/h, gone by 45 min.
128² × 4 h main runs 2026-10-16 (`runs/20261016_1319_single_cu_{hotspot,bubble}_240min_main/`, 11 GB each,
GIFs `cloud_gifs/20261016_single_cu_*_lapse_*.gif`): hot spot — first Cu at ~20 min, tops 3.2 km at 1 h,
4.5–4.8 km at 2–2.5 h, 5.6–6.2 km from 3 h, w up to 24 m/s, rain ≤ 0.03 mm/h (domain mean), cover ≤ 1.6 %;
all cloud within 3 km of the spot (centroid ~2 km downwind), background clear for 4 h. Bubble — top 7.5 km
at ≤ 20 min, w 29 m/s, rain ≤ 0.5 mm/h, gone by 40 min, clear 40 min–4 h.
**`congestus` variant (2026-10-17) — the first proper congestus.** Open lateral boundaries since 2026-10-19
(`BC_X/BC_Y='open'` in the variant; all earlier congestus runs are periodic unless tagged `_open`). Tropical sounding (θ 303 K ML to 900 m,
qv 15.2 g/kg, RH 0.85 to 3 km / 0.70 at 5 km, 2 K/km at 1.3–3 km, 6.5 K/km above + 2 K cap at 5.3–5.8 km,
CAPE 630, EL 6.8 km), 5 km moist hot spot + 1.5 K bubble, u 0 → 4 m/s over 0–5 km, 160². First try (4.5 K/km
above 3 km, +1.5 K cap, CAPE 1790) went to 8.4–9.3 km: the cap must sit where parcel buoyancy is already small.
Main run `runs/20261017_1511_single_cu_congestus_180min_main/` (4 threads, mean dt 1.8 s, 109 wall-min per
simulated hour, 13 GB; GIF `cloud_gifs/20261017_single_cu_congestus_0-180min.gif`): one broad tower (3–4 km)
over the spot for 3 h, tops 6–7.4 km, w 18–32 m/s, surface rain to ~2 mm/h (domain mean), new surges every
30–40 min; from ~75 min a detrainment shelf at 4.4–5.7 km spreads under the cap across the domain (that is the
cloud "beyond 6 km", not background convection — background BL stays clear). Weak cold pools (rain-shaft air
only ~0.2 K cooler): rain falls on the still-heated spot; `ENABLE_CLOUD_SHADING=True` is the obvious next test.
**Same run with open lateral BCs (2026-10-18)** `runs/20261018_0143_single_cu_congestus_180min_open/` (13 GB,
GIF `cloud_gifs/20261018_single_cu_congestus_open_0-180min.gif`; 4 threads, mean dt 1.5 s, 130 wall-min/h):
far-field low-level warming +0.16 K at 3 h vs +0.65 K periodic (recirculated subsidence gone); the tower
stays stronger (tops 7.2–8.0 km, cover to 39 %, rain to 3.8 mm/h vs 6.3–7.4 km / 28 % / 1.1 mm/h) because
the environment keeps the sounding instead of warming; net water import through the boundaries +13 % of the
domain water in 3 h, budget residual 0. The shelf leaves through the east (outflow) edge — cloud along it,
west edge clear; no boundary-forced updrafts (rms w within 1 km of the edge ≤ interior).
**Fragmented cloud base (2026-10-18).** The tower is one 3-D body (94–99 % of cloudy cells connected), but its
base is many narrow feeder columns (11–17 pieces < 500 m at 1.4–1.8 km in the 160² open run) fed by 70–80
separate sub-cloud updraft cores ~2 grid cells wide. 96² × 2 h tests (pieces 150 m above base / cores at 800 m,
60–120 min): disc spot 1–2 / 18–45; `SHF_SPOT_SHAPE='gaussian'` 2–4 / 40–86; Gaussian + 1000 m mixed layer at
RH 0.90 8–9 / 44–89 — reshaping the surface heating doesn't organise the inflow. A heated 400 m hill under the
spot (2026-10-19, docs/PLAN_terrain.md stage 4) doesn't organise it either (cores 40–81 vs 40–86 flat; calm: more
fragmented). Remaining candidates: 50 m resolution, cold pools.
**New warm rain + open BCs (2026-10-19/20, 96², 2 h, Seifert–Beheng Nc 300 cm⁻³ + KW78 evaporation).** Runs
`runs/*_tune96_{gauss_sb,hillcalm_sb}` (periodic) and `*_tune96_{gauss_sb_open,hillcalm_sb_open}`; GIFs
`cloud_gifs/20261019_single_cu_congestus96_*_sb*_0-120min.gif`. Rain aloft (qr > 0.1 g/kg) at 12 min when the tower
is already ~5.8 km (Kessler: 6–7 min at 2.1–2.4 km), at the ground from 27 min (Kessler 16–18), peak 7.3–7.4 mm/h
(domain mean; locally up to 162 mm/h) at 90–100 min. Real cold pools for the first time: lowest-level θ′ down to −1.1 (calm hill) / −2.2 K
(flat), domain mean −0.1 / −0.9 K at 2 h (Kessler runs never went below 0); the gust front ends the cell — updraft
cores at 800 m AGL fall from 50–110 to 1–7 by 120 min, outflow u_r +2…+4.5 m/s. Flat open at 90 min: inflow −3 m/s
inside 1.5 km and the largest base piece 3.9 km — the most organised base so far (gust-front convergence).
Periodic runs trap the 5–6 km detrainment sheet (high cover 58 % flat); open with shear exports it (12 %); open and
calm has no wind to export it, so it fills the 9.6 km domain (high cover 92 %, low 2 %) — a domain-size effect.
No boundary artefacts (rms w within 1 km of the edges 0.2–0.6 vs interior 0.6–1.3 m/s), budget residual 0.
Open issues: mature-stage tops 8.2–8.7 km overshoot the EL (6.8 km, tuned under Kessler) for 6–19 min; qr aloft
up to 11–13 g/kg in the 30 m/s cores.
**Retune and realism tests (2026-10-20, 72² open, Seifert–Beheng).** `precip_mmhr` in diag.csv is the DOMAIN MEAN; local
surface rain (ρ q_r V_t at the lowest level) is 15–20× larger. Sounding retune (`runs/*_snd72_*`): stronger cap + more
stable 1.3–3 km (`l3c4`) keeps tops ≤ 6.9 km. Trigger tests (`runs/*_trig_*`): without the bubble the hot spot alone
makes the congestus (first cloud 30 min, rain aloft 59 min); bubble and hot-spot strength only change timing — all
runs reach w_max 25–29 m/s, q_r aloft 5–8 g/kg, local rain 75–91 mm/h. Styria-like sounding (`runs/*_styria_c14`,
`_c18`: θ 302 K, q_v 11.6 g/kg, mixed layer 1.2 km, LCL 1.7 km, 0 °C at 3.8 km, −10 °C at 5.5 km, CAPE ~320 J/kg,
4.5 K/km above 3 km, stable layer 5.5–6 km; no bubble): rain aloft 68 min, at the ground 89–101 min, local rain
45–47 mm/h (half), q_r aloft 5.8–6.6 g/kg, tops 6.8–7.0 km, but w_max still ~25 m/s (realistic congestus 5–15). Cap
+5 K (`c14`) ≈ +7 K (`c18`). Remaining suspect: 100 m resolution → `styria_c14_dx50` (144² at 50 m). Diagnostics:
`scripts/analyze_updraft_dilution.py` (undiluted fraction of cores from q_t; > 1 at 2 km because hot-spot air is
moister than the reference surface air — use for comparison only) and local surface rain.
**50 m test (2026-10-20, `runs/*_styria_c14_dx50`, 144² at 50 m, 4 threads, 92 wall-min per simulated hour):** resolution
is not the cause — w_max 28.3 vs 24.5 m/s at 100 m, w99 8.3 vs 6.8, q_c 7.1 vs 6.4, q_r 5.5 vs 5.8 g/kg, local rain
51 vs 47 mm/h, same dilution (~0.8 at 3 km, ~0.6 at 4 km), same base fragmentation (2–5 pieces); faster life cycle
(peak 70–80 min, gone by 110). Hot-spot air is only +0.4–0.6 K / +0.2 g/kg (CAPE 440–480 J/kg, √(2·CAPE) 30–31 m/s), so
updrafts reach 80–90 % of parcel theory in every variant (real congestus ~50 %). Open suspects: one wide, sustained
plume over a 5 km spot (SHF 350 + LHF 400 W/m² there exceeds midday net radiation), and the SGS/numerics.
**Uniform "real" forcing (2026-10-21, `runs/*_real_{flat,hill}`, 128² at 100 m, open, U 3 m/s, Styria sounding, SHF 180 /
LHF 280 W/m² everywhere, no spot/bubble; hill = 400 m Gaussian, σ 1.5 km, at the centre; GIFs
`cloud_gifs/20261021_single_cu_real_{flat,hill400}_128sq_0-180min.gif`):** no congestus — a field of small to medium Cu, first
cloud 43 (flat) / 37 (hill) min, cover 1–4 %, base 1675 m, tops mostly 2–3 km (max 3.3 / 4.2 km), w_max 6–8 (hill up to 10) m/s,
q_c ≤ 2.3 g/kg, no surface rain (q_r aloft ≤ 0.14 g/kg). **Fetch artefact:** inflow is relaxed to the unheated initial sounding,
so the mixed layer warms along x (θ′ below 600 m 0.06 K at the inflow edge, 0.36–0.38 K at the outflow edge) and every cloud is in
the downstream half (x > 6.4 km). The hill doesn't trigger cloud over its summit; it adds a cloud street in its lee (cloud
frequency in the y-band through the hill at the last 1.6 km: 20 % vs 8 % flat, off-band ~10 % in both). Domain-mean θ_bl
stalls at +0.22 K after 75 min because the BL air is replaced every ~70 min. Real days need an inflow BL that is heated too
(time-varying boundary state, or a recycled/precursor inflow).
**Ice step 1 (2026-10-22, `runs/*_styria_c14_ice`, = `styria_c14` + `"ICE_SCHEME='mixed'"`, 72², 2 h, 44 wall-min on 4
threads; log `outputs/logs/styria_c14_ice.runner.log`):** small effect, as expected for a congestus whose coldest cloud is
−17 °C (ice ≤ 34 % of the condensate). Tops +100–400 m (max 7.33 vs 7.03 km), mean θ_v′ of w > 5 m/s cores at 4.5–6 km
+0.3–0.4 K at 90–120 min vs −0.9…+0.5 warm, cloud cover above 5 km lasts longer (16.6 vs 7.1 % at 120 min), w_max unchanged
(25.8 vs 24.5 m/s), surface rain 12 min later (101 vs 89 min), same total (0.126 vs 0.133 mm domain mean), local peak 31 vs
37 mm/h. Cloud ice cannot fall yet, so it piles up aloft → step 2 (precipitating ice + melting).
**Ice step 2 (2026-10-22, `runs/*_styria_c14_graupel`, same setup + `"ICE_SCHEME='graupel'"`, 46 wall-min, budget residual 0;
log `outputs/logs/styria_c14_graupel.runner.log`):** graupel from 60 min, up to 3.6 g/kg and 10.9 kt at 100–105 min (vs rain
9.8 kt), centred at ~4.9 km; rain aloft 65 min, at the ground 93 min (warm 89, step 1 101); peak local rain 29 mm/h, cold pool
−2.6 K (warm −2.4); tops 7.24 km, w_max 26 m/s (forcing, not microphysics). Total surface precip 0.089 mm vs 0.133 warm, because
**most graupel leaves through the open lateral boundaries** with the 5 km outflow (estimated from the snapshots: ~6 kt per 5 min
at 105–110 min, i.e. the whole drop 10.9 → 4.8 kt; mean graupel height stays at 4.9 km while it vanishes). The 7.2 km domain is
too small for a 5 km plume plus its anvil: judge precipitation of ice runs on ≥ 128² (12.8 km).
**128² warm vs graupel (2026-10-22, `runs/*_styria128_{warm,graupel}`, same Styria setup on 12.8 km, 2 threads each side by side,
109/116 wall-min per simulated hour, 1.1 GB each; `scripts/compare_microphysics.py`, output `outputs/logs/compare_styria128.txt`):**
graupel 3.2 g/kg max, 26 kt at 110 min (> rain 20 kt), centred at 4.3–4.9 km; ground rain from 93 min (warm 99), tops 7.8 vs
7.5 km, w_max 28 vs 27 m/s; local rain 23 vs 38 mm/h, cold pool −3.0 vs −3.8 K; surface total at 2 h 0.078 vs 0.108 mm, but
**both cells are still raining at 2 h** (warm peaks at 119 min), so totals are truncated. Graupel again leaves the domain — all
through the downwind (east) edge, carried by the 4 m/s wind at 4–5 km (export 4 → 17 t/s from 100 to 120 min; 39 % of the
graupel within 1 km of the edges at 120 min): slow (1.7–2.4 m/s) graupel/snow drifts downstream in the anvil for > 25 min
before melting. With the spot at the centre there are only 6.4 km of fetch downwind: put the spot upwind (`SHF_SPOT_X0`) and run
≥ 3 h to close the precipitation budget.
**Upwind spot, 3 h (2026-10-22, `runs/*_styria128up_{warm,graupel}`, 128², `SHF_SPOT_X0=3500.0` → 9.3 km downwind, 2 threads
each side by side, 102/111 wall-min per simulated hour, budget residual 0; `outputs/logs/compare_styria128up.txt`):** the first
cell now rains out inside the domain (no surface rain after ~130 min); a second cell starts at ~150 min in both runs (cut off at
3 h). Cell 1 (0–150 min): ground rain from 90 min in both; peak local rain 54 (warm) vs 32 mm/h (graupel), cold pool −3.2 vs
−3.1 K, tops 7.1 vs 7.7 km, w_max 25 in both. Rain + graupel aloft peak 33 kt (warm) vs 46 kt (24 rain + 23 graupel) at 105 min,
but surface precipitation 23.9 kt (0.146 mm) vs 14.1 kt (0.086 mm), plus ≈ 4.8 kt of rain + graupel exported through the east
edge (≈ 0 warm). So with graupel ~40 % more precipitation forms aloft but ~40 % less reaches the ground: the rest sublimates in
the dry (RH ~0.55) 4–5 km anvil air or evaporates after melting. Whether that is realistic (virga) or a too-strong sublimation
needs process-rate diagnostics (not yet in the model).

### land_cu run
Reproduction of every past land_cu run (exact overrides): `docs/RUNNING_LAND_CU.md`.
```bash
MPLBACKEND=Agg ./venv/bin/python3 run_land_cu.py 14400 > /dev/null 2>&1 &   # 4 h; output in runs/<ts>_land_cu_240min/
./venv/bin/python3 scripts/analyze_land_cu.py runs/<run_dir> 3600 7200       # metrics for a window
```
`console.log` in the run folder holds the progress lines; `runner.log` (if you redirect there)
adds dt and wall-time statistics at the end.

Runtime with numba (87 levels, convective state): 64×64 ≈ 0.30 s/step, 128×128 ≈ 1.05 s/step
(numpy: 1.23 / 5.4 s). dt is 5 s during spin-up and ~3.4 s with active convection at 100 m.
Measured 6-h runs (2026-10-12, 20-s snapshots): 64²/100 m alone on 4 threads 45 min wall
(mean dt 3.5 s); two 64² runs side by side on 2 threads each ~51 min; 128²/50 m on 4 threads
dt 2.4 s, ~28 wall-min per simulated hour (~2.75 h for 6 h), snapshots 12–28 MB (~27 GB per 6 h).
Check `uptime` / other CPU-heavy jobs before estimating — contention can halve the speed.
Recommended steady forcing (pilot_c / main_50m, drift θ_ml +0.15 K/h, cloud base +37 m/h, cover
12–15 %): `SHF=70.0 RAD_COOLING=-5.0 TAU_NUDGE_T=3600.0 TAU_NUDGE_Q=3600.0 NUDGE_Z_TOP=1800.0
W_SUBS=0.005 ENABLE_CLOUD_SHADING=True` (see `docs/EXPERIMENT_LOG.md`, 2026-10-12).
Long unattended runs: start them with `systemd-run --user --unit=<name> --working-directory="$PWD" /bin/bash -c '...'`
so they live outside VS Code's cgroup. A run started from the VS Code terminal or a Claude session
dies when VS Code is OOM-killed or crashes, and `profiles.npz` is only written at the end.
Inflow/outflow in x (validated 2026-10-12, 128×64 / 100 m, 12.8 km): fetch and test region agree
(cover 11.1/11.1 %, w′² 0.668/0.669), no jump at x = 0, water budget residual 0 (the fringe's water
exchange is its own `apply_fringe` budget term). Only the fringe itself is disturbed (variance dip in
its first half), so exclude it from statistics. Cost ≈ 2× a periodic run of the test-region size.
**Caveat (found 2026-10-14):** fetch and test region get the same inflow (the state at RECYCLE_X), so the
test region is a near-exact *twin* of the fetch shifted by RECYCLE_X: corr(w) 0.999 at its upstream end, 0.89 at
its downstream end (50 m, 5 m/s). The fetch/test agreement above is therefore trivial, the full domain looks
periodic with RECYCLE_X in GIFs, and the region statistics are not independent samples. Use it as a paired
control: patch effect = test region minus the fetch field 4.8 km upstream at the same time.
Warm stripe (2026-10-14): `SHF_STRIPE_AMP/_X0/_WIDTH/_TAPER` add a y-uniform SHF excess (LHF unchanged),
for response-vs-distance experiments; keep RECYCLE_X ≥ test-region length so every test x has a twin.
Pass string overrides quoted for the Python parser, e.g. in bash: `"LATERAL_BC='fringe'"`.
Variants without editing the case: `run_land_cu.py 10800 SHF=110.0 W_SUBS=0.005 TAG=steady`
(overrides are recorded in the run's `effective_config.txt`).

### Visualize
```bash
VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow ./venv/bin/python3 scripts/visualize/render_clouds.py \
    runs/<run_dir> --out cloud_gifs/YYYYMMDD_<description>.gif --fps 2 --smooth 0 --jobs 4
# --single <t_s> --out outputs/plots/frame.png   one frame
# --smooth 0 = raw qc field (no blur); 0.5 removes facets; 0.8 visibly bloats clouds
# --zexp vertical exaggeration (1.5), --mode volume (soft, slower)
# --xrange X0 X1 (km) crops x, e.g. to the test region of a fringe run (hides the twin fetch)
# --trange T0 T1 (s) only those snapshots, e.g. the 0–45 min life of the single_cu bubble cloud
# The camera backs off until the ground and the run's highest cloud level fit (since 2026-10-21; before, it was
# framed for shallow Cu and 7 km towers were cut off). single_cu: --zexp 1.0 keeps towers in proportion
# dx/dy and BC_X/BC_Y come from the run's effective_config.txt; in open directions shadows and smoothing
# don't wrap around (before 2026-10-18 they did, which drew a ghost sheet of outflowing cloud on the far edge).
# Clouds cut by the domain edge are closed with a flat face there.
# Terrain (gaussian_hill, agnesi_ridge from effective_config.txt) is drawn as the ground with 100 m contours.
# Before 2026-10-14 the renderer assumed 100 m,
# so earlier 50 m GIFs are stretched 2× horizontally (clouds look flatter/wider than they are).
```
Snapshots are written every `SNAPSHOT_EVERY = 60 s` (≈6.5 MB each at 64×64), one GIF frame each.

### Nested simulation
```bash
./venv/bin/python3 scripts/setup_nested.py --parent-snapshot <snap.npz> --cloud-center 44,56,44 \
  --nest-size 2.5 --dx-fine 25.0 --output nested_runs/ics/nested_ic.npz
# set RESTART_FROM in config_nested.py, then:
MPLBACKEND=Agg nohup ./venv/bin/python3 main_nested.py > outputs/logs/nested_sim.log 2>&1 &
./venv/bin/python3 scripts/archive_current_nested.py
```

---

## Diagnostics (`diag.csv`)

`micro_budget.npz` (since 2026-10-23, every PROFILE_EVERY): cumulative mass [kg] per level converted by each process of
`microphysics.PROCESSES` (cloud→rain, rain evaporation, ice autoconversion/collection, riming, shedding, rain freezing,
melting, graupel sublimation; surface rain/graupel at level 0). Accumulated in the kernels (`s.micro_acc`, Σ over y per
x and level); cloud condensation is not in it. Read with `scripts/compare_microphysics.py RUN … --budget T0 T1`.

- `cloud_frac` — column cloud cover (qc > 1e-5 kg/kg); land_cu target 0.05–0.20
- `CB`, `CT` — domain extremes: lowest / highest level with any qc > 0 (use
  `analyze_land_cu.py` for column-median base/top)
- `LCL` — from dewpoint depression of the lowest 200 m (tracks cloud base well)
- `BL_height` — level of max dθ/dz, i.e. the inversion, not the mixed-layer top
- `w_max`, `w_99`, `qc_max`, `qr_max`, `precip_mmhr`, `n_clouds`
- `cloud_persistence` — correlation of the cloud mask with 20 min earlier; target < 0.4
- `dt` — current adaptive time step

---

## Key fixes and lessons

- **Warm rain (2026-10-19):** Kessler (qc > 0.5 g/kg) made rain within 60 s of the first cloud (0.4–0.65 km
  deep), and the legacy evaporation was 70–100× weaker than Klemp–Wilhelmson, so rain reached the ground
  undiminished and cold pools were ~0.2 K. Runs before this date used `RAIN_SCHEME="kessler"
  RAIN_EVAP="legacy"`; pass those to reproduce them.
- **Anelastic projection (2026-10-11):** the Poisson RHS lacked ρ₀ on the horizontal
  divergence, and the projection used a constant-θ ρ₀. Fixed; residual ~1e-16 and the water
  budget closes exactly (`water_budget.csv` residual 0). Always pass `rho0=state.rho0`.
- **Lost run (2026-10-12):** main_50m was killed at 5 h 41 min by a system OOM event (external
  node processes, ~13 GB). profiles.npz can be rebuilt from the snapshots' `prof_*` arrays
  (θ is stored as θ′, so add θ₀ = profiles.theta[0] − snap_0.prof_theta of any land_cu run).
- **Flux-map bug (fixed 2026-10-12):** `forcing.build_flux_map` (SHF_SIGMA > 0) didn't remove the
  field mean (runs got SHF 35 instead of 70 W/m²), wasn't periodic, and drew SHF/LHF with a chance
  correlation of −0.56. Now: zero-mean, `mode='wrap'`, LHF made exactly uncorrelated with SHF,
  means restored after clipping. Runs with SHF_SIGMA > 0 before this date have wrong mean fluxes.
  SHF_CORR is the Gaussian σ: patches are ~2.5 σ across (500 m → ~6 warm patches on 6.4 km).
- **Default sounding is supersaturated** at ~950–1450 m (14.5 g/kg up to the inversion):
  any run on `CASE = "default"` starts with a cloud slab. Use `land_cu`.
- Old `ENABLE_CLOUD_SHADING` code was never called; shading now lives in `surface_layer.py`.
- Moisture budget (2026-09-21): QV_SCALE / nudging tuning — see `docs/EXPERIMENT_LOG.md`.
- Restart corruption (2026-09-21): never reuse a shared snapshot dir; copy from `restart_snapshots/`.
- **Singular pressure mode (fixed 2026-10-17):** the periodic solve swept the (0,0) mode before discarding
  it; with ρ₀ from `_constant_theta_rho0` its last pivot is exactly 0 and numba raised ZeroDivisionError
  (real runs only survived on round-off). A dummy λ for that mode fixes it; results are unchanged.
- Nested performance (2026-09-22): 10 m (250×250) is 25× slower than real time; use 25 m.
  WENO-3 advection was ~half of the numpy cost per step; now numba (`kernels.py`, 2026-10-12).
