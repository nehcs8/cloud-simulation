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
- `pressure.py` — anelastic FFT projection ∇·(ρ₀u)=0 (pass `rho0=state.rho0`)
- `kernels.py` — numba versions of WENO-3, momentum advection, SGS diffusion/K_M, saturation
  adjustment, Kessler microphysics, tridiagonal pressure solve. The numpy originals stay in
  `dynamics.py` / `thermodynamics.py` / `microphysics.py` / `pressure.py` as the reference
  (`USE_NUMBA = False`); keep both paths in sync and rerun `tests/test_kernels.py` after changes
- `timestepping.py` — RK3 + physics chain
- `thermodynamics.py` — θ_l/q_t conversions, q_sat (Buck)
- `microphysics.py` — Kessler warm rain, implicit sedimentation
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
`tests/test_single_cu.py` checks the single_cu state and prints parcel LCL/LFC/EL/CAPE (target
500–1500 J/kg); run it after any sounding change.
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
- `visualize/render_clouds.py` — fast lit iso-surface renderer → GIF (see Visualize)
- `visualize/visualize3d.py` — older volume renderer (slow without a GPU)
- `find_cloud.py`, `track_cloud.py`, `setup_nested.py`, `archive_current_nested.py`,
  `run_experiment.py`, `run_manager.py`, `debug_theta_l.py`, parameter-sweep scripts

### Data folders
- `runs/<YYYYMMDD_HHMM>_<CASE_NAME>/` — one folder per run: `diag.csv`, `console.log`,
  `config.py` copy, `effective_config.txt` (values after overrides), `profiles.npz`,
  `snapshots/snap_*.npz`, plus `water_budget.csv` / `runner.log` from `run_case.py`
- `experiments/` — older parent-run experiment series (Sept 2026)
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
MPLBACKEND=Agg ./venv/bin/python3 run_case.py single_cu:bubble 10800      # main run, 128², 3 h
```
Folders: `runs/<ts>_single_cu_<variant>_<min>min[_<tag>]/`. Always name the variant (the bare case
is the hot spot, but its folder name then lacks `_hotspot`). Tuning knobs: `SHF_SPOT_AMP/_DIAM`,
`BUBBLE_DTHETA/_RH/_Z0/_RX/_RZ`, `SHF`, `LHF`, `U_GEO`; sounding constants at the top of
`cases/single_cu.py`. Warm rain only, so no ice and no anvil above the ~3.9 km freezing level.
Cost estimate (not yet measured): 160 levels ≈ 1.85× the land_cu column, so ~1.9 s/step at 128²
on 4 threads; dt 1.5–3 s with 10–20 m/s updrafts → ~1–1.5 wall-h per simulated hour. Snapshots
~25–50 MB each at 128² (60 s interval → ~5–9 GB per 3 h).
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
# single_cu: use --zexp 1.0 — the camera is framed for shallow Cu, so 7 km tops at 1.5× leave the frame
# dx/dy come from the run's effective_config.txt. Before 2026-10-14 the renderer assumed 100 m,
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
- Nested performance (2026-09-22): 10 m (250×250) is 25× slower than real time; use 25 m.
  WENO-3 advection was ~half of the numpy cost per step; now numba (`kernels.py`, 2026-10-12).
