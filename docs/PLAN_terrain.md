# Plan: terrain (immersed-boundary / masked-cell topography)

Goal: real orography in the LES, first as idealised hills (test cases, and the natural organising trigger
for congestus: upslope flow converging over a heated summit), later the WegenerNet Feldbach region
(rolling hills, ~250–600 m a.s.l., moderate slopes) from a DEM. Builds on the open lateral boundaries
(docs/PLAN_open_boundaries.md, stages 1–4 done).

## Approach
Masked cells ("immersed boundary", staircase terrain), as in PALM and MicroHH: a cell is solid if its
centre lies below the terrain height h(x, y). Chosen over terrain-following coordinates because it keeps
every operator as it is (no metric terms), has no sloping-coordinate pressure-gradient error (a resting
atmosphere over a hill stays at rest), and fits the existing halo/kernels structure. Cost: slopes become
steps (50 m high at the current vertical spacing) — fine for gentle hills, crude for steep Alpine terrain;
terrain-following coordinates remain an option for that later.

## What exists (2026-10-18)
- `terrain.py`: `Terrain(h_2d, grid)` → solid mask (nx, ny, nz) and ground level index `k_sfc`; synthetic
  terrains (Gaussian hill, sine ridge, random field); `from_dem_file` is a stub (NotImplementedError).
- `simulation._create_terrain` selects it via `TERRAIN_TYPE` / `TERRAIN_H_MAX` / `TERRAIN_SIGMA` / `TERRAIN_FILE`.
- `timestepping._apply_boundary_conditions` zeroes u, v, w and resets θ_l, q_t, q_r inside the mask after
  every step.
- Not physical yet: the pressure solve ignores the terrain (flow passes through the hill and is then
  zeroed, so ∇·(ρ₀u) ≠ 0 next to it); stencils mix across solid walls; the surface layer always uses
  level 0; `forcing.apply_surface_fluxes` (which used `k_sfc`) is dead code; sedimentation rains into
  the solid; the water budget counts solid cells.

## Stages
1. **[done 2026-10-18] Pressure solve with solid cells.** `pressure._project_terrain` (+ `kernels.poisson_masked`),
   `tests/test_pressure_terrain.py`: zero flow through solid faces, divergence ~5e-8 of its pre-projection value
   at `TERRAIN_PCG_TOL` = 1e-8; flat-domain fallback identical to the direct solver (1 iteration, 1e-15).
   Warm start = previous φ scaled by the least-squares factor (RK3 stages differ by 1/3, 1/2, 1).
   Cost on 160² with a 500 m / σ 2 km hill (0.6 % solid): median 8 iterations, 1.2 s per projection
   (direct 0.23 s) → 6.65 s per model step, ~2.2× flat. The first RK stage of each step needs ~11 iterations
   (physics between steps changes the field). Faster options if needed: looser tolerance (1e-6 → ~5 it),
   or a cheaper/ better preconditioner near the terrain.
   Original stage-1 plan: Matrix-free anelastic operator ∇·(ρ₀∇φ) on fluid cells with zero
   normal gradient on solid faces (numba, 7-point, face masks); preconditioned conjugate gradient with the
   existing spectral solve (FFT/DCT of the terrain-free operator) as preconditioner. Expect ~5–20
   iterations → 1.5–3× the cost of a projection. Solid faces get zero normal velocity exactly.
   Flat runs keep the direct solver (bitwise identical).
   Test: random field over a hill → ∇·(ρ₀u) ≤ 1e-10 in fluid cells, u_n = 0 on solid faces, iterations count.
2. **[done 2026-10-18] Masked stencils.** `dynamics._ground_fill` (solid cells take their column's ground-level
   value before padding), `dynamics._terrain_masks` + `kernels.kdiff_xy(z)_masked` (no diffusive flux through
   solid faces, incl. horizontal diffusion of w); advection needs no mask (solid-face velocities are 0).
   `tests/test_terrain_stencils.py`: numba = numpy, uniform q_t unchanged, scalars conserved to 1e-17 with
   periodic BCs, values inside the terrain invisible to the fluid. Flat tendencies bitwise identical.
   Original stage-2 plan: Face masks (u-, v-, w-faces touching a solid cell). Advection: zero face velocity
   already gives zero flux; fill solid cells with neighbouring fluid values before the stencils so WENO
   reconstructions next to walls don't read placeholder values. Diffusion: multiply face fluxes by the
   face mask in `kdiff_*` (numba + numpy). Momentum: no-slip on solid faces. Flat runs bitwise identical.
3. **[done 2026-10-18] Surface physics on the terrain.** `surface_layer._apply_surface_layer_terrain` (fluxes into
   each column's `k_sfc` cell, stress at each face's ground level), `kernels.sedimentation_kb` + numpy
   (sweep ends at `k_sfc`, precipitation measured there), `Grid.z_agl()` (noise layer and bubble relative
   to the ground), fluid-only total water in `run_case.py`. `tests/test_terrain_physics.py`; flat 20-min
   congestus run with rain bitwise identical to the stage-2 code. First full run (500 m / σ 1.5 km hill,
   64², 30 min, background heating only): stable, water-budget residual 0, upslope flow 0.1–0.24 m/s toward
   the summit and +0.17 m/s at 300 m over it. Not done: wall drag/fluxes on vertical step faces, LCL /
   qv_bl diagnostics still use the lowest model levels, old terrain masking in
   `timestepping._apply_boundary_conditions` uses non-periodic boundary-face masks (harmless while the
   terrain is away from the edges).
   Original stage-3 plan: Surface layer at `k_sfc` per column (height above ground of the
   first fluid cell = Δz/2), flux maps / hot spot / shading unchanged in x, y; wall drag on vertical step
   faces (optional, small for gentle slopes). Sedimentation ends at `k_sfc` (surface precipitation at
   the ground). Initial noise/bubble relative to the ground. Diagnostics and the water budget exclude
   solid cells. Open boundaries: terrain tapered to flat over the relaxation zone so `open_bc.py` is unchanged.
4. **Validation.**
   - Resting, stably stratified atmosphere over a Gaussian hill, no fluxes: stays at rest (≤ 1e-6 m/s).
   - Linear mountain waves: uniform U, constant N, small ridge (Nh/U ≈ 0.1), open x / periodic y, compared
     with the analytic linear solution (vertical velocity pattern and wave drag).
   - Heated Gaussian hill, no wind: upslope flow and a single convergence zone over the summit; then the
     congestus case triggered by the hill instead of the hot spot (compare cloud-base fragmentation with
     the hot-spot runs).
5. **Real terrain.** `from_dem_file`: read a GeoTIFF (rasterio) for the WegenerNet Feldbach region,
   crop, area-average to the grid, smooth (σ ≈ 1–2 Δx), set the lowest point to z = 0. Candidate DEM:
   Austria's 10 m digital terrain model (open government data; access and licence to be checked).
   Domain for the region at 100 m: ~220 × 160 cells (~1.4× the 160² runs).

## Effort and cost
Stages 1–4: larger than the open-boundary work (the pressure solver is the hard part; stencil and
physics changes are mechanical with the existing tests as guard). Run cost with terrain: pressure 1.5–3×,
everything else unchanged → roughly 1.3–1.8× per step.

## Open questions
- Vertical spacing near the ground: 50 m steps; a finer lowest layer (25 m) would smooth slopes at extra cost.
- Wall fluxes on vertical step faces (heat, drag): needed for steep terrain, optional for Feldbach.
- Real-day forcing (WegenerNet, ERA5 or radiosonde column, open boundaries) is a separate plan.
