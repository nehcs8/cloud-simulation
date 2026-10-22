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
4. **[partly done 2026-10-18] Validation.**
   - `tests/test_terrain_rest.py`: resting stratified atmosphere over a 500 m hill stays exactly at rest
     (0.0 after 30 min — no sloping-coordinate pressure error with masked cells).
   - `tests/test_mountain_waves.py` (Witch-of-Agnesi ridge, linear hydrostatic theory): with 25 m levels
     near the ground the wave drag is 1.05–1.21× theory at 1–6 km, w pattern correlation 0.89, amplitude
     1.00 (passes). **The staircase acts ~0.6 Δz higher than the terrain:** 50 m levels give 1.24–1.47×
     (200 m ridge) and 1.66–1.92× (100 m ridge); a deeper sponge does not change it. Fixes: finer levels
     near the ground for terrain runs, or partial ("shaved") bottom cells that let a cell be partly solid
     (aperture/volume fractions in the pressure operator and fluxes, Adcroft et al. 1997).
   - **Partial bottom cells (2026-10-18)**, `TERRAIN_PARTIAL_CELLS = True` (`terrain.hc` fluid fractions,
     side-face apertures = smaller neighbour fraction, `TERRAIN_HFAC_MIN` 0.3; pressure operator, scalar fluxes
     ÷ hc, diffusion, surface layer, sedimentation and water budget use them; momentum advection unchanged).
     Unit tests pass in both modes. Mountain waves at Δz 50 m: 200 m ridge 1.12–1.36× (staircase 1.24–1.47),
     corr 0.83; 100 m ridge 1.38–1.90× (staircase 1.66–1.92), corr 0.60. Only a small gain: with partial
     cells the side walls are ~5 m, yet the bias stays ~0.5 Δz of effective extra height in every variant →
     the cause is not the steps but the z-level lowest-cell treatment of flow up a slope. Remaining options:
     finer levels near the ground (25 m: 1.05–1.21×), shaved cells with sloping cut faces (Adcroft et al. 1997)
     or terrain-following coordinates near the ground. Off by default.
   - **Smagorinsky K over terrain (fixed 2026-10-19).** `_smag_K` / `kernels.smag_KM` ignored the terrain: the
     solid cell under the ground (u = 0) next to 10 m/s fluid gave K ≈ 180 m²/s (≤ 12 elsewhere), which set the
     diffusive dt limit (0.073 s at Δz 12.5 m, 0.44 s at 25 m), and the centred du/dz in the first fluid cell saw
     the solid u = 0 as no-slip shear on the free-slip ground. Now K = 0 in solid cells and du/dz, dv/dz are
     one-sided at `k_sfc`. dt 0.4 s (12.5 m) / 1.07 s (25 m), 3–5× faster; drag only 0.02–0.04 lower.
     Mountain waves after the fix (2.5 h, M/M_theory at 1–6 km):
       Δz 25 m,   h₀ 200 m: 1.05–1.19, corr 0.89, amplitude 1.01 (PASS)
       Δz 12.5 m, h₀ 200 m: 1.01–1.19, corr 0.89, amplitude 1.02 (PASS)
       Δz 12.5 m, h₀ 200 m, partial cells: 1.03–1.23, corr 0.87 (PASS)
       Δz 12.5 m, h₀ 100 m: 1.22–1.58, corr 0.90, amplitude 1.13 (FAIL; was 1.66–1.92 / corr 0.50 at 50 m)
     **Halving Δz below 25 m changes nothing**, so the bias is not the near-ground vertical treatment. The
     excess is about the same in N/m for both ridges (+690 N/m at 200 m, +520 N/m at 100 m) — closer to an
     additive drag than an effective extra height. Controls (2026-10-19): flat ground with the same open BCs
     gives M = 0 and w = 0 for 2.5 h (the boundaries make no drag). **Δx 100 m** (512 × 4, Δz 25 m, h₀ 200 m):
     1.01–1.10, corr 0.91, amplitude 0.99 — mean excess over 1–6 km 6 % vs 12 % at Δx 200 m. **But** the
     100 m ridge at Δx 100 m, Δz 25 m gives 1.43–1.75 (corr 0.78, amplitude 1.41), between Δz 12.5 m
     (1.22–1.58) and Δz 50 m (1.66–1.92) at Δx 200 m: for the small ridge the excess follows Δz, not Δx.
     M(t) fluctuates by ±0.1 after 2 h, so the 200 m-ridge Δx gain is within the noise. **Cause unresolved.**
     The small ridge (Nh/U = 0.1) is the most linear case yet the worst, so the error is a roughly fixed
     amount (tens of m of effective height, or several hundred N/m), not a nonlinear-theory gap. Not yet
     tried: longer runs / time-averaging M over 1.5–2.5 h, wider domain, a 100 m ridge with Δz 12.5 m and
     Δx 100 m, momentum advection at the terrain (unchanged by partial cells).
   - **Heated hill as congestus trigger (2026-10-19).** `single_cu:congestus`, 96² at 100 m, periodic, Gaussian
     moist hot spot (`SHF_SPOT_SHAPE='gaussian'`) + bubble as in `tune96_gauss`, plus a 400 m Gaussian hill
     (σ 1.5 km) under the spot: `runs/20261019_1633_single_cu_congestus_120min_tune96_hill` (shear) and `…_hillcalm`
     (U = 0). Metrics from `scripts/analyze_cloud_base.py` (30/60/90/120 min, r < 3.5 km; cores 800 m above ground):
       flat (tune96_gauss): pieces at base+150 m 1/2/3/4, cores 86/46/64/40, u_r(50 m AGL, 0.5–1.5 km) −0.4…−1.9 m/s
       hill, shear:         pieces 4/7/4/3, cores 81/55/51/40, u_r −0.6…−1.3; tops ≤ 6.7 km, rain ≤ 2.3 mm/h
       hill, calm:          pieces 1/15/8/9, cores 109/66/56/70, u_r −1.0 at 30 min, then +1.4/+2.6 (rain outflow
                            from the summit), rain ≤ 5.7 mm/h, tops ≤ 7.8 km
     The hill doubles the early low-level inflow (30 min) but **does not organise the cloud base**: core counts
     match the flat run, and without wind the base is more fragmented. Once it rains over the summit the
     outflow replaces the upslope flow. GIFs: `cloud_gifs/20261019_single_cu_congestus96_hill{,_calm}_0-120min.gif`,
     flat reference `cloud_gifs/20261018_single_cu_congestus96_gauss_flat_0-120min.gif` (renderer now draws the
     terrain with 100 m contours). Remaining candidates for the fragmented base: 50 m resolution, cold pools.
     Rerun 2026-10-19/20 with Seifert–Beheng rain + KW78 evaporation (see CLAUDE.md): calm hill periodic
     (`…_hillcalm_sb`) and open (`…_hillcalm_sb_open`): one compact tower over the summit until ~80 min, then a real
     cold pool (θ′ −1.1 K) and outflow (+2.7…+4.5 m/s at 120 min) shut it down; the "wild" refiring of the Kessler
     run was the missing evaporation (summit stayed +3 K under 5 mm/h rain). Base still fragmented at 60–90 min
     (13–22 pieces).
   Original stage-4 plan:
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
