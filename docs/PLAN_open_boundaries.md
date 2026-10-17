# Plan: open lateral boundaries (non-periodic solver)

Goal: develop the model itself toward real-day simulations driven and checked with WegenerNet data.
First a working non-periodic model; real-day cases come after.

## Why
The model is doubly periodic. In the single_cu congestus run (2026-10-17, 16 km domain) this shows as:
- outflow at 5 km leaves one side and re-enters the other, so the shelf covers the domain by 160 min
- compensating subsidence stays in the domain, so the far environment warms (~0.6 K by 165 min, part of it surface heating)
- gravity waves from the tower re-enter instead of leaving
A bigger periodic domain only dilutes these, and cost grows with area. Real-day runs also need
inflow that is not the model's own outflow.

## Where periodicity lives in the code (2026-10-17)
- `pressure.py`: 2-D real FFT in x, y + tridiagonal in z
- `dynamics.py`: 24 × `np.roll` (WENO-3 advection, momentum advection, SGS diffusion, Coriolis averaging)
- `kernels.py`: `(i ± 1) % nx` / `(j ± 1) % ny` in the stencil loops (numba path)
- `timestepping._apply_boundary_conditions`: `u[nx] = u[0]` (same face)
- smaller pieces: `forcing.build_flux_map` / `spot_profile` (wrap filter, periodic distance),
  `forcing.cloud_transmittance` and the renderer's shadows (periodic shift), `lateral_bc.py` (fringe on
  the periodic grid), diagnostics that use periodic neighbours

## Stages
1. **[done 2026-10-17] Pressure solve with Neumann lateral BCs.** `pressure._project_open`, `BC_X`/`BC_Y`,
   `tests/test_pressure_open.py`; residual ~1e-16, 176 ms vs 216 ms (FFT) at 160²×160. With the normal velocity prescribed on the boundary faces,
   replace the FFT by a cosine transform (`scipy.fft.dctn`, type II on cell centres) in x and/or y; same
   tridiagonal in z and the same cost. Enforce zero net mass flux through the boundaries (solvability of the
   anelastic Poisson problem) by a uniform correction of the outflow normal velocity. Keep FFT as the periodic path.
   Test: divergence residual ~1e-15 with walls and with prescribed in/outflow; periodic path unchanged.
2. **[done 2026-10-17] Ghost cells (halo of 2) instead of wrap-around.** `halo.py` (index-map gather,
   numba), all stencils in `dynamics.py` / `kernels.py` read padded copies; periodic bitwise identical
   (tendencies + 10-min congestus run), no speed cost; `tests/test_halo.py`. Open = zero-gradient ghosts
   for now; a 30-min open/open congestus run is stable. Left periodic: `forcing.cloud_transmittance`
   (shadow shift), `build_flux_map` (wrap filter), `spot_profile`/bubble (periodic distance),
   `microphysics.apply_entrainment` (off by default). One `fill_halo(s)` routine per step fills ghost cells
   from the boundary condition (periodic copy, wall, inflow, outflow); all stencils in `dynamics.py` and
   `kernels.py` read the halo. Largest change by lines, mostly mechanical; `tests/test_kernels.py` (numba vs
   numpy) and the existing physics tests guard it. Periodic halo must reproduce today's results to round-off.
3. **Inflow / outflow conditions** (standard in CM1 / WRF): outflow normal velocity from an Orlanski /
   Klemp–Wilhelmson radiation condition, scalars advected out (upwind); inflow from prescribed profiles
   (θ, qv, u, v); a few-cell relaxation zone (Davies) at the boundaries; `LATERAL_BC = "periodic" | "open"`
   per direction.
4. **Validation.** Dry warm bubble / gravity-wave packet crossing the boundary with little reflection;
   water and mass budgets close (boundary fluxes as their own budget terms); the congestus rerun with open
   boundaries — same tower, shelf leaves the domain, no far-field warming from recirculation.
5. **Turbulent inflow from a precursor.** A cheap periodic background run writes inflow planes every few
   seconds; the open-boundary run reads them. Replaces the fringe recycling (and its fetch/test "twin" problem).

Effort: stages 1–4 a few days of focused work. Run cost per step about the same (+10–20 % for halo
filling); smaller domains become possible because self-interaction is gone.

## Later: real days with WegenerNet
WegenerNet (Univ. Graz): dense station networks in SE Styria (Feldbach region, ~150 stations on
~20 × 15 km, 5-min data: temperature, humidity, wind, precipitation, some soil/radiation) and the alpine
Johnsbachtal network. Uses in the model:
- **validation**: station time series and precipitation fields against the model at matching points
- **surface forcing / initial state**: near-surface T, q, soil moisture to build flux maps and the initial BL
WegenerNet does not give lateral boundaries or the free troposphere. Those would come from:
- a column (sounding + large-scale tendencies) from ERA5 / radiosondes (Graz) for a periodic or open domain
  of the network's size — the standard real-case LES set-up, no 3-D boundaries needed
- 3-D boundaries from a mesoscale model (e.g. WRF driven by ERA5, or a national analysis/forecast product
  such as GeoSphere Austria's) when the domain spans real terrain — the Feldbach region is hilly, so open
  boundaries plus terrain (`terrain.py` exists) are the end state
To check before then: WegenerNet data access/format, terrain data (DEM) for the region, ERA5 access
(`cdsapi` is installed, no `~/.cdsapirc` yet), and whether the region needs ice microphysics for the
target days (convective summer days: yes).
