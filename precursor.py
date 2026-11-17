"""
Concurrent precursor for open lateral boundaries (OPEN_INFLOW_REF = "precursor").

A small doubly periodic LES (PRECURSOR_NX × PRECURSOR_NY columns, same dx, dy, vertical grid, sounding and
uniform surface fluxes SHF / LHF, no hot spot / stripe / bubble) runs alongside the main domain with the
same dt. The main domain lies on the periodic tiling of the precursor: main cell (i, j) — including ghost
cells outside the domain — sits over precursor cell (i mod PRECURSOR_NX, j mod PRECURSOR_NY). Its turbulent,
heated boundary layer is fed in at every open face (west, east, south, north), at the points where the main
domain has inflow (open_bc.inflow_masks):

  * scalar ghost cells (θ_l′, q_t, q_r, q_g, n_r) are the precursor cells under them
    (open_bc.inflow_ghosts, inflow points only);
  * the boundary-face normal velocity relaxes to the precursor's face value with PRECURSOR_TAU_U at inflow
    points; outflow points keep the radiation condition (open_bc.radiation_tendency), so gust fronts and
    outflow from the main domain leave freely;
  * the inflow relaxation zones (OPEN_RELAX_WIDTH cells) relax toward the precursor cells under them,
    all fields incl. u, v, w (open_bc.apply_relaxation).

Main and precursor structures at the same tiled position move with the same mean wind, so the precursor
turbulence continues into the main domain from whichever side the wind blows. The precursor's horizontal-mean
profiles are the reference of the outflow faces too (s.open_ref). Requires no terrain at the open edges;
periodic main directions must be multiples of the precursor size (PRECURSOR_NY = 0: main NY). The tile repeats
along an edge every PRECURSOR_NX / NY cells. The precursor's horizontal-mean u, v are relaxed to u_g / V_GEO with
OPEN_TAU_NORMAL (a driving force: without Coriolis the periodic box would slowly lose its mean wind to surface
drag); turbulence is untouched.
Before 2026-10-28 only the west face (BC_Y = "periodic", wind along +x) was fed, and the whole west face u
followed the precursor, not only its inflow points.

The precursor State is flagged is_precursor so run_case's water budget ignores its physics calls.
"""
from contextlib import contextmanager

import numpy as np

import config as cfg

@contextmanager
def _as_precursor():
    """Temporarily switch the global config to the precursor's periodic box."""
    saved = {k: getattr(cfg, k) for k in ('NX', 'NY', 'BC_X', 'BC_Y', 'OPEN_INFLOW_REF', 'BUBBLE_DTHETA',
                                         'TERRAIN_TYPE', 'LATERAL_BC')}
    cfg.NX, cfg.NY = int(cfg.PRECURSOR_NX), int(cfg.PRECURSOR_NY or cfg.NY)
    cfg.BC_X, cfg.BC_Y = 'periodic', 'periodic'
    cfg.OPEN_INFLOW_REF, cfg.BUBBLE_DTHETA, cfg.TERRAIN_TYPE, cfg.LATERAL_BC = 'sounding', 0.0, 'flat', 'periodic'
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)


class Precursor:
    def __init__(self, main_grid):
        if cfg.BC_X != 'open' and cfg.BC_Y != 'open':
            raise ValueError('OPEN_INFLOW_REF = "precursor" needs BC_X or BC_Y = "open"')
        if cfg.RESTART_FROM is not None:
            raise ValueError('the precursor has no restart support')
        from grid import Grid
        from state import State
        with _as_precursor():
            self.g = Grid(terrain=None)
            self.s = State(self.g)
        self.s.is_precursor = True
        assert np.allclose(self.g.z_face, main_grid.z_face)
        for bc, n, n_pre, name in ((cfg.BC_X, main_grid.nx, self.g.nx, 'NX'), (cfg.BC_Y, main_grid.ny, self.g.ny, 'NY')):
            if bc != 'open' and n % n_pre:
                raise ValueError(f'periodic main {name} = {n} must be a multiple of the precursor\'s {n_pre}')
        self.shf = np.full((self.g.nx, self.g.ny), float(cfg.SHF))
        self.lhf = np.full((self.g.nx, self.g.ny), float(cfg.LHF))
        print(f"  Precursor: {self.g.nx}×{self.g.ny} periodic, SHF {cfg.SHF:.0f} / LHF {cfg.LHF:.0f} W/m², "
              f"feeds the inflow points of every open face")

    def step(self, dt: float, t: float) -> None:
        from timestepping import rk3_step
        s = self.s
        with _as_precursor():
            rk3_step(s, self.g, dt, self.shf, self.lhf, t=t)
            tau = cfg.OPEN_TAU_NORMAL
            if tau > 0:
                a = 1.0 - np.exp(-dt / tau)
                s.u += a * (s.u_g - s.u[:-1].mean(axis=(0, 1)))[None, None, :]
                s.v += a * (cfg.V_GEO - s.v.mean(axis=(0, 1)))[None, None, :]

    def dt_limit(self, cfl_dt) -> float:
        """Time-step limit of the precursor (cfl_dt: Simulation._cfl_dt taking (s, g))."""
        with _as_precursor():
            return cfl_dt(self.s, self.g)

    # ── what the main domain's open boundaries see ─────────────────────────────
    def at(self, key: str, I, J) -> np.ndarray:
        """Precursor field under main cells (or faces) I × J (index arrays, may lie outside the main domain):
        shape (len(I), len(J), nz[+1]). Face index n of u / v is face 0 again, so mod n_pre works for faces too."""
        f = getattr(self.s, key)
        return f[np.ix_(np.asarray(I) % self.g.nx, np.asarray(J) % self.g.ny)]

    def ghost_planes(self, key: str, H: int, nx: int, ny: int, faces) -> dict:
        """Precursor cells in the ghost cells of the given open faces ('w', 'e': (H, ny, nz); 's', 'n': (nx, H, nz)),
        ordered like the halo (west ghost g at x index g − H, east ghost g at nx + g, likewise in y)."""
        I, J, G = np.arange(nx), np.arange(ny), np.arange(H)
        where = dict(w=(G - H, J), e=(nx + G, J), s=(I, G - H), n=(I, ny + G))
        return {f: self.at(key, *where[f]) for f in faces}

    def mean_profiles(self) -> dict:
        s = self.s
        return {k: getattr(s, k).mean(axis=(0, 1)) for k in ('theta', 'theta_l', 'qv', 'qt')}
