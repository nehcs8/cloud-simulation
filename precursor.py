"""
Concurrent precursor for open lateral boundaries (OPEN_INFLOW_REF = "precursor").

A small doubly periodic LES (PRECURSOR_NX × NY columns, same dx, dy, vertical grid, sounding and
uniform surface fluxes SHF / LHF, no hot spot / stripe / bubble) runs alongside the main domain with the
same dt. Its turbulent, heated boundary layer is fed in at the west (x = 0) boundary of the main domain:

  * scalar ghost cells (θ_l′, q_t, q_r, q_g, n_r) at x = −H…−1 are the precursor cells
    (i0 − H … i0 − 1) mod PRECURSOR_NX (open_bc.inflow_ghosts, west face, inflow points only);
  * the west boundary-face normal velocity u[0] relaxes to the precursor face u[i0] with
    PRECURSOR_TAU_U (open_bc.radiation_tendency);
  * the inflow relaxation zone (OPEN_RELAX_WIDTH cells) relaxes cell d toward precursor column
    (i0 + d) mod PRECURSOR_NX, all fields incl. u, v, w (open_bc.apply_relaxation).

Main cell d and precursor column i0 + d sit at the same distance downstream of the "virtual" inflow, so
the precursor structures continue into the main domain at the mean wind. The other open faces (east)
use the precursor's horizontal-mean profiles as their reference (s.open_ref). Requires BC_X = "open",
BC_Y = "periodic", a mean wind along +x and no terrain at the west edge. The precursor's horizontal-mean
u, v are relaxed to u_g / V_GEO with OPEN_TAU_NORMAL (a driving force: without Coriolis the periodic
box would slowly lose its mean wind to surface drag); turbulence is untouched.

The precursor State is flagged is_precursor so run_case's water budget ignores its physics calls.
"""
from contextlib import contextmanager

import numpy as np

import config as cfg

i0 = 0                                   # precursor column that sits at the main west boundary


@contextmanager
def _as_precursor():
    """Temporarily switch the global config to the precursor's periodic box."""
    saved = {k: getattr(cfg, k) for k in ('NX', 'BC_X', 'BC_Y', 'OPEN_INFLOW_REF', 'BUBBLE_DTHETA',
                                         'TERRAIN_TYPE', 'LATERAL_BC')}
    cfg.NX, cfg.BC_X, cfg.BC_Y = int(cfg.PRECURSOR_NX), 'periodic', 'periodic'
    cfg.OPEN_INFLOW_REF, cfg.BUBBLE_DTHETA, cfg.TERRAIN_TYPE, cfg.LATERAL_BC = 'sounding', 0.0, 'flat', 'periodic'
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)


class Precursor:
    def __init__(self, main_grid):
        if cfg.BC_X != 'open' or cfg.BC_Y != 'periodic':
            raise ValueError('OPEN_INFLOW_REF = "precursor" needs BC_X = "open" and BC_Y = "periodic"')
        if cfg.RESTART_FROM is not None:
            raise ValueError('the precursor has no restart support')
        from grid import Grid
        from state import State
        with _as_precursor():
            self.g = Grid(terrain=None)
            self.s = State(self.g)
        self.s.is_precursor = True
        assert self.g.ny == main_grid.ny and np.allclose(self.g.z_face, main_grid.z_face)
        self.shf = np.full((self.g.nx, self.g.ny), float(cfg.SHF))
        self.lhf = np.full((self.g.nx, self.g.ny), float(cfg.LHF))
        print(f"  Precursor: {self.g.nx}×{self.g.ny} periodic, SHF {cfg.SHF:.0f} / LHF {cfg.LHF:.0f} W/m², "
              f"feeds the west boundary")

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

    # ── what the main domain's west boundary sees ──────────────────────────────
    def ghost_plane(self, key: str, H: int) -> np.ndarray:
        """(H, ny, nz) precursor cells for the west ghosts x = −H … −1."""
        f = getattr(self.s, key)
        return f[(i0 + np.arange(-H, 0)) % self.g.nx]

    def column(self, key: str, d: int) -> np.ndarray:
        """Precursor field at main cell (or face) index d of the west relaxation zone."""
        f = getattr(self.s, key)
        n = self.g.nx                                    # u has nx + 1 faces, face n == face 0
        return f[(i0 + d) % n]

    def mean_profiles(self) -> dict:
        s = self.s
        return {k: getattr(s, k).mean(axis=(0, 1)) for k in ('theta', 'theta_l', 'qv', 'qt')}
