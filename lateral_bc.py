"""
Inflow/outflow in x on the periodic FFT grid: fringe zone with recycled inflow.

The pressure solver needs periodic x, so the domain stays periodic, but the last
FRINGE_WIDTH of it is a fringe zone (Spalart & Watmuff 1993; Nordström et al. 1999)
relaxed toward a copy of the strip of the same width just upstream of RECYCLE_X.
Air leaving the fringe wraps into x = 0 looking like air at RECYCLE_X, so:

    [0, RECYCLE_X)              homogeneous fetch: a precursor that feeds itself
    [RECYCLE_X, Lx − FRINGE)    test region: air crosses it once, then leaves
    [Lx − FRINGE, Lx)           fringe: outflow is replaced by recycled inflow

Assumes the mean wind blows along +x. Applied once per time step after the physics;
the projection in the next RK stage removes the divergence the relaxation introduces.
"""

import numpy as np
import config as cfg

_cache = {}


def _smooth_step(x: np.ndarray) -> np.ndarray:
    """C∞ step: 0 for x ≤ 0, 1 for x ≥ 1 (the Nordström fringe function)."""
    x = np.clip(x, 0.0, 1.0)
    out = (x >= 1.0).astype(float)
    m = (x > 0.0) & (x < 1.0)
    out[m] = 1.0 / (1.0 + np.exp(1.0 / (x[m] - 1.0) + 1.0 / x[m]))
    return out


def fringe_setup(g) -> dict:
    """Indices and weights of the fringe zone and the recycling strip (cached per grid)."""
    key = (g.nx, g.dx, cfg.FRINGE_WIDTH, cfg.RECYCLE_X)
    if key in _cache:
        return _cache[key]
    n_f = int(round(cfg.FRINGE_WIDTH / g.dx))
    i_rec = int(round(cfg.RECYCLE_X / g.dx))
    if n_f < 4:
        raise ValueError(f"FRINGE_WIDTH={cfg.FRINGE_WIDTH} m is under 4 cells at dx={g.dx} m")
    if i_rec - n_f < 0 or i_rec > g.nx - n_f:
        raise ValueError(f"RECYCLE_X={cfg.RECYCLE_X} m: the recycling strip must lie in "
                         f"[{cfg.FRINGE_WIDTH}, Lx − FRINGE_WIDTH] = "
                         f"[{n_f * g.dx:.0f}, {(g.nx - n_f) * g.dx:.0f}] m")
    # Strength rises smoothly over the first half of the fringe, full over the second half
    xi_c = (np.arange(n_f) + 0.5) / n_f          # cell centres
    xi_f = np.arange(n_f) / n_f                  # u faces (west face of each cell)
    setup = dict(
        n_f=n_f, i_rec=i_rec,
        dst=slice(g.nx - n_f, g.nx), src=slice(i_rec - n_f, i_rec),
        lam_c=_smooth_step(2.0 * xi_c), lam_f=_smooth_step(2.0 * xi_f),
    )
    _cache[key] = setup
    return setup


def apply_fringe(s, g, dt: float) -> None:
    """Relax the fringe zone toward the recycling strip (u, v, w, θ_l, q_t, q_r)."""
    f = fringe_setup(g)
    dst, src = f['dst'], f['src']
    # Exact relaxation over dt: stable for any dt / FRINGE_TAU
    a_c = (1.0 - np.exp(-dt * f['lam_c'] / cfg.FRINGE_TAU))[:, None, None]
    a_f = (1.0 - np.exp(-dt * f['lam_f'] / cfg.FRINGE_TAU))[:, None, None]

    # Diagnostic θ, qv, qc are relaxed too, so they stay consistent with θ_l, q_t until the
    # end-of-step update (q_t = qv + qc holds since the relaxation is linear)
    for name in ('v', 'w', 'theta_l', 'qt', 'qr', 'qg', 'theta', 'qv', 'qc'):   # x index = cell index
        a = getattr(s, name)
        target = a[src].copy()
        a[dst] += a_c * (target - a[dst])
    target = s.u[src].copy()                               # u faces 0..nx-1; u[nx] = u[0]
    s.u[dst] += a_f * (target - s.u[dst])
    s.u[-1] = s.u[0]


def hetero_window(g) -> np.ndarray:
    """
    Weight (nx,) in [0, 1] confining flux patterns to the test region in "fringe" mode
    (cos² tapers of HETERO_TAPER at both ends); all ones for periodic x.
    """
    if cfg.LATERAL_BC != "fringe":
        return np.ones(g.nx)
    x0 = cfg.RECYCLE_X + cfg.HETERO_BUFFER
    x1 = g.nx * g.dx - cfg.FRINGE_WIDTH
    tp = max(cfg.HETERO_TAPER, 1e-9)
    if x1 - x0 < 2 * tp:
        raise ValueError(f"No room for flux patterns: test region [{x0:.0f}, {x1:.0f}] m "
                         f"is shorter than two tapers of {tp:.0f} m")
    ramp = lambda d: np.sin(0.5 * np.pi * np.clip(d / tp, 0.0, 1.0)) ** 2
    return ramp(g.x - x0) * ramp(x1 - g.x)
